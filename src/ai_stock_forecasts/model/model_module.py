
import os
from typing import Union
from lightning.pytorch import Callback, Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import EncoderNormalizer, QuantileLoss, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from pytorch_forecasting.tuning.tuner import Tuner

import torch
import json
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import numpy as np
import pandas as pd
from pandas import DataFrame

from ai_stock_forecasts.utils.s3_util import S3ParquetUtil
from lightning_utilities.core.rank_zero import rank_zero_only



if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')

"""
    Provides helpful logging.
"""
class StepPrint(Callback):
    def __init__(self, every=50): self.every = every
    def on_train_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        tl = m.get("train_loss")
        if tl is not None:
            print(f"[epoch {trainer.current_epoch}] train_loss={float(tl):.6f}", flush=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        vl = m.get("val_loss")
        if vl is not None:
            print(f"[epoch {trainer.current_epoch}] val_loss={float(vl):.6f}", flush=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.every == 0:
            mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            print(
                f"epoch={trainer.current_epoch} step={trainer.global_step} "
                f"batch={batch_idx} cuda_mem_GB={mem:.2f}",
                flush=True
            )


class ModelModule:
    def __init__(self, loss=QuantileLoss(quantiles=[0.3, 0.5, 0.7])):
        self.s3_util = S3ParquetUtil()

        self.model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
        self.ckpt_dir = os.path.join(self.model_dir, "checkpoints")

        self.model = None
        self._construct_loss(loss)
        self._construct_base_callbacks()

    def _construct_loss(self, loss):
        self.loss = loss
        self.output_size = len(self.loss.quantiles)
        self.mode = "quantiles"

    def _construct_base_callbacks(self):
        self.early_stop = EarlyStopping(
            monitor="val_loss",
            min_delta=0.0,
            patience=5,
            mode="min",
        )

        self.checkpoint_callback = ModelCheckpoint(
            save_top_k=-1,
            every_n_epochs=1,
            dirpath=self.ckpt_dir,
            filename="tft-epoch-{epoch:02d}",
            save_last=True,
        )

        self.ckpt_best_callback = ModelCheckpoint(
            dirpath=self.ckpt_dir,
            filename="tft-best-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        )

        self.step_logging_callback = StepPrint(every=50)

        self.callbacks = [ self.early_stop, self.checkpoint_callback, self.ckpt_best_callback, self.step_logging_callback ]

    def run_training(self, training_dataset: TimeSeriesDataSet, 
                     learning_rate: float, hidden_size: int,
                     attention_head_size: int, dropout: float,
                     hidden_continuous_size: int, lstm_layers: int,
                     reduce_on_plateau_patience: int, max_epochs: int,
                     accelerator: str, devices: int,
                     train_dataloader: DataLoader, val_dataloader: DataLoader,
                     gradient_clip_val: Union[None, float]):


        if (not isinstance(self.model, TemporalFusionTransformer)):
            self.model = TemporalFusionTransformer.from_dataset(
                training_dataset,
                learning_rate=learning_rate,
                hidden_size=hidden_size,
                attention_head_size=attention_head_size,
                dropout=dropout,
                loss=self.loss,
                output_size=self.output_size,
                hidden_continuous_size=hidden_continuous_size,
                lstm_layers=lstm_layers,
                log_interval=10,
                log_val_interval=1,
                reduce_on_plateau_patience=reduce_on_plateau_patience,
            )
        else:
            print('model is already loaded in, running fine tuning')

        self.trainer = Trainer(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            strategy="ddp" if devices > 1 else "auto",
            callbacks=self.callbacks,
            gradient_clip_val=gradient_clip_val,
        )

        self.trainer.fit(self.model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    @rank_zero_only
    def save_checkpoint(self):
        self.trainer.save_checkpoint(os.path.join(self.model_dir, 'tft_model.ckpt'))

    def load_model_from_checkpoint(self, model_id: str, map_location: str):
        with self.s3_util.load_best_model_checkpoint(model_id) as ckpt_path:
            self.model = TemporalFusionTransformer.load_from_checkpoint(ckpt_path, map_location=torch.device(map_location), weights_only=False)

    """ There is a bug with torch metrics where even if you specify map_location to something other than cuda
        loading in the model can still fail: https://github.com/pytorch/pytorch/issues/113973

        This method acts as a work around when attempting to run models trained on cuda but loaded in on mps or cpu.
    """
    def load_model_from_checkpoint_and_data(self, model_id: str, map_location: str,
                                            training_dataset: TimeSeriesDataSet, 
                                            learning_rate: float, hidden_size: int,
                                            attention_head_size: int, dropout: float,
                                            hidden_continuous_size: int, lstm_layers: int,
                                            reduce_on_plateau_patience: int):
        with self.s3_util.load_best_model_checkpoint(model_id) as ckpt_path:
            self.model = TemporalFusionTransformer.from_dataset(
                training_dataset,
                learning_rate=learning_rate,
                hidden_size=hidden_size,
                attention_head_size=attention_head_size,
                dropout=dropout,
                loss=self.loss,
                output_size=self.output_size,
                hidden_continuous_size=hidden_continuous_size,
                lstm_layers=lstm_layers,
                log_interval=10,
                log_val_interval=1,
                reduce_on_plateau_patience=reduce_on_plateau_patience,
            )
            ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
            self.model.load_state_dict(ckpt['state_dict'], strict=False)

    def run_batch_inference(self, dataloader: DataLoader, model_id: str, df: DataFrame, save_predictions: bool=True):
        if not isinstance(self.model, TemporalFusionTransformer):
            raise Exception('must load in model before loading predictions')

        trainer_kwargs = { "accelerator": "gpu", "devices": 1 }
        self.predictions = self.model.predict(
            dataloader,
            mode=self.mode,
            return_x=True,
            return_y=True,
            return_index=True,
            trainer_kwargs=trainer_kwargs
        )

        if save_predictions:
            self.s3_util.save_raw_predictions(model_id, self.predictions)
        self.convert_raw_predictions_to_simpler_format(df)
        if save_predictions:
            self.s3_util.save_human_readable_predictions(model_id, self.predictionsDF)

    def run_single_day_inference(self, dataloader: DataLoader, df: DataFrame):
        if not isinstance(self.model, TemporalFusionTransformer):
            raise Exception('must load in model before loading predictions')

        trainer_kwargs = { "accelerator": "gpu", "devices": 1 }
        self.predictions = self.model.predict(
            dataloader,
            mode=self.mode,
            return_x=True,
            return_y=True,
            return_index=True,
            trainer_kwargs=trainer_kwargs,
        )
        self.convert_raw_predictions_to_simpler_format(df)


    def load_raw_predictions(self, model_id: str, df: DataFrame):
        self.predictions = self.s3_util.load_raw_predictions(model_id)
        self.convert_raw_predictions_to_simpler_format(df)

    def load_human_readable_predictions(self, model_id: str):
        self.predictionsDF = self.s3_util.load_human_readable_predictions(model_id)

    def convert_raw_predictions_to_simpler_format(self, df: DataFrame):
        timestamps = self._get_timestamps(df)

        y_pred = self.predictions.output

        y_true = self.predictions.y[0]
        groups = self.predictions.x["groups"]
        symbols = self.predictions.index['symbol']
        # y_pred shape: (239, 14, 3)
        # y shape: (239, 14)
        # symbols: (239, 1)
        # timestamps: (239, 14)

        # pulls the y at timestamp you are predicting from
        enc_tgt = self.predictions.x["encoder_target"]
        current_y = enc_tgt[:, -1].detach().cpu().numpy() 

        y_true_np = y_true.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()
        ts_np = timestamps

        p30 = np.round(y_pred_np[:, :, 0], 10)
        p50 = np.round(y_pred_np[:, :, 1], 10)
        p70 = np.round(y_pred_np[:, :, 2], 10)

        self.predictionsDF = DataFrame({
            "symbol": symbols,
            "timestamp": ts_np,
            "current_y": current_y,
            "y": list(y_true_np),
            "y_pred_p30": list(p30),
            "y_pred_p50": list(p50),
            "y_pred_p70": list(p70),
        })

    def append_actuals_to_simple_predictions(self, df: DataFrame):
        # time_idx, timestamp, symbol, feature_a, feature_b, ...
        self.predictionsDF = self.predictionsDF.merge(df, on=['symbol', 'timestamp'], how='inner')

    def plot_mape_by_symbol(self):
        self.mapeResultDF = self.predictionsDF.copy()
        self.mapeResultDF['mape'] = ((self.mapeResultDF['y'] - self.mapeResultDF['y_pred_p50']).abs() / self.mapeResultDF['y']) * 100
        self.mapeResultDF = self.mapeResultDF.assign( mape_first=self.mapeResultDF["mape"].apply(lambda x: float(np.asarray(x).ravel()[0])) )
        self.mapeResultDF = self.mapeResultDF.groupby(['symbol']).mean()
        self.mapeResultDF = self.mapeResultDF.sort_values('mape_first', ascending=False)
        self.mapeResultDF = self.mapeResultDF.reset_index(drop=False)
        self.mapeResultDF[['symbol', 'mape_first']].plot(x='symbol', y='mape_first')
        plt.show()


    def upload_checkpoints_to_s3(self, model_id: str):
        self.s3_util.upload_checkpoints(self.ckpt_dir, self.model_dir, model_id)

    def quick_set_predictions_df(self, pred_file: str="src/ai_stock_forecasts/orchestration/val_predictions_series_level.csv"):
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 2000)
        pd.set_option('display.max_colwidth', None)
        self.predictionsDF = pd.read_csv(pred_file)

        def parse_series_string(x):
            if not isinstance(x, str):
                return x
            try:
                # Try standard JSON parsing first (compatible with the .tolist() fix)
                return json.loads(x.replace("'", '"'))
            except json.JSONDecodeError:
                # Fallback: Parse numpy's space-separated string format
                # Remove brackets and split by whitespace (handles newlines automatically)
                cleaned = x.replace('[', '').replace(']', '').strip()
                return [float(i) for i in cleaned.split()] if cleaned else []

        for col in ["y", "y_pred_p30", "y_pred_p50", "y_pred_p70"]:
            if col in self.predictionsDF.columns and self.predictionsDF[col].dtype == "object":
                self.predictionsDF[col] = self.predictionsDF[col].apply(parse_series_string)

    def find_optimal_hyperparameters(self, train_dataloader: DataLoader, val_dataloader: DataLoader):
        torch.serialization.add_safe_globals([EncoderNormalizer])
        study = optimize_hyperparameters(
            train_dataloader,
            val_dataloader,
            model_path="optuna_tft",
            n_trials=30,
            max_epochs=20,
            gradient_clip_val_range=(0.01, 1.0),
            hidden_size_range=(16, 128),
            hidden_continuous_size_range=(8, 64),
            attention_head_size_range=(1, 8),
            dropout_range=(0.1, 0.5),
            learning_rate_range=(1e-4, 1e-2),
            loss=self.loss,
            reduce_on_plateau_patience=4,
            use_learning_rate_finder=True,
        )

        print("Best trial value:", study.best_trial.value)
        print("Best parameters:")
        for k, v in study.best_trial.params.items():
            print(f"  {k}: {v}")

    def find_optimal_learning_rate(self, train_dataloader: DataLoader, val_dataloader: DataLoader,
                                   max_epochs: int, accelerator: str, devices: int):
        self.trainer = Trainer(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            strategy="ddp" if devices > 1 else "auto",
            callbacks=self.callbacks,
        )

        res = Tuner(self.trainer).lr_find(
            self.model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            max_lr=10.0,
            min_lr=1e-6,
        )

        print(f"suggested learning rate: {res.suggestion()}")
        fig = res.plot(show=True, suggest=True)
        fig.show()

    def plot_prediction(self, dataloader: DataLoader):
        trainer_kwargs = { "accelerator": "gpu", "devices": 1 }
        raw = self.model.predict(dataloader, mode='raw', return_x=True, return_y=True, trainer_kwargs=trainer_kwargs)

        out = raw.output
        x = raw.x
        idx = 0

        fig = self.model.plot_prediction(x, out, 0)

        fig.savefig("prediction.png", dpi=200, bbox_inches="tight")
        # fig.show()

    def _get_timestamps(self, df) -> np.ndarray:
        decoder_time_idx = self.predictions.x["decoder_time_idx"].detach().cpu().numpy()

        ts_lookup = (
            df[["time_idx", "timestamp"]]
            .drop_duplicates()
            .set_index("time_idx")["timestamp"]
            .sort_index()
        )

        current_time_idx = decoder_time_idx[:, 0] - 1
        current_timestamps = ts_lookup.reindex(current_time_idx).to_numpy()

        return current_timestamps

    def interpret_predictions(self, dataloader: DataLoader, output_path: str = ''):
        if not isinstance(self.model, TemporalFusionTransformer):
            raise Exception('must load in model before loading predictions')

        raw = self.model.predict(dataloader, mode='raw', return_x=True)

        interp = self.model.interpret_output(raw.output, reduction='sum')

        fig = self.model.plot_interpretation(interp)
        if output_path == "":
            plt.show()
        else:
            fig_list = fig if isinstance(fig, (list, tuple)) else [fig]
            with PdfPages(Path(output_path) / 'interpretation.pdf') as pdf:
                for fig in fig_list:
                    if fig is None:
                        continue
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)




