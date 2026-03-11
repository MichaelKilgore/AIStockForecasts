
import os
from typing import Union
from lightning.pytorch import Callback, Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import EncoderNormalizer, QuantileLoss, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.models.base import Prediction
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

        self.callbacks = [ self.checkpoint_callback, self.ckpt_best_callback, self.step_logging_callback ]
        # self.callbacks = [ self.early_stop, self.checkpoint_callback, self.ckpt_best_callback, self.step_logging_callback ]

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
            logger=False,
        )

        self.trainer.fit(self.model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    @rank_zero_only
    def save_checkpoint(self):
        self.trainer.save_checkpoint(os.path.join(self.model_dir, 'tft_model.ckpt'))

    def load_model_from_checkpoint(self, model_id: str, map_location: str, load_last_ckpt: bool=False):
        self.model_id = model_id
        self.map_location = map_location
        self.load_last_ckpt = load_last_ckpt

        self._load_model_from_checkpoint()

    def _load_model_from_checkpoint(self):
        with self.s3_util.load_best_model_checkpoint(self.model_id, pull_last_ckpt=self.load_last_ckpt) as ckpt_path:
            self.model = TemporalFusionTransformer.load_from_checkpoint(ckpt_path, map_location=torch.device(self.map_location), weights_only=False)

    """ There is a bug with torch metrics where even if you specify map_location to something other than cuda
        loading in the model can still fail: https://github.com/pytorch/pytorch/issues/113973

        This method acts as a work around when attempting to run models trained on cuda but loaded in on mps or cpu.
    """
    def load_model_from_checkpoint_and_data(self, model_id: str, map_location: str,
                                            training_dataset: TimeSeriesDataSet, 
                                            learning_rate: float, hidden_size: int,
                                            attention_head_size: int, dropout: float,
                                            hidden_continuous_size: int, lstm_layers: int,
                                            reduce_on_plateau_patience: int, load_last_ckpt: bool=False):
        with self.s3_util.load_best_model_checkpoint(model_id, load_last_ckpt) as ckpt_path:
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

    def run_batch_inference(self, dataloaders: list[DataLoader], model_id: str, df: DataFrame, save_predictions: bool=True, is_large: bool=False):
        if not isinstance(self.model, TemporalFusionTransformer):
            raise Exception('must load in model before loading predictions')

        trainer_kwargs = { "accelerator": "gpu", "devices": 1 }

        predictions = None
        for dl in dataloaders:
            pred = self.model.predict(
                dl,
                mode=self.mode,
                return_x=True,
                return_y=True,
                return_index=True,
                trainer_kwargs=trainer_kwargs
            )

            if predictions:
                predictions = self._append_predictions_chunk(predictions, pred)
            else:
                predictions = pred

        # TODO: for large models with many symbols and large prediction windows s3 doesn't allow you to save files larger than 5gb so we would need to break this up but for now just gonna skip saving it cause I don't really use the raw predictions for anything currently anyways.
        if save_predictions and not is_large:
            self.s3_util.save_raw_predictions(model_id, predictions)
        predictionsDF = self.convert_raw_predictions_to_simpler_format(predictions, df)
        if save_predictions:
            self.s3_util.save_human_readable_predictions(model_id, predictionsDF)


    def _append_predictions_chunk(self, predictions, p: Prediction):
        x_combined = {
            k: torch.cat([predictions.x[k], p.x[k]], dim=0)
            for k in predictions.x
        }

        y_combined = tuple(
            torch.cat([predictions.y[i], p.y[i]], dim=0)
            if predictions.y[i] is not None
            else None
            for i in range(len(predictions.y))
        )

        return Prediction(
            output=torch.cat([predictions.output, p.output], dim=0),

            x=x_combined,

            index=pd.concat([predictions.index, p.index], ignore_index=True),

            decoder_lengths=None,

            y=y_combined
        )

    def run_single_day_inference(self, dataloader: DataLoader, df: DataFrame) -> DataFrame:
        if not isinstance(self.model, TemporalFusionTransformer):
            raise Exception('must load in model before loading predictions')

        trainer_kwargs = { "accelerator": "gpu", "devices": 1 }
        predictions = self.model.predict(
            dataloader,
            mode=self.mode,
            return_x=True,
            return_y=True,
            return_index=True,
            trainer_kwargs=trainer_kwargs,
        )
        return self.convert_raw_predictions_to_simpler_format(predictions, df, True)


    def load_raw_predictions(self, model_id: str, df: DataFrame):
        predictions = self.s3_util.load_raw_predictions(model_id)
        self.convert_raw_predictions_to_simpler_format(predictions, df)

    def load_human_readable_predictions(self, model_id: str) -> DataFrame:
        return self.s3_util.load_human_readable_predictions(model_id)

    def convert_raw_predictions_to_simpler_format(self, predictions, df: DataFrame, is_single_day_inference: bool=False) -> DataFrame:
        timestamps = self._get_timestamps(predictions, df)

        y_pred =predictions.output

        y_true = predictions.y[0]
        symbols = predictions.index['symbol']
        # y_pred shape: (239, 14, 3)
        # y shape: (239, 14)
        # symbols: (239, 1)
        # timestamps: (239, 14)

        # pulls the y at timestamp you are predicting from
        # enc_tgt = torch.roll(predictions.y[0], shifts=-1, dims=0)

        y_true_np = y_true.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()
        ts_np = timestamps

        p30 = np.round(y_pred_np[:, :, 0], 10)
        p50 = np.round(y_pred_np[:, :, 1], 10)
        p70 = np.round(y_pred_np[:, :, 2], 10)

        predictionsDF = DataFrame({
            "symbol": symbols,
            "timestamp": ts_np,
            "y": list(y_true_np),
            "y_pred_p30": list(p30),
            "y_pred_p50": list(p50),
            "y_pred_p70": list(p70),
        })

        # make sure rows are ordered correctly first
        predictionsDF = predictionsDF.sort_values(["symbol", "timestamp"])

        if not is_single_day_inference:
            # extract the scalar you want from the per-row array
            predictionsDF["current_y"] = predictionsDF["y"].apply(lambda arr: arr[0])

            # shift within each symbol only
            predictionsDF["current_y"] = (
                predictionsDF.groupby("symbol")["current_y"].shift(1)
            )

            predictionsDF = predictionsDF[
                predictionsDF["current_y"].notna()
            ]
        else:
            predictionsDF["current_y"] = predictions.x["encoder_target"][:, -1].cpu().numpy()

        return predictionsDF


    def append_actuals_to_simple_predictions(self, predictionsDF: DataFrame, df: DataFrame) -> DataFrame:
        # time_idx, timestamp, symbol, feature_a, feature_b, ...
        return predictionsDF.merge(df, on=['symbol', 'timestamp'], how='inner')

    def plot_mape_by_symbol(self, predictionsDF):
        self.mapeResultDF = predictionsDF.copy()
        self.mapeResultDF['mape'] = ((self.mapeResultDF['y'] - self.mapeResultDF['y_pred_p50']).abs() / self.mapeResultDF['y']) * 100
        self.mapeResultDF = self.mapeResultDF.assign( mape_first=self.mapeResultDF["mape"].apply(lambda x: float(np.asarray(x).ravel()[0])) )
        self.mapeResultDF = self.mapeResultDF.groupby(['symbol']).mean()
        self.mapeResultDF = self.mapeResultDF.sort_values('mape_first', ascending=False)
        self.mapeResultDF = self.mapeResultDF.reset_index(drop=False)
        self.mapeResultDF[['symbol', 'mape_first']].plot(x='symbol', y='mape_first')
        plt.show()


    def upload_checkpoints_to_s3(self, model_id: str):
        self.s3_util.upload_checkpoints(self.ckpt_dir, self.model_dir, model_id)


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

    def _get_timestamps(self, predictions, df) -> np.ndarray:
        decoder_time_idx = predictions.x["decoder_time_idx"].detach().cpu().numpy()

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

        trainer_kwargs = { "accelerator": "gpu", "devices": 1 }
        raw = self.model.predict(dataloader, mode='raw', return_x=True, trainer_kwargs=trainer_kwargs)

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




