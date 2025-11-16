from datetime import datetime, timezone

import pandas as pd
import argparse
from alpaca.data import TimeFrame
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import NaNLabelEncoder, TimeSeriesDataSet

from ai_stock_forecasts.pytorch_datamodule_util.construct_time_series_dataset_util import \
    ConstructTimeSeriesDatasetUtil
from ai_stock_forecasts.s3.s3_util import S3ParquetUtil

from pytorch_forecasting.models import TemporalFusionTransformer
from lightning.pytorch import Trainer
from pytorch_forecasting.metrics import QuantileLoss

class Orchestration:
    def __init__(self):
        self.symbol_encoder = None
        self.dataset = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.model = None
        self.features_data = None
        self.s3_util = S3ParquetUtil()
        self.construct_time_series_dataset_util = ConstructTimeSeriesDatasetUtil()

    def load_dataset(self, symbols: list[str], features: list[str],
                     train_start: datetime, train_end: datetime,
                     validation_start: datetime, validation_end: datetime,
                     max_lookback_period: int, max_prediction_length: int,
                     time_frame: TimeFrame=TimeFrame.Day, accelerator: str = "cpu"):
        self.features_data = self.s3_util.get_features_data(symbols, features, time_frame)

        pivoted = self.construct_time_series_dataset_util.build_pivoted_with_time_idx(
            self.features_data
        )
        train_mask = (pivoted["timestamp"] >= train_start) & (pivoted["timestamp"] <= train_end)
        train_df = pivoted[train_mask].copy()

        self.symbol_encoder = NaNLabelEncoder().fit(pd.Series(symbols))
        self.dataset = TimeSeriesDataSet(
            train_df,
            time_idx="time_idx",
            group_ids=["symbol"],
            time_varying_known_reals=self.construct_time_series_dataset_util._get_known_features(
                pd.DataFrame([vars(r) for r in self.features_data])
            ),
            time_varying_unknown_reals=self.construct_time_series_dataset_util._get_unknown_features(
                pd.DataFrame([vars(r) for r in self.features_data])
            ),
            max_encoder_length=max_lookback_period,
            max_prediction_length=max_prediction_length,
            target="open",
            allow_missing_timesteps=True,
            categorical_encoders={"symbol": self.symbol_encoder},
        )

        batch_size = 64
        use_gpu = accelerator in ("gpu", "cuda")
        self.train_dataloader = self.dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=2, pin_memory=use_gpu)

        training_cutoff = train_df["time_idx"].max()
        val_source = pivoted[pivoted["timestamp"] <= validation_end].copy()

        validation_dataset = TimeSeriesDataSet.from_dataset(
            self.dataset,
            val_source,
            min_prediction_idx=training_cutoff + 1,
            stop_randomization=True,
        )

        print(validation_dataset)

        self.val_dataloader = validation_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=2, pin_memory=use_gpu)


    def train(self,
              learning_rate: float = 1e-4,
              hidden_size: int = 32,
              attention_head_size: int = 4,
              dropout: float = 0.1,
              hidden_continuous_size: int = 16,
              lstm_layers: int = 1,
              reduce_on_plateau_patience: int = 3,
              max_epochs: int = 50,
              accelerator: str = "cpu",
              devices: str = 1):
        loss = QuantileLoss(quantiles=[0.3, 0.5, 0.7])

        """ hidden_size = core model width, number of hidden units used in LSTM encoder/decoder and the gating/mixing layers inside TFT
                            typical values: 16, 32, 64, 128
                            higher = more expressive but slower and easier to overfit.
                            increase when running on beefer gpu / cpu
            attention_head_size = the number of attention heads
                            more heads means model can learn multiple types of temporal relationships in parallel
                            typical values: 1, 4, 8
                            higher = increases model expressiveness, smaller datasets prefer 1-4 heads
                            increase based on increase in data
                            chatgpt recommendation:
                                <50k rows <20 symbols = 1
                                50-500k rows 20-200 symbols = 2-4
                                500k rows 200-500 symbols = 4-8
                                5 million rows 1000 symbols = 8
            dropout = probability of dropping units during training to prevent overfitting.
                            typical values = 0.0-0.3
                            higher = more regularization
                            lower = more capacity, more risk of overfitting
            hidden_continuous_size = size in which data is compressed to.
                            typical values = 8, 16, 32
                            small = may compress too aggressively
                            large = more expressive, but slower and more prone to overfitting
            lstm_layers = number of stacked lstm layers
                            typical values = 1 or 2
                            more layers = deeper sequence modeling, higher compute and harder to train
                            mor ethan 2 rarely helps
            reduce_on_plateau_patience = how many validation epochs without improvement before the learning rate is reduced.
                            typical values 2 - 5
                            
        """
        self.model = TemporalFusionTransformer.from_dataset(
            self.dataset,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            attention_head_size=attention_head_size,
            dropout=dropout,
            loss=loss,
            output_size=len(loss.quantiles),
            hidden_continuous_size=hidden_continuous_size,
            lstm_layers=lstm_layers,
            log_interval=10,
            log_val_interval=1,
            reduce_on_plateau_patience=reduce_on_plateau_patience,
        )

        early_stop = EarlyStopping(
            monitor="val_loss",
            min_delta=0.0,
            patience=5,
            mode="min",
        )

        checkpoint = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            dirpath="checkpoints",
            filename="tft-best-{epoch:02d}-{val_loss:.4f}",
        )

        trainer = Trainer(max_epochs=max_epochs,
                          accelerator=accelerator,
                          devices=devices,
                          callbacks=[early_stop, checkpoint])
        trainer.fit(self.model, train_dataloaders=self.train_dataloader, val_dataloaders=self.val_dataloader)

        trainer.save_checkpoint("/opt/ml/model/tft_model.ckpt")

    def load_trained_model(self, path: str="tft_model.ckpt"):
        self.model = TemporalFusionTransformer.load_from_checkpoint(path)

    def save_predictions(self):
        # Ask explicitly for all quantiles
        predictions = self.model.predict(
            self.val_dataloader,
            mode="quantiles",
            return_x=True,
            return_y=True,
        )

        # y_pred: (n_samples, decoder_len, num_quantiles)
        y_pred = predictions.output

        print(y_pred.shape)

        y_true = predictions.y[0]

        if y_pred.ndim != 3:
            raise ValueError(f"Expected y_pred to be 3D (batch, decoder_len, num_quantiles), got {y_pred.shape}")

        n_samples, decoder_len, num_quantiles = y_pred.shape

        # sanity: y_true should match batch x decoder_len
        assert y_true.shape[0] == n_samples, f"y_true batch {y_true.shape[0]} != {n_samples}"
        assert y_true.shape[1] == decoder_len, f"y_true decoder_len {y_true.shape[1]} != {decoder_len}"

        # Flatten for dataframe
        y_true_flat = y_true.reshape(n_samples * decoder_len).cpu().numpy()
        y_pred_flat = y_pred.reshape(n_samples * decoder_len, num_quantiles).cpu().numpy()

        # Get symbol indices
        groups = predictions.x["groups"]          # (n_samples, ...)
        symbol_idx_batch = groups[:, 0]          # (n_samples,)
        symbol_idx_full = symbol_idx_batch.repeat_interleave(decoder_len)  # (n_samples * decoder_len,)

        symbols = self.symbol_encoder.inverse_transform(symbol_idx_full.cpu().numpy())

        # Final safety checks
        n_rows = n_samples * decoder_len
        assert len(symbols) == n_rows
        assert y_true_flat.shape[0] == n_rows
        assert y_pred_flat.shape[0] == n_rows

        df = pd.DataFrame({
            "symbol": symbols,
            "actual": y_true_flat,
            "pred_p30": y_pred_flat[:, 0],
            "pred_p50": y_pred_flat[:, 1],
            "pred_p70": y_pred_flat[:, 2],
        })

        df.to_csv("val_predictions.csv", index=False)
        print(df.head())

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--attention_head_size", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden_continuous_size", type=int, default=16)
    parser.add_argument("--lstm_layers", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=50)

    parser.add_argument("--train_start", type=str, default="2019-01-02")
    parser.add_argument("--train_end", type=str, default="2024-01-01")
    parser.add_argument("--val_start", type=str, default="2024-01-02")
    parser.add_argument("--val_end", type=str, default="2025-01-01")

    parser.add_argument("--max_lookback_period", type=int, default=1000)
    parser.add_argument("--max_prediction_length", type=int, default=14)

    parser.add_argument("--symbols_path", type=str, default="../constants/symbols.txt")

    parser.add_argument("--accelerator", type=str, default="cpu")
    parser.add_argument("--devices", type=str, default=1)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    with open(args.symbols_path, "r") as f:
        symbols = [line.strip() for line in f]

    orchestration = Orchestration()
    features = ['open', 'close', 'high', 'low', 'open', 'trade_count', 'volume', 'vwap']
    # features = ['open']

    train_start = datetime.fromisoformat(args.train_start).replace(tzinfo=timezone.utc)
    train_end = datetime.fromisoformat(args.train_end).replace(tzinfo=timezone.utc)
    val_start = datetime.fromisoformat(args.val_start).replace(tzinfo=timezone.utc)
    val_end = datetime.fromisoformat(args.val_end).replace(tzinfo=timezone.utc)

    orchestration.load_dataset(symbols, features,
                                train_start, train_end,
                               val_start, val_end,
                               args.max_lookback_period, args.max_prediction_length,
                               TimeFrame.Day, args.accelerator)
    orchestration.train(args.learning_rate, args.hidden_size, args.attention_head_size, args.dropout, args.hidden_continuous_size,
                        args.lstm_layers, 3, args.max_epochs, args.accelerator, args.devices)
    # orchestration.load_trained_model()
    # orchestration.save_predictions()
