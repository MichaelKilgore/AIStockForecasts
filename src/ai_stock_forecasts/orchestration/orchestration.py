from datetime import datetime, timezone

import pandas as pd
import numpy as np
from alpaca.data import TimeFrame
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import NaNLabelEncoder

from src.ai_stock_forecasts.pytorch_datamodule_util.construct_time_series_dataset_util import \
    ConstructTimeSeriesDatasetUtil
from src.ai_stock_forecasts.s3.s3_util import S3ParquetUtil

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
                     time_frame: TimeFrame=TimeFrame.Day):
        self.features_data = self.s3_util.get_features_data(symbols, features, time_frame)

        self.symbol_encoder = NaNLabelEncoder().fit(pd.Series(symbols))
        self.dataset = self.construct_time_series_dataset_util.get_time_series_dataset(self.features_data, train_start, train_end, max_lookback_period, max_prediction_length, self.symbol_encoder)

        batch_size = 64
        self.train_dataloader = self.dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=0)

        validation_dataset = self.construct_time_series_dataset_util.get_validation_time_series_dataset(self.dataset, self.features_data, train_start, validation_end)
        self.val_dataloader = validation_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=0)


    def train(self):
        loss = QuantileLoss(quantiles=[0.3, 0.5, 0.7])

        self.model = TemporalFusionTransformer.from_dataset(
            self.dataset,
            learning_rate=1e-4,
            hidden_size=16,
            attention_head_size=4,
            dropout=0.1,
            loss=loss,
            output_size=len(loss.quantiles),
            hidden_continuous_size=16,
            lstm_layers=2,
            log_interval=10,
            log_val_interval=1,
            reduce_on_plateau_patience=3,
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

        trainer = Trainer(max_epochs=10, accelerator="cpu", callbacks=[early_stop, checkpoint])
        trainer.fit(self.model, train_dataloaders=self.train_dataloader, val_dataloaders=self.val_dataloader)

        trainer.save_checkpoint("tft_model.ckpt")

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

if __name__ == "__main__":
    orchestration = Orchestration()
    orchestration.load_dataset(["GOOGL", "AAPL"], ["open"], datetime(2019, 1, 2, tzinfo=timezone.utc), datetime(2024, 1, 1, tzinfo=timezone.utc),
                               datetime(2024, 1, 2, tzinfo=timezone.utc), datetime(2025, 1, 1, tzinfo=timezone.utc),
                               1000, 14,
                               TimeFrame.Day)
    # orchestration.train()
    orchestration.load_trained_model()
    orchestration.save_predictions()
