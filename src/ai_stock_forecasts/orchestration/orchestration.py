from datetime import datetime, timezone

import numpy as np
import pandas as pd
import argparse
from alpaca.data import TimeFrame
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import NaNLabelEncoder, TimeSeriesDataSet
import torch
import json
from torch.utils.data import DataLoader

from ai_stock_forecasts.utils.plot_utils import plot_different_forecast_strategies_profits
from ai_stock_forecasts.pytorch_datamodule_util.construct_time_series_dataset_util import \
    ConstructTimeSeriesDatasetUtil
from ai_stock_forecasts.s3.s3_util import S3ParquetUtil

from pytorch_forecasting.models import TemporalFusionTransformer
from lightning.pytorch import Trainer
from pytorch_forecasting.metrics import QuantileLoss
from lightning.pytorch.callbacks import Callback

class StepPrint(Callback):
    def __init__(self, every=50): self.every = every
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.every == 0:
            mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            print(f"step={trainer.global_step} batch={batch_idx} cuda_mem_GB={mem:.2f}", flush=True)


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
                     time_frame: TimeFrame, accelerator: str = "cpu",
                     num_workers: int = 1, batch_size: int = 8):
        self.features_data = self.s3_util.get_features_data(symbols, features, time_frame)

        self.pivoted = self.construct_time_series_dataset_util.build_pivoted_with_time_idx(
            self.features_data
        )
        train_mask = (self.pivoted["timestamp"] >= train_start) & (self.pivoted["timestamp"] <= train_end)
        train_df = self.pivoted[train_mask].copy()

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

        use_gpu = accelerator in ("gpu", "cuda")
        self.train_dataloader = self.dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=num_workers, pin_memory=use_gpu)

        training_cutoff = train_df["time_idx"].max()
        val_source = self.pivoted[self.pivoted["timestamp"] <= validation_end].copy()

        validation_dataset = TimeSeriesDataSet.from_dataset(
            self.dataset,
            val_source,
            min_prediction_idx=training_cutoff + 1,
            stop_randomization=True,
        )

        self.val_dataloader = validation_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=num_workers, pin_memory=use_gpu)


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
              devices: str = "auto"):
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
                            more than 2 rarely helps
            reduce_on_plateau_patience = how many validation epochs without improvement before the learning rate is reduced.
                            typical values 2 - 5
                            
        """
        # assert isinstance(self.dataset, pd.DataFrame)

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

        print("about to fetch one train batch", flush=True)
        _ = next(iter(self.train_dataloader))
        print("fetched one train batch", flush=True)

        trainer = Trainer(max_epochs=max_epochs,
                          accelerator=accelerator,
                          devices=1,
                          callbacks=[early_stop, checkpoint, StepPrint(every=50)]
                          )
        trainer.fit(self.model, train_dataloaders=self.train_dataloader, val_dataloaders=self.val_dataloader)

        trainer.save_checkpoint("/opt/ml/model/tft_model.ckpt")

    def load_trained_model(self, path: str="tft_model.ckpt"):
        ckpt = torch.load(path, weights_only=False, map_location="cpu")
        for n, v in ckpt["state_dict"].items():
            if not torch.isfinite(v).all():
                print("❌ Bad tensor:", n, "contains NaN/Inf")
        self.model = TemporalFusionTransformer.load_from_checkpoint(path, map_location="cpu")

    def save_predictions(self, save_location: str="src/ai_stock_forecasts/orchestration/val_predictions_series_level.csv"):
        assert isinstance(self.model, TemporalFusionTransformer)
        assert isinstance(self.val_dataloader, DataLoader)
        assert isinstance(self.symbol_encoder, NaNLabelEncoder)

        predictions = self.model.predict(
            self.val_dataloader,
            mode="quantiles",
            return_x=True,
            return_y=True,
        )

        timestamps = self._get_decoder_timestamps(predictions)

        y_pred = predictions.output

        y_true = predictions.y[0]
        groups = predictions.x["groups"]
        symbol_idx_batch = groups[:, 0]
        symbols = self.symbol_encoder.inverse_transform(
            symbol_idx_batch.cpu().numpy()
        )
        # y_pred shape: (239, 14, 3)
        # y shape: (239, 14)
        # symbols: (239, 1)
        # timestamps: (239, 14)

        y_true_np = y_true.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()
        ts_np = timestamps

        p30 = np.round(y_pred_np[:, :, 0], 2)
        p50 = np.round(y_pred_np[:, :, 1], 2)
        p70 = np.round(y_pred_np[:, :, 2], 2)

        self.predictionsDF = pd.DataFrame({
            "symbol": symbols,
            "timestamp": ts_np[:, 0],
            "y": list(y_true_np),
            "y_pred_p30": list(p30),
            "y_pred_p50": list(p50),
            "y_pred_p70": list(p70),
        })

        self.predictionsDF.to_csv(save_location, index=False)
        # print(self.predictionsDF.head())


    def _get_decoder_timestamps(self, predictions) -> np.ndarray:
        assert isinstance(self.symbol_encoder, NaNLabelEncoder)


        decoder_time_idx = predictions.x["decoder_time_idx"].cpu().numpy()  # (B, L)
        B, L = decoder_time_idx.shape

        groups = predictions.x["groups"][:, 0].cpu().numpy()  # (B,)
        symbols = self.symbol_encoder.inverse_transform(groups)  # (B,)

        df_keys = pd.DataFrame({
            "symbol": np.repeat(symbols, L),              # (B*L,)
            "time_idx": decoder_time_idx.reshape(-1),     # (B*L,)
        })

        ts_lookup = (
            self.pivoted[["time_idx", "timestamp"]]
            .drop_duplicates()
        )

        df_join = df_keys.merge(ts_lookup, on=["time_idx"], how="left")

        timestamps = df_join["timestamp"].to_numpy().reshape(B, L)

        return timestamps

    def evaluate_validation_period_profit(self, interval_days: int=7, num_stocks_purchased: int=10):
        assert isinstance(self.predictionsDF, pd.DataFrame)

        money = 25000
        timestamps = (
            self.predictionsDF["timestamp"]
            .drop_duplicates()
            .sort_values()
            .reset_index(drop=True)
        )
        # print(f"the length of timestamps: {len(timestamps)}")
        current_ts = timestamps.min()
        while True:
            filtered = self.predictionsDF[self.predictionsDF["timestamp"] == current_ts]

            # determine what stocks to buy
            p30 = np.stack(filtered["y_pred_p30"].to_numpy())
            p50 = np.stack(filtered["y_pred_p50"].to_numpy())
            p70 = np.stack(filtered["y_pred_p70"].to_numpy())

            eps = 1e-8
            start = p50[:, 0]
            end = p50[:, interval_days]

            x = (end - start) / (start + eps) * 100.0
            band_width_pct = (p70 - p30) / (p30 + eps) * 100.0
            y = band_width_pct.mean(axis=1)
            score = x - y * 0.01
            # score = x
            filtered["x"] = x
            filtered["z"] = y
            filtered["score"] = score

            top_x = filtered.sort_values("score", ascending=False).head(num_stocks_purchased)
            # print(top_x['symbol'])

            total_profit = sum(((row[interval_days] - row[0]) / row[0]) * (25000 / num_stocks_purchased) for row in top_x["y"])
            # print(f"Total Profit (investing $500 per stock): {total_profit}")
            money += total_profit

            # go to 7 days later
            i = timestamps[timestamps == current_ts].index[0]
            if i + interval_days >= len(timestamps):
                break
            current_ts = timestamps[i + interval_days]

        # print(f"Total money made over 2024: {money - 25000}")
        absolute_difference = abs(money - 25000)
        average = (money + 25000) / 2
        difference = (absolute_difference / average) * 100
        # print(f"Percentage gained: {difference}")
        return difference

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

    parser.add_argument("--symbols_path", type=str, default="src/ai_stock_forecasts/constants/symbols.txt")

    parser.add_argument("--accelerator", type=str, default="cpu")
    parser.add_argument("--devices", type=str, default="auto")

    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    with open(args.symbols_path, "r") as f:
        symbols = [line.strip() for line in f]

    orchestration = Orchestration()
    features = ['open', 'close', 'high', 'low', 'trade_count', 'volume', 'vwap']
    # features = ['open']

    train_start = datetime.fromisoformat(args.train_start).replace(tzinfo=timezone.utc)
    train_end = datetime.fromisoformat(args.train_end).replace(tzinfo=timezone.utc)
    val_start = datetime.fromisoformat(args.val_start).replace(tzinfo=timezone.utc)
    val_end = datetime.fromisoformat(args.val_end).replace(tzinfo=timezone.utc)

    # orchestration.load_dataset(symbols, features,
    #                            train_start, train_end,
    #                            val_start, val_end,
    #                            args.max_lookback_period, args.max_prediction_length,
    #                            TimeFrame.Day, args.accelerator, args.num_workers, args.batch_size)
    #orchestration.train(args.learning_rate, args.hidden_size, args.attention_head_size, args.dropout, args.hidden_continuous_size,
    #                   args.lstm_layers, 3, args.max_epochs, args.accelerator, args.devices)

    #orchestration.load_trained_model("src/ai_stock_forecasts/orchestration/tft_model_expensive.ckpt")
    # result = orchestration.evaluate_validation_period_profit(7, 50)
    # print(f"result pre-expensive training: {result}")

    orchestration.quick_set_predictions_df("src/ai_stock_forecasts/orchestration/val_predictions_series_level_expensive.csv")

    results = {}
    for interval in range(1,14):
        for num_stocks in range(1, 51):
            results[(interval, num_stocks)] = orchestration.evaluate_validation_period_profit(interval, num_stocks)

    print(results)

    plot_different_forecast_strategies_profits(results)


