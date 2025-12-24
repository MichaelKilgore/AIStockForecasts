from datetime import datetime

import pandas as pd
from alpaca.data import TimeFrame
from pandas import DataFrame
from pytorch_forecasting.data.timeseries._timeseries import TimeSeriesDataSet

from ai_stock_forecasts.models.historical_data import HistoricalData

class ConstructTimeSeriesDatasetUtil:
    def __init__(self):
       self.known_features = ["timestamp"]

    def build_pivoted_with_time_idx(self, data: list[HistoricalData]):
        df = pd.DataFrame([vars(r) for r in data])
        pivoted = self._pivot_features(df)
        #pivoted = pivoted.sort_values(["symbol", "timestamp"])
        #pivoted["time_idx"] = pivoted.groupby("symbol").cumcount().astype("int64")
        pivoted = pivoted.sort_values(["timestamp", "symbol"])
        pivoted["time_idx"] = pd.factorize(pivoted["timestamp"])[0].astype("int64")
        return pivoted

    def get_time_series_dataset(self, data: list[HistoricalData], start: datetime, end: datetime, max_lookback_period: int, max_prediction_length: int, symbol_encoder):
        df = pd.DataFrame([vars(r) for r in data])
        filtered = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]

        return self._convert_historical_data_to_time_series_dataset(filtered, max_lookback_period, max_prediction_length, symbol_encoder)

    def get_validation_time_series_dataset(self, train_dataset, data: list[HistoricalData], start: datetime, end: datetime):
        df = pd.DataFrame([vars(r) for r in data])
        filtered = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
        pivoted_data = self._pivot_features(filtered)
        pivoted_data = pivoted_data.sort_values(["symbol", "timestamp"])
        pivoted_data["time_idx"] = pivoted_data.groupby("symbol").cumcount().astype("int64")

        training_cutoff = train_dataset.index.time.max()
        min_prediction_idx = training_cutoff + 1

        return TimeSeriesDataSet.from_dataset(train_dataset, pivoted_data, stop_randomization=True, min_prediction_idx=min_prediction_idx)

    def _convert_historical_data_to_time_series_dataset(self, df: DataFrame, max_lookback_period: int, max_prediction_length: int, symbol_encoder):
        pivoted_data = self._pivot_features(df)
        pivoted_data = pivoted_data.sort_values(["symbol", "timestamp"])
        pivoted_data["time_idx"] = pivoted_data.groupby("symbol").cumcount().astype("int64")

        return TimeSeriesDataSet(
                pivoted_data,
                time_idx="time_idx",
                group_ids=["symbol"],
                time_varying_known_reals=self._get_known_features(df),
                time_varying_unknown_reals=self._get_unknown_features(df),
                max_encoder_length=max_lookback_period,
                max_prediction_length=max_prediction_length,
                target="open",
                allow_missing_timesteps=True,
                categorical_encoders={
                    "symbol": symbol_encoder,  # <- same encoder you stored on Orchestration
                },
            )

    def _get_known_features(self, df: DataFrame) -> list[str]:
        distinct_features = df["feature"].unique().tolist()
        res = []
        for feature in distinct_features:
            if feature in self.known_features:
                res.append(feature)
        res.append("timestamp")
        return res

    def _get_unknown_features(self, df: DataFrame) -> list[str]:
        distinct_features = df["feature"].unique().tolist()
        res = []
        for feature in distinct_features:
            if feature not in self.known_features:
                res.append(feature)
        return res


    def _pivot_features(self, df: DataFrame):
        df["value"] = df.apply(lambda row: self._cast_value(row["value"], row["type"]), axis=1)

        pivoted = (
            df.pivot_table(
                index=["symbol", "timestamp"],
                columns="feature",
                values="value",
                aggfunc="first"
            ).reset_index()
        )

        return pivoted

    def _cast_value(self, value, type_str):
        """Convert string to the right Python type."""
        if type_str == "float":
            return float(value)
        elif type_str == "int":
            return int(value)
        elif type_str == "bool":
            return str(value).lower() in ("true", "1")
        elif type_str == "str":
            return str(value)
        else:
            return value

if __name__ == "__main__":
    records = [
        HistoricalData("AAPL", datetime(2025, 1, 1, 10), "open", "150.0", "float", datetime(2025, 1, 1, 10, 1), TimeFrame.Day, datetime(2025, 1, 1)),
        HistoricalData("AAPL", datetime(2025, 1, 1, 10), "close", "155.0", "float", datetime(2025, 1, 1, 10, 1), TimeFrame.Day, datetime(2025, 1, 1)),
        HistoricalData("AAPL", datetime(2025, 1, 2, 10), "open", "156.0", "float", datetime(2025, 1, 2, 10, 1), TimeFrame.Day, datetime(2025, 1, 2)),
        HistoricalData("AAPL", datetime(2025, 1, 2, 10), "close", "158.0", "float", datetime(2025, 1, 2, 10, 1), TimeFrame.Day, datetime(2025, 1, 2)),
    ]

    construct_timeseries_dataset_util = ConstructTimeSeriesDatasetUtil()

    res = construct_timeseries_dataset_util.get_time_series_dataset(records, datetime(2024, 1, 1), datetime(2026, 1, 3), 1, 1)

    print(res)
