
from datetime import datetime
import os
from typing import Union
from alpaca.data import TimeFrame, TimeFrameUnit
from pandas import DataFrame, factorize, to_numeric
from pytorch_forecasting import TimeSeriesDataSet
from ai_stock_forecasts.model.data.data_module import DataModule
from ai_stock_forecasts.s3.s3_util import S3ParquetUtil


"""
    Used for constructing the necessary dataloaders for training / validation / testing of
    TimeSeries pytorch models.
"""
class TrainingDataModule(DataModule):
    def __init__(self, symbols: list[str], features: list[str], time_frame: Union[TimeFrame, str],
                 max_lookback_period: int, max_prediction_length: int, is_df_cached: bool):
        if isinstance(time_frame, str):
            time_frame = TimeFrame(1, TimeFrameUnit(time_frame))

        self.s3_util = S3ParquetUtil()

        """
            The purpose of caching df is when we are running very large dataset across multiple GPUs,
            it is standard practice to create multiple identical datasets for each device run on.

            constructing our df one time is important because it saves on execution time and more importantly
            it reduces the maximum amount of memory required to construct our data by number of devices multiples.
        """
        self.is_df_cached = is_df_cached
        self.cache_dir = os.environ.get("SM_INPUT_DIR", "/tmp")
        self.cache_path = os.path.join(self.cache_dir, f"pivoted_{time_frame.amount_value}_{time_frame.unit_value}.parquet")

        super().__init__(symbols, features, time_frame, max_lookback_period, max_prediction_length)

        self.training_dataset = None
        self.validation_dataset = None
        self.test_dataset = None

        self.train_dataloader = None
        self.validation_dataloader = None
        self.test_dataloader = None

    def _construct_df(self):
        if self.is_df_cached:
            self.df.read_parquet(self.cache_path)

        features_data = self.s3_util.get_features_data(self.symbols, self.features, self.time_frame)

        pivoted_features_data = self._pivot_features_data(features_data)

        pivoted_features_data['time_idx'] = factorize(pivoted_features_data['timestamp'], sort=True)[0].astype('int64')
        pivoted_features_data = pivoted_features_data.sort_values(['time_idx', 'symbol'])

        for col in pivoted_features_data.columns:
            if col in self.known_categoricals:
                pivoted_features_data[col] = pivoted_features_data[col].astype(int).astype(str).astype('category')

        self.df = pivoted_features_data

    def _pivot_features_data(self, df: DataFrame):
        df['value'] = to_numeric(df['value'])
        df = df.drop_duplicates(subset=['symbol', 'timestamp', 'feature'], keep='first')
        wide = df.pivot(index=['symbol', 'timestamp'], columns='feature', values='value').reset_index()
        return wide

    def construct_training_and_validation_datasets(self, train_start: datetime, train_end: datetime,
                                                    validation_end: datetime):
        train_mask = (self.df["timestamp"] >= train_start) & (self.df["timestamp"] <= train_end)
        train_df = self.df.loc[train_mask].copy()

        self.training_dataset = TimeSeriesDataSet(
            train_df,
            time_idx="time_idx",
            group_ids=["symbol"],
            time_varying_known_reals=self._get_known_reals(),
            time_varying_unknown_reals=self._get_unknown_reals(),
            time_varying_known_categoricals=self._get_known_categoricals(),
            max_encoder_length=self.max_lookback_period,
            max_prediction_length=self.max_prediction_length,
            target="open",
            allow_missing_timesteps=True,
            categorical_encoders=self.categorical_encoders,
        )

        training_max_idx = train_df["time_idx"].max()

        validation_mask = (self.df["timestamp"] >= train_start) & (self.df["timestamp"] <= validation_end)
        validation_df = self.df.loc[validation_mask].copy()

        self.validation_dataset = TimeSeriesDataSet.from_dataset(
            dataset=self.training_dataset,
            data=validation_df,
            min_prediction_idx=training_max_idx + 1,
            stop_randomization=True,
        )

    def construct_test_dataset(self, train_start: datetime, validation_end: datetime, test_end: datetime):
        if self.training_dataset == None:
            raise Exception("You have to construct the training_dataset before executing this function")

        validation_mask = (self.df["timestamp"] >= train_start) & (self.df["timestamp"] <= validation_end)
        validation_df = self.df.loc[validation_mask].copy()
        validation_max_idx = validation_df["time_idx"].max()

        test_mask = (self.df["timestamp"] >= train_start) & (self.df["timestamp"] <= test_end)
        test_df = self.df.loc[test_mask].copy()

        self.test_dataset =  TimeSeriesDataSet.from_dataset(
            dataset=self.training_dataset,
            data=test_df,
            min_prediction_idx=validation_max_idx + 1,
            stop_randomization=True,
        )

    def construct_train_and_validation_dataloaders(self, batch_size: int, num_workers: int, pin_memory: bool):
        if self.training_dataset == None or self.validation_dataset == None:
            raise Exception("You have to construct the training_dataset before executing this function")

        self.train_dataloader = self.training_dataset.to_dataloader(train=True, batch_size=batch_size,
                num_workers=num_workers, pin_memory=pin_memory, persistent_workers=(num_workers > 0)) 
        self.validation_dataloader = self.validation_dataset.to_dataloader(train=False, batch_size=batch_size,
                num_workers=num_workers, pin_memory=pin_memory, persistent_workers=(num_workers > 0)) 

    def construct_test_dataloader(self, batch_size: int, num_workers: int, pin_memory: bool):
        if self.test_dataset == None:
            raise Exception("You have to construct the training_dataset before executing this function")

        self.test_dataloader = self.test_dataset.to_dataloader(train=False, batch_size=batch_size,
                num_workers=num_workers, pin_memory=pin_memory, persistent_workers=(num_workers > 0)) 

    def cache_df(self):
        self.df.to_parquet(self.cache_path, index=False)


