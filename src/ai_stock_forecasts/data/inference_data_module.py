
import logging
from datetime import datetime, timedelta
from typing import Optional, Union

import holidays
import numpy as np
import pandas as pd
from alpaca.data import TimeFrame, TimeFrameUnit
from pandas import DataFrame, factorize, to_numeric
from pytorch_forecasting import TimeSeriesDataSet
from torch.utils.data import DataLoader

from ai_stock_forecasts.cron.populate_historical_features_daily_grain_psql_table import (
    HistoricalFeaturesDailyPopulator,
)
from ai_stock_forecasts.data.data_module import DataModule
from ai_stock_forecasts.utils.get_historical_data_util import GetHistoricalDataUtil
from ai_stock_forecasts.utils.postgres_util import PostgresUtil


_TIME_FEATURES = {'year', 'month', 'day_of_month', 'day_of_week', 'hour_of_day', 'minute_of_day'}
# We always pull these from postgres regardless of self.features so downstream filters / fill logic work.
_ALWAYS_PULLED = ['open', 'close', 'high', 'low', 'volume']

# Columns that get zero-filled on future-dated skeleton rows so TimeSeriesDataSet doesn't choke on NaN.
_FUTURE_ZERO_FILL = ['range', 'body', 'lower_wick', 'upper_wick', 'vix_log', 'surprise', 'sandp500open']


class InferenceDataModule(DataModule):
    def __init__(self, symbols: list[str], features: list[str], time_frame: Union[TimeFrame, str],
                 max_lookback_period: int, max_prediction_length: int, target: str = 'open',
                 curr_date: Optional[datetime] = None, volume_filter: int = 100000):
        self.curr_date = curr_date if curr_date else datetime.now()

        self.postgres_util = PostgresUtil()
        self.get_historical_data_util = GetHistoricalDataUtil()
        self.volume_filter = volume_filter
        super().__init__(symbols, features, time_frame, max_lookback_period, max_prediction_length, target=target)

    def _construct_df(self):
        if not (self.time_frame.amount_value == 1 and self.time_frame.unit_value == TimeFrameUnit.Day):
            raise Exception(f'The time_frame: {self.time_frame} is not supported')

        min_date_needed = self._calendar_date_for_trading_days(
            self.max_lookback_period + 1, direction=-1, include_holidays=True,
        )
        max_date_needed = self._calendar_date_for_trading_days(
            self.max_prediction_length + 1, direction=1, include_holidays=False,
        )

        pg_features = self._features_to_pull_from_postgres()

        # 2. Pull from postgres, pivot long → wide. -> takes about a minute
        wide_df = self._pull_and_pivot(pg_features, min_date_needed, self.curr_date)

        # 3. This should filter out all rows where target doesn't exist which we solves multiple problem:
        #   1. removing rows on days where the market is closed
        #   2. removing any data missing target which we need since the model can't utilize that data without target.
        if 'close_log_return' in wide_df.columns:
            wide_df = wide_df[wide_df['close_log_return'].notna()]

        # 3. Detect missing trading days from the pivoted df (weekends / holidays fall away naturally).
        missing = self._missing_trading_days(wide_df, min_date_needed, self.curr_date)
        if missing:
            raise RuntimeError(
                f'missing {len(missing)} trading day(s) in historical_features: {missing}. '
                f'running HistoricalFeaturesDailyPopulator for {min(missing)}..{max(missing)}')

        # 4. Ensure every requested feature has a column, even if postgres returned no rows for it.
        for feat in pg_features:
            if feat not in wide_df.columns:
                wide_df[feat] = np.nan

        # 5. Volume filter on past data only (so the average isn't tanked by NaN future rows).
        if 'volume' in wide_df.columns and self.volume_filter > 0:
            avg_vol = wide_df.groupby('symbol')['volume'].mean()
            high_volume_symbols = avg_vol[avg_vol >= self.volume_filter].index
            wide_df = wide_df[wide_df['symbol'].isin(high_volume_symbols)]
        else:
            high_volume_symbols = wide_df['symbol'].unique()

        # 6. Future-window skeleton rows + forward-lookable is_earnings_day from finance_calendars.
        future_df = self._build_future_skeleton(list(high_volume_symbols), max_date_needed)
        joined_df = pd.concat([wide_df, future_df], ignore_index=True, sort=False)

        # 7. Fill future NaN values (mirrors the previous module's post-join cleanup).
        curr_ts = np.datetime64(self.curr_date.replace(tzinfo=None))
        if 'open' in joined_df.columns:
            future_mask = (joined_df['timestamp'] > curr_ts) & (joined_df['open'].isna())
        else:
            future_mask = joined_df['timestamp'] > curr_ts

        if 'close_log_return' in joined_df.columns:
            joined_df.loc[future_mask, 'close_log_return'] = 0.0

        zero_fill_cols = [c for c in _FUTURE_ZERO_FILL if c in joined_df.columns]
        if zero_fill_cols:
            joined_df[zero_fill_cols] = joined_df[zero_fill_cols].fillna(0.0)

        if 'is_earnings_day' in joined_df.columns:
            joined_df['is_earnings_day'] = joined_df['is_earnings_day'].fillna(False).astype(bool).astype(int)

        # 8. Derive time features post-pivot (covers past + future rows).
        if 'year' in self.features:
            joined_df['year'] = joined_df['timestamp'].dt.year
        if 'month' in self.features:
            joined_df['month'] = joined_df['timestamp'].dt.month
        if 'day_of_month' in self.features:
            joined_df['day_of_month'] = joined_df['timestamp'].dt.day
        if 'day_of_week' in self.features:
            joined_df['day_of_week'] = joined_df['timestamp'].dt.weekday

        # 9. Trim to model's expected window, then assign time_idx and cast categoricals.
        joined_df = self._filter_by_lookback_and_lookforward(joined_df)

        joined_df['time_idx'] = factorize(joined_df['timestamp'], sort=True)[0].astype('int64')
        joined_df = joined_df.sort_values(['time_idx', 'symbol'])

        for col in joined_df.columns:
            if col in self.known_categoricals:
                joined_df[col] = joined_df[col].astype(int).astype(str).astype('category')

        keep_cols = ['symbol', 'timestamp', 'time_idx']
        for c in self.features:
            if c not in keep_cols and c in joined_df.columns:
                keep_cols.append(c)
        if 'close' in joined_df.columns and 'close' not in keep_cols:
            keep_cols.append('close')

        self.df = joined_df[keep_cols]

    def _features_to_pull_from_postgres(self) -> list[str]:
        feats = {f for f in self.features if f not in _TIME_FEATURES}
        feats.update(_ALWAYS_PULLED)
        return sorted(feats)

    def _pull_and_pivot(self, features: list[str], start_date: datetime, end_date: datetime) -> DataFrame:
        start = datetime(start_date.year, start_date.month, start_date.day)
        end = datetime(end_date.year, end_date.month, end_date.day)

        long_df = self.postgres_util.get_features_data(
            self.symbols, features, self.time_frame, start_date=start, end_date=end,
        )

        if long_df.empty:
            return DataFrame(columns=['symbol', 'timestamp'])

        # Bool-typed features come back as 'True'/'False' strings.
        bool_mask = long_df['value'].isin(['True', 'False'])
        long_df['value'] = long_df['value'].where(~bool_mask, long_df['value'].map({'True': 1, 'False': 0}))
        long_df['value'] = to_numeric(long_df['value'])

        long_df['timestamp'] = pd.to_datetime(long_df['timestamp'], utc=True).dt.tz_convert(None).dt.normalize()

        long_df = long_df.drop_duplicates(subset=['symbol', 'timestamp', 'feature'], keep='first')
        wide_df = long_df.pivot(index=['symbol', 'timestamp'], columns='feature', values='value').reset_index()
        wide_df.columns.name = None
        return wide_df

    def _calendar_date_for_trading_days(self, n: int, direction: int, include_holidays: bool) -> datetime:
        nyse_h = holidays.financial_holidays('NYSE') if include_holidays else None
        cur = self.curr_date
        counted = 0
        while counted < n:
            cur = cur + timedelta(days=direction)
            if cur.weekday() < 5 and (nyse_h is None or cur.date() not in nyse_h):
                counted += 1
        return cur

    def _missing_trading_days(self, wide_df: DataFrame, start_date: datetime, end_date: datetime) -> list:
        nyse_h = holidays.financial_holidays('NYSE')

        start_d = start_date.date()
        end_d = end_date.date()

        expected = set()
        cur = start_d
        while cur <= end_d:
            if cur.weekday() < 5 and cur not in nyse_h:
                expected.add(cur)
            cur += timedelta(days=1)

        if wide_df.empty:
            return sorted(expected)

        present = set(pd.to_datetime(wide_df['timestamp']).dt.date)
        return sorted(expected - present)

    def _build_future_skeleton(self, symbols: list[str], max_date_needed: datetime) -> DataFrame:
        if not symbols:
            return DataFrame(columns=['symbol', 'timestamp', 'is_earnings_day', 'surprise'])

        future_start = self.curr_date.date() + timedelta(days=1)
        future_end = max_date_needed.date()

        future_weekdays = []
        cur = future_start
        while cur <= future_end:
            if cur.weekday() < 5:
                future_weekdays.append(cur)
            cur += timedelta(days=1)

        if not future_weekdays:
            return DataFrame(columns=['symbol', 'timestamp', 'is_earnings_day', 'surprise'])

        earnings_by_date: dict = {}
        for d in future_weekdays:
            try:
                series = self.get_historical_data_util.get_surprise(datetime(d.year, d.month, d.day))
            except Exception as e:
                logging.warning(f'failed to pull future earnings for {d}: {e}')
                series = pd.Series(dtype=float)
            if not series.empty:
                series = series[~series.index.duplicated(keep='first')]
                earnings_by_date[d] = set(series.index)
            else:
                earnings_by_date[d] = set()

        rows = []
        for sym in symbols:
            for d in future_weekdays:
                rows.append({
                    'symbol': sym,
                    'timestamp': pd.Timestamp(d),
                    'is_earnings_day': sym in earnings_by_date[d],
                    'surprise': 0.0,
                })

        return DataFrame(rows)

    def construct_inference_dataset(self, params) -> TimeSeriesDataSet:
        return TimeSeriesDataSet.from_parameters(
            params,
            self.df,
            predict=True,
            stop_randomization=True,
        )

    def construct_inference_dataloader(self, inference_dataset: TimeSeriesDataSet,
                                       batch_size: int, num_workers: int, pin_memory: bool) -> DataLoader:
        return inference_dataset.to_dataloader(train=False, batch_size=batch_size,
                num_workers=num_workers, pin_memory=pin_memory, persistent_workers=(num_workers > 0))

    def _filter_by_lookback_and_lookforward(self, df: DataFrame) -> DataFrame:
        curr_ts = np.datetime64(self.curr_date.replace(tzinfo=None))

        st = df[['symbol', 'timestamp']].drop_duplicates()
        st['rank'] = st.groupby('symbol')['timestamp'].rank(method='dense')

        last_rank = (
            st.loc[st['timestamp'] <= curr_ts]
              .groupby('symbol')['rank']
              .max()
              .rename('last_rank')
        )

        st = st.merge(last_rank, on='symbol', how='left')

        keep_st = st[
            st['last_rank'].notna()
            & (st['rank'] >= st['last_rank'] - self.max_lookback_period + 1)
            & (st['rank'] <= st['last_rank'] + self.max_prediction_length)
        ]

        return df.merge(keep_st[['symbol', 'timestamp']], on=['symbol', 'timestamp'], how='inner')


if __name__ == '__main__':
    with open('../constants/symbols.txt', 'r') as f:
        symbols = [line.strip() for line in f]

    obj = InferenceDataModule(
        symbols[:10],
        ['close_log_return', 'day_of_week', 'day_of_month', 'month', 'year', 'surprise', 'is_earnings_day',
         'range', 'body', 'lower_wick', 'upper_wick', 'vix_log', 'close', 'high', 'low', 'open', 'volume'],
        TimeFrame(1, TimeFrameUnit.Day),
        90, 14, target='close_log_return',
    )
