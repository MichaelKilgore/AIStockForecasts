
from datetime import datetime, timedelta
from typing import Optional, Union

from alpaca.data import TimeFrame, TimeFrameUnit
from ai_stock_forecasts.models.order import Order, OrderItem
from pandas import DataFrame, factorize
from pytorch_forecasting import TimeSeriesDataSet
from ai_stock_forecasts.backfill.backfill_features_util import BackfillFeaturesUtil
from ai_stock_forecasts.utils.date_util import get_next_market_open_day
from ai_stock_forecasts.utils.get_historical_data_util import GetHistoricalDataUtil
from ai_stock_forecasts.data.data_module import DataModule
from ai_stock_forecasts.models.stock_bar import StockBar
import pandas as pd
import numpy as np
import time

# TODO: We need to make the data construction more generic and extendable. Right now data construction is hard coded. I.E. we expect certain set of features.
class InferenceDataModule(DataModule):
    def __init__(self, symbols: list[str], features: list[str], time_frame: Union[TimeFrame, str],
                 max_lookback_period: int, max_prediction_length: int, target: str='open', curr_date: Optional[datetime] = None,
                 volume_filter: int=100000):
        self.curr_date = curr_date if curr_date else datetime.now()

        self.get_historical_data_util = GetHistoricalDataUtil()
        self.backfill_features_util = BackfillFeaturesUtil()
        self.volume_filter = volume_filter
        super().__init__(symbols, features, time_frame, max_lookback_period, max_prediction_length, target=target)

    """
    1. get yfinance data necessary
    2. get nasdaq features
    3. construct features

    goal:

    DF:
        symbol, timestamp, time_idx, close_log_return, day_of_week, day_of_month, month, year, surprise, is_earnings_day
    """
    def _construct_df(self):
        """ We check twice as far back as we technically need because each time step is not a single day its a single day
            the stock market is open. To make sure we get all the data we need multiplying by 2 should cover us for all use cases."""
        if (self.time_frame.amount_value == 1 and self.time_frame.unit_value == TimeFrameUnit.Day):
            min_date_needed = self.curr_date - timedelta(days=self.max_lookback_period * 2 + 5)
            max_date_needed = self.curr_date + timedelta(days=self.max_prediction_length * 2 + 5)
        else: # TODO: Add support for other timeframes
            raise Exception(f'The time_frame: {self.time_frame} is not supported')

        # stock_bars = self.get_historical_data_util.batch_get_historical_stock_prices(self.symbols[:2500], min_date_needed, self.curr_date, self.time_frame)
        stock_bars = self.get_historical_data_util.get_historical_stock_prices(self.symbols, min_date_needed, self.curr_date, self.time_frame)
        """ structure:
                        symbol, timestamp, close, other features..."""
        base_df = self._form_starting_df_from_base_features(stock_bars)

        # base_df = base_df
        high_volume_symbols = base_df.groupby('symbol').mean()
        high_volume_symbols = high_volume_symbols[high_volume_symbols['volume'] >= self.volume_filter]

        base_df = base_df[base_df['symbol'].isin(high_volume_symbols.index)]

        base_df = base_df.sort_values(['symbol', 'timestamp'])

        # calculate close_log_return
        base_df['close_log_return'] = np.log(
            base_df['close'].astype(float) /
                base_df.groupby('symbol')['close'].shift(1).astype(float)
        ).astype(float)
        base_df = base_df[base_df['close_log_return'] != 'nan']


        base_df['range'] = base_df['high'] - base_df['low']
        base_df['body'] = base_df['close'] - base_df['open']
        base_df['lower_wick'] = np.minimum(base_df['close'], base_df['open']) - base_df['low']
        base_df['upper_wick'] = base_df['high'] - np.maximum(base_df['open'], base_df['close'])

        # we get rate limited by yfinance for previous download if it was large
        time.sleep(30)
        # vix_log
        vix_records = self.get_historical_data_util.get_historical_vix('1y')

        vix_records['Date'] = vix_records['Date'].dt.tz_localize(None)

        vix_records['vix_log'] = np.log(vix_records['Close']).astype(str)

        vix_records['timestamp'] = vix_records['Date']

        vix_records['timestamp'] = pd.to_datetime(vix_records['timestamp'], utc=True).dt.tz_convert(None).dt.normalize()

        vix_records = vix_records[['timestamp', 'vix_log']]

        vix_records.set_index(['timestamp'])

        base_df['timestamp'] = pd.to_datetime(base_df['timestamp'], utc=True).dt.tz_convert(None).dt.normalize()
        # TODO: this is wrong base_df timestamp is 0 - 1300
        base_df = base_df.merge(
            vix_records,
            on="timestamp",
            how="left"
        )

        """for some of the feature data we don't have the same hour, min listed in the timestamp
            so we have to make sure we are grouping by correct time frame unit.

            TODO: we don't have support for min or hour timeframe, grouping seemed more confusing so leaving unsupported for now."""
        if self.time_frame.unit_value == TimeFrameUnit.Day:
            base_df['timestamp'] = pd.to_datetime(base_df['timestamp'], utc=True).dt.tz_convert(None).dt.normalize()
        else:
            raise Exception(f'TimeFrame: {self.time_frame} not supported')
        base_df = base_df.set_index(['symbol', 'timestamp'])

        """ structure:
                symbol, timestamp, surprise (is 0.0 if checking future date), is_earnings_day"""
        surprise_features = self.backfill_features_util.backfill_surprise_features(self.symbols, min_date_needed, max_date_needed, self.time_frame, return_df=True)

        """for some of the feature data we don't have the same hour, min listed in the timestamp
            so we have to make sure we are grouping by correct time frame unit.

            TODO: we don't have support for min or hour timeframe, grouping seemed more confusing so leaving unsupported for now."""
        if self.time_frame.unit_value == TimeFrameUnit.Day:
            surprise_features['timestamp'] = pd.to_datetime(surprise_features['timestamp'], utc=True).dt.tz_convert(None).dt.normalize()
        else:
            raise Exception(f'TimeFrame: {self.time_frame} not supported')
        surprise_features = surprise_features.set_index(['symbol', 'timestamp'])


        joined_df = base_df.join(surprise_features, how='outer').reset_index(drop=False)

        joined_df = joined_df[joined_df['symbol'].isin(high_volume_symbols.index)]

        # Filters out nan rows in the past (weekends) and rows in the future that are weekends
        curr_date = np.datetime64(self.curr_date.replace(tzinfo=None))
        joined_df = joined_df[ ( (joined_df['close_log_return'].notna()) & (joined_df['timestamp'] <= curr_date) ) | ( (joined_df['timestamp'] > curr_date) & (joined_df['timestamp'].dt.weekday < 5) )]

        # add close_log_return

        # Filters down to only the rows we need for lookback and lookforward
        joined_df = self._filter_by_lookback_and_lookforward(joined_df)

        # we have to set target in future to non NaN otherwise constructing the TimeSeries Dataset fails
        mask = (joined_df["timestamp"] > curr_date) & (joined_df["open"].isna())
        joined_df.loc[mask, "close_log_return"] = 0.0
        cols = ["range", "body", "lower_wick", "upper_wick", "vix_log"]

        joined_df[cols] = joined_df[cols].fillna(0.0)


        # include year and month feature
        if (self.time_frame.unit == TimeFrameUnit.Month 
                or self.time_frame.unit == TimeFrameUnit.Week
                or self.time_frame.unit == TimeFrameUnit.Day
                or self.time_frame.unit == TimeFrameUnit.Hour
                or self.time_frame.unit == TimeFrameUnit.Minute):
            if 'year' in self.features:
                joined_df['year'] = joined_df['timestamp'].dt.year
            if 'month' in self.features:
                joined_df['month'] = joined_df['timestamp'].dt.month

        # include day_of_month and day_of_week feature
        if (self.time_frame.unit == TimeFrameUnit.Day
                or self.time_frame.unit == TimeFrameUnit.Hour
                or self.time_frame.unit == TimeFrameUnit.Minute):
            if 'day_of_month' in self.features:
                joined_df['day_of_month'] = joined_df['timestamp'].dt.day
            if 'day_of_week' in self.features:
                joined_df['day_of_week'] = joined_df['timestamp'].dt.weekday

        # include hour_of_day feature
        if (self.time_frame.unit == TimeFrameUnit.Hour
                or self.time_frame.unit == TimeFrameUnit.Minute):
            if 'hour_of_day' in self.features:
                joined_df['hour_of_day'] = joined_df['timestamp'].dt.hour

        # include minute_of_day feature
        if (self.time_frame.unit == TimeFrameUnit.Hour
                or self.time_frame.unit == TimeFrameUnit.Minute):
            if 'minute_of_day' in self.features:
                joined_df['minute_of_day'] = joined_df['timestamp'].dt.minute

        joined_df['time_idx'] = factorize(joined_df['timestamp'], sort=True)[0].astype('int64')
        joined_df = joined_df.sort_values(['time_idx', 'symbol'])

        for col in joined_df.columns:
            if col in self.known_categoricals:
                joined_df[col] = joined_df[col].astype(int).astype(str).astype('category')

        joined_df = joined_df[['symbol', 'timestamp', 'close_log_return', 'range', 'body', 'lower_wick', 'upper_wick', 'vix_log', 'surprise', 'is_earnings_day', 'year', 'month', 'day_of_month', 'day_of_week', 'time_idx', 'close']]
        self.df = joined_df


    def construct_inference_dataset(self, params):
        self.inference_dataset = TimeSeriesDataSet.from_parameters(
            params,
            self.df,
            predict=True,
            stop_randomization=True,
        )

    def construct_inference_dataloader(self, batch_size: int, num_workers: int, pin_memory: bool):
        if self.inference_dataset == None:
            raise Exception("You have to construct the inference_dataset before executing this function")

        self.inference_dataloader = self.inference_dataset.to_dataloader(train=False, batch_size=batch_size,
                num_workers=num_workers, pin_memory=pin_memory, persistent_workers=(num_workers > 0)) 

    def is_it_time_to_order_again(self, latest_order_timestamp: datetime, interval_days: int=1):
        ts = pd.to_datetime(self.df["timestamp"], errors="coerce")

        latest_day = pd.Timestamp(latest_order_timestamp).normalize()
        curr_day = pd.Timestamp.now().normalize()

        s = self.df.loc[ts.le(latest_day), ["time_idx"]].sort_index()
        last_order_time_idx = s.iloc[-1]["time_idx"]

        s = self.df.loc[ts.le(curr_day), ["time_idx"]].sort_index()
        curr_order_time_idx = s.iloc[-1]["time_idx"]

        return (curr_order_time_idx - last_order_time_idx) >= interval_days

    def update_money_to_invest(self, order: Order) -> float:
        curr_date = np.datetime64(self.curr_date.replace(tzinfo=None))

        money_left = 0
        ts = pd.to_datetime(self.df["timestamp"], errors="coerce")

        for order_item in order.order_items:
            sub = self.df.loc[(self.df["symbol"] == order_item.symbol) & ts.le(curr_date)]
            if sub.empty:
                continue

            last_row = sub.loc[sub["time_idx"].idxmax()]
            latest_open = last_row["close_log_return"]

            money_left += float(latest_open) * float(order_item.quantity)

        return money_left
 

    def _form_starting_df_from_base_features(self, stock_bars: list[StockBar]) -> DataFrame:
        rows = []

        for stock_bar in stock_bars:
            new_row = {}

            new_row['symbol'] = stock_bar.symbol
            new_row['timestamp'] = stock_bar.timestamp

            for feature in self._get_alpaca_sourced_features():
                new_row[feature] = getattr(stock_bar, feature)

            rows.append(new_row)

        return DataFrame(rows)

    def _filter_by_lookback_and_lookforward(self, df: DataFrame) -> DataFrame:
        curr_ts = np.datetime64(self.curr_date)

        # unique (symbol, timestamp) pairs
        st = df[["symbol", "timestamp"]].drop_duplicates()

        # rank timestamps within each symbol
        st["rank"] = st.groupby("symbol")["timestamp"].rank(method="dense")

        # rank of the latest timestamp <= curr_ts for each symbol
        last_rank = (
            st.loc[st["timestamp"] <= curr_ts]
              .groupby("symbol")["rank"]
              .max()
              .rename("last_rank")
        )

        st = st.merge(last_rank, on="symbol", how="left")

        # keep last 60 up to curr_ts, plus next 2 after curr_ts
        keep_st = st[
            st["last_rank"].notna()
            & (st["rank"] >= st["last_rank"] - self.max_lookback_period + 1)
            & (st["rank"] <= st["last_rank"] + self.max_prediction_length)
        ]

        # filter original df to those (symbol, timestamp)
        df_filtered = df.merge(keep_st[["symbol", "timestamp"]], on=["symbol", "timestamp"], how="inner")

        return df_filtered

if __name__ == "__main__":
    # with open('src/ai_stock_forecasts/constants/symbols.txt', 'r') as f:
    with open('../constants/symbols.txt', 'r') as f:
        symbols = [line.strip() for line in f]

    obj = InferenceDataModule(symbols[:10], ['close_log_return', 'day_of_week', 'day_of_month', 'month', 'year', 'surprise', 'is_earnings_day', 'range', 'body', 'lower_wick', 'upper_wick', 'vix_log', 'close', 'high', 'low', 'open', 'volume'], TimeFrame(1, TimeFrameUnit.Day), 90, 14, target='close_log_return')







