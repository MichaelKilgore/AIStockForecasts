"""
Populate the `historical_features` postgres table at daily grain for a date
range (default: today only) across every symbol in many_symbols.txt + SPY.

Features written (per symbol, per trading day in range):
  - OHLCV base:    open, close, high, low, volume, vwap, trade_count
  - Wicks/body:    upper_wick, lower_wick, body, range
  - Time:          year, month, day_of_month, day_of_week
  - Log-return:    close_log_return  (uses extra yfinance lookback for prev close)
  - Market wide:   sandp500open, vix_log  (broadcast to every symbol)
  - Earnings:      surprise, is_earnings_day

By default, rows in the requested date range for the daily time_frame are
DELETEd before inserting (idempotent reruns). Pass --append to skip the delete.

Usage:
  # today
  python3 src/ai_stock_forecasts/cron/populate_historical_features_daily_grain_psql_table.py
  # explicit range (inclusive)
  python3 .../populate_historical_features_daily_grain_psql_table.py \\
      --start_date 2026-05-01 --end_date 2026-05-14
"""
import argparse
import logging
import math
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import List

import pandas as pd
import yfinance as yf
from alpaca.data import TimeFrame, TimeFrameUnit

from ai_stock_forecasts.models.historical_data import HistoricalData
from ai_stock_forecasts.utils.date_util import get_prev_market_open_day
from ai_stock_forecasts.utils.get_historical_data_util import GetHistoricalDataUtil
from ai_stock_forecasts.utils.postgres_util import PostgresUtil


_TIME_FRAME = TimeFrame(1, TimeFrameUnit.Day)
_TIME_FRAME_STR = 'Day'
_SYMBOLS_FILE = (
    Path(__file__).resolve().parents[3]
    / 'src/ai_stock_forecasts/constants/many_symbols.txt'
)
# Extra calendar-day lookback so weekend/holiday gaps don't break prev-close lookup
_PREV_CLOSE_LOOKBACK_DAYS = 10
# yfinance rate-limit guardrails
_OHLCV_CHUNK_SIZE = 200
_OHLCV_CHUNK_SLEEP_SECONDS = 3
_YF_MAX_RETRIES = 4
_YF_RETRY_SLEEP_SECONDS = 30


class HistoricalFeaturesDailyPopulator:
    def __init__(self, start_date: date, end_date: date, replace_existing: bool = True):
        self.start_date = start_date
        self.end_date = end_date
        self.replace_existing = replace_existing

        self.symbols = self._load_symbols()
        if 'SPY' not in self.symbols:
            self.symbols.append('SPY')

        self.pg = PostgresUtil()
        self.data_util = GetHistoricalDataUtil()
        self.updated = datetime.now(timezone.utc)

    def run(self):
        if self.replace_existing:
            self._delete_range()

        bars = self._fetch_ohlcv()
        if bars.empty:
            logging.warning('no bars returned by yfinance, nothing to write')
            return

        records: List[HistoricalData] = []
        records.extend(self._build_ohlcv_records(bars))
        records.extend(self._build_close_log_return_records(bars))
        records.extend(self._build_sandp500open_records(bars))
        records.extend(self._build_vix_log_records())
        records.extend(self._build_earnings_records())

        logging.info(f'inserting {len(records)} feature records into historical_features')
        df_out = self._records_to_df(records)
        if not df_out.empty:
            self.pg.upload_features_data_df(df_out, _TIME_FRAME)
        logging.info('done')

    @staticmethod
    def _load_symbols() -> List[str]:
        with open(_SYMBOLS_FILE) as f:
            return [line.split('|')[0].strip() for line in f if line.strip()]

    def _delete_range(self):
        with self.pg.conn.cursor() as cur:
            cur.execute(
                "DELETE FROM historical_features WHERE time_frame = %s AND date BETWEEN %s AND %s",
                (_TIME_FRAME_STR, self.start_date, self.end_date),
            )
            deleted = cur.rowcount
        self.pg.conn.commit()
        logging.info(
            f'deleted {deleted} existing rows for {self.start_date}..{self.end_date} '
            f'(time_frame={_TIME_FRAME_STR})'
        )

    def _fetch_ohlcv(self) -> pd.DataFrame:
        fetch_start = self.start_date - timedelta(days=_PREV_CLOSE_LOOKBACK_DAYS)
        start_str = fetch_start.strftime('%Y-%m-%d')
        end_str = (self.end_date + timedelta(days=1)).strftime('%Y-%m-%d')

        chunks = [
            self.symbols[i:i + _OHLCV_CHUNK_SIZE]
            for i in range(0, len(self.symbols), _OHLCV_CHUNK_SIZE)
        ]
        logging.info(
            f'downloading yfinance bars for {len(self.symbols)} symbols, '
            f'{fetch_start}..{self.end_date}, in {len(chunks)} chunks of {_OHLCV_CHUNK_SIZE}'
        )

        dfs: List[pd.DataFrame] = []
        for i, chunk in enumerate(chunks):
            chunk_df = self._download_ohlcv_chunk(chunk, start_str, end_str, i + 1, len(chunks))
            if chunk_df is not None and not chunk_df.empty:
                dfs.append(chunk_df)
            if i + 1 < len(chunks):
                time.sleep(_OHLCV_CHUNK_SLEEP_SECONDS)

        if not dfs:
            return pd.DataFrame()

        df = pd.concat(dfs, ignore_index=True)
        df = df.sort_values(['symbol', 'Date']).reset_index(drop=True)
        df['prev_close'] = df.groupby('symbol')['Close'].shift(1)
        return df

    @staticmethod
    def _download_ohlcv_chunk(chunk: List[str], start_str: str, end_str: str,
                              chunk_idx: int, total_chunks: int):
        for attempt in range(1, _YF_MAX_RETRIES + 1):
            try:
                df = yf.download(
                    tickers=chunk,
                    start=start_str,
                    end=end_str,
                    group_by='ticker',
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                )
            except Exception as e:
                logging.warning(
                    f'chunk {chunk_idx}/{total_chunks} attempt {attempt}/{_YF_MAX_RETRIES} '
                    f'raised: {e}'
                )
                time.sleep(_YF_RETRY_SLEEP_SECONDS)
                continue

            if df.empty:
                logging.warning(
                    f'chunk {chunk_idx}/{total_chunks} attempt {attempt} returned empty df; retrying'
                )
                time.sleep(_YF_RETRY_SLEEP_SECONDS)
                continue

            stacked = (
                df.stack(level='Ticker')
                  .reset_index()
                  .rename(columns={'Ticker': 'symbol'})
            )
            logging.info(
                f'chunk {chunk_idx}/{total_chunks} ok ({len(chunk)} symbols, {len(stacked)} rows)'
            )
            return stacked

        logging.error(f'chunk {chunk_idx}/{total_chunks} exhausted retries; skipping')
        return None

    def _in_range(self, ts_date: date) -> bool:
        return self.start_date <= ts_date <= self.end_date

    def _record(self, symbol: str, ts: datetime, feature: str, value, type_str: str) -> HistoricalData:
        return HistoricalData(symbol, ts, feature, str(value), type_str, self.updated, _TIME_FRAME, ts)

    def _bar_records(self, symbol: str, ts: datetime, open_v: float, close_v: float,
                     high_v: float, low_v: float, volume_v: float) -> List[HistoricalData]:
        return [
            self._record(symbol, ts, 'open',         open_v,                        'float'),
            self._record(symbol, ts, 'close',        close_v,                       'float'),
            self._record(symbol, ts, 'high',         high_v,                        'float'),
            self._record(symbol, ts, 'low',          low_v,                         'float'),
            self._record(symbol, ts, 'volume',       volume_v,                      'float'),
            self._record(symbol, ts, 'vwap',         -1,                            'float'),
            self._record(symbol, ts, 'trade_count',  -1,                            'int'),
            self._record(symbol, ts, 'year',         ts.year,                       'int'),
            self._record(symbol, ts, 'month',        ts.month,                      'int'),
            self._record(symbol, ts, 'day_of_month', ts.day,                        'int'),
            self._record(symbol, ts, 'day_of_week',  ts.weekday(),                  'int'),
            self._record(symbol, ts, 'upper_wick',   high_v - max(open_v, close_v), 'float'),
            self._record(symbol, ts, 'lower_wick',   min(open_v, close_v) - low_v,  'float'),
            self._record(symbol, ts, 'body',         close_v - open_v,              'float'),
            self._record(symbol, ts, 'range',        high_v - low_v,                'float'),
        ]

    def _build_ohlcv_records(self, bars: pd.DataFrame) -> List[HistoricalData]:
        records: List[HistoricalData] = []
        for row in bars.itertuples(index=False):
            ts_date = pd.Timestamp(row.Date).date()
            if not self._in_range(ts_date):
                continue
            if any(pd.isna(v) for v in (row.Open, row.Close, row.High, row.Low)):
                continue

            ts = datetime(ts_date.year, ts_date.month, ts_date.day)
            volume_v = float(row.Volume) if pd.notna(row.Volume) else 0.0
            records.extend(self._bar_records(
                row.symbol, ts,
                float(row.Open), float(row.Close), float(row.High), float(row.Low),
                volume_v,
            ))
        return records

    def _build_close_log_return_records(self, bars: pd.DataFrame) -> List[HistoricalData]:
        records: List[HistoricalData] = []
        for row in bars.itertuples(index=False):
            ts_date = pd.Timestamp(row.Date).date()
            if not self._in_range(ts_date):
                continue
            if pd.isna(row.Close) or pd.isna(row.prev_close):
                continue
            close_v = float(row.Close)
            prev = float(row.prev_close)
            if close_v <= 0 or prev <= 0:
                continue

            ts = datetime(ts_date.year, ts_date.month, ts_date.day)
            records.append(self._record(row.symbol, ts, 'close_log_return',
                                        math.log(close_v / prev), 'float'))
        return records

    def _build_sandp500open_records(self, bars: pd.DataFrame) -> List[HistoricalData]:
        spy_open_by_date: dict = {}
        for row in bars.itertuples(index=False):
            if row.symbol != 'SPY':
                continue
            ts_date = pd.Timestamp(row.Date).date()
            if not self._in_range(ts_date):
                continue
            if pd.isna(row.Open):
                continue
            spy_open_by_date[ts_date] = float(row.Open)

        records: List[HistoricalData] = []
        for d, spy_open in spy_open_by_date.items():
            ts = datetime(d.year, d.month, d.day)
            for sym in self.symbols:
                records.append(self._record(sym, ts, 'sandp500open', spy_open, 'float'))
        return records

    def _build_vix_log_records(self) -> List[HistoricalData]:
        vix = None
        for attempt in range(1, _YF_MAX_RETRIES + 1):
            try:
                vix = self.data_util.get_historical_vix()
                break
            except Exception as e:
                logging.warning(f'vix fetch attempt {attempt}/{_YF_MAX_RETRIES} failed: {e}')
                time.sleep(_YF_RETRY_SLEEP_SECONDS)
        if vix is None:
            logging.error('failed to fetch VIX after retries; skipping vix_log feature')
            return []

        vix['Date'] = vix['Date'].dt.tz_localize(None)
        vix = vix[(vix['Date'].dt.date >= self.start_date) & (vix['Date'].dt.date <= self.end_date)]

        records: List[HistoricalData] = []
        for row in vix.itertuples(index=False):
            if pd.isna(row.Close) or row.Close <= 0:
                continue
            ts_date = pd.Timestamp(row.Date).date()
            ts = datetime(ts_date.year, ts_date.month, ts_date.day)
            vix_log = math.log(float(row.Close))
            for sym in self.symbols:
                records.append(self._record(sym, ts, 'vix_log', vix_log, 'float'))
        return records

    def _build_earnings_records(self) -> List[HistoricalData]:
        records: List[HistoricalData] = []
        curr = self.start_date
        while curr <= self.end_date:
            ts = datetime(curr.year, curr.month, curr.day)
            try:
                surprise_series = self.data_util.get_surprise(ts)
            except Exception as e:
                logging.warning(f'failed to pull surprise for {curr}: {e}')
                surprise_series = pd.Series(dtype=float)

            if not surprise_series.empty:
                surprise_series = surprise_series[~surprise_series.index.duplicated(keep='first')]
                surprise_map = pd.to_numeric(surprise_series, errors='coerce').fillna(0.0).to_dict()
                earning_symbols = set(surprise_series.index)
            else:
                surprise_map = {}
                earning_symbols = set()

            for sym in self.symbols:
                records.append(self._record(sym, ts, 'surprise', surprise_map.get(sym, 0.0), 'float'))
                records.append(self._record(sym, ts, 'is_earnings_day', sym in earning_symbols, 'bool'))
            curr += timedelta(days=1)
        return records

    @staticmethod
    def _records_to_df(records: List[HistoricalData]) -> pd.DataFrame:
        return pd.DataFrame.from_records([
            {
                'symbol': r.symbol,
                'timestamp': r.timestamp,
                'feature': r.feature,
                'value': r.value,
                'type': r.type,
                'updated_timestamp': r.updated_timestamp,
                'date': r.date,
            }
            for r in records
        ])


def _parse_date(s: str) -> date:
    return datetime.strptime(s, '%Y-%m-%d').date()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start_date', type=_parse_date, default=None,
                        help='YYYY-MM-DD (default: most recent fully-closed market day)')
    parser.add_argument('--end_date',   type=_parse_date, default=None,
                        help='YYYY-MM-DD (default: most recent fully-closed market day)')
    parser.add_argument('--append', action='store_true',
                        help='skip the delete-by-date-range step (rows just get appended)')
    args = parser.parse_args()

    default_day = get_prev_market_open_day().date()
    start = args.start_date or default_day
    end = args.end_date or default_day
    if end < start:
        parser.error('--end_date must be >= --start_date')

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    HistoricalFeaturesDailyPopulator(start, end, replace_existing=not args.append).run()


if __name__ == '__main__':
    main()
