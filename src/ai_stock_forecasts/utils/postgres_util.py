import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import psycopg2
from alpaca.data import TimeFrame, TimeFrameUnit
from codetiming import Timer
from dotenv import load_dotenv
from psycopg2.extras import execute_values

from ai_stock_forecasts.models.historical_data import HistoricalData


_FEATURES_COLUMNS = (
    "symbol",
    "timestamp",
    "feature",
    "value",
    "type",
    "updated_timestamp",
    "time_frame",
    "date",
)


def _time_frame_to_str(tf: TimeFrame) -> str:
    unit = tf.unit_value.value
    return unit if tf.amount_value == 1 else f"{tf.amount_value}-{unit}"


def _as_utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


class PostgresUtil:
    def __init__(self):
        load_dotenv()

        self.conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "127.0.0.1"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "postgres"),
            dbname=os.getenv("POSTGRES_DB", "ai_stock_forecasts"),
        )

    def upload_features_data_df(self, df: pd.DataFrame, time_frame):
        if df.empty:
            return

        tf_str = _time_frame_to_str(time_frame)

        ts = pd.to_datetime(df["timestamp"], utc=True)
        updated_ts = pd.to_datetime(df["updated_timestamp"], utc=True)
        date_col = pd.to_datetime(df["date"]).dt.date

        rows = list(zip(
            df["symbol"].astype(str),
            ts.dt.to_pydatetime(),
            df["feature"].astype(str),
            df["value"].astype(str),
            df["type"].astype(str),
            updated_ts.dt.to_pydatetime(),
            [tf_str] * len(df),
            date_col,
        ))

        with self.conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO historical_features
                    (symbol, timestamp, feature, value, type, updated_timestamp, time_frame, date)
                VALUES %s
                """,
                rows,
                page_size=1000,
            )
        self.conn.commit()

        logging.info(f"inserted {len(rows)} rows into historical_features (time_frame={tf_str})")

    @Timer(name='PostgresUtil.get_features_data', text='{name} took {seconds:.2f}s', logger=logging.info)
    def get_features_data(
        self,
        symbols: List[str],
        features: List[str],
        time_frame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        tf_str = _time_frame_to_str(time_frame)

        logging.info(
            f"pulling feature data from postgres for features: {features}, "
            f"time_frame: {tf_str}, symbol_count: {len(symbols)}, "
            f"start_date: {start_date}, end_date: {end_date}"
        )

        query = """
            SELECT symbol, timestamp, feature, value, type, updated_timestamp, time_frame, date
            FROM historical_features
            WHERE feature   = ANY(%s)
              AND time_frame = %s
              AND symbol    = ANY(%s)
        """
        params: List[Any] = [list(features), tf_str, list(symbols)]

        if start_date is not None:
            query += " AND timestamp >= %s"
            params.append(_as_utc(start_date))

        if end_date is not None:
            query += " AND timestamp <= %s"
            params.append(_as_utc(end_date))

        with self.conn.cursor() as cur:
            cur.execute(query, tuple(params))
            rows = cur.fetchall()

        df = pd.DataFrame.from_records(rows, columns=list(_FEATURES_COLUMNS))

        if df.empty:
            return df

        if time_frame.unit_value in (TimeFrameUnit.Minute, TimeFrameUnit.Hour):
            h = pd.to_datetime(df["timestamp"], utc=True).dt.hour
            df = df[(h > 9) & (h < 16)].reset_index(drop=True)

        return df

    def add_transaction(self, model_id: str, symbol: str, timestamp: datetime, price: float, count: int, side: str):
        assert side in ('buy', 'sell'), f"side must be 'buy' or 'sell', got: {side}"

        logging.info(f'inserting transaction model_id: {model_id}, symbol: {symbol}, timestamp: {timestamp}, price: {price}, count: {count}, side: {side}')

        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO transactions (model_id, symbol, timestamp, price, count, side)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (model_id, symbol, timestamp, price, count, side),
            )
        self.conn.commit()


    def get_models_ranked_by_avg_weekly_performance(self) -> List[Dict[str, Any]]:
        query = """
            WITH buys AS (
                SELECT model_id, symbol, timestamp, price
                FROM transactions
                WHERE side = 'buy'
            ),
            matched AS (
                SELECT
                    b.model_id,
                    b.symbol,
                    s.timestamp AS sell_ts,
                    ((s.price - b.price) / b.price) * 100 AS pct_change
                FROM buys b
                JOIN LATERAL (
                    SELECT timestamp, price
                    FROM transactions s
                    WHERE s.side = 'sell'
                      AND s.model_id = b.model_id
                      AND s.symbol = b.symbol
                      AND s.timestamp > b.timestamp
                    ORDER BY s.timestamp ASC
                    LIMIT 1
                ) s ON TRUE
            ),
            weekly_avg AS (
                SELECT
                    model_id,
                    date_trunc('week', sell_ts) AS week,
                    AVG(pct_change) AS avg_weekly_pct
                FROM matched
                GROUP BY model_id, date_trunc('week', sell_ts)
            )
            SELECT
                model_id,
                AVG(avg_weekly_pct) AS avg_weekly_performance,
                COUNT(*) AS weeks_traded
            FROM weekly_avg
            GROUP BY model_id
            ORDER BY avg_weekly_performance DESC
        """

        with self.conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()

        return [
            {
                'model_id': model_id,
                'avg_weekly_performance': float(avg_weekly_performance),
                'weeks_traded': int(weeks_traded),
            }
            for model_id, avg_weekly_performance, weeks_traded in rows
        ]


    def get_model_weekly_performance(self, model_id: str) -> List[Dict[str, Any]]:
        query = """
            WITH buys AS (
                SELECT model_id, symbol, timestamp, price
                FROM transactions
                WHERE side = 'buy' AND model_id = %s
            ),
            matched AS (
                SELECT
                    b.symbol,
                    s.timestamp AS sell_ts,
                    ((s.price - b.price) / b.price) * 100 AS pct_change
                FROM buys b
                JOIN LATERAL (
                    SELECT timestamp, price
                    FROM transactions s
                    WHERE s.side = 'sell'
                      AND s.model_id = b.model_id
                      AND s.symbol = b.symbol
                      AND s.timestamp > b.timestamp
                    ORDER BY s.timestamp ASC
                    LIMIT 1
                ) s ON TRUE
            )
            SELECT
                to_char(date_trunc('week', sell_ts), 'YYYY-MM-DD') AS week,
                AVG(pct_change) AS avg_weekly_pct,
                COUNT(*) AS num_symbols
            FROM matched
            GROUP BY date_trunc('week', sell_ts)
            ORDER BY date_trunc('week', sell_ts) ASC
        """

        with self.conn.cursor() as cur:
            cur.execute(query, (model_id,))
            rows = cur.fetchall()

        return [
            {
                'week': week,
                'avg_weekly_pct': float(avg_weekly_pct),
                'num_symbols': int(num_symbols),
            }
            for week, avg_weekly_pct, num_symbols in rows
        ]


    def get_model_last_week_symbol_performance(self, model_id: str) -> List[Dict[str, Any]]:
        query = """
            WITH buys AS (
                SELECT model_id, symbol, timestamp, price
                FROM transactions
                WHERE side = 'buy' AND model_id = %s
            ),
            matched AS (
                SELECT
                    b.symbol,
                    b.price AS buy_price,
                    s.timestamp AS sell_ts,
                    s.price AS sell_price,
                    ((s.price - b.price) / b.price) * 100 AS pct_change
                FROM buys b
                JOIN LATERAL (
                    SELECT timestamp, price
                    FROM transactions s
                    WHERE s.side = 'sell'
                      AND s.model_id = b.model_id
                      AND s.symbol = b.symbol
                      AND s.timestamp > b.timestamp
                    ORDER BY s.timestamp ASC
                    LIMIT 1
                ) s ON TRUE
            ),
            latest_week AS (
                SELECT date_trunc('week', MAX(sell_ts)) AS week FROM matched
            )
            SELECT
                m.symbol,
                m.buy_price,
                m.sell_price,
                m.pct_change
            FROM matched m, latest_week lw
            WHERE date_trunc('week', m.sell_ts) = lw.week
            ORDER BY m.pct_change DESC
        """

        with self.conn.cursor() as cur:
            cur.execute(query, (model_id,))
            rows = cur.fetchall()

        return [
            {
                'symbol': symbol,
                'buy_price': float(buy_price),
                'sell_price': float(sell_price),
                'pct_change': float(pct_change),
            }
            for symbol, buy_price, sell_price, pct_change in rows
        ]


if __name__ == '__main__':
    u = PostgresUtil()

    u.add_transaction('test-model', 'AAPL', datetime.now(), 192.34, 5, 'buy')

    print(u.get_models_ranked_by_avg_weekly_performance())

    print(u.get_model_weekly_performance('test-model'))

    print(u.get_model_last_week_symbol_performance('test-model'))
