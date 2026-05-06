import logging
import os
from datetime import datetime
from typing import Any, Dict, List

import psycopg2
from dotenv import load_dotenv


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


if __name__ == '__main__':
    u = PostgresUtil()

    u.add_transaction('test-model', 'AAPL', datetime.now(), 192.34, 5, 'buy')

    print(u.get_models_ranked_by_avg_weekly_performance())
