import logging
import os
from datetime import datetime

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


if __name__ == '__main__':
    u = PostgresUtil()

    u.add_transaction('test-model', 'AAPL', datetime.now(), 192.34, 5, 'buy')
