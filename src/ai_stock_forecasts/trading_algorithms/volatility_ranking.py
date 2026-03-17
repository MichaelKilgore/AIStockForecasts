from collections import defaultdict

from datetime import timedelta
from ai_stock_forecasts.models.day_of_week import DayOfWeek
from ai_stock_forecasts.trading_algorithms.base_trading_module import BaseTradingModule
import pandas as pd
import numpy as np
import logging


class VolatilityRanking(BaseTradingModule):

    def __init__(self, num_stocks_purchased: int=10, day_of_week: DayOfWeek=DayOfWeek.tuesday, volatility_importance: float=0.3):
        super().__init__()

        self.num_stocks_purchased = num_stocks_purchased
        self.day_of_week = day_of_week
        self.volatility_importance = volatility_importance
        logging.info(f'set trading params to: num_stocks_purchased: {self.num_stocks_purchased}, day_of_week: {self.day_of_week}, volatility_importance: {self.volatility_importance}')


    def generate_buy_list(self, predictions: pd.DataFrame) -> pd.DataFrame:
        top_x = self._determine_top_x(predictions)

        return top_x

    """
        We expect predictions in this format (there will only be one timestamp though):

        "symbol": [ AAPL, AMD, ... ],
        "timestamp": [ ts1, ts2, ... ],
        "current_y": [ price, price, ... ],
        "y": [ [ ], [ ]... ],
        "y_pred_p30": [ [ ], [ ]... ],
        "y_pred_p50": [ [ ], [ ]... ],
        "y_pred_p70": [ [ ], [ ]... ],
        "close": [ price, price, ... ],

    """
    def _determine_top_x(self, df: pd.DataFrame) -> pd.DataFrame:

        df['avg_band_width'] = df.apply(
            lambda row: np.mean(np.array(row['y_pred_p70']) - np.array(row['y_pred_p30'])),
            axis=1
        )
        # least volatile first
        df['volatility_rank'] = df['avg_band_width'].rank(method='first', ascending=True).astype(int)

        p50 = np.stack(df['y_pred_p50'].values)

        df['profitability_score'] = p50[:, 0:5].sum(axis=1)

        # filter out non profitable stocks
        df = df[df['profitability_score'] > 0.0].copy()

        # most profitable first
        df["profitability_rank"] = df["profitability_score"].rank(
            ascending=False, method="first"
        ).astype(int)

        df['score'] = (
            df['volatility_rank'] * self.volatility_importance +
            df['profitability_rank'] * (1 - self.volatility_importance)
        )

        top_x = df.sort_values('score', ascending=True).head(self.num_stocks_purchased)

        return top_x








