from typing import DefaultDict, Optional
from pandas import DataFrame
import numpy as np
import pandas as pd
from scipy.stats import norm
from collections import defaultdict

from ai_stock_forecasts.models.day_of_week import DayOfWeek
from ai_stock_forecasts.trading_algorithms.base_trading_module import BaseTradingModule

"""
Simulates purchasing x stocks every y days over the entire predictions df passed.
"""
class SimpleXDaysAheadBuying(BaseTradingModule): 
    def __init__(self, interval_days: int=1, num_stocks_purchased: int=10,
                 uncertainty_multiplier:float=0.4, filter_out_x_most_volatile: int=0,
                 predicting_raw_num: bool=False, pivot_df: Optional[DataFrame] = None, day_of_week: DayOfWeek = Optional[DayOfWeek.tuesday]):
        super().__init__()

        self.interval_days: int = interval_days
        self.num_stocks_purchased: int = num_stocks_purchased
        self.uncertainty_multiplier: float = uncertainty_multiplier
        self.filter_out_x_most_volatile=filter_out_x_most_volatile
        self.predicting_raw_num = predicting_raw_num
        self.pivot_df = pivot_df
        self.day_of_week = day_of_week
        print(f"set trading params to: interval_days: {self.interval_days}, num_stocks_purchased: {self.num_stocks_purchased}, uncertainty_multiplier: {self.uncertainty_multiplier}, filter_out_x_most_volatile: {self.filter_out_x_most_volatile}")


    """
        TODO: Need to make use of base trading module simulate

        We expect predictions in this format:

            "symbol": [ AAPL, AMD, ... ],
            "timestamp": [ ts1, ts2, ... ],
            "current_y": [ price, price, ... ],
            "y": [ [ ], [ ]... ],
            "y_pred_p30": [ [ ], [ ]... ],
            "y_pred_p50": [ [ ], [ ]... ],
            "y_pred_p70": [ [ ], [ ]... ],
            "close": [ price, price, ... ],

        predicting_raw_num means that we are predicting something like 'open' which means
        in order for us to understand the models forecast we have to check y against current_y

        where-as for example a target of open_log_return. open_log_return is just log(open - open of prev day)
        so we don't need to do any check we can just sort by the p50.

    """
    def simulate(self, predictions: DataFrame) -> tuple[float, float, float]:
        if self.filter_out_x_most_volatile != 0:
            predictions = self._filter_out_x_most_volatile(predictions)

        return super().simulate(predictions)

    """
        We expect predictions in this format (there will only be one timestamp though):

            "symbol": [ AAPL, AMD, ... ],
            "timestamp": [ ts1 ],
            "current_y": [ price, price, ... ],
            "y": [ [ ], [ ]... ],
            "y_pred_p30": [ [ ], [ ]... ],
            "y_pred_p50": [ [ ], [ ]... ],
            "y_pred_p70": [ [ ], [ ]... ],
    """
    def generate_buy_list(self, predictions: DataFrame) -> DataFrame:
        top_x = self._determine_top_x(predictions)

        return top_x


    def _determine_top_x(self, predictions: DataFrame) -> DataFrame:
        p30 = np.stack(predictions["y_pred_p30"].to_numpy())
        p50 = np.stack(predictions["y_pred_p50"].to_numpy())
        p70 = np.stack(predictions["y_pred_p70"].to_numpy())

        if self.predicting_raw_num:
            current_y = predictions['current_y']

            eps = 1e-8
            start = current_y
            end = p50[:, self.interval_days-1]

            x = (end - start) / (start + eps) * 100.0
            band_width_pct = ( (p70 - p30) / (p30 + eps) ) * 100.0
            y = band_width_pct.mean(axis=1)

            """the general idea of this scoring mechanism is that
               we factor in a combination of which stocks are predicted
               to rise the most, but also factor in uncertainty, greater
               distances between p30 and p70 suggest the stock is volatile."""
            score = x - (y * self.uncertainty_multiplier)

            predictions.loc[:, "x"] = x
            predictions.loc[:, "z"] = y
            predictions.loc[:, "score"] = score

            top_x = predictions.sort_values("score", ascending=False).head(self.num_stocks_purchased)

            top_x = top_x[top_x["score"] > 0.0]

            return top_x
        else:
            eps = 1e-8
            x = p50[:, :self.interval_days].sum(axis=1)
            band_width_pct = np.abs(p70 - p30)
            y = band_width_pct.mean(axis=1)

            score = x - (y * self.uncertainty_multiplier)

            predictions.loc[:, "x"] = x
            predictions.loc[:, "z"] = y
            predictions.loc[:, "score"] = score

            top_x = predictions.sort_values("score", ascending=False).head(self.num_stocks_purchased)

            top_x = top_x[top_x["score"] > 0.0]

            return top_x

    def _filter_out_x_most_volatile(self, predictions: DataFrame) -> DataFrame:
        print(f'filtering out {self.filter_out_x_most_volatile} most volatile...')
        top_x_symbols = (
            self.pivot_df
                .groupby('symbol')['close_log_return']
                .std()
                .sort_values(ascending=False)
                .head(self.filter_out_x_most_volatile)
                .index
        )

        filtered_predictions = predictions[~predictions['symbol'].isin(top_x_symbols)]

        return filtered_predictions
