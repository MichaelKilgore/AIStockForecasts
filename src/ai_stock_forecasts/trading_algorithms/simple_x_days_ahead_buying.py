from typing import DefaultDict
from pandas import DataFrame
import numpy as np
import pandas as pd
from scipy.stats import norm

from ai_stock_forecasts.trading_algorithms.base_trading_module import BaseTradingModule

"""
Simulates purchasing x stocks every y days over the entire predictions df passed.
"""
class SimpleXDaysAheadBuying(BaseTradingModule): 
    def __init__(self, interval_days: int=1, num_stocks_purchased: int=10,
                 capital_gains_tax: float=0.35, compound_money:bool=True,
                 dont_buy_negative_stocks:bool=True, uncertainty_multiplier:float=0.4):
        super().__init__()
        
        self.interval_days: int = interval_days
        self.num_stocks_purchased: int = num_stocks_purchased
        self.capital_gains_tax: float = capital_gains_tax
        self.compound_money: bool = compound_money
        self.dont_buy_negative_stocks: bool = dont_buy_negative_stocks
        self.uncertainty_multiplier: float = uncertainty_multiplier
        print(f"set trading params to: interval_days: {self.interval_days}, num_stocks_purchased: {self.num_stocks_purchased}, capital_gains_tax: {self.capital_gains_tax}, compound_money: {self.compound_money}, dont_buy_negative_stocks: {self.dont_buy_negative_stocks}, uncertainty_multiplier: {self.uncertainty_multiplier}")


    """
        We expect predictions in this format:

            "symbol": [ AAPL, AMD, ... ],
            "timestamp": [ ts1, ts2, ... ],
            "current_y": [ price, price, ... ],
            "y": [ [ ], [ ]... ],
            "y_pred_p30": [ [ ], [ ]... ],
            "y_pred_p50": [ [ ], [ ]... ],
            "y_pred_p70": [ [ ], [ ]... ],
            "open": [ price, price, ... ],

        predicting_raw_num means that we are predicting something like 'open' which means
        in order for us to understand the models forecast we have to check y against current_y

        where-as for example a target of open_log_return. open_log_return is just log(open - open of prev day)
        so we don't need to do any check we can just sort by the p50.

    """
    def simulate(self, predictions: DataFrame, predicting_raw_num: bool) -> tuple:
        period_returns = []
        total_money = []
        money_made_per_day = DefaultDict(int)
        money = self.starting_money
        timestamps = (
            pd.Series(predictions["timestamp"])
            .drop_duplicates()
            .sort_values()
            .reset_index(drop=True)
        )

        current_ts = timestamps.min()
        while True:
            mask = predictions["timestamp"] == current_ts
            filtered = predictions.loc[mask].copy()

            top_x = self._determine_top_x(filtered, predicting_raw_num)

            total_profit = 0
            if len(top_x) > 0 and predicting_raw_num:
                for row, curr_price in zip(top_x["y"], top_x["current_y"]):
                    money_to_invest = money if self.compound_money else self.starting_money
                    money_invested_in_this_stock = (money_to_invest / len(top_x))
                    num_stocks_bought = ( money_invested_in_this_stock / curr_price )
                    stock_value_now = row[self.interval_days-1]
                    money_post_sell = (num_stocks_bought * stock_value_now)
                    total_profit += money_post_sell - (money_to_invest / len(top_x))
            elif len(top_x) > 0:
                i = timestamps[timestamps == current_ts].index[0]
                if i + self.interval_days >= len(timestamps):
                    break
                future_ts = timestamps[i + self.interval_days]
                future_mask = predictions["timestamp"] == future_ts

                future_price = predictions.loc[future_mask].copy()
                future_price['future_price'] = future_price['open']

                top_x['curr_price'] = top_x['open']
                top_x = top_x[['symbol', 'curr_price']].merge(
                        future_price[['symbol', 'future_price']],
                        on=['symbol'], how='inner')

                for _, curr_price, future_price in zip(top_x['symbol'], top_x['curr_price'], top_x['future_price']):
                    money_to_invest = money if self.compound_money else self.starting_money
                    money_invested_in_this_stock = (money_to_invest / len(top_x))
                    num_stocks_bought = ( money_invested_in_this_stock / curr_price )
                    stock_value_now = future_price
                    money_post_sell = (num_stocks_bought * stock_value_now)
                    total_profit += money_post_sell - (money_to_invest / len(top_x))

            period_returns.append(total_profit / money)
            total_money.append(money)
            # print(f"Total Profit (investing $500 per stock): {total_profit}")
            money += total_profit
            money_made_per_day[current_ts.day_name()] += total_profit

            # go to 7 days later
            i = timestamps[timestamps == current_ts].index[0]
            if i + self.interval_days >= len(timestamps):
                break
            current_ts = timestamps[i + self.interval_days]

        # print(f"Total money made over 2024: {money - 25000}")
        absolute_difference = money - self.starting_money
        money_left_post_tax = money - (absolute_difference*self.capital_gains_tax)
        difference = ((money_left_post_tax - self.starting_money) / self.starting_money) * 100
        print(f"money left post tax: {money_left_post_tax}")
        print(f"money made per day: {money_made_per_day}")

        sharpe, p_value = self._calculate_sharpe_ratio_and_p_value(period_returns)
        print(f"annual sharpe is: {sharpe}")

        return difference, sharpe, p_value

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


    def _determine_top_x(self, predictions: DataFrame, predicting_raw_num: bool) -> DataFrame:
        p30 = np.stack(predictions["y_pred_p30"].to_numpy())
        p50 = np.stack(predictions["y_pred_p50"].to_numpy())
        p70 = np.stack(predictions["y_pred_p70"].to_numpy())

        if predicting_raw_num:
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
            if (self.dont_buy_negative_stocks):
                top_x = top_x[top_x["score"] > 0.0]

            return top_x
        else:
            eps = 1e-8
            x = p50[:, self.interval_days-1]
            band_width_pct = np.abs(p70 - p30)
            y = band_width_pct.mean(axis=1)

            score = x - (y * self.uncertainty_multiplier)

            predictions.loc[:, "x"] = x
            predictions.loc[:, "z"] = y
            predictions.loc[:, "score"] = score

            top_x = predictions.sort_values("score", ascending=False).head(self.num_stocks_purchased)
            if (self.dont_buy_negative_stocks):
                top_x = top_x[top_x["score"] > 0.0]

            return top_x



