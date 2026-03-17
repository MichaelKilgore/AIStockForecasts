
from abc import abstractmethod
from typing import Optional

from pandas import DataFrame
import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime

from ai_stock_forecasts.models.day_of_week import DayOfWeek

from ai_stock_forecasts.utils.logging_util import verbose_log
import logging

from collections import defaultdict

class BaseTradingModule:
    starting_money = 25000
    tax_percentage = 0.35
    day_of_week: Optional[DayOfWeek]
    interval_days: Optional[int]
    predicting_raw_num: bool

    @abstractmethod
    def generate_buy_list(self, predictions: DataFrame) -> DataFrame:
        pass

    @abstractmethod
    def _determine_top_x(self, predictions: DataFrame) -> DataFrame:
        pass

    """
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
    def simulate(self, predictions: pd.DataFrame) -> tuple[float, float, float]:
        h = defaultdict(int)
        money = self.starting_money
        period_returns = []
        timestamps = (
            pd.Series(predictions["timestamp"])
            .drop_duplicates()
            .sort_values()
            .reset_index(drop=True)
        )

        curr_ts = timestamps.min()
        if self.day_of_week:
            curr_ts = self._get_next_day_of_week_timestamp(timestamps, curr_ts)

        while True:
            filtered = predictions.loc[predictions["timestamp"] == curr_ts].copy()

            top_x = self._determine_top_x(filtered)

            if self.day_of_week:
                future_ts = self._get_next_day_of_week_timestamp(timestamps, curr_ts)
            else:
                future_ts = self._get_next_timestamp(timestamps, curr_ts)

            total_profit = 0
            if len(top_x) > 0:
                future_price = predictions.loc[predictions['timestamp'] == future_ts].copy()
                future_price['future_price'] = future_price['close']

                top_x['curr_price'] = top_x['close']
                top_x = top_x[['symbol', 'curr_price']].merge(
                        future_price[['symbol', 'future_price']],
                        on=['symbol'], how='inner')

                for symbol, curr_price, future_price in zip(top_x['symbol'], top_x['curr_price'], top_x['future_price']):
                    if future_price / 4 > curr_price:
                        logging.info(f'reverse split likely, skipping symbol: {symbol}, ts: {curr_ts}, that had curr_price: {curr_price}, and future_price: {future_price}')
                        continue

                    h[symbol] += 1
                    money_invested_in_this_stock = ( self.starting_money / len(top_x) )
                    num_stocks_bought = ( money_invested_in_this_stock / curr_price )
                    money_post_sell = ( num_stocks_bought * future_price )
                    total_profit += money_post_sell - money_invested_in_this_stock

                verbose_log(f'for timestamp {curr_ts}, total_profit was: {total_profit}')
                verbose_log(f'top x for timestamp: {curr_ts}')
                verbose_log(top_x[['symbol', 'curr_price', 'future_price']])

            period_returns.append(total_profit / money)
            money += total_profit

            curr_ts = future_ts
            if not curr_ts:
                break

        stock_buys = sorted([(k,v) for k,v in h.items()], key=lambda x: x[1])
        for symbol, num_buys in stock_buys:
            verbose_log(f'purchased symbol: {symbol} {num_buys} times')

        absolute_difference = money - self.starting_money
        money_left_post_tax = money - (absolute_difference*self.tax_percentage)
        difference = ( ( money_left_post_tax - self.starting_money ) / self.starting_money ) * 100

        logging.info(f"money left post tax: {money_left_post_tax}")

        sharpe, p_value = self._calculate_sharpe_ratio_and_p_value(period_returns)
        logging.info(f"annual sharpe is: {sharpe}, p_value is: {p_value}")

        return (difference, sharpe, p_value)

    """ assumes we can generate 5% returns annually risk free.
        sharpe_annual > 1 is considered good, but greater than 2 or even 3 is ideal.
        p_two_sided of 0.11 for example means there is an 11% chance that these results could've been generated at random.
    """
    def _calculate_sharpe_ratio_and_p_value(self, period_returns) -> tuple[float, float]:
        rf_period = (1 + 0.05) ** (1 / len(period_returns)) - 1
        r = np.array(period_returns)
        excess = r - rf_period

        sharpe_daily = excess.mean() / excess.std(ddof=1)
        sharpe_annual = sharpe_daily * np.sqrt(len(period_returns))

        z = sharpe_daily * np.sqrt(len(period_returns))
        p_two_sided = 2 * norm.sf(abs(z))

        return sharpe_annual, float(p_two_sided)

    '''
        assumes timestamps is sorted in ascending order, returns None if next valid timestamp is not in the series.
    '''
    def _get_next_day_of_week_timestamp(self, timestamps: pd.Series, current_ts: np.datetime64) -> Optional[np.datetime64]:
        i = timestamps[timestamps == current_ts].index[0]
        j = 1

        while True:
            if i + j >= len(timestamps):
                return None
            if timestamps[(i + j)].day_name() == self.day_of_week.value:
                return timestamps[i + j]
                break
            j += 1

    def _get_next_timestamp(self, timestamps: pd.Series, current_ts: np.datetime64) -> Optional[np.datetime64]:
        i = timestamps[timestamps == current_ts].index[0]
        if i + self.interval_days >= len(timestamps):
            return None

        current_ts = timestamps[i + self.interval_days]

        return current_ts



