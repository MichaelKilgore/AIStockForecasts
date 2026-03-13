

from datetime import timedelta
from ai_stock_forecasts.models.day_of_week import DayOfWeek
from ai_stock_forecasts.trading_algorithms.base_trading_module import BaseTradingModule
import pandas as pd
import numpy as np


class VolatilityRanking(BaseTradingModule):
    def __init__(self, num_stocks_purchased: int=10, day_of_week: DayOfWeek=DayOfWeek.tuesday, volatility_importance: float=0.3):
        super().__init__()

        self.num_stocks_purchased = num_stocks_purchased
        self.day_of_week = day_of_week
        self.volatility_importance = volatility_importance
        print(f'set trading params to: num_stocks_purchased: {self.num_stocks_purchased}, day_of_week: {self.day_of_week}, volatility_importance: {self.volatility_importance}')

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
    def simulate(self, predictions: pd.DataFrame):
        timestamps = (
            pd.Series(predictions["timestamp"])
            .drop_duplicates()
            .sort_values()
            .reset_index(drop=True)
        )

        curr_ts = timestamps.min()
        curr_ts = self._get_next_timestamp(timestamps, self.day_of_week, curr_ts)

        money = self.starting_money

        period_returns = []

        while True:
            filtered = predictions.loc[predictions['timestamp'] == curr_ts].copy()

            top_x = self._determine_top_x(filtered)

            total_profit = 0
            if len(top_x) > 0:
                future_ts = self._get_next_timestamp(timestamps, self.day_of_week, curr_ts)
                future_mask = predictions['timestamp'] == future_ts

                future_price = predictions.loc[future_mask].copy()
                future_price['future_price'] = future_price['close']

                top_x['curr_price'] = top_x['close']
                top_x = top_x[['symbol', 'curr_price']].merge(
                        future_price[['symbol', 'future_price']],
                        on=['symbol'], how='inner')

                for symbol, curr_price, future_price in zip(top_x['symbol'], top_x['curr_price'], top_x['future_price']):
                    if future_price / 4 > curr_price:
                        print(f'reverse split likely, skipping symbol: {symbol}, ts: {curr_ts}, that had curr_price: {curr_price}, and future_price: {future_price}')
                        continue

                    money_invested_in_this_stock = (self.starting_money / len(top_x))
                    num_stocks_bought = ( money_invested_in_this_stock / curr_price )
                    stock_value_now = future_price
                    money_post_sell = (num_stocks_bought * stock_value_now)
                    total_profit += money_post_sell - (self.starting_money / len(top_x))

            # print(f"{curr_ts}: profit: {total_profit}")
            # print(top_x[['symbol', 'curr_price', 'future_price']])
            period_returns.append(total_profit / money)
            money += total_profit

            curr_ts = future_ts
            if not curr_ts:
                break

        absolute_difference = money - self.starting_money
        money_left_post_tax = money - (absolute_difference*0.35)
        difference = ((money_left_post_tax - self.starting_money) / self.starting_money) * 100

        print(f"money left post tax: {money_left_post_tax}")

        sharpe, p_value = self._calculate_sharpe_ratio_and_p_value(period_returns)
        print(f"annual sharpe is: {sharpe}")

        return difference, sharpe, p_value

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








