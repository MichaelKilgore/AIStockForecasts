from datetime import datetime, timedelta
from pandas import Series
import requests

from dotenv import load_dotenv
import os
from alpaca.data import StockHistoricalDataClient, TimeFrame, TimeFrameUnit
from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest

from ai_stock_forecasts.models.stock import Stock
from ai_stock_forecasts.models.stock_bar import StockBar

from finance_calendars import finance_calendars as fc


class GetHistoricalDataUtil:
    def __init__(self):
        load_dotenv()

        self._alpaca_key = os.getenv("ALPACA_KEY")
        self._alpaca_secret = os.getenv("ALPACA_SECRET")

        print("Alpaca Key:", self._alpaca_key)
        print("Alpaca Secret:", self._alpaca_secret)

        self.stock_client = StockHistoricalDataClient(self._alpaca_key, self._alpaca_secret)

    def get_latest_stock_prices(self, stocks: list[str]) -> list[Stock]:
        multisymbol_request_params = StockLatestQuoteRequest(symbol_or_symbols=stocks)

        latest_multisymbol_quotes = self.stock_client.get_stock_latest_quote(multisymbol_request_params)

        res = []
        for key,val in latest_multisymbol_quotes.items():
            res.append(Stock(key, val.bid_price))

        return res

    def get_historical_stock_prices(self, stocks: list[str], start: datetime, end: datetime, time_frame: TimeFrame = TimeFrame(amount=1, unit=TimeFrameUnit.Day)) -> list[StockBar]:
        print(f'Getting stock price history for {len(stocks)} symbols between {start} and {end}, with time_frame: {time_frame}')
        multisymbol_request_params = StockBarsRequest(symbol_or_symbols=stocks,
                                                      timeframe=time_frame,
                                                      start=start,
                                                      end=end,
                                                      adjustment="split")

        multisymbol_quotes = self.stock_client.get_stock_bars(multisymbol_request_params)

        res = []

        for symbol, bars in multisymbol_quotes.data.items():
            for bar in bars:
                res.append(StockBar(close=bar.close, high=bar.high, low=bar.low, open=bar.open,
                                    symbol=symbol, timestamp=bar.timestamp, trade_count=bar.trade_count,
                                    volume=bar.volume, vwap=bar.vwap, time_frame=time_frame))

        return res

    """
        returns DataFrame Series like:

        symbol value
        RBLX    -192.59
    """
    def get_surprise(self, date: datetime = datetime(2021, 8, 16, 0, 0)) -> Series:
        earnings = fc.get_earnings_by_date(date)
        if len(earnings) != 0:
            if 'surprise' in earnings.columns:
                return earnings['surprise']
            else:
                return Series(0.0, index=earnings.index, name="surprise")
        else:
            return Series()

if __name__ == "__main__":
    obj = GetHistoricalDataUtil()

    # curr = datetime(2025, 10, 30)
    #curr = datetime(2020, 4, 28)
    curr = datetime(2026, 1, 14)
    res = obj.get_surprise(curr)
    print(res)
    #print(res.index[res.index.duplicated()].unique())
    #print(res['KNDI'])
    # res = obj.get_historical_stock_prices(["SPY"], datetime(2020, 1, 1), datetime(2025, 11, 1))

    # res.sort()
    # for val in res:
    #     print(val)

