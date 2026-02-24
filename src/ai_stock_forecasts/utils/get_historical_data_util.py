from datetime import datetime, timedelta
from pandas import DataFrame, Series
import pandas as pd
import requests
from datetime import timezone

from dotenv import load_dotenv
import os
from alpaca.data import StockHistoricalDataClient, TimeFrame, TimeFrameUnit
from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest

from ai_stock_forecasts.models.stock import Stock
from ai_stock_forecasts.models.stock_bar import StockBar

from finance_calendars import finance_calendars as fc
from alpaca.data.enums import DataFeed
from zoneinfo import ZoneInfo
import yfinance as yf

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
                                                      feed=DataFeed.IEX,
                                                      adjustment="split")

        multisymbol_quotes = self.stock_client.get_stock_bars(multisymbol_request_params)

        res = []

        for symbol, bars in multisymbol_quotes.data.items():
            for bar in bars:
                res.append(StockBar(close=bar.close, high=bar.high, low=bar.low, open=bar.open,
                                    # symbol=symbol, timestamp=bar.timestamp.astimezone(ZoneInfo("America/New_York")), trade_count=bar.trade_count,
                                    symbol=symbol, timestamp=bar.timestamp, trade_count=bar.trade_count,
                                    volume=bar.volume, vwap=bar.vwap, time_frame=time_frame))

        return res

    def get_historical_vix(self, period: str='7y') -> DataFrame:
        ticker = yf.Ticker('^VIX')
        historical_data = ticker.history(period=period)

        return historical_data.reset_index()[['Date', 'Close']]


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
    # curr = datetime(2026, 1, 14)
    # res = obj.get_surprise(curr)
    # print(res)
    #print(res.index[res.index.duplicated()].unique())
    #print(res['KNDI'])
    # America/New_York
    # start = pd.Timestamp('2025-01-01', tzinfo=ZoneInfo("America/New_York"))
    # end = pd.Timestamp('2025-01-03', tzinfo=ZoneInfo("America/New_York"))
    # bars = data_client.get_stock_bars(bars_request).df.tz_convert('America/New_York', level='timestamp')
    start = datetime(2025, 1, 1)
    end = datetime(2026, 1, 1)

    # res = obj.get_historical_stock_prices(["VIX"], start, end, TimeFrame(1, TimeFrameUnit.Day))
    res = obj.get_historical_vix()

    print(res)

    # res.sort()
    # for val in res:
    #     print(val)

