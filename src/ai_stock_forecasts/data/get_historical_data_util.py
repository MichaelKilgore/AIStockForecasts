from datetime import datetime

from dotenv import load_dotenv
import os
from alpaca.data import StockHistoricalDataClient, TimeFrame
from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest

from src.ai_stock_forecasts.models.stock import Stock
from src.ai_stock_forecasts.models.stock_bar import StockBar


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

    def get_historical_stock_prices(self, stocks: list[str], start: datetime, end: datetime, time_frame: TimeFrame=TimeFrame.Day) -> list[StockBar]:
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

if __name__ == "__main__":
    obj = GetHistoricalDataUtil()

    res = obj.get_historical_stock_prices(["SPY"], datetime(2020, 1, 1), datetime(2025, 11, 1))

    res.sort()
    for val in res:
        print(val)
