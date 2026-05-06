from datetime import datetime
from typing import List, Union
from pandas import DataFrame
import yfinance as yf

from ai_stock_forecasts.models.stock import Stock

class YfinanceUtil:
    def get_historical_data(self, ticker: str, start: Union[str, datetime], end: Union[str, datetime]) -> DataFrame:
        obj = yf.Ticker(ticker)

        return obj.history(start=start, end=end, auto_adjust=True)

    def get_current_prices(self, tickers: List[str]) -> List[Stock]:
        objs = yf.Tickers(' '.join(tickers))

        return [ Stock(ticker, float(objs.tickers[ticker].fast_info['last_price'])) for ticker in tickers ]


if __name__ == '__main__':
    u = YfinanceUtil()

    res = u.get_historical_data('AAPL', '2020-01-01', '2026-01-01')

    print(res)

    print(u.get_current_prices(['AAPL', 'MSFT', 'GOOGL']))

# Date Open High Low Close Volume Dividends  Stock Splits
