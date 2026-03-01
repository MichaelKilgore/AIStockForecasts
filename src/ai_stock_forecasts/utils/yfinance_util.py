from datetime import datetime
from typing import Union
from pandas import DataFrame
import yfinance as yf

class YfinanceUtil:
    def get_historical_data(self, ticker: str, start: Union[str, datetime], end: Union[str, datetime]) -> DataFrame:
        obj = yf.Ticker(ticker)

        return obj.history(start=start, end=end, auto_adjust=True)


if __name__ == '__main__':
    u = YfinanceUtil()

    res = u.get_historical_data('AAPL', '2020-01-01', '2026-01-01')

    print(res)

# Date Open High Low Close Volume Dividends  Stock Splits
