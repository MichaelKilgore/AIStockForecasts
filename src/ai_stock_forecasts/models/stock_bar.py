from datetime import datetime

from alpaca.data import TimeFrame, TimeFrameUnit


class StockBar:
    def __init__(self, close: float, high: float, low: float,
                    open: float, symbol: str, timestamp: datetime,
                    trade_count: float, volume: float, vwap: float, time_frame: TimeFrame=TimeFrame(1, TimeFrameUnit.Day)):
        self.close = close
        self.high = high
        self.low = low
        self.open = open
        self.symbol = symbol
        self.timestamp = timestamp
        self.trade_count = trade_count
        self.volume = volume
        self.vwap = vwap
        self.time_frame = time_frame

    def __str__(self):
        return f"symbol: {self.symbol}, timestamp: {self.timestamp}, open price: {self.open}"

    def __lt__(self, other):
        if not isinstance(other, StockBar):
            return NotImplemented
        return self.timestamp < other.timestamp
