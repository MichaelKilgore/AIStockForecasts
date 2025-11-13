from datetime import datetime

from alpaca.data import TimeFrame


# This is the structure for records in our athena table: historical_data
# This table store historical stock feature data.
class HistoricalData:
    def __init__(self, symbol: str, timestamp: datetime, feature: str, value: str,
                 type: str, updated_timestamp: datetime, time_frame: TimeFrame, date: datetime):
        self.symbol = symbol
        self.timestamp = timestamp
        self.feature = feature
        self.value = value
        self.type = type
        self.updated_timestamp = updated_timestamp
        self.time_frame = time_frame
        self.date = date

    def __str__(self):
        return f"symbol: {self.symbol}, timestamp: {self.timestamp}, price: {self.price}"

