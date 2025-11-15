from datetime import datetime

from alpaca.data import TimeFrame

from src.ai_stock_forecasts.constants.extract_symbols_util import get_symbols
from src.ai_stock_forecasts.data.get_historical_data_util import GetHistoricalDataUtil
from src.ai_stock_forecasts.models.historical_data import HistoricalData
from src.ai_stock_forecasts.models.stock_bar import StockBar
from src.ai_stock_forecasts.s3.s3_util import S3ParquetUtil
import time
import pandas as pd

class BackfillFeaturesUtil:
    def __init__(self):
        self.s3_util = S3ParquetUtil()
        self.get_historical_data_util = GetHistoricalDataUtil()

    def backfill_base_features(self, features: list[str], symbols: list[str], start: datetime, end: datetime, time_frame: TimeFrame=TimeFrame.Day):
        self._validate_base_features(features)

        all_records = []
        for symbol in symbols:
            print(f"Pulling data for {symbol}...")
            all_records.extend(self.get_historical_data_util.get_historical_stock_prices([symbol], start, end, time_frame))
            time.sleep(0.2)

        converted_records = self._convert_stock_bar_records_to_historical_data_records(all_records, features)
        converted_records_sorted = sorted(converted_records, key=lambda x: x.date)

        print("Dropping duplicate records")
        df = pd.DataFrame([vars(r) for r in converted_records_sorted])
        df_unique = df.drop_duplicates()
        distinct_records = [HistoricalData(**row.to_dict()) for _, row in df_unique.iterrows()]

        self.s3_util.upload_features_data(distinct_records)

        print("Backfill complete...")

    def _convert_stock_bar_records_to_historical_data_records(self, records: list[StockBar], enabled_features: list[str]) -> list[HistoricalData]:
        res = []
        for record in records:
            for feature in enabled_features:
                value = getattr(record, feature, None)
                if value is None:
                    raise ValueError(f"The feature passed does not exist in the record: {feature}")

                res.append(HistoricalData(record.symbol, record.timestamp, feature, str(value), type(value).__name__, datetime.now(), record.time_frame, record.timestamp))

        return res

    def _validate_base_features(self, features: list[str]):
        for feature in features:
            if feature not in ["close", "high", "low", "open", "trade_count", "volume", "vwap"]:
                raise ValueError(f"The following feature: {feature} is not a base feature. Base features are all the features in StockBar class")

if __name__ == "__main__":
    obj = BackfillFeaturesUtil()

    with open("../constants/symbols.txt", "r") as f:
        symbols = [line.strip() for line in f]
    obj.backfill_base_features(['open', 'close', 'high', 'low', 'open', 'trade_count', 'volume', 'vwap'], symbols, datetime(2019, 1, 1), datetime(2025, 11, 1))
