from datetime import datetime, timedelta

from alpaca.data import TimeFrame, TimeFrameUnit

from ai_stock_forecasts.data.get_historical_data_util import GetHistoricalDataUtil
from ai_stock_forecasts.models.historical_data import HistoricalData
from ai_stock_forecasts.models.stock_bar import StockBar
from ai_stock_forecasts.s3.s3_util import S3ParquetUtil

import time

class BackfillFeaturesUtil:
    def __init__(self):
        self.s3_util = S3ParquetUtil()
        self.get_historical_data_util = GetHistoricalDataUtil()

    def backfill_base_features(self, features: list[str], symbols: list[str], start: datetime, end: datetime, time_frame: TimeFrame=TimeFrame(1, TimeFrameUnit.Day), include_time_features: bool = False):
        self._validate_base_features(features)

        all_records = []
        for symbol in symbols:
            print(f"Pulling data for {symbol}...")
            all_records.extend(self.get_historical_data_util.get_historical_stock_prices([symbol], start, end, time_frame))
            time.sleep(0.2)

        converted_records = self._convert_stock_bar_records_to_historical_data_records(all_records, features)
        del all_records

        print("Dropping duplicate records")
        distinct_records = sorted(
            set(converted_records),
            key=lambda x: x.timestamp
        )
        del converted_records

        final_records = distinct_records if not include_time_features else self._include_base_time_features(distinct_records, time_frame, distinct_records[0].updated_timestamp)
        del distinct_records

        print("Uploading data to s3")
        self.s3_util.upload_features_data(final_records, time_frame)

        print("Backfill complete...")

    # we don't need this right now as you can filter by interval directly in the api however it does seem like there is potential to fill in gaps
    # using this.
    def _filter_down_by_minute_interval_wanted(self, records: list[HistoricalData], interval: int=10) -> list[HistoricalData]:
        divisible_timestamps_seen = set()
        filtered_records = []
        """for every record we only want records for every 10 minutes
            however if an interval is missing we want to pull an earlier timestamp present."""
        for record in reversed(records):
            if (record.timestamp.minute % interval == 0):
                divisible_timestamps_seen.add((record.symbol, record.timestamp))
                filtered_records.append(record)
            else:
                # get latest divisible num
                minutes_off = record.timestamp.minute % interval
                latest_timestamp = record.timestamp + timedelta(minutes=(interval-minutes_off))
                if (record.symbol, latest_timestamp) not in divisible_timestamps_seen:
                    record.timestamp = latest_timestamp
                    filtered_records.append(record)

        return filtered_records

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

    def _include_base_time_features(self, records: list[HistoricalData], time_frame: TimeFrame = TimeFrame(1, TimeFrameUnit.Day), updated_time: datetime = datetime.now()) -> list[HistoricalData]:
        s = set()

        print("Attaching base time features to list of records")

        for record in records:
            s.add((record.symbol, record.timestamp))

        for key in s:
            # include year and month feature
            if (time_frame.unit == TimeFrameUnit.Month 
                    or time_frame.unit == TimeFrameUnit.Week
                    or time_frame.unit == TimeFrameUnit.Day
                    or time_frame.unit == TimeFrameUnit.Hour
                    or time_frame.unit == TimeFrameUnit.Minute):
                records.append(HistoricalData(key[0], key[1], 'year', str(key[1].year), type(key[1].year).__name__, updated_time, time_frame, key[1]))
                records.append(HistoricalData(key[0], key[1], 'month', str(key[1].month), type(key[1].month).__name__, updated_time, time_frame, key[1]))

            # include day_of_month and day_of_week feature
            if (time_frame.unit == TimeFrameUnit.Day
                    or time_frame.unit == TimeFrameUnit.Hour
                    or time_frame.unit == TimeFrameUnit.Minute):
                records.append(HistoricalData(key[0], key[1], 'day_of_month', str(key[1].day), type(key[1].day).__name__, updated_time, time_frame, key[1]))
                records.append(HistoricalData(key[0], key[1], 'day_of_week', str(key[1].weekday()), type(key[1].weekday()).__name__, updated_time, time_frame, key[1]))

            # include hour_of_day feature
            if (time_frame.unit == TimeFrameUnit.Hour
                    or time_frame.unit == TimeFrameUnit.Minute):
                records.append(HistoricalData(key[0], key[1], 'hour_of_day', str(key[1].hour), type(key[1].hour).__name__, updated_time, time_frame, key[1]))

            # include minute_of_day feature
            if (time_frame.unit == TimeFrameUnit.Hour
                    or time_frame.unit == TimeFrameUnit.Minute):
                records.append(HistoricalData(key[0], key[1], 'minute_of_day', str(key[1].minute), type(key[1].minute).__name__, updated_time, time_frame, key[1]))

        return records

if __name__ == "__main__":
    obj = BackfillFeaturesUtil()

    with open("src/ai_stock_forecasts/constants/symbols.txt", "r") as f:
        symbols = [line.strip() for line in f]

    symbols.append('SPY')

    # TODO: We stopped the program at 391, continue on to finish last 100 from step 14.
    i = 13*30
    while i < len(symbols):
        obj.backfill_base_features(['open', 'close', 'high', 'low', 'trade_count', 'volume', 'vwap'], symbols[i:min(i+30, len(symbols))], datetime(2020, 1, 1), datetime(2025, 12, 31), TimeFrame(10, TimeFrameUnit.Minute), True)
        i += 30

    #obj.backfill_base_features(['open', 'close', 'high', 'low', 'open', 'trade_count', 'volume', 'vwap'], symbols, datetime(2020, 1, 1), datetime(2025, 12, 31), TimeFrame(1, TimeFrameUnit.Day), True)

    #res = obj.get_historical_data_util.get_historical_stock_prices(['AAPL'], datetime(2020, 1, 1), datetime(2025, 11, 1), TimeFrame.Minute)
    #print(len(res))
    #print(sys.getsizeof(res))
 
