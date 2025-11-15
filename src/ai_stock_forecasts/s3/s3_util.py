from alpaca.data import TimeFrame
from dotenv import load_dotenv
import os
import boto3
import io
import pandas as pd
from datetime import datetime, timezone
from collections import defaultdict

from src.ai_stock_forecasts.models.historical_data import HistoricalData

class S3ParquetUtil:
    def __init__(self):
        load_dotenv()

        self.region = os.getenv("REGION_NAME")
        self.bucket = os.getenv("S3_BUCKET_NAME")
        self.prefix = os.getenv("ATHENA_TABLE_S3_PREFIX")

        access_key = os.getenv("ACCESS_KEY")
        secret_access_key = os.getenv("SECRET_ACCESS_KEY")

        self.s3 = boto3.client(
            "s3",
            region_name=self.region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_access_key,
        )

    def get_features_data(self, symbols: list[str], features: list[str], time_frame: TimeFrame=TimeFrame.Day) -> list[HistoricalData]:
        res = []
        for feature in features:
            print(f"pulling feature data for feature: {feature}, time_frame: {time_frame.unit_value.value}")
            prefix = f"{self.prefix}/feature={feature}/time_frame={time_frame.unit_value.value}/"

            continuation_token = None
            while True:
                list_kwargs = {
                    "Bucket": self.bucket,
                    "Prefix": prefix,
                }
                if continuation_token is not None:
                    list_kwargs["ContinuationToken"] = continuation_token

                response = self.s3.list_objects_v2(**list_kwargs)

                contents = response.get("Contents", [])
                if not contents:
                    break

                for obj in contents:
                    key = obj["Key"]
                    if not key.endswith(".parquet"):
                        continue

                    print(f"s3 get_object: {key}")
                    obj_resp = self.s3.get_object(Bucket=self.bucket, Key=key)
                    data = obj_resp["Body"].read()

                    df = pd.read_parquet(io.BytesIO(data))

                    print(f"appending results to final res for s3 key: {key}")
                    for row in df.itertuples(index=False):
                        res.append(
                            HistoricalData(
                                symbol=row.symbol,
                                timestamp=row.timestamp,
                                feature=row.feature,
                                value=row.value,
                                type=row.type,
                                updated_timestamp=row.updated_timestamp,
                                time_frame=time_frame,
                                date=row.date,
                            )
                        )

                if response.get("IsTruncated"):
                    continuation_token = response.get("NextContinuationToken")
                else:
                    break

        filtered_res = [r for r in res if r.symbol in symbols]

        return filtered_res


    def upload_features_data(self, records: list[HistoricalData], time_frame: TimeFrame=TimeFrame.Day):
        records_by_partitions: dict[str, list[HistoricalData]] = defaultdict(list)
        for r in records:
            partitions = r.feature + r.time_frame.unit_value.value
            records_by_partitions[partitions].append(r)

        for key, recs in records_by_partitions.items():
            rows = [self._to_row(r) for r in recs]
            df = pd.DataFrame(rows)

            buffer = io.BytesIO()
            df.to_parquet(buffer, index=False)
            buffer.seek(0)

            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            key = f"{self.prefix}/feature={recs[0].feature}/time_frame={time_frame.unit_value.value}/historical_data_{ts}.parquet"

            self.s3.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=buffer.getvalue(),
            )

            print(f"Uploaded {len(recs)} rows to s3://{self.bucket}/{key}")

    def _to_row(self, rec: HistoricalData) -> dict:
        return {
            "symbol": rec.symbol,
            "timestamp": rec.timestamp.replace(tzinfo=timezone.utc),
            "feature": rec.feature,
            "value": rec.value,
            "type": rec.type,
            "updated_timestamp": rec.updated_timestamp.replace(tzinfo=timezone.utc),
            "time_frame": rec.time_frame.unit_value,
            "date": rec.date.date(),
        }

if __name__ == "__main__":
    """test_record = HistoricalData(
        "TEST",
        datetime(1980, 1, 1),
        "open",
        "50.0",
        "float",
        datetime.now(),
        TimeFrame.Day,
        datetime(1980, 1, 1),
    )

    uploader = S3ParquetUtil()
    uploader.upload_records([test_record])"""

    s3_util = S3ParquetUtil()
    res = s3_util.get_features_data(["GOOGL"], ["open"], TimeFrame.Day)

    print(res[0].value)
