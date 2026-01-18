from contextlib import contextmanager
from pathlib import Path
import pickle
from alpaca.data import TimeFrame, TimeFrameUnit
from dotenv import load_dotenv
import os
import boto3
import io
import pandas as pd
from datetime import datetime, timezone
from collections import defaultdict
import tarfile
import tempfile

from pytorch_forecasting.models.base import Prediction
import cloudpickle

from ai_stock_forecasts.models.historical_data import HistoricalData

class S3ParquetUtil:
    def __init__(self):
        load_dotenv()

        self.region = os.getenv("REGION_NAME")
        self.bucket = os.getenv("S3_BUCKET_NAME")
        self.model_output_bucket = os.getenv("S3_MODEL_OUTPUT_BUCKET_NAME")
        self.prefix = os.getenv("ATHENA_TABLE_S3_PREFIX")

        access_key = os.getenv("ACCESS_KEY")
        secret_access_key = os.getenv("SECRET_ACCESS_KEY")

        self.s3 = boto3.client(
            "s3",
            region_name=self.region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_access_key,
        )

    """ returns a dataframe with the following columns always:
            timestamp: timestamp
            value: string
            type: string
            updated_timestamp: timestamp
            date: date
            symbol: string
            feature: string
            time_frame: string
    """
    def get_features_data(self, symbols: list[str], features: list[str], time_frame: TimeFrame=TimeFrame(1, TimeFrameUnit.Day)) -> pd.DataFrame:
        dfs = []
        for feature in features:
            print(f"pulling feature data for feature: {feature}, time_frame: {time_frame.unit_value.value}, prefix: {self.prefix}")
            if time_frame.amount_value == 1:
                prefix = f"{self.prefix}/feature={feature}/time_frame={time_frame.unit_value.value}/"
            else:
                prefix = f"{self.prefix}/feature={feature}/time_frame={time_frame.amount_value}-{time_frame.unit_value.value}/"

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

                    df = df[df["symbol"].isin(symbols)]
                    if time_frame.unit in (TimeFrameUnit.Minute, TimeFrameUnit.Hour):
                        h = df["timestamp"].dt.hour
                        df = df[(h > 9) & (h < 16)]

                    dfs.append(df)


                if response.get("IsTruncated"):
                    continuation_token = response.get("NextContinuationToken")
                else:
                    break

        final_df = pd.concat(dfs, ignore_index=True)

        return final_df


    def upload_features_data(self, records: list[HistoricalData], time_frame: TimeFrame=TimeFrame(1, TimeFrameUnit.Day)):
        records_by_partitions: dict[str, list[HistoricalData]] = defaultdict(list)
        for r in records:
            partitions = r.feature + r.time_frame.unit_value.value
            records_by_partitions[partitions].append(r)

        for key, recs in records_by_partitions.items():
            rows = (self._to_row(r) for r in recs)
            df = pd.DataFrame.from_records(rows)

            buffer = io.BytesIO()
            df.to_parquet(buffer, index=False)
            buffer.seek(0)

            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            key = None
            if time_frame.amount_value == 1:
                key = f"{self.prefix}/feature={recs[0].feature}/time_frame={time_frame.unit_value.value}/historical_data_{ts}.parquet"
            else:
                key = f"{self.prefix}/feature={recs[0].feature}/time_frame={time_frame.amount_value}-{time_frame.unit_value.value}/historical_data_{ts}.parquet"

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

    def save_raw_predictions(self, model_id: str, predictions: Prediction):
        key = 'model_predictions/'+model_id+'.pkl'
        print(f'saving raw predictions for model_id: {model_id} to {key}')
        buf = io.BytesIO()
        cloudpickle.dump(predictions, buf, protocol=pickle.HIGHEST_PROTOCOL)
        buf.seek(0)

        self.s3.put_object(Bucket=self.bucket, Key=key, Body=buf.getvalue())

    def save_human_readable_predictions(self, model_id: str, predictions: pd.DataFrame):
        key = 'model_predictions_readable/'+model_id+'.pkl'
        print(f'saving human readable predictions for model_id: {model_id} to {key}')
 
        buf = io.BytesIO()
        pickle.dump(predictions, buf, protocol=pickle.HIGHEST_PROTOCOL)
        buf.seek(0)

        self.s3.put_object(Bucket=self.bucket, Key=key, Body=buf.getvalue())

    def load_raw_predictions(self, model_id: str) -> pd.DataFrame:
        key = 'model_predictions/'+model_id+'.pkl'
        print(f'loading raw predictions for model_id: {model_id} from {key}')

        obj = self.s3.get_object(Bucket=self.bucket, Key=key)
        data = obj['Body'].read()
        return cloudpickle.loads(data)

    def load_human_readable_predictions(self, model_id: str):
        key = 'model_predictions_readable/'+model_id+'.pkl'
        print(f'loading human readable predictions for model_id: {model_id} from {key}')

        obj = self.s3.get_object(Bucket=self.bucket, Key=key)
        data = obj["Body"].read()
        return pickle.loads(data)

    @contextmanager
    def load_best_model_checkpoint(self, model_id: str):
        key = model_id + '/output/model.tar.gz'
        print(f'loading model ckpt for model_id: {model_id} from {key}')

        obj = self.s3.get_object(Bucket=self.model_output_bucket, Key=key)
        buf = io.BytesIO(obj["Body"].read())
        buf.seek(0)

        with tempfile.TemporaryDirectory() as tmp:
            with tarfile.open(fileobj=buf, mode="r:gz") as tf:
                """ On macOS, when files get copied/archived in certain ways, 
                    it can create companion files that start with ._ which contain 
                    Finder/resource-fork metadata. Those files can end up inside your 
                    tarball alongside the real files so we have to exlude files that start with '._'
                """
                ckpts = [
                    m for m in tf.getmembers()
                    if m.isfile() and m.name.endswith(".ckpt") and m.size and m.size > 0 and not Path(m.name).name.startswith("._")
                ]
                if not ckpts:
                    raise FileNotFoundError("No non-empty .ckpt files found in model.tar.gz")

                # Prefer ones whose *filename* contains 'best'
                def score(m):
                    name = Path(m.name).name.lower()
                    if "best" in name:
                        return 0
                    if "last" in name:
                        return 1
                    return 2

                ckpts.sort(key=score)
                member = ckpts[0]
                print(f'pulling this ckpt: {member.name}')

                tf.extract(member, path=tmp)
                ckpt_path = os.path.join(tmp, member.name)

                yield ckpt_path

    def upload_checkpoints(self, checkpoints_path: str, out_path: str, model_id: str):
        checkpoints = Path(checkpoints_path)
        key = f'{model_id}/output/model.tar.gz'

        print(f'uploading checkpoints path {checkpoints} to {key}')

        with tarfile.open(f'{out_path}/model.tar.gz', mode="w:gz") as tf:
            tf.add(checkpoints, arcname=checkpoints.name)

        out_dir = Path(out_path)
        tar_path = out_dir / 'model.tar.gz'
        with tar_path.open("rb") as f:
            self.s3.put_object(Bucket=self.model_output_bucket, Key=key, Body=f)


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
    res = s3_util.get_features_data(["GOOGL"], ["open"], TimeFrame(1, TimeFrameUnit.Day))

    print(res[0].value)
