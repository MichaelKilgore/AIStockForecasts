"""
Wipe s3://<bucket>/test_data/ then copy s3://<bucket>/historical_data_v4/ into it,
filtering every parquet under feature=*/time_frame=*/ down to AAPL only.
Non-parquet objects (and any parquet without a `symbol` column) are copied verbatim.
"""

import io
import logging
import os

import boto3
import pandas as pd
from dotenv import load_dotenv

SOURCE_PREFIX = "historical_data_v4/"
DEST_PREFIX = "test_data/"
SYMBOL = "AAPL"


def make_client():
    load_dotenv()
    return boto3.client(
        "s3",
        region_name=os.getenv("REGION_NAME"),
        aws_access_key_id=os.getenv("ACCESS_KEY"),
        aws_secret_access_key=os.getenv("SECRET_ACCESS_KEY"),
    )


def iter_keys(s3, bucket: str, prefix: str):
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            yield obj["Key"]
        if not resp.get("IsTruncated"):
            return
        token = resp.get("NextContinuationToken")


def wipe_dest(s3, bucket: str):
    logging.info(f"wiping s3://{bucket}/{DEST_PREFIX}")
    batch = []
    deleted = 0
    for key in iter_keys(s3, bucket, DEST_PREFIX):
        batch.append({"Key": key})
        if len(batch) == 1000:
            s3.delete_objects(Bucket=bucket, Delete={"Objects": batch})
            deleted += len(batch)
            batch = []
    if batch:
        s3.delete_objects(Bucket=bucket, Delete={"Objects": batch})
        deleted += len(batch)
    logging.info(f"deleted {deleted} objects")


def process_key(s3, bucket: str, src_key: str):
    dst_key = DEST_PREFIX + src_key[len(SOURCE_PREFIX):]

    is_feature_parquet = (
        src_key.endswith(".parquet")
        and "/feature=" in src_key
        and "/time_frame=" in src_key
    )

    if not is_feature_parquet:
        s3.copy_object(
            Bucket=bucket,
            Key=dst_key,
            CopySource={"Bucket": bucket, "Key": src_key},
        )
        logging.info(f"copied {src_key} -> {dst_key}")
        return

    body = s3.get_object(Bucket=bucket, Key=src_key)["Body"].read()
    df = pd.read_parquet(io.BytesIO(body))

    if "symbol" not in df.columns:
        s3.put_object(Bucket=bucket, Key=dst_key, Body=body)
        logging.info(f"copied (no symbol col) {src_key} -> {dst_key}")
        return

    filtered = df[df["symbol"] == SYMBOL]
    buf = io.BytesIO()
    filtered.to_parquet(buf, index=False)
    s3.put_object(Bucket=bucket, Key=dst_key, Body=buf.getvalue())
    logging.info(f"filtered {len(df)} -> {len(filtered)} rows  {src_key} -> {dst_key}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    bucket = os.getenv("S3_BUCKET_NAME") or "ai-stock-forecasts"
    s3 = make_client()

    wipe_dest(s3, bucket)

    for key in iter_keys(s3, bucket, SOURCE_PREFIX):
        process_key(s3, bucket, key)


if __name__ == "__main__":
    main()
