#!/usr/bin/env bash
# Build lambda.zip for the check_if_execute_buy_ran handler.
#
# Layout produced inside the zip:
#   check_if_execute_buy_ran.py        <- handler at root
#   ai_stock_forecasts/utils/...       <- imported helpers
#   ai_stock_forecasts/models/order.py
#   <site-packages of runtime deps>
#
# Handler string for the Lambda: check_if_execute_buy_ran.lambda_handler

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build/lambda"
ZIP_PATH="$REPO_ROOT/lambda.zip"
SRC="$REPO_ROOT/src/ai_stock_forecasts"

rm -rf "$BUILD_DIR" "$ZIP_PATH"
mkdir -p "$BUILD_DIR"

# 1. Runtime deps. boto3/botocore are already in the Lambda runtime, so skip them.
python3 -m pip install \
  --quiet \
  --target "$BUILD_DIR" \
  --platform manylinux2014_x86_64 \
  --only-binary=:all: \
  --implementation cp \
  --python-version 3.12 \
  alpaca-py python-dotenv requests

# 2. Application code — make ai_stock_forecasts a real package inside the zip.
mkdir -p "$BUILD_DIR/ai_stock_forecasts/utils" "$BUILD_DIR/ai_stock_forecasts/models"
touch "$BUILD_DIR/ai_stock_forecasts/__init__.py" \
      "$BUILD_DIR/ai_stock_forecasts/utils/__init__.py" \
      "$BUILD_DIR/ai_stock_forecasts/models/__init__.py"

cp "$SRC/utils/dynamodb_util.py"     "$BUILD_DIR/ai_stock_forecasts/utils/"
cp "$SRC/utils/telegram_bot_util.py" "$BUILD_DIR/ai_stock_forecasts/utils/"
cp "$SRC/models/order.py"            "$BUILD_DIR/ai_stock_forecasts/models/"

# 3. Handler at zip root.
cp "$SRC/lambda/check_if_execute_buy_ran.py" "$BUILD_DIR/"

# 4. Trim noise that bloats the zip.
find "$BUILD_DIR" -type d \( -name "__pycache__" -o -name "*.dist-info" -o -name "tests" \) -prune -exec rm -rf {} +
find "$BUILD_DIR" -type f -name "*.pyc" -delete

# 5. Zip.
( cd "$BUILD_DIR" && zip -qr "$ZIP_PATH" . )

echo "Built $ZIP_PATH ($(du -h "$ZIP_PATH" | cut -f1))"
