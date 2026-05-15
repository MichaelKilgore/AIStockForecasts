#!/usr/bin/env bash
# Repackage lambda.zip and push it to the existing check-if-execute-buy-ran
# Lambda function. Use this after editing the handler or any of the helper
# modules pulled into the zip by scripts/package_lambda.sh.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"
ZIP_PATH="$REPO_ROOT/lambda.zip"

FUNCTION_NAME="check-if-execute-buy-ran"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "missing $ENV_FILE" >&2
  exit 1
fi
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

: "${REGION_NAME:?REGION_NAME not set in .env}"

echo "[1/2] building lambda.zip"
"$REPO_ROOT/scripts/package_lambda.sh"

echo "[2/2] pushing code to $FUNCTION_NAME in $REGION_NAME"
aws lambda update-function-code \
  --function-name "$FUNCTION_NAME" \
  --zip-file "fileb://${ZIP_PATH}" \
  --region "$REGION_NAME" \
  --query '{FunctionArn:FunctionArn,LastUpdateStatus:LastUpdateStatus,CodeSha256:CodeSha256}' \
  --output table

aws lambda wait function-updated \
  --function-name "$FUNCTION_NAME" \
  --region "$REGION_NAME"

echo "Done."
