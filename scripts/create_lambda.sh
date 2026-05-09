#!/usr/bin/env bash
# One-shot creation of the check-if-execute-buy-ran Lambda:
#   1. builds lambda.zip via scripts/package_lambda.sh
#   2. creates (or reuses) the IAM execution role + attaches policies
#   3. creates the Lambda function pointing at the zip
#
# Re-running this script after the function exists is a no-op for the role
# and will fail at create-function — use `aws lambda update-function-code`
# (see scripts/package_lambda.sh comment block) to push new code.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"
ZIP_PATH="$REPO_ROOT/lambda.zip"

FUNCTION_NAME="check-if-execute-buy-ran"
ROLE_NAME="check-if-execute-buy-ran-role"
HANDLER="check_if_execute_buy_ran.lambda_handler"
RUNTIME="python3.12"
TIMEOUT=30
MEMORY=256

# --- load .env -------------------------------------------------------------
if [[ ! -f "$ENV_FILE" ]]; then
  echo "missing $ENV_FILE" >&2
  exit 1
fi
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

: "${REGION_NAME:?REGION_NAME not set in .env}"
: "${ORDERS_TABLE:?ORDERS_TABLE not set in .env}"
: "${TELEGRAM_BOT_TOKEN:?TELEGRAM_BOT_TOKEN not set in .env}"
: "${TELEGRAM_CHAT_ID:?TELEGRAM_CHAT_ID not set in .env}"
: "${ALPACA_KEY:?ALPACA_KEY not set in .env}"
: "${ALPACA_SECRET:?ALPACA_SECRET not set in .env}"
: "${ACCESS_KEY:?ACCESS_KEY not set in .env}"
: "${SECRET_ACCESS_KEY:?SECRET_ACCESS_KEY not set in .env}"

# --- build zip -------------------------------------------------------------
echo "[1/3] building lambda.zip"
"$REPO_ROOT/scripts/package_lambda.sh"

# --- IAM role --------------------------------------------------------------
echo "[2/3] ensuring IAM role $ROLE_NAME"
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"

if aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
  echo "  role already exists: $ROLE_ARN"
else
  aws iam create-role --role-name "$ROLE_NAME" \
    --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}' \
    >/dev/null
  aws iam attach-role-policy --role-name "$ROLE_NAME" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
  aws iam attach-role-policy --role-name "$ROLE_NAME" \
    --policy-arn arn:aws:iam::aws:policy/AmazonDynamoDBReadOnlyAccess
  echo "  created $ROLE_ARN — sleeping 10s for IAM propagation"
  sleep 10
fi

# --- create function -------------------------------------------------------
echo "[3/3] creating Lambda function $FUNCTION_NAME in $REGION_NAME"

ENV_JSON=$(cat <<EOF
{"Variables":{
  "REGION_NAME":"${REGION_NAME}",
  "ORDERS_TABLE":"${ORDERS_TABLE}",
  "TELEGRAM_BOT_TOKEN":"${TELEGRAM_BOT_TOKEN}",
  "TELEGRAM_CHAT_ID":"${TELEGRAM_CHAT_ID}",
  "ALPACA_KEY":"${ALPACA_KEY}",
  "ALPACA_SECRET":"${ALPACA_SECRET}",
  "ACCESS_KEY":"${ACCESS_KEY}",
  "SECRET_ACCESS_KEY":"${SECRET_ACCESS_KEY}"
}}
EOF
)

aws lambda create-function \
  --function-name "$FUNCTION_NAME" \
  --runtime "$RUNTIME" \
  --role "$ROLE_ARN" \
  --handler "$HANDLER" \
  --timeout "$TIMEOUT" \
  --memory-size "$MEMORY" \
  --zip-file "fileb://${ZIP_PATH}" \
  --region "$REGION_NAME" \
  --environment "$ENV_JSON" \
  --query '{FunctionArn:FunctionArn,State:State}' \
  --output table

echo
echo "Done. Test with:"
echo "  aws lambda invoke --function-name $FUNCTION_NAME --region $REGION_NAME \\"
echo "    --payload '{\"model_id\":\"<your-model-id>\"}' \\"
echo "    --cli-binary-format raw-in-base64-out /tmp/out.json && cat /tmp/out.json"
