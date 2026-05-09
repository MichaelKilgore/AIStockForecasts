#!/usr/bin/env bash
# Daily run of orchestration.py with only execute_buy enabled.
# All flags are passed explicitly so behavior does not depend on argparse defaults.

set -euo pipefail

REPO="/home/michael/Coding/AIStockForecasts"
LOG_DIR="$REPO/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/execute_buy_$(date +%Y-%m-%d).log"

MODEL_ID="ubuntu-with-even-more-recent-training"

cd "$REPO"
# shellcheck disable=SC1091
source venv/bin/activate

exec python3 src/ai_stock_forecasts/orchestration/orchestration.py \
  --model_id "$MODEL_ID" \
  --run_training False \
  --run_batch_inference False \
  --run_evaluation False \
  --execute_buy True \
  --run_checkpoint_upload False \
  --testing False \
  >>"$LOG_FILE" 2>&1

