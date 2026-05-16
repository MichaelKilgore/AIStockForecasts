#!/usr/bin/env bash
# Run orchestration.py with run_training enabled, resuming from last.ckpt and
# capped at 5 wall-clock hours via Lightning's max_time. Bails out if invoked
# more than MAX_LATENESS seconds past the most recent scheduled slot
# (SCHEDULE_DOW / SCHEDULE_HOUR) so a stale catch-up doesn't kick off a long
# training run at the wrong time.

set -euo pipefail

REPO="/home/michael/Coding/AIStockForecasts"
LOG_DIR="$REPO/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/training_$(date +%Y-%m-%d).log"

MODEL_ID="ubuntu-with-long-training"
MAX_HOURS_RUN=5

# Skip the run if we're more than MAX_LATENESS seconds past the most recent
# scheduled slot. SCHEDULE_DOW: 1=Mon..7=Sun (date +%u). Set to 0 to treat the
# schedule as daily (last slot = today's SCHEDULE_HOUR if past, else yesterday's).
SCHEDULE_DOW=0        # daily (cron handles Mon-Fri filter)
SCHEDULE_HOUR=10      # 10 AM, system local time (America/Chicago)
MAX_LATENESS=$((24 * 3600))

now_epoch=$(date +%s)
today_dow=$(date +%u)
today_hour=$((10#$(date +%H)))

if [[ "$SCHEDULE_DOW" -eq 0 ]]; then
  days_back=$([[ "$today_hour" -lt "$SCHEDULE_HOUR" ]] && echo 1 || echo 0)
else
  days_back=$(( (today_dow - SCHEDULE_DOW + 7) % 7 ))
  if [[ "$days_back" -eq 0 && "$today_hour" -lt "$SCHEDULE_HOUR" ]]; then
    days_back=7
  fi
fi

last_scheduled_date=$(date -d "${days_back} days ago" +%Y-%m-%d)
last_scheduled=$(date -d "${last_scheduled_date} ${SCHEDULE_HOUR}:00:00" +%s)
lateness=$(( now_epoch - last_scheduled ))

if (( lateness > MAX_LATENESS )); then
  echo "[$(date -Iseconds)] skipping: ${lateness}s past last scheduled slot ($(date -d "@$last_scheduled" -Iseconds)); > MAX_LATENESS=${MAX_LATENESS}s" >>"$LOG_FILE"
  exit 0
fi

cd "$REPO"
# shellcheck disable=SC1091
source venv/bin/activate

exec env PYTHONPATH=src python3 src/ai_stock_forecasts/orchestration/orchestration.py \
  --model_id "$MODEL_ID" \
  --run_training True \
  --resume_from_last_ckpt True \
  --max_hours_run "$MAX_HOURS_RUN" \
  --run_batch_inference False \
  --run_evaluation False \
  --execute_buy False \
  --run_checkpoint_upload False \
  --testing False \
  >>"$LOG_FILE" 2>&1
