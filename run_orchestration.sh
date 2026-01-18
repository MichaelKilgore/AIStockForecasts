#!/usr/bin/env bash
set -e

PROJECT_DIR=/home/ubuntu/AIStockForecasts
VENV_DIR=$PROJECT_DIR/.venv

cd $PROJECT_DIR

source $VENV_DIR/bin/activate

python3 src/ai_stock_forecasts/model/orchestration/orchestration.py
