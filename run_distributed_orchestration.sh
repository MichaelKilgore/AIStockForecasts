#!/bin/bash

rm -rf /tmp/pivoted_1_TimeFrameUnit.Day.parquet

PYTHONPATH=src torchrun --nproc_per_node=2 src/ai_stock_forecasts/orchestration/orchestration.py

rm -rf /tmp/pivoted_1_TimeFrameUnit.Day.parquet
