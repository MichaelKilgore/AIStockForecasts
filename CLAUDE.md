
run the following to test run_training for tft model:

source venv/bin/activate && ATHENA_TABLE_S3_PREFIX=test_data timeout 120 python3 src/ai_stock_forecasts/orchestration/orchestration.py --model_id tft-test-model --run_training True 2>&1 | tail -60

some other variables can also be set to test other paths:

--run_batch_inference
--run_evaluation
--execute_buy
--run_checkpoint_upload

You can also set model_id to lgbm-test-model to test LGBM model type as well.

Do not attempt to test running orchestration.py with any other model type as depending on your configuration these operations can take a long time.

## Variant models

A config entry may set `variant_of: <base_model_id>` to declare a variant. A variant reuses the base model's trained checkpoint and may only override `preferred_trading_strategy` (and its nested strategy params). Variants exist so the same trained model can be scheduled for `execute_buy` under different trading strategies, with each variant's trades tracked separately in the `transactions` table under its own model_id.

Rules:
- Only `execute_buy` is supported on a variant. `run_training`, `run_batch_inference`, `run_evaluation`, and `run_checkpoint_upload` will raise — run those against the base model.
- A variant may not point to another variant.
- A variant config may only contain `variant_of`, `preferred_trading_strategy`, and the nested strategy dict.

Scheduling a variant's `execute_buy` is identical to scheduling any other model_id.
