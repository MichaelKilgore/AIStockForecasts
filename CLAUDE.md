
run the following to test run_training for tft model:

source venv/bin/activate && ATHENA_TABLE_S3_PREFIX=test_data timeout 120 python3 src/ai_stock_forecasts/orchestration/orchestration.py --model_id tft-test-model --run_training True 2>&1 | tail -60

some other variables can also be set to test other paths:

--run_batch_inference
--run_evaluation
--execute_buy
--run_checkpoint_upload

You can also set model_id to lgbm-test-model to test LGBM model type as well.

Do not attempt to test running orchestration.py with any other model type as depending on your configuration these operations can take a long time.
