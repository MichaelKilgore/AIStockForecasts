#!/usr/bin/env bash

aws lambda invoke --function-name check-if-execute-buy-ran --region us-east-2 \
--payload '{"model_id":"ubuntu-with-even-more-recent-training"}' \
--cli-binary-format raw-in-base64-out /tmp/out.json && cat /tmp/out.json

