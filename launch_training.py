import sagemaker
from sagemaker.pytorch import PyTorch
import os

session = sagemaker.Session()
role = "arn:aws:iam::270673621735:role/SageMaker"

"""
cost us-east-2 (cheapest region apparently):
    1. g4dn.xlarge = $0.526 per hour | 4 vCPU, 16 GiB Memory, 125 GB NVMe SSD, Up to 25 Gigabit Network performance
    2. g4dn.2xlarge = $0.752 per hour | 8 vCPU, 32 GiB Memory, 225 GB NVMe SSD, Up to 25 Gigabit Network performance
    3. g4dn.4xlarge = $1.204 per hour | 16 vCPU, 64 GiB Memory, 225 GB NVMe SSD, Up to 25 Gigabit Network performance
    4. g4dn.8xlarge = $2.176 per hour | 32 vCPU, 128 GiB Memory, 900 GB NVMe SSD, 50 Gigabit Network performance
    5. g4dn.12xlarge = $3.912 per hour | 48 vCPU, 192 GiB Memory, 900 GB NVMe SSD, 50 Gigabit Network performance
    6. g4dn.16xlarge = $4.352 per hour | 64 vCPU, 256 GiB Memory, 900 GB NVMe SSD, 50 Gigabit Network performance
    7. g4dn.metal = $7.824 per hour | 96 vCPU, 384 GiB Memory, 2 x 900 GB NVMe SSD, 100 Gigabit Network performance
"""
estimator = PyTorch(
    entry_point="ai_stock_forecasts/orchestration/orchestration.py",
    source_dir="src",
    role=role,
    framework_version="2.8",
    py_version="py312",
    instance_type="ml.g4dn.xlarge",
    instance_count=1,
    hyperparameters={
        "hidden_size": 32,
        "attention_head_size": 4,
        "dropout": 0.1,
        "hidden_continuous_size": 16,
        "lstm_layers": 1,
        "learning_rate": 1e-4,
        "max_epochs": 50,
        "train_start": "2019-01-02",
        "train_end": "2024-01-01",
        "val_start": "2024-01-02",
        "val_end": "2025-01-01",
        "max_lookback_period": 1000,
        "max_prediction_length": 14,
        "symbols_path": "ai_stock_forecasts/constants/symbols.txt",
        "accelerator": "gpu"
    },
    dependencies=["src/requirements.txt"],
)

estimator.fit()

