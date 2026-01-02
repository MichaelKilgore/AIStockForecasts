import sagemaker
from sagemaker.pytorch import PyTorch

session = sagemaker.Session()
role = "arn:aws:iam::270673621735:role/SageMaker"

"""
cost us-east-2 (cheapest region apparently):
    1. g4dn.xlarge = $0.526 per hour | 4 vCPU, 16 GiB Memory, 125 GB NVMe SSD, Up to 25 Gigabit Network performance -> logs recommended 3 workers for val and train module
    2. g4dn.2xlarge = $0.752 per hour | 8 vCPU, 32 GiB Memory, 225 GB NVMe SSD, Up to 25 Gigabit Network performance
    3. g4dn.4xlarge = $1.204 per hour | 16 vCPU, 64 GiB Memory, 225 GB NVMe SSD, Up to 25 Gigabit Network performance
    4. g4dn.8xlarge = $2.176 per hour | 32 vCPU, 128 GiB Memory, 900 GB NVMe SSD, 50 Gigabit Network performance
    5. g4dn.12xlarge = $3.912 per hour | 48 vCPU, 192 GiB Memory, 900 GB NVMe SSD, 50 Gigabit Network performance
    6. g4dn.16xlarge = $4.352 per hour | 64 vCPU, 256 GiB Memory, 900 GB NVMe SSD, 50 Gigabit Network performance
    7. g4dn.metal = $7.824 per hour | 96 vCPU, 384 GiB Memory, 2 x 900 GB NVMe SSD, 100 Gigabit Network performance
    8. p3.2xlarge -> logs recommend 7 workers for val and train module
"""

""" hidden_size = core model width, number of hidden units used in LSTM encoder/decoder and the gating/mixing layers inside TFT
                    typical values: 16, 32, 64, 128
                    higher = more expressive but slower and easier to overfit.
                    increase when running on beefer gpu / cpu
    attention_head_size = the number of attention heads
                    more heads means model can learn multiple types of temporal relationships in parallel
                    typical values: 1, 4, 8
                    higher = increases model expressiveness, smaller datasets prefer 1-4 heads
                    increase based on increase in data
                    chatgpt recommendation:
                        <50k rows <20 symbols = 1
                        50-500k rows 20-200 symbols = 2-4
                        500k rows 200-500 symbols = 4-8
                        5 million rows 1000 symbols = 8
    dropout = probability of dropping units during training to prevent overfitting.
                    typical values = 0.0-0.3
                    higher = more regularization
                    lower = more capacity, more risk of overfitting
    hidden_continuous_size = size in which data is compressed to.
                    typical values = 8, 16, 32
                    small = may compress too aggressively
                    large = more expressive, but slower and more prone to overfitting
    lstm_layers = number of stacked lstm layers
                    typical values = 1 or 2
                    more layers = deeper sequence modeling, higher compute and harder to train
                    more than 2 rarely helps
    reduce_on_plateau_patience = how many validation epochs without improvement before the learning rate is reduced.
                    typical values 2 - 5
                    
"""

estimator = PyTorch(
    entry_point="ai_stock_forecasts/orchestration/orchestration.py",
    source_dir="src",
    role=role,
    framework_version="2.8",
    py_version="py312",
    instance_type="ml.p3.2xlarge",
    instance_count=1,
    hyperparameters={
        "hidden_size": 128,
        "attention_head_size": 8,
        "dropout": 0.1,
        "hidden_continuous_size": 32,
        "lstm_layers": 2,
        "learning_rate": 0.001,
        "max_epochs": 50,
        "train_start": "2019-01-02",
        "train_end": "2024-01-01",
        "val_start": "2024-01-02",
        "val_end": "2025-01-01",
        "max_lookback_period": 60,
        "max_prediction_length": 2,
        "symbols_path": "ai_stock_forecasts/constants/symbols.txt",
        "accelerator": "gpu",
        "num_workers": 7,
        "batch_size": 128
    },
    dependencies=["src/requirements.txt"],
)

estimator.fit()

