import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker import get_execution_role

session = sagemaker.Session()
role = get_execution_role()  # or hardcode ARN if running locally

estimator = PyTorch(
    entry_point="orchestration/orchestration.py",
    source_dir="ai_stock_forecasts",
    role=role,
    framework_version="2.9",
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
        "symbols_path": "constants/symbols.txt",  # packaged in source_dir
    },
    dependencies=["../requirements.txt"],    # install these before training
)

estimator.fit()

