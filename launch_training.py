import sagemaker
from sagemaker.pytorch import PyTorch

session = sagemaker.Session()
role = "arn:aws:iam::270673621735:role/SageMaker"

"""
cost us-east-2 (cheapest region apparently):
    1. g4dn.xlarge = $0.7364 per hour | 4 vCPU, 16 GiB Memory, 125 GB NVMe SSD, Up to 25 Gigabit Network performance, 1 NVIDIA T4 16gb gpu -> logs recommended 3 workers for val and train module
    2. g4dn.2xlarge = $0.752 per hour | 8 vCPU, 32 GiB Memory, 225 GB NVMe SSD, Up to 25 Gigabit Network performance
    3. g4dn.4xlarge = $1.204 per hour | 16 vCPU, 64 GiB Memory, 225 GB NVMe SSD, Up to 25 Gigabit Network performance
    4. g4dn.8xlarge = $2.176 per hour | 32 vCPU, 128 GiB Memory, 900 GB NVMe SSD, 50 Gigabit Network performance
    5. g4dn.12xlarge = $3.912 per hour | 48 vCPU, 192 GiB Memory, 900 GB NVMe SSD, 50 Gigabit Network performance
    6. g4dn.16xlarge = $4.352 per hour | 64 vCPU, 256 GiB Memory, 900 GB NVMe SSD, 50 Gigabit Network performance
    7. g4dn.metal = $7.824 per hour | 96 vCPU, 384 GiB Memory, 2 x 900 GB NVMe SSD, 100 Gigabit Network performance
    8. p3.2xlarge -> $3.06 per hour | logs recommend 7 workers for val and train module
    9. p3.8xlarge -> 32 vCPUs, 244 Gib of Memory, 4 NVIDIA Tesla V100 for a total of 64 VRAM. cost around 12$ per hour

good for just running batch inference maybe:
    1. ml.r7i.8xlarge -> 32 vCPUs, 256 Gib of Memory
"""

config_id = 'sagemaker-simple-daily-1-with-time-features'

estimator = PyTorch(
    entry_point="ai_stock_forecasts/model/orchestration.py",
    source_dir="src",
    role=role,
    framework_version="2.8",
    py_version="py312",
    instance_type="ml.g4dn.xlarge",
    instance_count=1,
    hyperparameters={
        "symbols_path": "ai_stock_forecasts/constants/symbols.txt",
        "config_path": "ai_stock_forecasts/constants/configs.yaml",
        "config_id": config_id,
    },
    dependencies=["src/requirements.txt"],
)

estimator.fit(job_name=config_id)

