"""
GSPO: Group Sequence Policy Optimization Implementation

Implementation of the GSPO algorithm by Zheng et al. (Qwen Team, Alibaba Inc.)
Optimized for H100 GPUs with comprehensive baseline comparisons.
"""

# Core imports that should always work
from .trainer import GSPOTrainer, GSPOConfig, create_math_reward_function, create_code_reward_function
from .dataset import GSPOCustomDataset, create_custom_gspo_dataset

# Optional imports for data loading
try:
    from .data_loader import DatasetLoader, create_reward_evaluator
    DATA_LOADER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: DatasetLoader not available due to missing dependencies: {e}")
    DATA_LOADER_AVAILABLE = False
    DatasetLoader = None
    create_reward_evaluator = None

__version__ = "0.1.0"
__author__ = "GSPO Implementation Team"

# Export only what's available
__all__ = [
    "GSPOTrainer",
    "GSPOConfig", 
    "GSPOCustomDataset",
    "create_math_reward_function",
    "create_code_reward_function", 
    "create_custom_gspo_dataset",
]

if DATA_LOADER_AVAILABLE:
    __all__.extend(["DatasetLoader", "create_reward_evaluator"]) 