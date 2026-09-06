"""
Reproducibility utilities for deterministic experiments.
Guarantees identical behavior across runs in PyTorch, NumPy, and Python standard library.
"""

import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """
    Sets deterministic seeds across all random number generators.

    Args:
        seed: Integer seed value (default: 42).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Guarantees deterministic convolution algorithms in cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """
    Worker initialization function for PyTorch DataLoader to ensure
    workers spawn with deterministic seeds.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
