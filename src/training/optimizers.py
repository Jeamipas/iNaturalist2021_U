"""
Optimizer and Learning Rate Scheduler factories.
"""

from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW, RMSprop
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau, _LRScheduler


def build_optimizer(
    model: nn.Module,
    opt_type: str = "adamw",
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    momentum: float = 0.9,
    param_groups: Optional[List[Dict[str, Any]]] = None
) -> torch.optim.Optimizer:
    """
    Builds optimizers for training.

    Args:
        model: PyTorch model.
        opt_type: 'sgd', 'sgd_momentum', 'adam', 'adamw', 'rmsprop'.
        lr: Base learning rate.
        weight_decay: L2 regularization factor.
        momentum: Momentum factor for SGD.
        param_groups: Optional custom parameter groups (e.g. for differential lr).

    Returns:
        torch.optim.Optimizer instance.
    """
    params = param_groups if param_groups is not None else [
        p for p in model.parameters() if p.requires_grad
    ]
    otype = opt_type.lower()

    if otype == "sgd":
        return SGD(params, lr=lr, weight_decay=weight_decay)

    elif otype in ["sgd_momentum", "sgd_m", "sgdm"]:
        return SGD(params, lr=lr, momentum=momentum, nesterov=True, weight_decay=weight_decay)

    elif otype == "adam":
        return Adam(params, lr=lr, weight_decay=weight_decay)

    elif otype == "adamw":
        return AdamW(params, lr=lr, weight_decay=weight_decay)

    elif otype == "rmsprop":
        return RMSprop(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    else:
        raise ValueError(f"Unknown optimizer: '{opt_type}'. Choose from 'sgd', 'sgd_momentum', 'adam', 'adamw', 'rmsprop'.")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    epochs: int = 10,
    min_lr: float = 1e-6,
    step_size: int = 5,
    gamma: float = 0.1
) -> Optional[Union[_LRScheduler, ReduceLROnPlateau]]:
    """
    Builds learning rate schedulers.

    Args:
        optimizer: PyTorch optimizer.
        scheduler_type: 'cosine', 'step', 'plateau', or 'none'.
        epochs: Total training epochs.
        min_lr: Minimum learning rate floor for cosine annealing.
        step_size: Epoch period for StepLR.
        gamma: Multiplicative factor of learning rate decay.

    Returns:
        Scheduler instance or None.
    """
    stype = scheduler_type.lower() if scheduler_type else "none"

    if stype == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)

    elif stype == "step":
        return StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif stype == "plateau":
        return ReduceLROnPlateau(optimizer, mode="max", factor=gamma, patience=2, min_lr=min_lr)

    elif stype in ["none", ""]:
        return None

    else:
        raise ValueError(f"Unknown scheduler_type: '{scheduler_type}'. Choose from 'cosine', 'step', 'plateau', 'none'.")
