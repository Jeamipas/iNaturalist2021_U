"""
Hybrid Optimizer: orchestrates multiple distinct optimizers across model submodules.
Implements the Moonshot AI (Kimi K2/K3) recipe:
- Muon (Newton-Schulz orthogonalization) for internal 2D/4D convolution and attention weight matrices
- AdamW for classification heads (Linear) and 1D normalization layers (LayerNorm / BatchNorm)
"""

from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from .optimizers import Muon


class HybridOptimizer(Optimizer):
    """
    Composite optimizer that dispatches parameter updates across specialized sub-optimizers
    while exposing a standard unified PyTorch Optimizer interface compatible with
    AMP GradScaler and learning rate schedulers.
    """

    def __init__(self, optimizers: List[Optimizer]):
        self.optimizers = optimizers
        # Collect and link all parameter groups from child optimizers
        param_groups = [g for opt in optimizers for g in opt.param_groups]
        super().__init__(param_groups, defaults={})
        for opt in optimizers:
            self.state.update(opt.state)

    def step(self, closure=None):
        loss = None
        for opt in self.optimizers:
            loss = opt.step(closure)
        return loss

    def zero_grad(self, set_to_none: bool = True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "optimizers": [opt.state_dict() for opt in self.optimizers],
            "state": self.state,
            "param_groups": self.param_groups
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        if "optimizers" in state_dict:
            for opt, sd in zip(self.optimizers, state_dict["optimizers"]):
                opt.load_state_dict(sd)
        super().load_state_dict(state_dict)


def build_hybrid_muon_adamw(
    model: nn.Module,
    muon_lr: float = 0.005,
    adamw_lr: float = 1e-3,
    weight_decay: float = 1e-4,
    momentum: float = 0.95
) -> HybridOptimizer:
    """
    Splits model parameters into:
    1. 2D/4D internal backbone tensors -> Muon (orthogonal momentum)
    2. Head parameters, biases, and 1D normalization tensors -> AdamW

    Args:
        model: Model containing .backbone and .head (or general nn.Module)
        muon_lr: Learning rate for 2D/4D internal weights
        adamw_lr: Learning rate for classifier head and 1D parameters
        weight_decay: L2 penalty for AdamW
        momentum: Momentum factor for Muon

    Returns:
        HybridOptimizer instance ready for training
    """
    muon_params = []
    adamw_params = []

    # If model has head attribute, separate backbone and head
    head_param_ids = set()
    if hasattr(model, "head") and isinstance(model.head, nn.Module):
        head_param_ids = set(id(p) for p in model.head.parameters())

    for p in model.parameters():
        if not p.requires_grad:
            continue

        if id(p) in head_param_ids:
            # Classification head always goes to AdamW for unconstrained logit calibration
            adamw_params.append(p)
        elif p.ndim >= 2:
            # 2D/4D hidden weights benefit from Newton-Schulz orthogonalization
            muon_params.append(p)
        else:
            # 1D biases and normalization gains go to AdamW
            adamw_params.append(p)

    optimizers = []
    if muon_params:
        opt_muon = Muon(muon_params, lr=muon_lr, momentum=momentum, weight_decay=weight_decay)
        optimizers.append(opt_muon)

    if adamw_params:
        use_fused = torch.cuda.is_available()
        try:
            opt_adamw = AdamW(adamw_params, lr=adamw_lr, weight_decay=weight_decay, fused=use_fused)
        except Exception:
            opt_adamw = AdamW(adamw_params, lr=adamw_lr, weight_decay=weight_decay)
        optimizers.append(opt_adamw)

    return HybridOptimizer(optimizers)
