"""
Optimizer and Learning Rate Scheduler factories.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW, RMSprop
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau, _LRScheduler


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    r"""
    Newton-Schulz iteration (order 5) to compute the zeroth power / orthogonal projection of matrix G.
    Coefficients: a = 3.4445, b = -4.7750, c = 2.0315 (Keller Jordan, 2024).
    Equalizes all singular values to 1.0, ensuring balanced gradient updates across all dimensions.
    """
    assert len(G.shape) == 2, f"Expected 2D matrix, got shape {G.shape}"
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    X /= (X.norm() + eps)
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    r"""
    Muon (Momentum Orthogonalized by Newton-Schulz) Optimizer.
    Invented by Keller Jordan; adopted by Moonshot AI (Kimi K2/K3) and Zhipu AI.
    Applies orthogonalized momentum updates to 2D weight matrices for ~2x sample efficiency.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.01
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0:
                    p.data.mul_(1.0 - lr * wd)

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)

                g_update = g.add(buf, alpha=momentum) if nesterov else buf

                orig_shape = g_update.shape
                if len(orig_shape) >= 2:
                    g_2d = g_update.reshape(orig_shape[0], -1)
                    ortho_update = zeropower_via_newtonschulz5(g_2d, steps=ns_steps).reshape(orig_shape)
                    scale = max(1.0, g_2d.size(0) / g_2d.size(1)) ** 0.5
                    p.data.add_(ortho_update, alpha=-lr * scale)
                else:
                    p.data.add_(g_update, alpha=-lr)

        return loss


class Lion(torch.optim.Optimizer):
    r"""
    Lion (EvoLved Sign Momentum) Optimizer from Google Research.
    Uses sign(momentum) and consumes ~50% less optimizer memory than AdamW.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0
    ):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0:
                    p.data.mul_(1.0 - lr * wd)

                state = self.state[p]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]

                # Update weights using sign of blended gradient and momentum
                update = exp_avg.mul(beta1).add(g, alpha=1.0 - beta1).sign()
                p.data.add_(update, alpha=-lr)

                # Decay the momentum buffer
                exp_avg.mul_(beta2).add_(g, alpha=1.0 - beta2)

        return loss


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
    use_fused = torch.cuda.is_available()

    if otype == "sgd":
        return SGD(params, lr=lr, weight_decay=weight_decay)

    elif otype in ["sgd_momentum", "sgd_m", "sgdm"]:
        return SGD(params, lr=lr, momentum=momentum, nesterov=True, weight_decay=weight_decay)

    elif otype == "adam":
        try:
            return Adam(params, lr=lr, weight_decay=weight_decay, fused=use_fused)
        except Exception:
            return Adam(params, lr=lr, weight_decay=weight_decay)

    elif otype in ["adamw", "fused_adamw"]:
        try:
            return AdamW(params, lr=lr, weight_decay=weight_decay, fused=use_fused)
        except Exception:
            return AdamW(params, lr=lr, weight_decay=weight_decay)

    elif otype == "muon":
        return Muon(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    elif otype == "lion":
        return Lion(params, lr=lr, weight_decay=weight_decay)

    elif otype == "rmsprop":
        return RMSprop(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    else:
        raise ValueError(
            f"Unknown optimizer: '{opt_type}'. Choose from 'sgd', 'sgd_momentum', 'adam', 'adamw', 'muon', 'lion', 'rmsprop'."
        )


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
