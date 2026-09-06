"""
Loss functions module:
- Standard Cross-Entropy
- Weighted Cross-Entropy (for Long-Tail)
- Multi-Class Focal Loss (for extreme class imbalance)
"""

from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    r"""
    Multi-class Focal Loss:
    FL(p_t) = - \alpha_t (1 - p_t)^\gamma \log(p_t)
    Down-weights easy examples and focuses training on hard / minority classes.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[Union[float, torch.Tensor]] = None,
        reduction: str = "mean"
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None and not isinstance(alpha, torch.Tensor):
            self.alpha = torch.tensor(alpha)
        else:
            self.alpha = alpha

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Raw logits of shape (N, C).
            targets: Class labels of shape (N,).
        """
        # Cross entropy computes -log(p_t)
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        p_t = torch.exp(-ce_loss)  # probability of true class

        # Modulating factor (1 - p_t)^gamma
        focal_weight = (1.0 - p_t) ** self.gamma

        loss = focal_weight * ce_loss

        # Apply class balancing alpha weight if provided
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def build_criterion(
    loss_type: str = "cross_entropy",
    class_weights: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
    label_smoothing: float = 0.0
) -> nn.Module:
    """
    Loss factory supporting standard, weighted, and focal losses.

    Args:
        loss_type: 'cross_entropy', 'weighted_cross_entropy', or 'focal_loss'.
        class_weights: Optional 1D Tensor of per-class weights.
        gamma: Focusing parameter for Focal Loss.
        label_smoothing: Label smoothing rate for regularizing cross-entropy.
    """
    ltype = loss_type.lower()

    if ltype == "cross_entropy":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    elif ltype in ["weighted_cross_entropy", "weighted_ce"]:
        if class_weights is None:
            raise ValueError("class_weights must be provided for weighted_cross_entropy.")
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    elif ltype == "focal_loss":
        return FocalLoss(gamma=gamma, alpha=class_weights)

    else:
        raise ValueError(f"Unknown loss_type: '{loss_type}'. Choose from 'cross_entropy', 'weighted_cross_entropy', 'focal_loss'.")
