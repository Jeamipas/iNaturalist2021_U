"""
Evaluation metrics module adhering to the course rubric:
- Top-1 Accuracy
- Top-5 Accuracy
- Macro F1-score
- Weighted F1-score
- Long-Tail Tier Breakdown (Frequent, Medium, Minoritary)
"""

from typing import Dict, List, Optional, Union
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


def compute_top_k_accuracy(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    k: int = 5
) -> float:
    """
    Computes Top-K accuracy given ground truth labels and predicted class probabilities.

    Args:
        y_true: 1D array of ground truth labels of shape (N,).
        y_probs: 2D array of class probabilities or logits of shape (N, C).
        k: Integer k for Top-k accuracy (default: 5).

    Returns:
        Top-k accuracy as float in [0.0, 1.0].
    """
    if y_probs is None or y_probs.shape[1] < k:
        return 0.0
    
    # Get top k indices along columns
    top_k_preds = np.argsort(y_probs, axis=1)[:, -k:]
    correct = [
        y_true[i] in top_k_preds[i]
        for i in range(len(y_true))
    ]
    return float(np.mean(correct))


def compute_classification_metrics(
    y_true: Union[List[int], np.ndarray, torch.Tensor],
    y_pred: Union[List[int], np.ndarray, torch.Tensor],
    y_probs: Optional[Union[np.ndarray, torch.Tensor]] = None,
    class_tiers: Optional[Dict[str, List[int]]] = None
) -> Dict[str, float]:
    """
    Computes comprehensive multi-class metrics required by the project rubric.

    Args:
        y_true: Ground truth target labels.
        y_pred: Predicted class labels (argmax).
        y_probs: Optional predicted logits or softmax probabilities.
        class_tiers: Optional dict mapping tier names ('frequent', 'medium', 'minority')
                     to lists of class labels for Long-Tail analysis.

    Returns:
        Dictionary containing all evaluation metrics.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_probs, torch.Tensor):
        y_probs = y_probs.detach().cpu().numpy()

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        "top1_acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    if y_probs is not None and y_probs.ndim == 2 and y_probs.shape[1] >= 5:
        metrics["top5_acc"] = compute_top_k_accuracy(y_true, y_probs, k=5)
    else:
        metrics["top5_acc"] = metrics["top1_acc"]

    # Compute Long-Tail Tier Breakdown if class_tiers is provided
    if class_tiers is not None:
        for tier_name, tier_classes in class_tiers.items():
            tier_set = set(tier_classes)
            tier_mask = np.isin(y_true, list(tier_set))

            if np.sum(tier_mask) > 0:
                tier_true = y_true[tier_mask]
                tier_pred = y_pred[tier_mask]
                metrics[f"{tier_name}_acc"] = float(accuracy_score(tier_true, tier_pred))
                metrics[f"{tier_name}_macro_f1"] = float(f1_score(tier_true, tier_pred, average="macro", zero_division=0))
                metrics[f"{tier_name}_recall"] = float(recall_score(tier_true, tier_pred, average="macro", zero_division=0))
                metrics[f"{tier_name}_count"] = int(np.sum(tier_mask))
            else:
                metrics[f"{tier_name}_acc"] = 0.0
                metrics[f"{tier_name}_macro_f1"] = 0.0
                metrics[f"{tier_name}_recall"] = 0.0
                metrics[f"{tier_name}_count"] = 0

    return metrics
