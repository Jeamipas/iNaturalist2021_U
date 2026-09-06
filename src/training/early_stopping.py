"""
Early Stopping mechanism with best model checkpointing and weight restoration.
"""

import copy
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn


class EarlyStopping:
    """
    Monitors validation performance and stops training early when metric ceases improving.
    Optionally restores the best weights at termination.
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 1e-4,
        mode: str = "max",
        metric_name: str = "val_macro_f1",
        checkpoint_path: Optional[str] = None,
        restore_best_weights: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode.lower()
        self.metric_name = metric_name
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.restore_best_weights = restore_best_weights

        self.best_score: Optional[float] = None
        self.counter: int = 0
        self.should_stop: bool = False
        self.best_weights: Optional[dict] = None

        if self.mode not in ["min", "max"]:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

    def __call__(self, current_score: float, model: nn.Module) -> bool:
        """
        Updates stopping state given the current validation metric.

        Returns:
            True if training should stop, False otherwise.
        """
        if self.best_score is None:
            self._save_checkpoint(current_score, model)
            return False

        if self.mode == "max":
            improved = (current_score - self.best_score) > self.min_delta
        else:
            improved = (self.best_score - current_score) > self.min_delta

        if improved:
            self._save_checkpoint(current_score, model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def _save_checkpoint(self, score: float, model: nn.Module) -> None:
        self.best_score = score
        if self.restore_best_weights:
            self.best_weights = copy.deepcopy(model.state_dict())

        if self.checkpoint_path:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "score": score,
                "metric_name": self.metric_name,
                "model_state_dict": model.state_dict()
            }, self.checkpoint_path)

    def restore(self, model: nn.Module) -> None:
        """Restores the best weights to the model."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
        elif self.checkpoint_path and self.checkpoint_path.exists():
            data = torch.load(self.checkpoint_path, map_location="cpu")
            model.load_state_dict(data["model_state_dict"])
