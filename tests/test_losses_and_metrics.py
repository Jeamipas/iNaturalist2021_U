"""
Tests for loss functions and classification metrics.
"""

import unittest
import torch
import numpy as np
from src.training.losses import build_criterion, FocalLoss
from src.utils.metrics import compute_classification_metrics, compute_top_k_accuracy


class TestLossesAndMetrics(unittest.TestCase):

    def test_focal_loss_computation_and_backward(self):
        criterion = FocalLoss(gamma=2.0)
        logits = torch.randn(8, 10, requires_grad=True)
        targets = torch.randint(0, 10, (8,))

        loss = criterion(logits, targets)
        self.assertTrue(torch.is_tensor(loss))
        self.assertGreater(loss.item(), 0.0)

        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_loss_factory(self):
        ce = build_criterion("cross_entropy")
        self.assertIsInstance(ce, torch.nn.CrossEntropyLoss)

        weights = torch.ones(5)
        w_ce = build_criterion("weighted_cross_entropy", class_weights=weights)
        self.assertIsInstance(w_ce, torch.nn.CrossEntropyLoss)

        focal = build_criterion("focal_loss", gamma=2.0)
        self.assertIsInstance(focal, FocalLoss)

    def test_metrics_computation(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 1, 2])
        # Logits of shape (6, 5)
        y_probs = np.eye(6, 5)

        tiers = {
            "frequent": [0, 1],
            "minority": [2]
        }

        metrics = compute_classification_metrics(y_true, y_pred, y_probs=y_probs, class_tiers=tiers)

        self.assertIn("top1_acc", metrics)
        self.assertIn("top5_acc", metrics)
        self.assertIn("macro_f1", metrics)
        self.assertIn("weighted_f1", metrics)
        self.assertIn("frequent_macro_f1", metrics)
        self.assertIn("minority_macro_f1", metrics)
        self.assertGreater(metrics["top1_acc"], 0.8)


if __name__ == "__main__":
    unittest.main()
