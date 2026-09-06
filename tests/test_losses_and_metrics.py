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
    def test_sota_optimizers(self):
        from src.training.optimizers import build_optimizer, Muon, Lion
        model = torch.nn.Linear(10, 5)

        opt_muon = build_optimizer(model, opt_type="muon", lr=0.02)
        self.assertIsInstance(opt_muon, Muon)

        opt_lion = build_optimizer(model, opt_type="lion", lr=1e-4)
        self.assertIsInstance(opt_lion, Lion)

        # Execute optimization step
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        opt_muon.step()
        opt_lion.step()


if __name__ == "__main__":
    unittest.main()
