"""
End-to-end smoke test for Trainer, EarlyStopping, and ExperimentTracker.
"""

import unittest
import shutil
from pathlib import Path
import torch
from src.data.dataset import SyntheticINatDataset
from src.data.dataloader import build_dataloaders
from src.models.cnn_custom import MiniINatCNN
from src.training.losses import build_criterion
from src.training.optimizers import build_optimizer
from src.training.trainer import Trainer
from src.training.early_stopping import EarlyStopping
from src.utils.tracking import ExperimentTracker


class TestTrainerSmoke(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = Path("./tmp_smoke_test")
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_trainer_fit_and_tracking(self):
        train_ds = SyntheticINatDataset(num_samples=32, num_classes=5, img_size=32, seed=42)
        val_ds = SyntheticINatDataset(num_samples=16, num_classes=5, img_size=32, seed=43)

        train_loader, val_loader = build_dataloaders(train_ds, val_ds, batch_size=8, num_workers=0)

        model = MiniINatCNN(num_classes=5, use_batchnorm=True, dropout_rate=0.1)
        criterion = build_criterion("cross_entropy")
        optimizer = build_optimizer(model, opt_type="adamw", lr=1e-3)

        early_stopping = EarlyStopping(
            patience=2,
            metric_name="val_macro_f1",
            checkpoint_path=str(self.tmp_dir / "best_model.pt")
        )

        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=torch.device("cpu"),
            early_stopping=early_stopping,
            use_amp=False
        )

        history, best_metrics, peak_vram, total_time = trainer.fit(
            train_loader, val_loader, epochs=2, verbose=False
        )

        self.assertEqual(len(history["train_loss"]), 2)
        self.assertIn("top1_acc", best_metrics)
        self.assertIn("macro_f1", best_metrics)

        # Test tracker
        tracker = ExperimentTracker(log_dir=str(self.tmp_dir / "logs"))
        tracker.log_experiment(
            exp_id="Smoke_01",
            model_name="MiniINatCNN",
            optimizer="AdamW",
            regularization="BN+Dropout",
            augmentation="None",
            transfer_learning="No",
            long_tail="No",
            metrics=best_metrics,
            training_time_sec=total_time,
            peak_vram_mb=peak_vram,
            param_count_m=0.1
        )

        md_table = tracker.to_markdown_table()
        self.assertIn("Smoke_01", md_table)
        self.assertIn("MiniINatCNN", md_table)


if __name__ == "__main__":
    unittest.main()
