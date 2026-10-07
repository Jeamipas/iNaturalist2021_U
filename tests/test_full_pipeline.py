"""Meaningful checks for full-corpus metrics, Muon resume, and accumulation."""
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image

from src.full_data import read_split
from src.full_experiment import StreamingMetrics, MuonAdam, make_model, make_optimizer, train_epoch, seed_all, run_experiment
from scripts.execute_project_notebook import requeue_current_allocation


class TestFullPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_metrics_include_unpredicted_species_without_dense_probabilities(self):
        metrics = StreamingMetrics(6)
        logits = torch.tensor([[9., 0, 0, 0, 0, -1], [0., 9, 0, 0, 0, -1], [0., 0, 9, 0, 0, -1]])
        metrics.update(logits, torch.tensor([0, 0, 2]))
        values = metrics.compute()
        self.assertAlmostEqual(values["top1"], 2 / 3)
        self.assertAlmostEqual(values["macro_f1"], (2 / 3 + 1) / 6)
        self.assertAlmostEqual(values["weighted_f1"], (4 / 3 + 1) / 3)
        self.assertEqual(values["top5"], 1)
        self.assertEqual(metrics.support.nbytes, 6 * 8)

    def test_muon_auxiliary_adam_and_scheduler_resume_match_continuous_training(self):
        seed_all(42)
        left = nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 3))
        right = copy.deepcopy(left)
        def optimizer(model):
            return MuonAdam([model[0].weight], [model[0].bias, *model[2].parameters()], 0.02, 0.001)
        opt_left, opt_right = optimizer(left), optimizer(right)
        sched_left = torch.optim.lr_scheduler.StepLR(opt_left, 1, gamma=0.9)
        sched_right = torch.optim.lr_scheduler.StepLR(opt_right, 1, gamma=0.9)
        x, y = torch.randn(7, 4), torch.tensor([0, 1, 2, 0, 1, 2, 0])
        def step(model, opt, sched):
            opt.zero_grad()
            nn.functional.cross_entropy(model(x), y).backward()
            opt.step()
            sched.step()
        step(left, opt_left, sched_left)
        right.load_state_dict(left.state_dict())
        opt_right.load_state_dict(copy.deepcopy(opt_left.state_dict()))
        sched_right.load_state_dict(sched_left.state_dict())
        step(left, opt_left, sched_left)
        step(right, opt_right, sched_right)
        for a, b in zip(left.parameters(), right.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual([g["lr"] for g in opt_left.param_groups], [g["lr"] for g in opt_right.param_groups])

    def test_accumulation_final_short_group_matches_actual_batch_mean(self):
        seed_all(8)
        actual = nn.Linear(2, 3)
        expected = copy.deepcopy(actual)
        x = torch.randn(7, 2) * 0.1
        y = torch.tensor([0, 1, 2, 0, 1, 2, 0])
        opt_actual = torch.optim.SGD(actual.parameters(), lr=0.01)
        opt_expected = torch.optim.SGD(expected.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        for start, end in [(0, 6), (6, 7)]:
            opt_expected.zero_grad()
            criterion(expected(x[start:end]), y[start:end]).backward()
            nn.utils.clip_grad_norm_(expected.parameters(), 1.0)
            opt_expected.step()
        config = {"id": "test", "mode": "full", "architecture": "mlp", "accumulation": 2, "batch_size": 3}
        with tempfile.TemporaryDirectory() as folder:
            metrics = train_epoch(actual, DataLoader(TensorDataset(x, y), batch_size=3), opt_actual, criterion,
                                  torch.amp.GradScaler("cuda", enabled=False), torch.device("cpu"), config, 1, Path(folder) / "status.json")
        self.assertEqual(metrics["samples"], 7)
        for a, b in zip(actual.parameters(), expected.parameters()):
            torch.testing.assert_close(a, b)

    def test_scratch_is_fully_trainable_and_initialization_matches(self):
        config = {"seed": 42, "architecture": "convnext_tiny", "pretrained": False, "mode": "full", "dropout": 0.2}
        left = make_model(config, classes=6)
        right = make_model({**config, "optimizer": "muon"}, classes=6)
        self.assertTrue(all(p.requires_grad for p in left.parameters()))
        for a, b in zip(left.parameters(), right.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        optimizer = make_optimizer(left, {**config, "optimizer": "muon", "muon_lr": 0.02, "lr": 0.001})
        matrix_ids = {id(p) for g in optimizer.children[0].param_groups for p in g["params"]}
        adam_ids = {id(p) for g in optimizer.children[1].param_groups for p in g["params"]}
        scales = [p for name, p in left.named_parameters() if name.endswith("layer_scale")]
        self.assertTrue(scales)
        self.assertTrue(all(id(p) in adam_ids and id(p) not in matrix_ids for p in scales))
        self.assertTrue(all(id(p) in adam_ids for p in left.classifier.parameters()))
        with self.assertRaises(ValueError):
            make_model({**config, "mode": "frozen"}, classes=6)

    def test_annotation_audit_rejects_subsets_and_missing_annotations(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "train_mini.json"
            content = {"images": [{"id": 0, "file_name": "train_mini/a/1.jpg"}], "annotations": [{"image_id": 0, "category_id": 0}], "categories": [{"id": 0}]}
            path.write_text(json.dumps(content))
            with self.assertRaises(ValueError):
                read_split(folder, "train_mini")
            content["annotations"] = []
            path.write_text(json.dumps(content))
            with self.assertRaises(ValueError):
                read_split(folder, "train_mini", expected_classes=1, expected_per_class=1)

    def test_epoch_boundary_resume_matches_continuous_training(self):
        config = dict(id="unit", architecture="mlp", size=4, pretrained=False, mode="full",
                      seed=42, classes=2, optimizer="adam", lr=0.001, backbone_lr=0.001,
                      weight_decay=0., dropout=0., augmentation=True, epochs=3, patience=10,
                      min_delta=0., batch_size=2, accumulation=1, workers=0)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            entries = []
            for i in range(6):
                name = f"image_{i}.jpg"
                Image.new("RGB", (8, 8), (i * 30, 80, 180)).save(root / name)
                entries.append((name, i % 2))
            for name in ["train_mini.json", "val.json"]:
                (root / name).write_text("{}")
            def read_fake(_root, split, *_args):
                return entries if split == "train_mini" else entries[:4], {}, {0: 0, 1: 1}, {}
            with patch("src.full_experiment.read_split", side_effect=read_fake):
                with patch.dict(os.environ, {"NATURALIST_SEGMENT_SECONDS": "1000000000"}):
                    continuous = run_experiment(config, root, root / "continuous", require_cuda=False)
                with patch.dict(os.environ, {"NATURALIST_SEGMENT_SECONDS": "0"}):
                    for epoch in (1, 2):
                        partial = run_experiment(config, root, root / "segmented", require_cuda=False)
                        self.assertEqual(partial["state"], "NEEDS_RESUME")
                        self.assertEqual(partial["next_epoch"], epoch + 1)
                        self.assertFalse((root / "segmented/unit/result.json").exists())
                    resumed = run_experiment(config, root, root / "segmented", require_cuda=False)
            self.assertEqual(continuous["metrics"], resumed["metrics"])
            self.assertEqual(continuous["initial_weights_sha256"], resumed["initial_weights_sha256"])
            a = torch.load(root / "continuous/unit/last.pt", weights_only=False)
            b = torch.load(root / "segmented/unit/last.pt", weights_only=False)
            for key in a["model"]:
                torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)

    def test_requeue_targets_only_own_array_task(self):
        with patch.dict(os.environ, {"SLURM_JOB_ID": "12346", "SLURM_ARRAY_JOB_ID": "12345", "SLURM_ARRAY_TASK_ID": "3"}, clear=True):
            with patch("scripts.execute_project_notebook.subprocess.run") as command:
                requeue_current_allocation()
                command.assert_called_once_with(["scontrol", "requeue", "12345_3"], check=True)
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                requeue_current_allocation()


if __name__ == "__main__":
    unittest.main()
