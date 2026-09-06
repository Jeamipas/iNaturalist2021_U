"""
Tests for dataset, transforms, samplers, and dataloaders.
"""

import unittest
import torch
from src.data.dataset import SyntheticINatDataset
from src.data.transforms import get_transforms
from src.data.sampler import create_strategic_subset, create_long_tail_subset, get_class_balanced_weights
from src.data.dataloader import build_dataloaders


class TestDataPipeline(unittest.TestCase):

    def setUp(self):
        self.num_classes = 10
        self.num_samples = 100
        self.dataset = SyntheticINatDataset(
            num_samples=self.num_samples,
            num_classes=self.num_classes,
            img_size=64,
            seed=42
        )

    def test_synthetic_dataset_len_and_item(self):
        self.assertEqual(len(self.dataset), self.num_samples)
        img, label = self.dataset[0]
        self.assertEqual(img.shape, (3, 64, 64))
        self.assertTrue(0 <= label < self.num_classes)

    def test_transforms_shapes(self):
        tf_train = get_transforms(split="train", img_size=128, aug_mode="standard")
        tf_val = get_transforms(split="val", img_size=128)
        self.assertIsNotNone(tf_train)
        self.assertIsNotNone(tf_val)

    def test_strategic_subset(self):
        indices, selected_classes, label_map = create_strategic_subset(
            self.dataset, n_classes=5, per_class=5, seed=42
        )
        self.assertEqual(len(selected_classes), 5)
        self.assertEqual(len(label_map), 5)
        self.assertTrue(len(indices) <= 25)

    def test_long_tail_subset(self):
        selected_classes = list(range(6))
        indices, label_map, tiers = create_long_tail_subset(
            self.dataset,
            selected_classes=selected_classes,
            many_shot_count=10,
            med_shot_count=5,
            few_shot_count=2,
            many_ratio=0.33,
            med_ratio=0.33,
            seed=42
        )
        self.assertIn("frequent", tiers)
        self.assertIn("medium", tiers)
        self.assertIn("minority", tiers)
        self.assertGreater(len(indices), 0)

    def test_class_balanced_weights(self):
        targets = [0, 0, 0, 0, 1, 1, 2]
        weights, sampler = get_class_balanced_weights(targets, num_classes=3)
        self.assertEqual(len(weights), 3)
        # Class 2 is rarest, so its weight should be higher than Class 0
        self.assertGreater(weights[2].item(), weights[0].item())

    def test_dataloader_builder(self):
        val_dataset = SyntheticINatDataset(num_samples=20, num_classes=self.num_classes, img_size=64)
        train_loader, val_loader = build_dataloaders(self.dataset, val_dataset, batch_size=8, num_workers=0)
        images, labels = next(iter(train_loader))
        self.assertEqual(images.shape[0], 8)
        self.assertEqual(labels.shape[0], 8)


if __name__ == "__main__":
    unittest.main()
