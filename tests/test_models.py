"""
Tests for model architectures: MLP, Custom CNN, and Transfer Learning wrappers.
"""

import unittest
import torch
from src.models.mlp import MLPBaseline
from src.models.cnn_custom import MiniINatCNN
from src.models.transfer import TransferCNN
from src.models.factory import build_model, count_parameters


class TestModels(unittest.TestCase):

    def test_mlp_forward_and_params(self):
        batch = torch.randn(4, 3, 32, 32)
        model = MLPBaseline(input_shape=(3, 32, 32), hidden_dims=[128, 64], num_classes=10)
        out = model(batch)
        self.assertEqual(out.shape, (4, 10))

        total, trainable, total_m, _ = count_parameters(model)
        self.assertGreater(total, 0)
        self.assertEqual(total, trainable)

    def test_mini_inat_cnn_variations(self):
        batch = torch.randn(4, 3, 64, 64)
        # Without BN
        m1 = MiniINatCNN(num_classes=10, use_batchnorm=False, dropout_rate=0.0)
        out1 = m1(batch)
        self.assertEqual(out1.shape, (4, 10))

        # With BN and Dropout
        m2 = MiniINatCNN(num_classes=10, use_batchnorm=True, dropout_rate=0.3)
        out2 = m2(batch)
        self.assertEqual(out2.shape, (4, 10))

    def test_transfer_cnn_feature_extraction(self):
        # Using resnet18 without downloading weights for quick offline test
        model = TransferCNN(backbone_name="resnet18", num_classes=10, mode="feature_extraction", pretrained=False)
        batch = torch.randn(2, 3, 128, 128)
        out = model(batch)
        self.assertEqual(out.shape, (2, 10))

        # Check that backbone is frozen
        head_ids = {id(p) for p in model.head.parameters()}
        backbone_grad = [p.requires_grad for p in model.backbone.parameters() if id(p) not in head_ids]
        self.assertTrue(all(not g for g in backbone_grad))

        # Head should be trainable
        head_grad = [p.requires_grad for p in model.head.parameters()]
        self.assertTrue(all(g for g in head_grad))

    def test_factory(self):
        m_mlp = build_model("mlp", num_classes=5, input_shape=(3, 16, 16), hidden_dims=[32])
        self.assertIsInstance(m_mlp, MLPBaseline)

        m_cnn = build_model("cnn_custom", num_classes=5)
        self.assertIsInstance(m_cnn, MiniINatCNN)


if __name__ == "__main__":
    unittest.main()
