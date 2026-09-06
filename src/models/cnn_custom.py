"""
Custom Convolutional Neural Network (MiniINatCNN).
Enhanced with toggles for Batch Normalization and Dropout for the Ablation Study.
"""

import torch
import torch.nn as nn


class MiniINatCNN(nn.Module):
    """
    Custom 3-block Convolutional Neural Network.
    Supports modular toggles for BatchNorm and Dropout to enable
    the systematic ablation experiments requested in Section 9.4.
    """

    def __init__(
        self,
        num_classes: int = 50,
        use_batchnorm: bool = False,
        dropout_rate: float = 0.0
    ):
        super().__init__()
        self.use_batchnorm = use_batchnorm
        self.dropout_rate = dropout_rate

        def make_block(in_c: int, out_c: int) -> nn.Sequential:
            block_layers = [nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)]
            if use_batchnorm:
                block_layers.append(nn.BatchNorm2d(out_c))
            block_layers.append(nn.ReLU(inplace=True))
            block_layers.append(nn.MaxPool2d(2))
            if dropout_rate > 0.0:
                block_layers.append(nn.Dropout2d(dropout_rate / 2.0))
            return nn.Sequential(*block_layers)

        self.block1 = make_block(3, 32)
        self.block2 = make_block(32, 64)
        self.block3 = make_block(64, 128)

        # Adaptive pool guarantees fixed feature vector regardless of input resolution
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        classifier_layers = [
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
        ]
        if dropout_rate > 0.0:
            classifier_layers.append(nn.Dropout(dropout_rate))

        classifier_layers.append(nn.Linear(256, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x
