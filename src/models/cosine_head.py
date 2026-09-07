"""
Cosine Normalized Classifier (ArcFace / Cosine Head).
Normalizes both feature embeddings and weight vectors onto a unit hypersphere,
preventing class norm imbalances and enforcing angular margin separation.
Widely used in Fine-Grained Visual Categorization (FGVC).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineClassifier(nn.Module):
    """
    Hyperspherical Cosine Classifier:
    logits = scale * (x / ||x||) @ (W / ||W||).T

    Args:
        in_features: Feature dimension from backbone (e.g. 768 for ConvNeXt-Tiny).
        num_classes: Number of biological species (50).
        scale: Inverse temperature scaling factor (default: 16.0).
    """

    def __init__(self, in_features: int, num_classes: int, scale: float = 16.0):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.scale = scale

        # Weight matrix: (num_classes, in_features)
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize input features across channels
        x_norm = F.normalize(x, p=2, dim=1)
        # Normalize class prototype weight vectors
        w_norm = F.normalize(self.weight, p=2, dim=1)
        # Cosine similarity matrix scaled by temperature
        cosine = F.linear(x_norm, w_norm)
        return self.scale * cosine
