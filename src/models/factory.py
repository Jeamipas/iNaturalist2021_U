"""
Unified model factory and parameter counting utilities.
"""

from typing import Dict, Tuple
import torch.nn as nn
from .mlp import MLPBaseline
from .cnn_custom import MiniINatCNN
from .transfer import TransferCNN


def count_parameters(model: nn.Module) -> Tuple[int, int, float, float]:
    """
    Counts total and trainable parameters of a model.

    Returns:
        (total_params, trainable_params, total_params_M, trainable_params_M)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable, total / 1e6, trainable / 1e6


def build_model(
    model_type: str,
    num_classes: int = 50,
    **kwargs
) -> nn.Module:
    """
    Factory function for all models in the project.

    Args:
        model_type: One of 'mlp', 'cnn_custom', 'resnet18', 'resnet50',
                    'convnext_tiny', 'efficientnet_b0'.
        num_classes: Number of output categories.
        **kwargs: Model-specific arguments (e.g., mode, dropout_rate, use_batchnorm).

    Returns:
        Instantiated nn.Module.
    """
    mtype = model_type.lower()

    if mtype == "mlp":
        return MLPBaseline(num_classes=num_classes, **kwargs)

    elif mtype in ["cnn_custom", "mini_inat_cnn"]:
        return MiniINatCNN(num_classes=num_classes, **kwargs)

    elif mtype in ["resnet18", "resnet50", "convnext_tiny", "efficientnet_b0", "swin_t", "swin_tiny", "swin"]:
        mode = kwargs.get("mode", "feature_extraction")
        pretrained = kwargs.get("pretrained", True)
        dropout_rate = kwargs.get("dropout_rate", 0.2)
        use_cosine_head = kwargs.get("use_cosine_head", False)
        return TransferCNN(
            backbone_name=mtype,
            num_classes=num_classes,
            mode=mode,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
            use_cosine_head=use_cosine_head
        )

    else:
        raise ValueError(f"Unknown model_type: '{model_type}'. Choose from 'mlp', 'cnn_custom', 'resnet18', 'resnet50', 'convnext_tiny', 'efficientnet_b0'.")


create_model = build_model

