"""
Transfer Learning and Fine-Tuning architectures (Part 6 of rubric).
Supports Feature Extraction, Partial Fine-Tuning, and Full Fine-Tuning across:
- ResNet (ResNet-18, ResNet-50)
- ConvNeXt (ConvNeXt-Tiny)
- EfficientNet (EfficientNet-B0)
"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn
from torchvision import models


class TransferCNN(nn.Module):
    """
    Wrapper for modern torchvision architectures supporting:
    - Experimento A: Feature Extraction (Backbone frozen, only classifier head trained)
    - Experimento B: Partial Fine-Tuning (Top convolutional stage unfrozen)
    - Experimento C: Full Fine-Tuning (All parameters trainable with differential lr)
    """

    def __init__(
        self,
        backbone_name: str = "resnet50",
        num_classes: int = 50,
        mode: str = "feature_extraction",
        pretrained: bool = True,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        self.num_classes = num_classes
        self.mode = mode.lower()

        # Build backbone and replace head
        self.backbone, self.in_features, self.head_attr = self._create_backbone(
            self.backbone_name, pretrained
        )

        # Build custom classification head with dropout
        self.head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.in_features, num_classes)
        )

        # Replace classification head on backbone
        setattr(self.backbone, self.head_attr, self.head)

        # Apply freezing mode
        self.set_fine_tuning_mode(self.mode)

    def _create_backbone(self, name: str, pretrained: bool):
        if name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            m = models.resnet18(weights=weights)
            return m, m.fc.in_features, "fc"

        elif name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            m = models.resnet50(weights=weights)
            return m, m.fc.in_features, "fc"

        elif name == "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            m = models.convnext_tiny(weights=weights)
            # In torchvision convnext, classifier is Sequential: LayerNorm2d, Flatten, Linear
            in_features = m.classifier[2].in_features
            # Replace classifier Sequential
            m.classifier = nn.Identity()
            return m, in_features, "classifier"

        elif name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            m = models.efficientnet_b0(weights=weights)
            in_features = m.classifier[1].in_features
            m.classifier = nn.Identity()
            return m, in_features, "classifier"

        else:
            raise ValueError(f"Unsupported backbone: {name}. Choose from resnet18, resnet50, convnext_tiny, efficientnet_b0.")

    def set_fine_tuning_mode(self, mode: str) -> None:
        """
        Configures parameter grad requirements for Feature Extraction, Partial FT, or Full FT.
        """
        self.mode = mode.lower()

        if self.mode == "full_fine_tuning":
            for param in self.parameters():
                param.requires_grad = True

        elif self.mode == "feature_extraction":
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Only head parameters are trainable
            for param in self.head.parameters():
                param.requires_grad = True

        elif self.mode == "partial_fine_tuning":
            # Freeze everything first
            for param in self.backbone.parameters():
                param.requires_grad = False

            # Unfreeze top stage
            if "resnet" in self.backbone_name:
                for param in self.backbone.layer4.parameters():
                    param.requires_grad = True
            elif "convnext" in self.backbone_name:
                # Unfreeze last stage (stages[3])
                for param in self.backbone.features[7].parameters():
                    param.requires_grad = True
            elif "efficientnet" in self.backbone_name:
                # Unfreeze last stage (features[7] and features[8])
                for param in self.backbone.features[7:].parameters():
                    param.requires_grad = True

            # Head is always trainable
            for param in self.head.parameters():
                param.requires_grad = True

        else:
            raise ValueError(f"Unknown mode: {mode}. Choose feature_extraction, partial_fine_tuning, or full_fine_tuning.")

    def get_parameter_groups(
        self,
        backbone_lr: float = 1e-4,
        head_lr: float = 1e-3,
        weight_decay: float = 1e-4
    ) -> List[Dict]:
        """
        Creates differential learning rate parameter groups:
        lower lr for pretrained backbone representations, higher lr for new head.
        """
        head_params = list(self.head.parameters())
        head_param_ids = set(id(p) for p in head_params)

        backbone_params = [
            p for p in self.parameters()
            if id(p) not in head_param_ids and p.requires_grad
        ]

        param_groups = [
            {"params": head_params, "lr": head_lr, "weight_decay": weight_decay}
        ]
        if backbone_params:
            param_groups.append(
                {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay}
            )

        return param_groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.head_attr == "fc":
            # For ResNet, the head was assigned directly to self.backbone.fc
            return self.backbone(x)
        else:
            # For ConvNeXt / EfficientNet where head replaced classifier
            features = self.backbone(x)
            return self.head(features)
