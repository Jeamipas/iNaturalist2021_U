"""
Transfer Learning and Fine-Tuning architectures (Part 6 of rubric & SOTA Elite).
Supports Feature Extraction, Partial Fine-Tuning, Deep Partial FT, and Full FT across:
- ConvNeXt (ConvNeXt-Tiny)
- Swin Transformer (Swin-T)
- ResNet (ResNet-18, ResNet-50)
- EfficientNet (EfficientNet-B0)
Also supports Hyperspherical Cosine Classifiers (ArcFace/Cosine Head) and Layer-wise LR Decay (LLRD).
"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn
from torchvision import models
from .cosine_head import CosineClassifier


class TransferCNN(nn.Module):
    """
    Wrapper for modern torchvision backbones and vision transformers supporting:
    - Feature Extraction (Backbone frozen, only classifier head trained)
    - Partial Fine-Tuning (Top stage unfrozen)
    - Deep Partial Fine-Tuning (Top 2 stages unfrozen)
    - Full Fine-Tuning with optional Layer-wise Learning Rate Decay (LLRD)
    - Standard Linear Head vs Hyperspherical Cosine Head
    """

    def __init__(
        self,
        backbone_name: str = "convnext_tiny",
        num_classes: int = 50,
        mode: str = "partial_fine_tuning",
        pretrained: bool = True,
        dropout_rate: float = 0.2,
        use_cosine_head: bool = False
    ):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        self.num_classes = num_classes
        self.mode = mode.lower()
        self.use_cosine_head = use_cosine_head

        # Build backbone and configure classification head
        self.backbone, self.in_features, self.head_attr, self.head = self._create_backbone(
            self.backbone_name, pretrained, num_classes, dropout_rate, use_cosine_head
        )

        # Apply freezing mode
        self.set_fine_tuning_mode(self.mode)

    def _create_backbone(
        self,
        name: str,
        pretrained: bool,
        num_classes: int,
        dropout_rate: float = 0.2,
        use_cosine_head: bool = False
    ):
        if name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            m = models.resnet18(weights=weights)
            in_features = m.fc.in_features
            head = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                CosineClassifier(in_features, num_classes) if use_cosine_head else nn.Linear(in_features, num_classes)
            )
            m.fc = head
            return m, in_features, "fc", head

        elif name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            m = models.resnet50(weights=weights)
            in_features = m.fc.in_features
            head = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                CosineClassifier(in_features, num_classes) if use_cosine_head else nn.Linear(in_features, num_classes)
            )
            m.fc = head
            return m, in_features, "fc", head

        elif name in ["convnext_tiny", "convnext"]:
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            m = models.convnext_tiny(weights=weights)
            in_features = m.classifier[2].in_features
            head = nn.Sequential(
                m.classifier[0],  # LayerNorm2d
                m.classifier[1],  # Flatten
                nn.Dropout(p=dropout_rate),
                CosineClassifier(in_features, num_classes) if use_cosine_head else nn.Linear(in_features, num_classes)
            )
            m.classifier = head
            return m, in_features, "classifier", head

        elif name in ["swin_t", "swin_tiny", "swin"]:
            weights = models.Swin_T_Weights.DEFAULT if pretrained else None
            m = models.swin_t(weights=weights)
            in_features = m.head.in_features
            head = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                CosineClassifier(in_features, num_classes) if use_cosine_head else nn.Linear(in_features, num_classes)
            )
            m.head = head
            return m, in_features, "head", head

        elif name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            m = models.efficientnet_b0(weights=weights)
            in_features = m.classifier[1].in_features
            head = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                CosineClassifier(in_features, num_classes) if use_cosine_head else nn.Linear(in_features, num_classes)
            )
            m.classifier = head
            return m, in_features, "classifier", head

        else:
            raise ValueError(f"Unsupported backbone: {name}. Choose from resnet18, resnet50, convnext_tiny, swin_t, efficientnet_b0.")

    def set_fine_tuning_mode(self, mode: str) -> None:
        """
        Configures parameter grad requirements for:
        - feature_extraction
        - partial_fine_tuning (Stage 4 only)
        - deep_partial_fine_tuning (Stages 3 & 4)
        - full_fine_tuning
        """
        self.mode = mode.lower()

        if self.mode == "full_fine_tuning":
            for param in self.parameters():
                param.requires_grad = True

        elif self.mode == "feature_extraction":
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.head.parameters():
                param.requires_grad = True

        elif self.mode == "partial_fine_tuning":
            # Freeze entire backbone first
            for param in self.backbone.parameters():
                param.requires_grad = False

            # Unfreeze top stage
            if "resnet" in self.backbone_name:
                for param in self.backbone.layer4.parameters():
                    param.requires_grad = True
            elif "convnext" in self.backbone_name:
                # Unfreeze last stage (features[7])
                for param in self.backbone.features[7].parameters():
                    param.requires_grad = True
            elif "swin" in self.backbone_name:
                # Unfreeze last Swin block (features[7]) and final norm
                for param in self.backbone.features[7].parameters():
                    param.requires_grad = True
                if hasattr(self.backbone, "norm"):
                    for param in self.backbone.norm.parameters():
                        param.requires_grad = True
            elif "efficientnet" in self.backbone_name:
                for param in self.backbone.features[7:].parameters():
                    param.requires_grad = True

            # Head is always trainable
            for param in self.head.parameters():
                param.requires_grad = True

        elif self.mode == "deep_partial_fine_tuning":
            # Freeze entire backbone first
            for param in self.backbone.parameters():
                param.requires_grad = False

            # Unfreeze top two stages (Stages 3 and 4)
            if "resnet" in self.backbone_name:
                for param in self.backbone.layer3.parameters():
                    param.requires_grad = True
                for param in self.backbone.layer4.parameters():
                    param.requires_grad = True
            elif "convnext" in self.backbone_name:
                # Unfreeze stage 3 (features[5]), downsample (features[6]), and stage 4 (features[7])
                for param in self.backbone.features[5:].parameters():
                    param.requires_grad = True
            elif "swin" in self.backbone_name:
                for param in self.backbone.features[5:].parameters():
                    param.requires_grad = True
                if hasattr(self.backbone, "norm"):
                    for param in self.backbone.norm.parameters():
                        param.requires_grad = True

            # Head is always trainable
            for param in self.head.parameters():
                param.requires_grad = True

        else:
            raise ValueError(f"Unknown mode: {mode}. Choose feature_extraction, partial_fine_tuning, deep_partial_fine_tuning, or full_fine_tuning.")

    def get_parameter_groups(
        self,
        backbone_lr: float = 1e-4,
        head_lr: float = 1e-3,
        weight_decay: float = 1e-4
    ) -> List[Dict]:
        """Differential learning rate parameter groups."""
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

    def get_layerwise_lr_groups(
        self,
        head_lr: float = 1e-3,
        decay_rate: float = 0.65,
        weight_decay: float = 1e-4
    ) -> List[Dict]:
        """
        Layer-wise Learning Rate Decay (LLRD).
        Earlier stages receive exponentially decayed learning rates to preserve universal features.
        """
        param_groups = [
            {"params": list(self.head.parameters()), "lr": head_lr, "weight_decay": weight_decay}
        ]

        if "convnext" in self.backbone_name or "swin" in self.backbone_name:
            # 4 main stages in features: 0-1 (stage 1), 2-3 (stage 2), 4-5 (stage 3), 6-7 (stage 4)
            stages = [
                (self.backbone.features[6:], 1),  # Stage 4
                (self.backbone.features[4:6], 2), # Stage 3
                (self.backbone.features[2:4], 3), # Stage 2
                (self.backbone.features[0:2], 4), # Stage 1 + Stem
            ]
            for stage_module, depth in stages:
                stage_params = [p for p in stage_module.parameters() if p.requires_grad]
                if stage_params:
                    stage_lr = head_lr * (decay_rate ** depth)
                    param_groups.append({"params": stage_params, "lr": stage_lr, "weight_decay": weight_decay})
        else:
            return self.get_parameter_groups(backbone_lr=head_lr * decay_rate, head_lr=head_lr, weight_decay=weight_decay)

        return param_groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
