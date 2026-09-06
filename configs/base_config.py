"""
Base configuration dataclass for all experiments.
Ultra-configurable, reproducible, and serialized.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple


@dataclass
class ExperimentConfig:
    # Experiment metadata
    exp_id: str = "E0_baseline"
    description: str = "Base experiment configuration"

    # Dataset paths and subsetting
    data_root: str = "../recursos/inat2021_sample"
    train_json: str = "../recursos/inat2021_sample/train_mini.json"
    val_json: str = "../recursos/inat2021_sample/val.json"
    train_images: str = "../recursos/inat2021_sample/train_mini"
    val_images: str = "../recursos/inat2021_sample/val"


    # Strategic sample parameters (for 4GB VRAM safety)
    n_classes: int = 50
    train_per_class: int = 35
    val_per_class: int = 10
    img_size: int = 224
    aug_mode: str = "standard"  # 'none', 'standard', 'full'

    # Model architecture
    model_type: str = "resnet50"  # 'mlp', 'cnn_custom', 'resnet18', 'resnet50', 'convnext_tiny', 'efficientnet_b0'
    mode: str = "feature_extraction"  # 'feature_extraction', 'partial_fine_tuning', 'full_fine_tuning'
    pretrained: bool = True
    dropout_rate: float = 0.2
    use_batchnorm: bool = True

    # Training hyperparameters
    batch_size: int = 32
    epochs: int = 10
    optimizer: str = "adamw"  # 'sgd', 'sgd_momentum', 'adam', 'adamw'
    lr: float = 1e-3
    backbone_lr: float = 1e-4
    weight_decay: float = 1e-4
    momentum: float = 0.9
    scheduler: str = "cosine"  # 'cosine', 'step', 'none'
    loss_type: str = "cross_entropy"  # 'cross_entropy', 'weighted_cross_entropy', 'focal_loss'
    focal_gamma: float = 2.0
    label_smoothing: float = 0.0

    # Reproducibility & Performance
    seed: int = 42
    use_amp: bool = True
    num_workers: int = 2

    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_metric: str = "val_macro_f1"

    # Long-tail challenge configuration
    is_long_tail: bool = False
    many_shot_count: int = 35
    med_shot_count: int = 15
    few_shot_count: int = 3

    # Output directory
    output_dir: str = "./outputs"

    def to_dict(self) -> Dict:
        return asdict(self)
