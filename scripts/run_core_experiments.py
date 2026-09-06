"""
Master runner script: Executes the comprehensive core benchmark suite (35+ experiments).
Adheres strictly to the 20-point rubric of Dra. Aurea Soriano-Vargas.
Optimized for RTX 5060 / 4GB-8GB VRAM with AMP fp16, channels_last, and RAM caching.
Retains the Top 10 champion checkpoints and logs live to TensorBoard.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.utils.seed import seed_everything
from src.data.dataset import INatDataset
from src.data.sampler import create_long_tail_subset
from src.data.transforms import get_transforms, get_batch_augmentations
from src.data.dataloader import build_dataloaders
from src.models.factory import build_model, count_parameters
from src.training.losses import build_criterion
from src.training.optimizers import build_optimizer
from src.training.early_stopping import EarlyStopping
from src.training.trainer import Trainer
from src.utils.tracking import ExperimentTracker
from src.utils.checkpoints import TopKCheckpointManager
from src.utils.visualization import plot_multiexperiment_4grid


def main():
    parser = argparse.ArgumentParser(description="Core experiment suite for iNaturalist 2021.")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to inat2021_sample dataset")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per experiment (default: 5)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--img_size", type=int, default=128, help="Image resolution (default: 128)")
    parser.add_argument("--limit_exps", type=int, default=None, help="Limit number of experiments for testing")
    args = parser.parse_args()

    print("=" * 80)
    print("  iNaturalist 2021: SUITE EXPERIMENTAL CORE COMPLETA (RÚBRICA OFICIAL + SOTA)")
    print("=" * 80)

    SEED = 42
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Dispositivo de computo: {device}")
    if device.type == "cuda":
        print(f"[*] GPU detectada: {torch.cuda.get_device_name(0)}")

    # 1. Resolver ruta del dataset preparado de 50 clases
    if args.data_dir:
        data_dir = Path(args.data_dir).resolve()
    else:
        sample_50 = ROOT_DIR.parent / "recursos" / "inat2021_sample"
        if sample_50.exists() and (sample_50 / "train_mini.json").exists():
            data_dir = sample_50
        else:
            data_dir = ROOT_DIR.parent / "recursos" / "sample_inat"

    train_json = data_dir / "train_mini.json"
    val_json = data_dir / "val.json"
    train_img_dir = data_dir / "train_mini"
    val_img_dir = data_dir / "val"

    if not (train_json.exists() and train_img_dir.exists()):
        raise FileNotFoundError(f"Dataset no encontrado en {data_dir}. Ejecuta primero scripts/prepare_dataset.py")

    print(f"[*] Directorio de datos activo: {data_dir.name} ({data_dir})")

    # 2. Datasets base
    IMG_SIZE = args.img_size
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size

    tf_train_std = get_transforms(split="train", img_size=IMG_SIZE, aug_mode="standard")
    tf_train_none = get_transforms(split="train", img_size=IMG_SIZE, aug_mode="none")
    tf_train_full = get_transforms(split="train", img_size=IMG_SIZE, aug_mode="full")
    tf_val = get_transforms(split="val", img_size=IMG_SIZE, aug_mode="none")

    print("[*] Precargando datasets en memoria RAM para máxima velocidad...")
    train_base = INatDataset(train_json, train_img_dir, transform=tf_train_std, cache_in_ram=True)
    val_base = INatDataset(val_json, val_img_dir, transform=tf_val, category_to_label=train_base.category_to_label, cache_in_ram=True)

    NUM_CLASSES = len(train_base.category_to_label)
    print(f"[*] Dataset cargado: {len(train_base)} train | {len(val_base)} val | {NUM_CLASSES} clases biológicas.")

    # 3. Mapeo de taxonomía: Label -> Kingdom (Animalia vs Plantae)
    label_to_kingdom = {}
    for cat_id, lbl in train_base.category_to_label.items():
        if cat_id in train_base.taxonomy:
            label_to_kingdom[lbl] = train_base.taxonomy[cat_id].get("kingdom", "Desconocido")
        else:
            label_to_kingdom[lbl] = "Desconocido"

    # 4. Gestores de seguimiento y checkpoints
    tracker = ExperimentTracker(log_dir=str(ROOT_DIR / "logs"))
    ckpt_manager = TopKCheckpointManager(checkpoint_dir=str(ROOT_DIR / "checkpoints"), k=10)
    all_histories: Dict[str, Dict[str, List[float]]] = {}

    criterion_ce = build_criterion("cross_entropy")

    # Helper para ejecutar y registrar un experimento
    def run_exp(
        exp_id: str,
        model: nn.Module,
        opt: torch.optim.Optimizer,
        criterion: nn.Module,
        train_ds,
        val_ds,
        model_name: str,
        opt_name: str,
        reg_name: str,
        aug_name: str,
        tl_name: str,
        lt_name: str,
        batch_aug=None,
        class_tiers=None,
        use_amp: bool = True,
        grad_accum_steps: int = 1,
        custom_epochs: Optional[int] = None
    ):
        ep_count = custom_epochs if custom_epochs is not None else EPOCHS
        print(f"\n" + "=" * 78)
        print(f"  >>> [{exp_id}] {model_name} | Opt: {opt_name} | Reg: {reg_name}")
        print(f"      Aug: {aug_name} | TL: {tl_name} | LT: {lt_name} | AMP: {use_amp}")
        print("=" * 78)

        tb_dir = str(ROOT_DIR / "logs" / "tb_runs" / f"{exp_id}_{model_name.replace(' ', '_')}_{opt_name.split()[0]}")
        train_l, val_l = build_dataloaders(train_ds, val_ds, batch_size=BATCH_SIZE, num_workers=0, seed=SEED)

        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=opt,
            device=device,
            use_amp=use_amp,
            class_tiers=class_tiers,
            label_to_kingdom=label_to_kingdom,
            batch_augmenter=batch_aug,
            grad_accum_steps=grad_accum_steps,
            tb_log_dir=tb_dir
        )

        history, best_metrics, peak_vram, total_time = trainer.fit(train_l, val_l, epochs=ep_count, verbose=True)
        _, _, total_m, _ = count_parameters(model)

        # 1. Registrar en tracker
        record = tracker.log_experiment(
            exp_id=exp_id,
            model_name=model_name,
            optimizer=opt_name,
            regularization=reg_name,
            augmentation=aug_name,
            transfer_learning=tl_name,
            long_tail=lt_name,
            metrics=best_metrics,
            training_time_sec=total_time,
            peak_vram_mb=peak_vram,
            param_count_m=total_m
        )

        # 2. Registrar en Top-10 Checkpoint Manager
        rank = ckpt_manager.register_and_save(
            model=model,
            exp_id=exp_id,
            model_name=model_name,
            metrics=best_metrics,
            hyperparams={
                "optimizer": opt_name,
                "batch_size": BATCH_SIZE,
                "epochs": ep_count,
                "img_size": IMG_SIZE,
                "use_amp": use_amp,
                "regularization": reg_name,
                "augmentation": aug_name
            }
        )

        # Guardar historial para visualizaciones
        all_histories[f"[{exp_id}] {model_name} | {opt_name}"] = history
        return record

    exp_counter = 0

    def should_continue():
        nonlocal exp_counter
        exp_counter += 1
        if args.limit_exps is not None and exp_counter > args.limit_exps:
            return False
        return True

    # =========================================================================
    # BLOQUE 1: BASELINE CON MULTILAYER PERCEPTRON (MLP) (Sección 6)
    # =========================================================================
    print("\n" + "#" * 80)
    print("  BLOQUE 1: BASELINE CON PERCEPTRÓN MULTICAPA (MLP)")
    print("#" * 80)

    # B01: MLP 32x32 con Adam
    if should_continue():
        train_mlp32 = INatDataset(train_json, train_img_dir, transform=get_transforms("train", img_size=32, aug_mode="none"), cache_in_ram=True)
        val_mlp32 = INatDataset(val_json, val_img_dir, transform=get_transforms("val", img_size=32, aug_mode="none"), category_to_label=train_mlp32.category_to_label, cache_in_ram=True)
        m_b01 = build_model("mlp", num_classes=NUM_CLASSES, input_shape=(3, 32, 32), hidden_dims=[512, 256])
        opt_b01 = build_optimizer(m_b01, opt_type="adam", lr=1e-3)
        run_exp("B01", m_b01, opt_b01, criterion_ce, train_mlp32, val_mlp32, "MLP (32x32)", "Adam (1e-3)", "None", "None", "No", "No")

    # B02: MLP 64x64 con Adam (Estudio de escalado de entrada y explosión paramétrica)
    if should_continue():
        train_mlp64 = INatDataset(train_json, train_img_dir, transform=get_transforms("train", img_size=64, aug_mode="none"), cache_in_ram=True)
        val_mlp64 = INatDataset(val_json, val_img_dir, transform=get_transforms("val", img_size=64, aug_mode="none"), category_to_label=train_mlp64.category_to_label, cache_in_ram=True)
        m_b02 = build_model("mlp", num_classes=NUM_CLASSES, input_shape=(3, 64, 64), hidden_dims=[1024, 512])
        opt_b02 = build_optimizer(m_b02, opt_type="adam", lr=1e-3)
        run_exp("B02", m_b02, opt_b02, criterion_ce, train_mlp64, val_mlp64, "MLP (64x64)", "Adam (1e-3)", "None", "None", "No", "No")

    # B03: MLP 32x32 con SGD+Momentum
    if should_continue():
        m_b03 = build_model("mlp", num_classes=NUM_CLASSES, input_shape=(3, 32, 32), hidden_dims=[512, 256])
        opt_b03 = build_optimizer(m_b03, opt_type="sgd_momentum", lr=1e-2, weight_decay=1e-4)
        run_exp("B03", m_b03, opt_b03, criterion_ce, train_mlp32, val_mlp32, "MLP (32x32)", "SGD+Momentum (1e-2)", "Weight Decay", "None", "No", "No")

    # =========================================================================
    # BLOQUE 2: ARQUITECTURAS CONVOLUCIONALES MODERNAS DESDE CERO (Sección 7)
    # =========================================================================
    print("\n" + "#" * 80)
    print("  BLOQUE 2: COMPARACIÓN DE ARQUITECTURAS CNN DESDE CERO (SCRATCH)")
    print("#" * 80)

    # A01: MiniINatCNN (Convolucional personalizada con BN y Dropout)
    if should_continue():
        m_a01 = build_model("cnn_custom", num_classes=NUM_CLASSES, use_batchnorm=True, dropout_rate=0.2)
        opt_a01 = build_optimizer(m_a01, opt_type="adamw", lr=1e-3)
        run_exp("A01", m_a01, opt_a01, criterion_ce, train_base, val_base, "MiniINatCNN (Scratch)", "AdamW (1e-3)", "BN+Dropout", "Standard", "No", "No")

    # A02: ResNet-18 desde cero
    if should_continue():
        m_a02 = build_model("resnet18", num_classes=NUM_CLASSES, pretrained=False)
        opt_a02 = build_optimizer(m_a02, opt_type="adamw", lr=1e-3)
        run_exp("A02", m_a02, opt_a02, criterion_ce, train_base, val_base, "ResNet-18 (Scratch)", "AdamW (1e-3)", "BN", "Standard", "No", "No")

    # A03: ResNet-50 desde cero
    if should_continue():
        m_a03 = build_model("resnet50", num_classes=NUM_CLASSES, pretrained=False)
        opt_a03 = build_optimizer(m_a03, opt_type="adamw", lr=1e-3)
        run_exp("A03", m_a03, opt_a03, criterion_ce, train_base, val_base, "ResNet-50 (Scratch)", "AdamW (1e-3)", "BN", "Standard", "No", "No")

    # A04: ConvNeXt-Tiny desde cero (Arquitectura convolucional moderna con kernels 7x7)
    if should_continue():
        m_a04 = build_model("convnext_tiny", num_classes=NUM_CLASSES, pretrained=False)
        opt_a04 = build_optimizer(m_a04, opt_type="adamw", lr=1e-3)
        run_exp("A04", m_a04, opt_a04, criterion_ce, train_base, val_base, "ConvNeXt-Tiny (Scratch)", "AdamW (1e-3)", "LayerNorm+DropPath", "Standard", "No", "No")

    # =========================================================================
    # BLOQUE 3: ESTUDIO DE OPTIMIZADORES TRADICIONALES Y SOTA (Sección 8)
    # =========================================================================
    print("\n" + "#" * 80)
    print("  BLOQUE 3: COMPARATIVA CONTROLADA DE OPTIMIZADORES (ResNet-18)")
    print("#" * 80)

    # O01: SGD estándar
    if should_continue():
        m_o01 = build_model("resnet18", num_classes=NUM_CLASSES, pretrained=False)
        opt_o01 = build_optimizer(m_o01, opt_type="sgd", lr=1e-2)
        run_exp("O01", m_o01, opt_o01, criterion_ce, train_base, val_base, "ResNet-18", "SGD Vainilla (1e-2)", "BN", "Standard", "No", "No")

    # O02: SGD con Nesterov Momentum
    if should_continue():
        m_o02 = build_model("resnet18", num_classes=NUM_CLASSES, pretrained=False)
        opt_o02 = build_optimizer(m_o02, opt_type="sgd_momentum", lr=1e-2, weight_decay=1e-4)
        run_exp("O02", m_o02, opt_o02, criterion_ce, train_base, val_base, "ResNet-18", "SGD+Momentum (1e-2)", "BN+WD", "Standard", "No", "No")

    # O03: Adam clásico
    if should_continue():
        m_o03 = build_model("resnet18", num_classes=NUM_CLASSES, pretrained=False)
        opt_o03 = build_optimizer(m_o03, opt_type="adam", lr=1e-3)
        run_exp("O03", m_o03, opt_o03, criterion_ce, train_base, val_base, "ResNet-18", "Adam Clásico (1e-3)", "BN", "Standard", "No", "No")

    # O04: AdamW (Weight Decay desacoplado)
    if should_continue():
        m_o04 = build_model("resnet18", num_classes=NUM_CLASSES, pretrained=False)
        opt_o04 = build_optimizer(m_o04, opt_type="adamw", lr=1e-3, weight_decay=1e-4)
        run_exp("O04", m_o04, opt_o04, criterion_ce, train_base, val_base, "ResNet-18", "AdamW (1e-3)", "BN+WD", "Standard", "No", "No")

    # O05: Muon (Newton-Schulz Quintic Orthogonalization) - SOTA Kimi k3
    if should_continue():
        m_o05 = build_model("resnet18", num_classes=NUM_CLASSES, pretrained=False)
        opt_o05 = build_optimizer(m_o05, opt_type="muon", lr=0.02, weight_decay=1e-4)
        run_exp("O05", m_o05, opt_o05, criterion_ce, train_base, val_base, "ResNet-18", "Muon (lr=0.02)", "BN+Orthogonal", "Standard", "No", "No")

    # O06: Lion (Google Brain Sign-based Momentum)
    if should_continue():
        m_o06 = build_model("resnet18", num_classes=NUM_CLASSES, pretrained=False)
        opt_o06 = build_optimizer(m_o06, opt_type="lion", lr=1e-4, weight_decay=1e-4)
        run_exp("O06", m_o06, opt_o06, criterion_ce, train_base, val_base, "ResNet-18", "Lion (1e-4)", "BN+SignMomentum", "Standard", "No", "No")

    # =========================================================================
    # BLOQUE 4: REGULARIZACIÓN Y DATA AUGMENTATION (ESTUDIO DE ABLACIÓN) (Sección 9)
    # =========================================================================
    print("\n" + "#" * 80)
    print("  BLOQUE 4: ESTUDIO SISTEMÁTICO DE ABLACIÓN Y DATA AUGMENTATION")
    print("#" * 80)

    train_none_ds = INatDataset(train_json, train_img_dir, transform=tf_train_none, cache_in_ram=True)
    train_full_ds = INatDataset(train_json, train_img_dir, transform=tf_train_full, cache_in_ram=True)

    # R01: Sin regularización (Base)
    if should_continue():
        m_r01 = build_model("cnn_custom", num_classes=NUM_CLASSES, use_batchnorm=False, dropout_rate=0.0)
        opt_r01 = build_optimizer(m_r01, opt_type="adamw", lr=1e-3, weight_decay=0.0)
        run_exp("R01", m_r01, opt_r01, criterion_ce, train_none_ds, val_base, "MiniINatCNN", "AdamW", "Ninguna (Sin BN/Drop)", "None", "No", "No")

    # R02: + Batch Normalization
    if should_continue():
        m_r02 = build_model("cnn_custom", num_classes=NUM_CLASSES, use_batchnorm=True, dropout_rate=0.0)
        opt_r02 = build_optimizer(m_r02, opt_type="adamw", lr=1e-3, weight_decay=0.0)
        run_exp("R02", m_r02, opt_r02, criterion_ce, train_none_ds, val_base, "MiniINatCNN", "AdamW", "+ BatchNorm", "None", "No", "No")

    # R03: + Dropout (p=0.3)
    if should_continue():
        m_r03 = build_model("cnn_custom", num_classes=NUM_CLASSES, use_batchnorm=True, dropout_rate=0.3)
        opt_r03 = build_optimizer(m_r03, opt_type="adamw", lr=1e-3, weight_decay=0.0)
        run_exp("R03", m_r03, opt_r03, criterion_ce, train_none_ds, val_base, "MiniINatCNN", "AdamW", "+ BatchNorm + Dropout(0.3)", "None", "No", "No")

    # R04: + Standard Data Augmentation (Crop + Flip)
    if should_continue():
        m_r04 = build_model("cnn_custom", num_classes=NUM_CLASSES, use_batchnorm=True, dropout_rate=0.3)
        opt_r04 = build_optimizer(m_r04, opt_type="adamw", lr=1e-3, weight_decay=0.0)
        run_exp("R04", m_r04, opt_r04, criterion_ce, train_base, val_base, "MiniINatCNN", "AdamW", "BN + Dropout", "+ Standard Aug", "No", "No")

    # R05: + Full Data Augmentation (Jitter + Rotation + RandomErasing) + Weight Decay
    if should_continue():
        m_r05 = build_model("cnn_custom", num_classes=NUM_CLASSES, use_batchnorm=True, dropout_rate=0.3)
        opt_r05 = build_optimizer(m_r05, opt_type="adamw", lr=1e-3, weight_decay=1e-4)
        run_exp("R05", m_r05, opt_r05, criterion_ce, train_full_ds, val_base, "MiniINatCNN", "AdamW", "BN + Drop + Weight Decay", "+ Full Augment", "No", "No")

    # R06: + SOTA Batch Augmentation (CutMix & MixUp en GPU)
    if should_continue():
        m_r06 = build_model("cnn_custom", num_classes=NUM_CLASSES, use_batchnorm=True, dropout_rate=0.3)
        opt_r06 = build_optimizer(m_r06, opt_type="adamw", lr=1e-3, weight_decay=1e-4)
        batch_aug = get_batch_augmentations(num_classes=NUM_CLASSES, mode="cutmix_or_mixup", cutmix_alpha=1.0, mixup_alpha=0.2)
        run_exp("R06", m_r06, opt_r06, criterion_ce, train_base, val_base, "MiniINatCNN", "AdamW", "BN + Drop + WD", "+ CutMix / MixUp (GPU)", "No", "No", batch_aug=batch_aug)

    # =========================================================================
    # BLOQUE 5: TRANSFER LEARNING Y FINE-TUNING (Sección 10)
    # =========================================================================
    print("\n" + "#" * 80)
    print("  BLOQUE 5: TRANSFER LEARNING (FEATURE EXTRACTION VS PARTIAL VS FULL FT)")
    print("#" * 80)

    # T01: ResNet-18 - Feature Extraction (Backbone congelado)
    if should_continue():
        m_t01 = build_model("resnet18", num_classes=NUM_CLASSES, mode="feature_extraction", pretrained=True)
        opt_t01 = build_optimizer(m_t01, opt_type="adamw", lr=1e-3)
        run_exp("T01", m_t01, opt_t01, criterion_ce, train_base, val_base, "ResNet-18 (Pretrained)", "AdamW (1e-3)", "BN", "Standard", "Feature Extraction (Frozen)", "No")

    # T02: ResNet-18 - Partial Fine-Tuning (Descongelado layer4 + LR Diferencial)
    if should_continue():
        m_t02 = build_model("resnet18", num_classes=NUM_CLASSES, mode="partial_fine_tuning", pretrained=True)
        opt_t02 = build_optimizer(m_t02, opt_type="adamw", param_groups=m_t02.get_parameter_groups(backbone_lr=1e-4, head_lr=1e-3))
        run_exp("T02", m_t02, opt_t02, criterion_ce, train_base, val_base, "ResNet-18 (Pretrained)", "AdamW (Diff LR)", "BN", "Standard", "Partial FT (layer4)", "No")

    # T03: ResNet-18 - Full Fine-Tuning (Descongelado total + LR Diferencial)
    if should_continue():
        m_t03 = build_model("resnet18", num_classes=NUM_CLASSES, mode="full_fine_tuning", pretrained=True)
        opt_t03 = build_optimizer(m_t03, opt_type="adamw", param_groups=m_t03.get_parameter_groups(backbone_lr=1e-5, head_lr=1e-3))
        run_exp("T03", m_t03, opt_t03, criterion_ce, train_base, val_base, "ResNet-18 (Pretrained)", "AdamW (Diff LR)", "BN", "Standard", "Full Fine-Tuning", "No")

    # T04: ResNet-50 - Feature Extraction
    if should_continue():
        m_t04 = build_model("resnet50", num_classes=NUM_CLASSES, mode="feature_extraction", pretrained=True)
        opt_t04 = build_optimizer(m_t04, opt_type="adamw", lr=1e-3)
        run_exp("T04", m_t04, opt_t04, criterion_ce, train_base, val_base, "ResNet-50 (Pretrained)", "AdamW (1e-3)", "BN", "Standard", "Feature Extraction (Frozen)", "No")

    # T05: ResNet-50 - Partial Fine-Tuning
    if should_continue():
        m_t05 = build_model("resnet50", num_classes=NUM_CLASSES, mode="partial_fine_tuning", pretrained=True)
        opt_t05 = build_optimizer(m_t05, opt_type="adamw", param_groups=m_t05.get_parameter_groups(backbone_lr=1e-4, head_lr=1e-3))
        run_exp("T05", m_t05, opt_t05, criterion_ce, train_base, val_base, "ResNet-50 (Pretrained)", "AdamW (Diff LR)", "BN", "Standard", "Partial FT (layer4)", "No")

    # T06: ConvNeXt-Tiny - Feature Extraction
    if should_continue():
        m_t06 = build_model("convnext_tiny", num_classes=NUM_CLASSES, mode="feature_extraction", pretrained=True)
        opt_t06 = build_optimizer(m_t06, opt_type="adamw", lr=1e-3)
        run_exp("T06", m_t06, opt_t06, criterion_ce, train_base, val_base, "ConvNeXt-Tiny (Pretrained)", "AdamW (1e-3)", "LayerNorm", "Standard", "Feature Extraction (Frozen)", "No")

    # T07: ConvNeXt-Tiny - Partial Fine-Tuning
    if should_continue():
        m_t07 = build_model("convnext_tiny", num_classes=NUM_CLASSES, mode="partial_fine_tuning", pretrained=True)
        opt_t07 = build_optimizer(m_t07, opt_type="adamw", param_groups=m_t07.get_parameter_groups(backbone_lr=5e-5, head_lr=1e-3))
        run_exp("T07", m_t07, opt_t07, criterion_ce, train_base, val_base, "ConvNeXt-Tiny (Pretrained)", "AdamW (Diff LR)", "LayerNorm", "Standard", "Partial FT (stages.3)", "No")

    # T08: ResNet-18 Partial FT + Muon Optimizer (Combinación SOTA)
    if should_continue():
        m_t08 = build_model("resnet18", num_classes=NUM_CLASSES, mode="partial_fine_tuning", pretrained=True)
        opt_t08 = build_optimizer(m_t08, opt_type="muon", lr=0.01, weight_decay=1e-4)
        run_exp("T08", m_t08, opt_t08, criterion_ce, train_base, val_base, "ResNet-18 (Pretrained)", "Muon (lr=0.01)", "BN", "Standard", "Partial FT + Muon", "No")

    # T09: ResNet-18 Partial FT + CutMix/MixUp (Combinación SOTA Transfer + Augment)
    if should_continue():
        m_t09 = build_model("resnet18", num_classes=NUM_CLASSES, mode="partial_fine_tuning", pretrained=True)
        opt_t09 = build_optimizer(m_t09, opt_type="adamw", param_groups=m_t09.get_parameter_groups(backbone_lr=1e-4, head_lr=1e-3))
        batch_aug_t = get_batch_augmentations(num_classes=NUM_CLASSES, mode="cutmix_or_mixup", cutmix_alpha=1.0, mixup_alpha=0.2)
        run_exp("T09", m_t09, opt_t09, criterion_ce, train_base, val_base, "ResNet-18 (Pretrained)", "AdamW (Diff LR)", "BN", "+ CutMix / MixUp", "Partial FT", "No", batch_aug=batch_aug_t)

    # =========================================================================
    # BLOQUE 6: BIODIVERSITY LONG-TAIL CHALLENGE (Sección 11)
    # =========================================================================
    print("\n" + "#" * 80)
    print("  BLOQUE 6: BIODIVERSITY LONG-TAIL CHALLENGE (DESBALANCE DE CLASES)")
    print("#" * 80)

    lt_classes = list(range(NUM_CLASSES))
    lt_indices, _, class_tiers = create_long_tail_subset(
        train_base,
        selected_classes=lt_classes,
        many_shot_count=50,
        med_shot_count=15,
        few_shot_count=3,
        many_ratio=0.25,
        med_ratio=0.35,
        seed=SEED
    )
    train_lt = Subset(train_base, lt_indices)
    print(f"[*] Partición Long-Tail construida: {len(train_lt)} imágenes (Many: {len(class_tiers['frequent'])} sp, Med: {len(class_tiers['medium'])} sp, Few: {len(class_tiers['minority'])} sp)")

    # L01: Long-Tail sin mitigación (Cross-Entropy clásica)
    if should_continue():
        m_l01 = build_model("resnet18", num_classes=NUM_CLASSES, mode="partial_fine_tuning", pretrained=True)
        opt_l01 = build_optimizer(m_l01, opt_type="adamw", lr=1e-3)
        run_exp("L01", m_l01, opt_l01, criterion_ce, train_lt, val_base, "ResNet-18 (Partial FT)", "AdamW", "BN", "Standard", "Partial FT", "Long-Tail (Sin Mitigación)", class_tiers=class_tiers)

    # L02: Long-Tail con Weighted Cross-Entropy
    if should_continue():
        # Calcular pesos inversos de clase
        targets_lt = [train_base.targets[i] for i in lt_indices]
        counts = torch.bincount(torch.tensor(targets_lt), minlength=NUM_CLASSES).float()
        weights = 1.0 / torch.clamp(counts, min=1.0)
        weights = (weights / weights.sum() * NUM_CLASSES).to(device)
        criterion_weighted = nn.CrossEntropyLoss(weight=weights)

        m_l02 = build_model("resnet18", num_classes=NUM_CLASSES, mode="partial_fine_tuning", pretrained=True)
        opt_l02 = build_optimizer(m_l02, opt_type="adamw", lr=1e-3)
        run_exp("L02", m_l02, opt_l02, criterion_weighted, train_lt, val_base, "ResNet-18 (Partial FT)", "AdamW", "BN", "Standard", "Partial FT", "Long-Tail (Weighted CE)", class_tiers=class_tiers)

    # L03: Long-Tail con Focal Loss (gamma=2.0)
    if should_continue():
        focal_loss = build_criterion("focal_loss", gamma=2.0)
        m_l03 = build_model("resnet18", num_classes=NUM_CLASSES, mode="partial_fine_tuning", pretrained=True)
        opt_l03 = build_optimizer(m_l03, opt_type="adamw", lr=1e-3)
        run_exp("L03", m_l03, opt_l03, focal_loss, train_lt, val_base, "ResNet-18 (Partial FT)", "AdamW", "BN", "Standard", "Partial FT", "Long-Tail (Focal Loss g=2.0)", class_tiers=class_tiers)

    # L04: Long-Tail con Focal Loss + CutMix/MixUp (SOTA Mitigación)
    if should_continue():
        focal_loss = build_criterion("focal_loss", gamma=2.0)
        m_l04 = build_model("resnet18", num_classes=NUM_CLASSES, mode="partial_fine_tuning", pretrained=True)
        opt_l04 = build_optimizer(m_l04, opt_type="adamw", lr=1e-3)
        batch_aug_lt = get_batch_augmentations(num_classes=NUM_CLASSES, mode="cutmix_or_mixup", cutmix_alpha=1.0, mixup_alpha=0.2)
        run_exp("L04", m_l04, opt_l04, focal_loss, train_lt, val_base, "ResNet-18 (Partial FT)", "AdamW", "BN", "+ CutMix / MixUp", "Partial FT", "Long-Tail (Focal + CutMix)", class_tiers=class_tiers, batch_aug=batch_aug_lt)

    # =========================================================================
    # BLOQUE 7: EFICIENCIA DE CÓMPUTO Y ACELERACIÓN DE HARDWARE (Sección 14)
    # =========================================================================
    print("\n" + "#" * 80)
    print("  BLOQUE 7: ESTUDIO DE EFICIENCIA DE HARDWARE (FP32 VS AMP FP16)")
    print("#" * 80)

    # E01: Sin AMP (FP32 tradicional)
    if should_continue():
        m_e01 = build_model("resnet18", num_classes=NUM_CLASSES, mode="feature_extraction", pretrained=True)
        opt_e01 = build_optimizer(m_e01, opt_type="adamw", lr=1e-3)
        run_exp("E01", m_e01, opt_e01, criterion_ce, train_base, val_base, "ResNet-18", "AdamW", "BN", "Standard", "Feature Extraction", "No", use_amp=False)

    # E02: Con AMP fp16 + channels_last
    if should_continue():
        m_e02 = build_model("resnet18", num_classes=NUM_CLASSES, mode="feature_extraction", pretrained=True)
        opt_e02 = build_optimizer(m_e02, opt_type="adamw", lr=1e-3)
        run_exp("E02", m_e02, opt_e02, criterion_ce, train_base, val_base, "ResNet-18", "AdamW", "BN", "Standard", "Feature Extraction", "No", use_amp=True)

    # E03: AMP + Gradient Accumulation (steps=2)
    if should_continue():
        m_e03 = build_model("resnet18", num_classes=NUM_CLASSES, mode="feature_extraction", pretrained=True)
        opt_e03 = build_optimizer(m_e03, opt_type="adamw", lr=1e-3)
        run_exp("E03", m_e03, opt_e03, criterion_ce, train_base, val_base, "ResNet-18", "AdamW", "BN", "Standard", "Feature Extraction", "No", use_amp=True, grad_accum_steps=2)

    # =========================================================================
    # CONSOLIDACIÓN FINAL, EXPORTACIÓN Y GRAFICACIÓN DE 4 PANELES
    # =========================================================================
    print("\n" + "=" * 80)
    print("  PROCESO EXPERIMENTAL COMPLETADO - GENERANDO ENTREGABLES")
    print("=" * 80)

    # 1. Guardar tabla final
    tracker.save_markdown_table("outputs/tabla_final_experimentos.md")
    tracker.save_json("outputs/tabla_final_experimentos.json")
    print("[OK] Tabla final guardada en outputs/tabla_final_experimentos.md")

    # 2. Generar gráfico 4-grid para los Top 5 experimentos
    top_entries = ckpt_manager.get_manifest()[:5]
    top_labels = [e["exp_id"] for e in top_entries]
    selected_histories = {}
    for hist_key, hist_val in all_histories.items():
        for top_id in top_labels:
            if f"[{top_id}]" in hist_key:
                selected_histories[hist_key] = hist_val
                break

    if selected_histories:
        plot_path = str(ROOT_DIR / "outputs" / "comparativa_top_modelos_4grid.png")
        plot_multiexperiment_4grid(
            selected_histories,
            title="Convergencia y Eficiencia del Top de Modelos (Pérdida y Precisión vs. Épocas y Tiempo)",
            save_path=plot_path
        )
        print(f"[OK] Gráfica de 4 paneles guardada en {plot_path}")

    # 3. Resumen del Top 10
    print("\n" + "=" * 80)
    print("  MANIFIESTO FINAL: TOP 10 MODELOS GUARDADOS")
    print("=" * 80)
    manifest = ckpt_manager.get_manifest()
    for entry in manifest:
        rank = entry.get("rank", "-")
        exp_id = entry.get("exp_id", "")
        model_name = entry.get("model_name", "")
        f1 = entry.get("metrics", {}).get("macro_f1", 0.0) * 100
        top1 = entry.get("metrics", {}).get("top1_acc", 0.0) * 100
        fpath = Path(entry.get("filepath", "")).name
        print(f"  #{rank:02d} | [{exp_id}] {model_name:<28} | Macro F1: {f1:5.2f}% | Top-1: {top1:5.2f}% | Archivo: {fpath}")
    print("=" * 80)


if __name__ == "__main__":
    main()
