"""
Extended Benchmark Suite: ConvNeXt-Tiny Multi-Optimizer & Epoch Analysis.
Evaluates:
- CX01: ConvNeXt-Tiny (Partial FT) | SGD + Momentum (Nesterov)
- CX02: ConvNeXt-Tiny (Partial FT) | Adam Clásico
- CX03: ConvNeXt-Tiny (Partial FT) | AdamW Baseline
- CX04: ConvNeXt-Tiny (Partial FT) | Muon (Newton-Schulz Orthogonal Momentum)
- CX05: ConvNeXt-Tiny (Partial FT) | Lion (Google Brain Sign Momentum)
- CX06: ConvNeXt-Tiny (Scratch)    | Muon (Newton-Schulz Orthogonal Momentum)
- CX07: ConvNeXt-Tiny (Partial FT) | AdamW + CosineAnnealingLR (8 Epochs)
- CX08: ConvNeXt-Tiny (Partial FT) | AdamW + CutMix/MixUp (GPU) + CosineAnnealingLR (8 Epochs)

Integrates live into TensorBoard, updates ExperimentTracker (tables and JSON),
and dynamically updates the Top 10 Champion Checkpoint ranking in checkpoints/.
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
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.utils.seed import seed_everything
from src.data.dataset import INatDataset
from src.data.transforms import get_transforms, get_batch_augmentations
from src.data.dataloader import build_dataloaders
from src.models.factory import build_model, count_parameters
from src.models.transfer import TransferCNN
from src.training.losses import build_criterion
from src.training.optimizers import build_optimizer
from src.training.trainer import Trainer
from src.utils.tracking import ExperimentTracker
from src.utils.checkpoints import TopKCheckpointManager
from src.utils.visualization import plot_multiexperiment_4grid


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="Extended ConvNeXt-Tiny Benchmark Suite")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--img_size", type=int, default=128, help="Image size")
    args = parser.parse_args()

    print("=" * 80)
    print("  iNaturalist 2021: SUITE EXTENDIDA CONVNEXT-TINY (MULTI-OPTIMIZADOR & ÉPOCAS)")
    print("=" * 80)

    SEED = 42
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Dispositivo de cómputo: {device}")
    if device.type == "cuda":
        print(f"[*] GPU detectada: {torch.cuda.get_device_name(0)}")

    # 1. Rutas del dataset de 50 clases (3,000 imágenes)
    data_dir = ROOT_DIR.parent / "recursos" / "inat2021_sample"
    if not (data_dir / "train_mini.json").exists():
        data_dir = ROOT_DIR / "data" / "inat2021_sample"

    print(f"[*] Carpeta de datos: {data_dir}")
    train_json = data_dir / "train_mini.json"
    val_json = data_dir / "val.json"
    train_img_dir = data_dir / "train_mini"
    val_img_dir = data_dir / "val"

    # Mapeo taxonómico a Reino
    with open(train_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    cat_to_kingdom = {cat["id"]: cat.get("kingdom", "Unknown") for cat in meta.get("categories", [])}

    tf_train = get_transforms("train", img_size=args.img_size, aug_mode="standard")
    tf_val = get_transforms("val", img_size=args.img_size, aug_mode="none")

    print("[*] Precargando datos en RAM (eliminando cuellos de botella de disco)...")
    train_base = INatDataset(train_json, train_img_dir, transform=tf_train, cache_in_ram=True)
    val_base = INatDataset(val_json, val_img_dir, transform=tf_val, category_to_label=train_base.category_to_label, cache_in_ram=True)


    NUM_CLASSES = len(train_base.category_to_label)
    label_to_kingdom = {label: cat_to_kingdom.get(cat_id, "Unknown") for cat_id, label in train_base.category_to_label.items()}
    print(f"[OK] {len(train_base)} muestras Train | {len(val_base)} muestras Val | {NUM_CLASSES} Especies")

    # Tracking y checkpoints
    tracker = ExperimentTracker(log_dir=str(ROOT_DIR / "logs"))
    ckpt_manager = TopKCheckpointManager(checkpoint_dir=str(ROOT_DIR / "checkpoints"), k=10)
    criterion_ce = build_criterion("cross_entropy")

    all_histories = {}

    def run_convnext_exp(
        exp_id: str,
        model: nn.Module,
        opt: torch.optim.Optimizer,
        criterion: nn.Module,
        model_name: str,
        opt_name: str,
        reg_name: str,
        aug_name: str,
        tl_name: str,
        epochs: int = 5,
        scheduler=None,
        batch_aug=None
    ):
        print("\n" + "=" * 78)
        print(f"  >>> [{exp_id}] {model_name} | Opt: {opt_name} | Épocas: {epochs}")
        print(f"      Reg: {reg_name} | Aug: {aug_name} | TL: {tl_name} | Sched: {scheduler is not None}")
        print("=" * 78)

        tb_dir = str(ROOT_DIR / "logs" / "tb_runs" / f"{exp_id}_{model_name.replace(' ', '_')}_{opt_name.split()[0]}")
        train_l, val_l = build_dataloaders(train_base, val_base, batch_size=args.batch_size, num_workers=0, seed=SEED)

        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=opt,
            device=device,
            scheduler=scheduler,
            use_amp=True,
            label_to_kingdom=label_to_kingdom,
            batch_augmenter=batch_aug,
            tb_log_dir=tb_dir
        )

        history, best_metrics, peak_vram, total_time = trainer.fit(train_l, val_l, epochs=epochs, verbose=True)
        _, _, total_m, _ = count_parameters(model)

        # 1. Registrar en tracker
        record = tracker.log_experiment(
            exp_id=exp_id,
            model_name=model_name,
            optimizer=opt_name,
            regularization=reg_name,
            augmentation=aug_name,
            transfer_learning=tl_name,
            long_tail="No",
            metrics=best_metrics,
            training_time_sec=total_time,
            peak_vram_mb=peak_vram,
            param_count_m=total_m,
            extra_metadata={"epochs": epochs, "scheduler": str(type(scheduler).__name__) if scheduler else "None"}
        )

        # 2. Registrar en Top-10 Checkpoint Manager
        rank = ckpt_manager.register_and_save(
            model=model,
            exp_id=exp_id,
            model_name=model_name,
            metrics=best_metrics,
            hyperparams={
                "optimizer": opt_name,
                "batch_size": args.batch_size,
                "epochs": epochs,
                "img_size": args.img_size,
                "use_amp": True,
                "regularization": reg_name,
                "augmentation": aug_name
            }
        )

        f1 = best_metrics.get("macro_f1", 0.0) * 100
        top1 = best_metrics.get("top1_acc", 0.0) * 100
        rank_str = f"TOP #{rank}" if rank is not None else "No clasifica al Top-10"
        print(f"[OK] [{exp_id}] Terminado: Val Macro-F1={f1:.2f}% | Top-1={top1:.2f}% | VRAM={peak_vram:.1f}MB | Tiempo={total_time:.1f}s | {rank_str}")

        all_histories[f"[{exp_id}] {model_name} ({opt_name.split()[0]})"] = history
        return history, best_metrics

    # =========================================================================
    # EJECUCIÓN DE LA SUITE CX01 - CX08
    # =========================================================================

    # CX01: ConvNeXt-Tiny (Partial FT) + SGD + Momentum
    m_cx01 = build_model("convnext_tiny", num_classes=NUM_CLASSES, mode="partial_fine_tuning", pretrained=True)
    pg_cx01 = m_cx01.get_parameter_groups(backbone_lr=1e-3, head_lr=1e-2)
    opt_cx01 = build_optimizer(m_cx01, opt_type="sgd_momentum", lr=1e-2, param_groups=pg_cx01)
    run_convnext_exp("CX01", m_cx01, opt_cx01, criterion_ce, "ConvNeXt-Tiny (Pretrained)", "SGD+Momentum (1e-2)", "LayerNorm", "Standard", "Partial FT (stages.3)", epochs=5)

    # CX02: ConvNeXt-Tiny (Partial FT) + Adam Clásico
    m_cx02 = build_model("convnext_tiny", num_classes=NUM_CLASSES, mode="partial_fine_tuning", pretrained=True)
    pg_cx02 = m_cx02.get_parameter_groups(backbone_lr=1e-4, head_lr=1e-3)
    opt_cx02 = build_optimizer(m_cx02, opt_type="adam", lr=1e-3, param_groups=pg_cx02)
    run_convnext_exp("CX02", m_cx02, opt_cx02, criterion_ce, "ConvNeXt-Tiny (Pretrained)", "Adam (1e-3)", "LayerNorm", "Standard", "Partial FT (stages.3)", epochs=5)

    # CX03: ConvNeXt-Tiny (Partial FT) + AdamW Baseline
    m_cx03 = build_model("convnext_tiny", num_classes=NUM_CLASSES, mode="partial_fine_tuning", pretrained=True)
    pg_cx03 = m_cx03.get_parameter_groups(backbone_lr=5e-5, head_lr=1e-3)
    opt_cx03 = build_optimizer(m_cx03, opt_type="adamw", lr=1e-3, param_groups=pg_cx03)
    run_convnext_exp("CX03", m_cx03, opt_cx03, criterion_ce, "ConvNeXt-Tiny (Pretrained)", "AdamW (Diff LR)", "LayerNorm", "Standard", "Partial FT (stages.3)", epochs=5)

    # CX04: ConvNeXt-Tiny (Partial FT) + Muon Optimizer (Newton-Schulz Orthogonal)
    m_cx04 = build_model("convnext_tiny", num_classes=NUM_CLASSES, mode="partial_fine_tuning", pretrained=True)
    trainable_p = [p for p in m_cx04.parameters() if p.requires_grad]
    opt_cx04 = build_optimizer(m_cx04, opt_type="muon", lr=0.01, param_groups=trainable_p)
    run_convnext_exp("CX04", m_cx04, opt_cx04, criterion_ce, "ConvNeXt-Tiny (Pretrained)", "Muon (lr=0.01)", "LayerNorm", "Standard", "Partial FT + Muon", epochs=5)

    # CX05: ConvNeXt-Tiny (Partial FT) + Lion Optimizer (Google Brain Sign Momentum)
    m_cx05 = build_model("convnext_tiny", num_classes=NUM_CLASSES, mode="partial_fine_tuning", pretrained=True)
    pg_cx05 = m_cx05.get_parameter_groups(backbone_lr=2e-5, head_lr=1e-4)
    opt_cx05 = build_optimizer(m_cx05, opt_type="lion", lr=1e-4, param_groups=pg_cx05)
    run_convnext_exp("CX05", m_cx05, opt_cx05, criterion_ce, "ConvNeXt-Tiny (Pretrained)", "Lion (1e-4)", "LayerNorm", "Standard", "Partial FT + Lion", epochs=5)

    # CX06: ConvNeXt-Tiny (Scratch) + Muon Optimizer
    m_cx06 = build_model("convnext_tiny", num_classes=NUM_CLASSES, pretrained=False)
    opt_cx06 = build_optimizer(m_cx06, opt_type="muon", lr=0.02)
    run_convnext_exp("CX06", m_cx06, opt_cx06, criterion_ce, "ConvNeXt-Tiny (Scratch)", "Muon (lr=0.02)", "LayerNorm+DropPath", "Standard", "No", epochs=5)

    # CX07: ConvNeXt-Tiny (Partial FT) + AdamW + CosineAnnealingLR (8 Épocas)
    m_cx07 = build_model("convnext_tiny", num_classes=NUM_CLASSES, mode="partial_fine_tuning", pretrained=True)
    pg_cx07 = m_cx07.get_parameter_groups(backbone_lr=5e-5, head_lr=1e-3)
    opt_cx07 = build_optimizer(m_cx07, opt_type="adamw", lr=1e-3, param_groups=pg_cx07)
    sched_cx07 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_cx07, T_max=8, eta_min=1e-6)
    run_convnext_exp("CX07", m_cx07, opt_cx07, criterion_ce, "ConvNeXt-Tiny (Pretrained)", "AdamW + CosineLR", "LayerNorm", "Standard", "Partial FT (8 Epochs)", epochs=8, scheduler=sched_cx07)

    # CX08: ConvNeXt-Tiny (Partial FT) + CutMix/MixUp on GPU + CosineAnnealingLR (8 Épocas)
    m_cx08 = build_model("convnext_tiny", num_classes=NUM_CLASSES, mode="partial_fine_tuning", pretrained=True)
    pg_cx08 = m_cx08.get_parameter_groups(backbone_lr=5e-5, head_lr=1e-3)
    opt_cx08 = build_optimizer(m_cx08, opt_type="adamw", lr=1e-3, param_groups=pg_cx08)
    sched_cx08 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_cx08, T_max=8, eta_min=1e-6)
    batch_aug_cx08 = get_batch_augmentations(num_classes=NUM_CLASSES, mode="cutmix_or_mixup", cutmix_alpha=1.0, mixup_alpha=0.2)
    run_convnext_exp("CX08", m_cx08, opt_cx08, criterion_ce, "ConvNeXt-Tiny (Pretrained)", "AdamW + CutMix + CosineLR", "LayerNorm", "+ CutMix / MixUp (8 Epochs)", "Partial FT (8 Epochs)", epochs=8, scheduler=sched_cx08, batch_aug=batch_aug_cx08)

    # =========================================================================
    # RE-EXPORTACIÓN DE ENTREGABLES
    # =========================================================================
    print("\n" + "=" * 80)
    print("  ACTUALIZANDO TABLAS FINALES Y ENTREGABLES...")
    print("=" * 80)

    outputs_dir = ROOT_DIR / "outputs"
    tracker.save_markdown_table(str(outputs_dir / "tabla_final_experimentos.md"))
    tracker.save_json(str(outputs_dir / "tabla_final_experimentos.json"))
    print(f"[OK] Tabla oficial guardada en {outputs_dir / 'tabla_final_experimentos.md'}")
    print(f"[OK] JSON consolidado guardado en {outputs_dir / 'tabla_final_experimentos.json'}")

    # Manifiesto Top 10
    print("\n" + "=" * 80)
    print("  RANKING ACTUALIZADO: TOP 10 MODELOS GUARDADOS EN DISCO")
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
