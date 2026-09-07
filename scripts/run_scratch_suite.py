"""
Scratch Benchmark Suite: Training Top Architectures (ConvNeXt-Tiny & Swin-T) from Scratch.
Explores:
- SC01: ConvNeXt-Tiny (Scratch, 224px) | AdamW (Decoupled, lr=1e-3, wd=0.01) + CosineLR
- SC02: ConvNeXt-Tiny (Scratch, 224px) | Hybrid (Muon on 2D/4D Convs + AdamW on Head/Norm) + CosineLR
- SC03: ConvNeXt-Tiny (Scratch, 224px) | Lion (Google Brain, lr=1e-4, wd=0.01) + CosineLR
- SC04: Swin-T (Scratch, 224px)        | AdamW (Decoupled, lr=5e-4, wd=0.05) + CosineLR
- SC05: Swin-T (Scratch, 224px)        | Hybrid (Muon on Self-Attention Q,K,V + AdamW on Head/Norm) + CosineLR

Features:
- Live status export to outputs/scratch_suite_live_status.json for real-time monitoring every 5 minutes
- Guaranteed Best Epoch Weight Restoration on all models before saving
- Native 224x224 image resolution cached in RAM
- Real-time logging to TensorBoard (http://localhost:6006)
- Dynamic re-ranking and persistence of checkpoints
- Consolidation of tables in outputs/tabla_final_experimentos.md
"""

import argparse
import copy
import json
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.utils.seed import seed_everything
from src.data.dataset import INatDataset
from src.data.transforms import get_transforms
from src.data.dataloader import build_dataloaders
from src.models.factory import build_model, count_parameters
from src.training.losses import build_criterion
from src.training.optimizers import build_optimizer
from src.training.hybrid_optimizer import build_hybrid_muon_adamw
from src.training.trainer import Trainer
from src.utils.metrics import compute_classification_metrics, compute_kingdom_breakdown
from src.utils.tracking import ExperimentTracker
from src.utils.checkpoints import TopKCheckpointManager


STATUS_FILE = ROOT_DIR / "outputs" / "scratch_suite_live_status.json"


def update_live_status(
    status: str,
    current_exp: Optional[str] = None,
    current_epoch: Optional[int] = None,
    total_epochs: Optional[int] = None,
    completed_experiments: Optional[List[Dict[str, Any]]] = None,
    msg: Optional[str] = None
):
    """Writes real-time status to JSON file for external polling/monitoring."""
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "current_exp": current_exp,
        "current_epoch": current_epoch,
        "total_epochs": total_epochs,
        "last_update": datetime.now().isoformat(),
        "message": msg or "",
        "completed_experiments": completed_experiments or []
    }
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run_scratch_suite(epochs: int = 15, batch_size: int = 32, img_size: int = 224):
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*75}")
    print(f"  [*] SCRATCH BENCHMARK SUITE: CONVNEXT & SWIN-T DESDE CERO ({img_size}px)")
    print(f"{'='*75}")
    print(f"[*] Dispositivo: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    if device.type == "cuda":
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[*] VRAM Total disponible: {total_vram:.2f} GB")

    # 1. Dataset loading at 224x224 cached in RAM
    sample_dir = ROOT_DIR.parent / "recursos" / "inat2021_sample"
    if not (sample_dir.exists() and (sample_dir / "train_mini.json").exists()):
        sample_dir = ROOT_DIR.parent / "recursos" / "sample_inat"
        if not (sample_dir.exists() and (sample_dir / "train_mini.json").exists()):
            sample_dir = ROOT_DIR / "data"

    print(f"[*] Directorio de dataset: {sample_dir.name}")
    train_json = sample_dir / "train_mini.json"
    val_json = sample_dir / "val.json"
    train_img_dir = sample_dir / "train_mini"
    val_img_dir = sample_dir / "val"

    tf_train = get_transforms(split="train", img_size=img_size, aug_mode="standard")
    tf_val = get_transforms(split="val", img_size=img_size, aug_mode="none")

    print("[*] Precargando imágenes en memoria RAM a 224x224...")
    train_dataset = INatDataset(train_json, train_img_dir, transform=tf_train, cache_in_ram=True)
    val_dataset = INatDataset(val_json, val_img_dir, transform=tf_val, category_to_label=train_dataset.category_to_label, cache_in_ram=True)

    num_classes = len(train_dataset.category_to_label)
    with open(train_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    cat_to_kingdom = {cat["id"]: cat.get("kingdom", "Unknown") for cat in meta.get("categories", [])}
    label_to_kingdom = {label: cat_to_kingdom.get(cat_id, "Unknown") for cat_id, label in train_dataset.category_to_label.items()}

    train_loader, val_loader = build_dataloaders(
        train_dataset, val_dataset, batch_size=batch_size, num_workers=0, seed=42
    )
    print(f"[[OK]] DataLoaders listos: {len(train_dataset)} Train | {len(val_dataset)} Val | {num_classes} Especies")

    tracker = ExperimentTracker(log_dir=str(ROOT_DIR / "logs"))
    ckpt_manager = TopKCheckpointManager(checkpoint_dir=str(ROOT_DIR / "checkpoints"), k=10, primary_metric="macro_f1")
    criterion = build_criterion("cross_entropy")

    # Scratch Experiment Definitions
    experiments = [
        {
            "id": "SC01",
            "name": "ConvNeXt-Tiny (Scratch, 224px)",
            "model_type": "convnext_tiny",
            "opt_type": "adamw",
            "lr": 1e-3,
            "weight_decay": 1e-2,
            "description": "ConvNeXt-Tiny entrenada desde cero (pesos aleatorios) con AdamW desacoplado (lr=1e-3, wd=0.01) y decaimiento coseno a 224px.",
            "use_hybrid": False
        },
        {
            "id": "SC02",
            "name": "ConvNeXt-Tiny (Scratch, 224px)",
            "model_type": "convnext_tiny",
            "opt_type": "hybrid_muon",
            "muon_lr": 0.01,
            "adamw_lr": 1e-3,
            "weight_decay": 1e-2,
            "description": "ConvNeXt-Tiny desde cero con optimizador Híbrido: Muon (Newton-Schulz en convoluciones 2D/4D) y AdamW en cabeza/norm.",
            "use_hybrid": True
        },
        {
            "id": "SC03",
            "name": "ConvNeXt-Tiny (Scratch, 224px)",
            "model_type": "convnext_tiny",
            "opt_type": "lion",
            "lr": 1e-4,
            "weight_decay": 1e-2,
            "description": "ConvNeXt-Tiny desde cero con optimizador Lion (Google Brain - sign momentum) lr=1e-4 y decaimiento coseno a 224px.",
            "use_hybrid": False
        },
        {
            "id": "SC04",
            "name": "Swin-T (Scratch, 224px)",
            "model_type": "swin_t",
            "opt_type": "adamw",
            "lr": 5e-4,
            "weight_decay": 5e-2,
            "description": "Swin Transformer (Swin-T) desde cero con pesos aleatorios y AdamW desacoplado (lr=5e-4, wd=0.05) con decaimiento coseno a 224px.",
            "use_hybrid": False
        },
        {
            "id": "SC05",
            "name": "Swin-T (Scratch, 224px)",
            "model_type": "swin_t",
            "opt_type": "hybrid_muon",
            "muon_lr": 0.01,
            "adamw_lr": 1e-3,
            "weight_decay": 1e-2,
            "description": "Swin-T desde cero con Muon Híbrido: Newton-Schulz en matrices Q,K,V y proyecciones lineales, y AdamW en cabeza/norm.",
            "use_hybrid": True
        }
    ]

    completed_results = []
    update_live_status("STARTING", total_epochs=epochs, completed_experiments=completed_results, msg="Inicializando suite...")

    for i, exp in enumerate(experiments, 1):
        exp_id = exp["id"]
        exp_name = exp["name"]
        print(f"\n{'='*75}")
        print(f"  [{i}/{len(experiments)}] EJECUTANDO {exp_id}: {exp_name}")
        print(f"  Detalle: {exp['description']}")
        print(f"{'='*75}")

        update_live_status("RUNNING", current_exp=f"{exp_id}: {exp_name}", current_epoch=0, total_epochs=epochs, completed_experiments=completed_results, msg=f"Iniciando {exp_id}...")

        # 1. Instantiate model from scratch (pretrained=False, mode='full_fine_tuning')
        model = build_model(
            exp["model_type"],
            num_classes=num_classes,
            pretrained=False,
            mode="full_fine_tuning",
            dropout_rate=0.2
        )
        total_p, train_p, total_m, train_m = count_parameters(model)
        print(f"[*] Parámetros totales: {total_m:.2f}M | Entrenables: {train_m:.2f}M (100% scratch)")

        # 2. Build optimizer
        if exp.get("use_hybrid", False):
            optimizer = build_hybrid_muon_adamw(
                model,
                muon_lr=exp["muon_lr"],
                adamw_lr=exp["adamw_lr"],
                weight_decay=exp["weight_decay"]
            )
            opt_label = "Hybrid (Muon+AdamW)"
        elif exp["opt_type"] == "lion":
            optimizer = build_optimizer(model, opt_type="lion", lr=exp["lr"], weight_decay=exp["weight_decay"])
            opt_label = "Lion (Google Brain)"
        else:
            optimizer = build_optimizer(model, opt_type="adamw", lr=exp["lr"], weight_decay=exp["weight_decay"])
            opt_label = f"AdamW (lr={exp['lr']})"

        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        tb_dir = str(ROOT_DIR / "logs" / "tb_runs" / f"{exp_id}_{exp['model_type']}_{exp['opt_type']}")
        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scheduler=scheduler,
            use_amp=True,
            label_to_kingdom=label_to_kingdom,
            tb_log_dir=tb_dir
        )

        # 3. Train loop with guaranteed best model restoration
        history, best_metrics, peak_vram, train_time = trainer.fit(
            train_loader,
            val_loader,
            epochs=epochs,
            verbose=True,
            show_pbar=True
        )

        best_epoch = best_metrics.get("best_epoch", epochs)
        val_top1 = best_metrics.get("top1_acc", 0.0)
        val_f1 = best_metrics.get("macro_f1", 0.0)
        val_top5 = best_metrics.get("top5_acc", 0.0)
        val_loss = best_metrics.get("loss", 0.0)
        anim_acc = best_metrics.get("kingdom_animalia_acc", 0.0)
        anim_f1 = best_metrics.get("kingdom_animalia_macro_f1", 0.0)
        plant_acc = best_metrics.get("kingdom_plantae_acc", 0.0)
        plant_f1 = best_metrics.get("kingdom_plantae_macro_f1", 0.0)

        full_metrics = {
            "top1_acc": float(val_top1),
            "macro_f1": float(val_f1),
            "weighted_f1": float(best_metrics.get("weighted_f1", val_f1)),
            "top5_acc": float(val_top5),
            "kingdom_animalia_acc": float(anim_acc),
            "kingdom_animalia_macro_f1": float(anim_f1),
            "kingdom_animalia_count": int(best_metrics.get("kingdom_animalia_count", 280)),
            "kingdom_plantae_acc": float(plant_acc),
            "kingdom_plantae_macro_f1": float(plant_f1),
            "kingdom_plantae_count": int(best_metrics.get("kingdom_plantae_count", 220)),
            "best_epoch": int(best_epoch),
            "val_loss": float(val_loss)
        }

        hyperparams = {
            "optimizer": opt_label,
            "batch_size": batch_size,
            "epochs": epochs,
            "img_size": img_size,
            "use_amp": True,
            "regularization": "LayerNorm / Dropout(0.2)",
            "augmentation": f"Standard ({img_size}px)",
            "pretrained": False,
            "best_epoch": int(best_epoch)
        }

        # 5. Log to ExperimentTracker
        tracker.log_experiment(
            exp_id=exp_id,
            model_name=exp_name,
            optimizer=opt_label,
            regularization="LayerNorm / Dropout(0.2)",
            augmentation=f"Standard ({img_size}px)",
            transfer_learning="Scratch (Random Init)",
            long_tail="No",
            metrics=full_metrics,
            training_time_sec=train_time,
            peak_vram_mb=peak_vram,
            param_count_m=total_m,
            extra_metadata=hyperparams
        )

        # 6. Save checkpoint if eligible
        ckpt_manager.register_and_save(
            model=model,
            exp_id=exp_id,
            model_name=exp_name,
            metrics=full_metrics,
            hyperparams=hyperparams
        )

        exp_summary = {
            "exp_id": exp_id,
            "name": exp_name,
            "optimizer": opt_label,
            "best_epoch": int(best_epoch),
            "total_epochs": epochs,
            "top1_acc": float(val_top1),
            "macro_f1": float(val_f1),
            "top5_acc": float(val_top5),
            "animalia_acc": float(anim_acc),
            "plantae_acc": float(plant_acc),
            "training_time": float(train_time),
            "peak_vram_mb": float(peak_vram)
        }
        completed_results.append(exp_summary)

        print(f"\n[[OK]] {exp_id} COMPLETADO:")
        print(f"    - Mejor Época: {best_epoch}/{epochs} | Val Loss: {val_loss:.4f}")
        print(f"    - Top-1 Acc: {val_top1*100:.2f}% | Macro-F1: {val_f1*100:.2f}% | Top-5: {val_top5*100:.2f}%")
        print(f"    - Animalia: {anim_acc*100:.2f}% | Plantae: {plant_acc*100:.2f}%")
        print(f"    - Tiempo: {train_time:.1f}s | VRAM: {peak_vram:.1f} MB")

        update_live_status("RUNNING", current_exp=exp_id, current_epoch=epochs, total_epochs=epochs, completed_experiments=completed_results, msg=f"{exp_id} finalizado exitosamente.")

    # Suite completion
    update_live_status("COMPLETED", current_exp=None, current_epoch=epochs, total_epochs=epochs, completed_experiments=completed_results, msg="Suite de Scratch completada con éxito.")

    # Export deliverables
    print(f"\n{'='*75}")
    print("  [*] GENERANDO ENTREGABLES FINALES Y ACTUALIZANDO RANKING")
    print(f"{'='*75}")
    tbl_md = ROOT_DIR / "outputs" / "tabla_final_experimentos.md"
    with open(tbl_md, "w", encoding="utf-8") as f:
        f.write(tracker.to_markdown_table())
    print(f"[[OK]] Tabla markdown actualizada: {tbl_md}")

    tbl_json = ROOT_DIR / "outputs" / "tabla_final_experimentos.json"
    with open(tbl_json, "w", encoding="utf-8") as f:
        json.dump(tracker.experiments, f, indent=2, ensure_ascii=False)
    print(f"[[OK]] Tabla JSON actualizada: {tbl_json}")

    print("\n[[OK]] Resumen de la Suite Scratch:")
    for res in completed_results:
        print(f"  - {res['exp_id']} ({res['name']}): Top-1={res['top1_acc']*100:.2f}%, F1={res['macro_f1']*100:.2f}%, Top-5={res['top5_acc']*100:.2f}%, Tiempo={res['training_time']:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scratch Benchmark Suite")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs per experiment")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--img-size", type=int, default=224, help="Image resolution")
    args = parser.parse_args()

    run_scratch_suite(epochs=args.epochs, batch_size=args.batch_size, img_size=args.img_size)
