"""
Stage 2 Orchestrator: Scale-Up to 10,000 Classes
Trains the top promising models on stratified subsets (50k or 100k images)
and evaluates on the full 100k validation set (val.json).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
import torch
import torch.nn as nn

# Root path setup
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.seed import seed_everything
from src.data.dataset import INatDataset
from src.data.transforms import get_transforms, get_val_transforms
from src.data.dataloader import build_dataloaders, create_eval_loader
from src.models.factory import build_model, count_parameters
from src.training.losses import build_criterion
from src.training.optimizers import build_optimizer
from src.training.trainer import Trainer
from src.training.early_stopping import EarlyStopping
from src.utils.tracking import ExperimentTracker
from src.utils.checkpoints import TopKCheckpointManager
from src.utils.metrics import compute_classification_metrics

STAGE2_MODELS = [
    {
        "id": "ST2_01",
        "name": "Swin-T (Transformer)",
        "arch": "swin_t",
        "optimizer": "muon",
        "lr": 0.001,
        "epochs": 3,
        "img_size": 224,
        "desc": "Champion Model: Swin-T + Muon Hybrid on 10,000 classes"
    },
    {
        "id": "ST2_02",
        "name": "ConvNeXt-Tiny (Deep FT)",
        "arch": "convnext_tiny",
        "optimizer": "adamw",
        "lr": 0.0005,
        "epochs": 3,
        "img_size": 224,
        "desc": "Top Convolutional: ConvNeXt-Tiny + AdamW (Stages 3+4)"
    },
    {
        "id": "ST2_03",
        "name": "ConvNeXt-Tiny (CosineHead)",
        "arch": "convnext_tiny",
        "optimizer": "adamw",
        "lr": 0.0005,
        "epochs": 3,
        "img_size": 224,
        "use_cosine_head": True,
        "desc": "Spherical Projection: ConvNeXt-Tiny + Cosine Head"
    },
    {
        "id": "ST2_04",
        "name": "ConvNeXt-Tiny (Muon Hybrid)",
        "arch": "convnext_tiny",
        "optimizer": "muon",
        "lr": 0.001,
        "epochs": 3,
        "img_size": 224,
        "desc": "ConvNeXt-Tiny + Muon Hybrid on 10,000 classes"
    },
    {
        "id": "ST2_05",
        "name": "Swin-T (AdamW)",
        "arch": "swin_t",
        "optimizer": "adamw",
        "lr": 0.0001,
        "epochs": 3,
        "img_size": 224,
        "desc": "Swin-T + Standard AdamW Decoupled"
    },
    {
        "id": "ST2_06",
        "name": "ConvNeXt-Tiny (CutMix/MixUp)",
        "arch": "convnext_tiny",
        "optimizer": "adamw",
        "lr": 0.0005,
        "epochs": 3,
        "img_size": 224,
        "use_cutmix": True,
        "desc": "ConvNeXt-Tiny with GPU CutMix / MixUp"
    },
    {
        "id": "ST2_07",
        "name": "ResNet-50 (Partial FT)",
        "arch": "resnet50",
        "optimizer": "adamw",
        "lr": 0.0005,
        "epochs": 3,
        "img_size": 224,
        "desc": "ResNet-50 Deep Residual Baseline"
    },
    {
        "id": "ST2_08",
        "name": "ResNet-18 (Focal Loss)",
        "arch": "resnet18",
        "optimizer": "adamw",
        "lr": 0.001,
        "epochs": 3,
        "img_size": 224,
        "loss": "focal_loss",
        "desc": "ResNet-18 with Focal Loss for 10k class imbalance"
    },
    {
        "id": "ST2_09",
        "name": "Swin-T (Scratch)",
        "arch": "swin_t",
        "optimizer": "muon",
        "lr": 0.002,
        "epochs": 5,
        "img_size": 224,
        "pretrained": False,
        "desc": "Top Scratch Vision Transformer trained from random weights"
    },
    {
        "id": "ST2_10",
        "name": "ConvNeXt-Tiny (Scratch)",
        "arch": "convnext_tiny",
        "optimizer": "muon",
        "lr": 0.002,
        "epochs": 5,
        "img_size": 224,
        "pretrained": False,
        "desc": "Top Scratch ConvNet trained from random weights with Muon"
    }
]


def run_experiment(exp_cfg, data_paths, device, num_classes=10000, batch_size=32):
    print("\n" + "=" * 75)
    print(f"  ETAPA 2: {exp_cfg['id']} - {exp_cfg['name']}")
    print(f"  Arquitectura: {exp_cfg['arch']} | Optimizador: {exp_cfg['optimizer']} | Clases: {num_classes:,}")
    print(f"  Descripción:  {exp_cfg['desc']}")
    print("=" * 75)

    img_size = exp_cfg.get("img_size", 224)
    pretrained = exp_cfg.get("pretrained", True)
    use_cosine_head = exp_cfg.get("use_cosine_head", False)

    # 1. Model Instantiation
    model = build_model(
        model_type=exp_cfg["arch"],
        num_classes=num_classes,
        pretrained=pretrained,
        use_cosine_head=use_cosine_head
    )
    total_p, train_p, total_m, train_m = count_parameters(model)
    print(f"[*] Parámetros Totales: {total_m:.2f} M | Entrenables: {train_m:.2f} M")

    # 2. Criterion
    loss_type = exp_cfg.get("loss", "cross_entropy")
    criterion = build_criterion(loss_type)

    # 3. Optimizer
    opt_type = exp_cfg["optimizer"]
    lr = exp_cfg.get("lr", 1e-3)
    optimizer = build_optimizer(model, opt_type=opt_type, lr=lr)

    # 4. Data Loaders
    tf_train = get_transforms(split="train", img_size=img_size, aug_mode="standard")
    tf_val = get_val_transforms(img_size=img_size)

    print("[*] Inicializando DataLoaders para 10,000 clases...")
    train_dataset = INatDataset(data_paths["train_json"], data_paths["train_img_dir"], transform=tf_train, cache_in_ram=False)
    val_dataset = INatDataset(data_paths["val_json"], data_paths["val_img_dir"], transform=tf_val, category_to_label=train_dataset.category_to_label, cache_in_ram=False)

    train_loader, val_loader = build_dataloaders(train_dataset, val_dataset, batch_size=batch_size, num_workers=4, seed=42)

    # 5. Trainer
    tb_log = str(ROOT_DIR / "logs" / "tb_runs" / exp_cfg["id"])
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        use_amp=True,
        tb_log_dir=tb_log
    )

    epochs = exp_cfg.get("epochs", 3)
    print(f"[*] Iniciando entrenamiento por {epochs} épocas...")
    history, best_metrics, peak_vram, elapsed = trainer.fit(
        train_loader, val_loader, epochs=epochs, verbose=True, show_pbar=True
    )

    print(f"\n[✓] {exp_cfg['id']} Completado en {elapsed:.1f}s | VRAM Pico: {peak_vram:.1f} MB")
    print(f"    - Top-1 Acc: {best_metrics.get('top1_acc', 0)*100:.2f}%")
    print(f"    - Top-5 Acc: {best_metrics.get('top5_acc', 0)*100:.2f}%")
    print(f"    - Macro-F1:  {best_metrics.get('macro_f1', 0)*100:.2f}%")

    # 6. Log to Tracker
    tracker = ExperimentTracker(log_dir=str(ROOT_DIR / "logs"))
    tracker.log_experiment(
        exp_id=exp_cfg["id"],
        model_name=f"{exp_cfg['name']} (10k Classes)",
        optimizer=f"{exp_cfg['optimizer']} (lr={lr})",
        regularization="LayerNorm" if "swin" in exp_cfg["arch"] or "convnext" in exp_cfg["arch"] else "BatchNorm",
        augmentation=f"Standard ({img_size}px)",
        transfer_learning="Pretrained" if pretrained else "Scratch (Random Init)",
        long_tail="Focal Loss" if loss_type == "focal_loss" else "No",
        metrics=best_metrics,
        training_time_sec=elapsed,
        peak_vram_mb=peak_vram,
        param_count_m=total_m,
        extra_metadata={"stage": 2, "num_classes": num_classes, "epochs": epochs}
    )

    # Update Markdown Table
    tracker.save_markdown_table(str(ROOT_DIR / "outputs" / "tabla_final_experimentos.md"), sort_by="macro_f1", ascending=False)
    tracker.save_json(str(ROOT_DIR / "outputs" / "tabla_final_experimentos.json"), sort_by="macro_f1", ascending=False)

    return best_metrics


def main():
    parser = argparse.ArgumentParser(description="Etapa 2: Scale-Up a 10,000 clases")
    parser.add_argument("--pilot_only", action="store_true", help="Correr únicamente la primera prueba piloto (Modelo Campeón)")
    parser.add_argument("--batch_size", type=int, default=32, help="Tamaño de lote")
    parser.add_argument("--data_dir", type=str, default="../recursos", help="Directorio con anotaciones y datos")
    args = parser.parse_args()

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = Path(args.data_dir).resolve()
    val_json = data_dir / "val.json"
    train_json = data_dir / "train_mini.json"

    val_img_dir = data_dir / "val"
    if not val_img_dir.exists():
        val_img_dir = data_dir / "inat_val" / "val"

    train_img_dir = data_dir / "train_mini"

    data_paths = {
        "val_json": val_json,
        "train_json": train_json,
        "val_img_dir": val_img_dir,
        "train_img_dir": train_img_dir
    }

    print("=" * 75)
    print("  BIODIVERSITY AT SCALE: ETAPA 2 (SCALE-UP A 10,000 CLASES)")
    print(f"  Dispositivo GPU: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Modo:            {'Piloto (Modelo Campeón)' if args.pilot_only else 'Suite Completa (10 Modelos)'}")
    print("=" * 75)

    experiments_to_run = [STAGE2_MODELS[0]] if args.pilot_only else STAGE2_MODELS

    for exp in experiments_to_run:
        run_experiment(exp, data_paths, device=device, num_classes=10000, batch_size=args.batch_size)

    print("\n[🎉] ¡Etapa 2 concluida exitosamente! Todos los resultados han sido registrados en outputs/.")


if __name__ == "__main__":
    main()
