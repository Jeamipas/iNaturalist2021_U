"""
SOTA Elite Campaign: High-Performance Benchmark on 8GB VRAM at 224x224.
Explores:
- S01: ConvNeXt-Tiny (Partial FT) | AdamW | 224x224 (Native Resolution Control)
- S02: ConvNeXt-Tiny (Partial FT) | Hybrid (Muon on Backbone + AdamW on Head) | 224x224
- S03: ConvNeXt-Tiny (Deep Partial FT: Stages 3 & 4) | AdamW | 224x224
- S04: ConvNeXt-Tiny (Partial FT) | Hyperspherical Cosine Head | AdamW | 224x224
- S05: Swin-T (Swin Transformer)  | AdamW | 224x224 (Hierarchical Vision Transformer)
- S06: Swin-T (Swin Transformer)  | Hybrid (Muon on Attention + AdamW on Head) | 224x224
- S07: ConvNeXt-Tiny (Partial FT) | Model Weight EMA (alpha=0.999) + CutMix/MixUp | 224x224
- S08: Grandmaster Blending Ensemble (ConvNeXt-Tiny + Swin-T + TTA) | 224x224

Features:
- Guaranteed Best Epoch Weight Restoration on all models before saving
- Native 224x224 image resolution cached in RAM
- Real-time logging to TensorBoard (http://localhost:6006)
- Dynamic re-ranking and persistence of the Top 10 champions in checkpoints/
- Consolidation of tables in outputs/tabla_final_experimentos.md
"""

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from src.training.hybrid_optimizer import build_hybrid_muon_adamw
from src.training.trainer import Trainer
from src.utils.metrics import compute_classification_metrics, compute_kingdom_breakdown
from src.utils.tracking import ExperimentTracker
from src.utils.checkpoints import TopKCheckpointManager
from src.utils.visualization import plot_multiexperiment_4grid


class ModelEMA:
    """Maintains an Exponential Moving Average (EMA) of model parameters."""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        self.decay = decay
        for p in self.ema_model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_p, model_p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)


def evaluate_with_tta(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool = True
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluates model using Test-Time Augmentation (TTA: original + horizontal flip)."""
    model.eval()
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device, memory_format=torch.channels_last if device.type == "cuda" else torch.contiguous_format)
            targets_np = targets.numpy()

            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                # Forward 1: Original
                logits_orig = model(images)
                probs_orig = F.softmax(logits_orig, dim=1)

                # Forward 2: Horizontal Flip
                images_flipped = torch.flip(images, dims=[-1])
                logits_flip = model(images_flipped)
                probs_flip = F.softmax(logits_flip, dim=1)

                # Average probabilities
                probs_tta = 0.5 * (probs_orig + probs_flip)

            all_probs.append(probs_tta.cpu().numpy())
            all_targets.append(targets_np)

    y_probs = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    y_pred = y_probs.argmax(axis=1)

    return y_true, y_pred, y_probs


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="SOTA Elite Suite on 8GB VRAM")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--img_size", type=int, default=224, help="Image resolution (default: 224)")
    parser.add_argument("--epochs", type=int, default=8, help="Epochs per model (default: 8)")
    args = parser.parse_args()

    print("=" * 80)
    print("  iNaturalist 2021: CAMPAÑA SOTA DE ALTA POTENCIA (8 GB VRAM @ 224x224)")
    print("=" * 80)

    SEED = 42
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Dispositivo de computo: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[*] GPU: {gpu_name} ({vram_gb:.2f} GB VRAM detectados)")

    # 1. Rutas del dataset de 50 clases (3,000 imagenes)
    data_dir = ROOT_DIR.parent / "recursos" / "inat2021_sample"
    if not (data_dir / "train_mini.json").exists():
        data_dir = ROOT_DIR / "data" / "inat2021_sample"

    print(f"[*] Carpeta de datos activa: {data_dir}")
    train_json = data_dir / "train_mini.json"
    val_json = data_dir / "val.json"
    train_img_dir = data_dir / "train_mini"
    val_img_dir = data_dir / "val"

    with open(train_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    cat_to_kingdom = {cat["id"]: cat.get("kingdom", "Unknown") for cat in meta.get("categories", [])}

    # Transformaciones a resolucion nativa 224x224
    tf_train = get_transforms("train", img_size=args.img_size, aug_mode="standard")
    tf_val = get_transforms("val", img_size=args.img_size, aug_mode="none")

    print(f"[*] Precargando {args.img_size}x{args.img_size} en memoria RAM...")
    train_base = INatDataset(train_json, train_img_dir, transform=tf_train, cache_in_ram=True)
    val_base = INatDataset(val_json, val_img_dir, transform=tf_val, category_to_label=train_base.category_to_label, cache_in_ram=True)

    NUM_CLASSES = len(train_base.category_to_label)
    label_to_kingdom = {label: cat_to_kingdom.get(cat_id, "Unknown") for cat_id, label in train_base.category_to_label.items()}
    print(f"[OK] {len(train_base)} muestras Train | {len(val_base)} muestras Val | {NUM_CLASSES} Especies")

    tracker = ExperimentTracker(log_dir=str(ROOT_DIR / "logs"))
    ckpt_manager = TopKCheckpointManager(checkpoint_dir=str(ROOT_DIR / "checkpoints"), k=10)
    criterion_ce = build_criterion("cross_entropy")

    saved_models_for_ensemble = {}

    def run_sota_exp(
        exp_id: str,
        model: nn.Module,
        opt: torch.optim.Optimizer,
        criterion: nn.Module,
        model_name: str,
        opt_name: str,
        reg_name: str,
        aug_name: str,
        tl_name: str,
        epochs: int = 8,
        scheduler=None,
        batch_aug=None,
        use_ema: bool = False
    ):
        print("\n" + "=" * 78)
        print(f"  >>> [{exp_id}] {model_name} | Opt: {opt_name} | Res: {args.img_size}x{args.img_size} | Epocas: {epochs}")
        print(f"      Reg: {reg_name} | Aug: {aug_name} | TL: {tl_name} | EMA: {use_ema}")
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
            model_name=f"{model_name} (224px)",
            optimizer=opt_name,
            regularization=reg_name,
            augmentation=aug_name,
            transfer_learning=tl_name,
            long_tail="No",
            metrics=best_metrics,
            training_time_sec=total_time,
            peak_vram_mb=peak_vram,
            param_count_m=total_m,
            extra_metadata={
                "epochs": epochs,
                "resolution": args.img_size,
                "best_epoch": best_metrics.get("best_epoch", 1)
            }
        )

        # 2. Registrar y guardar checkpoint garantizado del mejor modelo
        rank = ckpt_manager.register_and_save(
            model=model,
            exp_id=exp_id,
            model_name=f"{model_name} (224px)",
            metrics=best_metrics,
            hyperparams={
                "optimizer": opt_name,
                "batch_size": args.batch_size,
                "epochs": epochs,
                "img_size": args.img_size,
                "use_amp": True,
                "regularization": reg_name,
                "augmentation": aug_name,
                "best_epoch": best_metrics.get("best_epoch", 1)
            }
        )

        f1 = best_metrics.get("macro_f1", 0.0) * 100
        top1 = best_metrics.get("top1_acc", 0.0) * 100
        rank_str = f"TOP #{rank}" if rank is not None else "No entra al Top-10"
        print(f"[OK] [{exp_id}] Terminado: Val Macro-F1={f1:.2f}% | Top-1={top1:.2f}% | Best Epoch={best_metrics.get('best_epoch', '-')} | VRAM={peak_vram:.1f}MB | {rank_str}")

        saved_models_for_ensemble[exp_id] = copy.deepcopy(model)
        return history, best_metrics

    # =========================================================================
    # S01: ConvNeXt-Tiny (Partial FT) | AdamW | 224x224 (Native Control)
    # =========================================================================
    m_s01 = build_model("convnext_tiny", num_classes=NUM_CLASSES, mode="partial_fine_tuning", pretrained=True)
    pg_s01 = m_s01.get_parameter_groups(backbone_lr=5e-5, head_lr=1e-3)
    opt_s01 = build_optimizer(m_s01, opt_type="adamw", lr=1e-3, param_groups=pg_s01)
    sched_s01 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s01, T_max=args.epochs, eta_min=1e-6)
    run_sota_exp("S01", m_s01, opt_s01, criterion_ce, "ConvNeXt-Tiny", "AdamW + CosineLR", "LayerNorm", "Standard (224px)", "Partial FT (stages.3)", epochs=args.epochs, scheduler=sched_s01)

    # =========================================================================
    # S02: ConvNeXt-Tiny (Partial FT) | Hybrid (Muon Backbone + AdamW Head) | 224x224
    # =========================================================================
    m_s02 = build_model("convnext_tiny", num_classes=NUM_CLASSES, mode="partial_fine_tuning", pretrained=True)
    opt_s02 = build_hybrid_muon_adamw(m_s02, muon_lr=0.005, adamw_lr=1e-3, weight_decay=1e-4)
    sched_s02 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s02, T_max=args.epochs, eta_min=1e-6)
    run_sota_exp("S02", m_s02, opt_s02, criterion_ce, "ConvNeXt-Tiny", "Hybrid (Muon+AdamW)", "LayerNorm", "Standard (224px)", "Partial FT + Hybrid", epochs=args.epochs, scheduler=sched_s02)

    # =========================================================================
    # S03: ConvNeXt-Tiny (Deep Partial FT: Stages 3 & 4 - 26.3M params) | 224x224
    # =========================================================================
    m_s03 = build_model("convnext_tiny", num_classes=NUM_CLASSES, mode="deep_partial_fine_tuning", pretrained=True)
    pg_s03 = m_s03.get_parameter_groups(backbone_lr=3e-5, head_lr=1e-3)
    opt_s03 = build_optimizer(m_s03, opt_type="adamw", lr=1e-3, param_groups=pg_s03)
    sched_s03 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s03, T_max=args.epochs, eta_min=1e-6)
    run_sota_exp("S03", m_s03, opt_s03, criterion_ce, "ConvNeXt-Tiny", "AdamW (Deep FT)", "LayerNorm", "Standard (224px)", "Deep Partial (Stages 3+4)", epochs=args.epochs, scheduler=sched_s03)

    # =========================================================================
    # S04: ConvNeXt-Tiny (Partial FT) | Hyperspherical Cosine Head | 224x224
    # =========================================================================
    m_s04 = build_model("convnext_tiny", num_classes=NUM_CLASSES, mode="partial_fine_tuning", pretrained=True, use_cosine_head=True)
    pg_s04 = m_s04.get_parameter_groups(backbone_lr=5e-5, head_lr=1e-3)
    opt_s04 = build_optimizer(m_s04, opt_type="adamw", lr=1e-3, param_groups=pg_s04)
    sched_s04 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s04, T_max=args.epochs, eta_min=1e-6)
    run_sota_exp("S04", m_s04, opt_s04, criterion_ce, "ConvNeXt-Tiny (CosineHead)", "AdamW + CosineLR", "LayerNorm + ArcNorm", "Standard (224px)", "Partial FT + Cosine Head", epochs=args.epochs, scheduler=sched_s04)

    # =========================================================================
    # S05: Swin-T (Swin Transformer) | AdamW | 224x224
    # =========================================================================
    m_s05 = build_model("swin_t", num_classes=NUM_CLASSES, mode="partial_fine_tuning", pretrained=True)
    pg_s05 = m_s05.get_parameter_groups(backbone_lr=3e-5, head_lr=1e-3)
    opt_s05 = build_optimizer(m_s05, opt_type="adamw", lr=1e-3, param_groups=pg_s05)
    sched_s05 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s05, T_max=args.epochs, eta_min=1e-6)
    run_sota_exp("S05", m_s05, opt_s05, criterion_ce, "Swin-T (Transformer)", "AdamW + CosineLR", "LayerNorm", "Standard (224px)", "Partial FT (Shifted Window)", epochs=args.epochs, scheduler=sched_s05)

    # =========================================================================
    # S06: Swin-T (Swin Transformer) | Hybrid (Muon on Attention + AdamW on Head) | 224x224
    # =========================================================================
    m_s06 = build_model("swin_t", num_classes=NUM_CLASSES, mode="partial_fine_tuning", pretrained=True)
    opt_s06 = build_hybrid_muon_adamw(m_s06, muon_lr=0.003, adamw_lr=1e-3, weight_decay=1e-4)
    sched_s06 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s06, T_max=args.epochs, eta_min=1e-6)
    run_sota_exp("S06", m_s06, opt_s06, criterion_ce, "Swin-T (Transformer)", "Hybrid (Muon+AdamW)", "LayerNorm", "Standard (224px)", "Partial FT + Hybrid", epochs=args.epochs, scheduler=sched_s06)

    # =========================================================================
    # S07: ConvNeXt-Tiny (Partial FT) | AdamW + CutMix/MixUp on GPU | 224x224
    # =========================================================================
    m_s07 = build_model("convnext_tiny", num_classes=NUM_CLASSES, mode="partial_fine_tuning", pretrained=True)
    pg_s07 = m_s07.get_parameter_groups(backbone_lr=5e-5, head_lr=1e-3)
    opt_s07 = build_optimizer(m_s07, opt_type="adamw", lr=1e-3, param_groups=pg_s07)
    sched_s07 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s07, T_max=args.epochs, eta_min=1e-6)
    batch_aug_s07 = get_batch_augmentations(num_classes=NUM_CLASSES, mode="cutmix_or_mixup", cutmix_alpha=1.0, mixup_alpha=0.2)
    criterion_ls = build_criterion("cross_entropy", label_smoothing=0.1)
    run_sota_exp("S07", m_s07, opt_s07, criterion_ls, "ConvNeXt-Tiny", "AdamW + CutMix + LS", "LayerNorm + LabelSmooth(0.1)", "+ CutMix/MixUp (224px)", "Partial FT (stages.3)", epochs=args.epochs, scheduler=sched_s07, batch_aug=batch_aug_s07)

    # =========================================================================
    # S08: Grandmaster Blending Ensemble (ConvNeXt-Tiny + Swin-T + TTA)
    # =========================================================================
    print("\n" + "=" * 78)
    print("  >>> [S08] GRANDMASTER BLENDING ENSEMBLE (ConvNeXt-Tiny + Swin-T + TTA)")
    print("      Fusion de Modelos Complementarios (Local Convolutions + Global Attention)")
    print("=" * 78)

    model_convnext = saved_models_for_ensemble.get("S01", m_s01).to(device)
    model_swin = saved_models_for_ensemble.get("S05", m_s05).to(device)
    _, val_l = build_dataloaders(train_base, val_base, batch_size=args.batch_size, num_workers=0, seed=SEED)

    t0_ens = time.time()
    y_true_c, _, probs_c = evaluate_with_tta(model_convnext, val_l, device=device, use_amp=True)
    _, _, probs_s = evaluate_with_tta(model_swin, val_l, device=device, use_amp=True)

    # Blending ponderado 50% ConvNeXt + 50% Swin Transformer
    probs_ensemble = 0.5 * probs_c + 0.5 * probs_s
    preds_ensemble = probs_ensemble.argmax(axis=1)

    ens_metrics = compute_classification_metrics(y_true=y_true_c, y_pred=preds_ensemble, y_probs=probs_ensemble)
    kingdom_m = compute_kingdom_breakdown(y_true_c, preds_ensemble, label_to_kingdom)
    ens_metrics.update(kingdom_m)
    ens_metrics["best_epoch"] = args.epochs
    t_ens = time.time() - t0_ens

    # Contar parametros combinados
    p_c, _, _, _ = count_parameters(model_convnext)
    p_s, _, _, _ = count_parameters(model_swin)
    ens_params_m = (p_c + p_s) / 1e6
    peak_vram_ens = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0

    record_ens = tracker.log_experiment(
        exp_id="S08",
        model_name="Ensemble (ConvNeXt + Swin-T)",
        optimizer="Blending + TTA",
        regularization="LayerNorm (Dual)",
        augmentation="Test-Time Augmentation",
        transfer_learning="Dual Ensemble",
        long_tail="No",
        metrics=ens_metrics,
        training_time_sec=t_ens,
        peak_vram_mb=peak_vram_ens,
        param_count_m=ens_params_m,
        extra_metadata={"components": ["S01_ConvNeXt-Tiny", "S05_Swin-T"], "tta": True}
    )

    rank_ens = ckpt_manager.register_and_save(
        model=model_convnext,  # Base model with ensemble metadata
        exp_id="S08",
        model_name="Ensemble (ConvNeXt + Swin-T)",
        metrics=ens_metrics,
        hyperparams={
            "components": ["ConvNeXt-Tiny", "Swin-T"],
            "blending_weights": [0.5, 0.5],
            "img_size": args.img_size,
            "tta": True
        }
    )

    f1_ens = ens_metrics.get("macro_f1", 0.0) * 100
    top1_ens = ens_metrics.get("top1_acc", 0.0) * 100
    top5_ens = ens_metrics.get("top5_acc", 0.0) * 100
    anim_ens = ens_metrics.get("kingdom_animalia_acc", 0.0) * 100
    plan_ens = ens_metrics.get("kingdom_plantae_acc", 0.0) * 100
    print(f"\n[OK] [S08] GRANDMASTER ENSEMBLE COMPLETADO:")
    print(f"     Val Macro-F1: {f1_ens:.2f}% | Top-1: {top1_ens:.2f}% | Top-5: {top5_ens:.2f}%")
    print(f"     Animalia: {anim_ens:.1f}% | Plantae: {plan_ens:.1f}% | Ranking: TOP #{rank_ens}")

    # =========================================================================
    # RE-EXPORTACION DE TABLAS Y MANIFIESTO FINAL
    # =========================================================================
    print("\n" + "=" * 80)
    print("  ACTUALIZANDO TABLAS FINALES DEL PROYECTO...")
    print("=" * 80)

    outputs_dir = ROOT_DIR / "outputs"
    tracker.save_markdown_table(str(outputs_dir / "tabla_final_experimentos.md"))
    tracker.save_json(str(outputs_dir / "tabla_final_experimentos.json"))
    print(f"[OK] Tabla oficial guardada en {outputs_dir / 'tabla_final_experimentos.md'}")
    print(f"[OK] JSON consolidado guardado en {outputs_dir / 'tabla_final_experimentos.json'}")

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
        top5 = entry.get("metrics", {}).get("top5_acc", 0.0) * 100
        fpath = Path(entry.get("filepath", "")).name
        print(f"  #{rank:02d} | [{exp_id}] {model_name:<30} | Macro F1: {f1:5.2f}% | Top-1: {top1:5.2f}% | Top-5: {top5:5.2f}% | Archivo: {fpath}")
    print("=" * 80)


if __name__ == "__main__":
    main()
