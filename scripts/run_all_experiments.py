"""
Master runner script: Executes a complete first iteration of all project experiments (E1 to E11 + SOTA).
Adheres strictly to the rubric of Dra. Aurea Soriano-Vargas.
Optimized for 4GB VRAM with AMP, channels_last, and CUDA.
"""

import sys
import time
from pathlib import Path
import torch

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


def main():
    print("=" * 75)
    print("  iNaturalist 2021: Ronda Maestra de Experimentos (E1 - E11 + SOTA)")
    print("=" * 75)

    SEED = 42
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Dispositivo de cómputo: {device}")
    if device.type == "cuda":
        print(f"[*] GPU detectada: {torch.cuda.get_device_name(0)}")

    # 1. Rutas al dataset real de muestra en recursos/
    data_dir = ROOT_DIR.parent / "recursos" / "sample_inat"
    train_json = data_dir / "train_mini.json"
    val_json = data_dir / "val.json"
    train_img_dir = data_dir / "train_mini"
    val_img_dir = data_dir / "val"

    if not (train_json.exists() and train_img_dir.exists()):
        raise FileNotFoundError(f"Dataset de muestra no encontrado en {data_dir}")

    # 2. Datasets base
    IMG_SIZE = 128  # Resolución ágil para ronda preliminar
    EPOCHS = 3
    BATCH_SIZE = 32

    tf_train_std = get_transforms(split="train", img_size=IMG_SIZE, aug_mode="standard")
    tf_train_none = get_transforms(split="train", img_size=IMG_SIZE, aug_mode="none")
    tf_train_full = get_transforms(split="train", img_size=IMG_SIZE, aug_mode="full")
    tf_val = get_transforms(split="val", img_size=IMG_SIZE, aug_mode="none")

    train_base = INatDataset(train_json, train_img_dir, transform=tf_train_std)
    val_base = INatDataset(val_json, val_img_dir, transform=tf_val, category_to_label=train_base.category_to_label)

    NUM_CLASSES = len(train_base.category_to_label)
    print(f"[*] Dataset cargado: {len(train_base)} train | {len(val_base)} val | {NUM_CLASSES} clases.")

    tracker = ExperimentTracker(log_dir=str(ROOT_DIR / "logs"))
    criterion_ce = build_criterion("cross_entropy")

    # Helper para ejecutar un experimento
    def run_exp(exp_id, model, opt, criterion, train_ds, val_ds, model_name, opt_name, reg_name, aug_name, tl_name, lt_name, batch_aug=None, class_tiers=None):
        print(f"\n>>> [{exp_id}] {model_name} | Opt: {opt_name} | Reg: {reg_name} | Aug: {aug_name} | TL: {tl_name} | LT: {lt_name}")
        train_l, val_l = build_dataloaders(train_ds, val_ds, batch_size=BATCH_SIZE, num_workers=2, seed=SEED)
        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=opt,
            device=device,
            use_amp=True,
            class_tiers=class_tiers,
            batch_augmenter=batch_aug
        )
        history, best_metrics, peak_vram, total_time = trainer.fit(train_l, val_l, epochs=EPOCHS, verbose=True)
        _, _, total_m, _ = count_parameters(model)

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
        return record

    # =========================================================================
    # E1: Baseline MLP (Sección 6)
    # =========================================================================
    train_mlp_ds = INatDataset(train_json, train_img_dir, transform=get_transforms("train", img_size=32, aug_mode="none"))
    val_mlp_ds = INatDataset(val_json, val_img_dir, transform=get_transforms("val", img_size=32, aug_mode="none"), category_to_label=train_mlp_ds.category_to_label)
    mlp_model = build_model("mlp", num_classes=NUM_CLASSES, input_shape=(3, 32, 32), hidden_dims=[512, 256])
    mlp_opt = build_optimizer(mlp_model, opt_type="adam", lr=1e-3)
    run_exp("E1", mlp_model, mlp_opt, criterion_ce, train_mlp_ds, val_mlp_ds, "MLP Baseline (32x32)", "Adam (lr=1e-3)", "None", "None", "No", "No")

    # =========================================================================
    # E2: CNN A (ResNet-18 desde cero) (Sección 7)
    # =========================================================================
    cnn_a = build_model("resnet18", num_classes=NUM_CLASSES, pretrained=False)
    opt_a = build_optimizer(cnn_a, opt_type="adamw", lr=1e-3)
    run_exp("E2", cnn_a, opt_a, criterion_ce, train_base, val_base, "ResNet-18 (Scratch)", "AdamW (lr=1e-3)", "BN", "Standard", "No", "No")

    # =========================================================================
    # E3: CNN B (MiniINatCNN personalizada) (Sección 7)
    # =========================================================================
    cnn_b = build_model("cnn_custom", num_classes=NUM_CLASSES, use_batchnorm=True, dropout_rate=0.2)
    opt_b = build_optimizer(cnn_b, opt_type="adamw", lr=1e-3)
    run_exp("E3", cnn_b, opt_b, criterion_ce, train_base, val_base, "MiniINatCNN", "AdamW (lr=1e-3)", "BN+Dropout", "Standard", "No", "No")

    # =========================================================================
    # E4: Optimizador SGD + Momentum (Sección 8)
    # =========================================================================
    model_e4 = build_model("resnet18", num_classes=NUM_CLASSES, pretrained=False)
    opt_e4 = build_optimizer(model_e4, opt_type="sgd_momentum", lr=1e-2, weight_decay=1e-4)
    run_exp("E4", model_e4, opt_e4, criterion_ce, train_base, val_base, "ResNet-18", "SGD+Momentum (lr=1e-2)", "BN", "Standard", "No", "No")

    # =========================================================================
    # E5: Optimizador AdamW (Sección 8)
    # =========================================================================
    model_e5 = build_model("resnet18", num_classes=NUM_CLASSES, pretrained=False)
    opt_e5 = build_optimizer(model_e5, opt_type="adamw", lr=1e-3, weight_decay=1e-4)
    run_exp("E5", model_e5, opt_e5, criterion_ce, train_base, val_base, "ResNet-18", "AdamW (lr=1e-3)", "BN", "Standard", "No", "No")

    # =========================================================================
    # E6: Ablation Study (Sección 9.4)
    # =========================================================================
    # E6.1: Sin regularización ni augmentations
    m_abl1 = build_model("cnn_custom", num_classes=NUM_CLASSES, use_batchnorm=False, dropout_rate=0.0)
    opt_abl1 = build_optimizer(m_abl1, opt_type="adamw", lr=1e-3, weight_decay=0.0)
    train_abl_none = INatDataset(train_json, train_img_dir, transform=tf_train_none)
    run_exp("E6_Abl1", m_abl1, opt_abl1, criterion_ce, train_abl_none, val_base, "MiniINatCNN", "AdamW", "No BN, No Drop", "None", "No", "No")

    # E6.2: + BatchNorm + Dropout + Data Augmentation + Weight Decay (Completo)
    m_abl2 = build_model("cnn_custom", num_classes=NUM_CLASSES, use_batchnorm=True, dropout_rate=0.3)
    opt_abl2 = build_optimizer(m_abl2, opt_type="adamw", lr=1e-3, weight_decay=1e-4)
    train_abl_full = INatDataset(train_json, train_img_dir, transform=tf_train_full)
    run_exp("E6_AblFull", m_abl2, opt_abl2, criterion_ce, train_abl_full, val_base, "MiniINatCNN", "AdamW", "BN+Drop+WD", "Full Aug", "No", "No")

    # =========================================================================
    # E7: Transfer Learning - Feature Extraction (Sección 10.1)
    # =========================================================================
    model_e7 = build_model("resnet18", num_classes=NUM_CLASSES, mode="feature_extraction", pretrained=True)
    opt_e7 = build_optimizer(model_e7, opt_type="adamw", lr=1e-3)
    run_exp("E7", model_e7, opt_e7, criterion_ce, train_base, val_base, "ResNet-18 (Pretrained)", "AdamW", "BN+Dropout", "Standard", "Frozen Backbone", "No")

    # =========================================================================
    # E8: Transfer Learning - Partial Fine-Tuning (Sección 10.2)
    # =========================================================================
    model_e8 = build_model("resnet18", num_classes=NUM_CLASSES, mode="partial_fine_tuning", pretrained=True)
    param_groups_e8 = model_e8.get_parameter_groups(backbone_lr=1e-4, head_lr=1e-3)
    opt_e8 = build_optimizer(model_e8, opt_type="adamw", param_groups=param_groups_e8)
    run_exp("E8", model_e8, opt_e8, criterion_ce, train_base, val_base, "ResNet-18 (Pretrained)", "AdamW (Diff LR)", "BN+Dropout", "Standard", "Partial FT (layer4)", "No")

    # =========================================================================
    # E9: Transfer Learning - Full Fine-Tuning (Sección 10.3)
    # =========================================================================
    model_e9 = build_model("resnet18", num_classes=NUM_CLASSES, mode="full_fine_tuning", pretrained=True)
    param_groups_e9 = model_e9.get_parameter_groups(backbone_lr=1e-5, head_lr=1e-3)
    opt_e9 = build_optimizer(model_e9, opt_type="adamw", param_groups=param_groups_e9)
    run_exp("E9", model_e9, opt_e9, criterion_ce, train_base, val_base, "ResNet-18 (Pretrained)", "AdamW (Diff LR)", "BN+Dropout", "Standard", "Full Fine-Tuning", "No")

    # =========================================================================
    # E10 y E11: Biodiversity Long-Tail Challenge (Sección 11)
    # =========================================================================
    lt_classes = list(range(NUM_CLASSES))
    lt_indices, _, class_tiers = create_long_tail_subset(
        train_base,
        selected_classes=lt_classes,
        many_shot_count=20,
        med_shot_count=8,
        few_shot_count=2,
        many_ratio=0.25,
        med_ratio=0.35,
        seed=SEED
    )
    from torch.utils.data import Subset
    train_lt = Subset(train_base, lt_indices)

    # E10: Long-Tail sin corrección
    model_e10 = build_model("resnet18", num_classes=NUM_CLASSES, mode="partial_fine_tuning", pretrained=True)
    opt_e10 = build_optimizer(model_e10, opt_type="adamw", lr=1e-3)
    run_exp("E10", model_e10, opt_e10, criterion_ce, train_lt, val_base, "ResNet-18 (Partial FT)", "AdamW", "BN+Dropout", "Standard", "Partial FT", "Long-Tail (Sin Mitigación)", class_tiers=class_tiers)

    # E11: Long-Tail con mitigación (Focal Loss)
    focal_loss = build_criterion("focal_loss", gamma=2.0)
    model_e11 = build_model("resnet18", num_classes=NUM_CLASSES, mode="partial_fine_tuning", pretrained=True)
    opt_e11 = build_optimizer(model_e11, opt_type="adamw", lr=1e-3)
    run_exp("E11", model_e11, opt_e11, focal_loss, train_lt, val_base, "ResNet-18 (Partial FT)", "AdamW", "BN+Dropout", "Standard", "Partial FT", "Long-Tail (Focal Loss)", class_tiers=class_tiers)

    # =========================================================================
    # E12 (BONUS SOTA): Muon Optimizer (Newton-Schulz)
    # =========================================================================
    model_e12 = build_model("resnet18", num_classes=NUM_CLASSES, pretrained=False)
    opt_e12 = build_optimizer(model_e12, opt_type="muon", lr=0.02)
    run_exp("E12", model_e12, opt_e12, criterion_ce, train_base, val_base, "ResNet-18 (Scratch)", "Muon (Newton-Schulz)", "BN", "Standard", "No", "No")

    # =========================================================================
    # E13 (BONUS SOTA): CutMix / MixUp Data Mixing
    # =========================================================================
    batch_aug = get_batch_augmentations(num_classes=NUM_CLASSES, mode="cutmix_or_mixup")
    model_e13 = build_model("resnet18", num_classes=NUM_CLASSES, pretrained=False)
    opt_e13 = build_optimizer(model_e13, opt_type="adamw", lr=1e-3)
    run_exp("E13", model_e13, opt_e13, criterion_ce, train_base, val_base, "ResNet-18", "AdamW", "BN", "CutMix/MixUp (GPU)", "No", "No", batch_aug=batch_aug)

    # =========================================================================
    # Resumen y Tabla Final Oficial (Sección 14)
    # =========================================================================
    print("\n" + "=" * 80)
    print("  TABLA FINAL DE EXPERIMENTOS (Sección 14 de la Rúbrica Oficial)")
    print("=" * 80)
    md_table = tracker.to_markdown_table()
    print(md_table)

    # Guardar en archivo Markdown y CSV
    out_path = ROOT_DIR / "outputs" / "tabla_final_experimentos.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Tabla Final de Experimentos (iNaturalist 2021)\n\n" + md_table + "\n")

    csv_path = ROOT_DIR / "outputs" / "tabla_final_experimentos.csv"
    tracker.to_dataframe().to_csv(csv_path, index=False)
    print(f"\n[OK] Tabla guardada exitosamente en: {out_path} y {csv_path}")


if __name__ == "__main__":
    main()
