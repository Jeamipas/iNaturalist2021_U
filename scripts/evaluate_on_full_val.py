"""
Evaluation script to benchmark trained models on the validation dataset (Sample or Full 100k images).

Accelerations:
- Automatic Mixed Precision (AMP fp16)
- Memory format channels_last (NHWC)
- Multi-worker DataLoader with pin_memory=True
- GPU Tensor evaluation with batch-level metrics

Usage:
  python scripts/evaluate_on_full_val.py --checkpoint checkpoints/best_E8_ResNet18_PartialFT_AdamW.pt
"""

import argparse
import os
import sys
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import INatDataset
from src.data.transforms import get_val_transforms
from src.data.dataloader import create_eval_loader
from src.models.factory import create_model
from src.utils.metrics import compute_classification_metrics
from tqdm import tqdm


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool = True
):
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    start_time = time.time()
    pbar = tqdm(dataloader, desc="Evaluando en Validación", unit="batch")

    with torch.no_grad():
        for images, targets in pbar:
            if device.type == "cuda" and images.ndim == 4:
                images = images.to(device, memory_format=torch.channels_last, non_blocking=True)
            else:
                images = images.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)

            preds = outputs.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            all_probs.append(probs.cpu())

    total_time = time.time() - start_time
    all_preds_t = torch.cat(all_preds)
    all_targets_t = torch.cat(all_targets)
    all_probs_t = torch.cat(all_probs)

    metrics = compute_classification_metrics(all_targets_t, all_preds_t, y_probs=all_probs_t)

    images_per_sec = len(all_preds_t) / max(0.001, total_time)
    metrics["eval_time_sec"] = total_time
    metrics["images_per_sec"] = images_per_sec
    metrics["total_samples"] = len(all_preds_t)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluar modelo en dataset de validación.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_E8_ResNet18_PartialFT_AdamW.pt", help="Ruta al checkpoint .pt")
    parser.add_argument("--data_dir", type=str, default="../recursos/sample_inat", help="Directorio raíz de datos")
    parser.add_argument("--batch_size", type=int, default=64, help="Tamaño de lote para evaluación")
    parser.add_argument("--num_workers", type=int, default=4, help="Trabajadores DataLoader")
    parser.add_argument("--no_amp", action="store_true", help="Desactivar AMP fp16")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("  Evaluación de Modelo en Conjunto de Validación")
    print(f"  Dispositivo: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Checkpoint:  {args.checkpoint}")
    print(f"  Datos:        {args.data_dir}")
    print("=" * 70)

    val_json = Path(args.data_dir) / "val.json"
    val_img_dir = Path(args.data_dir) / "val"

    if not val_json.exists():
        print(f"[ERROR] No se encontró el archivo de anotaciones: {val_json}")
        return

    # Load dataset
    print("[*] Cargando dataset de validación...")
    val_dataset = INatDataset(
        json_path=str(val_json),
        images_root=str(val_img_dir),
        transform=get_val_transforms(img_size=224),
        cache_in_ram=False
    )
    print(f"    Total de imágenes de validación: {len(val_dataset):,}")
    print(f"    Total de clases: {len(val_dataset.category_to_label):,}")

    val_loader = create_eval_loader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Load checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[!] Checkpoint {ckpt_path} no encontrado. Creando modelo de prueba...")
        num_classes = len(val_dataset.category_to_label)
        model = create_model("resnet18", num_classes=num_classes, pretrained=True)
    else:
        print(f"[*] Cargando pesos desde {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        num_classes = len(val_dataset.category_to_label)
        model = create_model("resnet18", num_classes=num_classes, pretrained=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        else:
            model.load_state_dict(ckpt)

    model = model.to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    # Run evaluation
    metrics = evaluate_model(model, val_loader, device=device, use_amp=not args.no_amp)

    print("\n" + "=" * 70)
    print("  RESULTADOS DE EVALUACIÓN OFICIAL")
    print("=" * 70)
    print(f"  Muestras Evaluadas: {metrics['total_samples']:,}")
    print(f"  Tiempo Total:       {metrics['eval_time_sec']:.2f} s ({metrics['images_per_sec']:.1f} imgs/sec)")
    print(f"  Top-1 Accuracy:     {metrics['top1_acc'] * 100:.2f}%")
    print(f"  Top-5 Accuracy:     {metrics['top5_acc'] * 100:.2f}%")
    print(f"  Macro F1-Score:     {metrics['macro_f1'] * 100:.2f}%")
    print(f"  Weighted F1-Score:  {metrics['weighted_f1'] * 100:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
