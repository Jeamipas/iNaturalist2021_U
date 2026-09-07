"""
Export script to generate:
1. outputs/tabla_final_experimentos.md (Official Markdown Table per course rubric)
2. outputs/tabla_final_experimentos.json
3. outputs/comparativa_top_modelos_4grid.png (2x2 comparison grid across epochs and time)
"""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


from src.utils.tracking import ExperimentTracker

from src.utils.checkpoints import TopKCheckpointManager
from src.utils.visualization import plot_multiexperiment_4grid


def main():
    print("=" * 75)
    print("  GENERANDO ENTREGABLES FINALES DEL PROYECTO")
    print("=" * 75)

    outputs_dir = ROOT_DIR / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Exportar tablas finales
    tracker = ExperimentTracker(log_dir=str(ROOT_DIR / "logs"))
    tracker.save_markdown_table(str(outputs_dir / "tabla_final_experimentos.md"))
    tracker.save_json(str(outputs_dir / "tabla_final_experimentos.json"))
    print(f"[OK] Tabla oficial guardada en: {outputs_dir / 'tabla_final_experimentos.md'}")
    print(f"[OK] JSON consolidado guardado en: {outputs_dir / 'tabla_final_experimentos.json'}")

    # 2. Cargar historiales de TensorBoard para los modelos más representativos
    tb_dir = ROOT_DIR / "logs" / "tb_runs"
    runs_to_plot = [
        ("S06_Swin-T_(Transformer)_Hybrid", "S06: Swin-T | Hybrid Muon+AdamW (91.0%)"),
        ("S03_ConvNeXt-Tiny_AdamW", "S03: ConvNeXt-Tiny | Deep Partial FT (88.4%)"),
        ("S04_ConvNeXt-Tiny_(CosineHead)_AdamW", "S04: ConvNeXt-Tiny | Cosine Head (88.0%)"),
        ("S01_ConvNeXt-Tiny_AdamW", "S01: ConvNeXt-Tiny | 224px Native (87.4%)"),
        ("S05_Swin-T_(Transformer)_AdamW", "S05: Swin-T | AdamW (85.2%)"),
        ("CX02_ConvNeXt-Tiny_(Pretrained)_Adam", "CX02: ConvNeXt-Tiny | 128px Adam (79.2%)"),
        ("T05_ResNet-50_(Pretrained)_AdamW", "T05: ResNet-50 | 128px (68.0%)"),
        ("B01_MLP_(32x32)_Adam", "B01: MLP Baseline | 32px (9.8%)")
    ]

    histories = {}
    for folder_name, display_label in runs_to_plot:
        run_folder = tb_dir / folder_name
        if not run_folder.exists():
            continue

        try:
            ea = EventAccumulator(str(run_folder))
            ea.Reload()
            tags = ea.Tags().get("scalars", [])

            val_loss = [e.value for e in ea.Scalars("Loss/val")] if "Loss/val" in tags else []
            val_acc = [e.value / 100.0 for e in ea.Scalars("Accuracy/val_top1")] if "Accuracy/val_top1" in tags else []
            
            # Extract epoch times
            if "Time/Epoch_Sec" in tags:
                epoch_times = [e.value for e in ea.Scalars("Time/Epoch_Sec")]
            else:
                epoch_times = [1.0] * len(val_loss)

            if val_loss and val_acc:
                histories[display_label] = {
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "epoch_times": epoch_times
                }
        except Exception as e:
            print(f"[!] Error leyendo {folder_name}: {e}")

    # 3. Generar la gráfica cuádruple 2x2
    if histories:
        plot_path = str(outputs_dir / "comparativa_top_modelos_4grid.png")
        plot_multiexperiment_4grid(
            histories,
            title="Convergencia y Eficiencia del Benchmark (Pérdida y Precisión vs. Épocas y vs. Tiempo)",
            save_path=plot_path
        )
        print(f"[OK] Gráfico cuádruple 2x2 guardado en: {plot_path}")

    # 4. Mostrar resumen del Top 10
    ckpt_manager = TopKCheckpointManager(checkpoint_dir=str(ROOT_DIR / "checkpoints"), k=10)
    manifest = ckpt_manager.get_manifest()
    print("\n" + "=" * 75)
    print("  RANKING FINAL: TOP 10 MODELOS GUARDADOS EN DISCO")
    print("=" * 75)
    for entry in manifest:
        rank = entry.get("rank", "-")
        exp_id = entry.get("exp_id", "")
        model_name = entry.get("model_name", "")
        f1 = entry.get("metrics", {}).get("macro_f1", 0.0) * 100
        top1 = entry.get("metrics", {}).get("top1_acc", 0.0) * 100
        top5 = entry.get("metrics", {}).get("top5_acc", 0.0) * 100
        animalia = entry.get("metrics", {}).get("kingdom_animalia_acc", 0.0) * 100
        plantae = entry.get("metrics", {}).get("kingdom_plantae_acc", 0.0) * 100
        fpath = Path(entry.get("filepath", "")).name
        print(f"  #{rank:02d} | [{exp_id}] {model_name:<28} | Macro F1: {f1:5.2f}% | Top-1: {top1:5.2f}% | Top-5: {top5:5.2f}% | Animalia: {animalia:5.1f}% | Plantae: {plantae:5.1f}% | Archivo: {fpath}")
    print("=" * 75)


if __name__ == "__main__":
    main()
