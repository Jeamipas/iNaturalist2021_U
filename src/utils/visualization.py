"""
Visualization module for publication-quality plots:
- Loss & Accuracy learning curves
- Multi-optimizer / multi-architecture comparisons
- Ablation study bar charts
- Long-tail class-tier performance comparisons
"""

from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def set_plotting_style():
    """Sets a clean, scientific styling for all figures."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 11,
        "figure.titlesize": 14,
        "figure.dpi": 150
    })


def plot_training_curves(
    history: Dict[str, List[float]],
    title: str = "Training and Validation Curves",
    save_path: Optional[str] = None
) -> None:
    """
    Plots Train vs Val Loss and Train vs Val Accuracy across epochs.
    """
    set_plotting_style()
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Loss plot
    axes[0].plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#1f77b4", lw=2)
    axes[0].plot(epochs, history["val_loss"], "s--", label="Val Loss", color="#d62728", lw=2)
    axes[0].set_title("Loss vs. Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Accuracy / Macro F1 plot
    axes[1].plot(epochs, [x * 100 for x in history["train_acc"]], "o-", label="Train Top-1 Acc (%)", color="#2ca02c", lw=2)
    axes[1].plot(epochs, [x * 100 for x in history["val_acc"]], "s--", label="Val Top-1 Acc (%)", color="#ff7f0e", lw=2)
    if "val_macro_f1" in history:
        axes[1].plot(epochs, [x * 100 for x in history["val_macro_f1"]], "^:", label="Val Macro F1 (%)", color="#9467bd", lw=2)
    axes[1].set_title("Accuracy & F1 vs. Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score (%)")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()


def plot_comparison_curves(
    histories: Dict[str, Dict[str, List[float]]],
    metric: str = "val_loss",
    ylabel: str = "Validation Loss",
    title: str = "Model / Optimizer Comparison",
    save_path: Optional[str] = None
) -> None:
    """
    Plots a single metric over epochs for multiple experiments on the same figure.
    """
    set_plotting_style()
    plt.figure(figsize=(9, 5.5))

    for label, hist in histories.items():
        if metric in hist:
            values = hist[metric]
            if "acc" in metric or "f1" in metric:
                values = [v * 100 for v in values]
            epochs = range(1, len(values) + 1)
            plt.plot(epochs, values, marker="o", label=label, lw=2)

    plt.title(title, fontsize=13)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()


def plot_long_tail_comparison(
    baseline_metrics: Dict[str, float],
    mitigated_metrics: Dict[str, float],
    title: str = "Long-Tail Challenge: Desglose por Frecuencia de Clase",
    save_path: Optional[str] = None
) -> None:
    """
    Bar plot comparing Frequent, Medium, and Minority class performance
    between baseline and mitigated (e.g. Focal Loss) models.
    """
    set_plotting_style()
    tiers = ["Frequent", "Medium", "Minority"]
    tier_keys = ["frequent", "medium", "minority"]

    base_f1 = [baseline_metrics.get(f"{k}_macro_f1", 0.0) * 100 for k in tier_keys]
    mitig_f1 = [mitigated_metrics.get(f"{k}_macro_f1", 0.0) * 100 for k in tier_keys]

    x = np.arange(len(tiers))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, base_f1, width, label="Sin Mitigación (Cross-Entropy)", color="#7293CB")
    plt.bar(x + width/2, mitig_f1, width, label="Con Mitigación (Focal Loss / Balanced)", color="#D35E60")

    plt.ylabel("Macro F1-Score (%)", fontsize=12)
    plt.title(title, fontsize=13)
    plt.xticks(x, tiers, fontsize=11)
    plt.legend()
    plt.ylim(0, max(max(base_f1 + mitig_f1, default=50) + 15, 100))

    for i in range(len(tiers)):
        plt.text(x[i] - width/2, base_f1[i] + 1, f"{base_f1[i]:.1f}%", ha='center', va='bottom', fontsize=10)
        plt.text(x[i] + width/2, mitig_f1[i] + 1, f"{mitig_f1[i]:.1f}%", ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()


def plot_optimizer_error_convergence(
    optimizer_histories: Dict[str, Dict[str, List[float]]],
    title: str = "Convergencia de Error y Pérdida por Optimizador",
    save_path: Optional[str] = None
) -> None:
    """
    Plots training error rate (1 - Top-1) and validation loss across epochs for different optimizers
    (SGD+Momentum, Adam, AdamW, Muon, Lion).
    """
    set_plotting_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for opt_name, hist in optimizer_histories.items():
        epochs = range(1, len(hist["train_loss"]) + 1)
        # Training error rate: (1 - train_acc) * 100
        train_error = [(1.0 - acc) * 100 for acc in hist["train_acc"]]
        axes[0].plot(epochs, train_error, marker="o", lw=2, label=opt_name)

        # Validation Loss
        axes[1].plot(epochs, hist["val_loss"], marker="s", lw=2, linestyle="--", label=opt_name)

    axes[0].set_title("Tasa de Error en Entrenamiento (1 - Top-1 Acc %)", fontsize=12)
    axes[0].set_xlabel("Época", fontsize=11)
    axes[0].set_ylabel("Error de Entrenamiento (%)", fontsize=11)
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    axes[1].set_title("Pérdida en Validación (Val Loss)", fontsize=12)
    axes[1].set_xlabel("Época", fontsize=11)
    axes[1].set_ylabel("Cross-Entropy Loss", fontsize=11)
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()


def plot_multiexperiment_4grid(
    histories: Dict[str, Dict[str, List[float]]],
    title: str = "Convergencia y Eficiencia Multi-Experimento (Pérdida y Precisión vs. Épocas y Tiempo)",
    save_path: Optional[str] = None
) -> None:
    """
    Generates a publication-grade 2x2 comparison grid for multiple experiments:
    - Panel 1 (Top-Left): Val Loss vs. Epochs
    - Panel 2 (Top-Right): Val Loss vs. Elapsed Time (seconds)
    - Panel 3 (Bottom-Left): Val Top-1 Accuracy (%) vs. Epochs
    - Panel 4 (Bottom-Right): Val Top-1 Accuracy (%) vs. Elapsed Time (seconds)

    Each curve is identified by model, optimizer, and technique in the legend.
    """
    set_plotting_style()
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    # Color palette
    colors = plt.cm.tab10.colors

    for idx, (exp_label, hist) in enumerate(histories.items()):
        color = colors[idx % len(colors)]
        num_epochs = len(hist.get("val_loss", []))
        if num_epochs == 0:
            continue
        epochs = list(range(1, num_epochs + 1))

        # Calculate cumulative time in seconds
        epoch_times = hist.get("epoch_times", [1.0] * num_epochs)
        cum_time = list(np.cumsum(epoch_times))

        val_loss = hist["val_loss"]
        val_acc = [v * 100 for v in hist.get("val_acc", [0.0] * num_epochs)]

        # 1. Val Loss vs Epoch
        axes[0, 0].plot(epochs, val_loss, marker="o", lw=2, color=color, label=exp_label)

        # 2. Val Loss vs Time
        axes[0, 1].plot(cum_time, val_loss, marker="s", lw=2, color=color, label=exp_label)

        # 3. Val Accuracy vs Epoch
        axes[1, 0].plot(epochs, val_acc, marker="^", lw=2, color=color, label=exp_label)

        # 4. Val Accuracy vs Time
        axes[1, 1].plot(cum_time, val_acc, marker="d", lw=2, color=color, label=exp_label)

    # Subplot styling
    axes[0, 0].set_title("(A) Pérdida de Validación vs. Época", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Época", fontsize=11)
    axes[0, 0].set_ylabel("Validation Loss (Cross-Entropy)", fontsize=11)
    axes[0, 0].legend(fontsize=9, loc="upper right")
    axes[0, 0].grid(True, linestyle="--", alpha=0.6)

    axes[0, 1].set_title("(B) Pérdida de Validación vs. Tiempo Acumulado (s)", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Tiempo Transcurrido (segundos)", fontsize=11)
    axes[0, 1].set_ylabel("Validation Loss (Cross-Entropy)", fontsize=11)
    axes[0, 1].legend(fontsize=9, loc="upper right")
    axes[0, 1].grid(True, linestyle="--", alpha=0.6)

    axes[1, 0].set_title("(C) Precisión Top-1 vs. Época", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Época", fontsize=11)
    axes[1, 0].set_ylabel("Top-1 Accuracy (%)", fontsize=11)
    axes[1, 0].legend(fontsize=9, loc="lower right")
    axes[1, 0].grid(True, linestyle="--", alpha=0.6)

    axes[1, 1].set_title("(D) Precisión Top-1 vs. Tiempo Acumulado (s)", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Tiempo Transcurrido (segundos)", fontsize=11)
    axes[1, 1].set_ylabel("Top-1 Accuracy (%)", fontsize=11)
    axes[1, 1].legend(fontsize=9, loc="lower right")
    axes[1, 1].grid(True, linestyle="--", alpha=0.6)

    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
    else:
        plt.show()



