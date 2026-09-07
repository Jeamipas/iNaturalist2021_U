"""
Experiment Tracker: logs, aggregates, and renders comparison tables
adhering to Section 14 of the project rubric and efficiency benchmarks.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd


class ExperimentTracker:
    """
    Logs experiment configurations and evaluation metrics, saving them to disk
    and generating summary comparison tables in DataFrame and Markdown formats.
    """

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.log_dir / "experiments_history.json"
        self.experiments: List[Dict[str, Any]] = self._load_history()

    def _load_history(self) -> List[Dict[str, Any]]:
        if self.history_file.exists():
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def save_history(self) -> None:
        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(self.experiments, f, indent=2)

    def log_experiment(
        self,
        exp_id: str,
        model_name: str,
        optimizer: str,
        regularization: str,
        augmentation: str,
        transfer_learning: str,
        long_tail: str,
        metrics: Dict[str, float],
        training_time_sec: float,
        peak_vram_mb: float = 0.0,
        param_count_m: float = 0.0,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Logs an experiment entry.

        Args:
            exp_id: Identifier (e.g. 'E1', 'E2', ...).
            model_name: Model architecture name (e.g. 'MLP', 'ResNet-50', 'ConvNeXt-Tiny').
            optimizer: Optimizer used ('SGD+M', 'AdamW', etc.).
            regularization: Regularization setup (e.g. 'None', 'BN+Dropout', 'BN+Dropout+WD').
            augmentation: Augmentation description ('None', 'Standard', 'Full').
            transfer_learning: Transfer learning mode ('No', 'Frozen', 'Partial FT', 'Full FT').
            long_tail: Long tail setting ('No', 'Long-Tail Baseline', 'Long-Tail + FocalLoss').
            metrics: Dict with 'top1_acc', 'top5_acc', 'macro_f1', 'weighted_f1'.
            training_time_sec: Total training time in seconds.
            peak_vram_mb: Peak GPU memory used in MB.
            param_count_m: Model parameter count in Millions.
            extra_metadata: Any additional key-values to preserve.
        """
        record = {
            "exp_id": exp_id,
            "model": model_name,
            "optimizer": optimizer,
            "regularization": regularization,
            "augmentation": augmentation,
            "transfer_learning": transfer_learning,
            "long_tail": long_tail,
            "top1_acc": round(metrics.get("top1_acc", 0.0) * 100, 2),
            "top5_acc": round(metrics.get("top5_acc", 0.0) * 100, 2),
            "macro_f1": round(metrics.get("macro_f1", 0.0) * 100, 2),
            "weighted_f1": round(metrics.get("weighted_f1", 0.0) * 100, 2),
            "time_sec": round(training_time_sec, 1),
            "peak_vram_mb": round(peak_vram_mb, 1),
            "params_m": round(param_count_m, 2),
        }

        # Include long-tail specific breakdown if present
        for tier in ["frequent", "medium", "minority"]:
            if f"{tier}_macro_f1" in metrics:
                record[f"{tier}_f1"] = round(metrics[f"{tier}_macro_f1"] * 100, 2)
            if f"{tier}_recall" in metrics:
                record[f"{tier}_rec"] = round(metrics[f"{tier}_recall"] * 100, 2)

        if extra_metadata:
            record["extra"] = extra_metadata

        # Update if exp_id exists, else append
        for i, existing in enumerate(self.experiments):
            if existing.get("exp_id") == exp_id:
                self.experiments[i] = record
                self.save_history()
                return record

        self.experiments.append(record)
        self.save_history()
        return record

    def to_dataframe(self) -> pd.DataFrame:
        """Returns the experiments table as a pandas DataFrame."""
        return pd.DataFrame(self.experiments)

    def to_markdown_table(self) -> str:
        """Generates the Markdown table adhering to the project rubric."""
        df = self.to_dataframe()
        if df.empty:
            return "No experiments recorded yet."
        
        display_cols = [
            "exp_id", "model", "optimizer", "regularization",
            "augmentation", "transfer_learning", "long_tail",
            "top1_acc", "top5_acc", "macro_f1", "time_sec", "peak_vram_mb", "params_m"
        ]
        available_cols = [c for c in display_cols if c in df.columns]
        sub_df = df[available_cols].rename(columns={
            "exp_id": "ID",
            "model": "Modelo",
            "optimizer": "Optimizador",
            "regularization": "Regularización",
            "augmentation": "Aug.",
            "transfer_learning": "Transfer Learning",
            "long_tail": "Long-Tail",
            "top1_acc": "Top-1 (%)",
            "top5_acc": "Top-5 (%)",
            "macro_f1": "Macro-F1 (%)",
            "time_sec": "Tiempo (s)",
            "peak_vram_mb": "VRAM (MB)",
            "params_m": "Params (M)"
        })
        return sub_df.to_markdown(index=False)

    def save_markdown_table(self, file_path: str) -> None:
        """Saves the formatted Markdown table directly to disk."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        table_str = self.to_markdown_table()
        with open(path, "w", encoding="utf-8") as f:
            f.write(table_str)

    def save_json(self, file_path: str) -> None:
        """Saves all experiments records as JSON to disk."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.experiments, f, indent=2)

