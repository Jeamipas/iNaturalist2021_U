"""
Top-K Checkpoint Manager for deep learning experiments.
Maintains the K best-performing models on disk, ranked by validation Macro F1 / Top-1.
Saves checkpoints and writes an informative manifest (top10_manifest.json).
"""

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch


class TopKCheckpointManager:
    """
    Manages saving and pruning of the top K checkpoints across all experimental runs.
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        k: int = 10,
        primary_metric: str = "macro_f1",
        secondary_metric: str = "top1_acc"
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.k = k
        self.primary_metric = primary_metric
        self.secondary_metric = secondary_metric
        self.manifest_path = self.checkpoint_dir / "top10_manifest.json"

        # Load existing manifest if present
        self.entries: List[Dict[str, Any]] = []
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    self.entries = json.load(f)
            except Exception:
                self.entries = []

    def _sort_entries(self):
        """Sorts entries descending by primary metric, then secondary metric."""
        self.entries.sort(
            key=lambda x: (
                x.get("metrics", {}).get(self.primary_metric, 0.0),
                x.get("metrics", {}).get(self.secondary_metric, 0.0)
            ),
            reverse=True
        )

    def is_eligible(self, metrics: Dict[str, float]) -> bool:
        """Checks if the given metrics qualify for the top K."""
        if len(self.entries) < self.k:
            return True
        worst_primary = self.entries[-1].get("metrics", {}).get(self.primary_metric, -1.0)
        current_primary = metrics.get(self.primary_metric, 0.0)
        return current_primary >= worst_primary

    def register_and_save(
        self,
        model: torch.nn.Module,
        exp_id: str,
        model_name: str,
        metrics: Dict[str, float],
        hyperparams: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[int]:
        """
        Saves checkpoint if it enters the Top K and re-ranks all saved checkpoints.

        Returns:
            Rank integer (1-indexed) if saved, or None if it did not qualify.
        """
        if not self.is_eligible(metrics):
            return None

        # Clean exp_id for file naming
        safe_id = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in exp_id)
        score = metrics.get(self.primary_metric, 0.0)

        # Temporary save
        temp_filename = f"temp_{safe_id}_{score:.4f}.pt"
        temp_path = self.checkpoint_dir / temp_filename

        save_dict = {
            "exp_id": exp_id,
            "model_name": model_name,
            "metrics": metrics,
            "hyperparams": hyperparams or {},
            "metadata": metadata or {},
            "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()}
        }
        torch.save(save_dict, temp_path)

        # Create entry
        entry = {
            "exp_id": exp_id,
            "model_name": model_name,
            "primary_score": score,
            "metrics": metrics,
            "hyperparams": hyperparams or {},
            "metadata": metadata or {},
            "current_temp_path": str(temp_path)
        }

        self.entries.append(entry)
        self._sort_entries()

        # Prune if exceeded K
        if len(self.entries) > self.k:
            evicted = self.entries.pop()
            evicted_file = evicted.get("filepath") or evicted.get("current_temp_path")
            if evicted_file and os.path.exists(evicted_file):
                try:
                    os.remove(evicted_file)
                except Exception:
                    pass

        # Re-index and rename checkpoints to maintain top_01, top_02, ... format
        current_rank = None
        for rank_idx, e in enumerate(self.entries, start=1):
            if e["exp_id"] == exp_id:
                current_rank = rank_idx

            score_val = e.get("metrics", {}).get(self.primary_metric, 0.0)
            safe_name = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in e["model_name"])
            target_filename = f"top_{rank_idx:02d}_{e['exp_id']}_{safe_name}_f1_{score_val:.4f}.pt"
            target_path = self.checkpoint_dir / target_filename

            source_path = Path(e.get("current_temp_path") or e.get("filepath", ""))
            if source_path.exists() and source_path != target_path:
                if target_path.exists():
                    try:
                        target_path.unlink()
                    except Exception:
                        pass
                shutil.move(str(source_path), str(target_path))

            e["rank"] = rank_idx
            e["filepath"] = str(target_path)
            e.pop("current_temp_path", None)

        # Write updated manifest
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.entries, f, indent=2)

        print(f"\n[TOP-10 MANAGER] Model '{exp_id}' ranked #{current_rank} in Top 10! (Macro F1: {score*100:.2f}%)")
        return current_rank

    def get_manifest(self) -> List[Dict[str, Any]]:
        """Returns the current manifest entries."""
        return self.entries
