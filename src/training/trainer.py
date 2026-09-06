"""
Ultra-robust training and evaluation engine.
Features:
- Automatic Mixed Precision (AMP) with GradScaler for 4GB VRAM optimization
- Peak VRAM memory tracking
- Integrated Early Stopping and best model restoration
- Metric computation: Top-1, Top-5, Macro F1, Weighted F1, and Long-Tail tier breakdown
"""

import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from src.utils.metrics import compute_classification_metrics
from .early_stopping import EarlyStopping


class Trainer:
    """
    High-performance, reproducible PyTorch model trainer.
    Features:
    - Real-time tqdm.auto progress bar for epochs and batches
    - Real-time TensorBoard metric logging (Loss, Top-1/5 Acc, Macro F1, VRAM)
    - Automatic Mixed Precision (AMP) and channels_last hardware acceleration
    - Early Stopping and best model restoration
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: Optional[torch.device] = None,
        scheduler: Optional[object] = None,
        early_stopping: Optional[EarlyStopping] = None,
        use_amp: bool = True,
        class_tiers: Optional[Dict[str, List[int]]] = None,
        batch_augmenter: Optional[object] = None,
        grad_accum_steps: int = 1,
        tb_log_dir: Optional[str] = None
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.class_tiers = class_tiers
        self.batch_augmenter = batch_augmenter
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.show_pbar = True

        # TensorBoard integration
        self.writer = SummaryWriter(log_dir=tb_log_dir) if (SummaryWriter and tb_log_dir) else None

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model.to(self.device)
        if self.device.type == "cuda":
            # Channels Last memory format yields ~20-35% speedup on modern NVIDIA GPUs
            try:
                self.model = self.model.to(memory_format=torch.channels_last)
            except Exception:
                pass
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.benchmark = True

        # Automatic Mixed Precision for memory efficiency on 4GB VRAM
        self.use_amp = use_amp and (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def train_one_epoch(self, dataloader: DataLoader, epoch: int = 1, epochs: int = 1) -> Tuple[float, float]:
        """Runs one full training epoch with optional CutMix/MixUp and Gradient Accumulation."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        self.optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch:02d}/{epochs:02d} [Train]",
            leave=False,
            disable=not self.show_pbar
        )

        for step, (images, targets) in enumerate(pbar):
            if self.device.type == "cuda" and images.ndim == 4:
                images = images.to(self.device, memory_format=torch.channels_last, non_blocking=True)
            else:
                images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Apply batch-level SOTA augmentations (CutMix / MixUp) on GPU
            if self.batch_augmenter is not None:
                images, targets = self.batch_augmenter(images, targets)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                if self.grad_accum_steps > 1:
                    loss = loss / self.grad_accum_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == len(dataloader):
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            batch_size = targets.size(0)
            raw_loss = loss.item() * (self.grad_accum_steps if self.grad_accum_steps > 1 else 1.0)
            total_loss += raw_loss * batch_size

            true_labels = targets.argmax(dim=1) if targets.ndim == 2 else targets
            preds = outputs.argmax(dim=1)
            correct += (preds == true_labels).sum().item()
            total_samples += batch_size

            current_acc = (correct / total_samples) * 100
            pbar.set_postfix({"loss": f"{raw_loss:.4f}", "acc": f"{current_acc:.1f}%"})

        epoch_loss = total_loss / max(1, total_samples)
        epoch_acc = correct / max(1, total_samples)
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, epoch: int = 1, epochs: int = 1) -> Tuple[float, Dict[str, float]]:
        """Evaluates model on validation loader, returning loss and all rubric metrics."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        all_preds = []
        all_targets = []
        all_probs = []

        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch:02d}/{epochs:02d} [Val]  ",
            leave=False,
            disable=not self.show_pbar
        )

        for images, targets in pbar:
            if self.device.type == "cuda" and images.ndim == 4:
                images = images.to(self.device, memory_format=torch.channels_last, non_blocking=True)
            else:
                images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Softmax probabilities for Top-K
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()

            all_probs.append(probs)
            all_preds.append(preds)
            all_targets.append(targets.cpu().numpy())

        val_loss = total_loss / max(1, total_samples)
        y_true = np.concatenate(all_targets)
        y_pred = np.concatenate(all_preds)
        y_probs = np.concatenate(all_probs, axis=0)

        metrics = compute_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_probs=y_probs,
            class_tiers=self.class_tiers
        )
        return val_loss, metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        verbose: bool = True,
        show_pbar: bool = True
    ) -> Tuple[Dict[str, List[float]], Dict[str, float], float, float]:
        """
        Executes full training process across specified epochs with real-time logging.

        Returns:
            Tuple of (history_dict, best_val_metrics, peak_vram_mb, total_time_sec)
        """
        self.show_pbar = show_pbar
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        if verbose:
            dev_name = torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU"
            print(f"[*] Dispositivo Activo: {self.device} ({dev_name})")

        history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_top5_acc": [],
            "val_macro_f1": [],
            "val_weighted_f1": [],
            "epoch_times": []
        }

        best_val_metrics: Dict[str, float] = {}
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            ep_start = time.time()

            train_loss, train_acc = self.train_one_epoch(train_loader, epoch=epoch, epochs=epochs)
            val_loss, val_metrics = self.evaluate(val_loader, epoch=epoch, epochs=epochs)

            ep_time = time.time() - ep_start

            # Step scheduler if applicable
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["macro_f1"])
                else:
                    self.scheduler.step()

            # Record history
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_metrics["top1_acc"])
            history["val_top5_acc"].append(val_metrics["top5_acc"])
            history["val_macro_f1"].append(val_metrics["macro_f1"])
            history["val_weighted_f1"].append(val_metrics["weighted_f1"])
            history["epoch_times"].append(ep_time)

            # Log to TensorBoard if enabled
            if self.writer is not None:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("Loss/val", val_loss, epoch)
                self.writer.add_scalar("Accuracy/train_top1", train_acc * 100, epoch)
                self.writer.add_scalar("Accuracy/val_top1", val_metrics["top1_acc"] * 100, epoch)
                self.writer.add_scalar("Accuracy/val_top5", val_metrics["top5_acc"] * 100, epoch)
                self.writer.add_scalar("F1/val_macro", val_metrics["macro_f1"] * 100, epoch)
                self.writer.add_scalar("F1/val_weighted", val_metrics["weighted_f1"] * 100, epoch)
                if self.device.type == "cuda":
                    mem_mb = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
                    self.writer.add_scalar("Efficiency/VRAM_MB", mem_mb, epoch)

            if not best_val_metrics or val_metrics["macro_f1"] >= best_val_metrics.get("macro_f1", 0.0):
                best_val_metrics = {**val_metrics, "best_epoch": epoch, "val_loss": val_loss}

            if verbose:
                print(
                    f"Epoch {epoch:02d}/{epochs:02d} | "
                    f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}% | "
                    f"Val Loss: {val_loss:.4f}, Top-1: {val_metrics['top1_acc']*100:.2f}%, "
                    f"Top-5: {val_metrics['top5_acc']*100:.2f}%, F1: {val_metrics['macro_f1']*100:.2f}% | "
                    f"{ep_time:.1f}s"
                )

            # Early stopping evaluation
            if self.early_stopping is not None:
                score = val_metrics["macro_f1"] if self.early_stopping.mode == "max" else val_loss
                if self.early_stopping(score, self.model):
                    if verbose:
                        print(f"[*] Early stopping triggered at epoch {epoch}.")
                    break

        # Restore best weights if configured
        if self.early_stopping is not None and self.early_stopping.restore_best_weights:
            self.early_stopping.restore(self.model)

        if self.writer is not None:
            self.writer.flush()

        total_time = time.time() - start_time
        peak_vram = 0.0
        if self.device.type == "cuda":
            peak_vram = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)

        return history, best_val_metrics, peak_vram, total_time
