"""Training and analysis for the single full-data deliverable notebook.

Uses streaming classification metrics, explicit freezing, epoch-boundary resume,
and atomic checkpoints. Legacy subset runners and their trainer are not used.
"""
import copy
import hashlib
import json
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image

from src.full_data import atomic_json, digest, read_split


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def seed_worker(_):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class FullDataset(Dataset):
    def __init__(self, root, entries, size, augmentation, training=False):
        self.root, self.entries = Path(root), entries
        self.targets = [label for _, label in entries]
        steps = [transforms.RandomResizedCrop(size, scale=(0.8, 1.0)), transforms.RandomHorizontalFlip()] if training and augmentation else [transforms.Resize((size, size))]
        self.transform = transforms.Compose(steps + [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        name, target = self.entries[index]
        with Image.open(self.root / name) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, target


def long_tail_entries(entries, classes, seed):
    rng = random.Random(seed)
    species = list(range(classes))
    rng.shuffle(species)
    n_many, n_medium = int(classes * 0.2), int(classes * 0.4)
    tiers = {"many": species[:n_many], "medium": species[n_many:n_many + n_medium], "few": species[n_many + n_medium:]}
    quotas = {label: quota for tier, quota in [("many", 50), ("medium", 20), ("few", 5)] for label in tiers[tier]}
    by_class = defaultdict(list)
    for entry in entries:
        by_class[entry[1]].append(entry)
    selected = []
    for label in range(classes):
        candidates = by_class[label].copy()
        rng.shuffle(candidates)
        selected.extend(candidates[:quotas[label]])
    return selected, tiers


def make_model(config, classes=10000):
    seed_all(config["seed"])
    architecture = config["architecture"]
    if architecture == "mlp":
        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * config["size"]**2, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, classes))
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)
        return model
    if architecture == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if config["pretrained"] else None
        model = models.convnext_tiny(weights=weights)
        seed_all(config["seed"])  # Identical classifier initialization across transfer modes.
        model.classifier = nn.Sequential(model.classifier[0], model.classifier[1], nn.Dropout(config["dropout"]), nn.Linear(768, classes))
        head = model.classifier
        partial = model.features[7]
    elif architecture == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if config["pretrained"] else None
        model = models.resnet18(weights=weights)
        seed_all(config["seed"])
        model.fc = nn.Sequential(nn.Dropout(config["dropout"]), nn.Linear(512, classes))
        head, partial = model.fc, model.layer4
    else:
        raise ValueError(architecture)
    if not config["pretrained"] and config["mode"] != "full":
        raise ValueError("Scratch must train every parameter")
    if config["mode"] != "full":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in head.parameters():
            parameter.requires_grad = True
        if config["mode"] == "partial":
            for parameter in partial.parameters():
                parameter.requires_grad = True
        elif config["mode"] != "frozen":
            raise ValueError(config["mode"])
    return model


def train_mode(model, config):
    model.train()
    if config["mode"] in ["frozen", "partial"]:
        model.eval()  # Frozen BatchNorm/stochastic-depth remain fixed.
        head = model.classifier if config["architecture"] == "convnext_tiny" else model.fc
        head.train()
        if config["mode"] == "partial":
            (model.features[7] if config["architecture"] == "convnext_tiny" else model.layer4).train()


class MatrixMuon(torch.optim.Optimizer):
    """Project Muon: reshape conv kernels to matrices; five Newton–Schulz steps.

    Adapted from the existing project's Muon implementation, without its unrelated
    optimizers or claims about convergence. Used only for internal matrix weights.
    """
    def __init__(self, params, lr=0.02):
        super().__init__(params, dict(lr=lr, momentum=0.95))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(p)
                buffer = state["momentum"]
                buffer.mul_(group["momentum"]).add_(p.grad)
                update = p.grad.add(buffer, alpha=group["momentum"])
                matrix = update.reshape(update.shape[0], -1).float()
                transposed = matrix.shape[0] > matrix.shape[1]
                if transposed:
                    matrix = matrix.T
                matrix = matrix / (matrix.norm() + 1e-7)
                for _ in range(5):
                    gram = matrix @ matrix.T
                    matrix = 3.4445 * matrix + (-4.7750 * gram + 2.0315 * (gram @ gram)) @ matrix
                if transposed:
                    matrix = matrix.T
                scale = max(1.0, update.shape[0] / (update.numel() / update.shape[0]))**0.5
                p.add_(matrix.reshape_as(p).to(p.dtype), alpha=-group["lr"] * scale)


class MuonAdam(torch.optim.Optimizer):
    """One GradScaler/scheduler interface with resumable child states."""
    def __init__(self, matrices, other, matrix_lr, adam_lr):
        self.children = [MatrixMuon(matrices, matrix_lr), torch.optim.Adam(other, lr=adam_lr, weight_decay=0, fused=torch.cuda.is_available())]
        super().__init__([g for child in self.children for g in child.param_groups], {})

    def step(self, closure=None):
        for child in self.children:
            child.step()

    def state_dict(self):
        return {"children": [child.state_dict() for child in self.children]}

    def load_state_dict(self, state):
        for child, saved in zip(self.children, state["children"]):
            child.load_state_dict(saved)
        self.param_groups = [g for child in self.children for g in child.param_groups]


def make_optimizer(model, config):
    head = getattr(model, "classifier", getattr(model, "fc", model))
    head_ids = {id(p) for p in head.parameters()}
    if config["optimizer"] == "muon":
        matrices, other = [], []
        for name, p in model.named_parameters():
            if p.requires_grad:
                # ConvNeXt layer_scale is shaped [C, 1, 1] but represents a
                # per-channel vector. Only convolution/linear weight matrices
                # belong in Muon; learned scales also use auxiliary Adam.
                is_matrix_weight = name.endswith("weight") and p.ndim >= 2 and id(p) not in head_ids
                (matrices if is_matrix_weight else other).append(p)
        if not matrices or not other:
            raise ValueError("Muon needs internal matrices and auxiliary Adam parameters")
        return MuonAdam(matrices, other, config["muon_lr"], config["lr"])
    groups = [{"params": [p for p in head.parameters() if p.requires_grad], "lr": config["lr"]}]
    backbone = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    if backbone:
        groups.append({"params": backbone, "lr": config["backbone_lr"]})
    if config["optimizer"] == "sgd":
        return torch.optim.SGD(groups, momentum=0.9, weight_decay=config["weight_decay"])
    return torch.optim.Adam(groups, weight_decay=config["weight_decay"], fused=torch.cuda.is_available())


class StreamingMetrics:
    def __init__(self, classes):
        self.support = np.zeros(classes, dtype=np.int64)
        self.predicted = self.support.copy()
        self.true_positive = self.support.copy()
        self.top5 = self.count = 0

    def update(self, logits, targets):
        predicted = logits.argmax(1)
        truth = targets.cpu().numpy()
        pred = predicted.cpu().numpy()
        self.support += np.bincount(truth, minlength=len(self.support))
        self.predicted += np.bincount(pred, minlength=len(self.support))
        self.true_positive += np.bincount(truth[truth == pred], minlength=len(self.support))
        self.top5 += int((logits.topk(min(5, logits.shape[1]), dim=1).indices == targets[:, None]).any(1).sum())
        self.count += len(truth)

    def compute(self):
        f1 = np.divide(2 * self.true_positive, self.support + self.predicted, out=np.zeros_like(self.support, dtype=float), where=(self.support + self.predicted) > 0)
        return {"top1": float(self.true_positive.sum() / self.count), "top5": self.top5 / self.count, "macro_f1": float(f1.mean()), "weighted_f1": float(np.dot(f1, self.support) / self.count)}


def evaluate(model, loader, device, classes, criterion, amp, collect=False):
    model.eval()
    metrics = StreamingMetrics(classes)
    loss_sum = 0.0
    predictions, truths, confidence = [], [], []
    with torch.inference_mode():
        for images, targets in loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            if device.type == "cuda" and hasattr(model, "features"):
                images = images.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device.type, enabled=amp, dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, targets)
            loss_sum += float(loss) * len(targets)
            metrics.update(logits, targets)
            if collect:
                predictions.append(logits.argmax(1).cpu().numpy())
                truths.append(targets.cpu().numpy())
                confidence.append(logits.float().softmax(1).max(1).values.cpu().numpy())
    result = {**metrics.compute(), "loss": loss_sum / metrics.count, "samples": metrics.count}
    arrays = {"prediction": np.concatenate(predictions), "target": np.concatenate(truths), "confidence": np.concatenate(confidence)} if collect else None
    return result, arrays


def loader(dataset, config, epoch=0, training=False):
    return DataLoader(dataset, batch_size=config["batch_size"], shuffle=training, drop_last=False,
                      num_workers=config["workers"], worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(config["seed"] + epoch),
                      pin_memory=torch.cuda.is_available(), persistent_workers=False)


def atomic_checkpoint(path, state):
    temporary = Path(str(path) + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def train_epoch(model, data_loader, optimizer, criterion, scaler, device, config, epoch, progress_path):
    train_mode(model, config)
    optimizer.zero_grad(set_to_none=True)
    total_loss = correct = samples = 0
    accumulation = config["accumulation"]
    last_update = time.monotonic()
    for step, (images, targets) in enumerate(data_loader):
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        if device.type == "cuda" and hasattr(model, "features"):
            images = images.contiguous(memory_format=torch.channels_last)
        start_group = (step // accumulation) * accumulation * config["batch_size"]
        group_samples = min(accumulation * config["batch_size"], len(data_loader.dataset) - start_group)
        with torch.autocast(device.type, enabled=scaler.is_enabled(), dtype=torch.float16):
            logits = model(images)
            raw_loss = criterion(logits, targets)
            loss = raw_loss * len(targets) / group_samples
        if not torch.isfinite(raw_loss):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch}, batch {step}")
        scaler.scale(loss).backward()
        if (step + 1) % accumulation == 0 or step + 1 == len(data_loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        total_loss += float(raw_loss.detach()) * len(targets)
        correct += int((logits.argmax(1) == targets).sum())
        samples += len(targets)
        if time.monotonic() - last_update > 60:
            status = {"state": "TRAINING", "experiment": config["id"], "epoch": epoch,
                      "batch": step + 1, "batches": len(data_loader), "samples": samples,
                      "loss": total_loss / samples, "accuracy": correct / samples}
            atomic_json(progress_path, status)
            print(json.dumps(status), flush=True)
            last_update = time.monotonic()
    if samples != len(data_loader.dataset):
        raise RuntimeError("The epoch did not cover the entire training corpus")
    return {"loss": total_loss / samples, "top1": correct / samples, "samples": samples}


def run_experiment(config, data_root, output_root, require_cuda=True):
    allocation_started = float(os.environ.get("NATURALIST_ALLOCATION_STARTED", time.time()))
    segment_seconds = float(os.environ.get("NATURALIST_SEGMENT_SECONDS", 25200))
    output = Path(output_root) / config["id"]
    output.mkdir(parents=True, exist_ok=True)
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    train_entries, categories, labels, _ = read_split(data_root, "train_mini", config["classes"], 50)
    val_entries, _, val_labels, _ = read_split(data_root, "val", config["classes"], 10)
    if labels != val_labels:
        raise ValueError("Class mapping mismatch")
    data_hashes = {"train": digest(Path(data_root) / "train_mini.json"), "val": digest(Path(data_root) / "val.json")}
    identity = {"config_hash": config_hash, "data_hashes": data_hashes,
                "source_hashes": {name: digest(Path(__file__).parent / name) for name in ["full_experiment.py", "full_data.py"]}}
    if (output / "result.json").exists():
        saved = json.loads((output / "result.json").read_text())
        if saved["identity"] != identity:
            raise ValueError("Existing results have a different configuration/corpus; use another run directory")
        return saved
    tiers = None
    if config.get("long_tail"):
        train_entries, tiers = long_tail_entries(train_entries, config["classes"], config["seed"])
        atomic_json(output / "long_tail_manifest.json", {"seed": config["seed"], "tiers": tiers, "entries": train_entries})
    train_data = FullDataset(data_root, train_entries, config["size"], config["augmentation"], training=True)
    val_data = FullDataset(data_root, val_entries, config["size"], False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if require_cuda and device.type != "cuda":
        raise RuntimeError("A Slurm GPU allocation is required; CPU fallback is disabled")
    torch.set_num_threads(max(1, min(4, int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))))
    model = make_model(config, config["classes"]).to(device)
    if device.type == "cuda" and config["architecture"] != "mlp":
        model = model.to(memory_format=torch.channels_last)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
    initial_digest = hashlib.sha256()
    for p in model.state_dict().values():
        initial_digest.update(p.detach().cpu().numpy().tobytes())
    initial_hash = initial_digest.hexdigest()
    optimizer = make_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda ep: (ep + 1) / 3 if ep < 3 else 0.5 * (1 + np.cos(np.pi * (ep - 2) / max(1, config["epochs"] - 2))))
    weights = None
    if config.get("weighted_loss"):
        counts = np.bincount(train_data.targets, minlength=config["classes"])
        weights = torch.tensor(1.0 / counts, dtype=torch.float32, device=device)
        weights /= weights.mean()
    criterion = nn.CrossEntropyLoss(weight=weights)
    validation_criterion = nn.CrossEntropyLoss()  # Comparable unweighted validation loss.
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history, best_score, stale, start_epoch, best_epoch = [], -1.0, 0, 1, None
    last_path, best_path = output / "last.pt", output / "best.pt"
    if last_path.exists():
        saved = torch.load(last_path, map_location=device, weights_only=False)
        if saved["identity"] != identity:
            raise ValueError("Resume configuration/corpus mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        history, best_score, stale, best_epoch = saved["history"], saved["best_score"], saved["stale"], saved["best_epoch"]
        start_epoch = saved["epoch"] + 1
        random.setstate(saved["random"])
        np.random.set_state(saved["numpy"])
        torch.set_rng_state(saved["torch"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([r.cpu() for r in saved["cuda"]])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    environment = {"torch": torch.__version__, "torchvision": __import__("torchvision").__version__,
                   "numpy": np.__version__, "cuda": torch.version.cuda,
                   "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
                   "slurm_job": os.environ.get("SLURM_JOB_ID"), "deterministic_algorithms": True}
    atomic_json(output / "configuration.json", {"config": config, "identity": identity, "environment": environment, "initial_weights_sha256": initial_hash})
    try:
        for epoch in range(start_epoch, config["epochs"] + 1):
            if stale >= config["patience"]:
                break
            started = time.perf_counter()
            lrs = [g["lr"] for g in optimizer.param_groups]
            training = train_epoch(model, loader(train_data, config, epoch, True), optimizer, criterion, scaler, device, config, epoch, output / "status.json")
            atomic_json(output / "status.json", {"state": "VALIDATING", "epoch": epoch, "experiment": config["id"]})
            validation, _ = evaluate(model, loader(val_data, config), device, config["classes"], validation_criterion, scaler.is_enabled())
            if validation["samples"] != len(val_data):
                raise RuntimeError("Incomplete validation")
            if device.type == "cuda":
                torch.cuda.synchronize()
            row = {"epoch": epoch, "train": training, "validation": validation, "seconds": time.perf_counter() - started, "learning_rates": lrs,
                   "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0}
            history.append(row)
            print(json.dumps(row), flush=True)
            if validation["macro_f1"] > best_score + config["min_delta"]:
                best_score, best_epoch, stale = validation["macro_f1"], epoch, 0
                atomic_checkpoint(best_path, {"model": model.state_dict(), "epoch": epoch, "metrics": validation, "identity": identity, "config": config})
            else:
                stale += 1
            scheduler.step()
            atomic_checkpoint(last_path, {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                                         "history": history, "best_score": best_score, "stale": stale, "best_epoch": best_epoch, "epoch": epoch, "identity": identity,
                                         "random": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                                         "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else []})
            atomic_json(output / "history.json", history)
            remaining_epochs = epoch < config["epochs"] and stale < config["patience"]
            # Finish the epoch and persist all states before releasing the GPU.
            # Use a conservative measured estimate rather than starting an epoch
            # which is unlikely to fit in this allocation.
            estimated_epoch = max(h["seconds"] for h in history[-3:]) * 1.25
            elapsed = time.time() - allocation_started
            if remaining_epochs and elapsed + estimated_epoch >= segment_seconds:
                continuation = {"state": "NEEDS_RESUME", "experiment": config["id"],
                                "epoch": epoch, "next_epoch": epoch + 1,
                                "checkpoint": str(last_path), "elapsed_seconds": elapsed}
                atomic_json(output / "status.json", continuation)
                return continuation
        best = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best["model"])
        validation, arrays = evaluate(model, loader(val_data, config), device, config["classes"], validation_criterion, scaler.is_enabled(), collect=True)
        np.savez_compressed(output / "predictions.npz", **arrays)
        result = {"id": config["id"], "config": config, "identity": identity, "environment": environment,
                  "initial_weights_sha256": initial_hash, "best_epoch": best["epoch"], "epochs_completed": len(history),
                  "metrics": validation, "train_images": len(train_data), "val_images": len(val_data),
                  "parameters": sum(p.numel() for p in model.parameters()), "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                  "training_validation_seconds": sum(h["seconds"] for h in history),
                  "peak_vram_mb": max([h.get("peak_vram_mb", 0) for h in history] + [torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0]),
                  "checkpoint": str(best_path), "tiers": tiers}
        atomic_json(output / "result.json", result)
        atomic_json(output / "status.json", {"state": "COMPLETED", "best_epoch": best["epoch"], "metrics": validation})
        return result
    except Exception as error:
        atomic_json(output / "status.json", {"state": "FAILED", "error": repr(error), "resume_from": str(last_path)})
        raise


def show_eda(data_root, audit):
    import matplotlib.pyplot as plt
    entries, categories, mapping, meta = read_split(data_root, "train_mini")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    kingdoms = audit["species_by_kingdom"]
    axes[0].bar(kingdoms.keys(), kingdoms.values())
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].set_title("Especies por reino — corpus completo")
    axes[1].hist([im["width"] for im in meta["images"] if "width" in im], bins=30)
    axes[1].set_title("Ancho original — corpus completo")
    axes[2].bar(["Train/especie", "Val/especie"], [50, 10])
    axes[2].set_title("Mini está balanceado")
    plt.tight_layout()
    plt.show()
    by_class = defaultdict(list)
    for name, label in entries:
        by_class[label].append(name)
    label_to_cat = {label: category for category, label in mapping.items()}
    by_genus = defaultdict(list)
    for label, cat in label_to_cat.items():
        by_genus[categories[cat].get("genus", "Unknown")].append(label)
    chosen = next((sorted(ls)[:3] for _, ls in sorted(by_genus.items()) if len(ls) >= 3), [0, 1, 2])
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    rng = random.Random(42)
    for row, label in enumerate(chosen):
        names = rng.sample(by_class[label], 3)
        for ax, name in zip(axes[row], names):
            with Image.open(Path(data_root) / name) as image:
                image.load()
                ax.imshow(image.convert("RGB"))
            ax.set_title(categories[label_to_cat[label]]["name"], fontsize=8)
            ax.axis("off")
    fig.suptitle("Variabilidad intraespecie y especies del mismo género")
    plt.tight_layout()
    plt.show()


def show_results(plan, output_root, data_root=None):
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display
    root = Path(output_root)
    completed, rows = [], []
    for config in plan:
        path = root / config["id"] / "result.json"
        if path.exists():
            result = json.loads(path.read_text())
            completed.append(result)
            rows.append({"ID": config["id"], "Modelo": config["architecture"], "Optimizador": config["optimizer"], "Modo": config["mode"],
                         "Estado": "COMPLETED", **result["metrics"], "Época": result["best_epoch"], "Tiempo (s)": result["training_validation_seconds"],
                         "Parámetros": result["parameters"], "VRAM (MB)": result["peak_vram_mb"], "Train": result["train_images"], "Val": result["val_images"]})
        else:
            state_path = root / config["id"] / "status.json"
            state = json.loads(state_path.read_text())["state"] if state_path.exists() else "PENDING"
            rows.append({"ID": config["id"], "Modelo": config["architecture"], "Optimizador": config["optimizer"], "Modo": config["mode"], "Estado": state})
    table = pd.DataFrame(rows)
    pairs = {r["id"]: r for r in completed}
    if "E03" in pairs and "E04" in pairs:
        if pairs["E03"]["initial_weights_sha256"] != pairs["E04"]["initial_weights_sha256"]:
            raise RuntimeError("Adam–Muon initial weights differ; this comparison is invalid")
        print("Comparación Adam–Muon: SHA256 de pesos iniciales idéntico.")
    display(table)
    root.mkdir(parents=True, exist_ok=True)
    table.to_csv(root / "comparison.csv", index=False)
    atomic_json(root / "comparison.json", rows)
    if not completed:
        print("No hay métricas finales todavía. No se reutilizan resultados de subconjuntos.")
        return table
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for result in completed:
        history = json.loads((root / result["id"] / "history.json").read_text())
        epochs = [h["epoch"] for h in history]
        for ax, key, split in [(axes[0, 0], "loss", "train"), (axes[0, 1], "loss", "validation"), (axes[1, 0], "top1", "train"), (axes[1, 1], "macro_f1", "validation")]:
            ax.plot(epochs, [h[split][key] for h in history], label=result["id"])
            ax.set(xlabel="Época", ylabel=f"{split}: {key}")
    for ax in axes.flat:
        ax.legend()
        ax.grid(alpha=0.2)
    plt.tight_layout()
    fig.savefig(root / "learning_curves.png", dpi=160)
    plt.show()
    if data_root is None:
        return table
    best = max(completed, key=lambda r: r["metrics"]["macro_f1"])
    arrays = np.load(root / best["id"] / "predictions.npz")
    truth, prediction, confidence = arrays["target"], arrays["prediction"], arrays["confidence"]
    entries, categories, mapping, _ = read_split(data_root, "val")
    label_to_cat = {label: cat for cat, label in mapping.items()}
    per_species = []
    support = np.bincount(truth, minlength=len(mapping))
    predicted_count = np.bincount(prediction, minlength=len(mapping))
    tp = np.bincount(truth[truth == prediction], minlength=len(mapping))
    f1 = np.divide(2 * tp, support + predicted_count, out=np.zeros(len(mapping)), where=support + predicted_count > 0)
    for label, cat in label_to_cat.items():
        per_species.append({"label": label, "name": categories[cat]["name"], "kingdom": categories[cat].get("kingdom", ""), "genus": categories[cat].get("genus", ""), "support": int(support[label]), "true_positive": int(tp[label]), "f1": f1[label]})
    species_table = pd.DataFrame(per_species)
    species_table.to_csv(root / "best_model_species_metrics.csv", index=False)
    grouped = species_table.groupby("kingdom").agg(species=("name", "count"), macro_f1=("f1", "mean"), correct=("true_positive", "sum"), images=("support", "sum"))
    grouped["top1"] = grouped["correct"] / grouped["images"]
    display(grouped)
    pairs = Counter(zip(truth[truth != prediction].tolist(), prediction[truth != prediction].tolist()))
    confusion = [{"real": categories[label_to_cat[a]]["name"], "predicted": categories[label_to_cat[b]]["name"], "count": count,
                  "same_genus": categories[label_to_cat[a]].get("genus") == categories[label_to_cat[b]].get("genus")} for (a, b), count in pairs.most_common(20)]
    display(pd.DataFrame(confusion))
    atomic_json(root / "confused_species.json", confusion)
    correct = truth == prediction
    selections = [np.flatnonzero(correct)[np.argsort(confidence[correct])[-3:]],
                  np.flatnonzero(correct)[np.argsort(confidence[correct])[:3]],
                  np.flatnonzero(~correct)[np.argsort(confidence[~correct])[-3:]]]
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    for row, indices in enumerate(selections):
        for ax in axes[row]:
            ax.axis("off")
        for ax, index in zip(axes[row], indices):
            with Image.open(Path(data_root) / entries[index][0]) as image:
                ax.imshow(image.convert("RGB"))
            ax.set_title(f"Real: {categories[label_to_cat[int(truth[index])]]['name']}\nPred.: {categories[label_to_cat[int(prediction[index])]]['name']}\np={confidence[index]:.3f}", fontsize=8)
    fig.suptitle(f"{best['id']}: aciertos de alta/baja confianza y errores de alta confianza")
    plt.tight_layout()
    fig.savefig(root / "error_examples.png", dpi=160)
    plt.show()
    for result in completed:
        if result.get("tiers"):
            arr = np.load(root / result["id"] / "predictions.npz")
            group_rows = []
            for tier, tier_labels in result["tiers"].items():
                yt, yp = arr["target"], arr["prediction"]
                sup = np.bincount(yt, minlength=len(mapping))
                pc = np.bincount(yp, minlength=len(mapping))
                true_p = np.bincount(yt[yt == yp], minlength=len(mapping))
                fs = np.divide(2 * true_p, sup + pc, out=np.zeros(len(mapping)), where=sup + pc > 0)
                group_rows.append({"ID": result["id"], "tier": tier, "species": len(tier_labels), "macro_f1": float(fs[tier_labels].mean()),
                                   "recall": float((true_p[tier_labels] / sup[tier_labels]).mean())})
            display(pd.DataFrame(group_rows))
            atomic_json(root / result["id"] / "tier_metrics.json", group_rows)
    return table
