"""Official full-corpus preparation; no subset fallbacks or silent exclusions."""
import hashlib
import json
import os
import subprocess
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

SOURCE = "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/"
ARCHIVES = {
    "train_mini.json.tar.gz": "395a35be3651d86dc3b0d365b8ea5f92",
    "val.json.tar.gz": "4d761e0f6a86cc63e8f7afc91f6a8f0b",
    "train_mini.tar.gz": "db6ed8330e634445efc8fec83ae81442",
    "val.tar.gz": "f6f6e0e242e3d4c9569ba56400938afc",
}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_split(data_root, split, expected_classes=10000, expected_per_class=None):
    root = Path(data_root).resolve()
    path = root / ("train_mini.json" if split == "train_mini" else "val.json")
    meta = json.loads(path.read_text(encoding="utf-8"))
    categories = {int(c["id"]): c for c in meta["categories"]}
    labels = {category: label for label, category in enumerate(sorted(categories))}
    image_map = {int(im["id"]): im for im in meta["images"]}
    if len(image_map) != len(meta["images"]):
        raise ValueError("Duplicate image IDs")
    annotation_map = {}
    for ann in meta["annotations"]:
        image_id = int(ann["image_id"])
        if image_id in annotation_map or image_id not in image_map:
            raise ValueError("Each image must have exactly one existing annotation")
        annotation_map[image_id] = int(ann["category_id"])
    if set(annotation_map) != set(image_map):
        raise ValueError("Unannotated images or unknown image IDs")
    entries = []
    for im in meta["images"]:
        name = im["file_name"].replace("\\", "/")
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != split:
            raise ValueError(f"Invalid image path: {name}")
        cat = annotation_map[int(im["id"])]
        entries.append((name, labels[cat]))
    if len({name for name, _ in entries}) != len(entries):
        raise ValueError("Duplicate image paths")
    counts = Counter(label for _, label in entries)
    if len(categories) != expected_classes or len(counts) != expected_classes:
        raise ValueError(f"Expected {expected_classes} species, found {len(counts)}")
    if expected_per_class is not None and set(counts.values()) != {expected_per_class}:
        raise ValueError(f"Expected exactly {expected_per_class} images per species")
    return entries, categories, labels, meta


def audit_corpus(data_root, output=None, expected_classes=10000, train_per_class=50, val_per_class=10):
    root = Path(data_root)
    training, categories, labels, train_meta = read_split(root, "train_mini", expected_classes, train_per_class)
    validation, val_categories, val_labels, val_meta = read_split(root, "val", expected_classes, val_per_class)
    if labels != val_labels or categories != val_categories:
        raise ValueError("Training and validation taxonomy/mapping disagree")
    if {Path(p).name for p, _ in training} & {Path(p).name for p, _ in validation}:
        raise ValueError("Image filenames overlap across training and validation")
    grouped = defaultdict(set)
    for name, _ in training + validation:
        relative = Path(name)
        grouped[relative.parent].add(relative.name)
    missing = []
    for parent, expected in grouped.items():
        folder = root / parent
        available = {item.name for item in os.scandir(folder) if item.is_file()} if folder.is_dir() else set()
        missing.extend(str(parent / n) for n in expected - available)
    if missing:
        raise FileNotFoundError(f"{len(missing)} missing images; examples: {missing[:5]}")
    report = {
        "train_images": len(training), "val_images": len(validation), "species": len(labels),
        "train_per_species": train_per_class, "val_per_species": val_per_class,
        "missing_files": 0, "overlapping_filenames": 0,
        "train_json_sha256": digest(root / "train_mini.json"),
        "val_json_sha256": digest(root / "val.json"),
        "category_to_label": labels,
        "species_by_kingdom": dict(Counter(c.get("kingdom", "Unknown") for c in categories.values())),
        "species_by_supercategory": dict(Counter(c.get("supercategory", "Unknown") for c in categories.values())),
        "dimension_summary": {},
        "file_integrity_scope": "All paths checked; official archives verified by MD5 during preparation; sampled images decoded in EDA.",
    }
    for split, meta in [("train", train_meta), ("val", val_meta)]:
        dimensions = [(im.get("width"), im.get("height")) for im in meta["images"]]
        report["dimension_summary"][split] = {
            "images_with_dimensions": sum(w is not None and h is not None for w, h in dimensions),
            "min_width": min((w for w, _ in dimensions if w is not None), default=None),
            "max_width": max((w for w, _ in dimensions if w is not None), default=None),
            "min_height": min((h for _, h in dimensions if h is not None), default=None),
            "max_height": max((h for _, h in dimensions if h is not None), default=None),
        }
    if output:
        atomic_json(output, report)
    return report


def prepare_full_data(data_root):
    root = Path(data_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    status = root / "preparation_status.json"
    try:
        if all((root / p).exists() for p in ["train_mini.json", "val.json", "train_mini", "val"]):
            try:
                report = audit_corpus(root, root / "corpus_manifest.json")
                atomic_json(status, {"state": "READY", "report": report})
                return report
            except FileNotFoundError:
                pass  # Recover an interrupted extraction using verified archives.
        for name, expected_md5 in ARCHIVES.items():
            archive = root / "archives" / name
            archive.parent.mkdir(exist_ok=True)
            atomic_json(status, {"state": "DOWNLOADING", "archive": name})
            if not archive.exists():
                partial = Path(str(archive) + ".part")
                subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error", "--retry", "3", "--continue-at", "-", "--output", str(partial), SOURCE + name], check=True)
                if digest(partial, "md5") != expected_md5:
                    raise ValueError(f"Official MD5 mismatch: {name}; partial archive preserved for diagnosis")
                partial.replace(archive)
            elif digest(archive, "md5") != expected_md5:
                raise ValueError(f"Official MD5 mismatch: {name}")
            atomic_json(status, {"state": "EXTRACTING", "archive": name})
            print(f"Extracting verified archive: {name}", flush=True)
            with tarfile.open(archive, "r|gz") as tar:
                for member in tar:
                    if member.issym() or member.islnk():
                        raise ValueError("Archive links are not supported")
                    destination = (root / member.name).resolve()
                    if not destination.is_relative_to(root):
                        raise ValueError("Unsafe archive path")
                    tar.extract(member, root, filter="data")
        atomic_json(status, {"state": "AUDITING"})
        report = audit_corpus(root, root / "corpus_manifest.json")
        atomic_json(status, {"state": "READY", "report": report})
        return report
    except Exception as error:
        atomic_json(status, {"state": "FAILED", "error": repr(error)})
        raise
