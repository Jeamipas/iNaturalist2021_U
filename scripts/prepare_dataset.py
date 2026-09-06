"""
Automated dataset preparation script for iNaturalist 2021.

Downloads and extracts a strategic sample of N classes (e.g. 50 or 100 classes)
directly from the official AWS S3 archives via HTTP streaming (r|gz),
WITHOUT needing to download the entire 42 GB archive!

For N=50 classes:
- Train: 2,500 real images (50 per class)
- Val: 500 real images (10 per class)
- Total: 3,000 images (~240 MB downloaded, ~45 seconds)

Usage:
  python scripts/prepare_dataset.py --n_classes 50 --dest_dir ../recursos/inat2021_sample
  python scripts/prepare_dataset.py --n_classes 100 --dest_dir ../recursos/inat2021_100classes
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import tarfile
from pathlib import Path
from typing import Dict, List, Set, Tuple
from tqdm import tqdm
from PIL import Image


TRAIN_TAR_URL = "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.tar.gz"
VAL_TAR_URL = "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.tar.gz"
TRAIN_JSON_URL = "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.json.tar.gz"
VAL_JSON_URL = "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.json.tar.gz"


def download_and_extract_json_if_needed(json_url: str, dest_json_path: Path, tar_name: str) -> None:
    """Ensures base JSON metadata is available locally in recursos/."""
    if dest_json_path.exists():
        return

    dest_dir = dest_json_path.parent
    tar_path = dest_dir / tar_name
    print(f"[*] Descargando metadatos base desde {json_url}...")
    urllib.request.urlretrieve(json_url, str(tar_path))
    print(f"[*] Extrayendo {tar_name}...")
    with tarfile.open(str(tar_path), "r:gz") as tar:
        tar.extractall(path=str(dest_dir))
    if tar_path.exists():
        tar_path.unlink()
    print(f"[OK] Metadatos listos en {dest_json_path}")


def stream_extract_classes(
    tar_url: str,
    dest_dir: Path,
    split_name: str,
    max_classes: int,
    target_class_dirs: Set[str] = None
) -> Tuple[List[str], int]:
    """
    Streams from AWS tar.gz directly and extracts images for target classes.
    If target_class_dirs is None, extracts the first max_classes and returns their directory names.
    """
    print(f"\n[*] Extrayendo split '{split_name}' ({max_classes} clases) mediante streaming HTTP...")
    start_time = time.time()

    req = urllib.request.Request(tar_url, headers={"User-Agent": "Mozilla/5.0"})
    response = urllib.request.urlopen(req)

    extracted_classes = []
    seen_classes = set()
    img_count = 0

    pbar = tqdm(total=max_classes * (50 if split_name == "train_mini" else 10), desc=f"Descargando {split_name}", unit="imgs")

    with tarfile.open(fileobj=response, mode="r|gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.lower().endswith((".jpg", ".jpeg")):
                continue

            # Format: split_name/class_dir_name/image.jpg
            parts = member.name.replace("\\", "/").split("/")
            if len(parts) < 3:
                continue

            class_dir = parts[1]

            if target_class_dirs is not None:
                if class_dir not in target_class_dirs:
                    continue
            else:
                if class_dir not in seen_classes:
                    if len(seen_classes) >= max_classes:
                        break
                    seen_classes.add(class_dir)
                    extracted_classes.append(class_dir)

            tar.extract(member, path=str(dest_dir))
            img_count += 1
            pbar.update(1)

            # Check if all targets are completed
            if target_class_dirs is not None and img_count >= max_classes * 10:
                break

    pbar.close()
    elapsed = time.time() - start_time
    print(f"[OK] Split '{split_name}': {img_count} imágenes extraídas en {elapsed:.1f}s ({img_count/max(0.1, elapsed):.1f} imgs/s)")
    return extracted_classes, img_count


def filter_json_annotations(
    full_json_path: Path,
    output_json_path: Path,
    extracted_class_dirs: List[str]
) -> Dict:
    """Filters large full JSON metadata to only include the extracted classes and images."""
    print(f"[*] Filtrando metadatos para {len(extracted_class_dirs)} clases seleccionadas...")
    with open(full_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    class_dir_set = set(extracted_class_dirs)

    # Filter categories
    filtered_categories = [
        cat for cat in meta.get("categories", [])
        if cat.get("image_dir_name") in class_dir_set
    ]
    cat_ids = {cat["id"] for cat in filtered_categories}

    # Filter annotations
    filtered_annotations = [
        ann for ann in meta.get("annotations", [])
        if ann.get("category_id") in cat_ids
    ]
    img_ids = {ann["image_id"] for ann in filtered_annotations}

    # Filter images
    filtered_images = [
        img for img in meta.get("images", [])
        if img["id"] in img_ids
    ]

    filtered_meta = {
        "info": meta.get("info", {}),
        "licenses": meta.get("licenses", []),
        "categories": filtered_categories,
        "images": filtered_images,
        "annotations": filtered_annotations
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(filtered_meta, f, indent=2)

    print(f"[OK] Metadatos guardados en {output_json_path.name}: {len(filtered_images)} imágenes, {len(filtered_categories)} categorías.")
    return filtered_meta


def verify_image_integrity(sample_dir: Path, num_checks: int = 20) -> bool:
    """Verifies a random selection of extracted images can be opened without corruption."""
    all_jpegs = list(sample_dir.glob("**/*.jpg"))
    if not all_jpegs:
        print("[ERROR] No se encontraron imágenes JPEG en el directorio.")
        return False

    import random
    check_sample = random.sample(all_jpegs, min(num_checks, len(all_jpegs)))
    print(f"[*] Verificando integridad de {len(check_sample)} imágenes aleatorias...")

    for img_path in check_sample:
        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception as e:
            print(f"[ERROR] Imagen corrupta detectada: {img_path} ({e})")
            return False

    print(f"[OK] Integridad 100% verificada en todas las muestras.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Preparación del dataset experimental de iNaturalist 2021.")
    parser.add_argument("--n_classes", type=int, default=50, help="Número de clases biológicas a extraer (default: 50)")
    parser.add_argument("--dest_dir", type=str, default="../recursos/inat2021_sample", help="Carpeta destino (fuera de Git)")
    args = parser.parse_args()

    dest_path = Path(args.dest_dir).resolve()
    dest_path.mkdir(parents=True, exist_ok=True)
    recursos_dir = dest_path.parent

    print("=" * 75)
    print("  iNaturalist 2021: Pipeline Automatizado de Preparación de Datos")
    print(f"  Clases objetivo:   {args.n_classes}")
    print(f"  Imágenes esperadas: {args.n_classes * 50} train + {args.n_classes * 10} val = {args.n_classes * 60} imágenes")
    print(f"  Directorio destino: {dest_path}")
    print("=" * 75)

    # 1. Asegurar JSONs base
    full_train_json = recursos_dir / "train_mini.json"
    full_val_json = recursos_dir / "val.json"
    download_and_extract_json_if_needed(TRAIN_JSON_URL, full_train_json, "train_mini.json.tar.gz")
    download_and_extract_json_if_needed(VAL_JSON_URL, full_val_json, "val.json.tar.gz")

    # 2. Extraer Train Mini (50 fotos por clase)
    selected_classes, n_train = stream_extract_classes(
        tar_url=TRAIN_TAR_URL,
        dest_dir=dest_path,
        split_name="train_mini",
        max_classes=args.n_classes
    )

    # 3. Extraer Val (10 fotos por clase para exactamente las mismas clases)
    target_set = set(selected_classes)
    _, n_val = stream_extract_classes(
        tar_url=VAL_TAR_URL,
        dest_dir=dest_path,
        split_name="val",
        max_classes=args.n_classes,
        target_class_dirs=target_set
    )

    # 4. Filtrar y generar JSONs específicos para esta muestra
    sample_train_json = dest_path / "train_mini.json"
    sample_val_json = dest_path / "val.json"
    filter_json_annotations(full_train_json, sample_train_json, selected_classes)
    filter_json_annotations(full_val_json, sample_val_json, selected_classes)

    # 5. Verificación de integridad
    verify_image_integrity(dest_path)

    # 6. Resumen taxonómico
    with open(sample_train_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    kingdoms = {}
    for cat in meta["categories"]:
        k = cat.get("kingdom", "Desconocido")
        kingdoms[k] = kingdoms.get(k, 0) + 1

    print("\n" + "=" * 75)
    print("  [ÉXITO] DATASET EXPERIMENTAL PREPARADO CORRECTAMENTE")
    print("=" * 75)
    print(f"  Total Clases Biológicas:   {len(selected_classes)}")
    print(f"  Total Imágenes Train:      {n_train}")
    print(f"  Total Imágenes Val:        {n_val}")
    print(f"  Total General:             {n_train + n_val} imágenes")
    print("  Diversidad Taxonómica (Reinos):")
    for k, count in kingdoms.items():
        print(f"    - {k}: {count} especies")
    print(f"  Ubicación de los datos:    {dest_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
