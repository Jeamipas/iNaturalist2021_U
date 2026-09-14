"""
Generate Stratified Annotation Subset for 10,000 Classes
Selects exactly N images per class (e.g. N=5 -> 50,000 images, N=10 -> 100,000 images)
from the official train_mini.json (500,000 images, 10,000 species).
"""

import json
import random
from pathlib import Path
from collections import defaultdict

def build_stratified_json(
    source_json_path: Path,
    output_json_path: Path,
    images_per_class: int = 5,
    seed: int = 42
):
    print(f"[*] Leyendo metadatos completos desde: {source_json_path.name}...")
    with open(source_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    images_by_id = {item["id"]: item for item in meta["images"]}
    annotations_by_cat = defaultdict(list)

    for ann in meta["annotations"]:
        annotations_by_cat[ann["category_id"]].append(ann)

    print(f"    Total de categorías encontradas: {len(annotations_by_cat):,}")
    print(f"    Total de imágenes originales:   {len(meta['images']):,}")

    rng = random.Random(seed)
    selected_annotations = []
    selected_image_ids = set()

    for cat_id in sorted(annotations_by_cat.keys()):
        anns = annotations_by_cat[cat_id].copy()
        rng.shuffle(anns)
        chosen = anns[:images_per_class]
        selected_annotations.extend(chosen)
        for c in chosen:
            selected_image_ids.add(c["image_id"])

    selected_images = [
        images_by_id[img_id] for img_id in selected_image_ids if img_id in images_by_id
    ]

    print(f"[*] Subconjunto estratificado generado:")
    print(f"    - Especies cubiertas:        {len(annotations_by_cat):,}")
    print(f"    - Imágenes por especie:      {images_per_class}")
    print(f"    - Total de imágenes finales: {len(selected_images):,}")
    print(f"    - Total de anotaciones:      {len(selected_annotations):,}")

    subset_meta = {
        "images": selected_images,
        "annotations": selected_annotations,
        "categories": meta["categories"]
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(subset_meta, f)

    print(f"[✓] Archivo estratificado guardado en: {output_json_path}")
    print(f"    Tamaño en disco: {output_json_path.stat().st_size / (1024*1024):.2f} MB")


if __name__ == "__main__":
    recursos_dir = Path("c:/Users/jeanp/Documents/antigravity/Naturalist/recursos")
    train_json = recursos_dir / "train_mini.json"
    out_50k_json = recursos_dir / "train_50k_stratified.json"

    build_stratified_json(train_json, out_50k_json, images_per_class=5, seed=42)
