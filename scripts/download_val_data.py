"""
Script to download and extract the official iNaturalist 2021 Validation Dataset.

Sources (AWS Open Data / Caltech / Visipedia):
- val.tar.gz:        https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.tar.gz (~8.4 GB, 100k images)
  MD5: f6f6e0e242e3d4c9569ba56400938afc
- val.json.tar.gz:   https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.json.tar.gz (~25 MB)

Usage:
  python scripts/download_val_data.py --dest_dir ../recursos/inat_val
  python scripts/download_val_data.py --dest_dir ../recursos/inat_val --json_only
"""

import argparse
import hashlib
import os
import sys
import tarfile
from pathlib import Path
import urllib.request
from tqdm import tqdm


URLS = {
    "val_json": {
        "url": "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.json.tar.gz",
        "filename": "val.json.tar.gz",
        "expected_md5": None,
        "desc": "Validation Annotations JSON (~25 MB)"
    },
    "val_images": {
        "url": "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.tar.gz",
        "filename": "val.tar.gz",
        "expected_md5": "f6f6e0e242e3d4c9569ba56400938afc",
        "desc": "Validation Images (100k images, ~8.4 GB)"
    }
}


class DownloadProgressBar(tqdm):
    """tqdm wrapper for urllib.request hook."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: Path, desc: str) -> None:
    """Downloads a file with a live progress bar showing MB and speed."""
    print(f"\n[*] Descargando: {desc}")
    print(f"    URL: {url}")
    print(f"    Destino: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print(f"    [!] El archivo {output_path.name} ya existe. Omitiendo descarga...")
        return

    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, filename=str(output_path), reporthook=t.update_to)
    print(f"    [OK] Descarga completada: {output_path.name}")


def verify_md5(file_path: Path, expected_md5: str) -> bool:
    """Computes and validates MD5 checksum."""
    if expected_md5 is None:
        return True

    print(f"\n[*] Verificando integridad MD5 de {file_path.name}...")
    hasher = hashlib.md5()
    total_bytes = file_path.stat().st_size

    with open(file_path, "rb") as f, tqdm(total=total_bytes, unit="B", unit_scale=True, desc="Verificando MD5") as pbar:
        while chunk := f.read(1024 * 1024 * 8):  # 8 MB chunks
            hasher.update(chunk)
            pbar.update(len(chunk))

    calculated = hasher.hexdigest()
    if calculated.lower() == expected_md5.lower():
        print(f"    [OK] MD5 Válido: {calculated}")
        return True
    else:
        print(f"    [ERROR] MD5 Discrepancia! Esperado: {expected_md5}, Obtenido: {calculated}")
        return False


def extract_tar(archive_path: Path, extract_dir: Path) -> None:
    """Extracts a .tar.gz archive with progress reporting."""
    print(f"\n[*] Descomprimiendo {archive_path.name} en {extract_dir}...")
    extract_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, "r:gz") as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc=f"Extrayendo {archive_path.name}", unit="archivos"):
            tar.extract(member, path=str(extract_dir))
    print(f"    [OK] Extracción completada para {archive_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Descarga del dataset oficial de Validación iNaturalist 2021.")
    parser.add_argument(
        "--dest_dir",
        type=str,
        default="../recursos/inat_val",
        help="Carpeta destino (por defecto fuera del repo en ../recursos/inat_val)"
    )
    parser.add_argument(
        "--json_only",
        action="store_true",
        help="Descargar únicamente el archivo de anotaciones val.json.tar.gz (~25 MB)"
    )
    parser.add_argument(
        "--no_extract",
        action="store_true",
        help="Solo descargar sin extraer los .tar.gz"
    )

    args = parser.parse_args()
    dest_path = Path(args.dest_dir).resolve()
    dest_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  iNaturalist 2021 - Descarga de Datos de Validación Oficial")
    print(f"  Carpeta Destino: {dest_path}")
    print("=" * 70)

    # 1. Anotaciones JSON (~25 MB)
    json_info = URLS["val_json"]
    json_tar = dest_path / json_info["filename"]
    download_file(json_info["url"], json_tar, json_info["desc"])
    if not args.no_extract:
        extract_tar(json_tar, dest_path)

    # 2. Imágenes de Validación (100k imágenes, ~8.4 GB)
    if not args.json_only:
        img_info = URLS["val_images"]
        img_tar = dest_path / img_info["filename"]
        download_file(img_info["url"], img_tar, img_info["desc"])
        if img_info["expected_md5"]:
            valid = verify_md5(img_tar, img_info["expected_md5"])
            if not valid:
                print("    [ALERTA] El archivo descargado puede estar corrupto.")
        if not args.no_extract:
            extract_tar(img_tar, dest_path)

    print("\n" + "=" * 70)
    print("  [EXITO] Proceso de validación de datos completado.")
    print(f"  Archivos listos en: {dest_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
