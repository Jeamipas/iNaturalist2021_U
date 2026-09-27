"""Descarga y extrae iNaturalist 2021 Mini (train_mini + val) y sus anotaciones.

Uso:
    python descargar_datos.py                   # todo, en config.DATA_ROOT
    python descargar_datos.py --hilos 16        # más conexiones en paralelo
    python descargar_datos.py --solo-json       # sólo anotaciones (rápido)

Por qué no un simple `wget`: S3 entrega ~8 MB/s por conexión, así que los 42 GB
de train_mini tardarían ~1.5 h. Pidiendo trozos del archivo en paralelo (cabecera
HTTP `Range`) la descarga escala casi linealmente con el número de conexiones.

La descarga es reanudable: cada trozo terminado queda anotado en
`<archivo>.progreso.json`; si el proceso muere, al relanzarlo sólo baja lo que falta.
Sólo usa la biblioteca estándar (el venv no trae `requests`).
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from config import DATA_ROOT

BASE = "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/"
ARCHIVOS = {
    # nombre              carpeta que produce al extraer
    "train_mini.json.tar.gz": "train_mini.json",
    "val.json.tar.gz": "val.json",
    "train_mini.tar.gz": "train_mini",
    "val.tar.gz": "val",
}
TROZO = 256 * 2**20  # 256 MiB por petición


def tamano_remoto(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def bajar_trozo(url, destino, ini, fin, reintentos=8):
    """Descarga los bytes [ini, fin] y los escribe en su sitio con pwrite."""
    for intento in range(reintentos):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={ini}-{fin}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                fd = os.open(destino, os.O_WRONLY)
                try:
                    pos = ini
                    while True:
                        buf = r.read(4 * 2**20)
                        if not buf:
                            break
                        os.pwrite(fd, buf, pos)
                        pos += len(buf)
                finally:
                    os.close(fd)
            if pos != fin + 1:
                raise IOError(f"trozo incompleto {pos}/{fin + 1}")
            return ini
        except Exception as e:  # red inestable: reintentar con espera creciente
            print(f"  reintento {intento + 1} trozo {ini}: {e}", flush=True)
            time.sleep(2 ** intento)
    raise RuntimeError(f"no se pudo bajar el trozo {ini}-{fin}")


def descargar(nombre, hilos):
    url = BASE + nombre
    destino = DATA_ROOT / nombre
    progreso_f = DATA_ROOT / (nombre + ".progreso.json")
    total = tamano_remoto(url)

    if destino.exists() and destino.stat().st_size == total and not progreso_f.exists():
        print(f"[ok] {nombre} ya descargado ({total / 1e9:.2f} GB)")
        return destino

    hechos = set(json.loads(progreso_f.read_text())) if progreso_f.exists() else set()
    if not destino.exists():
        with open(destino, "wb") as f:
            f.truncate(total)  # archivo del tamaño final; los trozos se escriben en su offset

    trozos = [(i, min(i + TROZO, total) - 1) for i in range(0, total, TROZO)]
    pendientes = [t for t in trozos if t[0] not in hechos]
    print(f"[..] {nombre}: {total / 1e9:.2f} GB, {len(pendientes)}/{len(trozos)} trozos pendientes")

    lock = threading.Lock()
    t0, bajado = time.time(), 0
    with ThreadPoolExecutor(hilos) as ex:
        futs = {ex.submit(bajar_trozo, url, destino, a, b): (a, b) for a, b in pendientes}
        for fut in as_completed(futs):
            a, b = futs[fut]
            fut.result()
            with lock:
                hechos.add(a)
                progreso_f.write_text(json.dumps(sorted(hechos)))
                bajado += b - a + 1
                v = bajado / (time.time() - t0) / 1e6
                print(f"     {nombre}: {len(hechos)}/{len(trozos)} trozos  {v:.0f} MB/s", flush=True)
    progreso_f.unlink()
    return destino


def extraer(nombre, carpeta):
    if (DATA_ROOT / carpeta).exists():
        print(f"[ok] {carpeta} ya extraído")
        return
    print(f"[..] extrayendo {nombre}")
    # pigz descomprime en paralelo; si no está, tar usa gzip normal
    cmd = ["tar", "-I", "pigz", "-xf", nombre] if _hay("pigz") else ["tar", "-xzf", nombre]
    subprocess.run(cmd, cwd=DATA_ROOT, check=True)


def _hay(prog):
    return subprocess.run(["which", prog], capture_output=True).returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hilos", type=int, default=12)
    ap.add_argument("--solo-json", action="store_true")
    ap.add_argument("--borrar-tar", action="store_true", help="borra los .tar.gz tras extraer")
    args = ap.parse_args()

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Destino: {DATA_ROOT}")

    # Si ya existe la validación extraída con el nombre que usa torchvision
    # (`2021_valid`), la reutilizamos con un enlace en vez de bajar 8.4 GB otra vez.
    if not (DATA_ROOT / "val").exists() and (DATA_ROOT / "2021_valid").is_dir():
        os.symlink("2021_valid", DATA_ROOT / "val")
        print("[ok] val -> 2021_valid (validación ya presente)")

    for nombre, carpeta in ARCHIVOS.items():
        if args.solo_json and not nombre.endswith(".json.tar.gz"):
            continue
        if (DATA_ROOT / carpeta).exists():
            print(f"[ok] {carpeta} ya presente")
            continue
        descargar(nombre, args.hilos)
        extraer(nombre, carpeta)
        if args.borrar_tar:
            (DATA_ROOT / nombre).unlink()

    # Comprobación mínima: número de imágenes por partición
    for carpeta, esperado in [("train_mini", 500_000), ("val", 100_000)]:
        d = DATA_ROOT / carpeta
        if d.exists():
            n = sum(len(fs) for _, _, fs in os.walk(d, followlinks=True))
            estado = "ok" if n == esperado else "¡OJO!"
            print(f"[{estado}] {carpeta}: {n} imágenes (esperadas {esperado})")


if __name__ == "__main__":
    sys.exit(main())
