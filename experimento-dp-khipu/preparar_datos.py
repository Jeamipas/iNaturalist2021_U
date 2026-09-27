"""Prepara iNaturalist 2021 Mini para entrenar rápido. Se ejecuta UNA vez.

    python preparar_datos.py            # índices + imágenes redimensionadas + long-tail
    python preparar_datos.py --procesos 32

Pasos:
 1. Índice: lee los JSON (formato COCO) y produce una tabla por partición con
    ruta, etiqueta y taxonomía completa  ->  cache/{train,val}_indice.csv
 2. Redimensionado: decodifica los 600k JPEG una sola vez y los guarda como un
    array uint8 (N, 144, 144, 3) en disco (np.memmap)  ->  cache/{split}_144.u8
 3. Long-tail: construye con semilla fija el subconjunto desbalanceado y su
    control balanceado del mismo tamaño  ->  cache/longtail.npz

Por qué el paso 2: decodificar un JPEG de ~500 px cuesta ~5 ms de CPU. A
500k imágenes por época serían ~40 min de CPU por época, y la GPU se quedaría
esperando. Leer 144x144x3 bytes ya decodificados es casi gratis: el memmap de
train ocupa 31 GB y, tras la primera época, vive en la caché de páginas del SO.
El precio: se pierde resolución (ver "Limitaciones" en el README).
"""
import argparse
import json
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image

from config import CACHE, DATA_ROOT, RES_ALMACEN, SEMILLA

TAXONOMIA = ["kingdom", "phylum", "class", "order", "family", "genus", "supercategory"]


# ---------------------------------------------------------------- 1. índice
def construir_indice(split):
    """Une images + annotations + categories del JSON en una sola tabla."""
    d = json.loads((DATA_ROOT / f"{split}.json").read_text())
    img = pd.DataFrame(d["images"])[["id", "file_name", "width", "height"]]
    ann = pd.DataFrame(d["annotations"])[["image_id", "category_id"]]
    cat = pd.DataFrame(d["categories"])
    cat["especie"] = cat["name"]
    df = img.merge(ann, left_on="id", right_on="image_id").merge(
        cat[["id", "especie", "common_name", *TAXONOMIA]].rename(columns={"id": "category_id"}),
        on="category_id",
    )
    df = df.rename(columns={"category_id": "etiqueta"}).drop(columns=["image_id"])
    # Orden estable (por etiqueta y nombre de archivo): la fila i del CSV es la
    # imagen i del memmap. No depende del orden del JSON.
    df = df.sort_values(["etiqueta", "file_name"]).reset_index(drop=True)
    assert df["etiqueta"].nunique() == 10_000
    return df


# ---------------------------------------------------------- 2. redimensionar
def _cargar(ruta, res=RES_ALMACEN):
    """Lado corto -> res y recorte central cuadrado res x res."""
    im = Image.open(ruta)
    im.draft("RGB", (res, res))  # el decodificador JPEG reduce 2/4/8x gratis
    im = im.convert("RGB")
    w, h = im.size
    s = res / min(w, h)
    im = im.resize((max(res, round(w * s)), max(res, round(h * s))), Image.BICUBIC)
    w, h = im.size
    l, t = (w - res) // 2, (h - res) // 2
    return np.asarray(im.crop((l, t, l + res, t + res)))


def _trabajador(args):
    ruta_mm, n, ini, rutas = args
    mm = np.memmap(ruta_mm, dtype=np.uint8, mode="r+", shape=(n, RES_ALMACEN, RES_ALMACEN, 3))
    for k, r in enumerate(rutas):
        mm[ini + k] = _cargar(DATA_ROOT / r)
    mm.flush()
    return len(rutas)


def redimensionar(split, df, procesos):
    destino = CACHE / f"{split}_{RES_ALMACEN}.u8"
    hecho = CACHE / f"{split}_{RES_ALMACEN}.ok"
    if hecho.exists():
        print(f"[ok] {destino.name} ya existe")
        return
    n = len(df)
    np.memmap(destino, dtype=np.uint8, mode="w+", shape=(n, RES_ALMACEN, RES_ALMACEN, 3)).flush()
    bloque = 2000
    tareas = [(destino, n, i, df["file_name"].iloc[i:i + bloque].tolist()) for i in range(0, n, bloque)]
    hechas = 0
    with Pool(procesos) as p:
        for m in p.imap_unordered(_trabajador, tareas):
            hechas += m
            print(f"\r  {split}: {hechas}/{n}", end="", flush=True)
    print()
    hecho.touch()


# -------------------------------------------------------------- 3. long-tail
def construir_long_tail(df_train, n_max=50, n_min=1, semilla=SEMILLA):
    """Perfil exponencial de frecuencias (como CIFAR-LT / ImageNet-LT).

    Se baraja el orden de las especies con una semilla y la especie de rango r
    conserva  n(r) = n_max * (n_min/n_max)^(r/(C-1))  imágenes: de 50 a 1, factor
    de desbalance 50. Qué imágenes concretas se quedan también se sortea.
    Barajar las especies evita que el desbalance se alinee con la taxonomía
    (las etiquetas están ordenadas por reino/filo/...).
    """
    rng = np.random.default_rng(semilla)
    C = df_train["etiqueta"].nunique()
    orden = rng.permutation(C)  # orden[r] = especie de rango r
    rango = np.empty(C, dtype=int)
    rango[orden] = np.arange(C)
    n_por_clase = np.floor(n_max * (n_min / n_max) ** (rango / (C - 1)) + 1e-9).astype(int)
    n_por_clase = np.clip(n_por_clase, n_min, n_max)

    idx_lt = []
    for c, grupo in df_train.groupby("etiqueta").groups.items():
        idx_lt.append(rng.choice(np.asarray(grupo), size=n_por_clase[c], replace=False))
    idx_lt = np.sort(np.concatenate(idx_lt))

    # Control: mismo número total de imágenes pero repartido por igual. Así
    # "balanceado vs long-tail" compara sólo la FORMA de la distribución y no
    # la cantidad de datos (el train completo tiene 4x más imágenes).
    k = int(round(len(idx_lt) / C))
    idx_bal = np.sort(np.concatenate([
        rng.choice(np.asarray(g), size=k, replace=False)
        for _, g in df_train.groupby("etiqueta").groups.items()
    ]))

    # Grupos de frecuencia (umbrales escalados de Liu et al. 2019, ImageNet-LT)
    grupo = np.where(n_por_clase >= 20, 0, np.where(n_por_clase >= 5, 1, 2))
    return dict(idx_lt=idx_lt, idx_bal=idx_bal, n_por_clase=n_por_clase, grupo=grupo,
                nombres_grupo=np.array(["frecuentes", "intermedias", "minoritarias"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procesos", type=int, default=32)
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)

    for split in ["val", "train_mini"]:
        f = CACHE / f"{split}_indice.csv"
        if f.exists():
            df = pd.read_csv(f)
        else:
            df = construir_indice(split)
            df.to_csv(f, index=False)
        print(f"{split}: {len(df)} imágenes, {df['etiqueta'].nunique()} especies")
        redimensionar(split, df, args.procesos)

    lt = construir_long_tail(pd.read_csv(CACHE / "train_mini_indice.csv"))
    np.savez(CACHE / "longtail.npz", **lt)
    print(f"long-tail: {len(lt['idx_lt'])} imágenes; control balanceado: {len(lt['idx_bal'])}")
    for g, nombre in enumerate(lt["nombres_grupo"]):
        m = lt["grupo"] == g
        print(f"  {nombre:12s}: {m.sum():5d} clases, {lt['n_por_clase'][m].sum():6d} imágenes")


if __name__ == "__main__":
    main()
