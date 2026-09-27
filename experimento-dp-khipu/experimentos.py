"""Registro de TODOS los experimentos del proyecto.

Cada experimento se define como una modificación explícita de una receta base,
de modo que la diferencia entre dos experimentos se lee en una línea:

    python experimentos.py --listar          # tabla de experimentos
    python experimentos.py E03               # ejecutar uno (en un nodo GPU)
    python experimentos.py E03 --subconjunto 20000 --val-sub 5000 --epocas 2   # depurar

Receta base (CNN desde cero), elegida por ser la estándar de ImageNet:
  SGD + momentum 0.9 (Nesterov), lr 0.1 para batch 256, warmup de 1 época y
  coseno hasta 0, weight decay 5e-5, 16 épocas, entrada 128x128, aumento básico
  (recorte aleatorio + flip), bf16, early stopping con paciencia 4 sobre Top-1.
"""
import argparse
import dataclasses
import sys

from entrenamiento import Config, entrenar

BASE = dict(modelo="resnet18", optimizador="sgdm", lr=0.1, weight_decay=5e-5,
            aumento="basico", epocas=16, batch=256)

# Transfer Learning: ResNet-50 con pesos ImageNet. La cabeza (nueva, aleatoria)
# usa lr 0.1; el backbone 10x menos (0.01) para no destruir lo preentrenado.
TL = dict(BASE, modelo="resnet50", preentrenado=True, factor_lr_backbone=0.1)

EXPERIMENTOS = [
    # ---------------------------------------------------- Parte 2: MLP baseline
    Config("E01", "MLP 32px (baseline)", modelo="mlp", modelo_kw=dict(res=32, ocultas=(2048, 1024)),
           optimizador="adam", lr=1e-3, weight_decay=0.0, aumento="ninguno", batch=512, epocas=16),
    Config("E02", "MLP 64px (efecto de la resolución de entrada)", modelo="mlp",
           modelo_kw=dict(res=64, ocultas=(2048, 1024)),
           optimizador="adam", lr=1e-3, weight_decay=0.0, aumento="ninguno", batch=512, epocas=16),

    # ------------------------------------ Parte 3: arquitecturas CNN desde cero
    Config("E03", "ResNet-18 desde cero (receta base)", **BASE),
    Config("E04", "ResNet-50 desde cero", **dict(BASE, modelo="resnet50")),
    Config("E05", "EfficientNet-B0 desde cero", **dict(BASE, modelo="efficientnet_b0")),

    # ------------------------- Parte 4: optimizadores (ResNet-18, resto = E03)
    # E03 es SGD+momentum lr 0.1. Se barre el lr de SGDm y de Adam para medir
    # la sensibilidad al learning rate de cada optimizador.
    Config("E06", "SGD sin momentum, lr 0.1", **dict(BASE, optimizador="sgd")),
    Config("E07", "SGD+momentum, lr 0.02", **dict(BASE, lr=0.02)),
    Config("E08", "SGD+momentum, lr 0.5", **dict(BASE, lr=0.5)),
    Config("E09", "Adam, lr 1e-3", **dict(BASE, optimizador="adam", lr=1e-3)),
    Config("E10", "Adam, lr 1e-4", **dict(BASE, optimizador="adam", lr=1e-4)),
    Config("E11", "Adam, lr 1e-2", **dict(BASE, optimizador="adam", lr=1e-2)),
    Config("E12", "AdamW, lr 1e-3, wd 0.05 (desacoplado)", **dict(BASE, optimizador="adamw", lr=1e-3, weight_decay=0.05)),

    # ----------------- Parte 5: ablation study de regularización (ResNet-18 SGDm)
    # Se añade un componente cada vez. Early stopping activo en todos.
    Config("A1", "sin BN, sin dropout, sin aumento, sin wd",
           **dict(BASE, bn=False, dropout=0.0, aumento="ninguno", weight_decay=0.0)),
    Config("A2", "+ BatchNorm", **dict(BASE, dropout=0.0, aumento="ninguno", weight_decay=0.0)),
    Config("A3", "+ BN + Dropout 0.3", **dict(BASE, dropout=0.3, aumento="ninguno", weight_decay=0.0)),
    Config("A4", "+ BN + Dropout + aumento fuerte", **dict(BASE, dropout=0.3, aumento="fuerte", weight_decay=0.0)),
    Config("A5", "+ BN + Dropout + aumento fuerte + wd 5e-5",
           **dict(BASE, dropout=0.3, aumento="fuerte", weight_decay=5e-5)),
    Config("A6", "BN + aumento básico (sin dropout, sin wd)", **dict(BASE, aumento="basico", weight_decay=0.0)),

    # ------------------ Parte 6: Transfer Learning (ResNet-50 ImageNet, vs E04)
    Config("T1", "A: Feature Extraction (backbone congelado)", **dict(TL, ajuste="congelado")),
    Config("T2", "B: Partial Fine-Tuning (layer4 + cabeza)", **dict(TL, ajuste="parcial")),
    Config("T3", "C: Full Fine-Tuning", **dict(TL, ajuste="completo")),
    Config("T4", "C + aumento fuerte + label smoothing 0.1 (mejor modelo)",
           **dict(TL, ajuste="completo", aumento="fuerte", label_smoothing=0.1)),

    # --------------- Parte 7: Long-tail (receta de T3; ~125k imágenes de train)
    Config("L1", "balanceado, mismo nº de imágenes que el long-tail (control)", **dict(TL, datos="lt_balanceado")),
    Config("L2", "long-tail sin corrección", **dict(TL, datos="longtail")),
    Config("L3", "long-tail + Weighted Cross-Entropy (1/n)", **dict(TL, datos="longtail", perdida="ce_ponderada")),
    Config("L4", "long-tail + Logit Adjustment (tau=1)", **dict(TL, datos="longtail", perdida="logit_ajustado")),
    Config("L5", "long-tail + Weighted Sampling", **dict(TL, datos="longtail", muestreo="ponderado")),
]
POR_ID = {c.id: c for c in EXPERIMENTOS}


def listar():
    campos = ["modelo", "preentrenado", "ajuste", "optimizador", "lr", "weight_decay", "bn", "dropout",
              "aumento", "datos", "perdida", "muestreo", "epocas"]
    print(f"{'ID':4s} " + " ".join(f"{c[:11]:>11s}" for c in campos) + "  descripción")
    for c in EXPERIMENTOS:
        print(f"{c.id:4s} " + " ".join(f"{str(getattr(c, k))[:11]:>11s}" for k in campos) + f"  {c.descripcion}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("id", nargs="?")
    ap.add_argument("--listar", action="store_true")
    ap.add_argument("--subconjunto", type=int, help="nº de imágenes de train (depuración)")
    ap.add_argument("--val-sub", type=int, help="nº de imágenes de validación (depuración)")
    ap.add_argument("--epocas", type=int)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    if args.listar or not args.id:
        return listar()

    cfg = POR_ID[args.id]
    cambios = {}
    if args.epocas:
        cambios["epocas"] = args.epocas
    if args.subconjunto or args.val_sub or args.epocas:  # no pisar resultados reales
        cambios["id"] = cfg.id + "_debug"
    cfg = dataclasses.replace(cfg, **cambios)
    entrenar(cfg, args.subconjunto, args.val_sub, args.workers)


if __name__ == "__main__":
    sys.exit(main())
