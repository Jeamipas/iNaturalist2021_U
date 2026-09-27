"""Rutas y constantes globales del proyecto.

Todas se pueden sobreescribir con variables de entorno, así el mismo código
corre en Khipu o en otra máquina sin tocarlo.
"""
import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parent

# Dataset crudo (JPEG + JSON). Fuera del repositorio: son ~50 GB.
DATA_ROOT = Path(os.environ.get("INAT_ROOT", Path.home() / "40-data" / "inat"))

# Dataset preprocesado: imágenes redimensionadas en un array uint8 (memmap).
CACHE = Path(os.environ.get("INAT_CACHE", DATA_ROOT / "cache"))

# Resultados de los experimentos (métricas por época, predicciones, checkpoints).
RESULTADOS = Path(os.environ.get("INAT_RESULTADOS", RAIZ / "resultados"))

# Pesos preentrenados de torchvision. Los nodos GPU pueden no tener internet:
# se bajan antes desde el login con `python descargar_pesos.py`.
os.environ.setdefault("TORCH_HOME", str(Path.home() / ".cache" / "torch"))

# Resolución a la que se guardan las imágenes preprocesadas (lado del cuadrado).
# El entrenamiento recorta a RES_ENTRADA; la diferencia deja margen para el
# recorte aleatorio (data augmentation) sin volver a leer el JPEG.
RES_ALMACEN = 144
RES_ENTRADA = 128

SEMILLA = 42
NUM_CLASES = 10_000
