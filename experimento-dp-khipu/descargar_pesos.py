"""Descarga los pesos ImageNet de torchvision a $TORCH_HOME.

Los nodos GPU de Khipu no tienen salida a internet: ejecutar esto UNA vez desde
el nodo de login, antes de lanzar los experimentos de Transfer Learning.
"""
import config  # noqa: F401  (fija TORCH_HOME)
from modelos import PESOS

for nombre, w in PESOS.items():
    w.get_state_dict(progress=True)
    print(f"[ok] {nombre}: {w}")
