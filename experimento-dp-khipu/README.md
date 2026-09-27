# Biodiversity at Scale: clasificación fine-grained en iNaturalist 2021 Mini

Proyecto 1 de Aprendizaje Profundo (Práctica). Clasificación de 10 000 especies con 500 000 imágenes
de entrenamiento: primero un baseline MLP, después CNN modernas, optimizadores, regularización, transfer
learning y un escenario long-tail construido a partir del dataset balanceado.


> **Para quien lo ve en GitHub (carpeta `experimento-dp-khipu/`):** esta carpeta es autocontenida y es la versión
> entrenada en el clúster Khipu de UTEC. Los notebooks `p1`…`p8` **ya están ejecutados**: se leen
> directamente en GitHub con sus tablas y gráficos. Para reejecutar `p2`–`p7` en tu laptop, sin GPU ni
> dataset, basta con `pip install -r requirements.txt` y abrirlos desde esta carpeta: `resultados/` trae
> las métricas y predicciones de los 27 experimentos. `p1` y `p8` además necesitan las imágenes.
> Los pesos de los modelos (3 GB) no están en el repo.

## Pregunta de investigación e hipótesis

**Pregunta general (enunciado):** ¿qué estrategias de arquitectura, optimización, regularización y
transfer learning mejoran la clasificación fine-grained de especies a gran escala?

**Pregunta específica (propuesta):** con sólo 50 imágenes por especie y una resolución baja (128 px),
¿cuánto del rendimiento se debe al preentrenamiento en ImageNet y cuánto al fine-tuning, y compensa su coste?

**Hipótesis comprobable:**
- H1: Full Fine-Tuning de una ResNet-50 preentrenada (T3) supera a la misma red entrenada desde cero (E04)
  por **≥ 15 puntos de top-1**, con el mismo número de épocas.
- H2: Full Fine-Tuning (T3) supera a Feature Extraction (T1) por **≥ 5 puntos de top-1**, porque los
  rasgos de ImageNet no bastan para separar especies del mismo género.
- H3 (long-tail): Logit Adjustment (L4) recupera **más recall en las especies minoritarias** que
  Weighted Cross-Entropy (L3) y pierde menos en las frecuentes.

Se aceptan o se rechazan en `p8_evaluacion_errores.ipynb` §4. Si se prefiere otra pregunta, basta con
cambiarla aquí: los experimentos cubren todas las partes del enunciado.

## Estructura

Todo vive en la raíz del proyecto:

| Archivo | Qué hace |
|---|---|
| `config.py` | Rutas (dataset, caché, resultados), resolución y semilla. Cada ruta se puede cambiar con una variable de entorno. |
| `descargar_datos.py` | Descarga en paralelo y de forma reanudable `train_mini`, `val` y sus JSON, y los extrae. |
| `descargar_pesos.py` | Descarga los pesos de ImageNet de torchvision (los nodos GPU no tienen internet). |
| `preparar_datos.py` | Construye el índice imagen↔clase↔taxonomía, redimensiona las imágenes a un memmap uint8 de 144 px y construye el long-tail. |
| `datos.py` | Dataset por lotes sobre el memmap, DataLoaders y data augmentation en GPU. |
| `modelos.py` | MLP, CNN de torchvision con la cabeza de 10 000 clases y grupos de capas para fine-tuning. |
| `entrenamiento.py` | Bucle común: optimizador, scheduler, pérdidas, métricas, early stopping, checkpoints y reanudación. |
| `experimentos.py` | **Registro de los 26 experimentos**. `python experimentos.py --listar` los muestra. |
| `preparar.sbatch`, `lanzar.sbatch`, `encolar_todo.sh` | Jobs de SLURM para Khipu. |
| `sincronizar_repo.sh` | Copia la versión publicable (~60 MB) al repo compartido de GitHub. |
| `estado.sh` | Avance: experimentos terminados con sus métricas y época actual del que corre. |
| `00_experimentos.ipynb` | Catálogo: definición de cada experimento, qué cambia respecto a su referencia, arquitectura y estado. |
| `analisis.py` | Funciones comunes de los notebooks: carga de resultados, tablas, curvas. |
| `p1_exploracion.ipynb` | **Parte 1**: EDA, dificultades del problema, pipeline y construcción del long-tail. |
| `p2_mlp.ipynb` | **Parte 2**: baseline MLP (E01, E02). |
| `p3_cnn.ipynb` | **Parte 3**: arquitecturas CNN (E03, E04, E05). |
| `p4_optimizacion.ipynb` | **Parte 4**: optimizadores y sensibilidad al lr (E06–E12). |
| `p5_regularizacion.ipynb` | **Parte 5**: regularización, early stopping y ablación (A1–A6). |
| `p6_transfer.ipynb` | **Parte 6**: feature extraction, fine-tuning parcial y completo (T1–T4). |
| `p7_longtail.ipynb` | **Parte 7**: long-tail y mitigación (L1–L5). |
| `p8_evaluacion_errores.ipynb` | Tabla final, métricas por grupo taxonómico, análisis de errores, hipótesis y conclusiones. |

Los datos **no** están en el repositorio. Por defecto van a `~/40-data/inat` (variable `INAT_ROOT`).

## Cómo reproducirlo (Khipu)

```bash
cd ~/60-experiments/tarea-parcial-deeplearning
PY=~/.venvs/dl/bin/python

# 1. Descargar el dataset (~53 GB; 20-40 min) y los pesos preentrenados. Desde el login.
$PY descargar_datos.py            # reanudable: si se corta, relanzarlo
$PY descargar_pesos.py

# 2. Preprocesar (sólo CPU, ~30 min con 32 núcleos): índice + memmap 144 px + long-tail
sbatch preparar.sbatch            # -> ~/40-data/inat/cache/

# 3. Probar un experimento pequeño (unos 2 min) antes de lanzar todo
sbatch lanzar.sbatch E03 --subconjunto 20000 --val-sub 5000 --epocas 2

# 4. Lanzar todos los experimentos (se ejecutan uno tras otro)
./encolar_todo.sh                 # o solo algunos: ./encolar_todo.sh E01 E03 T3
squeue -u $USER                   # seguimiento;   tail -f logs/inat-E03-*.out

# 5. Analizar: abrir p2…p8 con el kernel dl y ejecutar todo (no necesitan GPU)
```

Si un job llega al límite de 8 h, basta con volver a lanzarlo: reanuda desde `resultados/<ID>/ultimo.pt`.
`encolar_todo.sh` se salta los experimentos que ya tienen `resumen.json`.

### Recursos en Khipu

- La QOS `a-postgrado` permite **1 GPU a la vez**, 32 CPU, 98 GB de RAM y 8 h por job. Por eso los
  experimentos van en cola, uno detrás de otro.
- Las RTX A6000 completas casi siempre están ocupadas por *shards* de otros usuarios, así que los jobs piden
  `--gres=shard:8`: comparten una A6000 de 48 GB. El caudal depende de la carga del nodo.
- Caudal medido en la prueba de humo (A6000 compartida, 128 px, bf16): ResNet-18 ≈ 3 600 img/s,
  ResNet-50 full FT ≈ 1 550 img/s, MLP ≈ 20 000 img/s.
- **Tiempo estimado del total ≈ 20 h de GPU:** ResNet-18 ≈ 40 min por experimento, ResNet-50 ≈ 1,5 h,
  long-tail ≈ 25 min.

## Experimentos

`python experimentos.py --listar` imprime la tabla completa. Cada uno es la receta base más un cambio explícito.

| Parte | IDs | Qué compara |
|---|---|---|
| 2 · MLP | E01, E02 | MLP a 32 px y a 64 px (efecto de la resolución en el número de parámetros) |
| 3 · CNN | E03, E04, E05 | ResNet-18, ResNet-50 y EfficientNet-B0 desde cero, misma receta |
| 4 · Optimización | E03, E06–E12 | SGD, SGD+momentum, Adam y AdamW; barrido de lr para SGDm y Adam |
| 5 · Regularización | A1–A6 | Ablación: BN → Dropout → aumento → weight decay; early stopping en todos |
| 6 · Transfer | T1–T4 (+E04) | Feature extraction, partial FT (layer4) y full FT de ResNet-50 ImageNet |
| 7 · Long-tail | L1–L5 | Balanceado del mismo tamaño, LT sin corrección, CE ponderada, logit adjustment y weighted sampling |

**Receta base:** SGD con momentum 0.9 (Nesterov) y lr 0.1 para batch 256, 1 época de warmup y después
coseno, weight decay 5e-5, 16 épocas, entrada 128×128, recorte aleatorio + flip, mixed precision bf16 y
early stopping (top-1 de validación, paciencia 4, mejora mínima 0,1 pp).

**Métricas:** Top-1, Top-5, Macro-F1 y Weighted-F1 sobre las 100 000 imágenes de validación; train/val
loss y accuracy por época; tiempo, img/s, memoria, parámetros y GFLOPs. En long-tail, además, recall,
precisión y F1 por grupo de frecuencia.

## Salidas

```
resultados/<ID>/config.json      configuración exacta
resultados/<ID>/historial.json   métricas por época
resultados/<ID>/resumen.json     métricas finales del mejor modelo, parámetros, FLOPs y tiempos
resultados/<ID>/preds.npz        top-5 y probabilidades de cada imagen de validación
resultados/<ID>/mejor.pt         pesos del mejor modelo
logs/                            salida de los jobs
```

## Reproducibilidad

- Semilla 42 (`config.SEMILLA`) para Python, NumPy, PyTorch y CUDA. El orden de los lotes se resiembra con
  `semilla + época`.
- El long-tail se construye por código con `numpy.random.default_rng(42)`, nunca a mano.
- `cudnn.benchmark` está activo por velocidad: dos ejecuciones difieren en ~0,01 pp. Para
  determinismo bit a bit: `TORCH_DETERMINISTA=1 sbatch lanzar.sbatch <ID>`.
- La validación nunca se usa para entrenar. Sí se usa para early stopping, como pide el enunciado; ver
  Limitaciones.

## Limitaciones del diseño

- **Resolución 128 px** (almacenada a 144 px con recorte central cuadrado). Es lo que permite ~26
  corridas sobre 500k imágenes en una GPU compartida. A cambio se pierden detalles finos y se penaliza
  el transfer desde ImageNet, que se preentrenó a 224 px.
- **Una semilla por experimento:** diferencias menores de ~0,5 pp pueden no ser significativas.
- **La misma validación** sirve para early stopping y para el reporte, así que la métrica tiene un ligero
  sesgo optimista.
- Los barridos de optimización y regularización se hacen con ResNet-18 por su coste.
