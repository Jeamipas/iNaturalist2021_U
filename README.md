# Biodiversity at Scale: Clasificación Fine-Grained de Especies con Deep Learning

Repositorio de código experimental para el proyecto de clasificación de especies a gran escala utilizando el dataset **iNaturalist 2021 (Mini)**.

## Entregable final: 26 históricos y cuatro ConvNeXt-Tiny

**Integrantes:** Jeanpier Garay, Diego Pacheco y Jesus Castillo.

El punto de entrada es [notebooks/00_entregable_reproducible.ipynb](notebooks/00_entregable_reproducible.ipynb),
ejecutado con tablas, EDA, curvas y ejemplos de errores. Contiene las fuentes
necesarias para preparar datos, gestionar Slurm, entrenar, evaluar y analizar.
Su modo predeterminado es `review`: funciona con la evidencia publicada, sin
GPU, conexión a Khipu, descarga del corpus ni nuevos entrenamientos.

Se conservan los archivos históricos de `experimento-dp-khipu/` y se añaden
cuatro ejecuciones completas en [outputs/convnext4_adam_muon_v3](outputs/convnext4_adam_muon_v3).
Cada carpeta N01–N04 incluye configuración, historial de 16 épocas, resumen,
predicciones de las 100000 imágenes de validación e identidad de inicialización.
Las 26 ejecuciones históricas y las cuatro nuevas se verifican sobre las mismas
etiquetas, con diez imágenes por cada una de las 10000 especies. El catálogo
histórico define L1, pero no publica resultados: no se cuenta como ejecución.

| ID | Arquitectura | Inicialización | Optimizador | Top-1 (%) | Top-5 (%) | Macro-F1 (%) | Checkpoint |
|---|---|---|---|---:|---:|---:|---:|
| N01 | ConvNeXt-Tiny | Desde cero | Adam | 19.222 | 37.155 | 18.221 | 15 |
| N02 | ConvNeXt-Tiny | Desde cero | Muon + Adam auxiliar | 29.558 | 50.514 | 28.583 | 16 |
| N03 | ConvNeXt-Tiny | ImageNet-1K V1, ajuste completo | Adam | 58.263 | 77.801 | 58.125 | 15 |
| N04 | ConvNeXt-Tiny | ImageNet-1K V1, ajuste completo | Muon + Adam auxiliar | 57.356 | 77.143 | 57.169 | 15 |

Cada modelo tiene 35510128 parámetros (aproximadamente 0.142 GB de pesos FP32;
no representa VRAM total de entrenamiento). Train usa las 500000 imágenes de
Mini. Los pares N01/N02 y N03/N04 comparten arquitectura, inicialización, datos,
aumentos y criterio de selección. Se conservan entrada 128 px, caché 144 px,
batch 256, BF16, warmup de una época, coseno por paso, paciencia 4 y mejora
estricta >0.001. Las tasas son Adam 0.001 y Muon 0.02, con Adam auxiliar 0.001;
el backbone preentrenado usa un factor 0.1. La extensión incluye el último lote
de 32 imágenes; el entrenamiento histórico descartaba ese lote. Los tiempos
son descriptivos por diferencias de GPU y contención. Early stopping estuvo
habilitado, pero no detuvo las cuatro ejecuciones antes de las 16 épocas.

### Revisión local reproducible

```sh
python -m pip install -r requirements-full.txt
```

Abrir el notebook con ese entorno y ejecutar de arriba abajo. `review` recalcula
Top-1, Top-5, Macro/Weighted-F1 de los 30 NPZ y verifica hashes, inicializaciones
y selección visual de N03. La EDA archivada proviene del reporte sobre el corpus
completo en Khipu; `prepare`/`report` la recalculan con el dataset disponible.
Los resultados quedan en `delivery_audit.json`, `comparison.csv` y las curvas
del directorio de resultados. `notebook_verification.json` registra la última
ejecución local completa de las ocho celdas de código, sin errores ni entrenamiento.

El notebook incorpora 23 archivos de fuentes/evidencia con checksum. Para
regenerar su fuente desde los módulos del repositorio:

```sh
python scripts/build_deliverable.py
```

Este comando elimina las salidas ejecutadas; volver a ejecutar el notebook
en `review` para recuperarlas. `evidence_manifest.json` registra archivos de
fuentes y resultados. `environment-freeze.txt` conserva el entorno efectivo
del entrenamiento; PyTorch 2.6.0 y torchvision 0.21.0 con CUDA se instalan
mediante `scripts/full_setup.sh` cuando se usa Khipu.

### Repetir el entrenamiento

En la celda de configuración, usar otro `RUN_ID` y otro `REMOTE_ROOT` propios,
actualizar `SETTINGS['run_id']` y `SETTINGS['remote_root']`, y seleccionar
`MODE='submit'`. Para un proyecto independiente, los valores predeterminados
`reuse_project_root=None` y `data_ready_job=None` preparan corpus y entorno
desde cero. La reserva original fue `shard:rtxa6000:8` en `ds001`; para solicitar
una GPU completa puede configurarse `gpu_gres='gpu:rtxa6000:1'` y `gpu_node=None`.
El envío, preparación, smoke CUDA, tareas y reporte pasan por Slurm. Los pesos
y checkpoints permanecen en Khipu y están excluidos de Git; el notebook permite
regenerarlos. Repetir bit a bit requiere mismo entorno/GPU y operadores deterministas.

Una semilla, validación reutilizada para selección/reporte y ajuste desigual
de hiperparámetros limitan las conclusiones. Estos resultados no demuestran
el máximo potencial de los optimizadores. Esta entrega corresponde al protocolo
original comparable con Diego; otros protocolos no se mezclan en sus tablas.

> **Curso**: Aprendizaje Profundo – Práctica  
> **Programa**: Maestría de Investigación en Inteligencia Artificial, UTEC Posgrado  
> **Profesor**: Dra. Aurea Soriano-Vargas  

---

## 📌 Descripción del Proyecto

El reconocimiento automático de especies biológicas a partir de imágenes representa un desafío crítico de visión por computador (*fine-grained visual classification*). A diferencia de tareas estándar de clasificación (e.g., ImageNet, CIFAR), este problema exhibe:
- **Alta similitud inter-especies**: Especies pertenecientes al mismo género u orden comparten fenotipos casi indistinguibles.
- **Gran variabilidad intra-especie**: Diferencias drásticas por etapa ontogenética (larva vs. adulto), dimorfismo sexual, variación estacional, pose y fondo.
- **Desafío Long-Tail**: Distribución con pocas especies comunes y una extensa cola de especies con escasas observaciones.

El objetivo de este proyecto es evaluar de forma rigurosa y controlada el impacto de:
1. **Modelos Base**: Multilayer Perceptron (MLP) y demostración de limitaciones espaciales.
2. **Arquitecturas CNN Modernas**: Conexiones residuales (ResNet) y diseños modernos (ConvNeXt / EfficientNet).
3. **Estrategias de Optimización**: Dinámica de convergencia con SGD + Momentum vs. AdamW.
4. **Regularización y Generalización**: Batch Normalization, Dropout, Weight Decay y Data Augmentation específico para biodiversidad.
5. **Transfer Learning y Fine-Tuning**: Comparación controlada entre Feature Extraction, Partial Fine-Tuning y Full Fine-Tuning.
6. **Biodiversity Long-Tail Challenge**: Modelado de desbalance y técnicas de mitigación (Focal Loss, Balanced Sampling, Weighted Loss).

---

## 📂 Estructura del Repositorio

```
iNaturalist2021_U/
│
├── .gitignore
├── README.md                          <- Documentación del proyecto y reproducción
├── requirements.txt                   <- Dependencias del proyecto
│
├── configs/                           <- Configuraciones modulares de experimentos
├── notebooks/                         <- Cuadernos Jupyter reproducibles
│   ├── 01_eda_dataset.ipynb           <- Exploración del dataset y taxonomía
│   ├── 02_baseline_mlp.ipynb          <- Baseline MLP y justificación
│   ├── 03_cnn_and_optimization.ipynb  <- CNNs modernas y optimizadores
│   ├── 04_regularization_ablation.ipynb <- Estudio de ablación
│   ├── 05_transfer_learning.ipynb     <- Feature extraction vs Fine-tuning
│   ├── 06_long_tail_challenge.ipynb   <- Desbalance extremo y mitigación
│   └── 07_error_analysis.ipynb        <- Análisis cuantitativo y cualitativo de errores
│
├── src/                               <- Código fuente modular
│   ├── data/                          <- Datasets, transformaciones y partición long-tail
│   ├── models/                        <- MLP, CNNs y wrappers de modelos preentrenados
│   ├── training/                      <- Loops de entrenamiento, early stopping y pérdidas
│   └── utils/                         <- Métricas (Top-1, Top-5, Macro F1), semillas y visualización
│
└── tests/                             <- Pruebas automatizadas de integridad de pipeline
```

---

## ⚙️ Instalación y Requisitos

### Requisitos de Entorno
- Python >= 3.10
- PyTorch >= 2.0 (con soporte CUDA si se dispone de GPU)
- torchvision, scikit-learn, matplotlib, seaborn, pyyaml, tqdm

```bash
git clone https://github.com/Jeamipas/iNaturalist2021_U.git
cd iNaturalist2021_U
pip install -r requirements.txt
```

---

## 📊 Dataset: iNaturalist 2021 Mini

Para descargar los datos oficiales requeridos:
- **Entrenamiento Mini (~42 GB)**: `https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.tar.gz`
- **Anotaciones Train**: `https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.json.tar.gz`
- **Validación (~8.4 GB)**: `https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.tar.gz`
- **Anotaciones Val**: `https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.json.tar.gz`

*Nota: Las imágenes y anotaciones no se incluyen en el repositorio de código.*
