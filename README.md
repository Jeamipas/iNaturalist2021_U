# Biodiversity at Scale: Clasificación Fine-Grained de Especies con Deep Learning

Repositorio de código experimental para el proyecto de clasificación de especies a gran escala utilizando el dataset **iNaturalist 2021 (Mini)**.

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
