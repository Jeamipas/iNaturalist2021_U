"""Funciones comunes a los notebooks de análisis (p2…p8).

Qué es este módulo
------------------
Los notebooks NO entrenan: leen lo que cada job de entrenamiento dejó en
disco y lo convierten en tablas y gráficos. Estas funciones hacen esa lectura
y ese formateo. Están aquí, y no copiadas en cada notebook, para que las
ocho partes calculen las métricas exactamente de la misma forma.

De dónde salen los datos
------------------------
Cada experimento (E01, T3, …) escribe al terminar, en resultados/<ID>/:

    config.json     hiperparámetros usados            (lo escribe entrenamiento.entrenar)
    historial.json  una entrada por época: train/val loss, top-1, tiempos…
    resumen.json    métricas finales del MEJOR modelo, parámetros, GFLOPs, tiempo total
    preds.npz       predicciones top-5 sobre validación (lo usan p7 y p8 directamente)

Uso típico en un notebook
-------------------------
    from analisis import *
    R = resultados()                                   # carga todo lo terminado
    definiciones([("E02", "E01")])                     # qué cambia entre experimentos
    curvas(R, ["E01", "E02"], "MLP")                   # 4 gráficos de entrenamiento
    tabla(R, ["E01", "E02"]).style.format(FMT)         # métricas finales

Índice de funciones
-------------------
    Carga         cargar, resultados, estado
    Definiciones  diferencias, definiciones   (qué configura cada experimento)
    Resultados    tabla, curvas, epoca_para   (qué obtuvo cada experimento)
"""
import dataclasses
import json
import os

import matplotlib.pyplot as plt
import numpy as np  # no se usa aquí: se reexporta a los notebooks con `from analisis import *`
import pandas as pd

from config import RESULTADOS
from entrenamiento import Config
from experimentos import EXPERIMENTOS, POR_ID

# Si se define la variable de entorno INAT_SUFIJO=_debug, se leen las carpetas
# resultados/<ID>_debug (las corridas de prueba con pocas imágenes) en vez de
# las reales. Sin la variable, SUFIJO = "" y se leen resultados/<ID>.
SUFIJO = os.environ.get("INAT_SUFIJO", "")

# Configuración con todos los valores por defecto de entrenamiento.Config.
# Sirve de "referencia vacía" para ver qué define un experimento por sí mismo.
DEFECTO = Config(id="(defecto)")

# Estilo común de todos los gráficos: algo más de resolución y sin los bordes
# superior y derecho (que no aportan información).
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})

# Formato de cada columna de `tabla` para mostrarla con .style.format(FMT):
# las métricas como porcentaje con 1 decimal, parámetros en millones, etc.
FMT = {c: "{:.1%}" for c in ["top1", "top5", "f1_macro", "f1_weighted", "train_top1", "brecha_top1"]} | {
    "params_M": "{:.1f}", "entrenables_M": "{:.1f}", "GFLOPs": "{:.2f}", "min_por_ep": "{:.1f}",
    "min_total": "{:.0f}", "brecha_loss": "{:.2f}", "img_s": "{:.0f}"}


# ======================================================================
# Carga de resultados
# ======================================================================
def cargar(id_):
    """Lee los resultados de UN experimento terminado.

    Parámetros
        id_ : identificador del experimento, p. ej. "E01".

    Devuelve
        None si el experimento no ha terminado (no existe su resumen.json).
        Si terminó, un diccionario con tres claves:
          "resumen" : dict con el contenido de resumen.json (métricas finales),
                      p. ej. {"val_top1": 0.0109, "parametros": 18647824, ...}
          "hist"    : DataFrame de pandas con una fila por época y columnas
                      epoca, train_loss, train_top1, val_loss, val_top1, val_top5,
                      val_f1_macro, lr, t_train, img_por_s, grad_norma_media, ...
          "dir"     : ruta de la carpeta, para abrir otros archivos (preds.npz).

    Ejemplo
        r = cargar("E01")
        r["resumen"]["val_top1"]     # 0.01086  (top-1 final)
        r["hist"].val_loss.min()     # la menor val loss de todas las épocas
    """
    d = RESULTADOS / (id_ + SUFIJO)
    if not (d / "resumen.json").exists():
        return None
    return dict(resumen=json.loads((d / "resumen.json").read_text()),
                hist=pd.DataFrame(json.loads((d / "historial.json").read_text())), dir=d)


def resultados():
    """Carga TODOS los experimentos terminados de una vez.

    Recorre la lista de experimentos de experimentos.py y llama a `cargar` con
    cada uno; los que aún no terminaron se omiten.

    Devuelve
        dict {ID: resultado de cargar(ID)}. Por convención se guarda en la
        variable R al inicio de cada notebook:

            R = resultados()
            "T3" in R                    # ¿terminó T3?
            R["T3"]["resumen"]["val_top1"]
            R["T3"]["hist"]              # su historial por épocas

    Las demás funciones reciben este R, así el disco se lee una sola vez.
    """
    return {c.id: r for c in EXPERIMENTOS if (r := cargar(c.id)) is not None}


def estado(id_):
    """Dice en qué punto está un experimento, como texto.

    Devuelve
        "terminado"            si existe resumen.json;
        "en curso (época N)"   si solo existe historial.json (el job está
                               corriendo o se cortó a la mitad);
        "pendiente"            si todavía no hay nada.

    Se usa en la columna "estado" de `definiciones`.
    """
    d = RESULTADOS / (id_ + SUFIJO)
    if (d / "resumen.json").exists():
        return "terminado"
    if (d / "historial.json").exists():
        return f"en curso (época {len(json.loads((d / 'historial.json').read_text()))})"
    return "pendiente"


# ======================================================================
# Definición de los experimentos (qué se configuró)
# ======================================================================
def _dict(c):
    """Convierte una Config en un dict de campos comparables.

    Función interna (el guion bajo inicial indica que no se usa desde los
    notebooks). `modelo_kw` es un dict dentro de otro dict; se pasa a texto
    para poder compararlo con `!=` y mostrarlo en una tabla.
    """
    d = dataclasses.asdict(c)
    d["modelo_kw"] = str(d["modelo_kw"]) if d["modelo_kw"] else ""
    return d


def diferencias(id_, ref):
    """Qué campos de la configuración cambian entre dos experimentos.

    Es la base de las "comparaciones controladas": si `id_` difiere de `ref`
    en un único campo, cualquier diferencia de resultado se atribuye a él.

    Parámetros
        id_ : experimento a describir, p. ej. "E02".
        ref : experimento de referencia, p. ej. "E01". Si no es un ID válido
              (p. ej. "(defecto)"), se compara con los valores por defecto.

    Devuelve
        dict {campo: "valor_en_ref → valor_en_id_"}, omitiendo id y descripción.

    Ejemplos
        diferencias("E02", "E01")   # {'modelo_kw': "{'res': 32, ...} → {'res': 64, ...}"}
        diferencias("E04", "E03")   # {'modelo': 'resnet18 → resnet50'}
        diferencias("T2", "T1")     # {'ajuste': 'congelado → parcial'}
    """
    a, b = _dict(POR_ID[id_]), _dict(POR_ID[ref] if ref in POR_ID else DEFECTO)
    return {k: f"{b[k]} → {a[k]}" for k in a if k not in ("id", "descripcion") and a[k] != b[k]}


def definiciones(pares):
    """Tabla que explica los experimentos de una parte del trabajo.

    Parámetros
        pares : lista de tuplas (ID, referencia). Cada experimento se describe
                por lo que cambia respecto a su referencia.

    Devuelve
        DataFrame indexado por ID con columnas:
          referencia · descripción · estado · cambios

    Ejemplo (p5_regularizacion: cada paso de la ablación frente al anterior)
        definiciones([("A2", "A1"), ("A3", "A2")])
        #     referencia  descripción         estado      cambios
        # A2  A1          + BatchNorm         terminado   bn: False → True
        # A3  A2          + BN + Dropout 0.3  terminado   dropout: 0.0 → 0.3
    """
    filas = [dict(ID=i, referencia=ref, descripción=POR_ID[i].descripcion, estado=estado(i),
                  cambios="; ".join(f"{k}: {v}" for k, v in diferencias(i, ref).items()) or "(referencia)")
             for i, ref in pares]
    return pd.DataFrame(filas).set_index("ID")


# ======================================================================
# Resultados (qué se obtuvo)
# ======================================================================
def tabla(R, ids):
    """Tabla de métricas finales de varios experimentos, una fila por ID.

    Parámetros
        R   : el diccionario de `resultados()`.
        ids : lista de IDs a incluir, en el orden deseado. Los que no han
              terminado se saltan sin error.

    Devuelve
        DataFrame indexado por ID (vacío si ninguno terminó). Columnas:

        Rendimiento en validación (con los pesos de la MEJOR época):
          top1, top5          accuracy top-1 y top-5
          f1_macro            F1 medio por especie (todas pesan igual)
          f1_weighted         F1 ponderado por nº de imágenes (≈ macro: val es balanceada)
        Generalización:
          train_top1          accuracy en train en esa misma época
          brecha_top1         train_top1 − top1   (grande = sobreajuste)
          brecha_loss         val_loss − train_loss en esa época
        Entrenamiento:
          mejor_ep            época elegida por early stopping
          épocas              épocas realmente corridas (< 16 si paró antes)
        Coste:
          params_M            parámetros totales, en millones
          entrenables_M       los que se actualizan (menos en transfer congelado/parcial)
          GFLOPs              cómputo de una pasada de 1 imagen de 128 px
          img_s               imágenes por segundo al entrenar (mediana, sin la 1.ª
                              época, que incluye la carga de datos al caché)
          min_total           minutos totales de entrenamiento
          gpu                 GPU donde corrió: los tiempos solo son comparables
                              entre filas con la misma GPU

    Para verla con porcentajes: tabla(R, ids).style.format(FMT)
    """
    filas = []
    for i in ids:
        if i not in R:
            continue
        r, h = R[i]["resumen"], R[i]["hist"]
        b = r["mejor_epoca"] - 1  # posición (desde 0) de la mejor época en el historial
        filas.append(dict(
            ID=i, descripción=POR_ID[i].descripcion,
            top1=r["val_top1"], top5=r["val_top5"], f1_macro=r["val_f1_macro"], f1_weighted=r["val_f1_weighted"],
            train_top1=h.train_top1.iloc[b], brecha_top1=h.train_top1.iloc[b] - r["val_top1"],
            brecha_loss=r["brecha_loss"], mejor_ep=r["mejor_epoca"], épocas=r["epocas_corridas"],
            params_M=r["parametros"] / 1e6, entrenables_M=r["parametros_entrenables"] / 1e6,
            GFLOPs=r["gflops"], img_s=h.img_por_s.iloc[1:].median() if len(h) > 1 else h.img_por_s.iloc[0],
            min_total=r["tiempo_train_s"] / 60, gpu=r["gpu"].replace("NVIDIA ", "").replace("A100-PCIE-40GB ", "A100 ")))
    return pd.DataFrame(filas).set_index("ID") if filas else pd.DataFrame()


def curvas(R, ids, titulo, etiquetas=None):
    """Dibuja las curvas de entrenamiento de varios experimentos superpuestas.

    Es el gráfico que pide el enunciado (Parte 4): cuatro paneles lado a lado,
    una línea por experimento, eje x = época:

        Training loss | Validation loss | Training accuracy | Validation accuracy

    Cómo leerlo:
      * train loss baja y val loss sube        → sobreajuste (p. ej. E01)
      * ambas bajan juntas                     → aprende y generaliza
      * val accuracy aún subiendo al final     → faltaron épocas
      * curva que sube antes que otra          → converge más rápido

    Parámetros
        R         : el diccionario de `resultados()`.
        ids       : experimentos a superponer; los no terminados se omiten.
        titulo    : título general de la figura.
        etiquetas : opcional, {ID: texto} para cambiar la leyenda. Por defecto
                    la leyenda es "ID · descripción".

    No devuelve nada: muestra la figura. Si ninguno terminó, lo avisa con print.
    """
    ids = [i for i in ids if i in R]
    if not ids:
        print("(sin resultados todavía)")
        return
    fig, axs = plt.subplots(1, 4, figsize=(17, 3.5))
    for i in ids:
        h = R[i]["hist"]
        lab = (etiquetas or {}).get(i, f"{i} · {POR_ID[i].descripcion}")
        for ax, col in zip(axs, ["train_loss", "val_loss", "train_top1", "val_top1"]):
            ax.plot(h.epoca, h[col], marker=".", label=lab)
    for ax, t in zip(axs, ["Training loss", "Validation loss", "Training accuracy (top-1)",
                           "Validation accuracy (top-1)"]):
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("época")
    for ax in axs[2:]:  # los dos paneles de accuracy, en porcentaje
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    axs[0].legend(fontsize=7)
    fig.suptitle(titulo)
    plt.tight_layout()
    plt.show()


def epoca_para(R, id_, fraccion=0.9):
    """Velocidad de convergencia: primera época que alcanza el 90 % del mejor top-1.

    Comparar el top-1 final no dice cuán rápido se llegó. Esta medida sí: si
    un experimento llega a su 90 % en la época 6 y otro en la 12, el primero
    converge el doble de rápido (en épocas), aunque ambos terminen igual.

    Parámetros
        R        : el diccionario de `resultados()`.
        id_      : experimento (debe estar terminado).
        fraccion : qué fracción del mejor top-1 cuenta como "casi convergido".

    Devuelve
        int, número de época (empieza en 1).

    Ejemplo
        epoca_para(R, "T3")    # 7: T3 llega al 90 % de su 50,4 % (≈45 %) en la época 7
    """
    h = R[id_]["hist"]
    return int(h.epoca[h.val_top1 >= fraccion * h.val_top1.max()].iloc[0])
