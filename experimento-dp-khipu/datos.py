"""Carga de datos: memmap -> lotes uint8 -> GPU, y data augmentation en GPU.

Flujo de un lote:

    memmap (N,144,144,3) uint8 en disco/caché
      │  DataLoader con workers: cada worker lee un LOTE entero de índices
      ▼
    tensor uint8 (B,144,144,3) en memoria fijada (pinned) ──► GPU
      │  AumentoGPU: recorte, flip, color, borrado, normalización
      ▼
    tensor float (B,3,128,128) listo para la red

Se transfiere uint8 (4x menos bytes que float32) y todo el aumento se hace en
GPU, vectorizado para el lote entero: la CPU sólo copia bytes.
"""
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler, WeightedRandomSampler
from torchvision.ops import roi_align

from config import CACHE, RES_ALMACEN, RES_ENTRADA

MEDIA = (0.485, 0.456, 0.406)  # estadísticas de ImageNet: las esperan los pesos
DESV = (0.229, 0.224, 0.225)   # preentrenados y no estorban al entrenar desde cero


def indice(split):
    """Tabla con ruta, etiqueta y taxonomía de cada imagen (fila i = imagen i)."""
    return pd.read_csv(CACHE / f"{split}_indice.csv")


def long_tail():
    return dict(np.load(CACHE / "longtail.npz"))


class LotesMemmap(Dataset):
    """Dataset cuyo __getitem__ recibe una LISTA de índices y devuelve el lote.

    Leer un lote de golpe con indexado avanzado de numpy es mucho más rápido
    que 256 llamadas de una imagen. Los índices se ordenan para que las lecturas
    del memmap sean lo más secuenciales posible; el orden dentro de un lote no
    afecta al gradiente.
    """

    def __init__(self, split, subconjunto=None):
        self.ruta = CACHE / f"{split}_{RES_ALMACEN}.u8"
        self.y = indice(split)["etiqueta"].to_numpy(np.int64)
        self.n_total = len(self.y)
        self.idx = np.arange(self.n_total) if subconjunto is None else np.asarray(subconjunto)
        self._mm = None  # se abre perezosamente en cada worker

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, lote):
        if self._mm is None:
            self._mm = np.memmap(self.ruta, dtype=np.uint8, mode="r",
                                 shape=(self.n_total, RES_ALMACEN, RES_ALMACEN, 3))
        i = np.sort(self.idx[np.asarray(lote)])
        return torch.from_numpy(np.ascontiguousarray(self._mm[i])), torch.from_numpy(self.y[i])

    def etiquetas(self):
        return self.y[self.idx]


def cargador(ds, batch_size, entrenamiento, semilla=0, pesos_muestreo=None, workers=12):
    """DataLoader por lotes. `pesos_muestreo` activa Weighted Sampling (long-tail)."""
    g = torch.Generator().manual_seed(semilla)  # mismo orden de lotes en cada ejecución
    if pesos_muestreo is not None:
        base = WeightedRandomSampler(torch.as_tensor(pesos_muestreo, dtype=torch.double),
                                     num_samples=len(ds), replacement=True, generator=g)
    elif entrenamiento:
        base = RandomSampler(ds, generator=g)
    else:
        base = SequentialSampler(ds)
    lotes = BatchSampler(base, batch_size, drop_last=entrenamiento)
    return DataLoader(ds, sampler=lotes, batch_size=None, num_workers=workers,
                      pin_memory=True, persistent_workers=workers > 0, prefetch_factor=4 if workers else None)


class AumentoGPU(nn.Module):
    """Data augmentation por lotes en GPU.

    nivel:
      "ninguno" : recorte central 128 (igual que en validación)
      "basico"  : recorte aleatorio 128 de 144 + flip horizontal
      "fuerte"  : Random Resized Crop (escala 0.35-1) + flip + color jitter
                  suave + Random Erasing

    Decisiones pensando en especies (ver discusión en p1_exploracion y p5_regularizacion):
      * sin flip vertical ni rotaciones grandes: las fotos tienen "arriba";
      * jitter de color SUAVE y sin cambio de tono (hue): el color distingue
        especies (p.ej. aves o setas casi idénticas salvo por el color);
      * la escala mínima de 0.35 no deja el organismo fuera del recorte casi nunca.
    """

    def __init__(self, nivel="basico", res=RES_ENTRADA):
        super().__init__()
        self.nivel, self.res = nivel, res
        self.register_buffer("media", torch.tensor(MEDIA).view(1, 3, 1, 1))
        self.register_buffer("desv", torch.tensor(DESV).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, x_u8, entrenamiento):
        x = x_u8.permute(0, 3, 1, 2).float().div_(255)  # (B,3,H,W) en [0,1]
        B, _, H, W = x.shape
        nivel = self.nivel if entrenamiento else "ninguno"

        if nivel == "fuerte":
            x = self._random_resized_crop(x)
        else:
            if nivel == "basico":  # un desplazamiento distinto por imagen, vía roi_align
                off = torch.randint(0, H - self.res + 1, (B, 2), device=x.device).float()
            else:
                off = torch.full((B, 2), (H - self.res) / 2, device=x.device)
            cajas = torch.cat([off, off + self.res], 1)
            x = self._recortar(x, cajas)

        if nivel in ("basico", "fuerte"):
            voltear = torch.rand(B, 1, 1, 1, device=x.device) < 0.5
            x = torch.where(voltear, x.flip(3), x)
        if nivel == "fuerte":
            x = self._color(x, 0.2)
            x = self._borrado(x, p=0.25)
        x = (x - self.media) / self.desv
        return x.contiguous(memory_format=torch.channels_last)

    def _recortar(self, x, cajas_xy):
        """Recorta cada imagen con su caja (x1,y1,x2,y2) y reescala a res x res."""
        B = x.shape[0]
        rois = torch.cat([torch.arange(B, device=x.device, dtype=x.dtype)[:, None], cajas_xy], 1)
        return roi_align(x, rois, output_size=self.res, spatial_scale=1.0, sampling_ratio=1, aligned=True)

    def _random_resized_crop(self, x, escala=(0.35, 1.0), razon=(3 / 4, 4 / 3)):
        B, _, H, W = x.shape
        d = x.device
        area = H * W * torch.empty(B, device=d).uniform_(*escala)
        r = torch.exp(torch.empty(B, device=d).uniform_(math.log(razon[0]), math.log(razon[1])))
        w = torch.sqrt(area * r).clamp(8, W)
        h = torch.sqrt(area / r).clamp(8, H)
        x0 = torch.rand(B, device=d) * (W - w)
        y0 = torch.rand(B, device=d) * (H - h)
        return self._recortar(x, torch.stack([x0, y0, x0 + w, y0 + h], 1))

    @staticmethod
    def _color(x, f):
        """Brillo, contraste y saturación aleatorios por imagen (factor 1±f)."""
        B = x.shape[0]
        a = lambda: torch.empty(B, 1, 1, 1, device=x.device).uniform_(1 - f, 1 + f)
        x = x * a()
        media = x.mean(dim=(1, 2, 3), keepdim=True)
        x = (x - media) * a() + media
        gris = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
        x = (x - gris) * a() + gris
        return x.clamp_(0, 1)

    @staticmethod
    def _borrado(x, p, escala=(0.02, 0.2)):
        """Random Erasing: tapa un rectángulo con ruido en una fracción p del lote."""
        B, _, H, W = x.shape
        d = x.device
        area = H * W * torch.empty(B, device=d).uniform_(*escala)
        r = torch.exp(torch.empty(B, device=d).uniform_(math.log(0.3), math.log(3.3)))
        h = torch.sqrt(area * r).clamp(1, H)
        w = torch.sqrt(area / r).clamp(1, W)
        y0 = torch.rand(B, device=d) * (H - h)
        x0 = torch.rand(B, device=d) * (W - w)
        yy = torch.arange(H, device=d).view(1, H, 1)
        xx = torch.arange(W, device=d).view(1, 1, W)
        m = ((yy >= y0.view(B, 1, 1)) & (yy < (y0 + h).view(B, 1, 1)) &
             (xx >= x0.view(B, 1, 1)) & (xx < (x0 + w).view(B, 1, 1)))
        m = m & (torch.rand(B, 1, 1, device=d) < p)
        return torch.where(m[:, None], torch.rand_like(x), x)
