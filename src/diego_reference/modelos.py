"""Modelos: MLP baseline y CNN de torchvision con la cabeza adaptada a 10.000 clases."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

from .config import NUM_CLASES


class MLP(nn.Module):
    """Perceptrón multicapa sobre la imagen aplanada.

    Recibe el mismo tensor (B,3,128,128) que las CNN y lo reduce a `res` x `res`
    antes de aplanar: la resolución de entrada es un hiperparámetro del MODELO,
    así todos los experimentos comparten el mismo pipeline de datos.

    Parámetros de la primera capa = 3·res²·h1. Con res=32 y h1=2048 son 6.3 M;
    con res=128 serían 100 M sólo en esa capa. Por eso el MLP obliga a bajar
    la resolución: su número de parámetros crece con el área de la imagen,
    mientras que el de una convolución no depende del tamaño de la entrada.
    """

    def __init__(self, res=32, ocultas=(2048, 1024), dropout=0.0, bn=True, num_clases=NUM_CLASES):
        super().__init__()
        self.res = res
        capas, d = [], 3 * res * res
        for h in ocultas:
            capas += [nn.Linear(d, h)]
            if bn:
                capas += [nn.BatchNorm1d(h)]
            capas += [nn.ReLU(inplace=True), nn.Dropout(dropout)]
            d = h
        self.cuerpo = nn.Sequential(*capas)
        self.fc = nn.Linear(d, num_clases)
        for m in self.modules():  # He/Kaiming: preserva la varianza con ReLU
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.interpolate(x, size=(self.res, self.res), mode="bilinear", antialias=True, align_corners=False)
        return self.fc(self.cuerpo(x.flatten(1)))


PESOS = {  # pesos ImageNet de torchvision
    "resnet18": tvm.ResNet18_Weights.IMAGENET1K_V1,
    "resnet50": tvm.ResNet50_Weights.IMAGENET1K_V2,
    "efficientnet_b0": tvm.EfficientNet_B0_Weights.IMAGENET1K_V1,
    "convnext_tiny": tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
    "mobilenet_v3_large": tvm.MobileNet_V3_Large_Weights.IMAGENET2K_V1 if hasattr(tvm.MobileNet_V3_Large_Weights, "IMAGENET2K_V1") else tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V1,
}


def crear_modelo(nombre, preentrenado=False, bn=True, dropout=0.0, num_clases=NUM_CLASES, **kw):
    if nombre == "mlp":
        return MLP(dropout=dropout, num_clases=num_clases, **kw)

    extra = dict(kw)
    if not bn:  # ablación: ResNet sin BatchNorm (cada BN se sustituye por identidad)
        assert nombre.startswith("resnet") and not preentrenado
        extra["norm_layer"] = lambda c: nn.Identity()
    m = getattr(tvm, nombre)(weights=PESOS[nombre] if preentrenado else None, **extra)

    # Sustituir la última capa por una de 10.000 salidas (con dropout opcional)
    if nombre.startswith("resnet"):
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, num_clases))
    else:  # efficientnet / convnext / mobilenet: classifier[-1] es la Linear final
        ult = m.classifier[-1]
        m.classifier[-1] = nn.Linear(ult.in_features, num_clases)
        if dropout > 0:
            m.classifier.insert(len(m.classifier) - 1, nn.Dropout(dropout))
    return m


def cabeza(m):
    """Módulo con la capa de clasificación (lo único que entrena Feature Extraction)."""
    return m.fc if hasattr(m, "fc") else m.classifier


def grupos_ajuste(m, nombre, modo):
    """Qué parámetros se entrenan en cada escenario de Transfer Learning.

    Devuelve (modulos_entrenables_backbone, modulos_congelados).
      "congelado" : sólo la cabeza                        (Experimento A)
      "parcial"   : último bloque del backbone + cabeza   (Experimento B)
      "completo"  : todo                                  (Experimento C)
    En ResNet-50 el "último bloque" es layer4: 15 M de 23.5 M parámetros del
    backbone y los rasgos más específicos de ImageNet (partes de objetos), que
    son los que más hay que readaptar a especies; layer1-3 (bordes, texturas)
    se transfieren bien sin tocarlos.
    """
    if nombre.startswith("resnet"):
        backbone = [m.conv1, m.bn1, m.layer1, m.layer2, m.layer3, m.layer4]
        ultimos = [m.layer4]
    else:
        bloques = list(m.features)
        backbone, ultimos = bloques, bloques[-2:]
    if modo == "congelado":
        return [], backbone
    if modo == "parcial":
        return ultimos, [b for b in backbone if all(b is not u for u in ultimos)]
    return backbone, []


def contar_parametros(m, solo_entrenables=False):
    return sum(p.numel() for p in m.parameters() if p.requires_grad or not solo_entrenables)


@torch.no_grad()
def gflops(m, res, dispositivo):
    """GFLOPs de una pasada hacia adelante de UNA imagen (multiplicación-suma = 2 FLOPs)."""
    from torch.utils.flop_counter import FlopCounterMode
    estado = m.training
    m.eval()
    x = torch.zeros(1, 3, res, res, device=dispositivo)
    with FlopCounterMode(display=False) as fc:
        m(x)
    m.train(estado)
    return fc.get_total_flops() / 1e9
