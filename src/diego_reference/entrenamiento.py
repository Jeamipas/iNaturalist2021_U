"""Bucle de entrenamiento y evaluación común a TODOS los experimentos.

Un experimento es sólo un diccionario de hiperparámetros (ver experimentos.py);
todo lo demás (datos, métricas, early stopping, registro) es idéntico entre
experimentos. Así las comparaciones son controladas: si dos experimentos
difieren, es porque difiere su configuración, no el código.

Salidas en resultados/<ID>/:
  config.json     configuración exacta usada
  historial.json  métricas por época (train/val loss, accuracy, F1, tiempos)
  resumen.json    métricas finales del mejor modelo + parámetros, FLOPs, tiempo
  preds.npz       top-5 predicciones y probabilidades sobre validación
  mejor.pt        pesos del mejor modelo (según early stopping)
"""
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NUM_CLASES, RES_ENTRADA, RESULTADOS
from ..diego_workflow import validate_identity, atomic_json, atomic_checkpoint, state_hash, needs_resume, selected_epoch
from .datos import AumentoGPU, LotesMemmap, cargador, long_tail
from .modelos import cabeza, contar_parametros, crear_modelo, gflops, grupos_ajuste


@dataclass
class Config:
    id: str
    descripcion: str = ""
    # modelo
    modelo: str = "resnet18"
    preentrenado: bool = False
    ajuste: str = "completo"          # congelado | parcial | completo
    bn: bool = True
    dropout: float = 0.0
    modelo_kw: dict = field(default_factory=dict)
    # datos
    datos: str = "completo"           # completo | longtail | lt_balanceado
    aumento: str = "basico"           # ninguno | basico | fuerte
    muestreo: str = "uniforme"        # uniforme | ponderado  (Weighted Sampling)
    # optimización
    optimizador: str = "sgdm"         # sgd | sgdm | adam | adamw
    lr: float = 0.1
    factor_lr_backbone: float = 1.0   # lr del backbone = lr * factor (fine-tuning)
    weight_decay: float = 5e-5
    batch: int = 256
    epocas: int = 16
    warmup: float = 1.0               # épocas de calentamiento lineal del lr
    clip: float = 5.0                 # norma máxima del gradiente
    # pérdida
    perdida: str = "ce"               # ce | ce_ponderada | logit_ajustado
    label_smoothing: float = 0.0
    # early stopping (sobre Top-1 de validación)
    paciencia: int = 4
    min_delta: float = 0.001          # mejora mínima de 0.1 puntos porcentuales
    semilla: int = 42
    muon_lr: float = 0.02


# ---------------------------------------------------------------- utilidades
def fijar_semillas(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    # cudnn.benchmark elige el algoritmo de convolución más rápido; introduce
    # un no-determinismo numérico mínimo (~1e-3 pp en accuracy) a cambio de
    # ~20% de velocidad. Para bit-exactitud: TORCH_DETERMINISTA=1.
    det = os.environ.get("TORCH_DETERMINISTA") == "1"
    torch.backends.cudnn.benchmark = not det
    torch.use_deterministic_algorithms(det, warn_only=True)


def f1_desde_predicciones(pred, y, C=NUM_CLASES):
    """Macro y weighted F1 en GPU con bincount (sin matriz de confusión C x C)."""
    tp = torch.bincount(y[pred == y], minlength=C).double()
    n_pred = torch.bincount(pred, minlength=C).double()
    n_real = torch.bincount(y, minlength=C).double()
    prec = tp / n_pred.clamp(min=1)
    rec = tp / n_real.clamp(min=1)
    f1 = 2 * prec * rec / (prec + rec).clamp(min=1e-12)
    presentes = n_real > 0
    macro = f1[presentes].mean().item()
    weighted = (f1 * n_real).sum().item() / n_real.sum().item()
    return macro, weighted, dict(f1=f1, prec=prec, rec=rec)


@torch.no_grad()
def evaluar(modelo, dl, aum, disp, guardar=False):
    modelo.eval()
    perdida, top1, top5, n = 0.0, 0, 0, 0
    preds, ys, top5_idx, top5_p = [], [], [], []
    for xb, yb in dl:
        xb, yb = xb.to(disp, non_blocking=True), yb.to(disp, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = modelo(aum(xb, entrenamiento=False))
        logits = logits.float()
        perdida += F.cross_entropy(logits, yb, reduction="sum").item()
        p5, i5 = logits.softmax(1).topk(5, 1)
        top1 += (i5[:, 0] == yb).sum().item()
        top5 += (i5 == yb[:, None]).any(1).sum().item()
        n += len(yb)
        preds.append(i5[:, 0]); ys.append(yb)
        if guardar:
            top5_idx.append(i5.short().cpu()); top5_p.append(p5.half().cpu())
    pred, y = torch.cat(preds), torch.cat(ys)
    f1m, f1w, _ = f1_desde_predicciones(pred, y)
    r = dict(loss=perdida / n, top1=top1 / n, top5=top5 / n, f1_macro=f1m, f1_weighted=f1w)
    if guardar:
        r["_preds"] = dict(y=y.cpu().numpy(), top5=torch.cat(top5_idx).numpy(), prob5=torch.cat(top5_p).numpy())
    return r


def programa_lr(total_pasos, pasos_warmup):
    """Calentamiento lineal y luego coseno hasta 0 (factor multiplicativo del lr)."""
    def f(paso):
        if paso < pasos_warmup:
            return (paso + 1) / pasos_warmup
        t = (paso - pasos_warmup) / max(1, total_pasos - pasos_warmup)
        return 0.5 * (1 + math.cos(math.pi * t))
    return f


def crear_optimizador(cfg, modelo, entrenables_backbone):
    ids_bb = {id(p) for m in entrenables_backbone for p in m.parameters()}
    grupos = {}
    for nombre, p in modelo.named_parameters():
        if not p.requires_grad:
            continue
        es_bb = id(p) in ids_bb
        sin_wd = p.ndim <= 1 or nombre.endswith("layer_scale")  # sesgos y parámetros de BN: sin weight decay
        clave = (es_bb, sin_wd)
        grupos.setdefault(clave, []).append(p)
    params = [dict(params=ps,
                   lr=cfg.lr * (cfg.factor_lr_backbone if es_bb else 1.0),
                   weight_decay=0.0 if sin_wd else cfg.weight_decay)
              for (es_bb, sin_wd), ps in grupos.items()]
    if cfg.optimizador == "muon":
        from ..diego_muon import DiegoMuon
        return DiegoMuon(modelo, cfg.lr, cfg.muon_lr, cfg.weight_decay, cfg.factor_lr_backbone)
    if cfg.optimizador == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, momentum=0.0)
    if cfg.optimizador == "sgdm":
        return torch.optim.SGD(params, lr=cfg.lr, momentum=0.9, nesterov=True)
    if cfg.optimizador == "adam":  # weight decay acoplado (L2 clásico)
        return torch.optim.Adam(params, lr=cfg.lr)
    if cfg.optimizador == "adamw":  # weight decay desacoplado
        return torch.optim.AdamW(params, lr=cfg.lr)
    raise ValueError(cfg.optimizador)


def crear_perdida(cfg, n_por_clase, disp):
    n = torch.as_tensor(n_por_clase, dtype=torch.float, device=disp).clamp(min=1)
    if cfg.perdida == "ce":
        return lambda z, y: F.cross_entropy(z, y, label_smoothing=cfg.label_smoothing)
    if cfg.perdida == "ce_ponderada":
        # peso ∝ 1/n_c. PyTorch divide por la suma de pesos del lote (media
        # ponderada), así que la escala de la pérdida no cambia: sólo cuánto
        # cuenta cada clase. Una imagen de clase con 1 ejemplo pesa 50x más.
        w = 1.0 / n
        w = w * n.sum() / (w * n).sum()
        return lambda z, y: F.cross_entropy(z, y, weight=w, label_smoothing=cfg.label_smoothing)
    if cfg.perdida == "logit_ajustado":
        # Menon et al. 2021: sumar log(prior) a los logits SÓLO al entrenar. El
        # modelo aprende p(y|x)/p(y) y en test (sin el término) predice balanceado.
        log_prior = torch.log(n / n.sum())
        return lambda z, y: F.cross_entropy(z + log_prior, y, label_smoothing=cfg.label_smoothing)
    raise ValueError(cfg.perdida)


# --------------------------------------------------------------- principal
def entrenar(cfg: Config, subconjunto=None, val_sub=None, workers=12):
    disp = torch.device("cuda")
    fijar_semillas(cfg.semilla)
    salida = RESULTADOS / cfg.id
    salida.mkdir(parents=True, exist_ok=True)
    identity, completed = validate_identity(cfg, salida)
    if completed is not None:
        return completed
    atomic_json(salida / "config.json", asdict(cfg))

    # ---- datos
    idx_train = None
    if cfg.datos in ("longtail", "lt_balanceado"):
        lt = long_tail()
        idx_train = lt["idx_lt"] if cfg.datos == "longtail" else lt["idx_bal"]
    if subconjunto:  # sólo para depurar
        base = np.arange(LotesMemmap("train_mini").n_total) if idx_train is None else idx_train
        idx_train = np.sort(np.random.default_rng(0).choice(base, subconjunto, replace=False))
    ds_tr = LotesMemmap("train_mini", idx_train)
    y_tr = ds_tr.etiquetas()
    n_por_clase = np.bincount(y_tr, minlength=NUM_CLASES)
    pesos = (1.0 / n_por_clase[y_tr]) if cfg.muestreo == "ponderado" else None
    dl_tr = cargador(ds_tr, cfg.batch, True, cfg.semilla, pesos, workers)
    idx_val = None if not val_sub else np.sort(np.random.default_rng(0).choice(100_000, val_sub, replace=False))
    dl_val = cargador(LotesMemmap("val", idx_val), 1024, False, workers=workers // 2)
    aum = AumentoGPU(cfg.aumento).to(disp)

    # ---- modelo
    modelo = crear_modelo(cfg.modelo, cfg.preentrenado, cfg.bn, cfg.dropout, **cfg.modelo_kw)
    modelo = modelo.to(disp, memory_format=torch.channels_last)
    if cfg.modelo == "mlp":
        entrenables_bb, congelados = [], []
    else:
        entrenables_bb, congelados = grupos_ajuste(modelo, cfg.modelo, cfg.ajuste)
    for m in congelados:
        m.requires_grad_(False)

    def modo_entrenamiento():
        modelo.train()
        for m in congelados:  # BN congeladas conservan las estadísticas de ImageNet
            m.eval()

    initial_hash = state_hash(modelo)
    atomic_json(salida / "initialization.json", {"initial_weights_sha256": initial_hash, "identity": identity})
    opt = crear_optimizador(cfg, modelo, entrenables_bb)
    pasos_ep = len(dl_tr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, programa_lr(cfg.epocas * pasos_ep, int(cfg.warmup * pasos_ep)))
    criterio = crear_perdida(cfg, n_por_clase, disp)

    info = dict(identity=identity, initial_weights_sha256=initial_hash, parametros=contar_parametros(modelo),
                parametros_entrenables=contar_parametros(modelo, True),
                gflops=gflops(modelo, RES_ENTRADA, disp),
                imagenes_train=len(ds_tr), pasos_por_epoca=pasos_ep,
                gpu=torch.cuda.get_device_name())
    print(json.dumps(info), flush=True)

    # ---- reanudar si el job anterior murió (límite de 8 h en Khipu)
    hist, ep0, mejor, sin_mejora, mejor_estado = [], 0, -1.0, 0, None
    ckpt = salida / "ultimo.pt"
    if ckpt.exists():
        s = torch.load(ckpt, map_location=disp, weights_only=False)
        modelo.load_state_dict(s["modelo"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
        hist, ep0, mejor, sin_mejora = s["hist"], s["epoca"] + 1, s["mejor"], s["sin_mejora"]
        mejor_estado = torch.load(salida / "mejor.pt", map_location="cpu")
        random.setstate(s["rng_python"]); np.random.set_state(s["rng_numpy"])
        torch.set_rng_state(s["rng_cpu"].cpu()); torch.cuda.set_rng_state(s["rng_cuda"].cpu())  # map_location los movió a GPU
        print(f"reanudando en la época {ep0}", flush=True)

    t_total = sum(h["t_train"] for h in hist)
    for ep in range(ep0, cfg.epocas):
        if sin_mejora >= cfg.paciencia:
            break
        atomic_json(salida / "status.json", {"state": "TRAINING", "epoch": ep + 1})
        modo_entrenamiento()
        dl_tr.sampler.sampler.generator.manual_seed(cfg.semilla + ep)  # orden reproducible por época
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        suma_l, aciertos, n, norma_max = torch.zeros((), device=disp), torch.zeros((), device=disp), 0, 0.0
        normas = []
        last_status = time.monotonic()
        for xb, yb in dl_tr:
            xb, yb = xb.to(disp, non_blocking=True), yb.to(disp, non_blocking=True)
            x = aum(xb, entrenamiento=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = modelo(x)
            loss = criterio(logits.float(), yb)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            normas.append(nn.utils.clip_grad_norm_(modelo.parameters(), cfg.clip))
            opt.step()
            sched.step()
            suma_l += loss.detach() * len(yb)
            aciertos += (logits.argmax(1) == yb).sum()
            n += len(yb)
            if time.monotonic() - last_status > 60:
                atomic_json(salida / "status.json", {"state": "TRAINING", "epoch": ep + 1, "samples": n, "total": len(ds_tr)})
                print(f"epoch={ep + 1} samples={n}/{len(ds_tr)}", flush=True)
                last_status = time.monotonic()
        if n != len(ds_tr):
            raise RuntimeError("Incomplete training corpus")
        torch.cuda.synchronize()
        t_train = time.time() - t0
        t_total += t_train
        normas = torch.stack(normas).float()

        t1 = time.time()
        atomic_json(salida / "status.json", {"state": "VALIDATING", "epoch": ep + 1})
        val = evaluar(modelo, dl_val, aum, disp)
        h = dict(epoca=ep + 1, train_loss=(suma_l / n).item(), train_top1=(aciertos / n).item(),
                 val_loss=val["loss"], val_top1=val["top1"], val_top5=val["top5"],
                 val_f1_macro=val["f1_macro"], val_f1_weighted=val["f1_weighted"],
                 lr=sched.get_last_lr()[0], t_train=t_train, t_val=time.time() - t1,
                 img_por_s=n / t_train, mem_gb=torch.cuda.max_memory_allocated() / 1e9,
                 grad_norma_media=normas.mean().item(), grad_norma_max=normas.max().item(),
                 perdida_no_finita=not math.isfinite((suma_l / n).item()))
        hist.append(h)
        print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in h.items()), flush=True)

        # ---- early stopping: Top-1 de validación, paciencia y mejora mínima
        if val["top1"] > mejor + cfg.min_delta:
            mejor, sin_mejora = val["top1"], 0
            mejor_estado = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}
            atomic_checkpoint(salida / "mejor.pt", mejor_estado)
        else:
            sin_mejora += 1
        h["train_samples"] = n
        atomic_json(salida / "historial.json", hist)
        atomic_checkpoint(ckpt, dict(modelo=modelo.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        hist=hist, epoca=ep, mejor=mejor, sin_mejora=sin_mejora,
                        rng_python=random.getstate(), rng_numpy=np.random.get_state(),
                        rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state()))
        if h["perdida_no_finita"]:
            print("pérdida no finita: se detiene (divergencia)", flush=True)
            break
        if sin_mejora >= cfg.paciencia:
            print(f"early stopping en la época {ep + 1} (mejor top-1 = {mejor:.4f})", flush=True)
            break

        if ep + 1 < cfg.epocas and needs_resume(hist):
            status = {"state": "NEEDS_RESUME", "epoch": ep + 1, "next_epoch": ep + 2}
            atomic_json(salida / "status.json", status)
            return status

    # ---- evaluación final con el MEJOR modelo (no el último)
    if mejor_estado is not None:
        modelo.load_state_dict(mejor_estado)
    final = evaluar(modelo, dl_val, aum, disp, guardar=True)
    preds = final.pop("_preds")
    np.savez_compressed(salida / "preds.npz", idx_val=np.arange(100_000) if idx_val is None else idx_val, **preds)
    mejor_ep = selected_epoch(hist, cfg.min_delta)
    resumen = dict(id=cfg.id, descripcion=cfg.descripcion, **info, **{f"val_{k}": v for k, v in final.items()},
                   mejor_epoca=mejor_ep, epocas_corridas=len(hist), tiempo_train_s=t_total,
                   tiempo_por_epoca_s=t_total / max(1, len(hist)),
                   brecha_loss=hist[mejor_ep - 1]["val_loss"] - hist[mejor_ep - 1]["train_loss"])
    resumen["imagenes_val"] = len(preds["y"])
    resumen["tiempo_val_s"] = sum(h["t_val"] for h in hist)
    resumen["torch_version"] = torch.__version__
    resumen["torchvision_version"] = __import__("torchvision").__version__
    resumen["precision"] = "bf16"
    resumen["deterministic_requested"] = os.environ.get("TORCH_DETERMINISTA") == "1"
    atomic_json(salida / "resumen.json", resumen)
    atomic_json(salida / "status.json", {"state": "COMPLETED", "metrics": {k:v for k,v in resumen.items() if k.startswith("val_")}})
    # Keep the final resume checkpoint for audit and reproducibility.
    print(json.dumps(resumen, indent=2), flush=True)
    return resumen
