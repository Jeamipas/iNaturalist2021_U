"""Audit, preparation and combined reporting for the minimal Diego extension."""
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from .full_data import atomic_json, audit_corpus, digest


def evidence(root):
    return json.loads((Path(root) / "references/diego_evidence.json").read_text(encoding="utf-8"))


def plan_from_evidence(root):
    original = next(e["config"] for e in evidence(root)["experiments"] if e["id"] == "E09")
    common = {**original, "modelo": "convnext_tiny", "muon_lr": .02,
              "modelo_kw": {"stochastic_depth_prob": .1}}
    return [{**common, "id": "N01", "descripcion": "ConvNeXt-Tiny desde cero con Adam"},
            {**common, "id": "N02", "descripcion": "ConvNeXt-Tiny desde cero con Muon + Adam auxiliar", "optimizador": "muon"},
            {**common, "id": "N03", "descripcion": "ConvNeXt-Tiny ImageNet: fine-tuning completo con Adam", "preentrenado": True, "factor_lr_backbone": .1},
            {**common, "id": "N04", "descripcion": "ConvNeXt-Tiny ImageNet: fine-tuning completo con Muon + Adam auxiliar", "preentrenado": True, "factor_lr_backbone": .1, "optimizador": "muon"}]


def selected_epoch(history, min_delta):
    best, selected = -1., None
    for row in history:
        if row["val_top1"] > best + min_delta:
            best, selected = row["val_top1"], row["epoca"]
    return selected


def state_hash(model):
    value = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value.update(name.encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def atomic_checkpoint(path, value):
    import torch
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def validate_identity(cfg, output):
    root = Path(os.environ["NATURALIST_ROOT"])
    data = Path(os.environ["INAT_ROOT"])
    cache = Path(os.environ["INAT_CACHE"])
    identity = {"configuration": asdict(cfg), "torch_determinista": os.environ.get("TORCH_DETERMINISTA", "0"),
                "data": {name: digest(data / name) for name in ("train_mini.json", "val.json")},
                "cache_manifest": digest(cache / "compatible_manifest.json"),
                "sources": {name: digest(root / name) for name in (
                    "src/diego_reference/entrenamiento.py", "src/diego_reference/modelos.py",
                    "src/diego_reference/datos.py", "src/diego_muon.py", "src/full_experiment.py",
                    "src/diego_workflow.py", "requirements-full.txt")}}
    if cfg.preentrenado:
        identity["pretrained_weights_sha256"] = digest(Path(os.environ["TORCH_HOME"]) / "hub/checkpoints/convnext_tiny-983f1562.pth")
    identity_path = Path(output) / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Existing run has different sources, configuration or data; use a new RUN_ID")
    atomic_json(identity_path, identity)
    result_path = Path(output) / "resumen.json"
    return identity, json.loads(result_path.read_text()) if result_path.exists() else None


def needs_resume(history):
    elapsed = time.time() - float(os.environ.get("NATURALIST_ALLOCATION_STARTED", time.time()))
    predicted_epoch = max(h["t_train"] + h["t_val"] for h in history[-3:]) * 1.25
    return elapsed + predicted_epoch >= float(os.environ.get("NATURALIST_SEGMENT_SECONDS", 25200))


def prepare_compatible_cache(root, data_root, workers=32):
    import numpy as np
    from .diego_reference import preparar_datos as prep
    from .diego_reference.config import CACHE, RES_ALMACEN
    root, data_root = Path(root), Path(data_root)
    corpus = audit_corpus(data_root)
    signatures = {"train": corpus["train_json_sha256"], "val": corpus["val_json_sha256"],
                  "preparation_source": digest(root / "src/diego_reference/preparar_datos.py"),
                  "pillow": __import__("PIL").__version__, "storage_size": RES_ALMACEN}
    CACHE.mkdir(parents=True, exist_ok=True)
    manifest_path = CACHE / "compatible_manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if previous and previous["signatures"] != signatures:
        raise ValueError("Diego cache identity changed; choose a separate cache directory")
    frames = {}
    for split, expected in (("val", 100000), ("train_mini", 500000)):
        index_path = CACHE / f"{split}_indice.csv"
        # Regenerate deterministically from all official annotations, never a sample.
        frame = prep.construir_indice(split)
        assert len(frame) == expected
        assert frame["etiqueta"].nunique() == 10000
        assert np.array_equal(np.unique(frame["etiqueta"]), np.arange(10000))
        frame.to_csv(index_path, index=False)
        mm = CACHE / f"{split}_{RES_ALMACEN}.u8"
        if (CACHE / f"{split}_{RES_ALMACEN}.ok").exists():
            if mm.stat().st_size != expected * RES_ALMACEN * RES_ALMACEN * 3:
                raise ValueError("Completed memmap has the wrong size")
        prep.redimensionar(split, frame, workers)
        frames[split] = frame
    lt = prep.construir_long_tail(frames["train_mini"])
    assert len(lt["idx_lt"]) == 120474 and len(lt["idx_bal"]) == 120000
    np.savez(CACHE / "longtail.npz", **lt)
    targets_hash = hashlib.sha256(frames["val"]["etiqueta"].to_numpy(dtype='<i8').tobytes()).hexdigest()
    historical = evidence(root)
    for experiment in historical["experiments"]:
        assert experiment["prediction_audit"]["y_sha256"] == targets_hash
        path = root / "references/khipu_diego" / historical["commit"][:12] / "resultados" / experiment["id"] / "preds.npz"
        if digest(path) != experiment["prediction_audit"]["preds_sha256"]:
            raise ValueError(f"Historical predictions checksum mismatch: {experiment['id']}")
    manifest = {"signatures": signatures, "train_images": 500000, "val_images": 100000,
                "species": 10000, "val_targets_sha256": targets_hash,
                "index_sha256": {s: digest(CACHE / f"{s}_indice.csv") for s in frames},
                "long_tail_images": len(lt["idx_lt"]), "balanced_lt_control_images": len(lt["idx_bal"]),
                "all_26_historical_prediction_checksums_verified": True}
    atomic_json(manifest_path, manifest)
    return manifest


def run_extension(config):
    from .diego_reference.entrenamiento import Config, entrenar
    from .diego_reference.config import RESULTADOS
    try:
        return entrenar(Config(**config), workers=28)
    except Exception as error:
        atomic_json(RESULTADOS / config["id"] / "status.json", {"state": "FAILED", "error": repr(error)})
        raise


def report(root, output_root, plan, with_images=False):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display
    root, output_root = Path(root), Path(output_root)
    historical = evidence(root)
    rows, histories = [], {}
    results = []
    for item in historical["experiments"]:
        s, c = item["summary"], item["config"]
        rows.append({"ID": "D-" + item["id"], "Origen": "khipu-diego", "Estado": "PUBLICADO Y VERIFICADO",
                     "Modelo": c["modelo"], "Optimizador": c["optimizador"], "Preentrenado": c["preentrenado"],
                     "Ajuste": c["ajuste"], "Datos": c["datos"], "Train": s["imagenes_train"], "Val": 100000,
                     "Top-1": s["val_top1"], "Top-5": s["val_top5"], "Macro-F1": s["val_f1_macro"],
                     "Weighted-F1": s["val_f1_weighted"], "Parámetros": s["parametros"],
                     "GPU": s["gpu"], "Train segundos": s["tiempo_train_s"],
                     "Val segundos": sum(h["t_val"] for h in item["history"]),
                     "Checkpoint época": item["selected_checkpoint_epoch"]})
        histories["D-" + item["id"]] = item["history"]
    for c in plan:
        directory = output_root / c["id"]
        result_path = directory / "resumen.json"
        if result_path.exists():
            s = json.loads(result_path.read_text())
            rows.append({"ID": c["id"], "Origen": "Extensión", "Estado": "COMPLETED", "Modelo": c["modelo"],
                         "Optimizador": c["optimizador"], "Preentrenado": c["preentrenado"], "Ajuste": c["ajuste"],
                         "Datos": c["datos"], "Train": s["imagenes_train"], "Val": s["imagenes_val"],
                         "Top-1": s["val_top1"], "Top-5": s["val_top5"], "Macro-F1": s["val_f1_macro"],
                         "Weighted-F1": s["val_f1_weighted"], "Parámetros": s["parametros"], "GPU": s["gpu"],
                         "Train segundos": s["tiempo_train_s"], "Val segundos": s["tiempo_val_s"],
                         "Checkpoint época": s["mejor_epoca"]})
            histories[c["id"]] = json.loads((directory / "historial.json").read_text())
            results.append(s)
        else:
            status = json.loads((directory / "status.json").read_text()) if (directory / "status.json").exists() else {"state": "PENDING"}
            rows.append({"ID": c["id"], "Origen": "Extensión", "Estado": status["state"], "Modelo": c["modelo"], "Optimizador": c["optimizador"]})
    by_id = {r["id"]: r for r in results}
    for left, right in (("N01", "N02"), ("N03", "N04")):
        if left in by_id and right in by_id:
            if by_id[left]["initial_weights_sha256"] != by_id[right]["initial_weights_sha256"]:
                raise RuntimeError(f"Invalid Adam–Muon pair {left}/{right}: initial weights differ")
    table = pd.DataFrame(rows)
    display(table)
    output_root.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_root / "comparison.csv", index=False)
    atomic_json(output_root / "comparison.json", rows)
    groups = {"convnext_scratch": ["N01", "N02"], "convnext_transfer": ["N03", "N04"],
              "convnext_regimenes": ["N01", "N02", "N03", "N04"],
              "arquitecturas": ["D-E03", "D-E04", "D-E05"],
              "optimizadores": ["D-E03", "D-E06", "D-E09", "D-E12"],
              "regularizacion": ["D-A1", "D-A2", "D-A3", "D-A4", "D-A5", "D-A6"],
              "transfer": ["D-E04", "D-T1", "D-T2", "D-T3", "D-T4"], "longtail": ["D-T3", "D-L2", "D-L3", "D-L4", "D-L5"]}
    for group, identifiers in groups.items():
        fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        for name in identifiers:
            if name not in histories:
                continue
            hist = histories[name]
            for ax, key in zip(axes.flat, ("train_loss", "val_loss", "train_top1", "val_top1")):
                ax.plot([h["epoca"] for h in hist], [h[key] for h in hist], label=name)
                ax.set(xlabel="Época", ylabel=key)
        for ax in axes.flat:
            ax.legend(); ax.grid(alpha=.2)
        fig.suptitle(group + " — recetas descriptivas; tiempos históricos dependen del hardware")
        fig.tight_layout()
        fig.savefig(output_root / f"curves_{group}.png", dpi=150)
        plt.show(); plt.close(fig)
    if with_images:
        report_errors(root, output_root, historical, results)
    return table


def report_errors(root, output_root, historical, results):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from PIL import Image
    from IPython.display import display
    from .diego_reference.config import CACHE, DATA_ROOT
    frame = pd.read_csv(CACHE / "val_indice.csv")
    candidates = [(e["summary"]["val_f1_macro"], "D-" + e["id"],
                   root / "references/khipu_diego" / historical["commit"][:12] / "resultados" / e["id"] / "preds.npz")
                  for e in historical["experiments"]]
    candidates += [(r["val_f1_macro"], r["id"], output_root / r["id"] / "preds.npz") for r in results]
    _, best_name, path = max(candidates)
    with np.load(path) as a:
        y, p, confidence = a["y"].astype(int), a["top5"][:, 0].astype(int), a["prob5"][:, 0].astype(float)
    assert np.array_equal(y, frame["etiqueta"].to_numpy())
    support, pc = np.bincount(y, minlength=10000), np.bincount(p, minlength=10000)
    tp = np.bincount(y[y == p], minlength=10000)
    f1 = np.divide(2*tp, support+pc, out=np.zeros(10000), where=support+pc > 0)
    species = frame.drop_duplicates("etiqueta").sort_values("etiqueta")[['etiqueta','especie','kingdom','genus']].copy()
    species["F1"] = f1; species["correct"] = tp; species["images"] = support
    species.to_csv(output_root / "best_species.csv", index=False)
    taxa = species.groupby("kingdom").agg(macro_f1=("F1","mean"), correct=("correct","sum"), images=("images","sum"))
    taxa["top1"] = taxa["correct"] / taxa["images"]
    display(taxa)
    confusions = pd.DataFrame({"real": y[y != p], "predicted": p[y != p]}).value_counts().head(20).reset_index(name="count")
    names = species.set_index("etiqueta")
    confusions["real_name"] = confusions["real"].map(names["especie"])
    confusions["predicted_name"] = confusions["predicted"].map(names["especie"])
    confusions["same_genus"] = confusions["real"].map(names["genus"]) == confusions["predicted"].map(names["genus"])
    display(confusions); confusions.to_csv(output_root / "confusions.csv", index=False)
    correct = y == p
    selections = [np.flatnonzero(correct)[np.argsort(confidence[correct])[-3:]],
                  np.flatnonzero(correct)[np.argsort(confidence[correct])[:3]],
                  np.flatnonzero(~correct)[np.argsort(confidence[~correct])[-3:]]]
    fig, axes = plt.subplots(3, 3, figsize=(12, 11))
    for row, indices in enumerate(selections):
        for ax in axes[row]: ax.axis("off")
        for ax, i in zip(axes[row], indices):
            with Image.open(DATA_ROOT / frame.iloc[i]["file_name"]) as im: ax.imshow(im.convert("RGB"))
            ax.set_title(f"Real: {names.loc[y[i], 'especie']}\nPred: {names.loc[p[i], 'especie']}\np={confidence[i]:.3f}", fontsize=8)
    fig.suptitle(best_name + " — aciertos alta/baja confianza y errores alta confianza")
    fig.tight_layout(); fig.savefig(output_root / "errors.png", dpi=150)
    plt.show(); plt.close(fig)
    lt = dict(np.load(CACHE / "longtail.npz"))
    tier_rows = []
    # Include T3 as a balanced descriptive reference; quantity also differs.
    for item in historical["experiments"]:
        if item["id"] not in ("T3", "L2", "L3", "L4", "L5"): continue
        path = root / "references/khipu_diego" / historical["commit"][:12] / "resultados" / item["id"] / "preds.npz"
        with np.load(path) as a: yt, yp = a["y"].astype(int), a["top5"][:, 0].astype(int)
        sup = np.bincount(yt,minlength=10000); pc = np.bincount(yp,minlength=10000)
        tp = np.bincount(yt[yt==yp],minlength=10000)
        fs = np.divide(2*tp,sup+pc,out=np.zeros(10000),where=sup+pc>0)
        precision = np.divide(tp,pc,out=np.zeros(10000),where=pc>0)
        for group, name in enumerate(lt["nombres_grupo"]):
            mask = lt["grupo"] == group
            tier_rows.append({"ID":"D-"+item["id"],"grupo":str(name),"especies":int(mask.sum()),
                              "train_LT_images":int(lt["n_por_clase"][mask].sum()),"macro_f1":float(fs[mask].mean()),
                              "precision":float(precision[mask].mean()),"recall":float((tp[mask]/sup[mask]).mean())})
    tiers = pd.DataFrame(tier_rows); display(tiers); tiers.to_csv(output_root / "longtail_groups.csv",index=False)
