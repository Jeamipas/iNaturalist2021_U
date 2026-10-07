"""Verificación local de las 30 ejecuciones originales, sin GPU ni inferencia."""
from pathlib import Path
import base64
import hashlib
import json

import numpy as np

RUN_ID = "convnext4_adam_muon_v3"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prediction_metrics(path):
    with np.load(path, allow_pickle=False) as arrays:
        y = arrays["y"].astype(np.int64)
        top5 = arrays["top5"].astype(np.int64)
        assert y.shape == (100000,) and top5.shape == (100000, 5), path
        assert np.all((y >= 0) & (y < 10000))
        assert np.all((top5 >= 0) & (top5 < 10000))
        support = np.bincount(y, minlength=10000)
        assert np.all(support == 10), "Evaluación incompleta o desbalanceada"
        p = top5[:, 0]
        count = np.bincount(p, minlength=10000)
        tp = np.bincount(y[y == p], minlength=10000)
        f1 = 2 * tp / (support + count)
        metrics = {"val_top1": float(np.mean(y == p)),
                   "val_top5": float(np.mean(np.any(top5 == y[:, None], axis=1))),
                   "val_f1_macro": float(f1.mean()),
                   "val_f1_weighted": float(np.average(f1, weights=support))}
        return metrics, hashlib.sha256(y.astype("<i8").tobytes()).hexdigest()


def audit_delivery(root):
    root = Path(root)
    out = root / "outputs" / RUN_ID
    reference = read(root / "references/diego_evidence.json")
    manifest = read(out / "evidence_manifest.json")
    for relative, expected in manifest["files"].items():
        assert digest(root / relative) == expected, relative
    records, labels = {}, set()
    for item in reference["experiments"]:
        exp = item["id"]
        path = root / "experimento-dp-khipu/resultados" / exp / "preds.npz"
        assert digest(path) == item["prediction_audit"]["preds_sha256"], exp
        metrics, yhash = prediction_metrics(path)
        assert yhash == item["prediction_audit"]["y_sha256"], exp
        labels.add(yhash)
        for key, value in metrics.items():
            assert np.isclose(value, item["summary"][key], rtol=0, atol=1e-7), (exp, key)
        records["D-" + exp] = metrics
    configurations, initializations = {}, {}
    for exp in ("N01", "N02", "N03", "N04"):
        directory = out / exp
        summary = read(directory / "resumen.json")
        config = read(directory / "config.json")
        history = read(directory / "historial.json")
        initialization = read(directory / "initialization.json")
        assert config == summary["identity"]["configuration"], exp
        assert summary["imagenes_train"] == 500000 and summary["imagenes_val"] == 100000
        assert summary["parametros"] == 35510128 and len(history) == 16
        assert summary["initial_weights_sha256"] == initialization["initial_weights_sha256"]
        for relative, expected in summary["identity"]["sources"].items():
            assert digest(root / relative) == expected, (exp, relative)
        metrics, yhash = prediction_metrics(directory / "preds.npz")
        labels.add(yhash)
        for key, value in metrics.items():
            assert np.isclose(value, summary[key], rtol=0, atol=1e-7), (exp, key)
        best, selected, stale = -1., None, 0
        for row in history:
            if row["val_top1"] > best + config["min_delta"]:
                best, selected, stale = row["val_top1"], row["epoca"], 0
            else:
                stale += 1
        assert stale < config["paciencia"]
        records[exp] = {**metrics, "checkpoint_epoch": selected,
                        "executed_epochs": len(history), "early_stopping_triggered": False}
        configurations[exp] = config
        initializations[exp] = summary["initial_weights_sha256"]
    for left, right in (("N01", "N02"), ("N03", "N04")):
        controls = lambda c: {k: v for k, v in c.items() if k not in ("id", "descripcion", "optimizador")}
        assert controls(configurations[left]) == controls(configurations[right])
        assert initializations[left] == initializations[right]
    assert len(records) == 30 and len(labels) == 1
    result = {"status": "VERIFIED", "experiments": 30, "validation_images_per_run": 100000,
              "labels_sha256": labels.pop(), "training_sources": "Coinciden con las identidades del entrenamiento",
              "paired_initializations_verified": True, "records": records}
    (out / "delivery_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("30 ejecuciones verificadas sobre las mismas 100000 imágenes; fuentes e inicializaciones emparejadas verificadas.")
    return result


def show_eda_snapshot(root):
    """Muestra EDA archivada; prepare/report la recalculan sobre el corpus."""
    from IPython.display import Image, display
    snapshot = read(Path(root) / "references/eda_snapshot.json")
    print(snapshot["scope"])
    for output in snapshot["outputs"]:
        data = output.get("data", {})
        if "image/png" in data:
            display(Image(data=base64.b64decode(data["image/png"])))
        elif "text/plain" in data:
            print("".join(data["text/plain"]))
        elif "text" in output:
            print("".join(output["text"]))


def show_n03_examples(root):
    """Reconstruye y verifica la selección visual del mejor modelo."""
    import csv
    import matplotlib.pyplot as plt
    from PIL import Image
    root = Path(root)
    source = root / "references/n03_errors"
    manifest = read(source / "manifest.json")
    assert digest(source / "especies.csv") == manifest["species_sha256"]
    path = root / "outputs" / RUN_ID / "N03/preds.npz"
    assert digest(path) == manifest["predictions_sha256"]
    with np.load(path, allow_pickle=False) as arrays:
        y, top5, prob5 = arrays["y"], arrays["top5"], arrays["prob5"]
        confidence = prob5[:, 0].astype(float)
        correct = y == top5[:, 0]
        masks = {"correctas_alta": correct & (confidence > .9),
                 "correctas_baja": correct & (confidence < .2),
                 "incorrectas_alta": ~correct & (confidence > .7),
                 "dificiles": ~np.any(top5 == y[:, None], axis=1) & (confidence < .05)}
        rng = np.random.default_rng(manifest["seed"])
        groups = {}
        for criterion, mask in masks.items():
            assert int(mask.sum()) == manifest["counts"][criterion]
            selected = rng.choice(np.flatnonzero(mask), 6, replace=False)
            samples = [s for s in manifest["samples"] if s["criterion"] == criterion]
            assert selected.tolist() == [s["prediction_row"] for s in samples]
            for sample in samples:
                i = sample["prediction_row"]
                assert int(y[i]) == sample["true_class"] and top5[i].tolist() == sample["top5"]
                assert float(confidence[i]) == sample["confidence"]
                assert digest(source / sample["image"]) == sample["file_sha256"]
                with Image.open(source / sample["image"]) as im:
                    assert im.size == (144, 144)
                    assert hashlib.sha256(np.asarray(im.convert("RGB")).tobytes()).hexdigest() == sample["pixels_sha256"]
            groups[criterion] = samples[:3]
    with (source / "especies.csv").open(encoding="utf-8", newline="") as stream:
        species = {int(row["etiqueta"]): row["especie"] for row in csv.DictReader(stream)}
    titles = {"correctas_alta": "Aciertos de alta confianza", "correctas_baja": "Aciertos de baja confianza",
              "incorrectas_alta": "Errores de alta confianza", "dificiles": "Real fuera de Top-5 y confianza <0.05"}
    for criterion, samples in groups.items():
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, sample in zip(axes, samples):
            with Image.open(source / sample["image"]) as im:
                ax.imshow(im.convert("RGB").crop((8, 8, 136, 136)))
            ax.axis("off")
            ax.set_title(f"Real: {species[sample['true_class']]}\nPred.: {species[sample['predicted_class']]}\np={sample['confidence']:.3f}", fontsize=9)
        fig.suptitle("N03: " + titles[criterion])
        fig.tight_layout()
        plt.show()
        plt.close(fig)
    print("24 ejemplos auditados; 12 ilustrados. Semilla 0, recorte real de 128 px; observaciones visuales sin inferencia causal.")
