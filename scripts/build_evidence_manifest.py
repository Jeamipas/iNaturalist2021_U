"""Registra checksums de fuentes y resultados archivados del entregable."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent.parent


def main():
    notebook = json.loads((ROOT / "notebooks/00_entregable_reproducible.ipynb").read_text(encoding="utf-8"))
    paths = [ROOT / "requirements-full.txt", ROOT / "references/diego_evidence.json",
             ROOT / "references/eda_snapshot.json"]
    paths += [ROOT / name for name in ("src/full_data.py", "src/full_experiment.py", "src/khipu_pipeline.py",
              "src/diego_workflow.py", "src/diego_muon.py", "src/delivery_evidence.py")]
    paths += sorted((ROOT / "src/diego_reference").glob("*.py"))
    output = ROOT / "outputs/convnext4_adam_muon_v3"
    for exp in ("N01", "N02", "N03", "N04"):
        paths += [output / exp / name for name in ("config.json", "resumen.json", "historial.json",
                  "identity.json", "initialization.json", "preds.npz")]
    paths += [output / "environment-freeze.txt"]
    paths += sorted(p for p in (ROOT / "references/n03_errors").iterdir() if p.is_file())
    files = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    manifest = {"scope": "26 históricos y cuatro ConvNeXt originales, sin ensambles ni resultados de protocolos posteriores",
                "original_training_bundle_sha256": "cf7e2b6ba686ffcfe5765fdc2be6c120162aa125ea16898744f4904436bfcf8f",
                "delivery_bundle_sha256": notebook["metadata"]["naturalist"]["source_sha256"], "files": files}
    (output / "evidence_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Manifiesto: {len(files)} archivos")


if __name__ == "__main__":
    main()
