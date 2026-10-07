"""Network-only staging of immutable historical prediction files."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.full_data import atomic_json
from scripts.stage_khipu_downloads import download


def main():
    source = json.loads((ROOT / "references/diego_evidence.json").read_text())
    status = ROOT / "cache/diego_predictions_status.json"
    try:
        for item in source["experiments"]:
            relative = f"resultados/{item['id']}/preds.npz"
            atomic_json(status, {"state": "DOWNLOADING", "id": item["id"]})
            download(f"https://raw.githubusercontent.com/Jeamipas/iNaturalist2021_U/{source['commit']}/experimento-dp-khipu/{relative}",
                     ROOT / "references/khipu_diego" / source["commit"][:12] / relative)
        atomic_json(status, {"state": "READY", "commit": source["commit"], "files": 26})
    except Exception as error:
        atomic_json(status, {"state": "FAILED", "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
