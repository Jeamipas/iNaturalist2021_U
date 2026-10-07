"""Network file transfers only; installation, verification and training use Slurm.

Khipu compute nodes cannot reach Internet. The SSH-access node downloads the
official inputs into shared project storage; CPU/GPU allocations use them offline.
No model is loaded or trained here.
"""
import argparse
import fcntl
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.full_data import ARCHIVES, SOURCE, atomic_json


def download(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    partial = Path(str(target) + ".part")
    subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error", "--retry", "5",
                    "--continue-at", "-", "--output", str(partial), url], check=True)
    partial.replace(target)  # MD5/SHA validation is performed by the CPU allocation.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--packages-only", action="store_true")
    args = parser.parse_args()
    locks = ROOT / "cache"
    locks.mkdir(exist_ok=True)
    with (locks / "network_download.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("A download process is already active", flush=True)
            return
        status = locks / "staging_status.json"
        packages = locks / "packages_status.json"
        wheelhouse = locks / "wheelhouse"
        wheelhouse.mkdir(exist_ok=True)
        try:
            if not packages.exists() or json.loads(packages.read_text()).get("state") != "READY":
                atomic_json(status, {"state": "DOWNLOADING_PACKAGES"})
                for requirements, index in [
                    (["torch==2.6.0", "torchvision==0.21.0"], "https://download.pytorch.org/whl/cu124"),
                    (["pip==24.3.1", "-r", str(ROOT / "requirements-full.txt")], "https://pypi.org/simple"),
                ]:
                    subprocess.run([sys.executable, "-m", "pip", "download", "--disable-pip-version-check",
                                    "--only-binary=:all:", "--dest", str(wheelhouse), "--index-url", index,
                                    *requirements], check=True)
                atomic_json(packages, {"state": "READY", "wheelhouse": str(wheelhouse)})
            # The wheelhouse can be ready while pretrained weights are absent.
            # File transfer only; checksum and model loading happen in Slurm.
            download("https://download.pytorch.org/models/convnext_tiny-983f1562.pth",
                     locks / "torch/hub/checkpoints/convnext_tiny-983f1562.pth")
            atomic_json(locks / "convnext_weights_status.json", {"state": "READY", "weights": "IMAGENET1K_V1"})
            if not args.packages_only:
                for name in ARCHIVES:
                    atomic_json(status, {"state": "DOWNLOADING_DATA", "archive": name})
                    print(f"Downloading official data archive: {name}", flush=True)
                    download(SOURCE + name, ROOT / "data/archives" / name)
                atomic_json(status, {"state": "READY", "archives": list(ARCHIVES)})
            print("Network staging complete", flush=True)
        except Exception as error:
            atomic_json(status, {"state": "FAILED", "error": repr(error)})
            raise


if __name__ == "__main__":
    main()
