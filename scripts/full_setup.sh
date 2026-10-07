#!/bin/bash
set -euo pipefail
ROOT="$1"
module load python3/3.11.11
cd "$ROOT"
python3 scripts/execute_project_notebook.py --materialize
# File transfers run on the network-access node; compute uses shared storage.
WAIT_START=$SECONDS
while [[ ! -f "$ROOT/cache/packages_status.json" ]]; do
    if [[ -f "$ROOT/cache/staging_status.json" ]] && python3 -c 'import json,sys; sys.exit(json.load(open(sys.argv[1])).get("state") != "FAILED")' "$ROOT/cache/staging_status.json"; then
        cat "$ROOT/cache/staging_status.json"
        exit 1
    fi
    if (( SECONDS - WAIT_START > 25200 )); then
        scontrol requeue "$SLURM_JOB_ID"
        exit 0
    fi
    sleep 30
done
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
    python3 -m venv "$ROOT/.venv"
fi
PY="$ROOT/.venv/bin/python"
"$PY" -m pip install --no-index --find-links "$ROOT/cache/wheelhouse" --disable-pip-version-check pip==24.3.1
"$PY" -m pip install --no-index --find-links "$ROOT/cache/wheelhouse" --disable-pip-version-check torch==2.6.0 torchvision==0.21.0 -r requirements-full.txt
"$PY" -m pip check
"$PY" -m ipykernel install --prefix "$ROOT/.venv" --name naturalist-full --display-name 'Naturalist full corpus'
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_HOME="$ROOT/cache/torch"
while [[ ! -f "$ROOT/cache/torch/hub/checkpoints/convnext_tiny-983f1562.pth" ]]; do
    if (( SECONDS > 25200 )); then
        scontrol requeue "$SLURM_JOB_ID"
        exit 0
    fi
    sleep 30
done
"$PY" -c 'from src.full_data import digest; from pathlib import Path; import os; p=Path(os.environ["TORCH_HOME"])/"hub/checkpoints/convnext_tiny-983f1562.pth"; assert digest(p).startswith("983f1562"), "ImageNet checkpoint SHA256 mismatch"'
"$PY" -m unittest discover -s tests -p test_full_pipeline.py -v
"$PY" -m unittest discover -s tests -p test_diego_extension.py -v
"$PY" -m pip freeze > "$ROOT/environment-freeze.txt"
while ! python3 -c 'import json,sys; sys.exit(json.load(open(sys.argv[1])).get("state") != "READY")' "$ROOT/cache/staging_status.json"; do
    if python3 -c 'import json,sys; sys.exit(json.load(open(sys.argv[1])).get("state") != "FAILED")' "$ROOT/cache/staging_status.json"; then
        cat "$ROOT/cache/staging_status.json"
        exit 1
    fi
    if (( SECONDS > 25200 )); then
        scontrol requeue "$SLURM_JOB_ID"
        exit 0
    fi
    sleep 30
done
while ! python3 -c 'import json,sys; sys.exit(json.load(open(sys.argv[1])).get("state") != "READY")' "$ROOT/cache/diego_predictions_status.json"; do
    if [[ -f "$ROOT/cache/diego_predictions_status.json" ]] && python3 -c 'import json,sys; sys.exit(json.load(open(sys.argv[1])).get("state") != "FAILED")' "$ROOT/cache/diego_predictions_status.json"; then
        cat "$ROOT/cache/diego_predictions_status.json"
        exit 1
    fi
    if (( SECONDS > 25200 )); then
        scontrol requeue "$SLURM_JOB_ID"
        exit 0
    fi
    sleep 30
done
export NATURALIST_MODE=prepare
"$PY" scripts/execute_project_notebook.py
