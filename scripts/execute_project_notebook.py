"""Execute the same deliverable notebook for preparation, a worker, or reporting."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "00_entregable_reproducible.ipynb"


def requeue_current_allocation():
    """Requeue only the worker's own array task (or its own preparation job)."""
    job = os.environ.get("SLURM_JOB_ID", "")
    if "SLURM_ARRAY_JOB_ID" in os.environ:
        job = os.environ["SLURM_ARRAY_JOB_ID"] + "_" + os.environ["SLURM_ARRAY_TASK_ID"]
    if not job or any(c not in "0123456789_" for c in job):
        raise RuntimeError("Continuation requires the current Slurm allocation")
    print(f"Checkpoint saved; requeuing own allocation {job}", flush=True)
    subprocess.run(["scontrol", "requeue", job], check=True)


class AllocationDeadline(RuntimeError):
    pass


def materialize():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    context = {"ROOT": ROOT}
    cell = next(c for c in notebook["cells"] if c.get("metadata", {}).get("role") == "source_bundle")
    exec("".join(cell["source"]), context)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialize", action="store_true")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if args.materialize:
        materialize()
        return
    if args.submit:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        context = {}
        config_cell = next(c for c in notebook["cells"] if c.get("metadata", {}).get("role") == "configuration")
        exec("".join(config_cell["source"]), context)
        materialize()
        sys.path.insert(0, str(ROOT))
        from src.khipu_pipeline import submit
        submit(ROOT, context["SETTINGS"], context["PLAN"])
        return
    import nbformat
    from nbclient import NotebookClient
    run_id = os.environ.get("NATURALIST_RUN_ID", "convnext4_adam_muon_v3")
    mode = os.environ.get("NATURALIST_MODE", "review")
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        mode = "train"
        index = int(os.environ["SLURM_ARRAY_TASK_ID"])
        definition = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        configuration = next(c for c in definition["cells"] if c.get("metadata", {}).get("role") == "configuration")
        os.environ["NATURALIST_MODE"] = mode
        context = {}
        exec("".join(configuration["source"]), context)
        os.environ["NATURALIST_EXPERIMENT"] = context["PLAN"][index]["id"]
    os.environ["NATURALIST_MODE"] = mode
    os.environ["NATURALIST_ROOT"] = str(ROOT)
    os.environ.setdefault("TORCH_HOME", str(ROOT / "cache/torch"))
    os.environ["NATURALIST_ALLOCATION_STARTED"] = str(time.time())
    expired = False
    def deadline(_signum, _frame):
        nonlocal expired
        expired = True
        raise AllocationDeadline("Allocation is approaching its eight-hour time limit")
    timed = mode in ("train", "prepare") and "SLURM_JOB_ID" in os.environ
    if timed:
        signal.signal(signal.SIGALRM, deadline)
        signal.alarm(int(os.environ.get("NATURALIST_DEADLINE_SECONDS", 27900)))
    # The kernel installed in this project's venv is used explicitly.
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    client = NotebookClient(notebook, timeout=None, kernel_name="naturalist-full", resources={"metadata": {"path": str(ROOT)}}, allow_errors=False)
    output = ROOT / "outputs" / run_id / "executions" / f"{mode}_{os.environ.get('NATURALIST_EXPERIMENT', 'all')}.ipynb"
    output.parent.mkdir(parents=True, exist_ok=True)
    if mode == "train" and "SLURM_JOB_ID" in os.environ:
        allocation_keys = ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID",
                           "SLURM_RESTART_COUNT", "SLURM_JOB_NODELIST", "SLURM_CPUS_PER_TASK",
                           "SLURM_JOB_GPUS", "CUDA_VISIBLE_DEVICES", "NATURALIST_GPU_GRES")
        allocation = {key: os.environ.get(key) for key in allocation_keys}
        allocation["recorded_unix"] = time.time()
        allocation_path = ROOT / "outputs" / run_id / os.environ["NATURALIST_EXPERIMENT"]
        allocation_path.mkdir(parents=True, exist_ok=True)
        (allocation_path / f"allocation_{os.environ['SLURM_JOB_ID']}_{os.environ.get('SLURM_RESTART_COUNT', '0')}.json").write_text(
            json.dumps(allocation, indent=2), encoding="utf-8")
    try:
        try:
            client.execute()
        except Exception:
            if not expired:
                raise
        finally:
            nbformat.write(notebook, output)
    finally:
        if timed:
            signal.alarm(0)
    status_path = ROOT / "outputs" / run_id / os.environ.get("NATURALIST_EXPERIMENT", "") / "status.json"
    current_status = json.loads(status_path.read_text()) if mode == "train" and status_path.exists() else {}
    if expired or current_status.get("state") == "NEEDS_RESUME":
        if mode == "train" and not any((status_path.parent / name).exists() for name in ("last.pt", "ultimo.pt")):
            raise RuntimeError("No epoch checkpoint was saved within eight hours; review this experiment before retrying")
        requeue_current_allocation()
        return
    if mode == "report":
        nbformat.write(notebook, NOTEBOOK)
    print(f"Executed notebook saved: {output}", flush=True)


if __name__ == "__main__":
    main()
