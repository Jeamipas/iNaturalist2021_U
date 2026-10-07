"""SSH/Slurm orchestration called from the deliverable notebook; stdlib only."""
import hashlib
import json
import os
import re
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def command(arguments):
    result = subprocess.run([str(a) for a in arguments], text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {result.stderr.strip()}")
    return result.stdout.strip()


def ssh(remote_command):
    return command(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "khipu", remote_command])


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def start_network_staging(remote):
    # Only network transfers on the SSH-access node. CPU preparation and model
    # work stay within Slurm. flock in the script prevents duplicate downloads.
    return ssh("module load python3/3.11.11; cd " + shlex.quote(remote)
               + "; nohup python3 scripts/stage_khipu_downloads.py > logs/network-staging.log 2>&1 < /dev/null & echo $!")


def gpu_arguments(settings):
    """Use the same explicit allocation for CUDA smoke and all four workers."""
    gres = settings.get("gpu_gres", "gpu:rtxa6000:1")
    node = settings.get("gpu_node")
    if not re.fullmatch(r"(?:gpu|shard):[a-zA-Z0-9_.]+:[1-9][0-9]*", gres):
        raise ValueError("Invalid GPU allocation")
    if node and not re.fullmatch(r"[a-zA-Z0-9_-]+", node):
        raise ValueError("Invalid GPU node")
    return ["--gres=" + gres] + (["--nodelist=" + node] if node else [])


def make_slurm_scripts(settings, count):
    root = settings["remote_root"]
    run = settings["run_id"]
    gpu_arguments(settings)
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", run) or not re.fullmatch(r"/home/jeanpier\.garay/proyectos/[a-zA-Z0-9_/-]+", root):
        raise ValueError("Invalid project path or run ID")
    environment = "\n".join([
        f"export NATURALIST_RUN_ID={shlex.quote(run)}",
        f"export NATURALIST_DATA_ROOT={shlex.quote(root + '/data')}",
        f"export TORCH_HOME={shlex.quote(root + '/cache/torch')}",
        f"export NATURALIST_GPU_GRES={shlex.quote(settings.get('gpu_gres', 'gpu:rtxa6000:1'))}",
        f"export NATURALIST_SCOPE={count}", "export PYTHONHASHSEED=42", "export CUBLAS_WORKSPACE_CONFIG=:4096:8",
        "export OMP_NUM_THREADS=4", "export MKL_NUM_THREADS=4",
        "export NATURALIST_SEGMENT_SECONDS=25200",  # Leave one hour for final evaluation/requeue.
        "export NATURALIST_DEADLINE_SECONDS=27900",  # Watchdog: 7h45m, below the 8h QOS limit.
    ])
    common = "#!/bin/bash\nset -euo pipefail\n" + environment + "\n"
    prepare = common + f"bash {shlex.quote(root + '/scripts/full_setup.sh')} {shlex.quote(root)}\n"
    train = common + f"module load python3/3.11.11\ncd {shlex.quote(root)}\nexport NATURALIST_MODE=train\n{shlex.quote(root + '/.venv/bin/python')} scripts/execute_project_notebook.py\n"
    report = common + f"module load python3/3.11.11\ncd {shlex.quote(root)}\nexport NATURALIST_MODE=report\n{shlex.quote(root + '/.venv/bin/python')} scripts/execute_project_notebook.py\n"
    scripts = {"prepare.sbatch": prepare, "train.sbatch": train, "report.sbatch": report}
    if settings.get("gpu_smoke"):
        scripts["smoke.sbatch"] = common + f"module load python3/3.11.11\ncd {shlex.quote(root)}\n{shlex.quote(root + '/.venv/bin/python')} scripts/smoke_diego_pair.py\n"
    return scripts


def submit(root, settings, plan):
    root = Path(root)
    output = root / "outputs" / settings["run_id"]
    manifest_path = output / "khipu_jobs.json"
    notebook = root / "notebooks" / "00_entregable_reproducible.ipynb"
    plan_hash = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    gpu_request = gpu_arguments(settings)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["plan_hash"] != plan_hash:
            raise ValueError("Existing Slurm submission uses a different plan; change run_id")
        if manifest.get("report_job"):
            print("Existing submission reused; no duplicate jobs.")
            print(status(manifest))
            return manifest
    else:
        manifest = {"run_id": settings["run_id"], "remote_root": settings["remote_root"], "plan_hash": plan_hash,
                    "local_root": str(root), "created_utc": datetime.now(timezone.utc).isoformat(), "experiments": [p["id"] for p in plan],
                    "resources": {"gpu_gres": settings.get("gpu_gres", "gpu:rtxa6000:1"),
                                  "node": settings.get("gpu_node"), "cpus_per_task": 32,
                                  "memory_gib": 96, "array_concurrency": 1}}
    remote = settings["remote_root"]
    # Read-only connection check and remote collision detection precede writes.
    print(ssh("id -un; hostname; pwd"))
    remote_manifest = remote + "/outputs/" + settings["run_id"] + "/khipu_jobs.json"
    recovered = ssh("if test -f " + shlex.quote(remote_manifest) + "; then cat " + shlex.quote(remote_manifest) + "; fi")
    if recovered:
        remote_value = json.loads(recovered)
        if remote_value["plan_hash"] != plan_hash:
            raise ValueError("Remote run directory has a different plan")
        manifest.update(remote_value)
        atomic_json(manifest_path, manifest)
        if manifest.get("report_job"):
            return manifest
    directories = [remote + "/" + part for part in ["notebooks", "scripts", "src", "tests", "references", "logs/slurm", "outputs/" + settings["run_id"], "jobs"]]
    ssh("mkdir -p " + " ".join(shlex.quote(p) for p in directories))
    shared = settings.get("reuse_project_root")
    if shared:
        if not re.fullmatch(r"/home/jeanpier\.garay/proyectos/[a-zA-Z0-9_/-]+", shared) or shared == remote:
            raise ValueError("Invalid shared project directory")
        for name in ("data", "cache", ".venv"):
            source, destination = shared + "/" + name, remote + "/" + name
            ssh("test -e " + shlex.quote(source) + "; if test ! -e " + shlex.quote(destination)
                + "; then ln -s " + shlex.quote(source) + " " + shlex.quote(destination) + "; fi")
    scripts_dir = output / "slurm"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    for name, source in make_slurm_scripts(settings, len(plan)).items():
        path = scripts_dir / name
        path.write_text(source, encoding="utf-8", newline="\n")
        command(["scp", path, f"khipu:{remote}/jobs/{name}"])
    for relative in ["notebooks/00_entregable_reproducible.ipynb", "scripts/execute_project_notebook.py", "scripts/full_setup.sh", "scripts/stage_khipu_downloads.py", "scripts/stage_diego_predictions.py", "references/diego_evidence.json", "src/full_data.py", "requirements-full.txt"]:
        command(["scp", root / relative, f"khipu:{remote}/{relative}"])
    ssh("bash -n " + shlex.quote(remote + "/scripts/full_setup.sh"))
    for name in make_slurm_scripts(settings, len(plan)):
        ssh("bash -n " + shlex.quote(remote + "/jobs/" + name))
    manifest["network_staging_pid"] = start_network_staging(remote)
    manifest["reference_staging_pid"] = ssh("module load python3/3.11.11; cd " + shlex.quote(remote)
        + "; nohup python3 scripts/stage_diego_predictions.py > logs/reference-staging.log 2>&1 < /dev/null & echo $!")

    def persist():
        atomic_json(manifest_path, manifest)
        command(["scp", manifest_path, "khipu:" + remote_manifest])

    def sbatch(arguments):
        response = ssh("sbatch --parsable " + " ".join(shlex.quote(str(a)) for a in arguments))
        job_id = response.split(";")[0]
        if not job_id.isdigit():
            raise RuntimeError("Unexpected sbatch response: " + response)
        return job_id

    if not manifest.get("prepare_job"):
        prepare_dependencies = []
        if settings.get("data_ready_job"):
            parent = str(settings["data_ready_job"])
            if not parent.isdigit():
                raise ValueError("Invalid data preparation job")
            parent_state = ssh("sacct -j " + parent + " -X -n -o State").strip()
            if parent_state != "COMPLETED":
                if parent_state not in ("RUNNING", "PENDING", "REQUEUED", "CONFIGURING"):
                    raise RuntimeError("Previous data preparation is not usable: " + parent_state)
                prepare_dependencies = ["--dependency=afterok:" + parent]
        manifest["prepare_job"] = sbatch(["--account=postgrado", "--qos=a-postgrado", "--partition=standard", f"--cpus-per-task={int(settings.get('prepare_cpus', 8))}", "--mem=" + settings.get('prepare_memory', '32G'), "--time=08:00:00", "--requeue", "--job-name=nat_diego_prepare", *prepare_dependencies,
                                          f"--output={remote}/logs/slurm/prepare-%j.log", remote + "/jobs/prepare.sbatch"])
        persist()
    if settings.get("gpu_smoke") and not manifest.get("smoke_job"):
        manifest["smoke_job"] = sbatch(["--account=postgrado", "--qos=a-postgrado", "--partition=gpu", *gpu_request, "--cpus-per-task=4", "--mem=16G", "--time=00:15:00", "--job-name=nat_diego_smoke",
                                       f"--dependency=afterok:{manifest['prepare_job']}", f"--output={remote}/logs/slurm/smoke-%j.log", remote + "/jobs/smoke.sbatch"])
        persist()
    if not manifest.get("train_job"):
        manifest["train_job"] = sbatch(["--account=postgrado", "--qos=a-postgrado", "--partition=gpu", *gpu_request, "--cpus-per-task=32", "--mem=96G", "--time=08:00:00", "--requeue",
                                        f"--array=0-{len(plan)-1}%1", "--job-name=" + settings.get('job_name', 'nat_pair'), f"--dependency=afterok:{manifest.get('smoke_job', manifest['prepare_job'])}",
                                        f"--output={remote}/logs/slurm/train-%A_%a.log", remote + "/jobs/train.sbatch"])
        persist()
    if not manifest.get("report_job"):
        manifest["report_job"] = sbatch(["--account=postgrado", "--qos=a-postgrado", "--partition=standard", "--cpus-per-task=4", "--mem=16G", "--time=01:00:00", "--job-name=nat_report",
                                         f"--dependency=afterok:{manifest['train_job']}", f"--output={remote}/logs/slurm/report-%j.log", remote + "/jobs/report.sbatch"])
        persist()
    print(json.dumps(manifest, indent=2))
    print(status(manifest))
    return manifest


def status(manifest):
    ids = ",".join(str(manifest[k]) for k in ["prepare_job", "smoke_job", "train_job", "report_job"] if manifest.get(k))
    if not ids:
        return "No jobs submitted"
    return ssh("squeue -j " + shlex.quote(ids) + " -o '%i %j %T %M %R'; sacct -j " + shlex.quote(ids) + " --format=JobID,State,Elapsed,ExitCode --noheader | head -30")


def collect(root, manifest):
    root = Path(root)
    remote = manifest["remote_root"]
    output = root / "outputs" / manifest["run_id"]
    output.mkdir(parents=True, exist_ok=True)
    # Fetch only result artifacts, not checkpoints, archives or images.
    files = ssh("find " + shlex.quote(remote + "/outputs/" + manifest["run_id"]) + " -type f \\( -name '*.json' -o -name '*.csv' -o -name '*.png' -o -name '*.npz' \\) -not -path '*/executions/*'").splitlines()
    for remote_file in files:
        relative = Path(remote_file).relative_to(Path(remote + "/outputs/" + manifest["run_id"]))
        local = output / relative
        local.parent.mkdir(parents=True, exist_ok=True)
        command(["scp", "khipu:" + remote_file, local])
    report_status = ssh("sacct -j " + shlex.quote(str(manifest.get("report_job", "0"))) + " -n -X -o State")
    if report_status.strip() == "COMPLETED":
        command(["scp", "khipu:" + remote + "/notebooks/00_entregable_reproducible.ipynb", root / "notebooks/00_entregable_reproducible.ipynb"])
    for filename in ["environment-freeze.txt", "data/preparation_status.json"]:
        if ssh("test -f " + shlex.quote(remote + "/" + filename) + " && echo exists || true"):
            command(["scp", "khipu:" + remote + "/" + filename, output / Path(filename).name])
    return status(manifest)
