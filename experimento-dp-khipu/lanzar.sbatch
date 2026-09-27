#!/bin/bash
# Un experimento por job:   sbatch lanzar.sbatch E03
# Límites de la QOS a-postgrado: 8 h, 32 CPU, 98 GB, 8 shards de GPU.
# Con --gres=shard:8 se comparte una RTX A6000 (48 GB) con otros usuarios;
# si un job no termina en 8 h, relanzarlo: reanuda desde resultados/<ID>/ultimo.pt.
#SBATCH -J inat
#SBATCH -p gpu
#SBATCH --gres=shard:8
# Sólo nodos con RTX A6000 (48 GB). En ag001 los shards caen en porciones MIG
# de 10 GB de la A100 compartidas con otros usuarios: ResNet-50 se quedó sin
# memoria (OOM) allí. g001 tiene una Tesla más pequeña.
#SBATCH --exclude=ag001,g001
#SBATCH -c 16
#SBATCH --mem=90G
#SBATCH -t 08:00:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
ID=${1:?"uso: sbatch lanzar.sbatch <ID>"}
scontrol update JobId="$SLURM_JOB_ID" JobName="inat-$ID" || true
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
~/.venvs/dl/bin/python -u experimentos.py "$ID" --workers 12 "${@:2}"
