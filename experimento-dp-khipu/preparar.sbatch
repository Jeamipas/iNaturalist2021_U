#!/bin/bash
# Preprocesado (sólo CPU): índices + memmap 144px + long-tail.  ~15-25 min.
#   sbatch preparar.sbatch
#SBATCH -J inat-preparar
#SBATCH -p standard
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
~/.venvs/dl/bin/python -u preparar_datos.py --procesos "$SLURM_CPUS_PER_TASK"
