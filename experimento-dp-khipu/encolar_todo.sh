#!/bin/bash
# Encola todos los experimentos (o los que se pasen como argumentos).
# La QOS sólo deja 1 GPU por usuario, así que corren uno tras otro solos.
#   ./encolar_todo.sh                 # todos
#   ./encolar_todo.sh E03 E04 T3      # algunos
#   DEP=12345 ./encolar_todo.sh       # esperar a que termine el job 12345 (preparar)
set -euo pipefail
cd "$(dirname "$0")"
IDS=("$@")
[ ${#IDS[@]} -eq 0 ] && IDS=($(~/.venvs/dl/bin/python -c "from experimentos import EXPERIMENTOS as E; print(*[c.id for c in E])"))
OPT=()
[ -n "${DEP:-}" ] && OPT=(--dependency=afterok:$DEP)
for id in "${IDS[@]}"; do
  if [ -f "resultados/$id/resumen.json" ]; then echo "$id ya terminado, se omite"; continue; fi
  sbatch "${OPT[@]}" -J "inat-$id" lanzar.sbatch "$id"
done
