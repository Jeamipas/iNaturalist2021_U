#!/bin/bash
# Copia la versión publicable del proyecto (~50 MB) a la carpeta experimento-dp-khipu/
# del repositorio compartido y muestra qué cambió. No hace commit ni push.
#   ./sincronizar_repo.sh [ruta_del_clon]      (por defecto ~/repos/iNaturalist2021_U)
set -euo pipefail
ORIGEN="$(cd "$(dirname "$0")" && pwd)"
CLON="${1:-$HOME/repos/iNaturalist2021_U}"
DEST="$CLON/experimento-dp-khipu"
mkdir -p "$DEST"
rsync -a --delete \
  --include='/*.py' --include='/*.sh' --include='/*.sbatch' --include='/*.ipynb' \
  --include='/README.md' --include='/requirements.txt' --include='/.gitignore' \
  --include='/resultados/' --include='/resultados/*/' \
  --include='/resultados/*/config.json' --include='/resultados/*/historial.json' \
  --include='/resultados/*/resumen.json' --include='/resultados/*/preds.npz' \
  --exclude='*' "$ORIGEN/" "$DEST/"
# experimentos sin terminar (sin resumen.json) no se publican
for d in "$DEST"/resultados/*/; do [ -f "$d/resumen.json" ] || rm -rf "$d"; done
du -sh "$DEST"
git -C "$CLON" status --short | head -20
