#!/bin/bash
# Resumen del avance:  ./estado.sh
cd "$(dirname "$0")"
PY=~/.venvs/dl/bin/python
echo "== Terminados"
for f in resultados/*/resumen.json; do
  [ -e "$f" ] || { echo "  (ninguno)"; break; }
  $PY -c "import json,sys;r=json.load(open('$f'));print(f\"  {r['id']:4s} top1={r['val_top1']:6.1%}  top5={r['val_top5']:6.1%}  macroF1={r['val_f1_macro']:6.1%}  {r['tiempo_train_s']/60:4.0f} min  {r['descripcion']}\")"
done
echo "== Corriendo"
squeue -u "$USER" -t R -h -o "%j %M %N" | while read nombre tiempo nodo; do
  id=${nombre#inat-}
  log=$(ls -t logs/${nombre}-*.out 2>/dev/null | head -1)
  ult=$(grep "^epoca=" "$log" 2>/dev/null | tail -1)
  ep=$(sed -E 's/^epoca=([0-9]+).*/\1/' <<<"$ult")
  tot=$($PY -c "from experimentos import POR_ID;print(POR_ID['$id'].epocas)" 2>/dev/null)
  top1=$(grep -oE "val_top1=[0-9.]+" <<<"$ult" | cut -d= -f2)
  echo "  $id en $nodo, $tiempo transcurrido: época ${ep:-0}/$tot, top1 val=${top1:--}"
done
echo "== En cola: $(squeue -u "$USER" -t PD -h | wc -l) jobs"
