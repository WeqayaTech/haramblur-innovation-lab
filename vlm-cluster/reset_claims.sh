#!/bin/bash
# Drop claims for cells that have no map json (dead runners). Run ONLY when no run_cells.sh is active on any pod.
for d in /workspace/exp22/claims/*/; do k=$(basename $d); [ -f "/workspace/exp22/eval/${k}_map.json" ] || { rm -rf "$d"; echo "released $k"; }; done
