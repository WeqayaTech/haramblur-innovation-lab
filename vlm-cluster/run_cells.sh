#!/bin/bash
# Claim-based work queue for EXP-2026-22 cells: any pod runs this against the same list; each cell is
# claimed atomically with mkdir on the shared volume, so pods can join or leave at any time.
#   usage: bash run_cells.sh <list.txt> [P] [threads]   list lines: "<ds> <run> <tag> <sz>" in priority order
# P cells run concurrently on this pod, each with (cores / P) threads. Cells with a map json are skipped.
set -u
LIST=$1; P=${2:-1}; TOVR=${3:-0}; CLAIMS=/workspace/exp22/claims; mkdir -p $CLAIMS
Q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || echo -1); PERIOD=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null || echo 100000)
CORES=$(nproc); [ "$Q" -gt 0 ] && CORES=$((Q / PERIOD))
export THREADS=$(( CORES / P )); [ "$TOVR" -gt 0 ] && export THREADS=$TOVR; [ "$THREADS" -lt 1 ] && export THREADS=1
source "$(dirname "$0")/matrix_lib.sh"
ME=$(hostname)
echo "CONFIG list=$LIST parallel=$P cores=$CORES threads/cell=$THREADS host=$ME $(date +%T)"; python -c "import ultralytics; print('ultralytics', ultralytics.__version__)"
n=0; pids=()
while read -r ds run tag sz; do
  [ -z "$ds" ] && continue; case $ds in \#*) continue;; esac
  key="${run}_${tag}_sz${sz}_${ds}"
  [ -f "/workspace/exp22/eval/${key}_map.json" ] && continue
  [ -d "$CLAIMS/$key" ] && continue                      # someone else has it (cheap pre-check)
  while :; do alive=0; for pid in "${pids[@]}"; do kill -0 $pid 2>/dev/null && alive=$((alive+1)); done; [ "$alive" -lt "$P" ] && break; sleep 10; done
  mkdir "$CLAIMS/$key" 2>/dev/null || continue          # claim only once a slot is free (atomic on the volume)
  echo "$ME $$ $(date +%T)" > "$CLAIMS/$key/owner"
  ( cell $ds $run $tag $sz; [ -f "/workspace/exp22/eval/${key}_map.json" ] || { echo "CELL_FAILED $key (claim released)"; rm -rf "$CLAIMS/$key"; } ) & pids+=($!); n=$((n+1)); sleep 3
done < "$LIST"
wait
echo "LIST_DONE $LIST claimed=$n $(date +%T)"
