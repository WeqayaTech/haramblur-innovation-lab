#!/bin/bash
# EXP-2026-22 single-pod queues (gpu | cpu | exports). For multi-pod runs use run_cells.sh with shard files.
set -u
QUEUE=${1:-cpu}
while ! grep -q VENV_READY /workspace/venv_setup.log 2>/dev/null; do sleep 15; done
source "$(dirname "$0")/matrix_lib.sh"
echo "CONFIG queue=$QUEUE threads=$T $(date +%T)"; python -c "import ultralytics; print('ultralytics', ultralytics.__version__)"
if [ "$QUEUE" = exports ]; then
  for sz in 640 416 320; do for run in $MODELS; do export_all $run $sz; done; done
  echo EXPORTS_DONE $(date +%T); exit 0
fi
if [ "$QUEUE" = gpu ]; then
  for run in $MODELS; do for sz in 640 416 320; do cell spotval $run pt $sz; done; done
  for sz in 640 416 320; do for run in $MODELS; do cell holdout $run pt $sz; done; done
  echo GPU_QUEUE_DONE $(date +%T)
else
  for sz in 640 416 320; do for run in $MODELS; do export_all $run $sz; done; done
  echo EXPORTS_DONE $(date +%T)
  for sz in 640 416 320; do for tag in fp32 fp16 int8 fdec; do cell spotval yolo11N-640 $tag $sz; done; done
  for run in y26n_humanshaped_v2 y26n_noe2e_warm50-2 y26s_humanshaped_smallpatch_v1 y26n_humanshaped_v2_distill_v1; do for sz in 640 416 320; do for tag in fp32 fp16 int8 fdec; do cell spotval $run $tag $sz; done; done; done
  for sz in 640 416 320; do for run in $MODELS; do cell holdout $run int8 $sz; cell holdout $run fdec $sz; done; done
  for sz in 640 416 320; do for run in $MODELS; do cell holdout $run fp32 $sz; cell holdout $run fp16 $sz; done; done
  echo CPU_QUEUE_DONE $(date +%T)
fi
