#!/bin/bash
# Join a fresh CPU/GPU pod to the EXP-2026-22 queue: local venv (container disk, ~4 min) then the runner.
#   bash /workspace/exp22/join_pod.sh <podname> [P]       P = parallel cells (default cores/4, max 8)
set -u
NAME=$1; CORES=$(nproc); P=${2:-$(( CORES / 4 ))}; [ "$P" -lt 1 ] && P=1; [ "$P" -gt 8 ] && P=8
V=/root/venvs/export_tflite; L=/workspace/exp22/logs; mkdir -p /root/venvs $L
if ! $V/bin/python -c "import ultralytics, ai_edge_litert, torch" 2>/dev/null; then
  rm -rf $V; export PIP_NO_CACHE_DIR=1
  PY=$(command -v python3.11 || command -v python3.12 || command -v python3.10); echo "python: $PY"
  $PY -m venv --without-pip $V && curl -sS https://bootstrap.pypa.io/get-pip.py | $V/bin/python
  if nvidia-smi > /dev/null 2>&1; then $V/bin/pip install -q --no-cache-dir torch==2.13.0 torchvision --index-url https://download.pytorch.org/whl/cu124
  else $V/bin/pip install -q --no-cache-dir torch==2.13.0 torchvision --index-url https://download.pytorch.org/whl/cpu; fi
  $V/bin/pip install -q --no-cache-dir ultralytics==8.4.146 ai-edge-litert==2.2.0 ai-edge-quantizer==0.9.0 litert-torch==0.9.4 pycocotools opencv-python-headless
fi
$V/bin/python -c "import ultralytics, ai_edge_litert, torch; print('VENV_OK', ultralytics.__version__, torch.__version__)" || { echo VENV_FAILED; exit 1; }
LIST=/workspace/exp22/lists/cells_all.txt; nvidia-smi > /dev/null 2>&1 && LIST=/workspace/exp22/lists/cells_pt.txt
(nohup bash /workspace/exp22/run_cells.sh $LIST $P > $L/${NAME}.log 2>&1 < /dev/null &)
echo "JOINED $NAME cores=$CORES P=$P list=$(basename $LIST) $(date +%T)"
