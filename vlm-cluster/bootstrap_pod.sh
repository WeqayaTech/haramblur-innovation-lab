#!/bin/bash
# New-pod bootstrap for EXP-2026-22 shards: build the venv (CPU torch on CPU pods), then run this pod's shard(s).
#   usage: bash /workspace/exp22/bootstrap_pod.sh <podname>     (shards: /workspace/exp22/shards/<podname>_{cpu,gpu}.txt)
set -u
NAME=$1; S=/workspace/exp22/shards; L=/workspace/exp22/logs; mkdir -p $L
if [ ! -x /root/venvs/export_tflite/bin/python ]; then
  python3 -m venv /root/venvs/export_tflite
  if nvidia-smi > /dev/null 2>&1; then /root/venvs/export_tflite/bin/pip install -q ultralytics==8.4.146 ai-edge-litert==2.2.0 ai-edge-quantizer==0.9.0 litert-torch==0.9.4 pycocotools
  else /root/venvs/export_tflite/bin/pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu && /root/venvs/export_tflite/bin/pip install -q ultralytics==8.4.146 ai-edge-litert==2.2.0 ai-edge-quantizer==0.9.0 litert-torch==0.9.4 pycocotools; fi
fi
/root/venvs/export_tflite/bin/python -c "import ultralytics, ai_edge_litert; print('VENV_OK', ultralytics.__version__)" || { echo VENV_FAILED; exit 1; }
Q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || echo -1); PP=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null || echo 100000); CORES=$(nproc); [ "$Q" -gt 0 ] && CORES=$((Q / PP))
P=$(( CORES / 5 )); [ $P -lt 1 ] && P=1; [ $P -gt 8 ] && P=8
if nvidia-smi > /dev/null 2>&1; then
  (nohup bash /workspace/exp22/run_cells.sh /workspace/exp22/lists/cells_pt.txt 1 > $L/${NAME}_gpu.log 2>&1 < /dev/null &); echo "launched gpu shard"
  P=$(( (CORES - 2) / 5 )); [ $P -lt 1 ] && P=1
fi
{ (nohup bash /workspace/exp22/run_cells.sh /workspace/exp22/lists/cells_all.txt $P > $L/${NAME}_cpu.log 2>&1 < /dev/null &); echo "launched cpu shard P=$P"; }
echo "BOOTSTRAP_DONE $NAME cores=$CORES $(date +%T)"
