#!/bin/bash
# Wait for the shared CPU venv on the volume, then start this pod's queue runner. usage: cpu_start.sh <name> <P>
NAME=$1; P=${2:-6}; V=/workspace/venvs/cpu_py311
mkdir -p /root/venvs /workspace/exp22/logs; ln -sfn $V /root/venvs/export_tflite
until $V/bin/python -c "import ultralytics, ai_edge_litert, torch" 2>/dev/null; do sleep 20; done
echo "VENV_OK $(date +%T)"
(nohup bash /workspace/exp22/run_cells.sh /workspace/exp22/lists/cells_all.txt $P > /workspace/exp22/logs/${NAME}_cpu.log 2>&1 < /dev/null &)
echo "LAUNCHED $NAME P=$P $(date +%T)"
