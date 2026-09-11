#!/bin/bash
# TF.js float + uint8 graph models for y26n_noe2e_warm50-2 and y26s_humanshaped_smallpatch_v1 at 640/416/320,
# following docs/TFJS_EXPORT_RECIPE.md + /workspace/tfjs_quant.sh (end2end=False -> raw (1,7,N), NMS in JS).
set -x
export CUDA_VISIBLE_DEVICES=""
OUT=/workspace/exports/tfjs_all; mkdir -p $OUT /root/tfjs_work
if [ ! -f /root/tfjsenv/bin/activate ]; then
  python3 -m venv /root/tfjsenv; source /root/tfjsenv/bin/activate
  pip install -q --upgrade pip
  pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
  pip install -q "ultralytics==8.4.82" onnx
  cp /workspace/exp18/train/y26n_noe2e_warm50-2/weights/best.pt /root/tfjs_work/probe.pt
  cd /root/tfjs_work; yolo export model=probe.pt format=tfjs imgsz=320 > /workspace/exports/tfjs_all/first_export_expected_fail.log 2>&1 || true
  pip uninstall -y -q tensorflow_decision_forests ydf || true
  python3 - <<'PY'
from pathlib import Path
f = Path('/root/tfjsenv/lib/python3.11/site-packages/tensorflowjs/converters/tf_saved_model_conversion_v2.py')
s = f.read_text()
s = s.replace('import tensorflow_decision_forests', 'try:\n  import tensorflow_decision_forests\nexcept Exception:\n  tensorflow_decision_forests = None')
f.write_text(s); print('patched')
PY
else
  source /root/tfjsenv/bin/activate
fi
echo ENV_READY
declare -A PT=( [y26n_noe2e_warm50-2]=/workspace/exp18/train/y26n_noe2e_warm50-2/weights/best.pt [y26s_humanshaped_smallpatch_v1]=/workspace/exp20/train/y26s_humanshaped_smallpatch_v1/weights/best.pt )
for m in "${!PT[@]}"; do
  cp "${PT[$m]}" /root/tfjs_work/$m.pt
  for SZ in 640 416 320; do
    cd /root/tfjs_work
    python3 - "$m" "$SZ" <<'PY'
import sys
from ultralytics import YOLO
m, sz = sys.argv[1], int(sys.argv[2])
model = YOLO(f'/root/tfjs_work/{m}.pt')
model.model.model[-1].end2end = False; model.model.end2end = False
model.export(format='tfjs', imgsz=sz, nms=False)
print('tfjs float export done', m, sz)
PY
    F=$OUT/${m}_${SZ}_float; Q=$OUT/${m}_${SZ}_uint8; rm -rf $F $Q; mkdir -p $Q
    cp -r /root/tfjs_work/${m}_web_model $F
    tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --quantize_uint8 '*' /root/tfjs_work/${m}_saved_model $Q 2>&1 | tail -2
    cp $F/metadata.yaml $Q/ 2>/dev/null || true
    echo "TFJS_DONE $m $SZ float=$(du -sh $F | cut -f1) uint8=$(du -sh $Q | cut -f1)"
  done
done
echo TFJS_BUILD_DONE
