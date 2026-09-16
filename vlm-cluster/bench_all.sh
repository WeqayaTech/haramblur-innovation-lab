#!/bin/bash
# CPU latency @640 for every model × {fp32, fp16, int8, fdec} on this (idle) pod -> /workspace/exp22/bench.json
source /root/venvs/export_tflite/bin/activate
X=/workspace/exports; E=/workspace/exp22/exports; E1=/workspace/exp21/exports; out=/workspace/exp22/bench.json; : > /workspace/exp22/bench.log
f() { local run=$1 tag=$2; for p in $E/$run/sz640/${run}_${tag}.tflite $X/${run}_calib500/sz640/${run}_${tag}.tflite $E1/$run/sz640/${run}_${tag}.tflite $E1/$run/${run}_fhead_decode_out.tflite; do [ "$tag" = fdec ] || [[ "$p" != *fhead_decode_out* ]] || continue; [ -s "$p" ] && { echo "$p"; return; }; done; }
python3 - << "PY" > /dev/null
PY
echo "{" > $out; first=1
for run in yolo11N-640 y26n_humanshaped_v2 y26n_noe2e_warm50-2 y26s_humanshaped_smallpatch_v1 y26n_humanshaped_v2_distill_v1; do
  [ $first = 1 ] || echo "," >> $out; first=0; echo "\"$run\": {" >> $out; f1=1
  for tag in fp32 fp16 int8 fdec; do p=$(f $run $tag); [ -s "$p" ] || { echo "MISSING $run $tag" >> /workspace/exp22/bench.log; continue; }
    r=$(OMP_NUM_THREADS=4 python3 /workspace/bench_tflite.py "$p" 2>&1 | grep -o "median=[0-9.]*ms.*p90=[0-9.]*ms" | sed "s/median=//; s/ms\tp90=/ · /; s/ms$//; s/ms *p90=/ · /")
    echo "$run $tag $p -> $r" >> /workspace/exp22/bench.log
    [ $f1 = 1 ] || echo "," >> $out; f1=0; echo "\"$tag\": \"$r\"" >> $out
  done; echo "}" >> $out
done; echo "}" >> $out; python3 -c "import json; print(json.load(open(\"$out\")))" >> /workspace/exp22/bench.log 2>&1; echo BENCH_ALL_DONE >> /workspace/exp22/bench.log
