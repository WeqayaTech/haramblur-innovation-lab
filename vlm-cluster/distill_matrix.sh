#!/bin/bash
# EXP-2026-21 addendum (2026-09-11): full precision matrix for one extra checkpoint
# (fp32 .pt on GPU, fp32 TFLite, INT8 W8A8, W8A8+float-decode fix) at 640/416/320 on Spotlight-val,
# plus the CPU latency bench that the previous pod never ran. Same data/scorer as quant_calib500_pipeline.sh.
# usage: bash distill_matrix.sh <run_name> <best.pt>      (logs: /workspace/exp21/<run>_matrix.log)
set -u
R=${1:-y26n_humanshaped_v2_distill_v1}; PT=${2:-/workspace/exp20/train/$R/weights/best.pt}
export TMPDIR=/workspace/tmp; mkdir -p /workspace/tmp
while ! grep -q VENV_READY /workspace/venv_setup.log 2>/dev/null; do sleep 15; done
source /root/venvs/export_tflite/bin/activate
Q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || echo -1); P=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null || echo 100000)
T=4; [ "$Q" -gt 0 ] && T=$((Q / P)); [ "$T" -lt 1 ] && T=1; echo "CONFIG run=$R threads=$T $(date +%T)"
DATA=/workspace/exp12/spotlight_oiv7_calib500.yaml; CAL=/workspace/exp12/calib_train500.txt
VLM=/workspace/data_inspection_tools/vlm-cluster; SV_IMG=/workspace/exp12/images_val4232
SV_GT=/workspace/spotlight/run/oiv7_val/labels; SV_IGN=/workspace/spotlight/run/oiv7_val/ignore_unk
E=/workspace/exp21/exports/$R; O=/workspace/exp21/eval; mkdir -p $E $O
RX=".*Detect_23;.*|.*_NormalizeCoords;.*|.*serving_default_output_0.*"
python -c "import ultralytics; print('ultralytics', ultralytics.__version__)"; sha256sum $CAL $PT

# ---- 1. latency bench first (idle CPU): the pending EXP-2026-21 item + this model ----
X=/workspace/exports; E2=/workspace/exp21/exports
bench() { echo "== BENCH $1"; OMP_NUM_THREADS=4 python3 /workspace/bench_tflite.py "$1" 2>&1 | grep -v XNNPACK | tail -3; }
for f in $X/y26n_humanshaped_v2_calib500/sz640/y26n_humanshaped_v2_fp32.tflite $X/y26n_humanshaped_v2_calib500/sz640/y26n_humanshaped_v2_int8.tflite \
         $E2/y26n_humanshaped_v2/y26n_humanshaped_v2_fhead_decode_out.tflite $E2/y26n_humanshaped_v2/y26n_humanshaped_v2_w8a32.tflite $E2/y26n_humanshaped_v2/y26n_humanshaped_v2_w8a16.tflite \
         $X/y26s_humanshaped_smallpatch_v1_calib500/sz640/y26s_humanshaped_smallpatch_v1_fp32.tflite $X/y26s_humanshaped_smallpatch_v1_calib500/sz640/y26s_humanshaped_smallpatch_v1_int8.tflite \
         $E2/y26s_humanshaped_smallpatch_v1/y26s_humanshaped_smallpatch_v1_fhead_decode_out.tflite; do [ -s "$f" ] && bench "$f" || echo "BENCH_MISSING $f"; done
echo BENCH_PART1_DONE $(date +%T)

# ---- 2. exports for the new checkpoint ----
export CUDA_VISIBLE_DEVICES=""
for SZ in 640 416 320; do
  d=$E/sz$SZ; mkdir -p $d; cp -n $PT $d/$R.pt
  if [ ! -s $d/${R}_fp32.tflite ]; then
    ( cd $d && OMP_NUM_THREADS=$T yolo export model=$d/$R.pt format=tflite imgsz=$SZ device=cpu > $d/export_fp32.log 2>&1 ); cp $d/$R.tflite $d/${R}_fp32.tflite 2>/dev/null
    echo "EXPORT fp32 $SZ bytes=$(stat -c%s $d/${R}_fp32.tflite 2>/dev/null) success=$(grep -c "export success" $d/export_fp32.log) $(date +%T)"
  fi
  if [ ! -s $d/${R}_int8.tflite ] || [ "$(stat -c%s $d/${R}_int8.tflite)" -gt "$(( $(stat -c%s $d/${R}_fp32.tflite) / 2 ))" ]; then
    rm -f $d/${R}_int8.tflite
    ( cd $d && OMP_NUM_THREADS=$T yolo export model=$d/$R.pt format=tflite imgsz=$SZ int8=True data=$DATA split=train device=cpu > $d/export_int8.log 2>&1 )
    echo "EXPORT int8 $SZ bytes=$(stat -c%s $d/${R}_int8.tflite 2>/dev/null) ratio=$(python3 -c "import os;print(round(os.path.getsize('$d/${R}_int8.tflite')/os.path.getsize('$d/${R}_fp32.tflite'),3))" 2>/dev/null) success=$(grep -c "export success" $d/export_int8.log) $(date +%T)"
  fi
  if [ ! -s $d/${R}_fdec.tflite ]; then
    OMP_NUM_THREADS=$T TF_NUM_INTRAOP_THREADS=$T python3 /workspace/float_head_quant.py --src $d/${R}_fp32.tflite --out $d/${R}_fdec.tflite --calib $CAL --imgsz $SZ --head-regex "$RX" > $d/export_fdec.log 2>&1
    echo "EXPORT fdec $SZ $(grep -a "^wrote" $d/export_fdec.log | cut -c1-140) $(date +%T)"
  fi
done
bench $E/sz640/${R}_fp32.tflite; bench $E/sz640/${R}_int8.tflite; bench $E/sz640/${R}_fdec.tflite; echo BENCH_DONE $(date +%T)

# ---- 3. evals: .pt on GPU first (fast), then CPU cells, 640 first ----
cell() {  # tag file imgsz device threads
  local tag=$1; local f=$2; local SZ=$3; local dev=$4; local th=$5
  local out=$O/${R}_${tag}_sz${SZ}_spotval
  [ -f "${out}_map.json" ] && { echo "SKIP_DONE $tag $SZ"; return 0; }
  [ -s "$f" ] || { echo "MISSING $f"; return 1; }
  echo "EVAL_START $tag $SZ $(date +%T)"
  if [ "$(find $out/raw -name '*.json' 2>/dev/null | wc -l)" != "4232" ]; then
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=$th MKL_NUM_THREADS=$th python3 $VLM/run_ultralytics_labels.py --engine ultralytics --model "$f" --imgsz $SZ --device $dev --images "$SV_IMG" --out "$out" --floor 0.001 > "$out.log" 2>&1
    echo "INFER_DONE $tag $SZ rc=$? n=$(find $out/raw -name '*.json' | wc -l) $(date +%T)"
  fi
  python3 $VLM/map_eval.py --raw "$out/raw" --gt-labels $SV_GT --ignore-labels $SV_IGN --expect-floor 0.001 --out "${out}_map.json" > "${out}_map.log" 2>&1
  python3 -c "import json; m=json.load(open('${out}_map.json')); pc=m['per_class']; print('RESULT $R $tag $SZ', m['map50_95'], m['map50'], m['map75'], {c:(v['ap50_95'],v['ap50']) for c,v in pc.items()})"
}
for SZ in 640 416 320; do cell pt $PT $SZ cuda:0 2; done
for SZ in 640 416 320; do for tag in fp32tflite int8 fdec; do f=$E/sz$SZ/${R}_$( [ $tag = fp32tflite ] && echo fp32 || echo $tag ).tflite; cell $tag $f $SZ cpu $T; done; done
echo MATRIX_DONE $(date +%T)
