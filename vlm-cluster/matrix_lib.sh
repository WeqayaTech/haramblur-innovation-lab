#!/bin/bash
# matrix_lib.sh — config + functions for the EXP-2026-22 matrix (sourced by full_matrix.sh and run_cells.sh).
# THREADS env overrides the per-cell thread count (used when several cells share a pod).
#   models   : 4 candidates + production yolo11N-640
#   precision: pt (fp32 .pt on GPU) | fp32 (TFLite) | fp16 (weight-cast) | int8 (W8A8, calib500) | fdec (W8A8 + float decode ops)
#   sizes    : 640 416 320
#   datasets : spotval (Spotlight-val 4,232) | holdout (haramblur_holdout 11,494; scored pooled + per collection + randoms FP arm)
# Two queues run concurrently: `gpu` (all pt cells) and `cpu` (exports, then TFLite cells in priority order).
# Every cell is skipped when its map json exists, so the script can be re-run / resumed on any pod.
#   usage: bash full_matrix.sh gpu|cpu      (logs: /workspace/exp22/<queue>.log)
set -u
export TMPDIR=/workspace/tmp; mkdir -p /workspace/tmp
source /root/venvs/export_tflite/bin/activate
Q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || echo -1); PERIOD=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null || echo 100000)
T=$(nproc); [ "$Q" -gt 0 ] && T=$((Q / PERIOD)); [ "$T" -lt 1 ] && T=1
VLM=/workspace/data_inspection_tools/vlm-cluster
DATA=/workspace/exp12/spotlight_oiv7_calib500.yaml; CAL=/workspace/exp12/calib_train500.txt
SV_IMG=/workspace/exp12/images_val4232; SV_GT=/workspace/spotlight/run/oiv7_val/labels; SV_IGN=/workspace/spotlight/run/oiv7_val/ignore_unk
H=/workspace/datasets/haramblur_holdout/labeling/full
E=/workspace/exp22/exports; O=/workspace/exp22/eval; mkdir -p $E $O
RX='.*Detect_[0-9]+;.*|.*_NormalizeCoords;.*|.*serving_default_output_0.*'
declare -A PT=( [y26n_humanshaped_v2]=/workspace/exports/y26n_humanshaped_v2_calib500/sz640/y26n_humanshaped_v2.pt
                [y26n_noe2e_warm50-2]=/workspace/exports/y26n_noe2e_warm50-2_calib500/sz640/y26n_noe2e_warm50-2.pt
                [y26s_humanshaped_smallpatch_v1]=/workspace/exports/y26s_humanshaped_smallpatch_v1_calib500/sz640/y26s_humanshaped_smallpatch_v1.pt
                [y26n_humanshaped_v2_distill_v1]=/workspace/exp21/exports/y26n_humanshaped_v2_distill_v1/sz640/y26n_humanshaped_v2_distill_v1.pt
                [yolo11N-640]=/workspace/model_v2/dataset_v2/yolo11N-640/weights/best.pt )
MODELS="yolo11N-640 y26n_humanshaped_v2 y26n_noe2e_warm50-2 y26s_humanshaped_smallpatch_v1 y26n_humanshaped_v2_distill_v1"
T=${THREADS:-$T}


# ---------- locate an existing export, or say where a new one goes ----------
resolve() {  # run tag sz -> path (existing preferred)
  local run=$1; local tag=$2; local sz=$3; local c
  case $tag in fp32) c="/workspace/exports/${run}_calib500/sz$sz/${run}_fp32.tflite /workspace/exp21/exports/$run/sz$sz/${run}_fp32.tflite";;
               fp16) c="/workspace/exports/${run}_calib500/sz$sz/${run}_fp16.tflite";;
               int8) c="/workspace/exports/${run}_calib500/sz$sz/${run}_int8.tflite /workspace/exp21/exports/$run/sz$sz/${run}_int8.tflite";;
               fdec) c="/workspace/exp21/exports/$run/sz$sz/${run}_fdec.tflite"; [ "$sz" = 640 ] && c="$c /workspace/exp21/exports/$run/${run}_fhead_decode_out.tflite";; esac
  for p in $c; do [ -s "$p" ] && { echo "$p"; return; }; done
  echo "$E/$run/sz$sz/${run}_${tag}.tflite"
}
export_all() {  # run sz
  local run=$1; local sz=$2; local d=$E/$run/sz$sz; mkdir -p $d; cp -n "${PT[$run]}" $d/$run.pt
  export CUDA_VISIBLE_DEVICES=""
  local f32=$(resolve $run fp32 $sz)
  if [ ! -s "$f32" ]; then
    ( cd $d && OMP_NUM_THREADS=$T yolo export model=$d/$run.pt format=tflite imgsz=$sz device=cpu > $d/export_fp32.log 2>&1 ); cp $d/$run.tflite $d/${run}_fp32.tflite 2>/dev/null; f32=$d/${run}_fp32.tflite
    echo "EXPORT $run fp32 $sz bytes=$(stat -c%s $f32 2>/dev/null) success=$(grep -c 'export success' $d/export_fp32.log) $(date +%T)"
  fi
  local f16=$(resolve $run fp16 $sz)
  if [ ! -s "$f16" ]; then
    python3 /workspace/fp16_cast.py "$f32" "$d/${run}_fp16.tflite" > $d/export_fp16.log 2>&1
    echo "EXPORT $run fp16 $sz $(tail -1 $d/export_fp16.log | cut -c1-120) $(date +%T)"
  fi
  local i8=$(resolve $run int8 $sz)
  if [ ! -s "$i8" ] || [ "$(stat -c%s "$i8")" -gt "$(( $(stat -c%s "$f32") / 2 ))" ]; then
    rm -f $d/${run}_int8.tflite
    ( cd $d && OMP_NUM_THREADS=$T yolo export model=$d/$run.pt format=tflite imgsz=$sz int8=True data=$DATA split=train device=cpu > $d/export_int8.log 2>&1 )
    echo "EXPORT $run int8 $sz bytes=$(stat -c%s $d/${run}_int8.tflite 2>/dev/null) success=$(grep -c 'export success' $d/export_int8.log) $(date +%T)"
  fi
  local fd=$(resolve $run fdec $sz)
  if [ ! -s "$fd" ]; then
    OMP_NUM_THREADS=$T TF_NUM_INTRAOP_THREADS=$T python3 /workspace/float_head_quant.py --src "$f32" --out $d/${run}_fdec.tflite --calib $CAL --imgsz $sz --head-regex "$RX" > $d/export_fdec.log 2>&1
    echo "EXPORT $run fdec $sz $(grep -a '^wrote' $d/export_fdec.log | cut -c1-120) $(date +%T)"
  fi
}

# ---------- one cell ----------
cell() {  # ds run tag sz
  local ds=$1; local run=$2; local tag=$3; local sz=$4
  local out=$O/${run}_${tag}_sz${sz}_${ds}; local f dev th
  if [ "$tag" = pt ]; then f=${PT[$run]}; if nvidia-smi > /dev/null 2>&1; then dev=cuda:0; th=2; else dev=cpu; th=$T; fi; else f=$(resolve $run $tag $sz); dev=cpu; th=$T; fi
  if [ -f "${out}_map.json" ]; then python3 -c "import json,sys; json.load(open(sys.argv[1]))" "${out}_map.json" 2>/dev/null && { echo "SKIP_DONE $ds $run $tag $sz"; return 0; } || { echo "CORRUPT_JSON $ds $run $tag $sz -> rescoring"; rm -f "${out}_map"*.json; }; fi
  # wait for an export another pod may still be writing: file present AND (int8) smaller than half its fp32 sibling
  local w=0; while true; do
    if [ -s "$f" ]; then
      if [ "$tag" = int8 ]; then local f32=$(resolve $run fp32 $sz); [ -s "$f32" ] && [ "$(stat -c%s "$f")" -lt "$(( $(stat -c%s "$f32") / 2 ))" ] && break
      else break; fi
    fi
    sleep 30; w=$((w+1)); [ $w -gt 90 ] && { echo "MISSING $ds $run $tag $sz $f"; return 1; }
  done
  local img=$SV_IMG; [ "$ds" = holdout ] && img=$H/images
  local n=4232; [ "$ds" = holdout ] && n=11494
  # reuse the spotval cells already scored under exp21 / quant_matrix_calib500 (map json only)
  if [ "$ds" = spotval ]; then
    local QM=/workspace/quant_matrix_calib500/$run; local X=/workspace/exp21/eval; local olds=""
    case $tag in pt)   olds="$X/${run}_pt_sz${sz}_spotval $QM/fp32_sz${sz}_spotval";;
                 fp32) olds="$X/${run}_fp32tflite_sz${sz}_spotval $QM/fp32tflite_sz${sz}_spotval";;
                 fp16) olds="$QM/fp16tflite_sz${sz}_spotval";;
                 int8) olds="$X/${run}_int8_sz${sz}_spotval $QM/sz${sz}_spotval";;
                 fdec) olds="$X/${run}_fdec_sz${sz}_spotval $X/${run}_fhead_decode_out_sz${sz}_spotval";; esac
    for old in $olds; do [ -f "${old}_map.json" ] && { cp "${old}_map.json" "${out}_map.json"; echo "REUSED $ds $run $tag $sz <- $old"; break; }; done
    [ -f "${out}_map.json" ] && { report_line $ds $run $tag $sz "$out"; return 0; }
  fi
  echo "EVAL_START $ds $run $tag $sz $(basename $f) $(date +%T)"
  # validate every existing sidecar (truncated files from an I/O event look present but do not parse) before trusting the count
  [ -d "$out/raw" ] && python3 - "$out/raw" <<'PYV'
import json, os, sys
d = sys.argv[1]; bad = 0
for f in os.listdir(d):
    p = os.path.join(d, f)
    try:
        if os.path.getsize(p) == 0: raise ValueError
        json.load(open(p))
    except Exception:
        os.remove(p); bad += 1
        lab = os.path.join(os.path.dirname(d), "labels", f[:-5] + ".txt")
        if os.path.exists(lab): os.remove(lab)
if bad: print(f"purged {bad} corrupt sidecars in {d}")
PYV
  if [ "$(find $out/raw -name '*.json' -size +0 2>/dev/null | wc -l)" != "$n" ]; then
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=$th MKL_NUM_THREADS=$th python3 $VLM/run_ultralytics_labels.py --engine ultralytics --model "$f" --imgsz $sz --device $dev --images "$img" --out "$out" --floor 0.001 > "$out.log" 2>&1
    echo "INFER_DONE $ds $run $tag $sz rc=$? n=$(find $out/raw -name '*.json' | wc -l) $(date +%T)"
  fi
  if [ "$ds" = spotval ]; then
    python3 $VLM/map_eval.py --raw "$out/raw" --gt-labels $SV_GT --ignore-labels $SV_IGN --expect-floor 0.001 --out "${out}_map.json" > "${out}_map.log" 2>&1
  else
    python3 $VLM/map_eval.py --raw "$out/raw" --gt-labels $H/labels_eval --ignore-labels $H/ignore --expect-floor 0.001 --out "${out}_map.json" > "${out}_map.log" 2>&1
    for c in child men shiekhs women women_hd; do   # the five collection scorers run concurrently (each loads the dump once)
      python3 $VLM/map_eval.py --raw "$out/raw" --gt-labels $H/labels_eval --ignore-labels $H/ignore --expect-floor 0.001 --stem-prefix "${c}__" --out "${out}_map_${c}.json" > "${out}_map_${c}.log" 2>&1 &
    done; wait
    python3 /workspace/randoms_fp.py "$out/raw" "${out}_randoms.json"
  fi
  rm -rf "$out/labels"   # derived from raw/, never read by the scorer — saves ~40% of the volume per cell
  # a partial dump (inference OOM-killed / crashed) must never be accepted as a result
  local got=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('n_images'))" "${out}_map.json" 2>/dev/null)
  [ "$got" = "$n" ] || { echo "CELL_PARTIAL $ds $run $tag $sz n_images=$got expected=$n -> dropped"; rm -f "${out}_map"*.json "${out}_randoms.json"; return 1; }
  report_line $ds $run $tag $sz "$out"
}
report_line() {
  local ds=$1; local run=$2; local tag=$3; local sz=$4; local out=$5
  python3 - "$ds" "$run" "$tag" "$sz" "$out" <<'PY'
import json, sys, os
ds, run, tag, sz, out = sys.argv[1:]
m = json.load(open(out + "_map.json")); pc = m.get("per_class", {})
line = f"RESULT {ds} {run} {tag} {sz} map50={m['map50']} map50_95={m['map50_95']} " + " ".join(f"{c}={v['ap50_95']}/{v['ap50']}" for c, v in pc.items())
if ds == "holdout":
    for c in ("child", "men", "shiekhs", "women", "women_hd"):
        p = f"{out}_map_{c}.json"
        if os.path.exists(p):
            try: mc = json.load(open(p)); line += f" | {c}: {mc['map50_95']}/{mc['map50']} n_gt={mc.get('n_gt')}"
            except Exception as e: line += f" | {c}: BAD_JSON"
    p = out + "_randoms.json"
    if os.path.exists(p): r = json.load(open(p)); line += f" | randoms: {r}"
print(line)
PY
}

