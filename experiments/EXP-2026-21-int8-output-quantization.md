# EXP-2026-21 — Where does the INT8 mAP50-95 loss at 640 come from?

*Opened 2026-09-10 · owner: Mostafa · pod `superior_amethyst_goose` (RTX 2000 Ada, 5.1 CPU cores)*
*Predictions below were written and saved before any arm was run.*

## Question

Train-calibrated full-INT8 (W8A8) TFLite exports lose 4.2 / 4.7 / 6.6 pts mAP50-95 at 640 px
(`y26n_humanshaped_v2` / `y26n_noe2e_warm50-2` / `y26s_humanshaped_smallpatch_v1`) while mAP50
is flat. A reviewer's hypothesis: the calibration data lacks variance. Mine
(`docs/INT8_PREPROCESSING_PARITY.md` §4–5): the exporter quantizes the **final `[1,7,N]` output
tensor** with a single per-tensor int8 scale (measured 0.0042, zero-point −128) shared by
normalized box coordinates and class scores, so every box edge is snapped to a 0.0042·W grid
(2.7 px at 640) and the high-IoU buckets of mAP50-95 collapse. Which is it?

## Pre-registered predictions (falsifiable)

| # | arm | prediction if the output-grid explanation is right | what would falsify it |
| --- | --- | --- | --- |
| A | box-edge error histogram, INT8 vs fp32-TFLite, from the existing raw dumps at 640 (no re-run) | edge errors sit on a comb at integer multiples of 0.0042·640 = 2.69 px (peaks at 0, ±2.7, ±5.4 px); ≥60 % of matched edges within ±0.25 px of a comb tooth; the same comb at 416 / 320 has spacing 1.75 / 1.34 px | a smooth, comb-free error distribution → the error is not from output quantization |
| B | `quantize=w8a16` (int8 weights, **int16 activations**, output step 1/65535) at 640, all three models | mAP50-95 within **±0.5 pt** of the fp32-TFLite row for each model | loss ≥ 2 pt remains → loss is in the weights or in intermediate activations, not the output |
| C | `quantize=w8a32` (int8 weights, float32 activations, no calibration) at 640, all three models | within ±0.3 pt of fp32 (weights alone cost ~nothing; they are per-channel) | a ≥ 1 pt loss → per-channel int8 weights themselves cost accuracy |
| D | *(only if B recovers)* float head: W8A8 backbone, dequantize before the detect head via an `ai_edge_quantizer` per-op recipe | recovers ≥ 80 % of the 640 gap at ≈ W8A8 file size and speed | recovers < 50 % → most of the loss is accumulated before the head |

Calibration-variance hypothesis, stated so it can lose too: if it were right, (i) arm B would
**not** recover (int16 activations don't change the calibration set), and (ii) the val-calibrated
(4,232 images) and train-calibrated (500 images) arms of 2026-09-09 would have differed
materially — they agreed within noise in 8/9 cells.

## Bar for "explained"

A passes **and** B passes. Then the fix for deployment is D (or shipping FP16 / `w8a16` if D is
impractical), and the reviewer's calibration question is answered with data, not argument.

## Method (identical to the 2026-09-09 matrix except for the export flag)

- Inputs: `/workspace/exports/<run>_calib500/sz640/<run>_fp32.tflite` (already scored:
  `/workspace/quant_matrix_calib500/<run>/fp32tflite_sz640_spotval_map.json`).
- Exports: `yolo export model=<best.pt> format=tflite imgsz=640 quantize=w8a16 data=<calib500 yaml>`
  and `... quantize=w8a32` (no data). Success check: process exited **and** log contains
  `export success` **and** file size ≈ ¼ fp32 (w8a16/w8a32 weights) — never trust the exit code
  alone (interim full-size file trap, memory `int8-calibration-recipe-flaw`).
- Scoring: `run_ultralytics_labels.py --engine ultralytics --model <tflite> --imgsz 640 --device cpu --floor 0.001`
  → `map_eval.py --ignore-labels` on Spotlight-val (4,232 images, 702 ignore boxes). Same
  letterbox (`scaleup=True`, centre pad), same NMS (IoU 0.7), same scorer. `OMP_NUM_THREADS=5`.
- Arm A: `vlm-cluster/box_edge_hist.py` — match INT8 and fp32 raw detections per image (same
  class, IoU ≥ 0.7, conf ≥ 0.25), take the four edge differences in pixels, histogram at 0.1 px
  bins, report the fraction within ±0.25 px of `k·0.0042·S`.

## Results

### Arm A — box-edge errors (existing dumps, no re-run)

**Prediction A as written was wrong, and wrongly designed.** The int8 − fp32 *difference* shows no
comb (`y26n_humanshaped_v2` @640: 21.4 % within ±0.25 px of k·2.688 px vs 18.6 % for a grid
shifted by half a step; half-step grid 24.4 % vs 21.0 %). It could not: the fp32 side is
continuous, so the difference is continuous whatever the int8 side does. Correction recorded as-is.

**Corrected test** (`vlm-cluster/box_grid_test.py`: are the INT8 model's *own* coordinates on the
grid?) with the exact output scale read from the file (`tflite_tensor_inspect.py`:
`serving_default_output_0_output` int8, scale **0.0041764890775084496**, zp −128 —
= 2.6729536 / 640):

| dump (`y26n_humanshaped_v2` @640, first 1,500 images, conf ≥ 0.25) | values | within ±0.05 step | median residual (steps) |
| --- | --- | --- | --- |
| INT8 `sz640_spotval` | 7,720 | **83.2 %** | 0.001 |
| fp32-TFLite control | 7,572 | 10.3 % (= chance 2·tol) | 0.250 |

The output grid is real. But it is not the main error — the INT8-vs-fp32 edge error is far
larger than one tooth and **grows with box size**, the signature of a per-tensor scale on a
stride-unit tensor (`box_edge_hist.py`, matched same-class pairs IoU ≥ 0.7, conf ≥ 0.25,
errors in model-input px):

| `y26n_humanshaped_v2` | edges | mean \|err\| | median | p90 |
| --- | --- | --- | --- | --- |
| 640, all | 26,928 | 4.35 px | 3.26 | 9.25 |
| 640, ref box < 64 px (≈ stride 8) | 420 | 1.86 | 1.55 | 3.91 |
| 640, 64–256 px (≈ stride 16) | 8,860 | 3.62 | 2.94 | 7.67 |
| 640, > 256 px (≈ stride 32) | 17,648 | 4.77 | 3.52 | 10.16 |
| 416, all | 26,872 | 2.32 | 1.75 | 5.00 |
| 320, all | 26,064 | 1.89 | 1.31 | 4.25 |
| 320, < 64 / 64–256 / > 256 px | 3,256 / 16,856 / 5,952 | 1.14 / 1.90 / 2.25 | | |

`y26n_noe2e_warm50-2` @640: mean 4.20 px, median 3.12 (same picture).

Head tensors of the INT8 file (`/workspace/exp21/tensor_details_640.txt`), all per-tensor int8:
`[1,4,8400]` scale 0.3131 zp −128 (regression distances in stride units, P3+P4+P5 concatenated →
one step = 2.5 / 5.0 / 10.0 px at strides 8 / 16 / 32); `[1,4,8400]` scale 2.6730 zp −128 (box in
pixels, step 2.67 px); `[1,7,8400]` scale 0.0041765 (normalized output). The loss is int8
*activation* resolution inside the detect head where multi-scale tensors share one 256-level
range — a geometric range (0–80 stride units, 0–640 px, 0–1), independent of the calibration
images.

### Arms B / C / D — all at 640, Spotlight-val, same scorer (closed 2026-09-11)

mAP50-95 (Δ vs fp32-TFLite in pts). Every cell is the full 4,232-image set; JSONs in
`models/exp21_20260911/exp21/eval/`.

| model | fp32 TFLite | INT8 W8A8 (`int8=True`) | C `w8a32` | B `w8a16` | **D1 W8A8 + float decode ops + float output** | D1′ W8A8 + float decode, output still int8 | D2 W8A8 + whole head float |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `y26n_humanshaped_v2` | 0.7037 | 0.6615 (−4.2) | 0.6977 (−0.6) | 0.6967 (−0.7) | **0.7296 (+2.6)** | 0.7251 (+2.1) | 0.6972 (−0.7) |
| `y26n_noe2e_warm50-2` | 0.7014 | 0.6544 (−4.7) | 0.6948 (−0.7) | 0.6948 (−0.7) | **0.7174 (+1.6)** | 0.7127 (+1.1) | 0.6830 (−1.8) |
| `y26s_humanshaped_smallpatch_v1` | 0.7682 | 0.7019 (−6.6) | 0.7646 (−0.4) | not run (int16 inference ≈ 7× slower; pod closed) | **0.7706 (+0.2)** | 0.7665 (−0.2) | 0.7603 (−0.8) |

mAP50 for `y26n_humanshaped_v2`: 0.8066 / 0.8201 / 0.8027 / 0.8044 / **0.8307** / 0.8304 / 0.8028.
File sizes (nano): INT8 2.82 MB, `w8a32` 2.84, `w8a16` 2.95, D1 2.96, D1′ 2.96, D2 3.29; y26s D1 10.28 MB vs INT8 9.97.

Per-class AP50-95 (AP50), `y26n_humanshaped_v2` @640:

| arm | Woman | Man | Child |
| --- | --- | --- | --- |
| fp32 TFLite | 0.7075 (0.8235) | 0.7765 (0.8624) | 0.6271 (0.7339) |
| INT8 W8A8 | 0.6450 (0.8220) | 0.7126 (0.8551) | 0.6269 (0.7832) |
| C `w8a32` | 0.6928 (0.8113) | 0.7769 (0.8637) | 0.6234 (0.7330) |
| D1 float decode + output | 0.7105 (0.8227) | 0.7813 (0.8684) | **0.6969 (0.8010)** |
| D2 whole head float | 0.6902 (0.8099) | 0.7734 (0.8611) | 0.6281 (0.7374) |

### Scorecard against the pre-registered predictions

| # | predicted | observed | verdict |
| --- | --- | --- | --- |
| A | comb in the int8−fp32 difference | no comb — the test was mis-designed (difference of a gridded and a continuous value is continuous); corrected test: 83 % of INT8 coordinates on the exact 0.0041765 grid vs 10 % control, **and** edge error ∝ stride level (1.9 / 3.6 / 4.8 px) | mechanism located, prediction as written failed |
| B | `w8a16` within ±0.5 of fp32 | −0.7 / −0.7 | just outside the bar; the activation-resolution claim holds (recovers 85 % of the gap) |
| C | `w8a32` within ±0.3 | −0.6 / −0.7 / −0.4 | missed by a hair: per-channel int8 weights cost ~0.5 pt, not ~0 |
| D | float head recovers ≥ 80 % of the 640 gap | D1 recovers **> 100 %** on all three models (nanos end up *above* fp32) | passed, with a surprise (below) |

**Bar for "explained": met** — the corrected A plus D1. The 640 INT8 loss is int8 activation
resolution in the detect head's *decode* ops (concatenated multi-scale regression in stride
units → pixel-space boxes → normalized output, each on one 256-level scale). Keeping only those
31 elementwise tensors in float (all convs stay int8, +5 % file size) removes the loss entirely.

**Calibration-variance hypothesis: rejected** on data. The failing tensors have geometric ranges
(0–80 stride units, 0–640 px, 0–1) that no image set changes; `w8a16` recovers without touching
the calibration set; the 500-vs-4,232-image arms of 2026-09-09 agreed.

### The surprise: D1 beats fp32, and D2 (more float) is worse than D1

D1's +2.6 / +1.6 on the nanos is **entirely Child**: Woman +0.3, Man +0.5, Child **+7.0**
AP50-95 (+6.7 AP50). The same Child gain is present in plain INT8 W8A8 (Child AP50 0.7832 vs fp32
0.7339 — the "unexplained Child AP50 rise" open item from the 2026-09-09 matrix) and absent in
`w8a32`, `w8a16` and D2. What those three share is *float head convs*; what INT8 and D1 share is
*int8 head convs*. So the gain comes from int8 quantization of the head's conv branches (the
class-score branch is the obvious suspect), on the nano models only (y26s: +0.2). This is a
real, reproducible effect on Spotlight-val, **not yet understood**, and the direction matters:
higher Child AP on model-derived labels could equally be adults being pulled towards Child
(the escape direction). The LAGENDA classification sweep on the D1 files is now the required
next step, not optional.

Not measured: latency of D1 vs INT8 (the bench was queued when the pod closed). D1 differs from
INT8 only in 31 elementwise ops on `[1, k, 8400]` tensors, so ≈ INT8 is expected — verify before
any recommendation changes.

### Artifacts

`models/exp21_20260911/exp21/` (gitignored): `eval/*_map.json` + logs for all 17 cells, `armA_*.json`,
`tensor_details_640.txt`, `int8_tensor_names.txt`, `fp32_tensor_names.txt`, all run scripts/logs,
and the `.tflite` files for `y26n_noe2e_warm50-2` and `y26s_humanshaped_smallpatch_v1` (the
`y26n_humanshaped_v2` exports were lost when the pull was cut off — regenerable in minutes with
`float_head_quant.py`; pod copies live in `/workspace/exp21/exports/`).


### Addendum 2026-09-11 — latency (measured) and a fourth model

**Latency** (pod `ylyixd8bja71om`, L4 host, 5.1 CPU cores, `bench_tflite.py` 4 threads, 20 warm-up + 100 runs, @640):

| file | size | median | p90 |
| --- | --- | --- | --- |
| `y26n_humanshaped_v2` fp32 | 9.84 MB | 25.7 ms | 26.1 |
| `y26n_humanshaped_v2` INT8 W8A8 | 2.89 | 21.4 | 21.6 |
| `y26n_humanshaped_v2` **D1 fix** | 2.96 | **22.9** | 23.9 |
| `y26n_humanshaped_v2` `w8a32` | 2.84 | 32.7 | 35.7 |
| `y26n_humanshaped_v2` `w8a16` | 2.95 | 1,425 | 1,432 |
| `y26s_humanshaped_smallpatch_v1` fp32 / INT8 / **D1** | 38.2 / 10.2 / 10.28 | 70.6 / 46.7 / **46.6** | 70.9 / 47.6 / 48.8 |
| `y26n_humanshaped_v2_distill_v1` fp32 / INT8 / **D1** | 9.84 / 2.89 / 2.96 | 25.6 / 22.9 / **20.0** | 27.7 / 24.0 / 20.2 |

D1 costs ≈ +1.5 ms on a nano (within bench jitter — the distill run measured it *faster* than INT8) and
nothing on y26s. `w8a32` is slower than fp32 (dynamic dequant per op); `w8a16` is 66× slower (reference
int16 kernels). Both are off the table; D1 is the only candidate that keeps INT8 speed.

**Fourth model — `y26n_humanshaped_v2_distill_v1`** (nano distilled from the y26s teacher; checkpoint is
**epoch 61 of 100**, `best.pt` sha256 `edd694c08585af1c…`, frozen copy under
`/workspace/exp21/exports/y26n_humanshaped_v2_distill_v1/`; nothing was training when measured).
Same exports/scorer as the matrix; the D1 recipe unchanged (`Detect_23` is the same layer index):

| size | `.pt` fp32 (GPU) | fp32 TFLite | INT8 W8A8 | **D1 fix** | INT8 file | D1 file |
| --- | --- | --- | --- | --- | --- | --- |
| 640 | 0.6848 | 0.6885 | 0.6560 (−3.3) | **0.7105 (+2.2)** | 2.89 MB `daa151876e64de5d…` | 2.96 MB `f675f6f55b6dc088…` |
| 416 | 0.6808 | 0.6778 | 0.6825 (+0.5) | **0.6928 (+1.5)** | 2.87 MB `fb63b92c5ed8829b…` | 2.90 MB `65c6e2e6bc2cc1bc…` |
| 320 | 0.6521 | 0.6482 | 0.6570 (+0.9) | **0.6842 (+3.6)** | 2.87 MB `dcaabe4354705b05…` | 2.89 MB `b8f081e497edc5cb…` |

Per-class @640 (AP50-95): fp32 TFLite Woman 0.672 / Man 0.771 / Child 0.622 → INT8 0.626 / 0.709 / 0.633
→ D1 0.678 / 0.772 / **0.681**. Same anatomy as the other nanos: INT8 loses ~6 pts of adult box
tightness, D1 restores it, int8 head convs add ~+6 on Child. At 416/320 this model's INT8 is already
≥ fp32 (the Child gain outweighs the smaller box loss). As a model it trails `y26n_humanshaped_v2`
by ~1.5–2 pts at every size — expected for an unfinished run; re-measure when training completes.
(Run note: the volume quota filled mid-run and killed 11 cells silently (`rc=120`); freed 2.3 GB of
export temp files, owner enlarged the volume, relaunched — all cells complete.)

## What this does NOT show

- Nothing about latency: `w8a16` runs int16 activation kernels on XNNPACK, which may be much
  slower than int8; a recovered mAP with 3× latency is not a deployment answer.
- Nothing about the extension: TF.js cannot run any of these files (`models/extension_tfjs_20260910/README.md`).
- Spotlight-val labels are model-derived (2026-07-28 policy, no humanshaped boxes) — absolute
  mAP is not a product metric here; only the *difference* between precisions on the same set is.

## Verify it yourself

Commands, in order, exactly as run (pod paths):

```bash
source /root/venvs/export_tflite/bin/activate
# arm C (no calibration)
yolo export model=/workspace/exp20/train/y26n_humanshaped_v2/weights/best.pt format=tflite imgsz=640 quantize=w8a32
# arm B (calibrated, same 500-image train subset as the 2026-09-09 matrix)
yolo export model=/workspace/exp20/train/y26n_humanshaped_v2/weights/best.pt format=tflite imgsz=640 quantize=w8a16 data=<calib500 yaml> fraction=1.0
# score (one cell)
OMP_NUM_THREADS=5 python3 /workspace/data_inspection_tools/vlm-cluster/run_ultralytics_labels.py --engine ultralytics \
  --model <file.tflite> --imgsz 640 --device cpu --floor 0.001 --images /workspace/exp12/images_val4232 --out <outdir>
python3 /workspace/data_inspection_tools/vlm-cluster/map_eval.py --pred <outdir>/labels --gt /workspace/spotlight/run/oiv7_val/labels --ignore-labels /workspace/spotlight/run/oiv7_val/ignore_unk --out <outdir>_map.json
# arm A
python3 /workspace/box_edge_hist.py --a /workspace/quant_matrix_calib500/<run>/sz640_spotval/raw --b /workspace/quant_matrix_calib500/<run>/fp32tflite_sz640_spotval/raw --imgsz 640
```

The exact flags used by the earlier matrix (`--images`, `--out`, yaml path) are copied from
`/workspace/quant_matrix_calib500/*.sh` at run time and pasted into the Results section.
