# Deployment — models and paths (2026-09-09)

All paths are on the RunPod network volume (`/workspace`). Attach the volume to any pod in its
datacenter; copy with `scp` over the pod's direct-TCP SSH port (the `ssh.runpod.io` proxy has no
scp). Verify every file you receive against the size and sha256 prefix below — an INT8 file that
is the same size as its fp32 sibling is **not quantized** (a known exporter failure mode).

## The three candidates

| model | what it is | source checkpoint (`.pt`, fp16-saved) | size | sha256 |
|---|---|---|---|---|
| `y26n_humanshaped_v2` | YOLO26n, current humanshaped labeling policy | `/workspace/exp20/train/y26n_humanshaped_v2/weights/best.pt` | 5.07 MB | `37825031b616abe6…` |
| `y26n_noe2e_warm50-2` | YOLO26n, previous holdout-mAP champion | `/workspace/exp18/train/y26n_noe2e_warm50-2/weights/best.pt` | 5.07 MB | `ec7bbfeb4642aff8…` |
| `y26s_humanshaped_smallpatch_v1` | YOLO26s (~3x the params), best accuracy measured | `/workspace/exp20/train/y26s_humanshaped_smallpatch_v1/weights/best.pt` | 20.31 MB | `b6bb781cd55ac263…` |

Classes in every artifact: `0 = Woman, 1 = Man, 2 = Child`.

**Recommended operating point: INT8 at 416 px** (see `docs/MODEL_COMPARISON.md`, section
"INT8 quantization accuracy matrix — train-calibrated, corrected"). INT8 costs no detection or
classification accuracy (mAP50 flat or up for the nanos); it costs box tightness, least at 416/320.
**If any accuracy loss is unacceptable, ship FP16** (below): identical to fp32, half the size.

> **2026-09-11 update — read before choosing INT8.** EXP-2026-21 found the cause of the 640 INT8
> loss (int8 resolution in the head's decode ops) and a fix that keeps every conv int8 but leaves
> 31 decode tensors float: `vlm-cluster/float_head_quant.py`. At 640 it scores 0.7296 / 0.7174 /
> 0.7706 mAP50-95 vs INT8 0.6615 / 0.6544 / 0.7019 (fp32 0.7037 / 0.7014 / 0.7682), +5 % file size.
> Latency measured 2026-09-11: it keeps INT8 speed (nano 22.9 vs 21.4 ms, y26s 46.6 vs 46.7). Its nano-only Child AP gain is **not yet explained**, so the
> recommendation below stands until both are done (`CLAUDE.md` open items). Do not ship plain
> `int8=True` at 640.

## TFLite / LiteRT exports — use these (train-calibrated, 2026-09-09)

Directory pattern: `/workspace/exports/<model>_calib500/sz<SZ>/<model>_{int8,fp32}.tflite`

| model | px | INT8 file | size | sha256 | fp32 file | size | sha256 |
|---|---|---|---|---|---|---|---|
| `y26n_humanshaped_v2` | 640 | `…_calib500/sz640/y26n_humanshaped_v2_int8.tflite` | 2.89 MB | `d2705be2b9cbc813…` | `…/sz640/y26n_humanshaped_v2_fp32.tflite` | 9.84 MB | `02bc0ce310750124…` |
| | 416 | `…_calib500/sz416/y26n_humanshaped_v2_int8.tflite` | 2.87 MB | `2a1456fce02dd916…` | `…/sz416/y26n_humanshaped_v2_fp32.tflite` | 9.78 MB | `83cd3cff76a68866…` |
| | 320 | `…_calib500/sz320/y26n_humanshaped_v2_int8.tflite` | 2.87 MB | `eb02ced0741d37f1…` | `…/sz320/y26n_humanshaped_v2_fp32.tflite` | 9.76 MB | `41d46fc904bddb1d…` |
| `y26n_noe2e_warm50-2` | 640 | `…_calib500/sz640/y26n_noe2e_warm50-2_int8.tflite` | 2.89 MB | `f3c54498879cf79b…` | `…/sz640/y26n_noe2e_warm50-2_fp32.tflite` | 9.84 MB | `da00892e286d053c…` |
| | 416 | `…_calib500/sz416/y26n_noe2e_warm50-2_int8.tflite` | 2.87 MB | `a13094d7adf97fba…` | `…/sz416/y26n_noe2e_warm50-2_fp32.tflite` | 9.78 MB | `35e01a3a890bd7f2…` |
| | 320 | `…_calib500/sz320/y26n_noe2e_warm50-2_int8.tflite` | 2.87 MB | `20e28eacf5211cd6…` | `…/sz320/y26n_noe2e_warm50-2_fp32.tflite` | 9.76 MB | `51e4f4094a94474f…` |
| `y26s_humanshaped_smallpatch_v1` | 640 | `…_calib500/sz640/y26s_humanshaped_smallpatch_v1_int8.tflite` | 10.20 MB | `2a8af38b47273da8…` | `…/sz640/y26s_humanshaped_smallpatch_v1_fp32.tflite` | 38.20 MB | `6a164fe846997e64…` |
| | 416 | `…_calib500/sz416/y26s_humanshaped_smallpatch_v1_int8.tflite` | 10.19 MB | `28a91ffd5d740816…` | `…/sz416/y26s_humanshaped_smallpatch_v1_fp32.tflite` | 38.14 MB | `b26a01dbfa106fc0…` |
| | 320 | `…_calib500/sz320/y26s_humanshaped_smallpatch_v1_int8.tflite` | 10.19 MB | `2f013bc0c09eb51a…` | `…/sz320/y26s_humanshaped_smallpatch_v1_fp32.tflite` | 38.12 MB | `8ba913e0539d8d5b…` |

`…` = `/workspace/exports/<model>`. Full export logs sit next to each file (`export_attempt1.log`,
`status.log`). INT8 calibration: 500 train-split images (`/workspace/exp12/calib_train500.txt`,
sha256 `55d0a78e…bff9`), Ultralytics 8.4.146 / ai-edge-litert 2.2.0.

## FP16 TFLite — half the size, identical accuracy (2026-09-10)

fp32 weights cast to float16 (no calibration). Measured **identical to fp32 within ±0.0004 mAP** in
all nine cells — the zero-risk way to halve the download, and the safe choice at 640 px where INT8
costs 4–7 pts mAP50-95. Not faster than fp32 on CPU (weights are dequantized at load); may be
faster on GPU/WebGPU delegates (unmeasured). Same I/O contract as the other TFLite files.

| model | px | file | size | sha256 |
|---|---|---|---|---|
| `y26n_humanshaped_v2` | 640 | `…_calib500/sz640/y26n_humanshaped_v2_fp16.tflite` | 5.13 MB | `06500a29039ed62a…` |
| | 416 | `…_calib500/sz416/y26n_humanshaped_v2_fp16.tflite` | 5.08 MB | `80a7d0ae0fb7d6ee…` |
| | 320 | `…_calib500/sz320/y26n_humanshaped_v2_fp16.tflite` | 5.06 MB | `d94400ac4c3bd721…` |
| `y26n_noe2e_warm50-2` | 640 | `…_calib500/sz640/y26n_noe2e_warm50-2_fp16.tflite` | 5.13 MB | `8dc0f4221e4ff8ed…` |
| | 416 | `…_calib500/sz416/y26n_noe2e_warm50-2_fp16.tflite` | 5.08 MB | `a26ab6dd495c4fb0…` |
| | 320 | `…_calib500/sz320/y26n_noe2e_warm50-2_fp16.tflite` | 5.06 MB | `57ff2e76c9d60964…` |
| `y26s_humanshaped_smallpatch_v1` | 640 | `…_calib500/sz640/y26s_humanshaped_smallpatch_v1_fp16.tflite` | 19.33 MB | `8aee8c4475e97714…` |
| | 416 | `…_calib500/sz416/y26s_humanshaped_smallpatch_v1_fp16.tflite` | 19.28 MB | `fc3b901a89bcfe63…` |
| | 320 | `…_calib500/sz320/y26s_humanshaped_smallpatch_v1_fp16.tflite` | 19.26 MB | `0ba546a1eacc1466…` |

## TF.js exports (for the current TF.js extension runtime)

All three candidates, all three sizes, float + uint8 (2026-09-10; `y26n_humanshaped_v2` set built
2026-09-02 with the same recipe):

| model | pod folder (SZ ∈ 640/416/320) | float | uint8 |
| --- | --- | --- | --- |
| `y26n_humanshaped_v2` | `/workspace/exports/humanshaped_tfjs/y26n_humanshaped_v2_<SZ>_{float,uint8}/` | 10.1–10.2 MB | 2.97–3.00 MB |
| `y26n_noe2e_warm50-2` | `/workspace/exports/tfjs_all/y26n_noe2e_warm50-2_<SZ>_{float,uint8}/` | 10.1–10.2 MB | 2.97–3.00 MB |
| `y26s_humanshaped_smallpatch_v1` | `/workspace/exports/tfjs_all/y26s_humanshaped_smallpatch_v1_<SZ>_{float,uint8}/` | 38.7–38.9 MB | 10.3–10.4 MB |

Each is a graph-model folder (`model.json`, `group1-shard*.bin`, `metadata.yaml`), exported with
`end2end=False, nms=False` so the output is the same raw `[1,7,N]` head the extension already
post-processes. **`uint8` = weight-only quantization** (`tensorflowjs_converter --quantize_uint8 '*'`;
activations stay float32) — accuracy ≈ the float/`.pt` rows, *not* the static-INT8 TFLite rows.
Local copy with drop-in instructions for the extension's `src/models/<id>/` tiers, IndexedDB-cache
caveat, and SHA256SUMS: `models/extension_tfjs_20260910/README.md` (gitignored, 216 MB). Builder:
`vlm-cluster/tfjs_build_two.sh`; recipe: `docs/TFJS_EXPORT_RECIPE.md` (pinned `ultralytics==8.4.82`;
newer versions no longer export TF.js).

## Input / output contract (identical for every TFLite and TF.js file above)

- **Input:** `float32`, NHWC `[1, S, S, 3]`, RGB, pixels scaled to `0..1`. INT8 files keep the
  same float32 interface (quantization is internal).
- **Output:** `[1, 7, N]` with `N = 8400 / 3549 / 2100` for 640 / 416 / 320. Rows are
  `[cx, cy, w, h, score_Woman, score_Man, score_Child]`, boxes in input-pixel coordinates.
  **This is the raw head — run NMS yourself** (same path the shipped extension already uses),
  then scale boxes back to the source frame.

## Thresholds — do not reuse 0.45 blindly

- Production today runs a single confidence threshold of **0.45** on the fp32 model.
- **INT8 redistributes confidence** (fewer boxes above 0.001, more above 0.25). Every INT8
  artifact needs its own threshold sweep before shipping — tooling: `vlm-cluster/conf_sweep.py`
  and `vlm-cluster/pr_curve.py`, procedure in `docs/HOLDOUT_BENCHMARK_HANDOFF.md`.
- `y26s_humanshaped_smallpatch_v1` peaks much higher than the nanos (F1-optimal on LAGENDA:
  Woman 0.65, Man 0.80, Child 0.60); 0.45 is too low for it. Per-model curves:
  https://claude.ai/code/artifact/24297c94-7dde-46be-900d-94487aa54e8b

## Superseded — do not ship

`/workspace/exports/humanshaped_v2_20260902/`, `/workspace/exports/noe2e_warm50-2_20260908/`,
`/workspace/exports/y26s_humanshaped_smallpatch_v1_20260909/` — earlier INT8 exports calibrated on
the validation set (retired; measurably worse at 320). Also `/workspace/exports/sop50_tfjs/` and
`/workspace/exports/y26n_sop50_20260902/` belong to `y26n_sop50`, not one of the three candidates.

## Before rollout (from the existing checklist, `docs/TFJS_EXPORT_RECIPE.md`)

1. Parity check the artifact against its `.pt` on ~30 real images: class agreement ~100%,
   mean |Δconf| < 0.01, mean box IoU > 0.98.
2. Threshold sweep for that exact artifact (above).
3. A/B in the extension's built-in compare mode before a percentage rollout.

Accuracy, size and latency for every row above: `docs/MODEL_COMPARISON.md`; consolidated report:
https://claude.ai/code/artifact/83339c8d-9d86-4d39-a880-a17fb79ff52e
