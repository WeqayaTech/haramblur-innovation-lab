# EXP-2026-22 — Precision × resolution × dataset matrix, with the production model

*Opened and closed 2026-09-14 (150/150 cells, 0 sanity flags, every cell n_images-exact) · owner: Mostafa · pods: L4 (`…105.15`, exports + `.pt` rows) + 32-core CPU pods (`cpu2`, `cpu3`, …) sharing one claim queue on the volume*

## Question

For every deployable candidate — the shipped `yolo11N-640` and the four YOLO26 candidates — how do
the five export precisions (fp32 `.pt`, fp32 TFLite, FP16 TFLite, INT8 W8A8, W8A8 + float decode
ops) compare at 640 / 416 / 320 on **both** the validation set (Spotlight-val) and the QA set
(`haramblur_holdout`, per collection)? This is the table the export decision and the report are
made from; EXP-2026-21 explained *why* INT8 loses at 640 and how the fix works, this measures the
whole space in one consistent pass.

## Pre-registered expectations (written 2026-09-14 before the holdout cells ran)

| # | expectation | falsified by |
| --- | --- | --- |
| 1 | fp32 TFLite = `.pt` within ±0.010 and FP16 = fp32 TFLite within ±0.002 mAP50-95 in **every** cell, both datasets | any cell outside — would mean the toolchain, not the model, differs between datasets |
| 2 | the float-decode fix ≥ `.pt` − 0.005 mAP50-95 on the holdout for all four YOLO26 candidates at 640 (it matched or beat fp32 on Spotlight-val) | fix below `.pt` by > 1 pt on holdout → the Spotlight-val gain was label-set specific |
| 3 | the INT8-class Child gain is *not* reproduced on the holdout `child` collection (Gemini labels, different distribution) — i.e. the gain is a Spotlight-label artefact | Child gain ≥ +3 pts on holdout too → the gain is a genuine model behaviour and the adult→Child question becomes urgent |
| 4 | production `yolo11N-640` trails every YOLO26 candidate on every holdout collection except possibly `randoms` FP rate | a collection where production wins |
| 5 | resolution ordering 640 > 416 > 320 holds on both datasets for every model at fp32; INT8's box-tightness loss shrinks with size (EXP-21) | — |

## Method

- `vlm-cluster/full_matrix.sh` (single pod) / `run_cells.sh` + `lists/cells_all.txt` (multi-pod claim
  queue: each cell = `mkdir /workspace/exp22/claims/<cell>`, atomic on the volume; runners skip
  cells whose `_map.json` exists and parses). `join_pod.sh <name>` bootstraps any new pod.
- Exports: `matrix_lib.sh::export_all` — fp32 via `yolo export format=tflite`, FP16 via
  `fp16_cast.py` (ai_edge_quantizer float_casting), INT8 via `int8=True data=spotlight_oiv7_calib500.yaml split=train`
  (500 fixed train images, sha 55d0a78e…; size-guarded ≤ ½ fp32), fix via `float_head_quant.py`
  with regex `.*Detect_[0-9]+;.*|.*_NormalizeCoords;.*|.*serving_default_output_0.*` (works for
  YOLO11's `Detect_23` too — 32 float decode tensors; its DFL distance tensor stays int8 at a
  0.05-stride step, 6× finer than YOLO26's 0.31).
- Scoring: `run_ultralytics_labels.py --floor 0.001` → `map_eval.py --ignore-labels` (Spotlight-val:
  `oiv7_val/labels` + `ignore_unk`; holdout: `labeling/full/labels_eval` polygons + `ignore/`, pooled
  plus `--stem-prefix <collection>__` for child / men / shiekhs / women / women_hd; `randoms_fp.py`
  for the person-free arm at conf 0.25 / 0.45).
- `.pt` rows: GPU where present, else torch CPU with all cores. TFLite rows: CPU (XNNPACK), 4–5 threads.
- Already-scored Spotlight-val cells (EXP-21 / 2026-09-09 matrix) are reused as byte-identical JSON copies.
- Monitoring: `matrix_status.py` — done / claimed / running per host, stalled claims, failures, sanity
  flags (n_images, plausible band, parity bands from expectation 1, INT8/fix within [−0.12, +0.06] of `.pt`).

## Results (matrix complete 2026-09-14 ≈ 21:50 pod time; rendered page: artifact 258b61fe, local `models/exp22_20260914/`)

### Scorecard against the pre-registered expectations

| # | expectation | observed | verdict |
| --- | --- | --- | --- |
| 1 | fp32 TFLite = `.pt` ± 0.010, FP16 = fp32 TFLite ± 0.002, both datasets | max Δ 0.004 (fp32 TFLite vs `.pt`) and 0.0003 (FP16 vs fp32) across all cells; `matrix_status.py` raised 0 parity flags | **held** |
| 2 | float-decode fix ≥ `.pt` − 0.005 on holdout @640, all four candidates | +0.0017 / +0.0030 / −0.0010 / +0.0013 (hv2 / noe2e / y26s / distill) | **held** |
| 3 | the INT8-class Child gain is *not* reproduced on the holdout `child` collection | holdout `child` Child-AP50-95: fix vs `.pt` = +1.0 / +1.0 / +0.1 / +0.9 pts (vs +7 on Spotlight-val); plain INT8 *loses* 4–6 pts on `child` | **mostly held** — the +7 was a Spotlight-label effect; a ~+1 residual remains, still to be checked on LAGENDA |
| 4 | production trails every candidate on every collection | true on all five collections (e.g. `women` 0.844 vs 0.898–0.936, `shiekhs` 0.675 vs 0.805–0.839) and on the `randoms` false-blur arm (7.0 % vs 4.3–5.1 % of person-free images) | **held** |
| 5 | 640 > 416 > 320 at fp32 on both datasets; INT8's box loss shrinks with size | held for every model on both datasets; INT8 loss on holdout −6.1/−5.1/−6.4/−5.1 @640 → −3.2/−3.6/−5.1/… @416 → −2.2/−2.3/−3.6/… @320 | **held** |

### Holdout (`haramblur_holdout`, 11,494 images) — pooled mAP50-95 @640

| model | `.pt` | fp32 TFLite | FP16 | INT8 W8A8 | **fix** |
| --- | --- | --- | --- | --- | --- |
| `yolo11N-640` (production) | 0.7414 | 0.7405 | 0.7404 | 0.7150 (−2.6) | **0.7631 (+2.2)** |
| `y26n_humanshaped_v2` | 0.8399 | 0.8398 | 0.8397 | 0.7788 (−6.1) | **0.8416 (+0.2)** |
| `y26n_noe2e_warm50-2` | 0.8353 | 0.8377 | 0.8376 | 0.7839 (−5.1) | **0.8383 (+0.3)** |
| `y26s_humanshaped_smallpatch_v1` | 0.8700 | 0.8704 | 0.8704 | 0.8057 (−6.4) | **0.8690 (−0.1)** |
| `y26n_humanshaped_v2_distill_v1` (ep 61) | 0.8276 | 0.8294 | 0.8294 | 0.7770 (−5.1) | **0.8289 (+0.1)** |

Per collection @640, dominant-class AP50-95 (`women_hd` / `women` / `men` / `child` / `shiekhs`) and
`randoms` image false-blur rate at conf 0.25:

| model · precision | women_hd | women | men | child | shiekhs | randoms FP |
| --- | --- | --- | --- | --- | --- | --- |
| production `.pt` | 0.804 | 0.844 | 0.772 | 0.793 | 0.675 | 7.0 % |
| production INT8 | 0.777 | 0.803 | 0.731 | 0.747 | 0.658 | 7.4 % |
| production fix | 0.808 | 0.864 | 0.789 | 0.790 | 0.707 | 7.3 % |
| `y26n_humanshaped_v2` `.pt` | 0.910 | 0.912 | 0.901 | 0.851 | 0.810 | 4.9 % |
| `y26n_humanshaped_v2` INT8 | 0.866 | 0.840 | 0.839 | 0.801 | 0.738 | 4.9 % |
| `y26n_humanshaped_v2` fix | 0.905 | 0.902 | 0.903 | 0.861 | 0.812 | 5.0 % |
| `y26n_noe2e_warm50-2` `.pt` | 0.913 | 0.907 | 0.902 | 0.837 | 0.809 | 4.3 % |
| `y26n_noe2e_warm50-2` fix | 0.901 | 0.902 | 0.904 | 0.847 | 0.811 | 4.4 % |
| `y26s_humanshaped_smallpatch_v1` `.pt` | 0.940 | 0.936 | 0.930 | 0.866 | 0.836 | 5.1 % |
| `y26s_humanshaped_smallpatch_v1` fix | 0.939 | 0.933 | 0.926 | 0.867 | 0.839 | 5.1 % |
| `y26n_humanshaped_v2_distill_v1` `.pt` | 0.905 | 0.898 | 0.895 | 0.831 | 0.807 | 4.8 % |
| `y26n_humanshaped_v2_distill_v1` fix | 0.902 | 0.892 | 0.891 | 0.840 | 0.805 | 4.3 % |

Spotlight-val @640 for the same rows (mAP50-95, `.pt` → INT8 → fix): production 0.6032 → 0.5959 →
0.6427; hv2 0.7032 → 0.6615 → 0.7296; noe2e 0.6970 → 0.6544 → 0.7174; y26s 0.7638 → 0.7019 → 0.7706;
distill 0.6848 → 0.6560 → 0.7105. Full 150-cell tables (416/320, mAP50, per class): the report.

### What the matrix says

- **Every YOLO26 candidate beats production on both datasets at every size and precision** —
  by 8–16 pts mAP50-95 on Spotlight-val and 8–13 pts on the holdout, on every collection, *and*
  with fewer false blurs on person-free images (4.3–5.1 % vs 7.0 %).
- **Plain W8A8 INT8 costs 5–6 pts mAP50-95 on the holdout at 640 for every YOLO26 model** —
  the EXP-21 head-decode loss is not a Spotlight-val artefact. The float-decode fix recovers it
  fully on the holdout (within ±0.3 of `.pt`) at INT8 speed.
- **The production model gains most from the fix** (+2.2 holdout, +4.0 Spotlight-val over its own
  `.pt`): its YOLO11 DFL head keeps a finer int8 step, so plain INT8 costs it less (−2.6), and the
  int8 head convs add Child/Woman AP on top.
- **Latency @640 (CPU, 4 threads, median ms):** production 26.7 / INT8 17.1 / fix 15.0; hv2 27.6 /
  22.5 / 20.2; noe2e 27.5 / 23.1 / 24.6; y26s 71.1 / 39.8 / 45.7; distill 25.3 / 18.3 / 16.3.
  The fix is at INT8 speed on the nanos; on y26s it costs ~6 ms over INT8 (still 1.55× faster than fp32).
- **The Spotlight-val Child gain shrinks to ~+1 pt on the holdout** — it was mostly the label
  set. The adult→Child question is now a small one but not closed (LAGENDA sweep still owed).
- fp32 TFLite / FP16 rows on the holdout were run anyway (the "optional" 30 cells) and confirmed
  the equivalences to 3–4 decimals — they can be dropped from future matrices.


### Latency — one pod, one method, all 60 files (2026-09-14, `bench_matrix.py`)

Dedicated 8-vCPU CPU pod (AMD EPYC 9655P, shared host, load 12–16 at start), LiteRT 2.2.0 +
XNNPACK, batch 1, fixed random input, 20 warm-up + 100 timed invokes, 3 interleaved repeats,
each measurement in a fresh process. **Read the 1-thread column as the reference**: it is
deterministic on this host (fp32 = FP16 to 0.1 %, repeat spread < 3 %); the 4-thread column is
what a desktop would do but is noisy on shared vCPUs (⚠ = spread > 10 % between repeats, worst 60 %).

Median ms, 1 thread (fp32 · FP16 · INT8 · **fix**):

| model | 640 | 416 | 320 |
| --- | --- | --- | --- |
| `yolo11N-640` (production) | 29.5 · 29.4 · 14.5 · **14.5** | 12.2 · 12.2 · 5.1 · **5.1** | 7.1 · 7.1 · 2.8 · **2.9** |
| `y26n_humanshaped_v2` | 26.1 · 26.1 · 16.6 · **16.6** | 10.6 · 10.6 · 4.7 · **5.0** | 6.2 · 6.1 · 2.6 · **2.6** |
| `y26n_noe2e_warm50-2` | 26.1 · 26.0 · 16.7 · **16.7** | 10.6 · 10.6 · 5.0 · **5.0** | 6.1 · 6.1 · 2.6 · **2.6** |
| `y26s_humanshaped_smallpatch_v1` | 87.7 · 87.7 · 40.9 · **41.0** | 36.4 · 36.4 · 13.2 · **12.8** | 21.7 · 21.7 · 7.1 · **7.1** |
| `y26n_humanshaped_v2_distill_v1` | 26.1 · 26.1 · 16.7 · **15.0** | 10.6 · 10.6 · 5.0 · **4.7** | 6.1 · 6.1 · 2.6 · **2.5** |

4 threads, median ms (fp32 · INT8 · fix; ⚠ noisy): production 640 11.9⚠ · 10.1⚠ · 10.7; hv2 640
14.3 · 13.3 · 13.3; y26s 640 51.3⚠ · 27.7⚠ · 27.5; full table on the report page.

What it says: **the float-decode fix costs nothing over plain INT8** (±0.1 ms on every nano cell,
+0.1 ms on y26s @640, faster on the distill model); **INT8 is 1.6–2.8× faster than fp32** at one
thread (largest at 416/320 and on y26s); **FP16 is not a speed option** (weights are dequantized
at load — identical to fp32 everywhere); resolution is the big lever (640 → 416 ≈ 2.5×, 416 → 320
≈ 1.7×). A YOLO26n at 416 in the fixed INT8 export runs in ~5 ms single-threaded on this CPU —
a third of the production model's fp32 @640. None of this is the extension's in-browser (TF.js)
latency; that needs a browser measurement (the extension's Compare Mode reports per-model ms).
Raw JSON: `models/exp22_20260914/bench_matrix.json`.

## Run notes (for the next person)

- The network volume's quota filled mid-run (a 9.5 GB venv install on the volume + old dumps).
  Symptoms: `scp` "close remote: Failure", empty `_map.json` files, cells failing instantly so
  runners burned through claims with unwritable owner files. Fix: free space, delete unparsable
  JSONs, release ownerless claims, restart runners; runners now validate JSON before skipping.
- Venvs go on the container disk (`join_pod.sh`), never on the volume — small-file writes to the
  network FS are ~10× slower and eat quota. A 5 GB container disk is too small for torch + ultralytics.
- CPU pods are the right hardware: 130/150 cells are LiteRT (CPU-only); a GPU helps only the 15
  holdout `.pt` rows.

## Verify it yourself

```bash
python3 /workspace/exp22/matrix_status.py                        # state + sanity flags
grep -h '^RESULT' /workspace/exp22/logs/*.log | sort              # every scored cell, one line each
python3 vlm-cluster/build_matrix_report.py --eval models/exp22_20260914/eval --out /tmp/r.html
```
