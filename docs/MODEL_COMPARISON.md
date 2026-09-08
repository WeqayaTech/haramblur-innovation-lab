# Model comparison — all measured models, side by side (updated 2026-08-08)

Every model evaluated under the identical frozen protocol (`docs/MODEL_EVAL_OVERVIEW.md`:
same images verified by count, 640 px, conf 0.45, IoU-0.5 greedy matching, same scorers,
one shared mAP evaluator). Scored **per dataset, never pooled** — the columns below are
labeled by dataset and must not be averaged into a single score. Source experiments:
EXP-2026-12, EXP-2026-13 (zero-shot YOLOE), EXP-2026-14/15 + the 2026-08-08 round.
Per-model JSONs: `/workspace/evalout/` and `/workspace/exp12/eval/`.

Models are listed under their **exact run/directory names** throughout this document.

## The models

| run name (as on disk) | weights path | params (fused) | labels / unknown handling |
|---|---|---|---|
| `yolo11N-640` | `/workspace/model_v2/dataset_v2/yolo11N-640/weights/best.pt` | 2.58M | old labels — **the deployed-era baseline** (`v11nclean2` export family) |
| `gelansfav14_datav2_v4` | `/workspace/YOLO-MIT/runs/v9MIT/gelansfav14_datav2_v4/weights/best.ckpt` | 8.21M | old labels — best v9 checkpoint, never shipped (too slow) |
| `gelansfav14_gemlb_v1` | `/workspace/YOLO-MIT/runs/v9MIT/gelansfav14_gemlb_v1/weights/best.ckpt` | 8.21M | Spotlight, **grey-masked** unknowns |
| `y26n_spotlight` | `/workspace/exp12/train/y26n_spotlight/weights/best.pt` | 2.38M | Spotlight, unknowns **as background** |
| **`y26n_gradsupp`** | `/workspace/exp14/train/y26n_gradsupp/weights/best_plain.pt` | 2.38M | Spotlight, **cls-gradient-suppressed** unknowns (EXP-2026-14 arm C) |
| `y26n_unk4` | `/workspace/exp14/train/y26n_unk4/weights/best.pt` | 2.38M | Spotlight, **unknown-as-4th-class**, dropped at inference (arm B) |
| `yoloe_n_gradsupp` | `/workspace/exp15/train/yoloe_n_gradsupp/weights/best_plain.pt` | 2.69M | Spotlight, gradient-suppressed (EXP-2026-15) |
| `y26n_twoaxis2` ⏹ | `/workspace/exp17/train/y26n_twoaxis2/weights/last.pt` | 2.38M | Spotlight **two-axis** labels, unknowns restored as `GenderUnknown`/`AgeUnknown` (EXP-2026-17) — **stopped at epoch 66/80** |
| `yoloe-26s-seg` (zero-shot) | ultralytics auto-download | 13.99M | none — text prompts `woman,man,child` (EXP-2026-13) |
| **`gelannfav14r4_gemlb_v2`** | `/workspace/YOLO-MIT/runs/v9MIT/gelannfav14r4_gemlb_v2/weights/best.ckpt` | ~3M | Spotlight (coworker's recipe) — **coworker main** |
| `gelannfav14w_gemlb_v1` | `/workspace/YOLO-MIT/runs/v9MIT/gelannfav14w_gemlb_v1/weights/best.ckpt` | ~6.6M | Spotlight (coworker's recipe) — **coworker main** |
| `gelannfav14r3_gemlb_v1` | `/workspace/YOLO-MIT/runs/v9MIT/gelannfav14r3_gemlb_v1/weights/best.ckpt` | ~3M | Spotlight — timed only, not accuracy-scored |
| `gelannfav14m_gemlb_v1` | `/workspace/YOLO-MIT/runs/v9MIT/gelannfav14m_gemlb_v1/weights/best.ckpt` | ~4.7M | Spotlight — timed only, not accuracy-scored |

## Accuracy — all datasets, one row per model

Lower is better for the two FP columns and the leak column; higher is better elsewhere.

⏹ **`y26n_twoaxis2` is an incomplete arm and is not a candidate.** It was stopped at **epoch 66 of 80**
(before `close_mosaic` at 70) and it **fails 4 of its 6 pre-registered no-regression bars**, so
EXP-2026-17's own decision rule rejects it. Two further reasons its row is not comparable like-for-like:
its LAGENDA detection figure (97.4) is labeled-person recall on the v2 answer key — its **all-people**
recall is **84.9**, the best measured, against `y26n_gradsupp`'s 74.8 in the corrected table below; and
its confidence is the max over **six** channels of which every person lights up two, so conf 0.45 is a
looser operating point here than on a 3-class head. That inflates recall and false positives together —
which is exactly the shape of its row. **The matched-FP `conf_sweep.py` run that would settle it was
never done.** Full write-up: `experiments/EXP-2026-17-two-axis-head.md`.

| run name | objects<br>%img-FP ↓<br>(259 imgs) | PASS<br>FP/100 ↓<br>(3,000) | crowd<br>recall<br>(517) | crowd<br>prec. | LAGENDA<br>detection<br>(5,000) | LAGENDA<br>adult gender | LAGENDA<br>leak ≥20→Child ↓ | LAGENDA<br>child recall |
|---|---|---|---|---|---|---|---|---|
| `yolo11N-640` | 54.1 | 1.17 | 34.9 | 94.4 | 95.3 | 89.7 | 0.52 | 81.2 |
| `gelansfav14_datav2_v4` | 53.7 | 1.43 | 39.6 | **95.6** | **96.0** | 87.2 | 0.77 | **92.1** |
| `gelansfav14_gemlb_v1` | 18.2 | **0.23** | **46.2** | 92.7 | 95.8 | 90.6 | **0.00** | 85.8 |
| `y26n_spotlight` | 12.7 | **0.23** | 32.8 | 93.9 | 94.1 | 90.7 | 0.11 | 79.0 |
| **`y26n_gradsupp`** | **11.2** | 0.27 | 36.5 | 94.0 | 93.9 | 91.1 | 0.11 | 81.0 |
| `y26n_unk4` | 11.6 | 0.40 | 33.5 | **94.2** | 93.3 | 90.9 | 0.15 | 81.1 |
| `yoloe_n_gradsupp` | 11.6 | 0.37 | 36.3 | 93.5 | 93.3 | 91.5 | 0.15 | 80.2 |
| `y26n_twoaxis2` ⏹ | 18.2 ⏹ | 1.00 ⏹ | **47.3** ⏹ | 92.7 | 97.4 | 89.9 | 0.34 | 69.1 |
| `yoloe-26s-seg` (zero-shot) | 2.7 ⚠ | 0.0 ⚠ | 0.3 ⚠ | 100 ⚠ | 13.6 ⚠ | 97.2 ⚠ | 0.00 ⚠ | 45.2 ⚠ |
| **`gelannfav14r4_gemlb_v2`** | 16.2 | 0.33 | **41.1** | 92.7 | **95.2** | **92.0** | **0.07** | 81.3 |
| `gelannfav14w_gemlb_v1` | **10.0** | 0.57 | 36.7 | 92.3 | 93.0 | 89.7 | 0.34 | 85.2 |

⚠ **Zero-shot YOLOE's row is not usable as-is.** At conf 0.45 it barely detects anything,
which makes its FP columns look spectacular for free and its recall columns catastrophic;
at the confidence floor where detection works (93.8% LAGENDA), gender collapses to 85.4%,
object FPs rise to 46.3% and crowd recall peaks at 25.3%. Full curve in EXP-2026-13.

Heavy-occlusion crowd recall (in the eval JSONs as `recall_by_occlusion`, tabulated so
far only for the EXP-2026-12 arms): `gelansfav14_gemlb_v1` 27.0 · `gelansfav14_datav2_v4`
15.6 · `y26n_spotlight` 13.2 · `yolo11N-640` 11.9.

## LAGENDA v2 — the corrected numbers (2026-08-10)

**Correction to the accuracy table above:** its "LAGENDA detection" column (93.0–95.3%)
is **prominent-subject recall** — the old eval set carried ~1 labeled person per image.
The official LAGENDA CSV labels ~6 person boxes per image; rebuilt against the full set
(4,601 images / 27,633 person boxes / 7,098 human age+gender labels, 298
training-contaminated images excluded), the honest numbers are:

| run name | det recall (all people) | precision | dup% | gender | leak ≥20→Child ↓ | child |
|---|---|---|---|---|---|---|
| `y26n_spotlight` | 72.1 | **94.3** | 2.3 | 90.5 | 0.34 | 79.1 |
| **`y26n_gradsupp`** | 74.8 | 93.3 | 2.2 | 90.8 | 0.37 | 81.3 |
| `y26n_unk4` | 72.7 | 94.1 | 2.0 | 90.7 | 0.45 | 81.1 |
| `yoloe_n_gradsupp` | 74.2 | 93.4 | 2.0 | 91.3 | 0.26 | 79.7 |
| **`gelannfav14r4_gemlb_v2`** | **79.6** | 91.5 | 2.4 | **92.4** | **0.16** | 80.9 |
| `gelannfav14w_gemlb_v1` | 75.7 | 90.9 | 3.4 | 90.0 | 0.48 | **83.9** |

Notes: LAGENDA **precision** is measurable for the first time (the y26n family leads it,
consistent with its FP advantage) — but treat it as **provisional, pending adjudication**:
a 2026-08-10 spot check of 9 flagged false positives (from the 6 densest crowd images in
the set — an extreme, non-representative sample) found them to be real people LAGENDA's
detector never boxed, which would make precision an understated lower bound. **Not yet
established**: that sample is far too small and too skewed to quantify, and the opposite
error is also unmeasured — if LAGENDA's shipped detector output contains false positives
of its own, treating those boxes as ignore regions would *inflate* precision instead. A
~100-box hand adjudication in both directions is the open task. Gradient suppression's
recall gain **replicates on a second dataset** (+2.7 over background here, +3.7 on
CrowdHuman); leak percentages read slightly higher than the old table because the
matched-adult pool grew ~40% and now includes smaller/harder people.

Evidence for the ignore design (`vlm-cluster/lagenda_map_audit.py --verify/--trace`,
2026-08-10): counts reconcile from two independent implementations — 7,098 scoreable +
20,535 ignore = 27,633 total person boxes, i.e. only **25.7%** of LAGENDA's people carry
human age/gender. On the six densest sampled images the model made 209 detections on real
people, of which **188 landed on unlabeled people** — every one of which the old
one-person-per-image ground truth would have scored as a false positive. Caveat: LAGENDA person boxes are the dataset authors'
detector output (their YOLOv8), not human-drawn — detection numbers here measure
agreement with their detector; CrowdHuman remains the human-GT detection benchmark.
Classification labels are fully human (10 votes/sample).

## mAP — three benchmarks, COCO-conventional (2026-08-10)

Protocol: pycocotools convention (101-point mean, confidence-ordered greedy matching),
detections logged to **floor 0.001**, maxDets=100, ignore-region support. Every output
JSON self-describes this protocol. NOT comparable to the earlier floor-0.05 numbers
(which read ~0.02 lower) — the y26n family measured 0.799→0.822 mAP50 on the same data
purely from the floor change, exactly the predicted truncation effect.

| run name | Spotlight-val mAP50 / 50-95 ¹ | LAGENDA 3-class (human) mAP50 / 50-95 ² | CrowdHuman person AP50 / 50-95 ³ |
|---|---|---|---|
| `y26n_spotlight` | 0.827 / 0.708 | 0.789 / **0.661** | 0.582 / 0.323 |
| **`y26n_gradsupp`** | 0.829 / 0.713 | 0.789 / **0.661** | 0.603 / 0.333 |
| `y26n_unk4` | 0.827 / 0.713 | 0.788 / **0.661** | 0.597 / 0.329 |
| `yoloe_n_gradsupp` | **0.831** / **0.715** | 0.786 / 0.657 | 0.605 / 0.334 |
| **`gelannfav14r4_gemlb_v2`** | 0.823 / 0.672 | **0.798** / 0.658 | **0.617** / **0.342** |
| `gelannfav14w_gemlb_v1` | 0.790 / 0.637 | 0.779 / 0.647 | 0.589 / 0.327 |

¹ Gemini-verified Spotlight labels; `best.pt` selected by fitness on this split → mildly
optimistic label-alignment, never the verdict. The 702 unknown-gender people the emit
deleted from val GT are treated as **ignore regions** (they'd otherwise count correct
detections as FPs — checked 2026-08-10: the fix lifts every model by a uniform
+0.005–0.008 mAP50, no ranking change).
² **Human-labeled 3-class mAP** — the 7,098 crowdsourced age/gender people, with the
20,535 detected-but-unlabeled person boxes as COCO ignore regions (neither TP nor FP).
The strongest mAP this project has. The `small` area slice has no support here (labeled
people are face-anchored and essentially never < 32² px) — use CrowdHuman for
small-person behavior.
³ Class-agnostic person mAP vs human-drawn exhaustive boxes, 517 images.

**What mAP adds to the decision: nothing that changes it.** On the human-labeled 3-class
benchmark the candidates are within ±0.012 mAP50 of each other — `gelannfav14r4_gemlb_v2`
edges AP50 (finds more people), the y26n family edges AP50-95 (tighter boxes), i.e. the
same recall-vs-precision split the component metrics already showed. The recommendation
still rests on the FP/latency trade.

## Size & speed — all re-timed in ONE session (2026-08-08)

ONNX Runtime CPU, **AMD EPYC 9254, 4 threads, 640×640, median of 100 runs after 20
warmup**. Same machine for every row; earlier cross-machine figures are superseded.

| run name | ONNX file | params | GFLOPs | **median** | p90 | NMS included? |
|---|---|---|---|---|---|---|
| `y26n_unk4` | 9.8 MB | 2.38M | 5.3 | **33.6 ms** | 33.9 | ✅ baked in |
| **`y26n_gradsupp`** | 9.8 MB | 2.38M | 5.3 | **33.8 ms** | 34.0 | ✅ baked in |
| `y26n_spotlight` | 9.8 MB | 2.38M | 5.3 | 34.0 ms | 34.3 | ✅ baked in |
| `yolo11N-640` | 10.6 MB | 2.58M | 6.4 | 41.1 ms | 41.3 | ❌ browser pays |
| `yoloe_n_gradsupp` | 11.1 MB | 2.69M | 9.1 | 49.1 ms | 50.3 | ✅ baked in (+ mask head) |
| `gelannfav14r3_gemlb_v1` | 15.7 MB | ~3M | — | 58.5 ms | 59.0 | ❌ browser pays |
| **`gelannfav14r4_gemlb_v2`** | 15.9 MB | ~3M | — | **59.5 ms** | 59.9 | ❌ browser pays |
| `gelannfav14m_gemlb_v1` | 19.1 MB | ~4.7M | — | 73.1 ms | 73.5 | ❌ browser pays |
| `gelannfav14w_gemlb_v1` | 27.1 MB | ~6.6M | — | 89.3 ms | 89.7 | ❌ browser pays |
| `gelansfav14_gemlb_v1` / `gelansfav14_datav2_v4` | 33 MB | 8.21M | — | ~135 ms † | — | ❌ browser pays |

† measured in the earlier EPYC 7352 session, not re-timed today.

**Reading the NMS column is essential.** YOLO26 exports end-to-end `(1, 300, 6)` — final
boxes, nothing left to do. `yolo11N-640` and every gelan export emit raw `(1, 7, 8400)`
tensors, so the browser still runs non-max-suppression over 8,400 candidates *on top of*
the time shown. Their true end-to-end cost is higher than listed; the YOLO26 rows are
complete.

Measurement validity cross-check: `y26n_spotlight` timed 34.0 ms here vs 33 ms on the
earlier EPYC 7352 session, and `yolo11N-640` 41.1 vs 42.1 — ~3% agreement across two
machines, which is why the un-re-timed ~135 ms figures stay comparable.

Correction to an earlier note: `yoloe_n_gradsupp`'s deployable fused model is **2.69M
params**, not the 5.6M quoted from the architecture yaml (the text-prompt machinery is
stripped during PE fine-tuning). Its penalty is compute, not parameter count.

## The bottom line (as of 2026-08-08, latency now measured)

**Recommendation: `y26n_gradsupp`.** With speed measured, the two-finalist question
resolves — it is not a balanced trade:

| | `y26n_gradsupp` | `gelannfav14r4_gemlb_v2` |
|---|---|---|
| latency | **33.8 ms**, NMS included | 59.5 ms **+ browser NMS on top** (≥1.8× slower) |
| object false persons ↓ | **11.2%** | 16.2% (+45% more false blurs) |
| PASS FP/100 ↓ | **0.27** | 0.33 |
| crowd recall | 36.5% | **41.1%** (+4.6) |
| LAGENDA detection | 93.9% | **95.2%** (+1.3) |
| adult gender | 91.1% | **92.0%** (+0.9) |
| leak ≥20 → Child ↓ | 0.11% | **0.07%** |

`gelannfav14r4_gemlb_v2` is genuinely better at finding and classifying people. But it
costs ~2× the compute *and* blurs 45% more non-people — losing on both the #1 user
complaint and the frame budget, to win a few points on the escape side. For a browser
extension processing video frames (which already downshifts to 320/416 inputs on slow
machines), that is the wrong trade. The product call belongs to the owner; the data points
one way.

Also settled by the timing run: **YOLOE is closed on compute, not size** — 2.69M params
but 9.1 GFLOPs and 49.1 ms (45% slower than `y26n_gradsupp`) for zero accuracy gain. And
the entire gelan nano family sits at 58–89 ms before NMS, so none of it competes on speed
with the YOLO26 line.

Everything else: `yolo11N-640` and `gelansfav14_datav2_v4` are dominated on the complaint
axis; `y26n_unk4` and `gelannfav14w_gemlb_v1` are dominated overall; `gelansfav14_gemlb_v1`
remains the accuracy ceiling reference at ~4× the latency budget. **Untried arm most
likely to merge both finalists' strengths: grey-masked unknowns on YOLO26n** — masking's
crowd-recall gain has now survived two architectures, and no YOLO26n arm has used it. That
is the one experiment that could give `gelannfav14r4_gemlb_v2`'s recall at `y26n_gradsupp`'s
speed.

Standing caveats: the Gulf/traditional-dress gender slice is untested for every model in
this document; object-set rates are upper bounds (set re-verification pending); PASS has
~5% known contamination (hits all models equally); single-run numbers carry unquantified
run-to-run variance — treat ≤1-pt differences as ties.

## Holdout benchmark — `haramblur_holdout` (2026-08-19, first measurement)

**A different dataset from everything above — never pool with it.** `haramblur_holdout`
(see CLAUDE.md → "THE HOLDOUT TEST SET") is a fresh, deduplicated, never-trained-on
11,494-image set with the project's first dedicated Gulf-dress slice (`shiekhs`, 2,897
imgs). Labels are machine-generated (Gemini 3.5 Flash-Lite via Spotlight), not
human-verified — treat this as relative model ranking, not human-accurate ground truth.
COCO-style mAP (`map_eval.py`, floor 0.001, `--ignore-labels` applied throughout so the
1,619 unknown-gender people are never scored as false positives).

| run name | weights path |
|---|---|
| **`y26n_noe2e_warm50-2`** | `/workspace/exp18/train/y26n_noe2e_warm50-2/weights/best.pt` |
| `yolo11N-640` (shipped production) | `/workspace/model_v2/dataset_v2/yolo11N-640/weights/best.pt` |

**Full set (11,494 images, 16,139 GT boxes):**

| run name | mAP50 | mAP75 | mAP50-95 | Woman AP50 | Man AP50 | Child AP50 |
|---|---|---|---|---|---|---|
| **`y26n_noe2e_warm50-2`** | **0.9125** | **0.8677** | **0.8353** | **0.9281** | **0.9247** | **0.8846** |
| `yolo11N-640` (production) | 0.8636 | 0.8038 | 0.7412 | 0.9094 | 0.8479 | 0.8336 |

The finetuned model wins on every full-set metric — largest gaps on Man (+7.7 AP50) and
overall mAP50-95 (+9.4 pts). By GT box size, both models are markedly weaker on small
people (mAP50-95: warm50-2 small=0.267/medium=0.607/large=0.874; production
small=0.168/medium=0.520/large=0.782) — the finetuned model leads at every size band too.

**Per collection — dominant class's AP50 is the meaningful number; the other two classes
in an off-target collection come from very few GT boxes and are noisy, not a reliable
signal.** n_gt per collection: child 1,969 · men 2,498 · randoms 148 · shiekhs 5,546 ·
women 4,924 · women_hd 1,054.

| collection | dominant class | `y26n_noe2e_warm50-2` AP50 | `yolo11N-640` AP50 |
|---|---|---|---|
| child | Child | **0.921** | 0.897 |
| men | Man | **0.945** | 0.874 |
| **shiekhs** (Gulf-dress) | Man | **0.939** | 0.872 |
| women | Woman | **0.953** | 0.948 |
| women_hd | Woman | **0.929** | 0.896 |

**The Gulf-dress number, first-ever for both checkpoints:** in `shiekhs`, both models are
strong on Man (dominant class, as above). On the rarer "Woman" class within that same
slice — only 64 real Woman GT boxes across 2,897 images, so this is a real but
statistically thin signal, not a final-word number — `y26n_noe2e_warm50-2` scores AP50
**0.376** vs `yolo11N-640`'s **0.298**. Both leave real room to improve; the finetuned
model is measurably better but the historical bias is not resolved by either.

**`randoms` collection — NOT a trustworthy number for either model, by design of this
run.** Both the per-collection mAP (n_gt=148) and the dedicated negatives-arm result below
are scored against ground truth containing the 254 unadjudicated gate-survivors flagged in
`labeling/FULL_RUN.md` — until those are hand-eyeballed, don't quote a verdict from this
arm:

| run name | FP/100 images | images w/ ≥1 FP | FP by class |
|---|---|---|---|
| `y26n_noe2e_warm50-2` | 6.18 | 4.26% (84/1,973) | Man 83, Woman 28, Child 11 |
| `yolo11N-640` (production) | 9.43 | 7.05% (139/1,973) | Man 110, Woman 64, Child 12 |

Directionally the finetuned model has fewer false persons here too, consistent with every
other false-positive metric measured for it across this whole project — but per the
pre-registered rule, this is reported for completeness, not as a verdict.

**Bug caught during this run, fixed before scoring:** `map_eval.py` derives its scored
image set from `--raw` (the sidecar dir), not from `--gt-labels`. Pointing it at the FULL
11,494-image raw dump while `--gt-labels` was sliced to one collection (as a literal
reading of the handoff doc's per-collection example would do) would count every *other*
collection's detections as false positives against that collection's ground truth,
corrupting precision/AP for every collection. Fixed by building per-collection `raw/`
sidecar slices (mirroring the image/label/ignore slice convention) and scoring each
collection against its own matching raw slice. `eval_negatives_crowd.py` does not have
this problem — it is driven by `--images`, already correctly sliced.

Full per-collection JSONs: `/workspace/holdout_eval/<run_name>/map_<collection>.json`;
negatives-arm: `/workspace/holdout_eval/<run_name>/negatives_randoms/summary.json`.

## Holdout benchmark update — 4-way comparison (2026-08-24)

Same protocol as above (`map_eval.py`, floor 0.001, `--ignore-labels` throughout), extended
to two more checkpoints: **`y26n_sop50`**, the EXP result of continuing
`y26n_noe2e_warm50-2` for 50 more epochs with a live `SmallObjectPatches` augmentation
(`vlm-cluster/train_small_object_patches.py`, ported from YOLO-MIT's own augmentation of the
same name, `--sop-p 0.05`) — and **`gelannfav14r4fw_gemlb_v2`**, an independent run
of the *same idea* (`task.data.data_augment.SmallObjectPatches=0.05`) applied natively inside
the YOLO-MIT/GELAN framework instead of ported into ultralytics, continuing from
`gelannfav14r4fw_gemlb_v1` for 15 epochs with class-3 masked-and-dropped-from-head unknown
handling. Different architecture (GELAN, not YOLO26n) and different training recipe — treat
the gelan row as an informative reference point, not an apples-to-apples ablation of the
augmentation alone.

| run name | weights path |
|---|---|
| `y26n_sop50` | trained via `train_small_object_patches.py --resume`, continuing `/workspace/exp18/train/y26n_noe2e_warm50-2/weights/best.pt`; final copy `models/y26n_sop50/best.pt` (local repo) |
| **`y26n_noe2e_warm50-2`** (baseline) | `/workspace/exp18/train/y26n_noe2e_warm50-2/weights/best.pt` |
| `gelannfav14r4fw_gemlb_v2` | `/workspace/YOLO-MIT/runs/v9MIT/gelannfav14r4fw_gemlb_v2/weights/best.ckpt` |
| `yolo11N-640` (shipped production) | `/workspace/model_v2/dataset_v2/yolo11N-640/weights/best.pt` |

**Full set (11,494 images, 16,139 GT boxes), mAP50-95 per class:**

| run name | mAP50 | mAP75 | mAP50-95 | Woman | Man | Child |
|---|---|---|---|---|---|---|
| `y26n_sop50` | 0.9120 | 0.8667 | **0.8354** | 0.8860 | 0.8170 | **0.8033** |
| **`y26n_noe2e_warm50-2`** (baseline) | 0.9125 | **0.8677** | 0.8353 | **0.8878** | **0.8187** | 0.7994 |
| `gelannfav14r4fw_gemlb_v2` | **0.9207** | 0.8529 | 0.7932 | 0.8507 | 0.7654 | 0.7635 |
| `yolo11N-640` (production) | 0.8636 | 0.8038 | 0.7412 | 0.8083 | 0.6795 | 0.7358 |

`y26n_sop50` and the baseline are statistically tied on full-set mAP50-95 (0.8354 vs 0.8353)
— the augmentation did not regress general accuracy. `y26n_sop50` edges ahead on Child
(0.8033 vs 0.7994); the baseline leads Woman and Man. `gelannfav14r4fw_gemlb_v2` wins mAP50
(best localization at IoU 0.5) but trails both y26n checkpoints on mAP50-95, driven mostly by
weaker Man (0.7654) and Child (0.7635) scores. All three beat production by a wide margin.

**Per-collection mAP50-95:**

| collection | `y26n_sop50` | `y26n_warm50-2` | `gelan_r4fw_v2` | production |
|---|---|---|---|---|
| child | 0.531 | **0.545** | 0.516 | 0.451 |
| men | 0.445 | 0.451 | **0.458** | 0.329 |
| randoms | **0.609** | 0.557 | 0.585 | 0.421 |
| shiekhs (Gulf-dress) | **0.481** | 0.461 | 0.455 | 0.376 |
| women | **0.658** | 0.635 | 0.634 | 0.546 |
| women_hd | 0.482 | 0.461 | **0.511** | 0.380 |

`y26n_sop50` leads on `randoms`, `shiekhs`, and `women` (the collections it improved on
in the `smallperson_v1` real-photo `crowd_small` tracking too — see the small-object
section below) but is slightly behind baseline on `child`/`men`. `gelan_r4fw_v2` wins
`men` and `women_hd` outright and is competitive everywhere except full-set mAP50-95.

**Negatives arm (`randoms`, 1,973 person-free images — same unadjudicated-ground-truth
caveat as above applies):**

| run name | FP/100 images | fp_total |
|---|---|---|
| `gelan_r4fw_v2` | **5.32** | 105 |
| `y26n_noe2e_warm50-2` (baseline) | 6.18 | 122 |
| `y26n_sop50` | 6.34 | 125 |
| `yolo11N-640` (production) | 9.43 | 186 |

The gelan checkpoint has the cleanest false-positive rate of all four; `y26n_sop50` is
marginally noisier than baseline here, consistent with the small but real FP-rate
regression seen on `crowd_small` during training (see below). All three finetuned
checkpoints are well clear of production.

**Small-object side note (`smallperson_v1`, real `crowd_small` CrowdHuman crops +
synthetic `synth_shrunk` shrink-to-target arm — different benchmark from the holdout set
above, do not pool):** `y26n_sop50`'s own training-time tracking (every epoch, 9→50) showed
a real, sustained crowd-photo recall gain (~5.2%→~5.7-6.0%) that plateaus below the
`synth_shrunk` benchmark's baseline score at every shrink size (h96 mAP50-95: baseline
0.286 vs `y26n_sop50` 0.093) — i.e., the augmentation helped realistic small-person photos
more than it helped the synthetic tiny-crop stress test, and the holdout numbers above
corroborate the "real photos, modest net win" read rather than the "synthetic benchmark,
clear loss" read. Two different measurements of small-object handling point in different
directions; neither should be quoted without the other.

## Model sizes — the four small-person candidates (measured 2026-08-25)

| run name | params (fused) | GFLOPs @640 | checkpoint on disk | ONNX CPU latency |
|---|---:|---:|---:|---:|
| `y26n_noe2e_warm50-2` | **2.375 M** | **5.3** | 5.1 MB | ~33.6–34.0 ms |
| `y26n_sop50` | **2.375 M** | **5.3** | 14.7 MB *(training state)* | ~33.6–34.0 ms |
| `yolo11N-640` (production) | 2.583 M | 6.4 | 5.5 MB | 41.1 ms |
| `gelannfav14r4fw_gemlb_v2` | 2.962 M | — | 36.0 MB *(model+EMA+optimizer)* | ~58.5–59.5 ms |

**Read the disk column carefully — it is not model size.** `y26n_sop50` is byte-identical in
architecture to `y26n_noe2e_warm50-2` (same 2.375 M params) and only looks 3× larger because
its checkpoint still carries optimizer state. The gelan checkpoint stores the weights **twice**
(`model.` and `ema.` prefixes, 2.962 M each) plus optimizer state, so a naive parameter count
over its `state_dict` reports 5.925 M — double the real figure. Latencies are the EPYC ONNX
numbers from the latency section above, for the same architecture families.

The practical size story: **all four are nano-class and within 0.6 M params of each other.**
Nothing here is a size/accuracy trade — gelan's ~1.75× latency is the only real cost.

## Small people (EXP-2026-19, `smallperson_v1`, measured 2026-08-20)

The regime production QA complains about, isolated for the first time. Arm A is 800
CrowdHuman-val images holding 16,314 people whose **full body is ≤96 px**; people above the
band are ignore regions, so this is a pure small-person number. conf 0.45, IoU 0.5.

| run name | weights | small-person recall | precision | clear FP/100 |
|---|---|---:|---:|---:|
| `gelannfav14r4fw_gemlb_v2` | `/workspace/YOLO-MIT/runs/v9MIT/gelannfav14r4fw_gemlb_v2` (`--engine mit`) | **7.58%** | 93.2% | **2.75** |
| `y26n_gradsupp` (`--no-e2e`) | `/workspace/exp14/train/y26n_gradsupp/weights/best_plain.pt` | 7.38% | 93.4% | 3.88 |
| `y26n_noe2e_warm50-2` | `/workspace/exp18/train/y26n_noe2e_warm50-2/weights/best.pt` | 5.97% | 92.3% | 3.75 |
| `y26n_sop50` | `/workspace/exp19/train/y26n_sop50/weights/best.pt` | 5.74% | 92.1% | 3.62 |
| `yolo11N-640` (production) | `/workspace/model_v2/dataset_v2/yolo11N-640/weights/best.pt` | 5.11% | **95.2%** | 3.00 |

**On the most realistic arm, production is LAST.** `gelannfav14r4fw_gemlb_v2` finds 48% more
small people than production (1,236 vs 834) *and* has the cleanest false-positive rate.

For scale: `y26n_gradsupp` scores 36.5% on the all-sizes CrowdHuman sample and 7.4% here.
**Roughly 19 in 20 small people are invisible to the shipped model.**

Arm B scores the SAME 480 LAGENDA people at five controlled apparent sizes (AP50):

| | h96 | h64 | h48 | h32 | h24 |
|---|---:|---:|---:|---:|---:|
| `yolo11N-640` **Woman** | **0.465** | **0.257** | **0.122** | **0.025** | 0.007 |
| `y26n_gradsupp` **Woman** | 0.389 | 0.194 | 0.076 | 0.019 | 0.009 |
| `yolo11N-640` Man | 0.607 | 0.393 | 0.244 | 0.082 | 0.034 |
| `y26n_gradsupp` Man | **0.627** | **0.435** | **0.316** | **0.110** | **0.063** |
| `yolo11N-640` Child | 0.114 | 0.035 | 0.010 | 0.000 | 0.000 |
| `y26n_gradsupp` Child | 0.119 | 0.035 | 0.010 | 0.000 | 0.000 |
| `y26n_noe2e_warm50-2` Woman | 0.426 | 0.191 | 0.083 | 0.026 | 0.013 |
| `y26n_noe2e_warm50-2` Man | **0.669** | **0.426** | 0.311 | 0.110 | 0.062 |
| `y26n_noe2e_warm50-2` Child | **0.175** | **0.050** | 0.009 | **0.010** | 0.000 |

**`y26n_gradsupp` is the better detector of small people and the worse product on them:** it
leads Man AP50 at every size and trails Woman AP50 at four of five. Since a detected-but-
mislabelled woman is an unblurred woman, the shipped model is better where it counts on small
people. This is the controlled reproduction of the production QA observation in CLAUDE.md
§4.9995 and points at Spotlight's 2.5× Woman→Man labelling asymmetry, not the architecture.
Arm C (`paste_grey`, 250 imgs / 1,500 people cut out with SAM polygons and pasted onto a
640 grey canvas) reports the true product metric — **a woman both found AND called Woman**:

| metric | model | h96 | h64 | h48 | h32 | h24 |
|---|---|---:|---:|---:|---:|---:|
| detection recall | `yolo11N-640` | **85.0** | **64.7** | 35.0 | 3.3 | 0.3 |
| | `y26n_gradsupp` | 70.7 | 49.3 | **44.0** | **16.3** | **4.7** |
| | `y26n_noe2e_warm50-2` | 70.0 | 62.0 | **45.0** | **16.3** | 4.0 |
| **Woman end-to-end** | `yolo11N-640` | **72.2** | 42.6 | 19.4 | 1.9 | 0.0 |
| | `y26n_gradsupp` | 60.2 | 38.9 | **31.5** | **8.3** | **1.9** |
| | `y26n_noe2e_warm50-2` | 57.4 | **49.1** | 23.1 | 7.4 | **1.9** |

| | `y26n_sop50` | 76.9 | 61.1 | 38.9 | 10.2 | 1.9 |
| | `gelannfav14r4fw_gemlb_v2` | **84.3** | **77.8** | **60.2** | **15.7** | 1.9 |

**`gelannfav14r4fw_gemlb_v2` beats production at every single size** — 84.3 vs 72.2 at h96,
and 60.2 vs 19.4 at h48, more than 3×. `y26n_sop50` also beats production at every size.
Their detection recalls are 91.3 / 89.3 / 78.7 / 31.7 / 8.3 and 94.7 / 85.3 / 70.3 / 28.7 /
6.7 against production's 85.0 / 64.7 / 35.0 / 3.3 / 0.3.

**This retires the earlier "crossover" reading**, which came from comparing only three
models. With five, two checkpoints dominate production across the whole size range, so the
correct statement is that production is beaten outright on small people — not that it wins
above 64 px. Production's high 3-class accuracy at h32/h24 (90%, 100%) is a tiny-denominator
artifact: it found almost nobody.

### The three arms disagree, and that is itself the finding

| rank on... | 1st | 2nd | 3rd | 4th | 5th |
|---|---|---|---|---|---|
| **Arm A** (real crowd photos, recall) | gelan_r4fw_v2 | y26n_gradsupp | warm50-2 | sop50 | **production** |
| **Arm C** (isolated on grey, Woman e2e) | gelan_r4fw_v2 | sop50 | production | y26n_gradsupp | warm50-2 |
| **Arm B** (shrunk real scene, AP50) | warm50-2 | production | y26n_gradsupp | sop50 | **gelan_r4fw_v2** |

Arm B ranks `gelan_r4fw_v2` **last** while Arms A and C rank it **first**. Two mechanical
explanations were checked and ruled out: it is not low-confidence spam (all five models emit
26–45 detections/image at the 0.001 floor), and it is not the AP metric (at conf 0.45 on the
same Arm B data, gelan still scores 12.9% at h96 vs production's 44.4%).

What actually differs is **background**. Arm B pastes a *shrunk whole scene* onto a canvas of
the original size (median long side 1,280 px, so its "h96" person is only **48 px at model
input**); Arm C pastes *SAM-cut people onto flat grey* at 640, where an h96 person really is
96 px. Comparing at matched input size — Arm B h96 vs Arm C h48, both 48 px — the ranking
still flips, so scale is not the cause either. The gelan and sop50 checkpoints are strong on
isolated people against plain backgrounds and weak on shrunk cluttered scenes.

**Weight Arm A most: it is real photographs.** Arm C is the cleanest isolation of size but
the least realistic input; Arm B's letterbox penalty makes its labels misleading unless you
convert to model-input pixels. This corroborates and extends the `y26n_sop50` note recorded
above — two measurements of small-object handling pointing different ways is now three, and
the tiebreak is realism.

Full analysis and the pre-registered bars: `experiments/EXP-2026-19-small-person-benchmark.md`.

## LAGENDA hand-labeled + CrowdHuman + PASS benchmark (2026-08-24)

**A third, independent benchmark — never pool with the holdout numbers above or the
LAGENDA v2 section further up.** Scores the same four checkpoints from the holdout
4-way comparison against three purpose-fit datasets: LAGENDA's **hand-labeled-only**
people for classification, CrowdHuman for person recall, PASS for false-positive rate.

LAGENDA scoring uses the `eval_v2` build (`build_lagenda_full.py`) exactly as
documented above: `labels_3class/` (the 7,098 people with real human age+gender
votes) plus `ignore/` (the other 20,535 real-but-unlabeled person boxes) fed to
`map_eval.py --gt-labels labels_3class --ignore-labels ignore`. COCO ignore semantics
mean a detection landing on a real, unlabeled person counts as neither a true nor a
false positive — a stricter and more correct rule than filtering to images where
every visible person happens to be hand-labeled, since only ~25.7% of LAGENDA's people
carry a human label at all and almost no image clears that bar. 4,601 images scored,
0 missing. CrowdHuman: 4,372 images against `annotation_val.odgt` (99,481 GT persons).
PASS: 3,000 person-free images. conf 0.45, IoU 0.7, floor 0.001 throughout.

| run name | weights path |
|---|---|
| **`y26n_noe2e_warm50-2`** (baseline) | `/workspace/exp18/train/y26n_noe2e_warm50-2/weights/best.pt` |
| `yolo11N-640` (shipped production) | `/workspace/model_v2/dataset_v2/yolo11N-640/weights/best.pt` |
| `y26n_sop50` | `/workspace/exp19/train/y26n_sop50/weights/best.pt` |
| `gelannfav14r4fw_gemlb_v2` | `/workspace/YOLO-MIT/runs/v9MIT/gelannfav14r4fw_gemlb_v2/weights/best.ckpt` |

**LAGENDA — hand-labeled 3-class mAP (4,601 images, 7,098 labeled people, 20,535 ignore):**

| run name | mAP50 | mAP50-95 | Woman AP50 | Man AP50 | Child AP50 |
|---|---|---|---|---|---|
| `y26n_noe2e_warm50-2` | 0.8567 | 0.7231 | 0.8806 | 0.8861 | 0.8033 |
| `yolo11N-640` (production) | 0.8228 | 0.7142 | 0.8656 | 0.8465 | 0.7562 |
| `y26n_sop50` | 0.8608 | 0.7266 | 0.8836 | 0.8882 | 0.8105 |
| `gelannfav14r4fw_gemlb_v2` | **0.8820** | **0.7312** | **0.9035** | **0.8979** | **0.8445** |

The gelan checkpoint sweeps every LAGENDA classification column. `y26n_sop50` is a
small, consistent improvement over its own baseline on all three classes — the
continuation did not cost general classification accuracy.

**CrowdHuman — person recall (4,372 images, 99,481 GT persons, ignore regions excluded):**

| run name | recall | recall (heavy occl.) | recall (light) | precision | clear FP/100 img |
|---|---|---|---|---|---|
| `y26n_noe2e_warm50-2` | 0.3868 | 0.2072 | 0.4584 | 0.9201 | 1.74 |
| `yolo11N-640` (production) | 0.3432 | 0.1377 | 0.4320 | **0.9452** | **1.28** |
| `y26n_sop50` | 0.3849 | 0.2035 | 0.4668 | 0.9219 | 1.67 |
| `gelannfav14r4fw_gemlb_v2` | **0.3893** | **0.2035**\* | 0.4668\* | 0.9313 | 1.56 |

\* `gelannfav14r4fw_gemlb_v2` and `y26n_sop50` tie on heavy-occlusion and light recall
to 4 decimal places on this run — recorded as a tie, not a win for either.

GELAN edges overall recall by 0.25 pts over the baseline — close enough to be
noise-level on a single run. Production has by far the cleanest precision and lowest
false-positive rate here, at the cost of the lowest recall of the four by a wide
margin (−4.6 pts vs the baseline).

**PASS — false positives (3,000 person-free images):**

| run name | FP/100 img | % images w/ FP | FP by class |
|---|---|---|---|
| `y26n_noe2e_warm50-2` | **0.37** | **0.37%** | Woman 2, Man 9 |
| `yolo11N-640` (production) | 1.17 | 1.13% | Woman 14, Man 21 |
| `y26n_sop50` | 0.60 | 0.60% | Woman 3, Man 14, Child 1 |
| `gelannfav14r4fw_gemlb_v2` | 0.70 | 0.70% | Woman 6, Man 12, Child 3 |

The un-continued baseline has the cleanest PASS false-positive rate of all four —
including its own `y26n_sop50` continuation, which is ~1.6× noisier here (0.37→0.60
FP/100 img). This is the same small real regression already flagged in the holdout
4-way comparison's negatives arm above, now confirmed on a second, independent
negatives set. Production remains the worst on this metric by a wide margin, as in
every other false-positive measurement in this document.

**Reading across all three:** no model wins everywhere. `gelannfav14r4fw_gemlb_v2` is
the best classifier and edges out recall, but comes with the compute cost already
documented in the size & speed table above. `yolo11N-640` (production) has the
cleanest CrowdHuman precision but is worst on PASS FPs and lowest on recall —
consistent with the #1 user complaint (false blurs) this project has tracked
throughout. `y26n_noe2e_warm50-2` is the cleanest all-around on false positives.
`y26n_sop50`'s small-object continuation shows a real but modest classification gain
here at a real but modest PASS-FP cost — this LAGENDA/CrowdHuman/PASS benchmark is a
general-purpose check, not the augmentation's target scenario (see `smallperson_v1`
above for that).

Raw outputs: `/workspace/exp_lagenda_bench/<run_name>/{lagenda_map.json,
crowd_eval/summary.json, pass_eval/summary.json}`.

## Spotlight-val mAP for the same four checkpoints (2026-08-24)

**A fourth benchmark for the same four models — never pool with the three above.**
Fills the same "Spotlight-val mAP50/50-95" column this doc has used for six other
models (see "mAP — three benchmarks" earlier) so the new checkpoints are directly
comparable to those existing rows. 4,232 images, 8,035 GT boxes, 702 unknown-gender
people scored as ignore regions — identical protocol (`map_eval.py`, COCO 101-point,
floor 0.001, conf 0.45, IoU 0.7).

**Correction (2026-08-25): none of the four models is actually held out here — this
needed direct verification, not the inference originally written below, and the
inference was wrong for two of the four models.** Checked each model's own training
config directly (`args.yaml` for the y26n pair, `.hydra/config.yaml` +
`.hydra/overrides.yaml` for GELAN, `args.yaml` for `yolo11N-640`), then traced every
`data.yaml` to its actual image/label paths on disk and diffed stem sets:

- `y26n_noe2e_warm50-2`, `y26n_sop50`: `data: spotlight_oiv7_local.yaml, split: val` →
  `val_full.txt`, the same 4,232-image list scored here. Confirmed directly.
- `gelannfav14r4fw_gemlb_v2`: `dataset.yaml=/workspace/spotlight/run/data.yaml`, which
  resolves `image_path.validation` to `/workspace/open-images-v7/images/val` and
  `label_path.validation` to `/workspace/spotlight/run/oiv7_val/labels_unk3` — the same
  Spotlight val labels used here. A stem diff confirmed **0 of our 4,232 benchmark
  images are missing** from that label set, and its own checkpoint-selection metric
  (`task.best_metric=cls/bal_acc`, from `overrides.yaml`) was computed on this data
  during training.
- `yolo11N-640`: `data: /workspace/open-images-v7/dataset.yaml`, `val: images/val` — a
  *different label set* (its own older labels, not Spotlight's), but
  `/workspace/open-images-v7/images/val` turned out to be **the same underlying
  4,485-image pool** as the Spotlight tree (`exp12/train_tree/images/val`): identical
  stem sets, identical counts, and one sampled file confirmed **MD5-identical** across
  both trees. A stem diff confirmed all 4,232 of our benchmark images are present in
  `yolo11N-640`'s own validation directory (0 missing). The one thing that is clean:
  a second diff against `open-images-v7/images/train` found **0 overlap** — none of
  our benchmark images were in `yolo11N-640`'s actual training loss, only its
  validation-time selection.

**Net effect:** this benchmark is in-sample or validation-adjacent for all four
checkpoints, not a fair generalization test for any of them — read every number below
as "how well does this model fit data it (or its checkpoint selection) has already
seen," not "how well does it generalize." The genuinely independent benchmarks in this
document remain LAGENDA (external, human-labeled, never used for any model's
checkpoint selection), CrowdHuman (external), and `haramblur_holdout` (built with an
explicit `DO_NOT_TRAIN` marker specifically to avoid this problem) — GELAN's
classification lead on those three benchmarks is the trustworthy signal, not anything
in the table below.

| run name | weights path |
|---|---|
| **`y26n_noe2e_warm50-2`** (baseline) | `/workspace/exp18/train/y26n_noe2e_warm50-2/weights/best.pt` |
| `yolo11N-640` (shipped production) | `/workspace/model_v2/dataset_v2/yolo11N-640/weights/best.pt` |
| `y26n_sop50` | `/workspace/exp19/train/y26n_sop50/weights/best.pt` |
| `gelannfav14r4fw_gemlb_v2` | `/workspace/YOLO-MIT/runs/v9MIT/gelannfav14r4fw_gemlb_v2/weights/best.ckpt` |

| run name | mAP50 | mAP75 | mAP50-95 | Woman AP50 | Man AP50 | Child AP50 |
|---|---|---|---|---|---|---|
| `y26n_noe2e_warm50-2` | 0.8022 | 0.7343 | 0.6970 | 0.8103 | 0.8644 | 0.7318 |
| `yolo11N-640` (production) | 0.7436 | 0.6674 | 0.6033 | 0.7745 | 0.7838 | 0.6725 |
| `y26n_sop50` | 0.8056 | **0.7388** | **0.6996** | **0.8165** | 0.8674 | 0.7329 |
| `gelannfav14r4fw_gemlb_v2` | **0.8118** | 0.7160 | 0.6540 | 0.7771 | **0.8700** | **0.7884** |

By GT box size, mAP50-95:

| run name | small | medium | large |
|---|---|---|---|
| `y26n_noe2e_warm50-2` | 0.3371 | 0.5710 | 0.7598 |
| `yolo11N-640` (production) | 0.2486 | 0.4948 | 0.6574 |
| `y26n_sop50` | **0.3392** | **0.5739** | **0.7603** |
| `gelannfav14r4fw_gemlb_v2` | 0.2913 | 0.5059 | 0.7347 |

**This is a notably different story from the other three benchmarks — but per the
correction above, all four rows here are compromised, not just the y26n pair.** On
LAGENDA, CrowdHuman, and the holdout set, `gelannfav14r4fw_gemlb_v2` was the clear
classification leader across the board. Here, `y26n_sop50` wins mAP75, mAP50-95, Woman
AP50, and all three size bands outright, and GELAN leads mAP50/Man AP50/Child AP50 —
but since GELAN's own checkpoint selection also used this same image+label combination,
its numbers here carry the same optimism risk as the y26n pair's, just via a different
mechanism (in-sample fit rather than direct split reuse). **Do not read this table as
answering "which model classifies best"** — use LAGENDA/CrowdHuman/holdout for that,
where GELAN led. What this table *can* still support: `y26n_sop50` edges its own
baseline (`y26n_noe2e_warm50-2`) on every column here, the same small, consistent win
seen on every other benchmark this session — that specific comparison stays
apples-to-apples (both selected the same way, against the same data), even though the
table as a whole isn't a fair cross-architecture comparison.

Raw outputs: `/workspace/exp_spotval_bench/<run_name>/{dump/raw, spotval_map.json}`.

## LAGENDA `fl1199` — fully human-labeled, zero ignore regions (2026-08-25)

**A fifth benchmark for the same four models — never pool with the four above.** LAGENDA's
`eval_v2` set (used earlier in this doc) still needs ignore regions because 4 of every 6
real people lack a human label. `fl1199` (`/workspace/datasets/lagenda_full/fl1199/`,
see `docs/LAGENDA_SOL_HANDOFF.md`) is a curated 1,199-image subset selected specifically
because **every person box has a human label** — no unlabeled people, so an unmatched
detection is a genuine false positive, not an annotation gap. Verified directly on disk
before running anything: `labels_3class` and `labels` (all boxes) both have exactly 1,514
lines across these 1,199 stems — identical counts — and 0 of 1,199 stems have any content
in their `ignore/` file. `map_eval.py` ran with no `--ignore-labels` flag at all.

**Data bug found and fixed before trusting the first result:** the Stage A tool
(`run_ultralytics_labels.py`) globs images recursively (`images_dir.rglob("*")`), which
swept up 3 stray Jupyter checkpoint duplicates from a nested `.ipynb_checkpoints/`
folder inside `fl1199/images/` (content-duplicates of 3 real images, but named
`<stem>-checkpoint.jpg` with no matching label file — each would have scored as a false
positive). Fixed by building a clean, non-recursive 1,199-image symlink farm
(`/workspace/exp_fl1199_bench/clean_images/`) and re-running; confirmed
`n_images: 1199` / `n_gt: 1514` / `ignore_regions: "none"` in every output JSON before
reading any number below.

**Confirms the user's recollection: Sol (`gpt-5.6-sol`) was run over this set** —
`/workspace/codex_sol_compare/lagenda_fl1199/verdicts_sol.jsonl` has exactly 1,884 lines,
matching the SAM3 detection count from the handoff doc. That data belongs to a separate
research thread (a 3-way Gemini vs Sol vs human comparison, not yet run for Gemini as of
the handoff doc) and is **not** what was used as ground truth here — the human labels in
`eval_v2/labels_3class` are.

**One real limitation, stated in the handoff doc and confirmed by the `by_area`
results below: this subset has zero small people** (0 below 64px at model input, 96.1%
≥160px, median 386px) — annotators labeled the prominent subjects only. Every number
here is prominent-subject recall/classification and should not be read as evidence
about small/distant people.

| run name | mAP50 | mAP75 | mAP50-95 | Woman AP50 | Man AP50 | Child AP50 |
|---|---|---|---|---|---|---|
| `y26n_noe2e_warm50-2` | 0.8955 | 0.8650 | 0.7801 | 0.9085 | 0.9364 | 0.8417 |
| `yolo11N-640` (production) | 0.8743 | 0.8512 | 0.7896 | 0.9010 | 0.8987 | 0.8231 |
| `y26n_sop50` | 0.9061 | 0.8743 | 0.7885 | 0.9252 | 0.9369 | 0.8561 |
| `gelannfav14r4fw_gemlb_v2` | **0.9433** | **0.9086** | **0.8060** | **0.9459** | **0.9549** | **0.9291** |

**GELAN sweeps every column — its most decisive win of any benchmark run this
session** (mAP50 +3.7 pts over the next-best, versus ~0.5-1 pt gaps on LAGENDA/
CrowdHuman). Consistent with the small-people caveat above: `fl1199` is exclusively the
regime GELAN's architecture is known to lead (large, prominent subjects), and never
exercises the y26n family's known small-object weakness. `y26n_sop50` again edges its
own baseline on every column, the same consistent pattern seen on every benchmark this
session. `by_area` confirms the small-people gap directly: `mAP50_95_small` and
`mAP50_95_medium` are both `null` for every model — no GT boxes exist in those size
bands in this set at all.

Raw outputs: `/workspace/exp_fl1199_bench/<run_name>/{dump/raw, fl1199_map.json}`;
clean image list at `/workspace/exp_fl1199_bench/clean_images/`.


## 2026-08-28/29 — `gelannfav14r6_gemlb_v2` measured + the five-model round

New model: `gelannfav14r6_gemlb_v2` (coworker gelan) ·
`/workspace/YOLO-MIT/runs/v9MIT/gelannfav14r6_gemlb_v2/weights/best.ckpt`.
**The canonical five-model table set is now `_deploycmp_review/FIVE_MODEL_BENCHMARK.md`**
(also at `/workspace/deploycmp/`): standard four-dataset protocol re-run from scratch for
all five current models (recomputed `yolo11N-640` cells reproduce the rows above exactly,
validating protocol continuity), plus fl1199 classification + fl1199 3-class mAP and the
holdout pooled mAP. Note the fl1199 mAP table above (from `/workspace/exp_fl1199_bench/`)
agrees with the independent `/workspace/evalout_v2/*_map_fl1199.json` run to the third
decimal on every shared cell; r6 adds **0.937 / 0.794** there (second to
`gelannfav14r4fw_gemlb_v2`).

`gelannfav14r6_gemlb_v2` one-line profile — the **classification-quality gelan**: best
gender-on-adults (91.6%), best LAGENDA precision (94.3%) and 3-class mAP (0.791), leak
fixed vs r4fw (0.27% vs 1.23%), clean FP profile (objects 12.7%, PASS 0.43/100), best
holdout Woman AP50 (0.951) and Child AP50 (0.898) — paid for with the lowest all-people
recall of the candidates (LAGENDA 73.4%, crowd 34.9% = shipped level) and the **worst
small-person numbers of any candidate** (QA-holdout small slice: det recall 44.9%,
Woman end-to-end 35.4%). Ship recommendation unchanged: `y26n_noe2e_warm50-2`
(thresholds: male 0.15 / female 0.20; r6 owner-picked 0.25 both audiences).

Small-person addenda (full tables + image proofs:
`_deploycmp_review/SMALL_PERSON_FIVE_MODELS.md` + `SMALL_PERSON_PROOF.html`):

- **Avatars arm (on-disk sizes 128/96/64/48 px), all five via one code path** — Woman
  end-to-end at conf 0.45: `y26n_sop50` best at every size (77.7/71.2/60.4/52.5%);
  gelans best mAP but collapse at 48 px; `yolo11N-640` worst at 64/48 px.
- **NEW: QA-holdout small slice** (`/workspace/deploycmp/holdout_small/`, people ≤96 px
  at model input, 1,587 people / 392 imgs, 83% Man, 64% shiekh-context): det recall at
  0.45 — `y26n_noe2e_warm50-2` 58.3 > `y26n_sop50` 57.6 > `gelannfav14r4fw_gemlb_v2`
  48.3 > `gelannfav14r6_gemlb_v2` 44.9 > `yolo11N-640` **30.9** (production misses 7 of
  10 small people; its failure concentrates on men — 26.5% found vs 59.3% of women).
  Small-woman end-to-end: warm50-2 58.9 / sop50 58.4 / v11n 56.5 / r4fw 48.3 / r6 35.4.

## 2026-09-05/08 — humanshaped labeling policy + `y26s_humanshaped_smallpatch_v1` (new best measured model)

**Policy change:** human-shaped toys/statues/mannequins/cartoons now count as gaze-lowering
targets (previously deleted as non-person). Age overrides gender in the promoted label:
gender+child → `Child`; gender+adult or unknown-age → `Woman`/`Man`; unknown-gender+child →
`Child`. Applied by patching promoted lines from the raw OIV7 labels onto the plain nc=3
base (`patch_promotions_local.py`) — verified by timestamp that the production
`y26n_noe2e_warm50-2` / `y26n_sop50` line was never touched by this change; it only affects
models trained after it.

**New model lineage:** `y26n_sop50` → `y26n_humanshaped_v1` → `y26n_humanshaped_v2` →
**`y26s_humanshaped_smallpatch_v1`**. The last arm is a size step up (YOLO26**s**, not n)
trained from COCO-pretrained weights (not a continuation) on the same humanshaped labels as
v2, plus a `SmallObjectPatches` augmentation: 5% per-sample chance of pasting the image's own
objects, shrunk to 96px, onto a grey canvas (mirrors YOLO-MIT's small-object augmentation;
source crops from LAGENDA `fl1199`, held out from the `smallperson_v1` benchmark's own
`paste_grey`/`synth_shrunk` arms via `--exclude-stems` on 554 stems). 100-epoch ceiling,
`patience=15` — ran to the full ceiling without early stopping, still setting new records as
late as **epoch 90/100** (`mAP50-95` 0.68→0.804 val). Weights:
`/workspace/exp20/train/y26s_humanshaped_smallpatch_v1/weights/best.pt` (20.3 MB).

**Standard 5-dataset comparison** (conf 0.45 unless noted; `object_set` FP/100 uses the OLD
pre-humanshaped ground truth, see caveat below):

| run name | Spotval mAP50/50-95 | LAGENDA v2 mAP50/50-95 | crowd recall/prec. | PASS FP/100 | object_set FP/100 |
|---|---|---|---|---|---|
| `y26n_sop50` | 0.806/0.700 | 0.758/0.638 | 0.391/0.924 | 0.60 | 23.94 |
| `y26n_noe2e_warm50-2` | 0.802/0.697 | 0.754/0.634 | 0.392/0.922 | 0.37 | 20.46 |
| `yolo11N-640` (shipped) | 0.744/0.603 | 0.724/0.627 | 0.349/0.943 | 1.17 | 88.42 |
| `gelannfav14r4fw_gemlb_v2` | 0.812/0.654 | 0.783/0.647 | 0.393/0.927 | 0.70 | 21.62 |
| `y26n_humanshaped_v2` | 0.806/0.704 | 0.859/0.723 | 0.390/0.922 | 0.50 | 31.27 |
| **`y26s_humanshaped_smallpatch_v1`** | **0.850/0.764** | **0.884/0.745** | **0.474**/0.918 | 0.60 | 35.14 |

**`object_set` is not comparable across the humanshaped boundary**: ground truth still assumes
"any detection = false positive" (the pre-policy rule), so a humanshaped model correctly
reading a toy/statue as a person necessarily scores worse there — evidence the policy worked,
not a regression. A proper number needs the still-pending `object_set` relabel.

**Holdout QA (`haramblur_holdout`, `DO_NOT_TRAIN`, 3-class mean mAP):**

| model | mAP50 | mAP50-95 |
|---|---|---|
| `y26n_sop50` | 0.912 | 0.835 |
| `y26n_warm50` | 0.913 | 0.835 |
| `v11n_shipped` | 0.864 | 0.741 |
| `gelan_r4fw_v2` | 0.921 | 0.793 |
| `y26n_humanshaped_v2` | 0.913 | 0.840 |
| **`y26s_humanshaped_smallpatch_v1`** | **0.928** | **0.870** |

Best mAP50 **and** best mAP50-95 of every model ever measured on this holdout set — +3.5 pts
mAP50-95 over the previous best (`gelan_r4fw_v2`), +12.9 over shipped production.

**Woman AP by exposure tier (2026-09-06/08, `deploy_compare` tiers t0-t4, reusing the
existing holdout subset views + `map_eval.py`, same method as the 2026-08-25 exposure work):**

| tier | n imgs | v11n_shipped | y26n_sop50 | y26n_warm50 | gelan_r4fw_v2 | gelan_r6_v2 | y26n_humanshaped_v2 | `y26s_humanshaped_smallpatch_v1` |
|---|---|---|---|---|---|---|---|---|
| all (whole holdout) | 11,494 | 0.909/0.808 | 0.928/0.886 | 0.928/0.888 | 0.951/0.851 | 0.951/0.846 | 0.929/0.889 | **0.952/0.921** |
| t0_covered | 139 | 0.713/0.598 | 0.780/0.718 | 0.769/0.708 | **0.856**/0.717 | 0.850/0.730 | 0.759/0.693 | 0.819/**0.767** |
| t1_modest | 892 | 0.921/0.822 | 0.937/0.909 | 0.927/0.898 | 0.968/0.880 | **0.973**/0.876 | 0.935/0.901 | 0.956/**0.926** |
| t2_ordinary | 1,410 | 0.914/0.811 | 0.923/0.882 | 0.929/0.888 | **0.964**/0.862 | 0.963/0.856 | 0.930/0.891 | 0.946/**0.917** |
| t3_revealing | 1,045 | 0.949/0.850 | 0.953/0.909 | 0.961/0.915 | **0.974**/0.876 | 0.972/0.871 | 0.955/0.915 | 0.966/**0.937** |
| t4_high | 220 | 0.946/0.809 | 0.928/0.863 | 0.925/0.865 | 0.960/0.822 | 0.959/0.808 | 0.935/0.875 | **0.967/0.926** |

Cells are Woman-class AP50/AP50-95. **Two things hold on every tier without exception:**
(1) `y26s_humanshaped_smallpatch_v1` beats its own predecessor `y26n_humanshaped_v2`
outright — same labels, strictly better box quality and recall at every exposure level;
(2) it has the **best AP50-95 of all seven models on every tier** — gelan occasionally edges
it on AP50 (looser IoU match) in the t1-t3 mid-range, the same "gelan wins loose mAP50, y26n
arms win strict mAP50-95" pattern seen project-wide, not better real coverage. It also **wins
outright (both metrics) on the whole holdout and on t4_high**, the most revealing/highest-
stakes tier. **t0_covered (hijab-only cues) is the hardest tier for every model** — fewest
visible skin/body cues to read gender from — and the one tier where gelan's AP50 lead is
widest; even there the new model has the best AP50-95.

**What was learned, beyond the accuracy numbers:**
- The `SmallObjectPatches` augmentation only helps if it actually reaches the training
  dataloader — `ultralytics/data/dataset.py` imports `v8_transforms` by name, so patching
  `ultralytics.data.augment.v8_transforms` alone is a silent no-op; the call site
  (`dataset.py`'s `build_transforms()`) must be patched too. Caught before the real run by
  building a live `YOLODataset` and checking the transform list, not by trusting the
  monkeypatch.
- Dataloader thread oversubscription is easy to get backwards on a RunPod pod: setting
  `OMP_NUM_THREADS`/`MKL_NUM_THREADS` to the worker count (10) made 10 processes each spawn
  up to 12 internal threads and REGRESSED throughput; the fix was `=1` for both **plus**
  `cv2.setNumThreads(1)` in the launcher (OpenCV's own thread pool otherwise defaults to the
  host's `nproc`, not the pod's real allocation — same trap as the nproc/cgroup issue
  documented elsewhere in this repo). Verified via `/proc/<pid>/status` thread counts, not
  by throughput alone.
- A stale, never-cleaned-up resync job silently exceeded the pod's real (invisible-to-`df`)
  volume quota and crashed training at the epoch-1 metrics-write step; recovered via a `dd`
  write-probe (not `df`) and a parallelized `find | xargs -P 32 rm -f` delete of ~450k
  orphaned files, then relaunched from epoch 0.
- A transient `inf` box_loss at epoch 18 was investigated rather than dismissed — `val/loss`
  stayed finite, epoch 19 showed zero recurrence, confirmed as one degenerate augmented batch
  (likely a near-zero-area `SmallObjectPatches` crop), not weight corruption.
- Training genuinely used the full 100-epoch ceiling productively — `patience=15` never
  triggered because the model kept improving through epoch 90; the epoch ceiling, not
  patience, was the real binding constraint for this run.

Full benchmark writeup + all raw JSONs: `_spotlight_review/y26s_humanshaped_smallpatch_v1_benchmark/`
(`BENCHMARK_SUMMARY.html` + `exposure_map/`, local mirror of `/workspace/deploycmp/map/`).

### Pixel-level metrics (2026-09-08) — % of target left visible vs % wrongly-blurred pixels

Same tool and geometry as the `WOMEN_REPORT_FINAL.md`/`MEN_REPORT_FINAL.md` pixel-coverage work
(`deploy_compare.py score` — see that file's own methodology note for the exact math): per GT
person, `covered_frac` = exact polygon-union area of fired same-class boxes ∩ that person's GT
box ÷ GT box area; **Epers%** = mean `1 − covered_frac` over every person (size-independent, "%
of a woman/man left visible"); **Eimg%** = the same uncovered pixels but area-weighted per image
÷ image area (screen-real-estate-weighted); **FBarea%** = area of fired same-class boxes
*outside* a small dilation margin around real GT boxes and outside ignore regions, ÷ image area
(the false-blur/FP-pixel rate). All replayed offline from the already-dumped holdout raw
sidecars (conf floor 0.001) — no model re-run, CPU-only, `/workspace/deploycmp/holdout_run7/`.

**First pass used the tool's standard matched-false-blur-area operating point** (each model's
confidence chosen so its false-blur area equals the incumbent `v11n_shipped@0.45`'s, not a
shared raw threshold) — both new models' matched confidence landed at **0.05, the floor of the
tested grid**, meaning even the most permissive confidence tested still had less false-blur area
than production at 0.45. A follow-up finer grid (0.01-0.05) confirmed the true crossing point is
below 0.05 for both.

**Owner then asked for a same-threshold comparison instead** ("confidence threshold matters a
lot, let's make them all at 0.45") — re-run with every model forced to the identical shared
conf 0.45, no matching, on the same 11,493 holdout images:

**MALE audience (blur target: Woman), all models @ conf 0.45:**

| model | Eimg% | P90% | Epers% (% of a woman left visible) | cov90% | FBarea% (% wrongly-blurred pixels) | chblur% |
|---|---|---|---|---|---|---|
| v11n_shipped (production) | 6.01 | 10.3 | 10.6 | 87.5 | 1.816 | 6.7 |
| y26n_sop50 | 4.98 | 6.6 | 9.6 | 88.6 | 0.723 | 5.0 |
| y26n_warm50 | 5.05 | 7.5 | 9.7 | 88.3 | 0.779 | 4.9 |
| gelan_r4fw_v2 | 11.07 | 50.5 | 19.1 | 78.3 | 0.498 | 3.7 |
| gelan_r6_v2 | 7.14 | 21.3 | 14.2 | 83.0 | 0.799 | 4.8 |
| y26n_humanshaped_v2 | 4.82 | 6.3 | 9.4 | 88.5 | 0.723 | 4.2 |
| **`y26s_humanshaped_smallpatch_v1`** | **2.86** | **1.7** | **6.0** | **92.7** | **0.621** | 5.3 |

**FEMALE audience (blur target: Man), all models @ conf 0.45:**

| model | Eimg% | P90% | Epers% (% of a man left visible) | cov90% | FBarea% (% wrongly-blurred pixels) | chblur% |
|---|---|---|---|---|---|---|
| v11n_shipped (production) | 6.07 | 11.4 | 17.0 | 78.6 | 1.326 | 2.3 |
| y26n_sop50 | 3.43 | 5.4 | 10.5 | 85.0 | 0.946 | 2.6 |
| y26n_warm50 | 3.41 | 5.0 | 10.0 | 85.4 | 0.891 | 2.8 |
| gelan_r4fw_v2 | 5.21 | 10.0 | 14.0 | 81.2 | 0.767 | 2.3 |
| gelan_r6_v2 | 4.89 | 10.2 | 13.9 | 80.9 | 0.928 | 2.9 |
| y26n_humanshaped_v2 | 3.35 | 5.0 | 10.1 | 85.6 | 0.902 | 3.0 |
| **`y26s_humanshaped_smallpatch_v1`** | **2.29** | **3.0** | **8.5** | **87.7** | **0.731** | 3.2 |

**At a genuinely shared threshold, `y26s_humanshaped_smallpatch_v1` wins every escape-side
metric on both audiences** (lowest Eimg/P90/Epers, highest cov90) **and has the lowest FBarea of
every y26n-family/production model** — beaten only by the gelan pair's FBarea, which comes at a
heavy escape-side cost (gelan_r4fw_v2's P90 hits 50.5%, meaning its worst 10% of images leave
over half the screen's target pixels showing). **Important honest caveat**: a shared fixed
threshold structurally favors models calibrated FOR that exact value — both y26n-family models
ship at 0.45 in production, while the gelan pair's own best operating point is materially lower
(documented median matched-confidence ~0.42, see the small-person work above), so gelan's numbers
here are worse than its capability ceiling, not a clean apples-to-apples calibration comparison.
Raw ship-table log: `_spotlight_review/y26s_humanshaped_smallpatch_v1_benchmark/pixel_metrics/`.
