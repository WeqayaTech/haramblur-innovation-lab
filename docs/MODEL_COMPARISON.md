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

**⚠️ CORRECTION (2026-09-09): the LAGENDA v2 column below is INVALID as a cross-model
comparison** — the two humanshaped models were scored on a different (smaller, corrected)
LAGENDA image set than every other model in this table, which alone explains most of the
apparent LAGENDA lead. See "LAGENDA benchmark-set mismatch found + corrected" near the end of
this document for the root cause and the full corrected table (every model gains ~10-13 mAP50
points once matched fairly, and the humanshaped models are no longer LAGENDA leaders). Every
OTHER column in the table below (Spotval, crowd, PASS, object_set) is unaffected — this bug was
specific to the LAGENDA dump.

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

## TFLite export size & CPU latency — fp32 vs INT8 (2026-09-08)

*Correction 2026-09-09: the INT8 files in this section were calibrated on the full Spotlight-val split (inherited exporter default). Sizes and latencies stand; for INT8 **accuracy** use the train-calibrated matrix in the last section of this file, "INT8 quantization accuracy matrix — train-calibrated, corrected (2026-09-09)".*

Two YOLO26n candidates for an export decision: **`y26n_humanshaped_v2`** (the newest
*complete* checkpoint under the current humanshaped-labeling policy,
`/workspace/exp20/train/y26n_humanshaped_v2/weights/best.pt`) and **`y26n_noe2e_warm50-2`**
(the pre-humanshaped-policy holdout mAP champion, `/workspace/exp18/train/y26n_noe2e_warm50-2/
weights/best.pt`). The two *newer* exp20 checkpoints (`y26n_humanshaped_smallpatch_v1`,
`y26n_humanshaped_v2_distill_v1`) were ruled out first: both stalled at epoch 2-3/100 when their
training pod was torn down mid-run and are not real candidates.

Both exported to `.tflite` with `yolo export format=tflite int8=True imgsz={640,416,320}
data=/workspace/exp12/spotlight_oiv7.yaml device=cpu` (INT8 calibrated on real Open Images v7 val
images, dynamic-range: float32 I/O, int8 weights), Ultralytics 8.4.144. `y26n_humanshaped_v2`'s
bundle already existed from a 2026-09-02 session; `y26n_noe2e_warm50-2` had no INT8 export
before this session — that gap is what this pass filled. Full bundles + READMEs + SHA256SUMS:
`models/humanshaped_v2_20260902/`, `models/noe2e_warm50-2_20260908/` (local repo, gitignored
binaries, reproducible from the pod paths above).

**Known export bug**: `y26n_noe2e_warm50-2`'s first `int8=True imgsz=640` attempt failed with
`AttributeError: '_OpNamespace' 'aten' object has no attribute 'cholesky'` (torch/onnx2tf gap);
an identical retry succeeded with no code change. Treat as transient, but budget one retry.

Latency measured with `vlm-cluster/bench_tflite.py` (`ai_edge_litert.Interpreter`, 4 threads, 20
warmup + 100 runs, median/p90) — **a different runtime AND different CPU from the ONNX/EPYC-9254
numbers in the "Size & speed" section above; do not cross-compare the two tables.**

| model | size | fp32 size | fp32 median / p90 | int8 size | int8 median / p90 |
|---|---|---|---|---|---|
| `y26n_humanshaped_v2` | 640 | 9.84 MB | 21.3 ms / 25.7 ms | 2.89 MB | 27.6 ms / 33.6 ms |
| `y26n_humanshaped_v2` | 416 | 9.78 MB | 9.5 ms / 10.3 ms | 2.87 MB | 9.7 ms / 10.1 ms |
| `y26n_humanshaped_v2` | 320 | 9.76 MB | 6.0 ms / 6.4 ms | 2.87 MB | **5.5 ms** / 5.9 ms |
| `y26n_noe2e_warm50-2` | 640 | 9.84 MB | **20.5 ms** / 24.5 ms | 2.89 MB | 27.3 ms / 27.6 ms |
| `y26n_noe2e_warm50-2` | 416 | 9.78 MB | 9.4 ms / 9.9 ms | 2.87 MB | 9.3 ms / 10.7 ms |
| `y26n_noe2e_warm50-2` | 320 | 9.76 MB | 6.1 ms / 6.8 ms | 2.87 MB | **5.6 ms** / 6.0 ms |

The two checkpoints are latency-twins (same architecture family, as expected). **The finding
that matters for the export decision: INT8 is not a free speedup.** It is a consistent ~3.4x
size reduction at every resolution, but at 640px it is ~30% *slower* than fp32 (dynamic-range
quantization's dequant overhead isn't hidden by XNNPACK at that resolution); at 416px it's a
wash; only at 320px does int8 edge out fp32 on speed. Size and speed are two separate levers
here — resolution (not precision) is the dominant speed lever.

## Threshold sweep — CrowdHuman recall/precision, PASS false positives (2026-09-08)

Offline confidence-threshold sweep via `vlm-cluster/conf_sweep.py` (replays already-dumped
floor-0.001 raw sidecars, no re-run) for the same two checkpoints, global (shared) threshold,
grid `0.05:0.95:0.01`, objective `@crowd_f1`. **CrowdHuman arm** (`--crowd`, exhaustively-boxed
GT, `annotation_val.odgt`, 4,372 images / 99,481 GT persons) gives real recall AND precision
directly; **PASS arm** (`--negatives pass:`, 3,000 person-free images) gives the false-persons
rate. Sidecars reused from existing dumps: `/workspace/mapdump/y26n_humanshaped_v2/{crowd,pass}/
raw`, `/workspace/exp_lagenda_bench/y26n_noe2e_warm50-2/{crowd,pass}/raw`.

| model | threshold | recall | precision | PASS img-FP rate |
|---|---|---|---|---|
| `y26n_humanshaped_v2` | 0.45 (current) | 0.390 | 0.922 | 0.47% |
| `y26n_humanshaped_v2` | 0.18 (best F1) | 0.580 | 0.718 | 2.43% |
| `y26n_noe2e_warm50-2` | 0.45 (current) | 0.387 | 0.920 | 0.37% |
| `y26n_noe2e_warm50-2` | 0.19 (best F1) | 0.567 | 0.731 | 2.10% |

**The two models are statistically near-identical on this sweep** — same recall/precision curve
shape, same trade at the same thresholds, well within run-to-run noise for a single-run number.
Lowering the threshold from 0.45 to ~0.18-0.19 buys +18-19 recall points at a cost of -19-20
precision points and a ~5x rise in PASS false-persons — a real trade, not a free lunch; this
sweep does not by itself recommend moving off 0.45. Full sweep tables (every threshold from 0.05
to 0.95), SVG charts, and self-contained HTML reports: `models/threshold_sweep_20260908/
{y26n_humanshaped_v2,y26n_noe2e_warm50-2}/` (local repo).

**Net read on the export decision**: these two checkpoints don't differentiate on latency, export
size, or the recall/precision/FP curve — the decision between them should rest on the
classification-accuracy numbers already in this document (holdout mAP, LAGENDA/CrowdHuman/PASS
above), not on export mechanics.

## LAGENDA classification sweep — recall & precision per class, IoU>=0.5 (2026-09-08)

The CrowdHuman/PASS sweep above answers "did you find a person"; this one answers "did you get
the *class* right" — the metric that actually matters for a gaze-lowering blur (an adult found
but mislabeled still escapes the blur). Same tool (`vlm-cluster/conf_sweep.py`), same two
checkpoints, but the `--lagenda` arm against LAGENDA v2's human-labeled ground truth
(`/workspace/datasets/lagenda_full/eval_v2/{gt.jsonl,labels}`, 4,601 images / 7,098 human
age+gender-labeled people), `--match-iou 0.5`, **0.05 grid increments** (0.05 to 0.95, 19
points, as requested — finer than the 0.01 grid used for the CrowdHuman sweep above). Sidecars
reused from the same floor-0.001 dumps as before: `/workspace/mapdump/y26n_humanshaped_v2/
lagenda/raw`, `/workspace/exp_lagenda_bench/y26n_noe2e_warm50-2/lagenda/raw`.

Columns: `det_recall` = labeled people found at all; `acc3` = 3-class accuracy on matched
people (the direct "did you classify correctly" number); `gender_acc` = Woman/Man accuracy on
matched adults; `W/M/C_recall` = **recall_e2e**, GT people of that class found AND written the
right label (an unmatched or mislabeled person counts against this); `W/M/C_prec` =
**precision_matched**, of matches the model wrote as that class, how many really are (LAGENDA is
label-anchored, so this is a matched-only proxy, not a true FP-based precision — see the
CrowdHuman/PASS sweep above for that); `leak` = matched GT age>=20 written 'Child' (adults
escaping the blur).

**`y26n_humanshaped_v2`:**

| thr | det_recall | acc3 | gender_acc | W_recall | W_prec | M_recall | M_prec | C_recall | C_prec | leak |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.995 | 0.881 | 0.903 | 0.897 | 0.878 | 0.898 | 0.908 | 0.816 | 0.848 | 0.004 |
| 0.10 | 0.994 | 0.881 | 0.903 | 0.896 | 0.877 | 0.897 | 0.908 | 0.815 | 0.849 | 0.004 |
| 0.15 | 0.992 | 0.882 | 0.904 | 0.896 | 0.877 | 0.896 | 0.909 | 0.816 | 0.850 | 0.004 |
| 0.20 | 0.991 | 0.882 | 0.905 | 0.895 | 0.878 | 0.895 | 0.910 | 0.815 | 0.850 | 0.004 |
| 0.25 | 0.986 | 0.883 | 0.906 | 0.891 | 0.880 | 0.893 | 0.911 | 0.812 | 0.850 | 0.004 |
| 0.30 | 0.984 | 0.884 | 0.906 | 0.889 | 0.881 | 0.893 | 0.913 | 0.810 | 0.849 | 0.004 |
| 0.35 | 0.979 | 0.886 | 0.907 | 0.884 | 0.883 | 0.893 | 0.915 | 0.806 | 0.850 | 0.003 |
| 0.40 | 0.974 | 0.888 | 0.909 | 0.883 | 0.885 | 0.890 | 0.919 | 0.807 | 0.851 | 0.003 |
| **0.45** | **0.965** | **0.892** | **0.911** | **0.879** | **0.887** | **0.884** | **0.923** | **0.803** | **0.854** | **0.003** |
| 0.50 | 0.952 | 0.896 | 0.916 | 0.868 | 0.892 | 0.881 | 0.928 | 0.795 | 0.857 | 0.003 |
| 0.55 | 0.933 | 0.903 | 0.922 | 0.857 | 0.900 | 0.875 | 0.935 | 0.778 | 0.863 | 0.002 |
| 0.60 | 0.911 | 0.908 | 0.926 | 0.841 | 0.906 | 0.859 | 0.939 | 0.767 | 0.868 | 0.002 |
| 0.65 | 0.881 | 0.916 | 0.932 | 0.820 | 0.912 | 0.837 | 0.946 | 0.744 | 0.878 | 0.002 |
| 0.70 | 0.845 | 0.925 | 0.939 | 0.790 | 0.924 | 0.817 | 0.954 | 0.720 | 0.884 | 0.001 |
| 0.75 | 0.800 | 0.937 | 0.948 | 0.759 | 0.935 | 0.786 | 0.964 | 0.685 | 0.900 | 0.001 |
| 0.80 | 0.753 | 0.948 | 0.957 | 0.725 | 0.944 | 0.755 | 0.973 | 0.643 | 0.917 | 0.001 |
| 0.85 | 0.683 | 0.958 | 0.966 | 0.659 | 0.955 | 0.705 | 0.979 | 0.577 | 0.929 | 0.001 |
| 0.90 | 0.570 | 0.970 | 0.977 | 0.559 | 0.970 | 0.605 | 0.986 | 0.477 | 0.946 | 0.000 |
| 0.95 | 0.323 | 0.990 | 0.992 | 0.316 | 0.990 | 0.376 | 0.997 | 0.251 | 0.977 | 0.000 |

**`y26n_noe2e_warm50-2`:**

| thr | det_recall | acc3 | gender_acc | W_recall | W_prec | M_recall | M_prec | C_recall | C_prec | leak |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.994 | 0.878 | 0.904 | 0.897 | 0.870 | 0.898 | 0.901 | 0.804 | 0.858 | 0.003 |
| 0.10 | 0.993 | 0.878 | 0.904 | 0.895 | 0.871 | 0.896 | 0.902 | 0.806 | 0.857 | 0.003 |
| 0.15 | 0.991 | 0.879 | 0.905 | 0.895 | 0.872 | 0.894 | 0.903 | 0.806 | 0.857 | 0.003 |
| 0.20 | 0.989 | 0.881 | 0.906 | 0.895 | 0.873 | 0.893 | 0.905 | 0.806 | 0.859 | 0.003 |
| 0.25 | 0.986 | 0.882 | 0.907 | 0.892 | 0.875 | 0.892 | 0.906 | 0.808 | 0.860 | 0.003 |
| 0.30 | 0.984 | 0.883 | 0.908 | 0.892 | 0.877 | 0.891 | 0.908 | 0.807 | 0.858 | 0.003 |
| 0.35 | 0.980 | 0.885 | 0.908 | 0.890 | 0.878 | 0.890 | 0.911 | 0.805 | 0.860 | 0.003 |
| 0.40 | 0.972 | 0.888 | 0.910 | 0.886 | 0.882 | 0.888 | 0.913 | 0.796 | 0.861 | 0.003 |
| **0.45** | **0.964** | **0.891** | **0.914** | **0.882** | **0.884** | **0.887** | **0.918** | **0.787** | **0.864** | **0.003** |
| 0.50 | 0.951 | 0.895 | 0.917 | 0.873 | 0.887 | 0.879 | 0.923 | 0.781 | 0.866 | 0.003 |
| 0.55 | 0.934 | 0.900 | 0.922 | 0.860 | 0.891 | 0.871 | 0.931 | 0.772 | 0.870 | 0.002 |
| 0.60 | 0.907 | 0.906 | 0.927 | 0.843 | 0.895 | 0.854 | 0.939 | 0.748 | 0.876 | 0.002 |
| 0.65 | 0.877 | 0.912 | 0.932 | 0.819 | 0.902 | 0.832 | 0.945 | 0.729 | 0.882 | 0.002 |
| 0.70 | 0.843 | 0.921 | 0.940 | 0.794 | 0.910 | 0.813 | 0.953 | 0.701 | 0.891 | 0.001 |
| 0.75 | 0.801 | 0.932 | 0.947 | 0.766 | 0.925 | 0.784 | 0.960 | 0.667 | 0.901 | 0.001 |
| 0.80 | 0.748 | 0.946 | 0.958 | 0.724 | 0.941 | 0.750 | 0.971 | 0.628 | 0.916 | 0.001 |
| 0.85 | 0.678 | 0.959 | 0.967 | 0.656 | 0.956 | 0.701 | 0.979 | 0.573 | 0.932 | 0.001 |
| 0.90 | 0.562 | 0.975 | 0.981 | 0.551 | 0.973 | 0.601 | 0.988 | 0.471 | 0.958 | 0.000 |
| 0.95 | 0.315 | 0.989 | 0.992 | 0.311 | 0.988 | 0.378 | 0.998 | 0.220 | 0.972 | 0.000 |

**Reading it**: the classic threshold trade is fully visible here — every class's recall falls
and its precision-matched rises monotonically as the threshold climbs, from (0.05: recall ~0.90,
precision ~0.85-0.91) to (0.95: recall ~0.22-0.38, precision ~0.97-1.00). `acc3` and
`gender_acc_adults` *rise* with threshold too, but that is a **survivorship effect, not a real
accuracy gain** — at 0.95 almost nothing survives to be scored (`n_matched` collapses), and the
few detections that do are the easy, high-confidence ones; it is not evidence that the model
"classifies better" at a stricter cut. **Again the two checkpoints track each other almost
exactly at every threshold** (largest gap: Child recall at low thresholds, `y26n_humanshaped_v2`
+1-3pts over `y26n_noe2e_warm50-2`) — this sweep does not separate them either. Full 19-point
CSVs/JSONs/SVGs (plus per-class response charts): `models/threshold_sweep_20260908/
{y26n_humanshaped_v2,y26n_noe2e_warm50-2}_lagenda/`.

### Real per-class precision/recall curves + F1-optimal threshold (2026-09-09)

The table above uses `conf_sweep.py`'s LAGENDA arm, whose "precision" is a **matched-only
proxy** — it can't see a false detection on a person LAGENDA never labeled. `vlm-cluster/
pr_curve.py` (new, imports `map_eval.py`'s loading/matching so it stays consistent with every
AP number in this doc) computes the **real, FP-based per-class precision/recall curve** instead
— the same convention as CrowdHuman/PASS, but per class: `--gt-labels labels_3class
--ignore-labels ignore` against the same LAGENDA v2 set, IoU>=0.5, one curve point per detection
confidence (not just the 19-point grid). This is the direct analogue of Ultralytics'
`metrics.box.px`/`rx`, built from this repo's own scorer instead of `model.val()`.

**Interactive precision-vs-recall chart (hover any point for its exact threshold):**
[claude.ai/code/artifact/24297c94-7dde-46be-900d-94487aa54e8b](https://claude.ai/code/artifact/24297c94-7dde-46be-900d-94487aa54e8b)
— three panels (Woman/Man/Child), both checkpoints overlaid, 0.05-increment markers, the 0.45
baseline ringed, full 19-threshold data table included. Raw curves (thousands of points per
class, full precision):
`models/pr_curves_20260909/{y26n_humanshaped_v2,y26n_noe2e_warm50-2}.json`.

**F1-optimal threshold per class** (max of `2*recall*precision/(recall+precision)` over the full
curve, not just the 0.05 grid):

| class | `y26n_humanshaped_v2` | `y26n_noe2e_warm50-2` |
|---|---|---|
| Woman | thr 0.05* → recall 0.897, precision 0.878 (F1 0.887) | thr 0.30 → recall 0.892, precision 0.877 (F1 0.884) |
| Man | thr 0.50 → recall 0.881, precision 0.928 (F1 0.904) | thr 0.45 → recall 0.887, precision 0.918 (F1 0.902) |
| Child | thr 0.15 → recall 0.816, precision 0.850 (F1 0.833) | thr 0.25 → recall 0.808, precision 0.860 (F1 0.833) |

\* **Woman's F1 has no real interior peak on either model** — it is flat-to-declining across the
whole tested range, so 0.05 wins by sitting at the grid floor, not because there is a knee there;
treat Man's and Child's optima as the meaningful numbers, Woman's as inconclusive without
testing below 0.05. Man and Child both land close to the current shipped 0.45 — **nothing in
this curve argues for moving off 0.45** as a shared threshold. Per-class thresholds are
technically supported by this pipeline (each class's curve is independent of the others' cutoffs
at the filtering step) if a genuinely different cut per class is ever wanted.

## LAGENDA benchmark-set mismatch found + corrected (2026-09-09)

**The question that triggered this:** why did `y26n_humanshaped_v2` score so much higher than
`y26n_noe2e_warm50-2` on LAGENDA mAP (0.859 vs 0.754 mAP50, a ~10.4pt gap) when the only stated
difference was training on humanshaped/cartoon-promoted labels, and the two checkpoints are
statistically indistinguishable on every other LAGENDA measurement in this document (the
classification sweep and PR-curve sections above)?

**Root cause found: the two models were never scored on the same LAGENDA images.**
`y26n_humanshaped_v2`'s LAGENDA raw dump (`/workspace/mapdump/y26n_humanshaped_v2/lagenda/raw`)
has **4,601 images** — the corrected LAGENDA v2 set, after this project's own documented fix
that excludes 298 images overlapping the OIV7 train split (see `docs/DATASET_REGISTRY.md` /
the LAGENDA contamination note). `y26n_noe2e_warm50-2`'s LAGENDA dump used for the original
comparison (`/workspace/mapdump/y26n_noe2e_warm50-2/lagenda/raw`) has **4,899 images** — a
stale dump from before that correction was applied. **Every model in this document except the
two humanshaped ones was scored on that same stale 4,899-image set** — the humanshaped models
were the only ones ever benchmarked on the corrected 4,601-image list. `n_gt` (7,098 labeled
people) is identical either way, confirming the 298 excluded images have zero human-labeled
people in LAGENDA's sparse ground truth — so those extra images function as pure background
images in the mAP calculation: any legitimate detection a model makes on a real-but-unlabeled
person in those 298 images gets scored as a false positive, dragging down precision-based AP
for every model that was tested against them, while the two humanshaped models were never
tested there at all.

**Verified by direct re-scoring**, not just by counting images. Every model with an existing raw
sidecar dump was re-scored on the identical, matched 4,601-image set (`map_eval.py`, no model
re-run — for models whose dump already existed at 4,601 images the existing file was reused; for
the rest, a subset raw-sidecar view was built from the stale 4,899-image dump, restricted to the
matching stems):

| model | original (stale 4,899-img) mAP50 | **corrected (matched 4,601-img) mAP50 / mAP50-95** |
|---|---|---|
| `yolo11N-640` (production) | 0.7238 | 0.8228 / 0.7142 |
| `y26n_sop50` | 0.7584 | 0.8608 / 0.7266 |
| `y26n_noe2e_warm50-2` | 0.7544 | 0.8567 / 0.7231 |
| `gelannfav14r4fw_gemlb_v2` | 0.7826 | 0.8820 / 0.7312 |
| `gelan_r4v2` | 0.7976 | 0.8948 / 0.7401 |
| `gelan_r6_v2` | 0.7914 | 0.8935 / 0.7349 |
| `gelan_w_v1` | 0.7789 | 0.8786 / 0.7318 |
| `y26n_gradsupp` | 0.7890 | 0.8953 / 0.7527 |
| `y26n_spotlight` | 0.7894 | 0.8947 / 0.7522 |
| `y26n_unk4` | 0.7885 | 0.8941 / 0.7529 |
| `yoloe_n` | 0.7857 | 0.8903 / 0.7472 |
| `y26n_humanshaped_v2` | n/a (already on corrected set) | 0.8588 / 0.7233 |
| `y26s_humanshaped_smallpatch_v1` | n/a (already on corrected set) | 0.8842 / 0.745 |

**Conclusion: the humanshaped-vs-non-humanshaped LAGENDA gap does not exist.** Once every model
is scored on the identical image set, `y26n_humanshaped_v2` and `y26n_noe2e_warm50-2` land
within 0.2 mAP50 points of each other (0.8588 vs 0.8567) — noise, not a training-data effect,
confirming the classification-sweep and PR-curve findings elsewhere in this document.
**Every prior claim in this document (and in `experiments/EXP-2026-20-yolo26s-humanshaped-
smallpatch.md`) that a humanshaped model "wins" or is "best" on LAGENDA mAP is WRONG and should
be read as corrected here** — five other models (`y26n_gradsupp`, `y26n_spotlight`,
`y26n_unk4`, `gelan_r4v2`, `gelan_r6_v2`) now outscore both humanshaped models on LAGENDA mAP50
once fairly measured. This does **not** affect the other datasets in the standard comparison
(Spotlight-val, CrowdHuman, PASS, object_set, the holdout QA set, or the pixel-level/exposure-
tier work) — those were independently verified to use consistent image sets across models and
are unaffected by this specific bug.

**Why this wasn't caught sooner:** `n_images` was recorded correctly in every `map_eval.py`
output all along (`4601` vs `4899` is visible in every JSON's own metadata) — the comparison
table simply never cross-checked that field across rows before this investigation. **Lesson for
future comparisons: always diff `n_images`/`n_gt` across every row of a cross-model table before
trusting a gap** — this is now a standing check, not a one-off fix.

**Reproduce:** `map_eval.py --raw <raw_dir> --gt-labels /workspace/datasets/lagenda_full/eval_v2/
labels_3class --ignore-labels /workspace/datasets/lagenda_full/eval_v2/ignore --expect-floor
0.001 --out <out>.json`. Corrected raw sidecar dirs used: `/workspace/exp_lagenda_bench/
{yolo11N-640,y26n_sop50,y26n_noe2e_warm50-2,gelannfav14r4fw_gemlb_v2}/lagenda/raw` (already at
4,601 images) and `/tmp/subset4601_{gelan_r4v2,gelan_r6_v2,gelan_w_v1,y26n_gradsupp,
y26n_spotlight,y26n_unk4,yoloe_n}` (built as symlink subsets of the stale 4,899-image mapdump
dumps, restricted to `y26n_humanshaped_v2`'s 4,601 stems — a strict subset relationship,
verified via `comm` before scoring).

## INT8 quantization accuracy matrix — train-calibrated, corrected (2026-09-09)

**Why this exists:** the first INT8 accuracy pass (the "val-calibrated" numbers quoted in the
2026-09-08 TFLite section above and in the previous session's `/workspace/quant_matrix/`) inherited
ultralytics' export defaults (`split=val`, `fraction=1.0`) and so calibrated every INT8 model's
quantization ranges on **the same 4,232 Spotlight-val images it was then scored on**. That is
unsupervised distribution leakage that flatters int8 on that eval set, and 10x more calibration
images than range statistics need (ultralytics' own source recommends >300). Nobody chose it; see
memory `int8-calibration-recipe-flaw`. This section redoes the matrix with calibration on a
fixed 500-image **train**-split subset and evaluation on val, and keeps the flawed arm as a
comparison column so the leakage delta is visible.

### Calibration methodology (identical for all 9 exports)

```
sort /workspace/exp12/train_full.txt | awk 'NR % 950 == 1' | head -500 > /workspace/exp12/calib_train500.txt
sed 's#^train:.*#train: /workspace/exp12/calib_train500.txt#' /workspace/exp12/spotlight_oiv7.yaml > /workspace/exp12/spotlight_oiv7_calib500.yaml
yolo export model=<run>.pt format=tflite imgsz=<SZ> int8=True data=/workspace/exp12/spotlight_oiv7_calib500.yaml split=train device=cpu
```

- `calib_train500.txt`: 500 lines, sha256 `55d0a78e75b436b0ace227f49ea027d462b6b0a749c507db36ec51027551bff9`,
  first three entries `train_tree/images/train/000002c707c9895e.jpg`, `00cb3fbb7339f271.jpg`,
  `01bf4d4cb674e46c.jpg` (deterministic: sorted list, every 950th line). Zero overlap with
  `val_full.txt` by construction (different split).
- `split=train` is what makes the exporter read the `train:` key; with the default `split=val`
  the yaml's `val:` line (= the eval set) is used regardless of what `train:` says.
- Ultralytics 8.4.146 (export path `litert_torch 0.9.4`, same as the 2026-09-09 y26s bundle;
  ai-edge-litert 2.2.0, ai-edge-quantizer 0.9.0, TensorFlow 2.21.0, torch 2.13.0), Python 3.11,
  `device=cpu`. Post-training INT8 with a representative-dataset calibration, float32 I/O — the
  same `int8=True` recipe as the 2026-09-08 bundles (that section's "dynamic-range" wording was
  loose: with a `data=` yaml the exporter runs a calibration pass, which is what this whole
  section is about).
- Export dirs: `/workspace/exports/<run>_calib500/sz<SZ>/<run>_int8.tflite` with the full
  export log (`export_attempt1.log`) and a `status.log` next to each. The `.tflite` bundles
  (~135 MB total) stay on the pod — the RunPod proxy SSH route has no scp, and they are
  reproducible from the three lines above.

### Success criterion for an INT8 export (all three, never just one)

1. The `yolo export` process exited **and** its log contains both `Quantized model size` and
   `export success`.
2. The output file is ~1/4 of the fp32 TFLite: **~2.87 MB for the YOLO26n models, ~10.2 MB for
   YOLO26s.** A ~9.8 MB (nano) / ~38 MB (small) file is **not quantized**.
3. The `Running Calibration::` bar totals `208,000` iterations (500 images × 416 quantization
   params); with the old val recipe it was `1,760,928` (4,232 × 416).

Why the size rule is the load-bearing one: ultralytics writes a full-size interim `.tflite` at
the final path *before* calibration. A killed/failed calibration leaves that file in place, exit
code 0 or not, and a "file exists" check calls it done. **This exact failure was found in the
previous run**: `/workspace/exports/y26s_humanshaped_smallpatch_v1_20260909/sz640/` holds a
38,199,244-byte "int8" file whose calibration log ends at 90% (1,583,619/1,760,928) when the
pod was torn down — so the val-calibrated `y26s_humanshaped_smallpatch_v1` @640 row in the
comparison column below is **fp32 TFLite, not INT8**, which is why it scored above the fp32
checkpoint. It is retained, struck through, as the cautionary example.

### The matrix — Spotlight-val (4,232 images, 8,035 GT people, 702 ignore boxes), `map_eval.py`, pycocotools 101-pt, floor 0.001

Three rows per model × size. **fp32 `.pt`** is the checkpoint itself (the 640 rows are the
standing numbers from `/workspace/evalout_v2/<run>_map_spotval.json`, reused verbatim; the
416/320 rows were run this session on the GPU with the same script). **fp32 TFLite** is a fresh
un-calibrated export under the same toolchain as the INT8 rows — it is the *right* baseline for
a quantization-only delta, because it removes the export path from the comparison. **INT8
TFLite** is the train-calibrated export. Every row is `n_images=4232, n_gt=8035` (asserted, not
assumed — see the LAGENDA section above for why).

| model | imgsz | precision / path | mAP50 | mAP50-95 | Woman AP50 | Man AP50 | Child AP50 | mAP50-95 S / M / L | val-calibrated INT8 (flawed arm) mAP50 / mAP50-95 | train − val calib Δ |
|---|---|---|---|---|---|---|---|---|---|---|
| `y26n_humanshaped_v2` | 640 | fp32 `.pt` | 0.8060 | 0.7035 | 0.8226 | 0.8595 | 0.7358 | 0.3384 / 0.5764 / 0.7666 | |  |
| `y26n_humanshaped_v2` | 640 | fp32 TFLite | 0.8066 | 0.7037 | 0.8235 | 0.8624 | 0.7339 | 0.3367 / 0.5790 / 0.7667 | |  |
| `y26n_humanshaped_v2` | 640 | **INT8 TFLite, train-calibrated** | 0.8201 | 0.6615 | 0.8220 | 0.8551 | 0.7832 | 0.2915 / 0.5012 / 0.7460 | 0.8114 / 0.6525 | +0.0087 / +0.0090 |
| `y26n_humanshaped_v2` | 416 | fp32 `.pt` | 0.8005 | 0.6886 | 0.8275 | 0.8325 | 0.7414 | 0.3215 / 0.5540 / 0.7604 | |  |
| `y26n_humanshaped_v2` | 416 | fp32 TFLite | 0.7983 | 0.6884 | 0.8240 | 0.8381 | 0.7328 | 0.3214 / 0.5523 / 0.7599 | |  |
| `y26n_humanshaped_v2` | 416 | **INT8 TFLite, train-calibrated** | 0.8198 | 0.6774 | 0.8339 | 0.8394 | 0.7862 | 0.3156 / 0.5240 / 0.7635 | 0.8094 / 0.6687 | +0.0104 / +0.0087 |
| `y26n_humanshaped_v2` | 320 | fp32 `.pt` | 0.7865 | 0.6623 | 0.8221 | 0.8103 | 0.7270 | 0.2924 / 0.5111 / 0.7417 | |  |
| `y26n_humanshaped_v2` | 320 | fp32 TFLite | 0.7862 | 0.6633 | 0.8222 | 0.8132 | 0.7231 | 0.3006 / 0.5112 / 0.7431 | |  |
| `y26n_humanshaped_v2` | 320 | **INT8 TFLite, train-calibrated** | 0.8166 | 0.6679 | 0.8477 | 0.8293 | 0.7727 | 0.2842 / 0.5204 / 0.7465 | 0.7913 / 0.6345 | +0.0253 / +0.0334 |
| `y26n_noe2e_warm50-2` | 640 | fp32 `.pt` | 0.8022 | 0.6971 | 0.8105 | 0.8643 | 0.7319 | 0.3368 / 0.5711 / 0.7599 | |  |
| `y26n_noe2e_warm50-2` | 640 | fp32 TFLite | 0.8063 | 0.7014 | 0.8116 | 0.8668 | 0.7405 | 0.3365 / 0.5772 / 0.7626 | |  |
| `y26n_noe2e_warm50-2` | 640 | **INT8 TFLite, train-calibrated** | 0.8115 | 0.6544 | 0.7998 | 0.8572 | 0.7774 | 0.3026 / 0.5029 / 0.7322 | 0.8029 / 0.6435 | +0.0086 / +0.0109 |
| `y26n_noe2e_warm50-2` | 416 | fp32 `.pt` | 0.8037 | 0.6921 | 0.8393 | 0.8407 | 0.7311 | 0.3221 / 0.5574 / 0.7652 | |  |
| `y26n_noe2e_warm50-2` | 416 | fp32 TFLite | 0.8020 | 0.6921 | 0.8375 | 0.8447 | 0.7237 | 0.3243 / 0.5587 / 0.7641 | |  |
| `y26n_noe2e_warm50-2` | 416 | **INT8 TFLite, train-calibrated** | 0.8103 | 0.6665 | 0.8391 | 0.8413 | 0.7505 | 0.2950 / 0.5214 / 0.7514 | 0.8074 / 0.6671 | +0.0029 / -0.0006 |
| `y26n_noe2e_warm50-2` | 320 | fp32 `.pt` | 0.7849 | 0.6595 | 0.8259 | 0.8180 | 0.7108 | 0.2878 / 0.5070 / 0.7407 | |  |
| `y26n_noe2e_warm50-2` | 320 | fp32 TFLite | 0.7859 | 0.6629 | 0.8234 | 0.8177 | 0.7167 | 0.2962 / 0.5138 / 0.7414 | |  |
| `y26n_noe2e_warm50-2` | 320 | **INT8 TFLite, train-calibrated** | 0.8087 | 0.6629 | 0.8478 | 0.8307 | 0.7475 | 0.2833 / 0.5108 / 0.7463 | 0.7914 / 0.6368 | +0.0173 / +0.0261 |
| `y26s_humanshaped_smallpatch_v1` | 640 | fp32 `.pt` | 0.8500 | 0.7637 | 0.8742 | 0.8910 | 0.7847 | 0.3951 / 0.6378 / 0.8339 | |  |
| `y26s_humanshaped_smallpatch_v1` | 640 | fp32 TFLite | 0.8527 | 0.7682 | 0.8759 | 0.8918 | 0.7905 | 0.3999 / 0.6458 / 0.8362 | |  |
| `y26s_humanshaped_smallpatch_v1` | 640 | **INT8 TFLite, train-calibrated** | 0.8487 | 0.7019 | 0.8679 | 0.8828 | 0.7955 | 0.3660 / 0.5514 / 0.7896 | ~~0.8527 / 0.7682~~ (unquantized, see text) | n/a |
| `y26s_humanshaped_smallpatch_v1` | 416 | fp32 `.pt` | 0.8528 | 0.7610 | 0.8771 | 0.8749 | 0.8064 | 0.3581 / 0.6305 / 0.8365 | |  |
| `y26s_humanshaped_smallpatch_v1` | 416 | fp32 TFLite | 0.8523 | 0.7606 | 0.8773 | 0.8801 | 0.7995 | 0.3699 / 0.6338 / 0.8346 | |  |
| `y26s_humanshaped_smallpatch_v1` | 416 | **INT8 TFLite, train-calibrated** | 0.8416 | 0.7095 | 0.8708 | 0.8739 | 0.7801 | 0.3225 / 0.5727 / 0.7990 | 0.8447 / 0.7148 | -0.0031 / -0.0053 |
| `y26s_humanshaped_smallpatch_v1` | 320 | fp32 `.pt` | 0.8319 | 0.7323 | 0.8533 | 0.8580 | 0.7845 | 0.3014 / 0.5886 / 0.8172 | |  |
| `y26s_humanshaped_smallpatch_v1` | 320 | fp32 TFLite | 0.8370 | 0.7350 | 0.8619 | 0.8600 | 0.7892 | 0.3266 / 0.5905 / 0.8189 | |  |
| `y26s_humanshaped_smallpatch_v1` | 320 | **INT8 TFLite, train-calibrated** | 0.8497 | 0.7181 | 0.8780 | 0.8730 | 0.7982 | 0.3061 / 0.5770 / 0.8030 | 0.8327 / 0.6959 | +0.0170 / +0.0222 |

**Quantization-only deltas** (INT8 minus fp32 TFLite, same size, same toolchain):

| model | imgsz | Δ mAP50 (int8 − fp32 TFLite) | Δ mAP50-95 | Δ Woman | Δ Man | Δ Child | Δ mAP50-95 small |
|---|---|---|---|---|---|---|---|
| `y26n_humanshaped_v2` | 640 | +0.0135 | -0.0422 | -0.0015 | -0.0073 | +0.0493 | -0.0452 |
| `y26n_humanshaped_v2` | 416 | +0.0215 | -0.0110 | +0.0099 | +0.0013 | +0.0534 | -0.0058 |
| `y26n_humanshaped_v2` | 320 | +0.0304 | +0.0046 | +0.0255 | +0.0161 | +0.0496 | -0.0164 |
| `y26n_noe2e_warm50-2` | 640 | +0.0052 | -0.0470 | -0.0118 | -0.0096 | +0.0369 | -0.0339 |
| `y26n_noe2e_warm50-2` | 416 | +0.0083 | -0.0256 | +0.0016 | -0.0034 | +0.0268 | -0.0293 |
| `y26n_noe2e_warm50-2` | 320 | +0.0228 | +0.0000 | +0.0244 | +0.0130 | +0.0308 | -0.0129 |
| `y26s_humanshaped_smallpatch_v1` | 640 | -0.0040 | -0.0663 | -0.0080 | -0.0090 | +0.0050 | -0.0339 |
| `y26s_humanshaped_smallpatch_v1` | 416 | -0.0107 | -0.0511 | -0.0065 | -0.0062 | -0.0194 | -0.0474 |
| `y26s_humanshaped_smallpatch_v1` | 320 | +0.0127 | -0.0169 | +0.0161 | +0.0130 | +0.0090 | -0.0205 |

### What the matrix says

1. **The TFLite export path is faithful.** fp32 TFLite vs fp32 `.pt` agree within ±0.005
   mAP50 and ±0.005 mAP50-95 in all 9 cells (largest gap: `y26s_humanshaped_smallpatch_v1`
   @320, 0.8370 vs 0.8319). So any int8-vs-fp32 difference below is quantization, not export.
   Corollary: the val-calibrated `y26s_humanshaped_smallpatch_v1` @640 "int8" number from the
   previous session (0.8527 / 0.7682) is **identical to this session's fp32 TFLite row** — the
   definitive confirmation that that file was never quantized.

2. **INT8 costs fine localization, not IoU-0.5 detection — and the pattern is stronger than
   the val-calibrated arm suggested.** mAP50-95 drops at 640 (−4.2 / −4.7 / −6.6 pts for
   `y26n_humanshaped_v2` / `y26n_noe2e_warm50-2` / `y26s_humanshaped_smallpatch_v1`) and at
   416 (−1.1 / −2.6 / −5.1), and is flat at 320 (+0.5 / 0.0 / −1.7). Small-object mAP50-95
   drops in every one of the 9 cells (−0.6 to −4.7 pts). mAP50, meanwhile, does **not** drop for
   the nanos — it *rises* in all six nano cells (+0.5 to +3.0 pts, largest at 320) — and for
   y26s moves −0.4 / −1.1 / +1.3.

3. **`y26s_humanshaped_smallpatch_v1` is the least INT8-robust of the three, not the most.**
   It has the largest mAP50-95 loss at both 640 and 416 and is the only model whose mAP50 goes
   down at those sizes. If the export decision is "y26s at 416 in INT8", budget ≈ −1.1 mAP50 /
   −5.1 mAP50-95 against its fp32 TFLite (0.8523 / 0.7606 → 0.8416 / 0.7095) — it still beats
   both nanos' fp32 on every column, at 3.7x their file size (10.2 MB vs 2.87 MB). Hypothesis,
   not measured: the larger model's wider activation ranges quantize worse under per-tensor
   min/max calibration.

4. **The mAP50 *rise* under INT8 is real in the numbers and unexplained.** A background
   diagnostic over the raw sidecars (`/workspace/dets_diag.log`, first cells) shows INT8 emits
   *fewer* raw boxes above the 0.001 floor than fp32 TFLite (e.g. `y26n_humanshaped_v2` @640:
   41,935 vs 44,432) but *more* above 0.25 (8,936 vs 8,428) — a confidence redistribution, not
   extra low-confidence boxes padding the PR tail. Two consequences: (a) **do not reuse fp32
   deployment thresholds for an INT8 export** — re-sweep per precision (the 2026-09-08
   threshold-sweep tooling applies unchanged); (b) the per-class movement is not uniform:
   Child AP50 rises +3.7 to +5.3 pts for the nanos at every size while Woman/Man move within
   ±1 pt at 640/416. Whether that Child gain is better-ranked real children or extra
   Child-labelled boxes on adults (the adult-escape direction) is **not answerable from mAP** —
   it needs the LAGENDA classification sweep run on the INT8 files. Open item.

5. **Train-calibration on 500 images was as good as or better than val-calibration on 4,232
   in 7 of 8 comparable cells** (column "train − val calib Δ": +0.3 to +2.5 mAP50, +0.9 to
   +3.3 mAP50-95; tie at `y26n_noe2e_warm50-2` @416; −0.3 / −0.5 at y26s @416). The direction is
   the *opposite* of "leakage flatters the val arm": calibrating on the eval images did not buy
   the val arm anything, and at 320 the 4,232-image calibration was clearly worse (−1.7 to −2.5
   mAP50). Consistent with the range-widening mechanism in memory
   `int8-calibration-recipe-flaw` (more calibration images → more outlier activations → wider
   per-tensor ranges → coarser steps). **Retire the val-calibrated arm**: it is neither a
   cleaner baseline nor a better export. Export time also fell from 13–20 min to 0.8–7.7 min
   per model on 13.6 cores.

### What this does NOT show (harshest reader)

- **Spotlight-val only, and its labels come from the same pipeline that trained these models.**
  That bias is identical for the fp32 and INT8 rows of the *same* model, so within-model
  precision deltas are valid; the cross-model gaps in this table are **not** a ranking (LAGENDA
  `fl1199`, CrowdHuman, PASS, `object_set`, and `haramblur_holdout` sections above do that). It
  says nothing about the Gulf-dress misread, object-set false positives, crowd recall, or the
  adult→Child leak rate under INT8 — point 4 above is the closest it gets, and it is a flag,
  not a measurement.
- **One calibration draw.** Every INT8 row is a single 500-image subset (the hash above). The
  variance across calibration draws was not measured, so any arm-to-arm gap under ~1 pt
  (several cells in the val-vs-train column, the y26s mAP50 deltas at 640/416) may be draw
  noise. The 4–7 pt mAP50-95 losses at 640 are far outside that and stand.
- **The val-vs-train comparison is not a controlled experiment.** Source (val vs train) and
  count (4,232 vs 500) changed together, and for the two nano bundles the toolchain may also
  differ (`humanshaped_v2_20260902` / `noe2e_warm50-2_20260908` were exported under Ultralytics
  8.4.144; `/workspace/exports_build.log` mentions `onnx2tf`; the y26s bundle and everything
  here used `litert_torch 0.9.4`). It shows the old arm was not better; it does not isolate why.
- **Latency was not re-measured.** The 2026-09-08 size/latency table stands (same architectures;
  file sizes here match it to within 0.1%): INT8 is a 3.4x size win everywhere and a speed win
  only at 320.
- **The fp32 416/320 `.pt` rows ran on a GPU (RTX 4090, torch 2.13.0+cu130) while every TFLite
  row ran on CPU/XNNPACK.** The ±0.005 agreement in point 1 says this didn't matter here, but
  the two device paths are not byte-identical.

### Verify it yourself (verbatim; every threshold and its source)

```
# calibration subset (deterministic) + yaml
sort /workspace/exp12/train_full.txt | awk 'NR % 950 == 1' | head -500 > /workspace/exp12/calib_train500.txt
sha256sum /workspace/exp12/calib_train500.txt   # 55d0a78e75b436b0ace227f49ea027d462b6b0a749c507db36ec51027551bff9
sed 's#^train:.*#train: /workspace/exp12/calib_train500.txt#' /workspace/exp12/spotlight_oiv7.yaml > /workspace/exp12/spotlight_oiv7_calib500.yaml
# INT8 export (per run/size, cwd = /workspace/exports/<run>_calib500/sz<SZ>, checkpoint copied in as <run>.pt)
OMP_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=2 yolo export model=<run>.pt format=tflite imgsz=<SZ> int8=True \
  data=/workspace/exp12/spotlight_oiv7_calib500.yaml split=train device=cpu
# fp32 TFLite export (same dir, no calibration)
yolo export model=<run>.pt format=tflite imgsz=<SZ> device=cpu
# inference (raw sidecar logs every box >= --floor; --conf 0.45 default only sets the 'kept' flag, map_eval ignores it)
OMP_NUM_THREADS=T MKL_NUM_THREADS=T python3 /workspace/data_inspection_tools/vlm-cluster/run_ultralytics_labels.py \
  --engine ultralytics --model <file.tflite or best.pt> --imgsz <SZ> --device cpu|cuda:0 \
  --images /workspace/exp12/images_val4232 --out <out> --floor 0.001
# scoring (never model.val(): different AP convention, no ignore regions)
python3 /workspace/data_inspection_tools/vlm-cluster/map_eval.py --raw <out>/raw \
  --gt-labels /workspace/spotlight/run/oiv7_val/labels --ignore-labels /workspace/spotlight/run/oiv7_val/ignore_unk \
  --expect-floor 0.001 --out <out>_map.json
```

Thresholds: `--floor 0.001` (raw-sidecar floor; `map_eval.py --expect-floor` asserts the dump was
made at it), IoU 0.50:0.05:0.95 and 101-point interpolation inside `map_eval.py`, ignore regions
IoA ≥ 0.5 (702 unknown-gender boxes), `max_dets_per_image=100`. NMS IoU 0.7 (`--iou` default
in `run_ultralytics_labels.py`), applied identically to every row.

Pipeline scripts as run (all on the volume): `/workspace/quant_calib500_pipeline.sh` (export →
infer → score, 9 jobs, size check + `export_ok` marker), `/workspace/fp32tflite.sh` (fp32 TFLite
rows), `/workspace/fp32_smallsz.sh` (fp32 `.pt` 416/320 rows), `/workspace/repair_calib500.sh`
(see below). Outputs: `/workspace/quant_matrix_calib500/<run>/{sz,fp32tflite_sz,fp32_sz}<SZ>_spotval{,_map.json,.log}`;
exports `/workspace/exports/<run>_calib500/sz<SZ>/<run>_{int8,fp32}.tflite` (+ `export_attempt1.log`,
`export_fp32.log`, `status.log`). Local copies of all 28 `_map.json` files (9 INT8, 9 fp32
TFLite, 6 fp32 `.pt` 416/320, the 3 standing fp32 @640 JSONs, and the val-calibrated y26s @416
scored this session): `models/quant_matrix_calib500_20260909/` (gitignored).

**Incident that shaped the run (so the next person doesn't re-learn it):** the network volume
hit its quota mid-run (`Errno 122`), which (a) killed one inference job at 1,729/4,232, (b)
wrote three 0-byte `_map.json` files that parsed as "done", and (c) left 47 raw sidecars missing
in `y26n_noe2e_warm50-2` @640 that a plain resume would have skipped forever — the script's
resume test is "label file exists", not "raw JSON exists". Every INT8 directory was therefore
re-validated (`repair_calib500.sh`: parse every raw JSON, delete unparseable ones, delete label
files for stems without a valid raw JSON, resume, rescore, assert `n_images == 4232`) before any
number above was taken. Memory: `run-ultralytics-labels-resume-is-label-based`,
`runpod-volume-and-gemini-storage-caps`.
