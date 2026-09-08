# Model round-up — unknown-handling arms, YOLOE, and the two gelan mains (2026-08-08)

Five new models measured on the identical protocol used for every model so far (same
images verified by count, 640 px, conf 0.45, IoU-0.5 matching, same scorers, one shared
mAP evaluator). Full side-by-side of all ten models measured to date:
`docs/MODEL_COMPARISON.md`. Method: `docs/MODEL_EVAL_OVERVIEW.md`.

## What was tried

- **Unknown-label handling on YOLO26n** — Spotlight drops ~12% of detected people as
  unknown-gender (median ~58 px). Training with them as background cost ~7 pts of crowd
  recall (measured earlier). Two fixes tested: **4-class Unknown** (`y26n_unk4`, dropped
  at inference) and **cls-gradient suppression** (`y26n_gradsupp` — the unknown teaches
  the detector WHERE a person is but is excluded from the classifier's gradient).
- **YOLOE fine-tune** (`yoloe_n_gradsupp`) — the promptable/open-vocab architecture,
  fine-tuned nano with the same gradient-suppression treatment.
- **The two gelan mains** — `gelannfav14w_gemlb_v1` and `gelannfav14r4_gemlb_v2`,
  measured on the same benchmarks for the first time.

## Results

Lower is better for the FP columns (the "it blurs books" complaint); higher is better for
recall/gender; the leak column is adults ≥20 wrongly called Child (they escape the blur).

Models are named exactly as their run directories. Baseline = `y26n_spotlight`
(EXP-2026-12, unknowns dropped as background).

| run name | objects<br>%img-FP ↓ | PASS<br>FP/100 ↓ | crowd<br>recall | crowd<br>prec. | LAGENDA<br>detect | LAGENDA<br>gender | leak ≥20<br>→Child ↓ | child<br>recall | mAP50 /<br>50-95 | latency | NMS<br>incl.? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `y26n_spotlight` (baseline) | 12.7 | **0.23** | 32.8 | 93.9 | 94.1 | 90.7 | 0.11 | 79.0 | 0.799 / 0.685 | 34.0 ms | ✅ |
| **`y26n_gradsupp`** | **11.2** | 0.27 | 36.5 | 94.0 | 93.9 | 91.1 | 0.11 | 81.0 | 0.799 / 0.689 | **33.8 ms** | ✅ |
| `y26n_unk4` | 11.6 | 0.40 | 33.5 | **94.2** | 93.3 | 90.9 | 0.15 | 81.1 | 0.801 / 0.690 | 33.6 ms | ✅ |
| `yoloe_n_gradsupp` | 11.6 | 0.37 | 36.3 | 93.5 | 93.3 | 91.5 | 0.15 | 80.2 | **0.801 / 0.691** | 49.1 ms | ✅ |
| **`gelannfav14r4_gemlb_v2`** | 16.2 | 0.33 | **41.1** | 92.7 | **95.2** | **92.0** | **0.07** | 81.3 | 0.788 / 0.645 | 59.5 ms † | ❌ † |
| `gelannfav14w_gemlb_v1` | **10.0** | 0.57 | 36.7 | 92.3 | 93.0 | 89.7 | 0.34 | **85.2** | 0.753 / 0.610 | 89.3 ms † | ❌ † |

Also timed (not accuracy-scored this round): `gelannfav14r3_gemlb_v1` 58.5 ms ·
`gelannfav14m_gemlb_v1` 73.1 ms — both raw-tensor outputs.

† raw-tensor output — the browser still runs NMS over 8,400 candidates on top of the time
shown, so the gelan numbers understate their true cost. All latencies measured in one
session on one machine (AMD EPYC 9254, ONNX Runtime CPU, 4 threads, 640 px, median of 100
runs after 20 warmup); `y26n_spotlight` and `yolo11N-640` agreed within ~3% with the
earlier EPYC 7352 session, so the older `gelansfav14_*` figures (~135 ms) remain
comparable.

## What we learned

1. **Gradient suppression is the better unknown-handling design** — +3.7 pts crowd recall
   over treating unknowns as background, versus +0.7 for the 4-class approach, while also
   *improving* the object false-positive rate (12.7 → 11.2%) and holding the leak at
   0.11%. Between the two designs it wins clearly.
2. **But it only recovers about half of what grey-masking achieves.** Our pre-registered
   bar was +5 pts; it landed at +3.7, so formally the arm did not pass. Masking (the gelan
   line's approach) still leads on crowd recall across two architectures.
   **Suggested next arm: grey-masked unknowns on YOLO26n** — nobody has tried that
   combination, and it's the one most likely to merge both sides' strengths.
3. **YOLOE is closed.** The fine-tuned nano ties our YOLO26n on essentially every metric
   at 2.3× the parameters plus a segmentation head. Both of its pre-registered win
   conditions failed. Combined with the earlier zero-shot rejection, the architecture has
   now been answered end-to-end with numbers — no further YOLOE work planned.
4. **`gelannfav14r4_gemlb_v2` is the strongest model on the "adults escaping the blur"
   side** measured to date at nano scale: best crowd recall (41.1%), best LAGENDA
   detection (95.2%), best gender (92.0%), best leak (0.07%). Its cost is the false
   positive side — 16.2% of object-set images gain a false person, ~45% more than
   y26n_gradsupp's 11.2%.
5. `gelannfav14w_gemlb_v1` trades the opposite way (best object-FP at 10.0%) but has the
   worst leak (0.34%) and lowest gender (89.7%) of the round, and is ~2.8× the parameters
   of the nano candidates.

## The decision — now that latency is measured

**Proposed: `y26n_gradsupp` as the production candidate.** The last missing measurement
came in and it isn't a close trade:

| | y26n_gradsupp | gelannfav14r4_gemlb_v2 |
|---|---|---|
| latency | **33.8 ms** (final boxes) | 59.5 ms **+ browser NMS** — at least 1.8× slower |
| object false persons ↓ | **11.2%** | 16.2% — 45% more false blurs |
| crowd recall | 36.5% | **41.1%** |
| detection / gender / leak | 93.9 / 91.1 / 0.11 | **95.2 / 92.0 / 0.07** |

r4v2 genuinely finds and classifies people better. But it asks for ~2× the compute *and*
45% more false blurs to get there — losing on the #1 user complaint and on the frame
budget, to win a few points on the escape side. For a browser extension that already
downshifts to 320/416 inputs on slower machines, that's the wrong direction.

Two more things the timing run settled: the **whole gelan nano family runs 58–89 ms**
before NMS, so none of it competes with the YOLO26 line on speed; and **YOLOE loses on
compute, not size** — its deployable model is only 2.69M params but 9.1 GFLOPs / 49.1 ms
for no accuracy gain (correcting an earlier note that cited 5.6M params from the
architecture spec; the text-prompt machinery is stripped during fine-tuning).

**The experiment that could end the trade-off entirely: grey-masked unknowns on YOLO26n.**
Masking is what gives the gelan line its recall advantage, it has now worked on two
architectures, and no YOLO26n arm has used it. That arm would plausibly deliver r4v2-level
recall at y26n speed.

## 2026-08-10 addendum — LAGENDA corrected + state-of-the-art mAP

Follow-up to the mAP question raised on this brief. Two findings and a new metric:

1. **Our LAGENDA eval had been using ~1 labeled person per image; the official CSV has
   ~6.** So every historical "LAGENDA detection recall 93–95%" was *prominent-subject*
   recall. Rebuilt against all 27,633 person boxes (298 training-contaminated images
   excluded), real people-recall is **72–80%**, and LAGENDA precision is measurable for
   the first time (91–94%; the y26n family leads it). Rankings are unchanged;
   `gelannfav14r4_gemlb_v2` still leads recall (79.6%), and gradient suppression's gain
   replicates on this second dataset (+2.7 over background). Model rows updated in
   `docs/MODEL_COMPARISON.md`.
2. **mAP is now COCO-conventional**: floor 0.001 (was 0.05 — worth ~+0.02 mAP50
   uniformly), maxDets=100, 101-point pycocotools convention, and ignore-region support —
   applied on BOTH partially-labeled benchmarks: LAGENDA's 20.5k unlabeled person boxes
   AND Spotlight-val's 702 deleted unknown-gender people are ignore regions (the latter
   checked explicitly: uniform +0.005–0.008 mAP50 for every model, no ranking change).
   Principle now on record: absence of a label is not evidence of absence — incomplete GT
   is ignored, never scored against.
3. **The new headline metric: human-labeled 3-class mAP on LAGENDA** (7,098 crowdsourced
   age/gender people; the ~20.5k detected-but-unlabeled person boxes are COCO ignore
   regions). Results: `gelannfav14r4_gemlb_v2` **0.798** mAP50 / 0.658 mAP50-95 · y26n
   family 0.788–0.789 / 0.661 · `yoloe_n_gradsupp` 0.786 / 0.657 · `gelannfav14w_gemlb_v1`
   0.779 / 0.647. **All candidates within ~0.01 of each other** — r4v2 edges AP50 (finds
   more people), y26n edges AP50-95 (tighter boxes). mAP confirms the component-metric
   story rather than changing the decision: the trade is still recall vs FPs+latency.

## Method notes (for anyone quoting these numbers)

- **Human ground truth** decides: LAGENDA (age/gender), CrowdHuman (boxes), PASS and the
  object set (no people). None of these were used in training or checkpoint selection.
- **mAP is a supporting metric only.** It is computed COCO-style (101-point
  interpolation, IoU 0.50–0.95, per class then averaged, detections confidence-ranked) by
  one shared evaluator, against the Spotlight **val** split (4,232 images / 8,035 people)
  whose labels are Gemini-verified, not human. So it measures *agreement with our training
  labels*, not truth — and because `best.pt` is selected by fitness on that same val
  split, it is mildly optimistic and not a clean held-out test number. Two further
  deviations from strict COCO: a uniform 0.05 confidence floor (reads ~0.02 low; compare
  within this table only) and no small/medium/large area breakdown.
- Single run per model, no variance estimate — **treat differences of ≤1 point as ties.**
- Object-set rates are upper bounds (set re-verification pending); PASS has ~5% known
  contamination; both hit all models equally.
- **The Gulf/traditional-dress gender slice is still untested for every model here** —
  it remains our biggest evaluation gap.

## Questions for the team

1. **@[retrain owner]** — two on your models: (a) were the gelan ONNX files we timed
   produced by the exporter *after* the confidence-bug fix? Latency shouldn't move (same
   graph), but worth confirming before we quote it. (b) How is `best.ckpt` selected —
   val fitness, last epoch, something else? It affects how comparable the mAP column is
   between our families.
2. **@[everyone]** — sanity-check the recommendation: `y26n_gradsupp` (33.8 ms, 11.2%
   object FPs) over `gelannfav14r4_gemlb_v2` (59.5 ms + NMS, 16.2% object FPs, +4.6 crowd
   recall). Anyone see a reason the recall matters more than speed + false blurs here?
3. **@[retrain owner]** — worth running grey-masked unknowns on YOLO26n as the next arm?
   That's the combination the data points at, and it's the one shot at r4v2's recall at
   y26n's speed.
4. **@[everyone]** — who can help hand-check a Gulf/traditional-dress slice? Every gender
   number in this table carries that caveat.
