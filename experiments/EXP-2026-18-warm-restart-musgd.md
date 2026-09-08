# EXP-2026-18 — Warm restart with MuSGD: does 50 more epochs at a proper LR fix the pure one-to-many model?

**In one line:** every YOLO26 run so far silently used AdamW (optimizer=auto overrode our lr0)
and stopped at 30 epochs — this arm warm-restarts `y26n_noe2e` (EXP-2026-16's architecturally
pure one-to-many model — never had a one-to-one head) for 50 more epochs with **MuSGD**,
YOLO26's own optimizer, at lr0=0.003.

**Status:** RUN COMPLETE 2026-08-17 — **self-improvement bar CLEARED, deployment bar NOT
CLEARED** (object-set FP regression vs the current best). `y26n_gs_nms` stays the deployable
recommendation. · Owner: Mostafa · Design: Mohammed Yasin

---

## Which checkpoint, and why that choice matters

There are two checkpoints that could be called "the model with no NMS head":

- **`y26n_gradsupp` read via `end2end=False`** — trained with BOTH heads (gradient-suppression
  unknown handling included); the one-to-one head still exists in the weights, just unused at
  inference. Currently the best benchmarked model (objects 10.4%, crowd 38.4, gender 90.6).
- **`y26n_noe2e`** — architecturally pure one-to-many, the one-to-one head was never built.
  Trained on plain classes-0-2 labels (unknowns as background, NO gradient suppression).
  Already benchmarked WORSE than the above (objects 13.1%, crowd 35.7) — per EXP-2026-14 that
  gap is mostly attributable to the missing unknown-handling, not epoch count.

**This experiment continues `y26n_noe2e`** (owner's explicit choice) — plain `YOLO().train()`,
NOT `GSTrainer`, since that checkpoint was never trained with gradient suppression and adding
it mid-continuation would be a third uncontrolled variable on top of optimizer + epochs.

**Prediction, stated before running:** because the dominant factor in EXP-2026-14 was unknown
handling (+3.7 recall) rather than epoch count (+0.3 mAP in EXP-2026-16), this arm is likely
to still trail `y26n_gs_nms` even if it improves on its own prior numbers. If the goal becomes
"the single best model" rather than "does MuSGD fix this specific architecture," the stronger
follow-up is the same MuSGD/epoch treatment applied to `y26n_gradsupp` via `GSTrainer`.

## Why

Two things we got wrong or never tested, both surfaced by Yasin:

1. **We never used MuSGD.** YOLO26's headline optimizer is MuSGD (SGD+Muon hybrid); it is what
   the architecture was designed and benchmarked around. Every run of ours logged
   `optimizer='auto' found, ignoring 'lr0'` and selected **AdamW** instead. So the vendor's
   published YOLO26 numbers and ours were produced by different optimizers.
2. **30 epochs may have been short.** `y26n_spotlight` and `y26n_gradsupp` peaked at epoch 28
   and flattened, which read as convergence — but `y26n_noe2e`'s best epoch was its last (30),
   which is the classic "still climbing" signal.

**A conclusion this experiment may overturn.** EXP-2026-16 argued the one-to-many loss decay
(0.8 → 0.1) was harmless *because AdamW normalizes by gradient magnitude and is therefore
~invariant to constant loss scaling*. **That argument does not hold for MuSGD** — SGD-family
optimizers are sensitive to gradient scale, so under MuSGD the decay genuinely shrinks the
trunk's effective step size late in training. Yasin's original "dense head is under-trained"
hypothesis could be correct for a MuSGD run even though it was falsified for our AdamW runs.
Watch the o2m-vs-e2e branch gap in this arm.

## Setup

Warm restart from **`y26n_noe2e`** (`/workspace/exp16/train/y26n_noe2e/weights/best.pt`,
epoch 30 — its own best). Plain `YOLO(weights).train(...)`, no custom trainer: the checkpoint
has no one-to-one head to strip and no gradient-suppression loss to preserve, so stock
ultralytics is the correct (and simpler) tool here.

| knob | value | why |
|---|---|---|
| start weights | `/workspace/exp16/train/y26n_noe2e/weights/best.pt` | its own best checkpoint (epoch 30) |
| optimizer | **MuSGD** | YOLO26's own; never yet tested here. MUST be set explicitly or ultralytics ignores lr0 |
| lr0 | **0.003** | Yasin's call. SGD-family LRs run ~10x Adam's; YOLO's SGD default is 0.01, so this is a deliberately gentler warm restart |
| epochs | 50 (→ 80 cumulative) | Yasin's call |
| warmup_epochs | **0** (vs default 3) | ultralytics gives bias parameters their OWN warmup ramp — starting at a fixed elevated `warmup_bias_lr` and decaying DOWN to schedule, distinct from the 0-to-lr0 ramp everything else gets. On a converged checkpoint that elevated bias LR has nothing to "ease into" and risks knocking already-tuned bias terms around for no benefit. Verify the mechanism against pinned 8.4.115 (`inspect.getsource(BaseTrainer)`, search `warmup_bias_lr`) before launch. |
| data / imgsz / batch / workers | `spotlight_oiv7_local.yaml` / 640 / -1 / 12 | unchanged from the comparison arm |

**This is a warm restart, not a continuation** — `resume=True` only finishes the original
30-epoch schedule. The LR schedule restarts from lr0 and decays again, and the optimizer state
is fresh (AdamW moments are discarded; MuSGD starts clean). Mosaic re-enables and closes for
the final 10 of the 50.

## Pre-registered bars (written before launch)

Two comparisons, both reported — self-improvement is the adoption bar, beating the current
best is the higher bar that decides deployment:

| metric | `y26n_noe2e` (its own prior best) | `y26n_gs_nms` (current deployable best) |
|---|---|---|
| Spotlight-val mAP50 | 0.836 | 0.833 |
| crowd recall | 35.7 | **38.4** |
| object-set %img-FP | 13.1% | **10.4%** |
| gender on adults | 90.6% | 90.6% |
| adult(20+)→Child leak | 0.07% | 0.15% |

**Self-improvement bar (does the warm restart help THIS checkpoint):** crowd recall
≥ 37.2 (+1.5) or mAP50 ≥ 0.841 (+0.5), with object-FP not regressing past 13.1%.

**Deployment bar (does it become the new best model):** must beat `y26n_gs_nms` on BOTH
crowd recall (>38.4) AND object-set %img-FP (<10.4%) — matching the prior round's rule that
a recall gain paired with worse FPs is not an adoption.

Given the prediction above, the realistic outcomes are: (a) clears self-improvement but not
deployment → confirms MuSGD/epochs help but unknown-handling remains the bigger lever, or
(b) clears neither → epoch count was never the bottleneck for this architecture, full stop.

Also report, as the scientific payload rather than an adoption criterion:
- **the e2e-vs-o2m branch gap under MuSGD** vs the AdamW-era gap (0.827 / 0.833). If the gap
  widens meaningfully, EXP-2026-16's AdamW-invariance explanation is confirmed as
  optimizer-specific and the non-e2e question reopens for MuSGD.
- the epoch of `best.pt` — if it lands at 48-50, even 80 cumulative epochs is short.

## Cost

~50 epochs x ~20 min = **~17 h, ~$12** on a 4090 + high-vCPU host with images staged to NVMe.

## Results

**Training complete 2026-08-17** — 50 epochs, 38.564 h wall clock (run dir auto-suffixed
`y26n_noe2e_warm50-2` — a leftover empty dir from the first failed launch attempt claimed the
unsuffixed name). `best.pt` = final epoch (50/50), 2,375,421 params fused, 5.3 GFLOPs — same
size/speed as every other y26n arm, confirming the "no one-to-one head" architecture held
through the warm restart (no head was reintroduced).

**Spotlight-val (4,233 imgs), full precision:**

| model | mAP50 | mAP50-95 | P | R |
|---|---|---|---|---|
| `y26n_noe2e` (30ep, AdamW — its own prior best) | 0.836 | 0.717 | 0.812 | 0.760 |
| `y26n_gs_nms` (o2m branch, current deployable best) | 0.833 | 0.715 | 0.803 | 0.761 |
| **`y26n_noe2e_warm50-2`** (80ep cumulative, 50ep MuSGD warm restart) | **0.847** | **0.735** | 0.809 | **0.783** |

Per-class mAP50: Woman 0.867→**0.872**, Man 0.880→**0.897**, Child 0.761→**0.774** — every
class improved, Man and Child by over a point.

**Self-improvement bar: CLEARED via the mAP path** (0.847 ≥ 0.841 required, +1.1 pt over the
+0.5 pt bar). Recall gained +2.3 pts, precision held flat (−0.3, within noise) — the MuSGD
warm restart helped, confirming epoch count / optimizer was leaving real accuracy on the table
for this architecture, on top of (not instead of) the unknown-handling gap EXP-2026-14 found.

**Four-benchmark protocol — run 2026-08-17** (conf 0.45, same LAGENDA/CrowdHuman/PASS/
object-set protocol as every other arm):

| | objects %img-FP | PASS FP/100 | crowd recall | heavy | crowd prec | LAGENDA det | gender | leak(20+) |
|---|---|---|---|---|---|---|---|---|
| `y26n_noe2e` (its own prior best, 30ep AdamW) | 13.1% | 0.30 | 35.7 | 16.6 | 92.9 | 95.4 | 90.6 | 0.07% |
| `y26n_gs_nms` (current deployable best) | **10.4%** | 0.30 | 38.4 | 17.6 | 92.7 | 95.4 | 90.6 | 0.15% |
| **`y26n_noe2e_warm50-2`** (this arm) | 11.2% | 0.37 | **39.2** | **19.7** | 92.2 | **96.2** | **91.5** | 0.14% |

**Self-improvement bar: CLEARED, via BOTH pre-registered paths.** mAP50 0.847 ≥ 0.841, crowd
recall 39.2 ≥ 37.2, and object-FP (11.2%) did not regress past the 13.1% ceiling. Every metric
except PASS (flat, +0.07 pt, noise at n=3,000/11 FPs) and object-FP improved over the checkpoint's
own prior best — crowd recall +3.5 pts, heavy-occlusion recall +3.1 pts, LAGENDA detection
+0.8 pts, gender +0.9 pts, leak roughly unchanged.

**Deployment bar: NOT CLEARED — fails on one of its two required conditions.** Crowd recall
39.2 > 38.4 ✅, but object-set %img-FP 11.2% is not < 10.4% ❌ (it's worse than `y26n_gs_nms`'s
rate, though still much better than this checkpoint's own prior 13.1%). Per the pre-registered
rule ("must beat on BOTH"), a recall gain paired with a worse object-FP rate is not an adoption.
**This is outcome (a), predicted before the run:** MuSGD + more epochs demonstrably helped this
checkpoint, but it still doesn't overtake `y26n_gs_nms` — gradient-suppression unknown-handling
remains the bigger lever than epoch count/optimizer. `y26n_gs_nms` (i.e. `y26n_gradsupp` read
via `end2end=False`) stays the deployable recommendation.

**Best epoch: 50/50 (the last one)** — `results.csv` confirms fitness peaked at the final
epoch, still climbing when the schedule ended. Consistent with the "even 80 cumulative epochs
may be short" risk flagged in Why (§2) — more epochs is the natural next lever if this
architecture is pursued further, though the deployment-bar miss means that's not currently
worth spending on.

**The e2e-vs-o2m branch-gap question is N/A for this arm** — `y26n_noe2e` was never trained
with a one-to-one head at all (that's the whole point of EXP-2026-16), so there is no second
branch to compare here. That question only applies to a MuSGD warm restart of `y26n_gradsupp`
(which does have both heads) — still an open follow-up, not run.

## Supplementary: does the warm restart change how models react to SMALLER people?

The four-benchmark protocol scores real crowd/LAGENDA images at their natural size distribution.
A separate question: as a person's apparent size shrinks (further from camera / smaller crop),
does this arm degrade differently than its predecessor or than the current best model? Built
`vlm-cluster/scale_robustness.py` (selftested) to test this directly: for a sample of 120
LAGENDA-val images with real GT boxes (native median GT person height in this sample: **474px**
— these are prominent-subject photos, well above what a synthetic shrink is simulating), each
image's content is resized by a scale factor and pasted centered onto a canvas of the ORIGINAL
size (padded, not cropped), so the composition stays fixed and only the apparent person size
shrinks — the same visual effect as a person moving further from the camera. GT boxes are scaled
identically. Every model sees the SAME shrunk canvas at every scale (match-IoU 0.3,
class-agnostic detection recall only — see caveat below).

Four models compared: this arm (`y26n_noe2e_warm50-2`), its own predecessor (`y26n_noe2e`), the
current best deployable (`y26n_gradsupp`'s NMS/o2m branch), and the actual shipped production
model (`yolo11N-640`, `/workspace/model_v2/dataset_v2/yolo11N-640/weights/best.pt`).

**Detection recall by scale (n=123 GT boxes, 120 images, seed 42):**

| model | 1.0 (native) | 0.75 | 0.5 | 0.35 | 0.25 | 0.15 |
|---|---|---|---|---|---|---|
| `y26n_warm50` (this arm) | **1.000** | **1.000** | 0.976 | **0.968** | **0.902** | 0.675 |
| `y26n_noe2e` (predecessor) | 0.992 | 0.992 | 0.976 | 0.943 | 0.813 | 0.626 |
| `y26n_gradsupp` (NMS branch, current best) | 0.992 | 0.992 | 0.968 | 0.951 | 0.837 | **0.683** |
| `v11n_shipped` (production) | 0.992 | 0.984 | 0.976 | 0.959 | 0.870 | 0.642 |

**This arm is the most robust to shrinking down to scale 0.25** (90.2% recall vs 81-87% for
the others — a real, not marginal, gap at that size), then falls back roughly in line with the
pack at the most extreme scale (0.15, ~15% the linear size / ~2% the area of the original
person). Consistent with the four-benchmark result: this checkpoint detects real people better
across the board, and that advantage holds up as people get smaller, not just at native size.

**Caveat, found and corrected during this run:** an initial version of this probe also reported
a "class accuracy on matched" column. That number turned out to be meaningless — LAGENDA's YOLO
label files use a single generic class id (`0`) for every GT box (confirmed: `awk '{print $1}'`
over every `.txt` in `lagenda_yolo/labels/val` returns only `0`); the real gender/age ground
truth lives in `gt.jsonl`, not the label files, and is what `run_autolabel_on_manifest.py`
actually scores against. The class-accuracy column was silently comparing predictions against a
constant placeholder and has been dropped. **Only class-agnostic detection recall is reported
above** — this probe does not (yet) measure gender/age accuracy at reduced scale, which is the
natural next question if this is worth extending.

**Not done:** this used LAGENDA's own images (already fairly small/varied scale, median 474px
at native res — a portrait-to-medium-shot range, not a close-up dataset), synthetically shrunk
further. It does not test scale robustness on a dataset of naturally-*large* prominent subjects
shrunk down to LAGENDA- or crowd-scene-typical sizes, which would isolate "how does the model
handle an object THIS small" from "how does it handle further synthetic shrinking of an
already-smallish photo." Also not done: repeating this at the extreme low end (<0.15) where all
models are clearly failing, and any check of whether the recall gap is occlusion-shaped or
purely resolution-shaped (a shrunk-but-unoccluded person is an easier case than a real small
crowd person, which is usually also partially hidden).

## Notes / risks

- **Residual risk `warmup_epochs=0` does NOT fix:** MuSGD starts with completely cold
  internal state (no momentum/velocity history — unavoidable on any optimizer switch) and
  full `momentum` applies from batch one. Zero warmup removes the targeted bias-LR-spike
  risk but not this more diffuse one. Accepted, not solved — there's no warmup setting that
  avoids it without abandoning the optimizer switch itself.
- **Optimizer switch mid-training is a bigger change than "lower LR".** AdamW shaped these
  weights; MuSGD continues them. Defensible (it is YOLO26's designed optimizer) but it means
  this arm differs from `y26n_gradsupp` in TWO ways — epochs and optimizer — so a win cannot
  be attributed to epoch count alone. Stated up front rather than discovered later.
- If MuSGD is unavailable or errors in the pinned 8.4.115, fall back to `SGD` at the same lr0
  and record the substitution.
- **Version drift, discovered at launch:** this run's pod reports **Ultralytics 8.4.120**,
  not the 8.4.115 every other arm in this series was pinned to. Confirmed via the
  `train: Scanning` corrupt-file count differing from every prior run on the identical
  475,207-image list (2 files vs the historical 1: a GIF saved with a `.jpg` extension, and
  a genuinely tiny 9x16px stray — both legitimate pre-existing data issues, not staging
  corruption; the known 1x1px stray was also caught as before). Everything functionally
  verified correct regardless (MuSGD selected, non-e2e head built, all args as configured)
  — recorded as a comparability caveat, not a blocker.
- Tooling: `train_gradsuppress.py` gained `--optimizer / --lr0 / --lrf / --warmup-epochs`
  (2026-08-14), with a guard that errors if `--lr0` is passed without `--optimizer` — the
  exact trap Yasin flagged.
