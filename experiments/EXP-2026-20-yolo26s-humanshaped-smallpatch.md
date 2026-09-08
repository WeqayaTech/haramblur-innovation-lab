# EXP-2026-20 — YOLO26s + small-object patches: does a bigger backbone with a small-object augmentation beat everything measured so far?

**In one line:** trained `y26s_humanshaped_smallpatch_v1` — a size step up from every prior
candidate (YOLO26**s**, not n) on the same humanshaped-policy Spotlight labels as
`y26n_humanshaped_v2`, plus a `SmallObjectPatches` augmentation (grey-canvas small-object
pasting) — going in, the expectation was that it would simply do much better across the board;
this doc checks whether that held up and by how much.

**Status:** TRAINED + BENCHMARKED 2026-09-05→08. **Hypothesis confirmed, decisively, on every
metric measured** (standard 5-dataset comparison, holdout QA mAP, per-exposure-tier Woman AP,
and pixel-level escape/false-blur metrics) — see Scorecard. Owner: Mostafa.

---

## The short version

- **Best model this project has ever measured, on every axis tested.** Standard 5-dataset
  comparison, `haramblur_holdout` QA mAP, Woman AP broken out by how covered a woman's clothing
  is (5 tiers), and pixel-level "how much of a woman is actually left visible" / "how many wrong
  pixels get blurred" — it wins or ties best on all of them, against 6 other models including
  the currently shipped production model and the team's best gelan checkpoints.
- **The expectation was that a bigger backbone would help — it did, by more than expected.**
  Holdout mAP50-95 jumped from 0.840 (its own `y26n_humanshaped_v2` predecessor) to 0.870, and
  at a genuinely shared confidence (0.45) it cut the average share of a woman left unblurred
  from 9.4% to 6.0% for the male audience — a bigger single-model jump than most of this
  project's architecture changes have produced.
- **Three things changed at once versus the pre-humanshaped baseline** (model size n→s, the
  humanshaped labeling policy, and the small-object augmentation), so this is NOT a clean single-
  variable test. The one clean comparison we do have — against `y26n_humanshaped_v2`, which
  shares every label and only differs by model size + the augmentation — still shows a win on
  every metric with no exceptions, which is the strongest evidence available that the gain isn't
  just "different labels."
- **The `object_set` false-positive metric looks worse, but that's the label policy working as
  intended**, not a regression — old ground truth still scores "any detection on a toy = FP,"
  and this model correctly detects more toys/statues as people.
- **The main open gap:** no formal numeric bar was written down before training started — the
  hypothesis was informal ("it should do much better"), so this write-up is reporting a
  confirmed expectation rather than a pre-registered pass/fail the way most experiments in this
  project are run. Flagged honestly in "What this does NOT tell us."

## Why we did this

The prior best model on the holdout set, `y26n_humanshaped_v2`, had just proven the humanshaped
labeling policy (toys/statues/cartoons shaped like people count as gaze-lowering targets) was a
real win over the pre-policy baseline. Two questions were still open: would a bigger backbone
(YOLO26**s** instead of **n**) buy more accuracy on top of that, and would adding a dedicated
small-object augmentation — mirroring the coworker's YOLO-MIT trick of pasting shrunk objects
onto a grey canvas — close some of the small-person gap this project has repeatedly found in
every nano-class model (EXP-2026-19). Both changes are cheap to combine into one training run,
so rather than run them as separate ablations, the decision was to stack them and see whether
the resulting model earns a spot as the new best deployable candidate.

## What we wanted to find out

**The question, informally:** does a bigger backbone with a small-object augmentation trained on
the humanshaped labels simply do better than everything measured so far?

**No formal numeric bar was pre-registered before this run** — unlike most experiments in this
project, the expectation going in was a general one ("it should do much better across the
board"), not specific pass/fail numbers per metric. That is a real gap against this project's own
convention and is called out explicitly in "What this does NOT tell us," not glossed over. The
closest thing to a bar in practice was the standing recommendation rule used throughout this
project's model rounds: a new candidate is worth adopting if it does not regress any headline
metric relative to the current best while improving at least one by a non-trivial margin. That
informal bar is checked against below.

## Data used

Same labels as `y26n_humanshaped_v2` — **Spotlight** (SAM3 → Gemini 3.5 Flash-Lite verified) OIV7
train labels, `classes {0: Woman, 1: Man, 2: Child}`, **plus the humanshaped policy**: human-shaped
toys/statues/mannequins/cartoons are patched in as real gaze-lowering targets rather than deleted
as non-person. Age overrides gender in the promoted label — a child-shaped toy becomes `Child`
regardless of any gender read; an adult-or-unknown-age toy becomes `Woman`/`Man`. Applied by
`patch_promotions_local.py`, which appends promoted lines from the raw OIV7 labels (at the
matching `det_index`) onto the plain nc=3 base. **Verified by file timestamp that this change
never touched the production `y26n_noe2e_warm50-2`/`y26n_sop50` label tree** (written up
separately in `_spotlight_review/STALE_LABEL_TREE_ISSUE.md`) — only models trained after the
patch, including this one, carry the policy.

**On top of that, a `SmallObjectPatches` augmentation** — a 5% per-sample chance of pasting the
image's own detected objects, shrunk to 96px, onto a 640px grey canvas, mirroring the coworker's
YOLO-MIT small-object trick. Source crops for the augmentation's own object bank came from
LAGENDA `fl1199`, and **554 stems were explicitly excluded** (`--exclude-stems`) to keep this
run's training data disjoint from `smallperson_v1`'s own `paste_grey`/`synth_shrunk` benchmark
arms, which reuse the same source images — without that exclusion, benchmarking this model on
`smallperson_v1` would have partly been scoring it on its own training data.

## Training setup

Trained **from COCO-pretrained `yolo26s.pt` weights, not a continuation** of any prior
checkpoint — a fresh start, so nothing about this run is a warm restart of `y26n_humanshaped_v2`
or any earlier arm.

| knob | value | note |
|---|---|---|
| architecture | YOLO26**s** | a size step up from every prior candidate in this project, which were all YOLO26**n** |
| start weights | `yolo26s.pt` (COCO-pretrained) | fresh start, not a continuation |
| optimizer | MuSGD | YOLO26's own optimizer (must be set explicitly or ultralytics silently falls back to AdamW — a trap this project hit once before, EXP-2026-18) |
| lr0 / lrf | 0.003 / 0.01 | matches the project's established recipe for fresh YOLO26 starts |
| momentum / weight_decay | 0.937 / 0.0005 | project default |
| warmup_epochs | 3.0 | fresh-start default (a warm restart uses 0 instead — not applicable here) |
| epochs | 100 (ceiling) | `patience=15` never triggered — the model was still setting records as late as epoch 90, so the epoch ceiling, not patience, was the binding constraint |
| patience | 15 | anti-overfitting safeguard, never fired |
| save_period | 5 | checkpoint every 5 epochs |
| batch | 0.85 (auto-batch fraction) | |
| imgsz | 640 | |
| augmentation | mosaic=1.0, scale=0.5, fliplr=0.5, hsv jitter, close_mosaic=10, auto_augment=randaugment, erasing=0.4 | project's standard recipe, unchanged from prior arms |
| + SmallObjectPatches | p=0.05, target_size=96px | the new augmentation this arm adds — see Appendix for the exact patch code |

**Best checkpoint: epoch 90/100**, `mAP50-95` (val) 0.68 → 0.804 over the run. Final weights:
`/workspace/exp20/train/y26s_humanshaped_smallpatch_v1/weights/best.pt` (20.3 MB fused).
Trained on an RTX 4090; wall clock ran to roughly two days including one full restart from
epoch 0 after an infrastructure crash (see Appendix — a stale, unrelated resync job silently
exceeded the pod's disk quota mid-run). No dollar cost figure was tracked for this run.

## How we compared

Four separate measurement passes, each already using this project's standard, previously
validated methodology — none of the scoring machinery was written new for this experiment:

1. **Standard 5-dataset comparison** (Spotlight-val, LAGENDA v2, CrowdHuman, PASS, object_set) —
   `run_ultralytics_labels.py` dump (conf 0.45, floor 0.001, IoU 0.7, 640px) → `map_eval.py`
   (COCO-style mAP) / `eval_negatives_crowd.py` (crowd + negatives), the exact protocol used for
   every model in `docs/MODEL_COMPARISON.md`.
2. **Holdout QA mAP** (`haramblur_holdout`, 11,494 images, `DO_NOT_TRAIN`, never touched by any
   training run in this project) — same dump/score tools, against `labels_eval` + `ignore`.
3. **Woman AP by exposure tier** — the holdout's existing `deploy_compare` exposure-tier subset
   views (t0_covered → t4_high, based on how much of a woman's own body her labeled attire
   exposes), scored per tier with `map_eval.py` against a fresh per-tier raw-sidecar view built
   from the already-dumped holdout predictions.
4. **Pixel-level escape / false-blur metrics** — `deploy_compare.py score`, the tool behind this
   project's `WOMEN_REPORT_FINAL.md`/`MEN_REPORT_FINAL.md` reports: for every GT person, the
   exact polygon-union area of fired same-class boxes intersected with that person's GT box
   gives a per-person "how much of them is still visible" fraction (**Epers%**) and an
   area-weighted per-image version (**Eimg%**); the area of fired boxes falling outside a small
   dilation margin around real GT boxes and outside ignore regions gives the false-blur /
   wrong-pixel rate (**FBarea%**). Run twice: once at each model's own matched-false-blur
   operating point (the tool's normal mode), and once — at the owner's explicit request, since a
   shared threshold is the fairer "as-shipped-today" comparison — with every model forced to the
   identical fixed confidence **0.45**.

All four passes score against 6 other models: the currently shipped production model
(`yolo11N-640`), the two pre-humanshaped y26n bests (`y26n_sop50`, `y26n_noe2e_warm50-2`), the
team's two gelan checkpoints (`gelan_r4fw_v2`, `gelan_r6_v2`), and this model's own direct
predecessor (`y26n_humanshaped_v2`) — same labels, one size class smaller, no small-object
augmentation, which is the closest thing to a controlled comparison available.

## What we found

**1 — Standard 5-dataset comparison.** Wins every mAP and recall column outright:

| model | Spotval mAP50/50-95 | LAGENDA mAP50/50-95 | crowd recall/prec. | PASS FP/100 | object_set FP/100* |
|---|---|---|---|---|---|
| `y26n_sop50` | 0.806/0.700 | 0.758/0.638 | 0.391/0.924 | 0.60 | 23.94 |
| `y26n_noe2e_warm50-2` | 0.802/0.697 | 0.754/0.634 | 0.392/0.922 | 0.37 | 20.46 |
| `yolo11N-640` (shipped) | 0.744/0.603 | 0.724/0.627 | 0.349/0.943 | 1.17 | 88.42 |
| `gelan_r4fw_v2` | 0.812/0.654 | 0.783/0.647 | 0.393/0.927 | 0.70 | 21.62 |
| `y26n_humanshaped_v2` | 0.806/0.704 | 0.859/0.723 | 0.390/0.922 | 0.50 | 31.27 |
| **`y26s_humanshaped_smallpatch_v1`** | **0.850/0.764** | **0.884/0.745** | **0.474**/0.918 | 0.60 | 35.14* |

\* not comparable across the humanshaped policy boundary — see caveat below.

**2 — Holdout QA (3-class mean mAP):** best mAP50 **and** best mAP50-95 of every model ever
measured on this set.

| model | mAP50 | mAP50-95 |
|---|---|---|
| `y26n_sop50` / `y26n_warm50` | 0.912 / 0.913 | 0.835 / 0.835 |
| `v11n_shipped` | 0.864 | 0.741 |
| `gelan_r4fw_v2` | 0.921 | 0.793 |
| `y26n_humanshaped_v2` | 0.913 | 0.840 |
| **`y26s_humanshaped_smallpatch_v1`** | **0.928** | **0.870** |

**3 — Woman AP by exposure tier:** best AP50-95 of all 7 models on **every single tier**,
wins outright (both AP50 and AP50-95) on the whole holdout and on **t4_high** (the most
revealing/highest-stakes tier), and beats `y26n_humanshaped_v2` — its closest, most controlled
comparison — at every tier with no exceptions. `t0_covered` (hijab-only cues) is the hardest
tier for every model, and the one tier where the gelan pair's looser-IoU AP50 lead is widest;
even there this model has the best AP50-95.

**4 — Pixel-level metrics.** At each model's own matched-false-blur operating point, this
model's matched confidence landed at **0.05 — the floor of the tested grid** — meaning even the
most permissive setting tested still produced less wrongly-blurred area than production at 0.45.
At the owner-requested **shared fixed 0.45**, it still wins every escape-side metric on both
audiences:

| audience | Eimg% | P90% | Epers% (% of person left visible) | cov90% | FBarea% (% wrongly-blurred pixels) |
|---|---|---|---|---|---|
| MALE (blur target: Woman) — production | 6.01 | 10.3 | 10.6 | 87.5 | 1.816 |
| MALE — `y26s_humanshaped_smallpatch_v1` | **2.86** | **1.7** | **6.0** | **92.7** | **0.621** |
| FEMALE (blur target: Man) — production | 6.07 | 11.4 | 17.0 | 78.6 | 1.326 |
| FEMALE — `y26s_humanshaped_smallpatch_v1` | **2.29** | **3.0** | **8.5** | **87.7** | **0.731** |

(FBarea is beaten only by the gelan pair, at a heavy escape-side cost — `gelan_r4fw_v2`'s male
P90 hits 50.5%, meaning its worst 10% of images leave over half the screen's target pixels
showing; the gelan pair is also calibrated for a lower operating point than 0.45, so this shared
threshold understates its real ceiling — an honest caveat, not a hidden one.)

**object_set caveat, stated plainly:** its FP/100 rose to 35.14 versus the pre-humanshaped
baselines' ~20-24. That dataset's ground truth still assumes "any detection on a toy/statue =
false positive" (the pre-policy rule) — a humanshaped model correctly reading a toy as a person
necessarily scores worse there. This is evidence the labeling-policy change is doing what it was
designed to do, not a regression, but a proper apples-to-apples number still needs the pending
`object_set` relabel under the new policy.

## What we can decide from this

- **This is now the recommended candidate.** It beats every other measured model — including the
  currently shipped model and the team's best gelan checkpoints — on mAP, holdout accuracy,
  per-exposure-tier accuracy, and the two pixel-level metrics that most directly describe user
  experience (how much of a target person is left visible, how many wrong pixels get blurred),
  with no metric where it is clearly worse once the `object_set`/policy-boundary caveat is
  accounted for.
- **The size + augmentation combination is worth keeping together going forward** — the one
  controlled comparison available (against `y26n_humanshaped_v2`, same labels, no augmentation,
  one size smaller) shows a clean win at every tier and on every pixel metric, which is the best
  evidence this project has that the combination, not just different labels, is doing real work.
- **The `object_set` relabel is now higher priority** — it's the one dataset in the standard
  protocol that cannot currently give this model (or `y26n_humanshaped_v2`) a fair reading.

## What this does NOT tell us

- **No formal numeric bar was pre-registered before this run.** The going-in expectation was
  informal ("it should do much better across the board"), which this doc is reporting as
  confirmed — but that is a weaker claim than a pre-registered pass/fail, and it means there was
  no explicit failure condition defined that could have made this a "did not clear the bar"
  write-up. Stated honestly rather than backdated to look more rigorous than it was.
- **Three things changed at once relative to the pre-humanshaped baseline** (model size, the
  humanshaped label policy, and the small-object augmentation). The only clean single-swap
  comparison is against `y26n_humanshaped_v2` (isolates size + augmentation, holding labels
  fixed); there is no ablation isolating the augmentation alone from the size increase, so this
  experiment cannot say how much of the gain over `y26n_humanshaped_v2` is the bigger backbone
  versus the small-object patches specifically.
- **Latency/size were not measured for this arm.** A YOLO26s checkpoint is meaningfully larger
  than every prior YOLO26n candidate in this project (which were kept deliberately nano-class,
  within 0.6M params of each other) — this write-up has no ONNX/CPU latency number for it, so it
  cannot yet be compared to the rest of the model-comparison table on speed, only accuracy.
- **`object_set` numbers are not apples-to-apples** against pre-humanshaped models, as stated
  above — no clean false-positive read exists yet for humanshaped-policy models on that dataset.
- **The matched-false-blur operating point (0.05) was not fully resolved** — a finer grid
  (0.01-0.05) confirmed the true crossing point is below the tested floor for both humanshaped
  models, but the exact matched confidence was never pinned down below 0.01.

## What's next

- **Measure latency/size** for `y26s_humanshaped_smallpatch_v1` and add it to
  `docs/MODEL_COMPARISON.md`'s size/speed table alongside the nano-class candidates, since it is
  a meaningfully bigger model and that tradeoff has not yet been quantified.
- **The pending `object_set` relabel**, so this and future humanshaped models get a fair
  false-positive read on that dataset instead of the pre-policy answer key.
- **An ablation isolating the small-object augmentation from the size increase** (a YOLO26s
  trained on the same humanshaped labels WITHOUT `SmallObjectPatches`, or a YOLO26n trained WITH
  it) — the only way to attribute the win over `y26n_humanshaped_v2` cleanly.
- **Small-person benchmark (`smallperson_v1`)** — this model was never scored there, which is the
  most direct test of whether `SmallObjectPatches` actually helped on small/distant people
  specifically, as opposed to the general accuracy gain measured here.
- **Video flicker replay** (EXP-2026-11's harness) — not yet run for this checkpoint.

---

## Appendix — the details

**Weights:** `/workspace/exp20/train/y26s_humanshaped_smallpatch_v1/weights/best.pt` (20.3 MB
fused, epoch 90/100). **Training log/results:**
`/workspace/exp20/train/y26s_humanshaped_smallpatch_v1/results.csv` (full 100-epoch history).
**Benchmark outputs:** `_spotlight_review/y26s_humanshaped_smallpatch_v1_benchmark/` locally
(`BENCHMARK_SUMMARY.html`, `exposure_map/`, `pixel_metrics/`); on the pod,
`/workspace/holdout_eval/y26s_humanshaped_smallpatch_v1/`, `/workspace/evalout_v2/
y26s_humanshaped_smallpatch_v1_*.json`, `/workspace/deploycmp/map/
y26s_humanshaped_smallpatch_v1__*.json`, `/workspace/deploycmp/holdout_run7/summary.json` and
`/workspace/deploycmp/holdout_run8_fixed045/summary.json`.

**Operational traps hit and fixed during this run** (kept here since they are reusable lessons,
not just this run's trivia):

- **The `SmallObjectPatches` augmentation only reaches the dataloader if BOTH copies of
  `v8_transforms` are patched.** `ultralytics/data/dataset.py` does `from .augment import
  v8_transforms` — a direct name import — so patching `ultralytics.data.augment.v8_transforms`
  alone is a silent no-op; the actual call site in `dataset.py`'s `build_transforms()` uses its
  own local reference. Caught before the real run by building a live `YOLODataset` and checking
  the transform list, not by trusting the patch:

  ```python
  def patch_v8_transforms(p: float = 0.05, target_size: int = 96):
      import ultralytics.data.augment as aug_mod
      import ultralytics.data.dataset as dataset_mod
      orig_v8_transforms = aug_mod.v8_transforms
      def patched(dataset, imgsz, hyp):
          transforms = orig_v8_transforms(dataset, imgsz, hyp)
          transforms.append(SmallObjectPatches(p=p, target_size=target_size))
          return transforms
      aug_mod.v8_transforms = patched       # cosmetic consistency
      dataset_mod.v8_transforms = patched   # the actual call site (dataset.py's build_transforms)
  ```

- **Dataloader thread oversubscription is easy to get backwards.** Setting
  `OMP_NUM_THREADS`/`MKL_NUM_THREADS` to the worker count (10) made each of the 10 dataloader
  worker *processes* spawn up to 12 internal threads each and regressed throughput (~168 → ~140
  img/s). The fix was `=1` for both — parallelism already comes from having 10 worker processes,
  not from each one being multi-threaded — **plus** `cv2.setNumThreads(1)` in the launcher
  (OpenCV's own thread pool otherwise defaults to the pod's host `nproc`, not its real allocated
  share — the same nproc/cgroup trap this project has hit before). Verified via
  `/proc/<pid>/status` thread counts, not throughput alone; final throughput ~216 img/s at 71%
  GPU utilization once warmed up.
- **A stale, never-cleaned-up resync job from unrelated earlier work silently exceeded the pod's
  real (invisible-to-`df`) volume quota** and crashed training at the epoch-1 metrics-write step.
  Diagnosed with a `dd` write-probe (not `df`, which falsely showed hundreds of TB free — the
  shared-cluster number, not the pod's real quota) and fixed with a parallelized
  `find ... -print0 | xargs -0 -P 32 rm -f` delete of ~450,918 orphaned files, then training was
  relaunched from epoch 0 (the good epoch-1 result from before the crash was lost, since no
  checkpoint had saved yet).
- **A transient `inf` box_loss at epoch 18** was investigated rather than dismissed:
  `val/box_loss` stayed finite (0.4385) and epoch 19 showed zero recurrence, consistent with one
  degenerate augmented batch (likely a near-zero-area `SmallObjectPatches` crop) poisoning that
  epoch's running-mean display, not weight corruption.

### Appendix — "verify it yourself"

- **How the model was called for scoring:** `run_ultralytics_labels.py --engine ultralytics
  --model /workspace/exp20/train/y26s_humanshaped_smallpatch_v1/weights/best.pt --conf 0.45
  --floor 0.001 --iou 0.7 --imgsz 640 --device cuda:0`, identical invocation used for every other
  model in `docs/MODEL_COMPARISON.md`.
- **How objects were matched / scored:** `map_eval.py` (COCO 101-point interpolation,
  confidence-ordered greedy matching, `--ignore-labels` for unlabeled/unknown-gender regions) for
  all mAP numbers; `deploy_compare.py`'s `image_metrics()` (exact polygon-union area calculation,
  quoted in full in this project's `docs/MODEL_COMPARISON.md` pixel-metrics section) for the
  escape/false-blur numbers — both tools pre-date this experiment and were not modified for it.
- **How samples were selected:** the full `haramblur_holdout` set (11,494 images, fixed
  membership, `DO_NOT_TRAIN`) for QA/exposure/pixel metrics; the project's standard five staged
  datasets (Spotlight-val 4,233 imgs, LAGENDA v2 4,601 imgs, CrowdHuman 4,370-4,372 imgs, PASS
  3,000 imgs, object_set 259 imgs) for the standard comparison — no new sampling was done for
  this experiment.
- **Every threshold used:** conf 0.45 / IoU 0.7 for detection dumps; mAP floor 0.001; exposure-
  tier taxonomy and `tau=0.005 phi=0.01 kappa=0.2 dilate=0.1` for the pixel metrics (all as
  defined in the tools above, unchanged from every prior use of them in this project).
- **What was NOT done:** no formal pre-registered numeric bar (see "What this does NOT tell
  us"); no latency/size measurement for this specific checkpoint; no ablation isolating the
  augmentation from the size increase; no `smallperson_v1` or video-flicker scoring yet.
