# Experiment: Does splitting gender and age into two independent labels beat one entangled 3-class head?

**In one line:** Retrain YOLO26n so every person carries two labels — a gender and an age — instead of one `{Woman, Man, Child}` class, and see whether separating the two questions buys back crowd recall and stops the teen band corrupting the gender read.

**Status:** **STOPPED at epoch 66 of 80 by owner decision, 2026-08-17.** Measured and documented as a baseline; not resumed. 1 arm (~$14 of the $25 cap) · Owner: Mostafa · frozen weights `/workspace/exp17/train/y26n_twoaxis2/weights/last.pt` (epoch 66), with `epoch{0,10,20,30,40,50,55,60,65}.pt` alongside

---

## The short version

- **The mechanism works. The model does not clear the bar.** Every person really does get two independent
  readings and both are learned — but at epoch 66 the arm **fails 4 of the 6 pre-registered
  "must not regress" bars**, so by its own decision rule it is not adopted and `y26n_gradsupp` stands.
- **It wins crowd recall by more than anything we have measured: 47.3 vs 36.5** (+10.8; the bar was +3).
  It also wins LAGENDA all-people detection recall, 84.9 vs 74.8.
- **It pays for that in false positives and in gender.** Object-set false persons 18.2 % of images against
  the 11.2 % bar, PASS 1.00 per 100 against 0.23, gender on adults 89.9 % against 91.1 %, adult→Child leak
  0.335 % against 0.11 %.
- **Recall and false positives moved in the SAME direction, which is the signature of a lower operating
  point rather than a better model.** Box confidence here is the max over six channels and every person is
  trained to light up two of them, so conf 0.45 is not the threshold it is on a 3-class head. **The
  matched-FP sweep that would settle this was never run** — the run was stopped first. Until it is, the
  crowd-recall win is unproven.
- **The abstention channel — the reason for the whole design — did not learn.** `AgeUnknown` recall is
  **16.9 %**; the model answers Adult instead. The teen-band must-win bar (≥ 60 %) is missed widely.
- **Separating the axes revealed something the 3-class view cannot show:** 113 of 200 collapsed errors are
  *pure age failures wearing a gender label*. `Child → Woman` looks like gender confusion; in 58 of 64
  cases the gender was read correctly and the age was not.
- **Read every number as an 82.5 %-trained model.** The run stopped before `close_mosaic` (epoch 70), where
  classification metrics normally move most. The two FP gaps are large (1.6× and 4.3× the bars), so
  finishing seems unlikely to close them — but that is an opinion, not a measurement.

## Why we did this

Today the detector answers one 3-way question per person, so gender and age share a single channel. That costs us in three places we have already measured:

- **The teen band.** SAM3, Qwen2.5-VL-7B and y26n all fade between 13 and 17. With one entangled class, an uncertain age corrupts an otherwise certain gender — a person who is unmistakably a woman but ambiguously 15 has no way to say "sure about the gender, not about the age."
- **The 196,119 people Spotlight threw away.** They were dropped because the 3-class taxonomy has no slot for "person, gender unreadable" (median ~58 px). CLAUDE.md records this as the leading hypothesis for why y26n loses small-person recall against the shipped model.
- **Threshold tuning is spent.** `conf_sweep.py` showed conf 0.45 is already on the Pareto front for `y26n_gradsupp`, and per-viewer-gender thresholds collapse back to one config. Two axes give two genuinely independent knobs.

## What we wanted to find out

**Can one YOLO26n answer gender and age as two independent questions, without giving up anything on the two axes the product is judged on — false persons, and adults escaping the blur?**

Bars, written before running. Against `y26n_gradsupp` (the current recommendation) on the identical four-dataset protocol at imgsz 640 / conf 0.45 / NMS IoU 0.7:

**Must not regress (any failure = not adopted):**

| Measure | Dataset | Bar |
|---|---|---|
| Images gaining a false person | object set | ≤ 11.2 % |
| False persons per 100 images | PASS | ≤ 0.23 |
| Adult (GT ≥ 20) read as Child | LAGENDA | ≤ 0.11 % |
| Gender on adults | LAGENDA | ≥ 91.1 % |
| Detection recall | LAGENDA | ≥ 93.9 % |
| Recall | CrowdHuman | ≥ 36.5 |

**Must win at least one of (otherwise it is a tie and we keep the simpler model):**

- **Crowd recall ≥ 39.5** (+3 or better) — the 196k-unknowns hypothesis: restoring those people as gender-unknown-but-real should buy back small-person recall.
- **Teen band (GT 13–17): ≥ 60 % answered `AgeUnknown`** — the separation hypothesis: the model should learn to abstain rather than guess, while still committing on gender.

> **Pre-registration amendment, 2026-08-15 — recorded before training, after Phase 0, with no results in hand.** The teen bar was written assuming `AgeUnknown` would mostly *be* the teen band. Phase 0 showed it is 3.1 % teen band and 96.9 % labeler abstention, so the model will see only ~5,526 teen-band examples against ~173,146 unreadable-age ones. The bar is **kept unchanged** — moving it now would be exactly the goalpost-shifting this project forbids — but it must be read for what it now tests: whether abstention learned from one population (unreadable, small, occluded) **generalizes** to another (visible teenagers). A miss is therefore weaker evidence against the design than originally intended, and a hit is stronger evidence for it. The abstention ceilings below are unchanged and remain the guard.

**Abstention ceilings (so accuracy cannot be bought by refusing to answer):**

- Gender coverage ≥ 90 % on LAGENDA adults (i.e. `GenderUnknown` on ≤ 10 % of them).
- Age coverage ≥ 90 % on GT ≥ 18.

**Decision rule.** Any "must not regress" failure disqualifies the arm regardless of what it wins. If it passes all six and wins neither tiebreak, it is a tie and `y26n_gradsupp` stands. Only if it clears a must-win do we consider it for deployment — and then the browser work in the deployment note becomes real.

## How we did it

1. **The answer key.** Four frozen datasets, scored separately, never pooled (`docs/MODEL_EVAL_PROTOCOL.md`): LAGENDA (human apparent age + gender), CrowdHuman (exhaustively boxed crowds), PASS (verified-empty scenes), the 259 hand-verified doll/statue/toy images. Same physical copies, same matcher, same operating point as every model in `docs/MODEL_COMPARISON.md`.

2. **The labels.** Re-emitted from the Spotlight verdicts that already exist — **no new API spend and no relabeling** — and written to the pod's local disk as part of staging rather than to the network volume, since they are a deterministic $0 re-emit and the volume quota is invisible to `df`. `verdicts_batch.jsonl` stored Gemini's gender for *every* detection, including the ones the 3-class emit collapsed to `Child` and discarded the gender of. Each person becomes two rows with identical geometry:

   ```
   0 0.412 0.533 0.104 0.288     # gender: Woman
   3 0.412 0.533 0.104 0.288     # age:    Adult
   ```

   Classes `{0 Woman, 1 Man, 2 GenderUnknown | 3 Adult, 4 Child, 5 AgeUnknown}`.

3. **What we trained.** YOLO26n, 30 epochs, same recipe as EXP-2026-12/14, with a patched loss (`train_twolabel.py`). The patch is needed because the label format alone is not enough — see "Definitions that matter".

4. **How we compared.** The two-axis predictions are **collapsed back to the production label space** at decode time, so every existing scorer runs unmodified and the numbers drop straight into the comparison table. On top of that, the two axes are scored natively via the new `two_axis_full` translation scheme, which is the only way to see gender accuracy *on children* and age accuracy independent of gender.

5. **Definitions that matter.**

   - **Why the loss had to change.** The two-rows-per-person file is legal and loads fine, but `TaskAlignedAssigner` gives each anchor exactly one ground-truth box and a one-hot target. Under stock training the two rows *split* the person's anchors instead of sharing them, and no anchor is ever supervised on both axes. Measured, not assumed: the selftest asserts stock produces **1 hot channel per anchor** where ours produces **2**.
   - **Where `AgeUnknown` comes from — corrected by Phase 0.** The rule fires on two triggers: an estimated age inside **13–17** (the band EXP-2026-10 measured as close to a coin flip), or Gemini answering `age_group: unknown`. It was designed as the teen-band class. **Phase 0 measured it and it is not one.** Of 178,672 `AgeUnknown` people, the teen band accounts for **5,526 (3.1 %)** and the labeler's own abstention for **173,146 (96.9 %)**.

     So what the class actually is: **"the labeler could not read this person's age"** — overwhelmingly the same small, occluded population as the 222,795 `GenderUnknown`. The teen band is a rounding error on top, worth keeping because it is principled and costs nothing, but it is not the story and must not be written up as one.

     The band stays at 13–17. Narrowing it moves 3 % of the class; widening past 17 would relabel genuine adults (the histogram shows 18–19 sitting in the 15–19 bucket, and `estimated_age` is heavily round-number biased — 30–34 holds 346,188 while 40–44 holds 74,167, so Gemini answers "30", "35", "45"). Neither direction is worth a re-emit.
   - **Child cutoff stays 12**, unchanged from production and from `spotlight_run.CHILD_AGE_MAX`.
   - **The collapse is the blur policy**, written once: `Child` if the age axis says Child; otherwise the gender, with an unreadable age defaulting to **Adult** (= blur = the safe direction) and an unreadable gender kept as its own class so it scores as an abstention rather than as a wrong answer.

## What we found

### Phase 0 (complete) — the labels, before any training

Counted over all 1,579,649 Spotlight verdicts. Two predictions made before running were confirmed exactly: `not a person` 192,388 and `people` 1,387,261 (the 3-class emit's 1,191,142 kept **plus** the 196,119 unknown-gender people it dropped).

| | count | share of people |
|---|---:|---:|
| Adult | 1,147,057 | 82.69 % |
| Child | 61,532 | 4.44 % |
| AgeUnknown | 178,672 | 12.88 % |
| — of which teen band | 5,526 | 3.1 % *of AgeUnknown* |
| — of which labeler abstained | 173,146 | 96.9 % *of AgeUnknown* |
| Woman | 355,559 | 25.63 % |
| Man | 808,907 | 58.31 % |
| GenderUnknown | 222,795 | 16.06 % |

Four things that change how this experiment should be read:

- **`AgeUnknown` is a labeler-abstention class, not a teen class** (see "Definitions that matter"). Under the old 3-class labels those 173,146 people were all asserted **Adult**, because `merge()` falls through to gender whenever `age_group != child`. The old labels claimed an age nobody could read.
- **The class collapses to Adult, so it has no product downside.** A person the model marks `AgeUnknown` is still blurred if their gender is the opposite one — identical behaviour to today. What changes is only that training no longer forces a guess.
- **It gives uncertainty a home on the safe side.** Previously an uncertain person could land only in Adult or Child, and Child is the direction that lets someone escape the blur. A third option that behaves as Adult means the model can express doubt *without* routing it into the consequential error. If the adult→Child leak improves, this is the mechanism.
- **Gender is imbalanced 2.27 : 1 toward Man** (808,907 vs 355,559). Not new — the 3-class labels carried it too — but Woman is the blur target for a male viewer, so the product's most important class is the scarcer one, and it points the same direction as the unattributed 2.5× Woman→Man Spotlight asymmetry.

Also observed, not acted on: `estimated_age` is heavily round-number biased (30–34 → 346,188 vs 40–44 → 74,167), and a handful of garbage ages exist (20 negatives, one ~3000). All resolve to Adult and are harmless, but nobody should use `estimated_age` finely.

### Training results

**How far it got.** 66 of 80 epochs, then stopped by owner decision. Two interruptions, both recorded
because they cost real time: the container OOM-killed the dataloader workers at epoch 51 and the parent
survived *deadlocked* — a log that simply stopped growing, unnoticed for 12.8 h (`oom_kill 7`, container
ceiling 56.8 GiB; ultralytics gives the val loader `workers * 2`, so `--workers 8` meant 24 persistent
workers whose copy-on-write share of the 2.5 GB label cache grows monotonically). It resumed at 4 workers
and ran clean to epoch 66. `close_mosaic` (epoch 70) never engaged.

**A `--resume` trap found on the way, now fixed.** `Model.train()` rewrites `resume=True` to the path the
YOLO object was built from, and if that file carries no epoch/optimizer state it **silently starts a new
run** behind one warning line (`ultralytics 8.4.115 engine/model.py:805-814`). `--weights yolo26n.pt` is
exactly such a file (`epoch=-1`, no optimizer), so the resume command as originally written would have
discarded 50 epochs and retrained from scratch. `train_twolabel.py` now resolves the checkpoint itself,
proves it is resumable, and carries a selftest contract for it.

#### Scorecard against the pre-registered bars

Epoch 66, imgsz 640 / conf 0.45 / NMS IoU 0.7, same four datasets and same scorers as every row in
`docs/MODEL_COMPARISON.md`.

**Must not regress — 4 of 6 FAILED:**

| Measure | Dataset | Bar | Epoch 66 | |
|---|---|---|---|---|
| Images gaining a false person | object set | ≤ 11.2 % | **18.15 %** (30.1 FP/100) | ❌ |
| False persons per 100 images | PASS | ≤ 0.23 | **1.00** | ❌ |
| Adult (GT ≥ 20) read as Child | LAGENDA | ≤ 0.11 % | **0.335 %** (13 / 3,883) | ❌ |
| Gender on adults | LAGENDA | ≥ 91.1 % | **89.85 %** | ❌ |
| Detection recall (labeled people) | LAGENDA | ≥ 93.9 % | 97.39 % | ✅ |
| Recall | CrowdHuman | ≥ 36.5 | 47.30 | ✅ |

**Must win at least one — 1 of 2:**

| | Bar | Epoch 66 | |
|---|---|---|---|
| Crowd recall | ≥ 39.5 | **47.30** (+10.8 over `y26n_gradsupp`) | ✅ |
| Teen band answered `AgeUnknown` | ≥ 60 % | ~17 % (`AgeUnknown` recall 16.9 %) | ❌ |

**Abstention ceilings — both PASS**, so the accuracy above was not bought by refusing to answer:

| | Bar | Epoch 66 | |
|---|---|---|---|
| Gender coverage on adults | ≥ 90 % | 98.5 % (`GenderUnknown` on 18 / 1,220) | ✅ |
| Age coverage on GT ≥ 18 | ≥ 90 % | 100 % (zero Adult → `AgeUnknown`) | ✅ |

**Decision rule applied as written — any must-not-regress failure disqualifies the arm regardless of what
it wins. Four failed. Not adopted; `y26n_gradsupp` stands.**

#### Against the field

| run | objects %img-FP ↓ | PASS FP/100 ↓ | crowd recall | crowd prec | LAGENDA det (all people) | gender on adults | leak ≥20 ↓ | child recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `y26n_gradsupp` | **11.2** | **0.27** | 36.5 | **94.0** | 74.8 | **91.1** | 0.37 | **81.3** |
| `gelansfav14_gemlb_v1` | 18.2 | 0.23 | 46.2 | 92.7 | — | 90.6 | — | 85.8 |
| `gelannfav14r4_gemlb_v2` | 16.2 | 0.33 | 41.1 | — | 79.6 | 92.4 | **0.16** | 80.9 |
| **`y26n_twoaxis2` @ep66** | 18.15 | 1.00 | **47.30** | 92.7 | **84.91** | 89.85 | 0.335 | 69.1 |

Crowd recall (heavy occlusion 25.1, partial 41.6, light 55.7) and all-people detection recall are the best
this project has measured. Everything on the false-positive axis is the worst of the four.

#### The two axes, scored separately (epoch 64 · 800 LAGENDA images · 1,220 people)

The view no other tool in the repo can produce — `vlm-cluster/twoaxis_report.py`, written for this head
alone, with the visual report at `experiments/assets/exp17/twoaxis_report_epoch64.html`.

| layer | result |
|---|---|
| detection (labeled people, IoU ≥ 0.5) | 1,220 / 1,221 = 99.9 % |
| **gender axis** | 1,066 / 1,220 = **87.4 %** |
| **age axis** | 925 / 1,220 = **75.8 %** |
| collapsed product label | 1,020 / 1,220 = 83.6 % |

Gender axis: `Woman→Man` 99 against `Man→Woman` 37 — a **2.7 : 1 asymmetry reproducing the unattributed
2.5× Woman→Man Spotlight correction ratio inside the trained model.** Woman is the blur target for a male
viewer, so this is the product-critical direction. Age axis: 187 of its 295 errors are the abstention band
collapsing (`AgeUnknown→Adult` 158, `AgeUnknown→Child` 29).

**What actually causes the collapsed errors** — the decomposition only a two-axis view can do:

| n | collapsed error | gender axis | age axis | to blame |
|---:|---|---|---|---|
| 58 | Child → **Woman** | Woman → Woman ✓ | Child → Adult ✗ | **age only** |
| 27 | Woman → Man | Woman → Man ✗ | Adult → Adult ✓ | gender only |
| 26 | Child → **Man** | Man → Man ✓ | Child → Adult ✗ | **age only** |
| 20 | Man → Child | Man → Man ✓ | AgeUnknown → Child ✗ | age only |
| 17 | Man → Woman | Man → Woman ✗ | Adult → Adult ✓ | gender only |

**Totals: age only 113 · gender only 44 · both 43.** In the collapsed 4-class view — the only view every
other model supports — all 64 `Child → Woman` cases read as gender confusion. Fifty-eight are age failures.
That misattribution is the strongest argument the experiment produced *for* keeping the axes apart,
independent of whether this checkpoint is ever adopted. The 20 `AgeUnknown → Child` cases are
13–17-year-olds the policy says to treat as adults, predicted Child and so never blurred — the escape
direction, at the teen band.

#### Validation mAP (epoch 64; the epoch-66 re-run did not finish before the pod was released)

Spotlight OIV7 val, 4,232 images / 8,035 GT, floor 0.001, maxDets 100, 702 ignore regions — identical
protocol and tool as every row below.

| run | mAP50 | mAP50-95 | Woman | Man | Child |
|---|---:|---:|---:|---:|---:|
| `yoloe_n_gradsupp` | 0.8312 | 0.7150 | 0.858 | 0.878 | 0.757 |
| `y26n_gradsupp` | 0.8291 | 0.7133 | 0.859 | 0.878 | 0.750 |
| `gelan_r4v2` | 0.8231 | 0.6723 | 0.802 | 0.869 | **0.799** |
| **`y26n_twoaxis2` @ep64** | **0.6961** | **0.6031** | 0.697 | 0.822 | 0.569 |

−13.3 mAP50. The obvious artifact was checked and ruled out: `map_eval.CLASS_NAMES` is `{0,1,2}` so the 4th
class is dropped rather than scored, but at conf 0.45 the model emits only **84 `UnknownGender` boxes of
7,357** (1.1 %). The deficit sits exactly where the split changed the supervision — Child −0.181,
Woman −0.162, Man only −0.056. Detection is not the cause: this model finds *more* people than the
baselines and labels them less accurately.

#### The mechanism was verified four ways before any of the above was believed

An early read of `train_batch0.jpg` suggested only one label per person. It is a rendering artifact — the
two rows share identical geometry, so `plot_images` paints the age row exactly over the gender row and only
classes 3/4/5 stay visible (`assets/exp17/train_batch0_age_row_on_top.jpg`). What settles it:

1. **Ground truth reaches the model with both axes.** The dataset's own class histogram
   (`assets/exp17/gt_class_histogram.jpg`): Woman 355,558 + Man 808,896 + GenderUnknown 222,793 =
   **1,387,247**; Adult 1,147,040 + Child 61,532 + AgeUnknown 178,671 = **1,387,243**. Both equal the
   emit's `kept` (1,387,261) to within the corrupt images ultralytics dropped.
2. **Training supervises both groups on the same anchor.** The selftest asserts exactly two hot channels
   per foreground anchor, one per group, and that cls gradient reaches both — and that *stock* ultralytics
   gives only one, which is why the patch exists at all.
3. **Inference produces both.** Mean argmax confidence 0.75 gender / 0.79 age on the same box.
4. **A cross-axis leak is structurally impossible.** `split_scores` takes an argmax *within* each channel
   group, so a gender reading can never come back "Adult".

What is *not* delivered as two predictions: the emitted YOLO label file carries one collapsed class per
person (per-axis detail lives only in the raw sidecar), and the `end2end` export cannot carry two axes at
all. Both are properties of the output path, not of the head.

## What we can decide from this

- **The two-axis head is not adopted.** Four must-not-regress failures, decision rule applied as written.
  `y26n_gradsupp` remains the recommendation in `docs/MODEL_COMPARISON.md`.
- **The core claim — that separating the axes buys crowd recall — is supported but unproven.** +10.8 crowd
  recall and +10.1 LAGENDA all-people recall are the largest gains measured in this project. They arrive
  together with 1.6–4.3× the false positives, which is what an effectively lower operating point looks
  like. One CPU-only `conf_sweep.py` run at matched FP would separate the two explanations. It was never run.
- **The abstention channel, as built, does not work.** 16.9 % recall. Phase 0 had already warned that
  `AgeUnknown` is 96.9 % labeler-abstention and only 3.1 % teen band, so the model was being asked to
  transfer abstention from unreadable/small/occluded people to visible teenagers. It did not.
- **One concrete suspect for why age lost:** the loss normalizes BCE across both channel groups with a
  single `cls_norm`, so the two axes share one gradient budget with no per-group balance. If gender is the
  easier axis it dominates — consistent with age being the weaker axis and its rarest channel collapsing
  first. Untested.
- **Keep per-axis reporting whichever head ships.** The collapsed 4-class view systematically misattributes
  age failures as gender confusion (113 against 44). That is a flaw in how we look at *every* model, not a
  property of this one.

## What this does NOT tell us

- **Nothing about the Gulf/traditional-dress bias.** LAGENDA is general photos and CrowdHuman is general crowds. This is the same gap every experiment since EXP-2026-01 has flagged, and it remains the single biggest untested question in the project. The two-axis head does not address it and must not be described as if it does.
- **The unknowns and the two-axis head move together.** The primary arm both separates the axes *and* restores the 196k people Spotlight dropped. If it wins crowd recall, this experiment alone **cannot say which change did it.** Isolating that needs a second arm (two-axis with unknown-gender people excluded), deliberately not run yet because it is only worth $11 if the primary arm wins.
- **The teen-band result will be a generalization test, not a direct one.** Only 5,526 of the training people are teen-band `AgeUnknown`; 173,146 are unreadable-age. Whatever the model does on visible teenagers is transfer from a different population, and the write-up must say so.
- **The teen-band bar rewards abstention, and abstention is cheap.** A model that answers `AgeUnknown` constantly would clear it. That is exactly why the abstention ceilings are pre-registered alongside it — read the two together or not at all.
- **In-training mAP is meaningless here** and is not evidence of anything. `DetectionValidator` scores the 6 channels as if they were mutually exclusive, and `best.pt` is selected on that fitness, so both `best.pt` and `last.pt` get evaluated through the external protocol.
- **This is an 82.5 %-trained model, and the comparisons are against fully-trained ones.** The run stopped
  at epoch 66 of 80, before `close_mosaic` (epoch 70) — normally where classification metrics move most.
  Every "fails the bar" verdict below should be read with that attached. The two FP gaps are 1.6× and 4.3×
  their bars, so finishing is unlikely to close them, but nobody measured that.
- **The operating point is NOT matched to the models it is compared against, and this may explain the whole
  result.** Box confidence is `max` over six channels and every person is trained to light up two of them,
  so conf 0.45 admits more boxes here than on a 3-class head — inflating recall *and* false positives
  together, which is exactly the pattern observed. **`conf_sweep.py` at matched FP is the test that would
  settle it and it was never run.** No adoption or rejection conclusion about the *design* should be drawn
  before it is; the rejection recorded above is of this checkpoint at this operating point.
- **Single run, no variance estimate.** Per the project convention, differences of ≤ 1 point are ties.
- **Nothing is deployed by this.** The browser cannot consume a 6-channel head without a JS change (see below), and no export or parity check has been done.

## What's next

No follow-up experiment is scheduled — the owner stopped this arm on 2026-08-17 to document it rather than
iterate. Recorded so the next person does not have to rediscover it:

**Free, CPU-only, and the single highest-value thing left** — `conf_sweep.py` on the epoch-66 sidecars at
matched FP (constrain `objects` %img-FP to 11.2 % and PASS to 0.27, then re-read crowd recall). It replays
the existing dumps with no GPU and no model re-run, and it is the only way to tell whether +10.8 crowd
recall is a real gain or a lower threshold in disguise. **Needs the dumps under
`/workspace/exp17/dump_epoch66/` and `/workspace/mapdump/y26n_twoaxis2_ep66/` on the network volume — they
survive the pod, but a new pod is required to reach them.**

**If the design is ever revisited**, in rough order of expected value:

1. **Per-group loss balance** — give each axis its own normalizer/weight instead of one shared `cls_norm`.
   The cheapest change that addresses the measured failure (age losing to gender, rarest channel collapsing).
2. **Rethink `AgeUnknown` before spending GPU.** It is 96.9 % labeler-abstention and 3.1 % teen band; asking
   one channel to carry both may be the design error. A dedicated teen-band label, or explicit abstention
   supervision, is a different experiment.
3. **Emit two rows per person** so the output format matches the label format. No retrain — it re-decodes
   existing checkpoints. Keep a collapsed file alongside so the legacy scorers keep working.
4. **The unknowns-ablation arm** is still the only way to attribute any crowd-recall gain between "axes
   separated" and "196k unknowns restored". Not worth $11 until the matched-FP question is answered.

**Unaffected by any of this:** the standing EXP-2026-14 recommendation — **grey-masked unknowns on
YOLO26n** — is still open and still untried, and the **Gulf-dress slice** remains the highest-value
unblocked work in the project, with the 30,567 Woman→Man Spotlight corrections as its ready-made sample.
The `Woman→Man` 2.7 : 1 asymmetry measured here is fresh evidence for doing it.

## What was measured, and what was not

| | status |
|---|---|
| LAGENDA (detection + classification), CrowdHuman, PASS, object set | ✅ epoch 66 |
| Per-axis accuracy + HTML predicted-vs-label report | ✅ epoch 64 (`assets/exp17/twoaxis_report_epoch64.html`) |
| Spotlight-val mAP | ✅ epoch 64 · ❌ epoch-66 re-run cut off when the pod was released |
| Matched-FP `conf_sweep.py` | ❌ never run — the open question |
| Flicker / temporal replay, ONNX latency, export parity | ❌ not attempted; nothing here is deployable |
| Gulf/traditional-dress slice | ❌ still the project-wide gap |

## Deployment note (not part of this experiment)

The extension hardcodes the raw `[1, 4+C, N]` layout with class indices 0/1/2 and runs NMS in JS (`docs/DEPLOYMENT_LEARNING_CONTEXT.md:32-33`). Two axes ship on that same raw path — the tensor becomes `(1, 10, 8400)` and the JS does two argmaxes instead of one, applying the collapse rule above. The `end2end=True` `(1,300,6)` export **structurally cannot** carry two axes: its row has a single `cls` scalar. So export with `end2end=False`, which is what `y26nraw1` already ships. `decode_two_axis` in `run_ultralytics_labels.py` is the reference implementation for that JS.

---

## Appendix — the details

### Commands

```bash
# Phase 0 — the gate, before anything else (minutes, free)
python3 exp17_phase0_counts.py --verdicts /workspace/spotlight/run/oiv7_train/verdicts_batch.jsonl

# 1. transform the labels IN-POD, onto local NVMe, as part of staging.
#    Reads the volume, writes local: the two-axis labels are a deterministic
#    $0 function of verdicts_batch.jsonl, so a second copy on the network
#    volume would only spend quota that `df` cannot see.
python3 parallel_emit.py --two-axis \
    --raw-labels /workspace/spotlight/raw/oiv7_train \
    --run        /workspace/spotlight/run/oiv7_train \
    --out        $TREE/labels/train --workers 24

# 2. train
python3 train_twolabel.py --data /workspace/exp17/twoaxis.yaml \
    --weights yolo26n.pt --epochs 30 --imgsz 640 --batch -1 \
    --project /workspace/exp17/train --name y26n_twoaxis

# 3. dump the four datasets (both best.pt and last.pt)
python3 run_ultralytics_labels.py --model <ckpt> --map two-axis \
    --images <dataset> --out /workspace/exp17/<arm>/<dataset> \
    --conf 0.45 --floor 0.05 --iou 0.7 --imgsz 640

# 4. score — existing scorers, unchanged, with the collapsed class names
--classes Woman,Man,Child,UnknownGender      # conf_sweep.py
--class-names Woman,Man,Child,UnknownGender  # run_autolabel_on_manifest.py, eval_negatives_crowd.py
```

Full phased commands: `EXP-2026-17-POD-RUNBOOK.md`.

### Plain-word definitions

- **Coverage / abstention** — how often the model answers `GenderUnknown` or `AgeUnknown` instead of committing. Reported separately from accuracy, because a model that never commits has a great accuracy and is useless.
- **Escape** — an adult read as a Child, so the blur does not apply and the viewer sees them. The consequential direction.
- **False person** — a box drawn on something that is not a person. Dolls, statues and mannequins are not people; a photo or poster of a real person **is** (the pre-registered EXP-2026-03 ruling).

### Appendix — "verify it yourself"

Written before running. Every snippet below is quoted from the source as it stands, with `file:line`.

**1. The taxonomy and the age-unknown rule** — `vlm-cluster/two_axis.py:56-74`. Note the ordering: the band is checked *before* the child call, so a "child" verdict with an estimated age of 15 abstains rather than being trusted.

```python
    if v is None or v.get("verdict") not in ("real_person", "depiction"):
        return None
    gid = GENDER_ID.get(v.get("gender"), GENDER_UNKNOWN)

    age, grp = v.get("estimated_age"), v.get("age_group")
    if lowconf_age_unknown and v.get("confidence") == "low":
        aid = AGE_UNKNOWN
    elif isinstance(age, (int, float)) and band[0] <= age <= band[1]:
        aid = AGE_UNKNOWN
    elif grp not in ("child", "adult"):
        aid = AGE_UNKNOWN
    elif grp == "child" and (age is None or age <= CHILD_AGE_MAX):
        aid = CHILD
    else:
        aid = ADULT
    return gid, aid
```

**2. The collapse back to the scored label space** — `vlm-cluster/two_axis.py:88-94`. This is the blur policy; nothing else in the pipeline re-implements it.

```python
    if aid == CHILD:
        return C_CHILD
    if gid == WOMAN:
        return C_WOMAN
    if gid == MAN:
        return C_MAN
    return C_UNKNOWN_GENDER if keep_unknown_gender else None
```

**3. Pairing the two rows back into one person** — `vlm-cluster/train_twolabel.py:135-141`. Pairing is by **exact (image, box) equality**; augmentation applies the same transform to both rows of a pair, so they stay bit-identical. A broken pair raises rather than training a half-supervised person.

```python
        key = torch.cat([bi[:, None], box], 1)
        uniq, inv = torch.unique(key, dim=0, return_inverse=True)
        multihot = torch.zeros(uniq.shape[0], self.nc, device=self.device, dtype=box.dtype)
        multihot[inv, cls] = 1.0
```

**4. Class-agnostic assignment and the multi-hot target** — `vlm-cluster/train_twolabel.py:191-211`. Every GT label is set to `nc`, which indexes an appended class-agnostic channel, so an anchor competes on "how person-ish is this" and never on one axis's score. The alignment weight that comes back is written into *both* of the person's hot channels.

```python
        pd_sig = pred_scores.detach().sigmoid()
        pd_aug = torch.cat([pd_sig, pd_sig.max(-1, keepdim=True).values], dim=-1)
        ...
        align_w = target_scores[..., self.nc]  # (b, h*w)
        mh_sel = gt_multihot.gather(1, target_gt_idx.unsqueeze(-1).expand(-1, -1, self.nc))
        cls_target = align_w.unsqueeze(-1) * mh_sel * fg_mask.unsqueeze(-1)
```

Box/DFL are untouched and keep weighting by that same alignment weight, so a person whose gender *and* age are both unreadable still supervises localization — asserted in the selftest.

**5. The decode** — `vlm-cluster/run_ultralytics_labels.py:213-224`. Confidence is the max over all six channels and NMS is class-agnostic, deliberately identical to stock single-label inference so it can be parity-checked.

```python
    scores = p[:, 4:]                            # already sigmoid
    conf = scores.amax(1)
    keep = conf >= floor
    ...
    idx = torchvision.ops.nms(xyxy, conf, iou)[:max_det]
    xyxy = ops.scale_boxes(in_shape, xyxy[idx].clone(), orig_shape)
```

**6. What the self-tests actually prove** (all run, all passing, CPU-only, no downloads, no data):

| Check | Command | Result |
|---|---|---|
| Stock ultralytics really does drop an axis | `train_twolabel.py --selftest` | stock assigner: **1 hot channel per anchor**; ours: **2** |
| Both axes receive gradient on the same anchor | `train_twolabel.py --selftest` | passes on the plain *and* end-to-end paths |
| Doubly-unknown person still supervises the box | `train_twolabel.py --selftest` | box loss 2.296 (yolo11n) / 1.748 (yolo26n) |
| Broken pair raises instead of half-training | `train_twolabel.py --selftest` | passes |
| Checkpoint pickles and reloads | `train_twolabel.py --selftest` | passes; `--export-plain` verified to load with the module absent from `sys.path` |
| Two-axis emit preserves geometry byte-for-byte | `parallel_emit.py --selftest` | passes |
| **Collapsing two-axis labels reproduces the existing 3-class emit exactly** | `parallel_emit.py --selftest` | passes **with the band on and off** |
| Decode matches ultralytics' own NMS | `run_ultralytics_labels.py --selftest` | 300/300 boxes, **max box delta 0.00, max conf delta 0.00** |
| Sidecar keeps `cls`/`conf` meaning, adds axes | `run_ultralytics_labels.py --selftest` | passes; label lines stay 5-field |
| The pre-existing trainer is unaffected | `train_gradsuppress.py --selftest` | passes, reproducing its recorded numbers (10 anchors / 2.181, 1 / 1.757) |

The collapse-equivalence check is the load-bearing one: it means the two-axis labels are a strict **refinement** of the existing labels, not a different labeling. People in the teen band were already adults under `merge()`, and `AgeUnknown` collapses back to Adult, so the 3-class projection is identical either way. The band changes what the model is *taught*, not what the old label space says.

**7. Environment.** `ultralytics==8.4.115` (the pin the loss copies by line number — an unpinned install would silently diverge from the copied method bodies). The self-tests above were run against that exact version.

**8. Said honestly — what was NOT done.**

- **Nothing has been trained.** Every number in this appendix is a self-test result about mechanism, not about accuracy. There is no model yet.
- **The 13–17 band is a judgement call**, not a calibrated threshold. It is taken from EXP-2026-10's measured gradient, but no sweep was run to choose it, and no one has checked how many training people it moves — that is Phase 0's second count, and it should be filled in here before training.
- **Pairing assumes augmentation keeps duplicate boxes bit-identical.** This is argued from the transforms applying identically to identical inputs, and the trainer raises loudly if it is ever false, but it has **not** been observed across a real augmented epoch. The 1-epoch smoke in the runbook is what would catch it.
- **The decode parity check uses a synthetic head output, not a real forward pass.** That is deliberate — a randomly initialized network emits a near-constant score field, so a real-model comparison would test NMS tie-breaking rather than arithmetic — but it means the parity evidence covers the decode maths, while the model→sidecar path is covered only by a shape-and-plumbing smoke.
- **No export, no browser parity check, no latency measurement.** None of the deployment claims have been tested.
