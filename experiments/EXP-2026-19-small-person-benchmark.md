# EXP-2026-19 — A benchmark for small people

**Status:** dataset built 2026-08-20; models scored the same day. Bars below were written
before any model result was read.

## The question

Production QA keeps hitting the same thing: the model misses small/distant people. We have
never been able to say *how* bad it is, because no dataset we own isolates that regime — the
four standard benchmarks all mix a prominent foreground subject with everyone else, and one
pooled recall number hides the small-person failure inside it.

So: **given a person who is small in the frame, how often does each candidate model find them,
and does it still get their gender right?**

## Why a new set, and why two arms

`conf_sweep.py` already reports `det_recall_small|med|large` bands, and EXP-2026-18's
`scale_robustness.py` already shrinks images. Neither gives us a *dataset* — something staged
on the volume that any tool, any model, and any teammate can be pointed at, and that QA can
look through. That is what this builds.

Two arms, because neither alone is trustworthy:

**Arm A — `crowd_small` (real).** Real images that genuinely contain small people, from
CrowdHuman val, whose boxes are **human and exhaustive**. This is the only sound source we have
for small-person *recall*: a machine-labelled corpus cannot measure it, because the labeler's
own misses become "no label", and absence of a label is not evidence of absence — the bug class
this project has hit four times. CrowdHuman has no gender/age labels, so Arm A measures
**Component 1 only**.

**Arm B — `synth_shrunk` (controlled).** People who were photographed LARGE and labelled at a
size where the label is trustworthy, then shrunk-and-padded onto a canvas of the original size
so they land on a target pixel height, labels carried along. This is the only way to measure
**gender and age on small people** without new human labelling, and it isolates apparent size
as the single changing variable. Its limitation is stated up front: LANCZOS-downscaled people
are cleaner than real distant people — no motion blur, no haze, no focus loss — so Arm B is an
**upper bound** on real-world small-person performance, never a substitute for Arm A.

## How "small" is defined

A person's height is recorded two ways, and they are not interchangeable:

- **native px** — pixels in the source image. What a QA person sees at 100% zoom, and what the
  request asked for ("people spanning 96 px or less").
- **model-input px** — `native_h × 640 / max(W, H)`, the height after aspect-preserving
  letterbox. What actually decides detectability. A 96 px person in a 4000 px-wide photo is
  **15 px** to the network; the same 96 px person in a 640 px photo is 96 px.

Selection for these arms is on **native height ≤ 96 px** (the product question), with the
input-space height recorded per person in the manifest. The 96 px edge is also COCO's
medium/large boundary, so the numbers stay readable next to `mAP_medium`.

**People larger than the band are not deleted — they become ignore regions.** A model is
therefore neither credited for the big person in the foreground nor penalised for finding them.
That makes the headline a pure small-person number.

## Pre-registered bars

No model is adopted or rejected on this set — it is a measuring instrument, and this run is the
first reading. The bars are therefore about whether the instrument works, plus one product
threshold stated before looking.

**Set validity (must all hold, or the set is not fit for purpose):**

- **V1 — it isolates the hard regime.** Each model's small-person recall on `crowd_small` must
  be materially below its own all-sizes CrowdHuman recall from the standard protocol (e.g.
  `y26n_gradsupp` 36.5%). If the set scores the same as the pooled benchmark, it isolates
  nothing.
- **V2 — it discriminates.** Spread between best and worst model ≥ 3 pts. A set on which every
  model scores identically cannot inform a choice.
- **V3 — the ignore machinery works.** Zero scored GT boxes above 96 px, and every out-of-band
  person present as an ignore region (verified by re-parsing the emitted files, not asserted).

**Product reading (stated before the numbers):**

- Small-person recall **< 25%** ⇒ the QA complaint is real and quantified: roughly three in
  four small people are invisible to the blur.
- **25–50%** ⇒ a real, large gap worth a dedicated fix.
- **> 50%** ⇒ small people are mostly handled and QA's complaint is about something narrower.

**What this run does NOT settle** (write it down now, not after):

- CrowdHuman carries no gender or age, so Arm A says nothing about whether a *woman* who is
  small gets blurred — only whether a *person* who is small gets found. Arm B is what speaks to
  that, under its upscaling caveat.
- Precision on this set is a lower bound, not a true precision: a detection that lands on a
  real person CrowdHuman failed to annotate counts against the model. Same caveat as
  EXP-2026-03 §2.7.
- Nothing here is temporal. EXP-2026-11 found 90–99% of production dropouts are threshold gaps
  under hysteresis, so equal per-frame recall can still look much worse to someone watching
  video.

## What the census found (before any model ran)

Reading the person-height distribution of everything we own was supposed to be a sizing
exercise. It turned out to be the finding.

| source | people | median height | people ≤ 96 px |
|---|---:|---:|---:|
| CrowdHuman val (human, exhaustive) | 99,481 | 162 px | **30,209** (by full-body box) |
| `haramblur_holdout` (machine labels) | 16,139 | 280 px | 2,289 (14%) |
| LAGENDA, human-labelled 3-class | 7,098 | 507 px | **0** |

**LAGENDA's human-labelled set contains no small people at all** — 5th-percentile height
198 px. That is the answer to why "how good is the model on small women?" has never had a
number: the only human gender/age benchmark this project owns has never contained a small
person. It is also why Arm B has to be synthetic, and why LAGENDA is the right *source* for
it (every person large and human-labelled).

The holdout is nearly as lopsided (median 280 px), so it is a poor small-person source too.
CrowdHuman is the only real arm available, and it is a good one.

One number worth keeping: CrowdHuman's median person is 162 px native but **59.5 px at
model input**. Half of that benchmark is already "small" in the only space that matters,
which is invisible when heights are quoted natively.

## What was built

| arm | contents | state |
|---|---|---|
| `crowd_small` | 800 images · 16,326 people ≤96 px full-body · 23,072 ignore regions · occlusion light 6,121 / partial 6,762 / heavy 3,443 | built, 11/12 checks pass |
| `synth_shrunk` | 1,500 images = 300 LAGENDA × {96,64,48,32,24} px, human gender/age carried through | built, not yet verified |

`crowd_small` height bands (native, visible box): tiny `<16` 485 · verysmall `16–32` 3,706 ·
small `32–64` 7,979 · medium `64–96` 4,115.

### The one check that failed, and how it was closed

27 of 16,326 scored people have a *visible* box taller than 96 px. 15 are float round-trip
noise (≤0.01 px). The other 8 are genuine, and the cause is CrowdHuman, not us: **548 people
in this set carry a `vbox` taller than their `fbox`** — the largest is a 622 px visible box
on a person whose full-body box is ≤96 px. Sizing by `max(fbox_h, vbox_h)` removes it. Until
that re-mine happens, 0.05% of the set is people who are not small. Recorded rather than
silently tolerated by loosening the check.

The tolerance is now `BAND_EPS_PX = 0.01`, guarded by selftests proving a box at
exactly 96.0 px passes and one at 96.5 px still fails — so the epsilon cannot hide a
real out-of-band person. **The crowd arm now passes 12/12.**

### V3 was earned, not assumed

The first build of this set reported **every person as unoccluded**, because the emit copied
`vbox` into `fbox` and `occ_ratio = area(vbox)/area(fbox)` is then 1.0 by construction — the
light/partial/heavy breakdown was silently dead. That is also why size is now measured on the
full-body box: selecting on visible extent had been quietly mixing "small person" with "large
person mostly hidden". Both were found by reading the emitted files back rather than trusting
the manifest, and both are now negative controls in `verify_small_person_set.py --selftest`.

## Model results

Two models, per the owner's narrowing: shipped production `yolo11N-640` vs the fine-tuned
YOLO26n (`y26n_gradsupp`, one-to-many head — MODEL_COMPARISON's recommended deployable).
`y26n_noe2e_warm50-2` was run alongside at negligible cost. All at conf 0.45, IoU 0.5
greedy matching, the standard protocol, nothing tuned for this set.

### Arm A — real small people (800 images, 16,314 people ≤96 px full-body)

| model | recall | found | precision | light occl. | partial | heavy | clear FP/100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `gelannfav14r4fw_gemlb_v2` | **7.58%** | **1,236** | 93.2% | **10.4%** | **3.7%** | **0.6%** | **2.75** |
| `y26n_gradsupp` | 7.38% | 1,204 | 93.4% | 10.2% | 3.5% | 0.4% | 3.88 |
| `y26n_noe2e_warm50-2` | 5.97% | 974 | 92.3% | 8.0% | 3.4% | 0.4% | 3.75 |
| `y26n_sop50` | 5.74% | 936 | 92.1% | 7.7% | 3.2% | 0.4% | 3.62 |
| `yolo11N-640` (production) | 5.11% | 834 | **95.2%** | 7.5% | 1.7% | 0.1% | 3.00 |

**Production is last on the most realistic arm.** `gelannfav14r4fw_gemlb_v2` finds 48% more
small people than production *and* carries the cleanest false-positive rate of the five.

**Roughly 19 in 20 small people are invisible to production, and 13 in 14 to the fine-tuned
YOLO26n.** The QA complaint is not a perception — it is the largest measured failure in this
project. For context, `y26n_gradsupp` scores 36.5% on the all-sizes CrowdHuman benchmark;
restricted to people ≤96 px it scores 7.4%.

The fine-tuned YOLO26n finds **44% more small people** than production (1,204 vs 834) for
1.8 points of precision and ~0.9 extra false persons per 100 images.

### Arm B — the same 480 people at five apparent sizes (AP50)

| model | h96 | h64 | h48 | h32 | h24 |
|---|---:|---:|---:|---:|---:|
| `yolo11N-640` mAP50 | 0.395 | 0.228 | 0.125 | 0.036 | 0.013 |
| `y26n_gradsupp` mAP50 | 0.378 | 0.222 | 0.134 | 0.043 | 0.024 |
| — | | | | | |
| `yolo11N-640` **Woman** | **0.465** | **0.257** | **0.122** | **0.025** | 0.007 |
| `y26n_gradsupp` **Woman** | 0.389 | 0.194 | 0.076 | 0.019 | 0.009 |
| `yolo11N-640` Man | 0.607 | 0.393 | 0.244 | 0.082 | 0.034 |
| `y26n_gradsupp` Man | **0.627** | **0.435** | **0.316** | **0.110** | **0.063** |
| `yolo11N-640` Child | 0.114 | 0.035 | 0.010 | 0.000 | 0.000 |
| `y26n_gradsupp` Child | 0.119 | 0.035 | 0.010 | 0.000 | 0.000 |

Performance roughly halves for every halving of apparent size, on both models, on every
class. Children become undetectable below ~32 px (AP50 = 0.000) — the harmless direction
for this product, since an unfound child is left unblurred, which is the correct outcome.

### Arm C — `paste_grey`: the same people, cut out and isolated

250 images / 1,500 people. SAM polygons cut each person out and paste six of them onto a
640 px grey canvas at each target height. No scene context, no occlusion, no crowding — the
cleanest isolation of apparent size, and the least realistic of the three arms. It is also
the only arm that reports the real product metric end to end: **a woman both FOUND and
CALLED Woman**.

| metric | model | h96 | h64 | h48 | h32 | h24 |
|---|---|---:|---:|---:|---:|---:|
| detection recall | `yolo11N-640` | **85.0** | **64.7** | 35.0 | 3.3 | 0.3 |
| | `y26n_gradsupp` | 70.7 | 49.3 | **44.0** | **16.3** | **4.7** |
| | `y26n_warm50-2` | 70.0 | 62.0 | **45.0** | **16.3** | 4.0 |
| **Woman end-to-end** | `yolo11N-640` | **72.2** | 42.6 | 19.4 | 1.9 | 0.0 |
| | `y26n_gradsupp` | 60.2 | 38.9 | **31.5** | **8.3** | **1.9** |
| | `y26n_warm50-2` | 57.4 | 49.1 | 23.1 | 7.4 | 1.9 |
| | `y26n_sop50` | 76.9 | 61.1 | 38.9 | 10.2 | 1.9 |
| | `gelannfav14r4fw_gemlb_v2` | **84.3** | **77.8** | **60.2** | **15.7** | 1.9 |
| 3-class acc (on found) | `yolo11N-640` | 70.6 | 63.4 | 64.8 | 90.0 | 100.0 |
| | `y26n_gradsupp` | 70.8 | 60.8 | 63.6 | 61.2 | 57.1 |
| | `y26n_warm50-2` | **76.2** | **69.4** | **68.1** | 59.2 | 66.7 |

**With five models the crossover disappears: `gelannfav14r4fw_gemlb_v2` and `y26n_sop50`
beat production at every size** (84.3 / 77.8 / 60.2 / 15.7 and 76.9 / 61.1 / 38.9 / 10.2 vs
72.2 / 42.6 / 19.4 / 1.9). The "production wins above 64 px" reading was an artifact of
comparing only three checkpoints and is retired.

Among the original three, production did lead above ~64 px and lose below ~48 px. At 32 px it finds 3.3% of people against `y26n_gradsupp`'s 16.3% — five
times more — and its Woman end-to-end rate is 1.9% against 8.3%. At 24 px production reaches
**0.0%** Woman end-to-end while both YOLO26n models still return 1.9%.

Production's 3-class accuracy at h32/h24 (90%, 100%) is an artifact of a tiny denominator —
it only found 3.3% and 0.3% of people. Accuracy on found people is not a product metric when
almost nothing is found.

The child→adult leak runs 40–55% at h96 on every model. That is the over-blur direction and
is acceptable for this product.

### Arm D — `avatars`: simulated profile pictures

240 images, 1,336 head-and-shoulders portraits laid out on a 640 canvas at 160/112/80/56 px,
masked to circles and rounded squares the way Facebook and LinkedIn render them. 334 people
at every size with an identical class mix (126 Woman / 128 Man / 80 Child), so size is again
the only moving variable.

**Why it was worth building.** A social feed is mostly portrait crops, which is a different
distribution from the full-body people these detectors were trained on. And because the crops
are taken from the original photograph and **never upscaled**, the people stay sharp at every
size — this is the only arm that separates *small* from *low quality*. Every other arm
confounds the two.

| metric | model | 160 px | 112 px | 80 px | 56 px |
|---|---|---:|---:|---:|---:|
| detection recall | `yolo11N-640` | 74.9 | 76.3 | 67.7 | 47.6 |
| | `y26n_noe2e_warm50-2` | 81.1 | 79.6 | 76.3 | **67.4** |
| | `y26n_sop50` | 80.5 | **80.5** | **78.1** | 65.9 |
| | `gelannfav14r4fw_gemlb_v2` | **82.0** | 76.0 | 65.6 | 47.6 |
| **Woman end-to-end** | `yolo11N-640` | 72.2 | 71.4 | 65.1 | 41.3 |
| | `y26n_noe2e_warm50-2` | 78.6 | 72.2 | 69.0 | 57.9 |
| | `y26n_sop50` | **79.4** | **76.2** | **72.2** | **58.7** |
| | `gelannfav14r4fw_gemlb_v2` | 69.8 | 67.5 | 61.1 | 38.9 |
| class acc (on found) | `gelannfav14r4fw_gemlb_v2` | 81.0 | **81.5** | **81.7** | **80.5** |
| | others | 81.6 / 83.0 / 82.9 | 75.3 / 77.4 / 77.7 | 75.2 / 74.5 / 74.7 | 71.1 / 73.3 / 74.5 |
| Child end-to-end | `gelannfav14r4fw_gemlb_v2` | **73.8** | **60.0** | **50.0** | **23.8** |
| | `yolo11N-640` | 40.0 | 25.0 | 13.8 | 1.2 |

**Three findings.**

1. **Even a large, perfectly sharp avatar is not reliably handled.** At 160 px — a big,
   clear portrait — only 72–79% of women are both found *and* called Woman. Roughly one in
   four profile-picture women goes unblurred, and that is with image quality at its best.
   Sharpness is not the binding constraint; framing is.
2. **The crowd-photo ranking does not transfer.** `gelannfav14r4fw_gemlb_v2` wins Arm A
   outright and is **last** here on Woman end-to-end at every size, despite carrying the
   best class accuracy (81–82%). It classifies portraits well and finds fewer of them.
   Any "best model" claim has to name the arm.
3. **Production collapses at feed size.** At 56 px — an ordinary comment-thread avatar —
   `yolo11N-640` reaches 41.3% Woman end-to-end against ~58% for both YOLO26n arms, a
   17-point gap on the metric that decides whether a blur happens.

Child end-to-end is the one place gelan dominates (73.8% vs 40.0% at 160 px). Children are
correctly left unblurred, so that is a win in the harmless direction.

**The mAP block tells a different story than conf 0.45, and the difference is
calibration.** `gelannfav14r4fw_gemlb_v2` posts the best MAP50 at every avatar size
(0.757 / 0.739 / 0.691 / 0.629 vs production's 0.613 / 0.605 / 0.557 / 0.456) while
finishing last on conf-0.45 Woman end-to-end. Its median matched-detection confidence is
**0.420 — under the shipped 0.45 threshold** — against 0.574 for `y26n_noe2e_warm50-2`.
Move the cut to 0.25 and its 56 px recall goes 47.6% → 74.6%, level with the y26n arms.

So the ranking-quality of gelan's boxes is the best measured here; what it lacks is
confidence calibrated to where we cut. A fixed 0.45 comparison scores calibration as if it
were capability, and `conf_sweep.py` already exists to pick the cut per model and per
class. That should happen before the next candidate comparison.

**Caveats, stated plainly.** The crop is a geometric approximation from the person's box —
the top 34% of its height, centred — not a face detection, so framing is approximate. The
source is LAGENDA general photography, not real scraped profile pictures, so lighting and
pose are photographic rather than selfie-typical. And avatars sit on a flat light canvas,
which is closer to a real feed than `paste_grey`'s grey but still not a rendered UI.

### The finding that matters

**The fine-tuned YOLO26n is better at finding small people and worse at calling them women.**
It leads `Man` AP50 at every single size (0.627/0.435/0.316/0.110/0.063 vs
0.607/0.393/0.244/0.082/0.034) and trails `Woman` AP50 at four of five
(0.389/0.194/0.076/0.019 vs 0.465/0.257/0.122/0.025). At h48 it has **30% more Man AP and
38% less Woman AP** than production.

For a blur product serving a male viewer, `Woman` is the product metric — a woman who is
detected but labelled "Man" is a woman who is not blurred.

**But Arm C shows this holds only down to a point.** The shipped model is the better product
on *moderately* small people (64–96 px) and the *worse* one below ~48 px, where it collapses
to near-zero detection while the fine-tuned models keep finding several times more. The
honest statement is a crossover, not "production is better on small people".

This is the first controlled reproduction of the owner's production QA observation recorded
in CLAUDE.md §4.9995 ("the shipped model beats yolo26n on women recall and loses on FP"),
and it discriminates between the ranked hypotheses listed there. It is the signature of
**hypothesis (c)** — Spotlight's UNATTRIBUTED 2.5× Woman→Man labelling asymmetry (30,567 vs
12,102) taught the model to call women "Man". A pure operating-point difference (hypothesis
a) would move both classes together; it does not. Detection loss from the dropped ~58 px
unknowns (hypothesis b) would cost recall without a class-specific swing; Arm A shows
y26n's detection is *better*, so that cannot be the whole story either.

**Before the next training run, this points at the labels, not the architecture.** The
30,567 Woman→Man Spotlight corrections are already named across CLAUDE.md as the ready-made
probe sample.

### Scorecard against the pre-registered bars

| bar | verdict |
|---|---|
| **V1** — isolates the hard regime | **PASS**, decisively: 7.4% here vs 36.5% all-sizes for the same model |
| **V2** — discriminates (≥3 pt spread) | **FAIL on Arm A** — best-to-worst spread is 2.27 pts (7.38 vs 5.11), under the bar I registered. **PASS on Arm B**, where the Woman-AP50 gap reaches 7.6 pts at h96 and 4.6 at h48 |
| **V3** — ignore machinery works | **PASS** — verified by re-parsing the emitted files; 6,462 detections fell on ignore regions and were correctly excluded. The crowd arm now passes **12/12** and each synth/paste arm **7/7** |
| product reading | **< 25% ⇒ "the complaint is real and quantified"** — both models are at 5–7%, far below the lowest band I pre-registered |

V2's failure on Arm A is recorded as failed, not reinterpreted. The honest reading is that
the crowd arm compresses model differences because *everything* fails there; Arm B, where
the same people are scored at five controlled sizes, is where the models separate — and it
separates them in the direction that matters to the product.

