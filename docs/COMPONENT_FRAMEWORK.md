# The three-component evaluation framework

**In one line:** every labeling system we test — SAM3, Qwen2.5-VL-7B, MiVOLO V2, the replacement
pipeline, anything future — gets measured against one or more of **three named components**, each
with its own datasets, metrics, and bars, and the results are never pooled across components.

Adopted 2026-07-14. This replaces the looser "component-selection framing" from EXP-2026-02 §2.6
(which EXP-2026-03 already corrected once). Experiments reference this doc instead of re-deriving
the split.

---

## Why decompose at all

The labeling pipeline does three different jobs, and our first three experiments showed they fail
independently:

- SAM3's *detection* looked solved on LAGENDA (98.7% recall) and then broke on crowds (77%) and
  human-shaped objects (62% false-person rate) — a detection problem invisible to classification
  metrics.
- SAM3 and Qwen2.5-VL-7B — completely different architectures — share the same *age* failure band
  (teens), while both are near-perfect on *gender*. A single pooled "accuracy" number would have
  hidden that the age axis is the unsolved one.
- The Gulf/traditional-dress bias is purely a *gender* question on adults; it has nothing to do
  with detection or age.

A monolithic score can't tell you which component to replace. Component scores can.

## The three components

The components mirror the pipeline itself: find the person, decide adult vs child, and (adults
only) decide gender. Each component's question assumes the previous one was answered correctly.

### Component 1 — Person Detection

**Job:** image → a box for every person, and no boxes where there is no person.

**What counts as a person** (pre-registered in EXP-2026-03 §2.2, owner's ruling): anything a
viewer is meant to lower their gaze from in the Islamic sense — a real human **and any printed or
photographic depiction of one** (posters, billboards, magazine covers). A doll, mannequin, statue,
sculpture, or toy is **not** a person; a detection there is a false positive.

**Metrics:** detection recall (stratified by occlusion where GT supports it), detection precision,
duplicates per matched person (fragmentation), false positives per 100 person-free images, % of
human-shaped-object images gaining a false person label.

**Datasets** (staged under `/workspace/datasets/`, reusable):
| Dataset | What it tests | Structural limit |
|---|---|---|
| LAGENDA (`/workspace/lagenda_eval/`) | recall on prominent subjects | ~1 labeled person/image → precision unmeasurable |
| CrowdHuman (`crowdhuman/`, 500-img seed-51 sample) | recall + precision + duplicates in exhaustively-labeled crowds | no age/gender labels |
| PASS (`pass_3k/`, 3,000 verified-empty scenes) | baseline hallucination rate | negatives only |
| Object set (`object_set/`, 259 hand-verified doll/statue/toy images) | false-person rate on human-shaped objects | negatives only; expensive to rebuild — reuse, don't re-verify |

**Tooling:** `vlm-cluster/eval_negatives_crowd.py` (both modes) + the detection-recall side of
`run_autolabel_on_manifest.py`. Matching is class-agnostic (`match_boxes`, greedy mutual-exclusive,
IoU ≥ 0.5 default).

### Component 2 — Age group classification (Adult vs Child)

**Job:** given a correctly-detected person, decide **Adult vs Child**. Production cutoff is ≤12 =
Child, but the finding from both completed classification experiments is that models have their own
implicit boundary (~18ish with a teen fade) regardless of where we draw ours.

**Error direction weighting (from the product goal):** the consequential error is an **adult read
as Child — that adult escapes the blur**. A ~12-year-old over-blurred as adult is acceptable. So
the headline metric is the adult→Child leak rate, not child recall. Do not call child recall a
"safety floor."

**Metrics:** adult→Child rate (the one that matters), child recall, accuracy in the teen band
(13–17), the implicit-boundary curve (child-rate by actual age band), coverage/abstain rate.

**Datasets:** LAGENDA (5,000-person human-checked sample — the standard bench so every system's
teen curve is directly comparable); MSP60K / WIDER as follow-on options. Clean cutoffs only exist
at band edges 9/10, 12/13, 17/18 (VLM band-granularity limit).

**Tooling:** `translation.py` schemes `child_vs_adult` / `child_vs_adult_9` / `child_vs_adult_18`
+ `eval_taxonomy.py`; for detector-style systems, `run_autolabel_on_manifest.py`'s
`child_age_diagnostics` (implicit boundary + cutoff sweep).

### Component 3 — Adult gender classification (Man vs Woman)

**Job:** given a correctly-detected **adult**, decide Man vs Woman. This is the component the
product actually blurs on, and the one carrying the known domain risk: **men in Gulf/traditional
dress (thobe, ghutra) misread as women**.

**Error direction weighting:** a wrong gender on an adult means the wrong people get blurred AND
the right ones don't — both directions defeat the purpose. Both matter.

**Metrics:** gender accuracy on adults (with coverage/abstain reported separately), and —
critically — the same metric **on domain slices**: general photos vs the Gulf/traditional-dress
slice. A general-domain number does NOT stand in for the domain slice.

**Datasets:** LAGENDA (general-domain); a hand-checked **Gulf/traditional-dress slice — still
does not exist, and is the single biggest open gap across all experiments** (gender on adults in
known attire is a reliable human label even where age isn't).

**Tooling:** `translation.py` scheme `gender_only` + `eval_taxonomy.py`; for detector-style
systems, the gender-only-error split in `run_autolabel_on_manifest.py`.

## Where mAP fits (and why it is not on this scoreboard)

mAP is deliberately absent here: it pools localization + classification + confidence-ranking
into one number (the monolithic score this framework decomposes), it integrates over all
confidence thresholds while the product runs at a fixed conf 0.45, and three of the four
benchmark datasets can't support it structurally (LAGENDA labels ~1 person/image → unlabeled
real people would poison precision; PASS and the object set have zero positives → mAP
undefined). mAP IS computed where it's well-defined and useful — against the exhaustively
labeled **Spotlight val split** as a *label-alignment* metric (one shared evaluator,
`vlm-cluster/map_eval.py`; see EXP-2026-12) — and lives in the experiment docs, not this
scoreboard. Component metrics at the production threshold remain the decision-grade numbers.

## Rules of the framework

1. **Every evaluation names its component(s) up front.** "We measured system X on Component 1
   using CrowdHuman + PASS" — not "we measured accuracy."
2. **Never pool across components or across datasets.** Each dataset answers one question for one
   component. (EXP-2026-03 house rule, now global.)
3. **Condition on the upstream component.** Component 2 is scored only on correctly-detected
   people; Component 3 only on correctly-detected adults. A monolithic system (SAM3's 3-prompt
   mode does all three jobs in one pass) gets decomposed scores by conditioning — exactly what
   EXP-2026-02's gender-only-error and child diagnostics already did.
4. **A system may be measured on any subset of components.** A pure detector is Component 1 only;
   MiVOLO V2 is Components 2+3 only; that's fine and expected.
5. **Pre-register bars per component per dataset** before running (experiment template rule), and
   weight errors by the product's error direction (adult escaping blur is the bad one).
6. **End-to-end pipeline accuracy is a separate measurement, not a substitute.** Component scores
   tell you what to fix; only a full-pipeline run tells you the shipped label quality.

## Scoreboard — every measured number, by component

Status as of 2026-08-05. ✅/❌/⚠️ are against each experiment's pre-registered bars.

### Component 1 — Person Detection

| System | Dataset | Headline result | Source |
|---|---|---|---|
| SAM3, production 3-prompt (woman/man/child) | LAGENDA | recall **98.7%** ✅ (prominent subjects only) | EXP-2026-02 |
| SAM3, production 3-prompt | CrowdHuman | recall **77.1%** ❌ (light occl. 82.7% vs 90% bar; heavy 61%) · precision **86.6%** ✅ lower bound · duplicates **1.1%** ✅ | EXP-2026-03 |
| SAM3, production 3-prompt | PASS | **8.4 FPs/100** empty images ⚠️ (5.4% of images; mostly "Man") | EXP-2026-03 |
| SAM3, production 3-prompt | Object set | **62.2%** of images gain a false person ❌ (dolls 88.8%, 2.3 FPs/img, mostly "Woman") | EXP-2026-03 |
| SAM3, single "person" prompt | CrowdHuman | recall **78.5%*** (light 84.7%, +2.1 vs 3-prompt — under the 5-pt adoption rule) · precision **84.3%** ❌ (down from 86.6%) · duplicates 1.1% ✅ | EXP-2026-04 |
| SAM3, single "person" prompt | PASS | **27.4 FPs/100** ❌ — ~3× the 3-prompt rate; the gendered prompts were an accidental specificity filter | EXP-2026-04 |
| SAM3, single "person" prompt | Object set | **68.7%** of images gain a false person ❌ (worse than 62.2%; 193 vs 144 FPs/100) — prompting amplifies, not fixes, form-matching. **⚠ 2026-07-15: gallery review found real people in the object set — BOTH object-set rates (62.2% and 68.7%) are upper bounds pending a re-verification pass + CPU re-score (EXP-2026-04 §2.5)** | EXP-2026-04 |
| Replacement pipeline Stage 0 (+ person-verifier gate) | same | not built | — |
| Production YOLO-MIT v9 (never-shipped GELAN ckpt, conf 0.45) | CrowdHuman / PASS / Object set | crowd recall **39.6%** · PASS **1.43 FPs/100** · objects **53.7%** img-FP | EXP-2026-12 |
| Shipped `v11nclean2` baseline (YOLO11n, old labels) | same three | crowd **34.9%** · PASS **1.17** · objects **54.1%** ❌ (the users' blurs-books bug, measured) | EXP-2026-12 |
| YOLO26n fine-tuned on Spotlight labels | same three | crowd **32.8%** (lost small-person recall vs MIT) · PASS **0.23** ✅ · objects **12.7%** ✅ best measured | EXP-2026-12 |
| gelansfav14_gemlb_v1 (coworker Spotlight retrain, grey-masked unknowns) | same three | crowd **46.2%** best of the four · PASS **0.23** ✅ · objects **18.2%** | EXP-2026-12 |
| YOLOE-26s — zero-shot prompts `woman,man,child` | LAGENDA | recall **13.6%** ❌ at frozen conf 0.45; **93.8%** at the 0.05 floor — but gender then fails (see C3); no threshold passes both | EXP-2026-13 |
| YOLOE-26s — zero-shot | CrowdHuman | recall **25.3%** at the conf floor ❌ — **worst detector measured** (SAM3 77.1%, best VLM 59.8%); precision 95.7% | EXP-2026-13 |
| YOLOE-26s — zero-shot | PASS / Object set | PASS **2.5 FPs/100** ✅ · objects **46.3%** img-FP at usable recall (frozen-config 2.7% = nothing-detected artifact) | EXP-2026-13 |
| YOLOE-26s + distractor vocab `statue,mannequin,doll` | all four | **NOT adopted**: FP drop 22.5%/14.1% rel ❌ (bar ≥30%) at zero recall cost (≤0.2 pt) ✅ — trick safe but under-powered on this model | EXP-2026-13 |

### Component 2 — Age group (Adult vs Child)

| System | Dataset | Headline result | Source |
|---|---|---|---|
| SAM3, production 3-prompt | LAGENDA | adult→Child is **93% of its classification errors** (the bad direction) · child recall 97.0% ✅ · implicit fade: 99% "child" ≤9 → 36% at 15–19 → 0% by 30+ | EXP-2026-02 |
| Qwen2.5-VL-7B (verifier prompt) | LAGENDA | child recall 99.6% · **teens read as child ~68%** · abstains ~19% · fixed ~18 implicit boundary | EXP-2026-01 |
| MiVOLO V2 | LAGENDA | **untested — main candidate**, drops into the existing harness as-is | EXP-2026-05 (planned) |
| EXP-2026-12 four-way (adult(≥20)→Child leak) | LAGENDA | MIT v9 **0.77%** · shipped v11n **0.52%** · y26n-ft **0.11%** · gemlb_s **0.00%** — Spotlight labels fix the leak direction | EXP-2026-12 |
| YOLOE-26s zero-shot (prompted `child`) | LAGENDA | **adult(≥18)→Child ≤ 0.73% at EVERY conf threshold** ✅ (0.00% at frozen config) — best age direction measured; `child` = under-10 detector (call rate 100% at 0–4, 0% at 10+), teens→adult = safe/blurred | EXP-2026-13 |

Two architectures, same teen failure band → if MiVOLO matches it, the answer is an ensemble + a
default-to-adult policy for the ambiguous ~10–19 band (adult = blurred = the acceptable error),
not a better single model.

### Component 3 — Adult gender (Man vs Woman)

| System | Dataset | Headline result | Source |
|---|---|---|---|
| SAM3, production 3-prompt | LAGENDA (general domain) | gender-only error **~1.2%** | EXP-2026-02 |
| Qwen2.5-VL-7B | LAGENDA (general domain) | **~99.3%** when it commits, at every age | EXP-2026-01 |
| any system | **Gulf/traditional-dress slice** | **UNTESTED — the single biggest open gap.** General-domain numbers above do not cover it; Qwen2.5-VL-7B likely shares the production model's attire bias (same-flavor training data) | Track 2/3, not scheduled |
| Production YOLO-MIT | Gulf/traditional-dress | known weakness (thobe/ghutra men → "Woman") — the bias that motivated this whole lab; not yet quantified | docs/AUDIT_HISTORY.md |
| EXP-2026-12 four-way (adult gender on matched, LAGENDA) | LAGENDA (general domain) | MIT v9 **87.2%** · shipped v11n **89.7%** · y26n-ft **90.7%** · gemlb_s **90.6%** | EXP-2026-12 |
| YOLOE-26s zero-shot | LAGENDA (general domain) | **97.2%** at frozen conf 0.45 ✅ but only on 13.6% of people; **85.4%** ❌ at the conf floor where detection works — gender quality and recall trade against each other, no threshold passes both | EXP-2026-13 |

## What the scoreboard says to do (2026-07-14 reading)

- **Component 1 is not solved** — SAM3 needs a person-verifier gate (matches human FORM, never
  asks "real person?") and better crowd recall. EXP-2026-04 answered the prompt question:
  **person-prompt Stage 0 is rejected** — a single "person" prompt makes SAM3 fire more liberally
  everywhere (~3× PASS hallucinations, worse on dolls) for only +2 points of light-occlusion
  crowd recall; occlusion, not prompting, is the binding constraint. (\* crowd numbers pending
  one unresolved image with 310 GT persons — see the experiment doc §2.5.)
- **Component 2 is the unsolved classifier** — teen band fails across architectures; MiVOLO V2
  (EXP-2026-05) decides ensemble vs default-to-adult policy.
- **Component 3 is decided by a dataset that doesn't exist yet** — build the Gulf-dress slice.
  ~99% general-domain gender means nothing until that slice is measured.

**2026-08-05 addendum (EXP-2026-12/13):** the four-way retrain comparison shows Spotlight
labels — not architecture — fix the FP and leak problems (objects 53.7→12.7%, leak
0.77→0.11% for y26n; gemlb shows masking unknowns also recovers crowd recall). Zero-shot
YOLOE is rejected for every seat (crowd 25.3% at best; detection and gender never pass at
the same threshold), and its distractor-vocabulary trick is validated safe but under-powered
(22.5% rel FP drop vs 30% bar) — kept on the shelf for future promptable detectors. Its one
keeper: a prompted `child` class never leaks adults to Child at any threshold — evidence a
conservative child vote exists for a future age ensemble.
