# EXP-2026-13 — Can YOLOE (promptable, open-vocabulary) serve as a production candidate — and can distractor prompts kill the doll/statue failure?

**In one line:** point YOLOE at our benchmarks with the text vocabulary `woman, man, child` —
zero training — and measure all three components; then test the one trick no closed-vocabulary
model can do: adding `statue, mannequin, doll` as *distractor* classes to absorb the
human-shaped-object false positives that SAM3 fails at 62% and users experience as blurred books.

**Status:** Phases A+B RUN COMPLETE 2026-08-05 (L4 pod, ~2.5 h GPU, $0 API). Q1 = no at every
operating point; distractor vocabulary NOT adopted (B1 fail). Phases C/D/E did not run, per
pre-registered conditions. Remaining: optional gallery adjudication of absorbed crops. ·
Owner: Mostafa

---

## The short version

**Zero-shot YOLOE is rejected for every seat, and not because of the threshold.** At the
frozen production config (conf 0.45) it detects almost nothing (LAGENDA recall 13.6%, crowd
0.27%). The offline confidence sweep — free, replayed from the raw sidecars — shows there is
no operating point that works: at conf 0.05 detection nearly reaches its bar (93.8%) but
adult gender collapses to 85.4% (bar 95%), and crowd recall tops out at **25.3%**, the worst
of any detector this lab has measured. **The distractor-vocabulary trick is safe but
under-powered:** it stole essentially zero real people (recall loss ≤ 0.2 pt — B2 passes)
but cut the object-set false-person rate only 22.5% relative vs the pre-registered 30% bar
(B1 fails) — not adopted.

Two findings survive the rejection. First, **the prompted `child` class has the best age
*direction* we've ever measured**: adult(≥18)→Child stays ≤ 0.73% across the entire
confidence curve, because "child" acts as an under-10 detector and pushes teens to adult —
the safe, blurred direction. Second, the distractor mechanism itself is validated harmless,
so it stays on the shelf as a $0 trick for any *future* promptable detector that can actually
detect. The AGPL license question is moot — nothing here earns a seat.

## Why we did this

Every detector we've measured fails the same way: it matches human FORM and never asks "real
person?" (SAM3: 62.2% of human-shaped-object images gain a false person; the production model's
trained-in symptom is blurring books and backgrounds). YOLOE ("Real-Time Seeing Anything",
THU-MIG, ICCV 2025; integrated in `ultralytics` with new YOLOE-26 variants) is a real-time
detector whose classes are **text prompts**. That buys three things worth measuring:

1. **Zero-shot 3-class production behavior for free** — prompt `woman, man, child` and the
   prompt order *is* the production class mapping. If accuracy is usable, that's a candidate
   with no training run at all, exportable to ONNX with the vocabulary baked in
   (re-parameterized to a plain YOLO graph, zero inference overhead — so runtime speed is a
   normal YOLO's).
2. **Distractor vocabulary** — add `statue, mannequin, doll` to the prompt list and *discard*
   those detections. A closed-vocab model must decide person-or-nothing; YOLOE can hand the
   statue a better-fitting label than "woman". Nothing in our stack can currently do this at
   detector speed — today's fix is the Flash-Lite gate at $0.54/1k crops.
3. **A possibly-better fine-tune base** — linear-probe or full fine-tune on the Spotlight
   labels, directly comparable to EXP-2026-12's YOLO26 arm.

**License, flagged up front:** YOLOE is **AGPL-3.0 twice over** (THU-MIG repo AND the
ultralytics integration; weights included). Same owner/legal gate as EXP-2026-12 — if the
answer is no for shipping, the zero-shot arm still matters as a potential *labeling-pipeline*
seat (server-side, where AGPL is a different conversation), so Phases A/B run regardless.

## What we wanted to find out

Three questions, each with its own bars:

- **Q1:** Is zero-shot text-prompted YOLOE accurate enough on Components 1–3 to be worth
  anything (production candidate, or detector/gate seat in the labeling pipeline)?
- **Q2:** Do distractor prompts cut the human-shaped-object false-positive rate without
  stealing real people?
- **Q3 (conditional):** Does fine-tuning YOLOE on Spotlight labels beat YOLO26 trained on the
  same data (EXP-2026-12 Phase B), i.e. does open-vocab pretraining transfer?

Components measured (per `docs/COMPONENT_FRAMEWORK.md`): 1, 2, 3 — decomposed by conditioning,
scored per dataset, never pooled.

### Pre-registered bars (written 2026-08-03, before running anything)

**Phase A — zero-shot, prompts `woman, man, child`** (`yoloe-26s-seg.pt` primary,
`yoloe-26m-seg.pt` secondary if the s-run is healthy):

- **LAGENDA detection recall ≥ 95%** (SAM3's 98.7% is the anchor; zero-shot gets 3.7 pts).
- **Gender on correctly-detected adults ≥ 95%**; **adult(GT ≥ 18) → Child rate is the headline
  Component 2 number** — reported with the full age-band curve (the teen fade every other
  architecture shows), barred at **≤ 2%**.
- **CrowdHuman: recall ≥ 60% at precision ≥ 85%** (beats every commercial VLM measured in
  EXP-2026-06 — best was Sol 59.8% — while respecting that SAM3's 77.1% is a labeler, not
  zero-shot).
- **PASS: ≤ 10 FPs/100 images** (SAM3's 8.4 is the anchor).
- **Object set: image-FP rate < 62.2%** (strictly better than SAM3's measured failure).
- **Worth-anything gate:** LAGENDA detection + gender bars both pass. Fail either → Q1 answered
  no; Phase B still runs (it's cheap and answers a different question), Phase C does not.

**Phase B — distractor arm, prompts `woman, man, child` + `statue, mannequin, doll`**
(detections claimed by a distractor class are discarded; **`poster` is deliberately NOT a
distractor** — posters/photos of people COUNT as people per the pre-registered EXP-2026-03
ruling):

- **B1: object-set image-FP rate drops ≥ 30% relative** vs Phase A's own rate (e.g. 40% → ≤28%).
- **B2: the cost is ~nothing** — LAGENDA detection recall loss ≤ 1 pt AND CrowdHuman recall
  loss ≤ 1 pt vs Phase A (distractors must not steal real people; the failure mode to watch is
  a person in a shop window absorbed by `mannequin`).
- Adopt the distractor vocabulary iff B1 AND B2.

**Phase C — fine-tune on Spotlight labels (conditional on the Q1 gate):** same bars as
EXP-2026-12 Phase B/C (that doc is the master for the fine-tune comparison), plus: report
linear-probe vs full fine-tune, and zero-shot → fine-tuned delta per component.

**Phase D — export sanity (runs if A or B pass their bars):** `set_classes(...)` then
`export(format="onnx")`; the exported fixed-vocab model must (a) emit **only** the baked class
ids (a known ultralytics bug class — out-of-vocabulary ids appeared in an OpenVINO export,
issue #23250), and (b) match native detections on 20 held-out images at IoU ≥ 0.9 / same class
/ |Δconf| ≤ 0.05 for ≥ 95% of boxes. Fail → the deployment story is broken regardless of
accuracy; report as a blocker.

**Phase E — flicker (conditional on Q1 gate):** the 4 EXP-2026-11 clips via
`video_flicker_probe.py --engine ultralytics --prompts "woman,man,child"`; compared
row-for-row against the production baseline table. Reported, not barred — a zero-shot model
isn't expected to win flicker, but the `dubai_souk` Woman↔Man flip rate is the first look at
whether *prompted* gender is steadier on Gulf dress than trained gender.

## How we plan to do it

1. **The answer keys** — identical to EXP-2026-12: CrowdHuman seed-51 500 / PASS 3k / object
   set 259 for Component 1, LAGENDA 5,000-person manifest for Components 2+3, the 4 flicker
   clips for temporal. All already staged; nothing rebuilt.
2. **Phase A.** `run_ultralytics_labels.py --model yoloe-26s-seg.pt --prompts "woman,man,child"`
   over the four datasets (prompt order = class ids 0/1/2 = the production mapping, so
   `run_autolabel_on_manifest.py` and `eval_taxonomy` conventions apply unchanged). Component 1
   via `eval_negatives_crowd.py`; Components 2+3 via `run_autolabel_on_manifest.py`.
3. **Phase B.** Same command + `--distractors "statue,mannequin,doll"` over object set, PASS,
   LAGENDA, CrowdHuman. Distractor hits are excluded from label files but fully logged in the
   raw sidecars (`excluded: "distractor"`), so B1/B2 need no re-run to re-analyze — and a
   different distractor list can be *scored* offline from the sidecars only if it's a subset;
   any new word means a re-run (pre-registered: the list above is frozen for this experiment).
4. **Phase C (conditional).** `YOLOEPETrainer` full fine-tune + the documented linear-probe
   recipe on `/workspace/spotlight/run/oiv7_train/labels/` (classes 0-2 variant, same data
   caveat as EXP-2026-12). Same $100 GPU cap, pre-registered.
5. **Phase D.** Export + the two sanity checks above, CPU-only.
6. **Phase E.** Flicker probe + `flicker_metrics.py` at conf 0.45, raw floor 0.05.

GPU cost estimate: Phases A+B ≈ 2–3 h on an L4 (~$1–2); Phase C is the only real spend.

## What we found

### Phase A — zero-shot `yoloe-26s-seg.pt`, prompts `woman,man,child`, frozen config (conf 0.45) — run 2026-08-05, L4 pod

**Q1 gate: FAIL — recorded before any further analysis, per the pre-registered rule.**
LAGENDA detection recall came in at **13.6%** against the 95% bar. Adult gender on the people
it did find passes (97.2%), but the gate needs both. Q1 = no at the frozen config; **Phase C
(fine-tune) does not run.** Prompt-order assertion passed on the pod before anything ran
(`{0: woman, 1: man, 2: child, 3: statue, 4: mannequin, 5: doll}`).

| pre-registered bar | measured | verdict |
|---|---|---|
| LAGENDA detection recall ≥ 95% | **13.6%** (682/5,000) | ❌ FAIL |
| Gender on detected adults ≥ 95% | 97.2% (633/651) | ✅ |
| adult(GT ≥ 18) → Child ≤ 2% | **0.0%** (0/600) | ✅ |
| CrowdHuman recall ≥ 60% @ precision ≥ 85% | **0.27%** (32/11,641) @ 100% precision | ❌ FAIL |
| PASS ≤ 10 FPs/100 images | 0.0 (0/3,000 images) | ✅\* |
| Object set image-FP < 62.2% | 2.7% (7/259; 3 Man + 4 Woman) | ✅\* |

\* **The FP-side passes are not discrimination evidence at this operating point.** A model
that emits almost nothing (1,088 detections across 4,899 LAGENDA images; 32 across 517 crowd
images) gets clean negatives for free. Judge the FP numbers only at a threshold where recall
is usable — see the confidence-sweep addendum below.

**The failure signature is confidence calibration, not blindness.** conf 0.45 is a
*trained-model* production setting; zero-shot open-vocab confidences run structurally lower.
The raw sidecars logged every detection down to 0.05 (log-raw principle), so the operating
curve is re-derivable offline with zero GPU re-run — that sweep is the pre-registered-style
follow-up (new section, thresholds stated before scoring; the 0.45 frozen-config verdict
above stands regardless).

What the detected slice shows (biased slice — detection itself is skewed: Woman 19.5% /
Man 14.7% / Child 2.6% recall-by-class):

- 3-class accuracy on matched people 94.9%; adult gender 97.2%; **zero adults of any age
  called Child** (catches_adults = 100% at every cutoff — the consequential direction is
  perfectly clean on this sample, n=600 at ≥18).
- The `child` prompt behaves as an **under-10 detector**: call rate 100% at ages 0–4, 60% at
  5–9, 0% at 10+ — an even sharper fade than SAM3 or Qwen2.5-VL-7B's teen band, in the safe
  direction (teens read as adults → blurred).
- Box quality on matches is good: 82.6% of matched IoUs ≥ 0.9, only 2.5% below 0.75; zero
  dual-class survivors.
- Crowd scorer note: `n_images_missing` 3,853 = odgt entries outside our frozen 517-image
  sample — expected, not an error.

### Phase A addendum — offline confidence sweep from the raw sidecars (2026-08-05)

Labels re-emitted from the sidecars at four extra thresholds (zero GPU re-run), scored with
the unchanged scorers. NMS ran once at the 0.05 floor — same accepted replay trade as
EXP-2026-11/12.

| conf | LAGENDA recall | adult gender | adult(≥18)→Child | crowd recall | crowd precision | objects %img-FP | PASS FP/100 |
|---|---|---|---|---|---|---|---|
| 0.05 | 93.8% | 85.4% | 0.73% | 25.3% | 95.7% | 46.3% | 2.5 |
| 0.10 | 82.1% | 86.9% | 0.64% | 14.5% | 97.8% | 38.2% | 1.1 |
| 0.20 | 56.7% | 90.9% | 0.39% | 5.3% | 98.3% | 23.5% | 0.3 |
| 0.30 | 35.1% | 94.2% | 0.07% | 1.8% | 99.1% | 10.4% | 0.2 |
| 0.45 (frozen) | 13.6% | 97.2% | 0.00% | 0.3% | 100.0% | 2.7% | 0.0 |

**Q1 is a no at EVERY operating point, not just the frozen config.** The calibration
hypothesis is answered: fixing the threshold cannot rescue the model, because the two bars
trade against each other —

1. **Recall and gender move in opposite directions.** At 0.05, detection nearly reaches its
   bar (93.8% vs 95%) but gender collapses to 85.4% (bar 95%); at thresholds where gender
   passes, detection is 14–35%. There is no threshold where both hold. Low-conf detections
   are also gender-unreliable — confidence is doing double duty.
2. **Crowd recall is an architecture failure, not a threshold artifact: 25.3% at the floor.**
   Worst crowd detector we have measured — below SAM3 (77.1%), every commercial VLM
   (best 59.8%), and Flash-Lite (49.2%). No seat (production or labeling-pipeline detector)
   survives this.
3. **Objects at usable recall: 46.3% image-FP** — better than SAM3's 62.2%, roughly the
   shipped model's 54.1%, far worse than fine-tuned y26n's 12.7%. The frozen-config 2.7% was
   indeed a nothing-detected artifact.

One genuinely notable behavior, recorded for the age-component discussion: **the
adult→Child leak stays ≤ 0.73% at every threshold** including the floor — the prompted
`child` class is conservative everywhere on the curve, failing only in the safe direction
(under-10 detector; teens → adult → blurred). Zero-shot YOLOE's *age semantics* are the
best we've measured even though its detection is the worst.

**Consequences (per pre-registered rules):** Phase C does not run; the `yoloe-26m-seg.pt`
secondary arm does not run (conditional on a healthy s-run); Phase E does not run. Phase B
runs as planned — it answers the separate distractor-vocabulary question.

### Phase B amendment — pre-registered BEFORE the distractor run (2026-08-05)

The B1/B2 comparison is meaningless at conf 0.45 (Phase A emits almost nothing there: 7
object FPs total). Amendment, stated before Phase B runs or is scored: **B1 and B2 are
evaluated at conf 0.05 and 0.10** (both reported), re-emitted offline from both arms'
sidecars by the same rule (`excluded is None and conf ≥ t`). Bars unchanged: B1 = object-set
image-FP rate drops ≥ 30% relative vs Phase A *at the same threshold*; B2 = LAGENDA and
crowd recall each lose ≤ 1 pt vs Phase A at the same threshold. The distractor mechanism
acts at inference (distractor classes compete for boxes in NMS/assignment), so a GPU re-run
with the 6-word vocabulary is required; the run itself still writes labels at 0.45 for
directory-layout consistency.

### Phase B — distractor arm results (run 2026-08-05, scored per the amendment above)

| conf | objects img-FP (A → B) | rel. drop — B1 bar ≥ 30% | LAGENDA recall loss | crowd recall loss | B2 bar ≤ 1 pt each | PASS FP/100 (A → B) |
|---|---|---|---|---|---|---|
| 0.05 | 46.3% → 35.9% (196 → 148 FP boxes) | **22.5%** ❌ | 0.20 pt | 0.01 pt | ✅ | 2.47 → 1.90 |
| 0.10 | 38.2% → 32.8% (136 → 116 FP boxes) | **14.1%** ❌ | 0.08 pt | 0.00 pt | ✅ | 1.07 → 0.87 |

**Verdict: the distractor vocabulary is NOT adopted** (adoption rule, pre-registered: B1 AND
B2; B1 fails at both thresholds). Q2's answer splits cleanly in two:

- **The safety half is a clean yes.** Distractors stole essentially no real people — recall
  losses of 0.00–0.20 pt against a 1-pt allowance, on both LAGENDA and CrowdHuman. The feared
  person-in-a-shop-window-absorbed-by-`mannequin` failure mode did not materialize at
  measurable scale.
- **The power half is a no.** A ~15–23% relative FP cut is real and free, but under the 30%
  bar — most human-shaped-object FPs survive because the model genuinely reads them as
  `woman`/`man` rather than as a better-fitting distractor word.

Because the mechanism is proven harmless, it remains a $0 candidate pre-filter for any future
promptable detector that passes detection bars — the negative result is about YOLOE-26s's
vocabulary discrimination, not about the trick. (Per the pre-registered scope note, this
verdict is specific to the frozen list `statue, mannequin, doll`.)

Pending, informational only (does not affect the verdict): the 40-crop adjudication gallery
(`/workspace/exp13/distract_adjudication.html`, seed 20260805) tallying correctly-absorbed
objects vs stolen people, and the per-dataset absorbed-detection counts from the re-emit log.

**Phase D did not run** — its pre-registered condition was "A or B pass their bars"; neither
did. Phases C and E were already off (Q1 gate).

### mAP addendum (2026-08-05) — team-requested, informational

Computed with the shared evaluator (`vlm-cluster/map_eval.py`) from the raw sidecars (floor
0.05; mAP integrates over confidence, so the frozen-config threshold doesn't apply). Per
`docs/COMPONENT_FRAMEWORK.md` §"Where mAP fits", these are label-alignment numbers, not
decision-grade component metrics, and they stay in this doc rather than the scoreboard.

**Spotlight OIV7 val (3-class, 4,232 images / 8,035 GT people)** — the same GT as the
EXP-2026-12 four-way val mAP, so YOLOE's row drops straight into that comparison:

| zero-shot `yoloe-26s-seg` | AP50 | AP75 | AP@[.5:.95] |
|---|---|---|---|
| Woman | 0.410 | 0.369 | 0.349 |
| Man | 0.561 | 0.531 | 0.511 |
| Child | 0.376 | 0.335 | 0.318 |
| **mean** | **0.449** | **0.412** | **0.393** |

**CrowdHuman person-mAP (class-agnostic — GT has no gender; all arms' predictions collapsed
to one class; same 517-image frozen sample, 11,641 GT persons; odgt `mask`/ignore GT boxes
dropped, but detections inside ignore regions were NOT excluded — a small equal FP tax on
every arm, stated not hidden). NOT comparable to the 3-class OIV7 numbers above:**

| arm | AP50 | AP75 | AP@[.5:.95] |
|---|---|---|---|
| gelansfav14_gemlb_v1 | **0.655** | 0.372 | 0.371 |
| YOLO-MIT v9 (mit_prod) | 0.651 | **0.390** | **0.382** |
| shipped v11n (v11n_pt) | 0.631 | 0.382 | 0.373 |
| y26n_ft | 0.571 | 0.318 | 0.318 |
| YOLOE-26s zero-shot | 0.249 | 0.191 | 0.166 |
| YOLOE-26s + distractors | 0.249 | 0.191 | 0.166 |

Readings: (a) YOLOE's crowd deficit is confirmed at *every* confidence ranking, not just the
working threshold — 0.249 AP50 vs 0.57–0.66 for the trained arms; (b) the distractor arm is
identical to 3 decimal places (0.2488 vs 0.2486) — the no-stealing verdict holds under
ranking too; (c) among trained arms the ordering matches the recall story (gemlb ≥ MIT >
v11n > y26n_ft), with v11n outranking y26n_ft on AP despite similar conf-0.45 recall —
AP rewards its below-threshold detections; (d) LAGENDA / PASS / object-set mAP were
deliberately NOT computed (structurally invalid: non-exhaustive labels would poison
precision; negatives have no positives — see the framework doc).

### Size & latency addendum (2026-08-05) — informational export

Team-requested after the verdict; this is NOT Phase D (its class-id and parity sanity
checks were still skipped per the pre-registered rules — the accuracy bars had already
failed, so nothing rides on this export).

- Params: **13.99M fused** (15.27M counting the text-prompt machinery), 35.4 GFLOPs at 640.
- Fixed-vocab (`woman,man,child`) ONNX: **41.8 MB**; NMS-free end-to-end seg graph —
  outputs (1, 300, 38) = 6 detection fields + 32 mask coefficients, plus (1, 32, 160, 160)
  mask protos, so latency includes segmentation work a det-only graph wouldn't pay.
- Latency: **109.6 ms median** (ONNX Runtime CPU, EPYC 9254, 4 threads, 100 warm runs), with
  `y26n_spotlight` re-timed in the same session at 34.0 ms (vs its 33 ms on the EXP-2026-12
  benchmark machine — so these numbers transfer within ~3%). **3.2× slower than
  y26n_spotlight at 4.3× the file size**, with mAP50 0.449 vs 0.799 — the size/speed row
  agrees with the accuracy verdict from every angle.

## What we can decide from this

- **Production-model seat: no.** No operating point passes detection + gender together;
  crowd recall 25.3% at best. The AGPL license question never needs answering for this model.
- **Labeling-pipeline detector seat: no.** SAM3 keeps it (77.1% crowd recall vs 25.3%);
  YOLOE doesn't come close even as a zero-shot convenience.
- **Distractor-vocabulary pre-filter: not adopted here, but the trick is validated safe**
  (zero measurable person-stealing). Re-test it for free on any future promptable detector
  that passes detection bars — the 30% power bar, not safety, is what failed.
- **For the age-component discussion (EXP-2026-05 / ensemble design):** a prompted `child`
  class is the first age mechanism measured that *never* leaks adults to Child (≤ 0.73%
  across the whole confidence curve) — it fails teens in the safe direction (→ adult →
  blurred). If a future ensemble needs a conservative child vote, text-prompted
  open-vocabulary semantics are now evidence that such a vote exists.

## What this does NOT tell us

Written before running:

- **The Gulf-dress slice is still untested** — prompted gender might behave differently on
  thobe/ghutra than trained gender, and nothing here measures that beyond the dubai_souk
  temporal peek. The slice remains the biggest open gap in the lab.
- **Zero-shot LVIS numbers (YOLOE-26s ≈ 28.6 mAP) predict nothing about our 3-class task** —
  that's why we measure; vendor numbers are cited for context only.
- **LAGENDA is recall-side only; object-set rates are upper bounds** (EXP-2026-04 §2.5
  re-verification pending); **PASS is ~5% contaminated** (EXP-2026-08) — scored full-set for
  comparability AND with flagged images excluded, both reported. Same caveats as EXP-2026-12.
- **The distractor result is vocabulary-specific.** A pass/fail on `statue, mannequin, doll`
  says nothing about other lists; teddy bears (15% FP rate for SAM3) have no distractor in
  this list by design — adding `toy` risks absorbing children holding toys, and that trade is
  its own future experiment.
- **Prompt wording is a hidden knob.** We freeze one vocabulary and don't sweep synonyms
  (`lady`, `guy`, `kid`...); a failed bar could in principle be a wording failure, not an
  architecture failure. Recorded as the first follow-up if bars are near-missed.

## What's next

- **No prompt-wording sweep.** The pre-registered trigger was *near-missed* bars; crowd
  recall missed by 35 points at the confidence floor — that is an architecture verdict, and
  synonym sweeps don't move it. (Recorded so nobody re-litigates it later.)
- Optional closeout: eyeball `/workspace/exp13/distract_adjudication.html` (40 crops) and
  drop the absorbed-vs-stolen tally into the Phase B section for the record.
- Append the YOLOE rows to the four-way comparison in `EXP-2026-12-TEAM-BRIEF.md` (the team
  asked for YOLOE results — the answer is a clean rejection with numbers).
- The queue behind this per EXP-2026-12 feedback: the **4-class `labels_unk3` YOLO26 arm**
  (zero-code), then the cls-gradient-suppression arm; the Gulf-dress slice
  (30,567 Woman→Man Spotlight corrections) remains the biggest open gap in the lab.

---

## Appendix — the details

### Frozen configuration

| knob | value | source |
|---|---|---|
| ultralytics version | pinned `8.4.115` | runbook Phase 0 |
| weights | `yoloe-26s-seg.pt` (primary), `yoloe-26m-seg.pt` (secondary) | auto-download |
| prompt vocabulary | `woman, man, child` — order = class ids 0/1/2 | this doc, frozen |
| distractor vocabulary | `statue, mannequin, doll` — frozen; `poster` excluded on purpose | this doc + EXP-2026-03 §2.2 person ruling |
| working conf / NMS IoU / imgsz | 0.45 / 0.7 / 640 | production settings (see EXP-2026-12 table) |
| log-raw floor | 0.05 | log-raw-readings principle |
| match IoU | 0.5 | `match_boxes` default |

### Appendix — "verify it yourself" (written BEFORE running)

**How the model is called** — `vlm-cluster/run_ultralytics_labels.py`, prompted branch:

```python
from ultralytics import YOLOE
self.model = YOLOE(model_path)
vocab = list(prompts) + list(distractors or [])
try:
    self.model.set_classes(vocab)               # documented 8.4.x path
except TypeError:
    self.model.set_classes(vocab, self.model.get_text_pe(vocab))
```

**How distractors are excluded** — `map_detection`; the class id threshold is simply "first
`len(prompts)` ids are real", which holds because `set_classes` assigns ids in list order:

```python
if mode == "prompts":
    return (cls, None) if cls < n_prompt else (None, "distractor")
```

Every distractor hit is preserved in the raw sidecar
(`{"cls_name": "statue", "kept": false, "excluded": "distractor", "conf": ...}`), so B1/B2 are
re-derivable offline and the counterfactual (what Phase A would have called that box) is
answerable from the Phase A run of the same images.

**Label-file contract, matching, flicker machinery** — identical to EXP-2026-12's appendix
(5-field lines because 6-field lines are silently dropped by `seg_boxes`; empty file on zero
detections; unchanged `match_boxes` IoU ≥ 0.5; unchanged `probe_video`/`flicker_metrics.py`).
One doc is the master for shared machinery; this one only records what differs.

**Selftests run 2026-08-03 (no GPU/network):** `run_ultralytics_labels.py --selftest` includes
the distractor case (a distractor-only image must produce an *empty* label file, and the
sidecar must carry the excluded detection) — passes.

**Honestly NOT done at design time:** the distractor list and the 30%-reduction bar are
judgment calls, not calibrated; prompt wording is not swept; the `set_classes` id-order
assumption is asserted from documentation and verified live in runbook Phase 0 (print
`model.names` after `set_classes`, abort if order ≠ prompt order); YOLOE visual prompts and
prompt-free mode are deliberately out of scope.
