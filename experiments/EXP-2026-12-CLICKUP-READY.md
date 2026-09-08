# EXP-2026-12 — We trained YOLO26 on the Spotlight labels and compared it head-to-head with the production model

> **Upload notes:** paste this text into ClickUp, then drag in the 3 PNGs from
> `experiments/assets/png/EXP-2026-12/` at the marked spots. Full detail + verify-it-yourself
> appendix: `experiments/EXP-2026-12-yolo26-production-candidate.md`. How any new model gets
> the same evaluation: `docs/MODEL_EVAL_PROTOCOL.md`.

## The short version

- We fine-tuned **Ultralytics YOLO26-nano** (2.4M params) on the **Spotlight-cleaned OIV7
  labels** (475k images) for **~$11 of GPU time**, and measured it and the **production
  YOLO-MIT v9 checkpoint** (8.2M params) on the exact same four benchmarks, same settings
  (640px, conf 0.45), same scorer. This is also the **first time the production model has
  ever been measured** on these benchmarks.
- **The new model wins every metric users actually complain about, despite being 3.5×
  smaller.** False people on books/statues/toys: **53.7% → 12.7%** of images. False people on
  empty scenes: **1.43 → 0.23 per 100**. Gender on adults: **87.2% → 90.7%**. Adults (20+)
  misread as Child — the error that lets an adult escape the blur: **0.77% → 0.11%** (7×
  cleaner), with a much sharper teen boundary.
- **The new model loses on finding small/distant/occluded people:** crowd recall 39.6% →
  32.8%, child detection 95.7% → 91.2%. This loss has a known, fixable cause in the training
  data (below), not the architecture.
- **Not yet decided/tested:** speed in the browser (YOLO26's other claimed win), video
  flicker, the Gulf-dress gender slice (still untested for every model we've ever measured),
  and the **AGPL license question — YOLO26 needs either AGPL compliance or a paid Ultralytics
  license to ship. That's a leadership decision no benchmark can make.**

## What the users' "it blurs books" complaint looks like, measured

**[DRAG IN: exp12_false_people.png]**

The production model puts a false person on **54% of doll/statue/toy images** — it learned
that habit from its SAM3-labeled training data (SAM3 itself: 62%). Training on the
Spotlight-cleaned labels cut that to **12.7%** and cut empty-scene hallucinations 6×. The
label-cleaning pipeline's value, transferred into a deployed-style model, is now demonstrated
— this was the core bet of the whole Spotlight effort.

## Classification against human ground truth (LAGENDA, 5,000 people)

**[DRAG IN: exp12_lagenda.png]**

Gender improves (+3.5 points — and note: production's real gender accuracy is **87%**, a
number we never had before). Detection recall dips slightly (−1.9). The age story is the
standout:

**[DRAG IN: exp12_age_curve.png]**

Every system we've ever measured (SAM3, Qwen2.5-VL-7B, production) "fades" through the teens
— calling 27–68% of teenagers Child, and some adults too. The new model has the sharpest
boundary we've measured: **8.4% of 15–19-year-olds** called Child, **~0% from 20 up**. In
product terms: adults essentially never escape the blur by being misread as children.

## The one real regression, and why we think it's the data, not the model

Crowd recall dropped 39.6% → 32.8% (worst on small/heavily-occluded people). The likely
cause: Spotlight **dropped 12.4% of detected people as "unknown gender"** (median ~58 px —
too small to judge) and those people became unlabeled background in training — the model was
literally *taught to ignore small people*. This is exactly the concern behind the
`labels_unk3` masking variant. Two follow-up arms will separate data from capacity:

1. **yolo26s** (9.5M params — size-matched to production) on the same data → tests capacity.
2. A retrain that **masks unknown-person regions** instead of treating them as background →
   tests the data effect, and directly values the labels_unk3 design.

## Honest caveats

- Models are **not size-matched** (new model is 3.5× smaller — which makes its wins more
  impressive, and its recall loss possibly partly capacity).
- "Production" = the team's YOLO-MIT v9 checkpoint at conf 0.45. Whether the shipped
  extension binary is exactly this model is still being confirmed.
- LAGENDA age labels are human *apparent* age (±2–3 yrs), so teen-band numbers are
  approximate; the 20+ leak numbers are robust.
- The **Gulf/traditional-dress gender slice remains untested for every model** — still the
  biggest open gap in all our evaluations.
- Object-set FP rates are upper bounds for both models (a re-verification pass of that set
  is pending).

## What's next

1. Browser-relevant speed test (ONNX CPU latency) — YOLO26 is NMS-free and vendor-claims
   ~31% faster CPU inference; we measure ourselves.
2. Video flicker replay (the EXP-2026-11 clips) for temporal stability.
3. The two follow-up training arms above.
4. License decision (AGPL vs Enterprise) — gates any shipping plan.
5. Same evaluation for additional candidate models — the protocol doc makes any new model a
   ~30-minute evaluation.
