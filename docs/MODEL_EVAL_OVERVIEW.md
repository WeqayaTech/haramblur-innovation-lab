# How we evaluate models — the short version

One page, high level. The full rules live in `docs/COMPONENT_FRAMEWORK.md` and
`docs/MODEL_EVAL_PROTOCOL.md`; this is the why, plus exact metric definitions.

## The two failures we evaluate against

Every model is judged against the two ways HaramBlur actually fails its users:

1. **False positives — blurring things that aren't people.** The #1 user complaint (books,
   statues, backgrounds getting blurred) and the direct motivation for the Spotlight
   relabeling and every retrain since. A product that blurs junk gets disabled — and a
   disabled product protects nobody.
2. **An adult escaping the blur** — missed by the detector, misread as a child, or given
   the wrong gender. This defeats the app's purpose (gaze-lowering) even when the user
   doesn't file a complaint about it.

Both directions matter and they trade against each other (a stricter detector blurs less
junk but misses more people), which is why we never report one pooled "accuracy" number —
it hides which direction just got worse.

## Three components, because models fail in three different ways

Every model is scored as three separate jobs, each conditioned on the previous one:

1. **Detection** — find every person; put nothing on non-people.
2. **Age** — given a found person: adult or child?
3. **Gender** — given a found adult: man or woman?

These fail independently (a model can be near-perfect on gender while missing 1 in 4
people in crowds), so a combined score can't tell you what to fix.

## Four datasets, one question each — scored per dataset, never pooled

| dataset | what it asks |
|---|---|
| LAGENDA — 5,000 people, human age/gender labels | age + gender quality on found people |
| CrowdHuman — 517 imgs, 11,641 people, exhaustively boxed | detection recall/precision in hard scenes |
| PASS — 3,000 verified person-free scenes | hallucination rate on ordinary images |
| Object set — 259 hand-checked dolls/statues/toys | the blurred-books bug, directly |

## The metrics, precisely

All matching is greedy one-to-one, IoU ≥ 0.5, each GT box claimable once. All label-file
metrics are at the production operating point: 640 px input, confidence 0.45.

**False-positive side (the complaint):**

- **Object-set image-FP rate** = images gaining ≥ 1 detection ÷ 259. (No people exist in
  these images, so every detection is a false person.) Also reported: FP boxes per 100
  images.
- **PASS FPs/100** = total detections ÷ 3,000 × 100. Same logic on ordinary empty scenes.
- **Crowd precision** = matched detections ÷ all detections, on CrowdHuman (detections
  inside annotated ignore-regions excluded). **Duplicates** = extra detections landing on
  an already-claimed person ÷ matched detections.

**Escape side (the purpose):**

- **Detection recall** = GT people matched by any detection ÷ all GT people. Reported on
  CrowdHuman (stratified by occlusion level) and on LAGENDA. An unmatched person = an
  adult who can never be blurred.
- **Adult→Child leak** = matched people with GT age ≥ 20 that the model labeled Child ÷
  all matched GT-≥20 adults. The single most consequential classification number (Child =
  deliberately unblurred). The full age-band curve (child-rate by GT age band) is reported
  with it, because every architecture fades through the teens.
- **Gender accuracy on adults** = correctly gendered ÷ matched GT adults. Wrong gender
  blurs the wrong people AND un-blurs the right ones — both directions count.
- **Child recall (≤ 12)** = GT children labeled Child ÷ matched GT children. Reported, but
  it is the acceptable-error direction (an over-blurred child is safe), so it never
  outranks the leak.

**Supporting:**

- **mAP (AP50 / AP75 / AP@[.5:.95])** — area under the precision-recall curve, sweeping
  the model's own confidence ranking (COCO 101-point interpolation), per class then
  averaged. Threshold-free, so it cross-checks that a verdict isn't a conf-0.45 artifact —
  but it averages over all thresholds and both failure directions, so it is never the
  decision number. Computed only where GT is exhaustive (Spotlight val 3-class;
  CrowdHuman as single-class person-mAP).
- **Latency** = median of 100 warm ONNX Runtime CPU inferences, 4 threads, 640 px, on one
  machine per session — cross-machine numbers are never compared. Model size = params +
  ONNX bytes.

## The rules that keep the numbers honest

- **Absence of a label is not evidence of absence.** On partially-labeled data, a
  detection landing on a real-but-unlabeled person must be *ignored* (COCO iscrowd
  semantics), never scored as a false positive. This bug class was found and fixed four
  times in this project (LAGENDA's 1-person-per-image sampling, PASS survivors, object-set
  re-verification, Spotlight-val's deleted unknowns) — new benchmarks start from this rule.
- **Bars pre-registered** before a run starts — results can't move goalposts.
- **Identical protocol per model**: same images (verified by count), same settings, same
  matching code.
- **Raw outputs logged to conf 0.05** in sidecars — threshold sweeps and mAP replay
  offline, no re-runs.
- **Every experiment ships a verify-it-yourself appendix** — actual code and thresholds,
  so reviewers check the mechanism, not a summary.

## What a model must show to win a seat

Beat the incumbent on the complaint metrics (object/PASS FPs) and hold or improve the
escape metrics (recall, leak, gender) — or vice versa — with **no regression** on the
other side, at a size and latency the product can ship. Accuracy, size, and speed are one
table; winning a single column is not winning.

## The known blind spot

The Gulf/traditional-dress gender slice — the bias that motivated this lab — still has no
hand-checked evaluation set. Every gender number we publish carries that caveat until the
slice exists.
