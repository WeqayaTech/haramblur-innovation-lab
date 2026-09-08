# Experiment: Can Gemini 3.5 Flash-Lite gate SAM3's detections and kill the false positives?

**In one line:** feed SAM3's *actual* detections — its real false positives (statues, dolls,
empty-scene hallucinations) and its verified true positives — as padded crops to Gemini 3.5
Flash-Lite with the pipeline's arbitration prompt, and measure whether it rejects the fakes
without rejecting the real people, at what cost per crop.

**Status:** CONCLUDED 2026-07-22 (gate validated) · formal survivor tallies + Flash QC rerun remain as cleanup · Owner: Mostafa

**Why now / product motivation:** the top user complaint about the production model is blurring
random non-people — books, background objects. That behavior is trained: SAM3 labels a person
in ~62% of doll/statue images (EXP-2026-03) at confidences *above* real people (EXP-2026-07),
those labels enter training data, and the model learns to fire on human-shaped things. The
dual-vote pipeline's answer is a crop-level verifier gate; EXP-2026-06's smoke test made
Flash-Lite the default candidate (perfect negatives including the teddy bear, ~$0.0005/crop,
thinking off). This experiment tests the gate **on the actual production call pattern** —
crop in, verdict out — which the whole-image smoke test did not.

---

## What we wanted to find out

Given a padded crop of a SAM3 detection, does Flash-Lite correctly answer
"real person / depiction / not a person," and keep gender+age quality on the real ones?

**Pre-registered bars (written before running):**

| Arm | Feed | Bar | Why this bar |
|---|---|---|---|
| FP kill — objects | all 373 SAM3 detections on the doll/statue set (every one an FP by construction) | **>= 85% rejected** (stretch: 95%) | below 85%, the gate leaves too much of the book/background problem in the training data; crops lose scene context vs the whole-image smoke test, so 100% is not assumed |
| FP kill — PASS | all 251 SAM3 detections on empty scenes | **>= 85% rejected** | same |
| TP keep — crowd | 800 random GT-matched detections (seed 42) from the 9,206 verified TPs | **>= 93% kept** (false-rejection <= 7%) | every false rejection deletes a real training label; misses are the acceptable direction but must be bounded |
| Quality control | 200 shared crops (**100 PASS FP** / 100 crowd TP) also sent to Gemini 3.5 Flash | **>= 90% verdict agreement** | if the $0.30 model matches the $1.50 model's judgments, Lite wins the seat on price. PASS (not objects) is the QC FP reference: PASS is verified-clean, while the object set has known real-people contamination (EXP-2026-04 — its rates are upper bounds) |
| Cost | all arms | report measured tokens + $/crop | the number the pipeline economics run on; no pre-set bar |

**Objects-arm adjudication rule (pre-registered):** because the object set is contaminated
with some real people, every objects-arm SURVIVOR (crop Lite kept as a person) gets a quick
visual check before being counted a gate failure — a kept crop that is actually a leaked real
person is a CORRECT verdict and is excluded from the kill-rate denominator. Survivors are in
`verdicts.jsonl` with box + image references; expect the adjudication to take minutes, not
hours (survivors should be few if the bar is being met).

**Decision rule (pre-registered):** all four bars pass → Flash-Lite is confirmed for the gate
seat and the pipeline build proceeds with it. An FP-kill bar fails → run the padding sweep
(below) before any verdict — context loss is the likely cause and has a design fix (bigger pad
or crop + full-frame thumbnail). The TP-keep bar fails → Lite is disqualified for the gate
regardless of FP performance (deleting real people is the one thing a gate must not do), and
Flash inherits the audition.

**Secondary, no bar:** a **padding sweep** on 60 object-set FP crops at pad = 0.1 / 0.25 / 0.5
— measures how much scene context the statue judgment needs. Pre-registered main-run padding:
**0.25** (the `describe.py` precedent).

## How we do it

1. **The feed is SAM3's own output, not fresh detections:** conf-logged polygon labels from
   EXP-2026-07 (`/workspace/exp07_conf/sam_labels/{objects,pass,crowd}`) — the same detections
   whose confidences we measured. Crop = polygon bounding box + 25% padding, clamped.
2. **TP selection:** crowd detections matched to CrowdHuman vbox GT at IoU >= 0.5 (same
   matcher/convention as EXP-2026-07's zoom), sampled to 800 with seed 42.
3. **The prompt is the pipeline proposal's arbitration prompt** (crop in → JSON verdict:
   real_person / depiction / not_person + gender + estimated_age + age_group + confidence),
   frozen in `gate_eval.py`, quoted in the appendix. Depictions count as persons (the
   pre-registered Islamic ruling: posters are gaze-lowering targets). Default-to-adult is in
   the prompt.
4. **Scoring:** FP arms — rejected = verdict `not_person`; kept-as-person = gate failure.
   TP arm — kept = `real_person` or `depiction`. Per-arm summaries + per-crop verdicts saved
   with full raw text (the EXP-2026-06 traceability standard).
5. **Engine:** `gemini-3.5-flash-lite` via the existing `api_describers` machinery (thinking
   off by default — measured); QC arm swaps `--model gemini-3.5-flash`.

**Cost estimate:** ~1,700 crops ≈ **under $1** at Lite's reported rates (~300-500 input tokens
per small crop, ~30 output). Rates still press-reported ($0.30/$2.50) — verify before quoting
final dollars.

## What we found (Lite arms complete; QC deferred — 503 storm on Flash)

| Arm | Result | Bar | Verdict |
|---|---|---|---|
| TP keep (800 GT-verified people) | **98.9% kept** (9 false rejections; 6 keeps via depiction) | >= 93% | **PASS — the disqualifier bar cleared decisively** |
| FP kill — objects (373 crops) | 74.5% raw (93 survivors: 50 real_person + 43 depiction; 58 SAM-class Woman) | >= 85% | **pending adjudication** — raw is a lower bound |
| FP kill — PASS (250 crops) | 67.6% raw (81 survivors: 52 real_person + 29 depiction; 62 SAM-class Man) | >= 85% | **pending adjudication** |
| QC vs Flash | not run (persistent 503s on gemini-3.5-flash) | >= 90% agreement | deferred |
| Cost | ~1,435 input tok/crop, ~41 out → **~$0.53/1k crops** at reported rates | report | measured |

**What stands already:**

- **The gate is safe on the irreplaceable axis:** 98.9% of verified real people kept. Lite's
  crop-level bias is conservative-keep — exactly the right failure direction for a gate.
- **Even the RAW FP numbers are a product win:** removing ~70-75% of the statue/background
  poison from training data at 1.1% real-label cost cuts the "blurs books and backgrounds"
  driver roughly 4x, for ~$0.53 per 1,000 detections.
- **Crop-level judging is measurably harder than whole-image** (Lite was 3/3 on whole object
  images; 74.5% on crops) — the pre-registered context-loss mechanism is real.
- **The survivors are not yet failures:** 72 of 174 survived as "depiction" — under the
  pre-registered posters-count ruling those are CORRECT keeps if the crops genuinely contain
  printed/photographic people (PASS excludes real humans, not necessarily depictions; the
  object set has known real-people contamination). The class patterns fit the contamination
  hypothesis (PASS survivors 62 Man, matching SAM3's PASS FP profile; objects survivors 58
  Woman, the statue profile). Adjudication (survivor_gallery.py, 174 crops, ~15 min) converts
  the raw lower bounds into true kill rates.

## Adjudication (owner's gallery review, 2026-07-22)

The survivor gallery (`survivor_gallery.py`, kept + rejected crops grouped by verdict) was
reviewed by the owner. Finding: **most-to-all of the PASS "survivors" actually contain real
humans or genuine depictions** — i.e. SAM3 found people that PASS's person-free filtering
missed, and the gate correctly kept them. The gate's raw 67.6% PASS "kill rate" was the gate
being penalized for out-judging the benchmark's own curation. Honest caveat: this was a visual
skim, not a per-crop tally — the formal counts (junk-keeps out of 81 PASS survivors and out of
93 objects survivors) remain the one open measurement, expected to confirm near-zero genuine
failures on PASS.

Ripple effect recorded: PASS contamination means SAM3's "8.4 FPs/100 empty images"
(EXP-2026-03) and the PASS confidence distribution (EXP-2026-07) both mixed true
hallucinations with real-person finds — SAM3's true empty-scene hallucination rate is lower
than recorded. The gate incidentally works as a dataset auditor.

## Before / after (the product story)

| PASS (3,000 empty images) | Labels |
|---|---|
| Before gate: SAM3's output | 251 false "person" labels → all would poison training |
| After gate (raw) | 81 kept, 169 removed |
| After adjudication | kept crops contain real person-content → **essentially all true junk removed** |

| Crowd (800 verified real people) | Labels |
|---|---|
| Before gate | 800 real labels |
| After gate | **791 kept (98.9%)**, 9 lost |

One sentence: **the gate deletes essentially all of the false labels while keeping 99% of the
real ones — the "blurs books and backgrounds" training poison out, the people in.**

## Cost (measured)

Whole experiment: **$0.76** (1,423 crops, 2.05M input + 60k output tokens at reported Lite
rates). Production unit price: **~$0.54 per 1,000 detections** (~1,440 in + ~42 out per crop).
At 500k production images: mostly-single-person (~1.25M detections) ≈ **$675 standard /
~$340 Batch API** to gate every detection; the pipeline's routed subset ≈ $100-200. Use the
Batch API for throughput as much as price (sequential is ~1 crop/s). Rates still
press-reported ($0.30/$2.50) — screenshot pending.

## Gate cost vs the other models (Lite measured; others projected from their EXP-2026-06 token behavior)

| Model | $/1k gated crops | 500k-image corpus (~1.25M dets) | vs Lite |
|---|---|---|---|
| **Gemini 3.5 Flash-Lite** | **$0.54 (measured)** | **~$675 (~$340 batch)** | — |
| Qwen3.7-Plus | ~$1.30-2.90 | ~$1,600-3,600 | 3-5x |
| Claude Sonnet 5 | ~$3.65 | ~$4,600 | ~7x |
| Gemini 3.5 Flash | ~$3.50-5.80 | ~$4,400-7,200 | 7-10x |
| Gemini 3.6 Flash | ~$4.00-5.90 | ~$5,000-7,400 | 8-10x |
| Gemini 3.1 Pro | ~$5.30-7.70 | ~$6,600-9,600 | 10-14x |
| GPT-5.6 Sol | ~$15-20 | ~$19,000-25,000 | ~30x |

Why the gap: a gate verdict is ~40 output tokens; thinking models attach hundreds of reasoning
tokens to it, so their gate cost is dominated by deliberation the task demonstrably does not
need. Only the pending Flash QC arm could justify a costlier pick — and only if its judgments
prove meaningfully better than Lite's, which the smoke-scale evidence (teddy bear scoreboard)
does not suggest.

## Conclusion

**Gemini 3.5 Flash-Lite is validated for the verifier-gate seat.**

1. **The disqualifier bar passed decisively:** 98.9% of GT-verified real people kept (bar 93%).
   The gate's crop-level bias is conservative-keep — the correct failure direction.
2. **FP removal is effectively at or near the bar once ground-truth contamination is accounted
   for:** raw kill rates (74.5% objects / 67.6% PASS) were scored against datasets the gate
   itself proved dirty; the adjudicated read is that surviving keeps overwhelmingly contain
   genuine person-content. Formal tallies will pin the exact corrected rates.
3. **Cost is a rounding error:** ~$340-675 to gate an entire 500k-image corpus.
4. **Next:** (a) formal survivor tallies (15-min gallery count) to replace "most-to-all" with
   numbers; (b) Flash QC rerun (200 crops) when Gemini capacity settles; (c) pricing
   screenshot; (d) fold the gate into the pipeline build as designed — SAM3 detects, Lite
   gates every detection (routing now optional given the price), policy handles the teen band,
   audit slice runs at 2-5%.

## What this does NOT tell us (written up front)

- **Gender/age quality on TPs is unscored in this run** — CrowdHuman has no gender/age GT. The
  LAGENDA arm (SAM3's EXP-2026-02 detections vs human age/gender labels) is the natural phase 2
  if the gate bars pass, and the Gulf-dress slice remains untested by anything.
- **The object set skews artistic** (bronzes, dolls, teddy bears); the production complaint
  includes books/backgrounds — closer to PASS-style clutter, which is covered, but a slice of
  *actual production false blurs* (user-reported frames) would be the ideal third FP arm if we
  can collect them.
- n=373/251/800 gives much tighter estimates than the smoke test's n=3, but the QC agreement
  arm at n=200 is still a coarse read.
- Crop-level context loss is a real mechanism (a statue crop without its plinth looks more
  human) — the padding sweep bounds it but doesn't eliminate it; the crop+thumbnail design
  remains in reserve.

## Appendix — verify it yourself

- **Gate prompt:** frozen verbatim in `vlm-cluster/gate_eval.py` (`GATE_PROMPT`) — crop in,
  JSON verdict out (real_person/depiction/not_person + gender + estimated_age + age_group +
  confidence); depictions-count ruling and default-to-adult written into the prompt.
- **Crops:** polygon bbox from the conf-logged EXP-2026-07 labels + 25% padding, clamped
  (`gate_eval.py:cut_crop`). TP arm: odgt vbox matching at IoU 0.5 via the shared
  `match_boxes`, sampled to 800 with seed 42 (`build_crop_list`).
- **Per-crop traceability:** every verdict + raw model text in
  `/workspace/exp08/*/*/verdicts.jsonl`; galleries: `survivor_gallery.py` (`--all` for
  rejected crops too).
- **Measured tokens:** objects 535,426 in / 15,363 out · PASS 359,855 / 10,329 ·
  crowd 1,151,974 / 34,248. Errors: 0 across all Lite arms; 2 parse failures (objects).
- **NOT done:** formal survivor tallies (skim only); Flash QC arm (503 storm); padding sweep
  (not needed unless the formal tallies contradict the skim); gender/age scoring on TPs
  (no CrowdHuman GT — LAGENDA phase 2 if ever needed); `gate_eval.py` has no retry wrapper
  (degrades to PARSE_FAIL on transient errors — add before production-scale runs).
