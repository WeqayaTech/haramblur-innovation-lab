# The SPOTLIGHT pipeline (v1): SAM3 detects → mask-highlighted crops → Flash-Lite gates, labels, and refines

**Name (adopted 2026-07-24):** *Spotlight* — for every detected person, the pipeline puts them
alone in the spotlight (their exact mask outlined) and asks the judge to rule on that one
person. Stages: **Detect → Spotlight → Verify → Merge → Emit+Audit.**

**In one line:** SAM3 finds every person and supplies geometry; every detection becomes a
padded crop with that person's own mask *highlighted* (not cut out); Gemini 3.5 Flash-Lite
judges the highlighted person — real? gender? age? better box? — and the merge rules assemble
the final training label.

**Status:** design spec, 2026-07-23. This is the buildable v1 of
`PIPELINE_PROPOSAL_dual_vote_labeler.md` — simplified from four models to two because
EXP-2026-08 measured that gating **every** detection costs ~$0.54/1k (Batch ~half), which makes
the dual-pipeline "route only disagreements" economics optional rather than necessary. The
dual-vote design (second detector + MiVOLO) remains the planned v2 upgrade path, not a v1
blocker. Every seat assignment below is backed by a measured number (sources cited inline).

---

## Why these two models in these two seats

| Seat | Model | Evidence |
|---|---|---|
| Detector + geometry | SAM3, frozen production config (woman/man/child prompts, conf 0.4, NMS 0.7) | Best crowd recall we've measured: 78.2% vs Flash-Lite's 49.2% on the same 100 images (EXP-2026-09); heavy occlusion 62.5% vs 21.4%. Its levers are exhausted: "person" prompt tested and rejected (EXP-2026-04), conf gating measured and rejected as the filter (EXP-2026-07), best-setup arms confirmed (EXP-2026-09 Part 2). |
| Gate + labels + box refinement | Gemini 3.5 Flash-Lite (Batch API) | Gate validated: TP-keep 98.9%, junk removal essentially complete, $0.54/1k detections (EXP-2026-08). Near-perfect statue/doll rejection incl. the teddy bear that fooled 5 other models (EXP-2026-06). Tightest boxes measured: matched-IoU median 0.869 in crowds vs SAM3's 0.81 (EXP-2026-09), 0.914 on LAGENDA. Cheapest model tested by 3–30× at gate-style calls. |

Why NOT the other arrangements, measured:
- **Not Lite as detector** — it misses half the crowd (49.2%) and is nearly blind to heavy
  occlusion (21.4%); a whole-image Lite pass must never override SAM3's detections.
- **Not running both as detectors** — union recall is +1.6 pts over SAM3 alone for ~2× cost
  (EXP-2026-09).
- **Not SAM3's own confidence as the gate** — its FPs score the same as its TPs; statue FPs
  score *higher* than real people (EXP-2026-07).

## The stages

### Stage 1 — Detection (SAM3, frozen production)

Run the unmodified production `autolabel_sam.py` (prompts woman/man/child, conf 0.4,
NMS IoU 0.7) over the image set. Output per detection: **segment polygon (the mask), derived
box, provisional class ∈ {Woman, Man, Child}**.

- The config stays frozen — EXP-2026-04/07/09 closed every tuning lever. Any future change
  re-opens validation.
- Optional cost knob, **off by default**: the EXP-2026-07 conf pre-filter (~0.55–0.6 floor)
  kills ~half of ordinary FPs but costs 10–14% of real people. Since misses are unrecoverable
  and full gating is cheap, only turn this on under hard budget pressure.
- Known ceiling, accepted for v1: ~22% of people in dense crowds are never detected
  (EXP-2026-09). If crowd-heavy data becomes a priority, the v2 detector bake-off
  (dual-vote doc, Gate 1) is the answer — not tuning SAM3.

### Stage 2 — Crop construction with MASK HIGHLIGHTING

For each detection, build the arbitration image:

1. **Crop the detection box padded by 25%** (the EXP-2026-08 gate geometry — validated).
2. **Highlight the person's own SAM3 mask on the crop** — a semi-transparent color tint over
   the mask region plus a solid outline of the mask boundary. Nothing is removed or blacked
   out.
3. The prompt refers to "the highlighted person."

Why highlighting beats the alternatives we considered:

- **vs plain crop:** a padded crowd crop contains 2–4 people; "label this person" is ambiguous.
  We hit exactly this in EXP-2026-09's human adjudication gallery — unusable until the target
  was marked on the crop. What fixed it for the human is what fixes it for the VLM.
- **vs mask-out (delete background):** cutting context removes the clothing/body cues gender
  and age depend on, and a wrong mask *pre-bakes* SAM3's error into what the judge sees —
  hiding the very mistakes we want caught. Highlighting keeps all pixels visible: if the tint
  covers a lamppost or half a neighbor, Lite can still see that and say so.
- **vs point marker:** a point in an overlap is ambiguous (front person or back person?) and
  carries no extent, so it can't support "return a corrected box for this claim."
- **Cost:** none. Same crop, same pixel count, same input tokens — highlighting changes pixel
  values, not billing.

Rendering parameters to freeze after the validation smoke test (Gate V2 below): tint color and
alpha (start ~35% alpha; color must not plausibly recolor clothing — validation checks for
label drift), outline width ~2–3 px, drawn at full crop resolution.

**Mask-source requirement for the production runner (added 2026-07-23):** occluded people's
masks have disconnected parts, but the YOLO label format flattens them into ONE polygon with
stitching bridges. Experiments on frozen labels must reverse-engineer the parts from that
polygon (rasterize → morphological opening → per-part contours — a scale-dependent heuristic
whose worst case is a clumsy outline, with `highlight_quality` as the self-report). **The
production runner must not inherit this heuristic:** it runs SAM3 fresh, so it must render
highlights directly from the in-memory mask components (or persist per-part polygons in a
sidecar next to the YOLO labels), flattening to single-polygon YOLO only for the training
output. That makes part-accurate outlines guaranteed instead of recovered.

### Stage 3 — Arbitration (one Flash-Lite call per detection, Batch API)

One frozen prompt, one JSON verdict per crop, about the highlighted person only:

```json
{
  "verdict":       "real_person | depiction | not_person",
  "gender":        "man | woman | unknown",
  "age_group":     "child | adult | unknown",   // child = appears 12 or younger
  "estimated_age": <number>,
  "box_correction": [x1, y1, x2, y2] | "ok",     // crop pixel coords, the highlighted person only
  "highlight_quality": "good | covers_wrong_object | covers_multiple_people",
  "confidence":    "high | low"
}
```

Prompt carries the settled rulings (same as EXP-2026-06/08): a **printed/photographic depiction
of a real person counts as a person** (posters are gaze-lowering targets); dolls, mannequins,
statues, cartoons are not; **men in thobe/ghutra are men — judge by the face, not the robe**;
gender `unknown` only when no cue is visible at all.

Operational notes (from measured behavior):
- **Batch API, not sequential** — sequential is ~1 crop/s (EXP-2026-08); Batch is ~half price.
- **Retry wrapper required** before production (the EXP-2026-08 cleanup item; Flash's QC arm
  was lost to a 503 storm once).
- **Malformed-JSON tolerance:** ~4% of Flash-Lite responses in EXP-2026-09 drifted to
  per-coordinate box keys with typos; the parser must repair-or-flag, never silently drop
  (flagged crops → retry once, then route to audit).
- `highlight_quality` is the escape hatch for bad masks: `covers_wrong_object` means SAM3's
  mask missed the person entirely — treat as existence-suspect, route to audit.

### Stage 4 — Merge rules (assembling the final label)

Per detection, in order:

1. **Existence:** `not_person` → **delete the detection**. (The validated gate: 98.9% TP-keep,
   EXP-2026-08.) `depiction` → keep (posters count).
2. **Gender:**
   - Lite commits (man/woman) and agrees with SAM3 → done.
   - Lite commits and *disagrees* → **interim rule: Lite wins** (VLM gender is ~99% accurate
     when committed, EXP-2026-01; SAM3's gender-only error ~1.2%, EXP-2026-02 — but on
     *conflicts specifically* the winner must be measured). **This rule is provisional until
     the EXP-2026-09 Phase 5 adjudication tallies land** (Gate V1 below); if adjudication says
     otherwise, flip it.
   - Lite says `unknown` → SAM3's class stands, detection tagged `low-confidence-gender`
     (droppable from training by config; the dual-vote doc's "drop rather than guess" stance).
3. **Age — the teen policy (unchanged from the team-reviewed proposal):** no model is trusted
   in the 10–19 band (six models measured failing it the same way). Final **Child** label only
   when Lite says child AND `estimated_age` < 10. Anything reading 10–19 by either voter →
   **adult by rule** (over-blurring a borderline 12-year-old is acceptable; a teen adult
   escaping the blur is not). This also neutralizes SAM3's dominant error direction — 93% of
   its class errors are adult→Child (EXP-2026-02), and EXP-2026-09 found 24 SAM3-Child vs
   Lite-adult conflicts on 1,133 shared people.
4. **Geometry:** SAM3's mask/box is the default. Lite's `box_correction` **replaces the box
   only if all sanity checks pass**: box within crop bounds; IoU with SAM3's box ≥ 0.3
   (identity-swap guard); area change bounded (≤ 3× grow/shrink). Evidence for letting Lite
   refine at all: its boxes are tighter on the same people (0.845 vs 0.824 mean matched-IoU,
   median 0.869 vs 0.81), while SAM3 carries a loose-box tail (recall drops 17.7 pts from
   IoU 0.5→0.7; 1 in 3 boxes under 0.75) — EXP-2026-09.
   - **Open integration decision (blocker for the box-refinement part only):** training labels
     today are segment *polygons*. A corrected box can't patch a polygon. Options: (a) v1
     fixes class only and keeps SAM3 geometry; (b) clip the polygon to the corrected box;
     (c) write box labels where corrected. Requires first verifying byte-for-byte what training
     consumes (the still-open EXP-2026-02 closeout item). **Recommendation: ship v1 as (a) —
     class + existence correction only — and add box refinement as v1.1** once the
     training-format question is settled.

### Stage 5 — Audit & bias certification (unchanged from the team-reviewed proposal)

- **2–5% random audit** of accepted labels through the same pipeline with human review — a
  thermometer on the "accepted" tier, not a label source.
- **The Gulf/traditional-dress slice** (~100–200 hand-checked crops) certifies both models on
  the bias that motivated this whole effort — still unbuilt, still the single biggest untested
  gap (flagged in every experiment since EXP-2026-01). No model-vs-model agreement can detect
  a shared bias; only this slice can.
- Route the small scraped bias-mitigation dirs (e.g. `dataset_arab_traditional`) through the
  pipeline **first** — they're the original deploy target and the highest-value cleanup.

## Cost (token-profile estimate; verify before quoting)

Measured base: the EXP-2026-08 gate call ≈ $0.54/1k detections (verdict-only, ~40 output
tokens). This pipeline's richer verdict (gender/age/box/quality fields) ≈ 70–100 output
tokens → **≈ $0.7–1.0 per 1k detections standard, roughly half on Batch.** A 500k-image
mostly-single-person corpus (~1.25M detections) ≈ **$900–1,250 standard / ~$450–650 Batch**.
Mask highlighting adds nothing (same pixels). Caveats: Flash-Lite rates are still
press-reported ($0.30/$2.50 — the pricing-screenshot verification is an open item), and these
are projections from measured token counts, not a measured bill.

## What we are explicitly NOT doing in v1, and why

- **No whole-image VLM detection pass** — 49.2% crowd recall would silently delete people.
- **No point-prompt disambiguation** — ambiguous under overlap, extent-less (can't anchor a
  box correction).
- **No background mask-out** — destroys context and hides mask errors from the judge.
- **No second detector / MiVOLO yet** — that's the dual-vote v2; its gates (detector bake-off,
  EXP-2026-05 MiVOLO teen curve) stay on the roadmap and this v1 doesn't foreclose them: the
  merge layer already speaks (box, class, verdict), so a third voter slots in.
- **No trust in any model for ages 10–19** — policy handles it.

## Validation gates before production (all pre-registered, all cheap)

| Gate | What it decides | Status |
|---|---|---|
| **V1 — EXP-2026-09 Phase 5 adjudication** | The gender-conflict rule in Stage 4.2 (does Lite actually beat SAM3 on conflicts?) and confirms the Child↔adult conflict direction | Gallery built, tallies pending (the one open human step) |
| **V2 — highlight validation (EXP-2026-10 candidate)** | (a) *No-bias check:* highlighted vs plain crops on unambiguous single-person cases — label flip rate must be ~0; (b) *right-person rate* on overlapping pairs (model returns the highlighted person's box; scored vs the intended GT box) ≥ 95%; (c) *box-refinement quality:* corrected boxes beat SAM3's IoU on the loose-box tail without identity swaps | Not designed yet; reuses exp09 `per_person.jsonl` + gate_eval machinery; < $1, no GPU |
| **V3 — Gulf-dress slice** | Certifies gender on the target bias for both models | Unbuilt (Track 2) |
| **V4 — engineering hardening** | Retry wrapper, Batch integration, malformed-JSON repair, pricing screenshot | Partially open (EXP-2026-08 cleanup list) |

Build order: V2 tool + run (it's the only new science), land V1 tallies, then build the
pipeline runner against the frozen prompt — V3 can certify in parallel before the first
production-scale run.
