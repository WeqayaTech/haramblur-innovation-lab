# VLM auto-labeling — accuracy & cost, five models (EXP-2026-06)

**Source:** EXP-2026-06 smoke test, 2026-07-16/22 — 7 models scored against human ground truth on 4 datasets (Gemini 3.6 Flash and 3.5 Flash-Lite added the week they released). Token counts measured from vendor API responses; $/token rates hand-verified 2026-07-17 for the original five; **3.6 and Flash-Lite rates are press-reported, verification pending.**

> ⚠️ **n = 3 images per dataset per model.** Smoke-test scale: these numbers pick which models advance to the ~250-image run — they are not final accuracy verdicts.

## Bottom line

- **Eliminating SAM3's false positives: works.** SAM3 wrongly labels a person in ~62% of doll/statue/toy images; the five VLMs made 4 such errors in 30 chances — 3 of them on the same teddy bear. Claude Sonnet 5: zero.
- **VLM as sole labeler: no.** Every VLM trails SAM3's 77% crowd recall (best 59.8%) — but Gemini's 96–98% box precision beats SAM3's 86.6%, and false labels poison training while misses only shrink it.
- **Recommended architecture: SAM3 detects → VLM verifies & classifies.**
- **New (2026-07-22): Gemini 3.5 Flash-Lite is the presumptive verifier-gate model** — perfect
  negatives at smoke scale (including the teddy bear that fooled 4 of 7 models), perfect
  single-person classification, no thinking overhead, and ~5x cheaper than everything
  (~$1.30/1k mostly-single-person images; a crop-verdict gate call ≈ $0.0005). Pending:
  pricing verification + the scale-up.

## Accuracy by dataset (7 models)

| Metric | GPT-5.6 Sol | Gemini 3.5 Flash | Gemini 3.1 Pro | Claude Sonnet 5 | Qwen3.7-Plus | Gemini 3.6 Flash | **Gemini 3.5 Flash-Lite** |
|--------|------------|------------------|----------------|-----------------|--------------|------------------|---------------------------|
| LAGENDA found · gender · age | 3/3 · 3/3 · 3/3 | 3/3 · 3/3 · 3/3 | 3/3 · 3/3 · 3/3 | 2/3¹ · 2/2 · 1 teen→child² | 3/3 · 3/3 · 3/3 | 2/3 · 2/2 · 1 teen→child | **3/3 · 3/3 · 3/3** |
| Crowd recall @IoU 0.5 | **59.8%** | 55.2% | 50.6% | 31.0% | 56.3% | 41.4% | 42.5% |
| Crowd precision | 49.5% | **98.0%** | 95.7% | 36.0% | 60.5% | 81.8% | 61.7% |
| Crowd AP50 / AP@[.50:.95] | 0.456 / 0.205 | **0.545** / 0.335 | 0.503 / **0.352** | 0.167 / 0.044 | 0.468 / 0.210 | 0.414 / 0.237 | 0.269 / 0.137 |
| Box tightness (matched IoU, crowd) | 0.750 | 0.818 | **0.849** | 0.657 | 0.752 | 0.801 | 0.797 |
| PASS false persons (empty scenes) | 1 (low-conf) | **0** | **0** | **0** | **0** | **0** | **0** |
| Object-set false persons (dolls/statues) | **0** | 1 (teddy) | 1 (teddy) | **0** | 1 (teddy) | 1 (teddy) | **0 — rejected the teddy** |

¹ Claude detected and correctly classified this person — its box just fell a hair under the IoU 0.5 matching bar (0.465). Box looseness, not blindness.
² A 17-year-old read as child — the teen band, the same failure zone SAM3 and Qwen2.5-VL-7B show. Gemini 3.6 made the identical error.
³ Flash-Lite tightness detail: crowd matched-IoU median 0.835, 67.6% at COCO-strict >=0.75; **LAGENDA (single-person) matched-IoU mean 0.914 on 3/3 matches — among the tightest single-person boxes measured (3.6 Flash: 0.924 but on only 2/3).** Its low crowd AP (0.269) reflects its 61.7% precision, not box quality — consistent with the gate-not-detector role.

**Teddy-bear scoreboard** (the one object that fools models): rejected by Sol, Claude, **Flash-Lite** · fooled Flash, Pro, 3.6, Qwen.

**Reference — production SAM3 on the same benchmarks:** crowd recall 77% / precision 86.6%; object-set false-person rate 62.2% of images (upper bound, re-verification pending).

**How to read the profiles:** Geminis = tight boxes, conservative (their misses are truly undetected); Sol & Qwen = see the most people, box them loosely; Claude = sees plenty (56% recall at relaxed IoU 0.3) but has the loosest boxes.

## Estimated price per 1,000 images, by image type (7 models)

| Image type | **Flash-Lite 3.5**† | Qwen3.7-Plus | Gemini 3.5 Flash | Gemini 3.1 Pro | Claude Sonnet 5 | Gemini 3.6 Flash† | GPT-5.6 Sol |
|------------|---------------------|--------------|------------------|----------------|-----------------|-------------------|-------------|
| Free of people (empty) | **~$0.50** | ~$1 | ~$2.50* | ~$3.20* | ~$1.70 | ~$5.20 | ~$11 |
| Couple of people (1–5) | **~$1.15** | ~$9 | ~$4.70* | ~$5.80* | ~$7 | ~$12.80 | ~$35 |
| Crowd (20–70 people) | **~$3.50** | ~$10.50 | ~$24.60 | ~$13* | ~$20 | ~$28.70 | ~$94 |
| **Mixed, mostly single person** (10% empty / 80% sparse / 10% crowd) | **~$1.30** | ~$8.10 | ~$6.40* | ~$6.30* | ~$7.90 | ~$13.70 | ~$38.40 |

† Flash-Lite ($0.30/$2.50) and 3.6 Flash ($1.50/$7.50) rates are press-reported, pending pricing-page verification. Flash-Lite ships with **thinking OFF by default** (~10 output tokens on empty scenes) — the rate advantage and the token advantage stack, which is why it is ~5x cheaper than everything else. Its quality on exactly this image type: perfect at smoke scale (see accuracy table).

## Estimated price per 1,000 PERSONS labeled (different unit!)

If your budget unit is labeled person instances (training data), not images processed. Crowds
are the most expensive images but the cheapest persons: each image has a fixed cost (the image
itself as input tokens) that spreads over ~35 people in a crowd vs ~5 in a sparse image, and
each extra person only adds ~60–70 output tokens. Example: 1,000 persons = ~28 crowd images
(Sol: ~$2.70) or ~190 sparse images (Sol: ~$6.50).

| Source images | **Flash-Lite 3.5**† | Qwen3.7-Plus | Gemini 3.5 Flash | Gemini 3.1 Pro | Claude Sonnet 5 | GPT-5.6 Sol |
|---------------|---------------------|--------------|------------------|----------------|-----------------|-------------|
| Sparse (1–5 people/img) | **~$0.29** | ~$2.60 | ~$1.30* | ~$1.60* | ~$1.50 | ~$6.50 |
| Crowds (20–70 people/img) | **~$0.18** | ~$0.40 | ~$1.50 | ~$0.90* | ~$0.80 | ~$2.70 |

Verified rates ($/1M input / output): Sol 5/30 · Flash 1.50/9 · Pro 2/12 · Sonnet 2/10 · Qwen 0.40/1.60 list.

## Pricing notes

- **\* = lower bound.** Google bills thinking tokens as output; our first-pass accounting missed them (~3.5× more billed output on the corrected re-run). Starred Gemini figures rise ~2–3× on people-heavy images; Flash's crowd figure is already corrected.
- **Two price time bombs:** Claude Sonnet 5 is intro-priced — **+50% on Sep 1, 2026** ($3/$15). Qwen's 20% promo (~$0.32/$1.28 actual) will lapse.
- **Cost drivers:** fixed per-image cost (prompt + image input tokens, scales with resolution) + ~60–70 output tokens per person found. Crowds cost more per image, less per person.
- **100k mixed images ballpark:** Qwen $200–900 · Flash $300–900 · Pro $400–1,100 · Sonnet $500–1,500 · Sol $2,000–8,000. Batch APIs (~50% off, all vendors) can halve these at scale.

## Verifier-gate cost per 1,000 crops (NEW — measured in EXP-2026-08)

The gate call pattern (SAM3 detection crop in → keep/reject verdict out) is the pipeline's
actual production use of a VLM. Flash-Lite's figure is **measured** (1,423 real crops); others
are projected from their measured token behavior at the same workload.

| Model | $/1k gated crops | Gate a 500k-image corpus (~1.25M detections) | vs Lite |
|-------|------------------|----------------------------------------------|---------|
| **Gemini 3.5 Flash-Lite** | **$0.54 (measured)** | **~$675 standard / ~$340 Batch API** | — |
| Qwen3.7-Plus | ~$1.30–2.90 | ~$1,600–3,600 | 3–5× |
| Claude Sonnet 5 | ~$3.65 | ~$4,600 | ~7× |
| Gemini 3.5 Flash | ~$3.50–5.80 | ~$4,400–7,200 | 7–10× |
| Gemini 3.6 Flash | ~$4.00–5.90 | ~$5,000–7,400 | 8–10× |
| Gemini 3.1 Pro | ~$5.30–7.70 | ~$6,600–9,600 | 10–14× |
| GPT-5.6 Sol | ~$15–20 | ~$19,000–25,000 | ~30× |

Why Lite wins structurally: a gate verdict is ~40 output tokens — thinking models attach
hundreds of reasoning tokens to that tiny answer, so their cost is deliberation the task
doesn't need. And Lite's gate *quality* is validated (EXP-2026-08): **98.9% of verified real
people kept; essentially all false labels removed** (its "failures" turned out to be the
benchmark's contamination — real people the gate correctly recognized). The whole validation
experiment cost $0.76.

## Next steps

1. ~~Gate experiment~~ **DONE (EXP-2026-08):** Flash-Lite validated for the gate seat — 98.9%
   real-people keep rate, near-total false-label removal, $0.54/1k crops. Pipeline is now:
   SAM3 detects → Lite gates every detection → teen-band default-to-adult policy → 2–5% audit.
2. **Pipeline build** with the validated components; gate the current 500k production corpus
   (~$340–675 via Batch API).
3. **Scale-up of the labeler comparison** (~250 images/dataset, ~$50–150) if a whole-image
   VLM labeler is ever needed — currently optional, since SAM3+gate covers the design.
4. Cleanup: Flash QC rerun (200 crops), formal survivor tallies, pricing screenshots for
   Flash-Lite and 3.6 (rates still press-reported).

## The exact prompt used (frozen, identical for every model)

`{width}` / `{height}` are replaced with the image's pixel dimensions; nothing else changes
between models or datasets. Lives in `vlm-cluster/vlm_detect_eval.py` (`DETECT_PROMPT`).

```
You are annotating images for a person-detection dataset. Find EVERY person
in this image and return a JSON object, nothing else.

What counts as a person:
- Every real human being, of any age, including partially visible or
  occluded people (even if only a head, torso, or limb is visible).
- A printed or photographic depiction of a real human — a poster,
  billboard, magazine cover, or photo within the photo — COUNTS as a person.
- A doll, mannequin, statue, sculpture, toy, cartoon, or drawing is NOT a
  person. Do not include them.
- If the image contains no persons, return an empty list.

For each person report:
- "box_2d": the bounding box as [x1, y1, x2, y2] in pixels, where (x1, y1)
  is the top-left corner and (x2, y2) the bottom-right. The image is
  {width} pixels wide and {height} pixels tall.
- "gender": "man" or "woman" as your best estimate from visible cues;
  "unknown" ONLY if no gender cue is visible at all (fully covered figure,
  too small, back turned with no other cues).
- "age_group": "child" if the person appears 12 years old or younger,
  otherwise "adult". Use "unknown" ONLY if you truly cannot tell.
- "estimated_age": your single best numeric age estimate (a number, not a
  range). Always give a number even when age_group is "unknown".
- "confidence": "high" if you are sure this is a real person (or printed
  photo of one), "low" if it might be a statue/doll/reflection or is barely
  visible.

Return EXACTLY this JSON structure:
{"people": [{"box_2d": [x1, y1, x2, y2], "gender": "...",
"age_group": "...", "estimated_age": N, "confidence": "..."}, ...]}

Return only the JSON. Do not identify or name anyone.
```

**Known model quirks found during the run** (all handled in the harness):
- **Gemini** (Flash & Pro) ignores the requested pixel format and always returns y-first 0–1000 boxes (`[ymin, xmin, ymax, xmax]`).
- **Qwen3.7-Plus** returns x-first 0–1000 boxes, and on one image emitted a JSON block, wrote "Wait, I need to scale the coordinates…", and appended a corrected block — the parser now takes the final answer.
- **OpenAI and Anthropic** honor the prompt's pixel format as asked.
