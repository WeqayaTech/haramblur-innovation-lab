# Spotlight pipeline — results (EXP-2026-10, run 2026-07-26)

> Paste into a ClickUp doc; `|` tables convert automatically. Drag the four PNGs from
> `experiments/assets/png/EXP-2026-10/` into the marked spots.

**Pipeline:** SAM3 detects every person and provides masks → each detection becomes a padded
crop with that person's mask outlined → Gemini 3.5 Flash-Lite confirms it's a real person and
corrects the gender/age label → rejected detections are deleted, anything over 12 defaults to
adult. Every raw reading is logged.

**Run:** 5,236 detections · CrowdHuman 150 images · LAGENDA 150 images · PASS 500 images ·
7 parse failures (0.13%) · 0 API errors · total spend **$5.50**.

---

## 1. Headline — pipeline vs today's SAM3 labels

[DRAG IN: exp10_vs_sam3.png]

| What we measured | SAM3 labels today | Spotlight v1 |
| --- | --- | --- |
| Label accuracy (Woman/Man/Child) | 88.7% | **99.3%** |
| Adults NOT leaking into "Child" | ~89.5% | **100%** (0 of 95 aged 18+) |
| Gender correct on adults | ~98.8% | **99.05%** |
| Genuine junk removed (statues, hallucinations) | none | **~92%** |
| Real people kept | — | **98.5–99.3%** |

SAM3 figures come from EXP-2026-02 on a different sample; a like-for-like re-measurement is
still owed.

## 2. The age result — nobody over 17 is labeled a child

[DRAG IN: exp10_age_gradient.png]

| Ground-truth age | People | Labeled "Child" |
| --- | --- | --- |
| 0–12 | 32 | **100%** — every real child caught |
| 13–15 | 17 | 76.5% |
| 16–17 | 7 | 14.3% |
| 18–19 | 5 | **0%** |
| 20–29 | 27 | **0%** |
| 30+ | 63 | **0%** |

The only disagreements are 13–17-year-olds. LAGENDA's ages are human *apparent-age* guesses
(±2–3 years), so a model saying 12 where an annotator said 13 is annotation noise, not an adult
escaping the blur. **No person aged 18 or over was ever labeled a child.**

## 3. Per dataset — each answers a different question

[DRAG IN: exp10_per_dataset.png]

| Dataset | Question it answers | Result |
| --- | --- | --- |
| CrowdHuman (150 imgs, 3,853 dets) | Does the gate keep real people in crowds? | **98.48%** kept (2,534 / 2,573) |
| LAGENDA (150 imgs, 1,331 dets) | Are the labels right vs human ground truth? | **99.34%** kept · **99.3%** label accuracy · **99.05%** gender |
| PASS (500 imgs, 52 dets) | Does it delete hallucinations on empty scenes? | **~92%** of genuine junk removed |
| Object set (dolls/statues) | Does it reject human-shaped objects? | running — lean prompt scored 74.5% |

Two raw metrics were investigated and set aside, with evidence: CrowdHuman's "false positive"
count and PASS's raw kill rate both penalised the pipeline for keeping **real people the
benchmarks never annotated** (occluded figures, people in posters and mirrors). Owner reviewed
every survivor image individually before either metric was reinterpreted.

## 4. Running the full 500,000 images — time and money

[DRAG IN: exp10_cost.png]

Everything below scales from **measured** throughput: SAM3 labels ~1.3 s/image on an L4;
each Gemini call takes ~1.3 s and costs **~$1.00 per 1,000 people labeled** (1,838 input + 178
output tokens each). **Cost and time scale with PEOPLE, not images** — so the only number that
really matters is how crowded the corpus is.

### Money — Gemini API only

| 500,000 images | People to label | Standard API | **Batch API** |
| --- | --- | --- | --- |
| Ordinary photos (~2.5 people/img) | 1.25M | $1,250 | **$625** |
| Multi-person (~6.5/img) | 3.25M | $3,250 | **$1,625** |
| Crowd-heavy (~26/img) | 12.9M | $12,850 | **$6,425** |

Human annotation of the same 500k images ≈ **$175,000**. This whole validation experiment cost
**$5.50**. (SAM3 labeling runs on GPU pods we already pay for — it is not part of these figures.)

### Time

| Stage | Ordinary | Multi-person | Crowd-heavy |
| --- | --- | --- | --- |
| SAM3 labeling — 1 GPU pod | 7.5 days | 7.5 days | 7.5 days |
| SAM3 labeling — 4 pods in parallel | **~2 days** | ~2 days | ~2 days |
| Gemini — one process at a time | 19 days | 49 days | 193 days ❌ |
| Gemini — 10 parallel processes | ~2 days | ~5 days | ~19 days |
| Gemini — **Batch API** | **~1 day** | **~1–2 days** | **~2–3 days** |

**Realistic plan: ~2–4 days end to end, ~$625 of API spend** for an ordinary-photo corpus —
SAM3 labeling across a few pods, crops sent through the Batch API. Sequential API calls are not
viable at this scale (19 to 193 days depending on crowding), so **Batch API integration is the
one piece of engineering required before a production run.** It also halves the bill.

### Caveats on these numbers

- Dollar figures use press-reported Flash-Lite rates ($0.30 / $2.50 per 1M tokens); **token
  counts are measured, the rates still need vendor confirmation.**
- Batch turnaround is Google's stated target, not something we have measured.
- Parallel-process throughput assumes account rate limits allow it — unverified.

## 5. What we also learned

- **Box refinement is not worth shipping.** Gemini proposed tighter boxes on 10% of detections,
  but they regress the average fit (0.752 vs SAM3's 0.787). SAM3's geometry always wins; the
  suggestions are still logged for later analysis.
- **A production bug was found**: the auto-labeler's duplicate-removal step compares full-size
  masks pairwise on one CPU core — it stalled outright on a dense crowd image. Fixed in our copy
  with a provably identical shortcut; **the same bug is in the production script.**
- **Analysis fields work.** Every detection now carries apparent race, garment, head covering,
  exposed body parts, orientation, pose, occlusion, skin tone, and more — enough to build the
  adversarial slices (e.g. thobe vs abaya, back-facing people) the team asked for.

## 6. Still open before production

1. **The Gulf/traditional-dress slice** — the bias that motivated this whole effort remains
   untested; no public benchmark covers it.
2. Object-set false-positive number (running).
3. A ~100-label human audit of accepted labels — catches biases both models might share.
4. Batch API integration (halves the cost) and vendor-confirmed pricing.
