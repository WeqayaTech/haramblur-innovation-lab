# Pipeline v1 — team brief: pilot findings, accuracy, cost (confirm before full run)

**The pipeline in one sentence:** SAM3 detects every person and provides masks/boxes; each
detection is sent to Gemini Flash-Lite as a padded crop with that person's mask outlined; Lite
confirms it's a real person, corrects the gender/age label, and can tighten the box; anything
aged over 12 defaults to adult, and rejected detections are deleted.

**Status:** 25-image pilot complete, one blocker found AND fixed (details below). Asking for
team sign-off on the fixed config before the full 500-images-per-dataset run.

---

## 1. What each stage takes in and puts out (concrete example)

| Stage | Input | Output | Example |
|---|---|---|---|
| **1. SAM3 detect** (frozen production: prompts woman/man/child, conf 0.4, NMS 0.7 — labels already on disk from EXP-2026-02/03, no GPU needed) | Raw image, e.g. a 1360×800 crowd photo | One YOLO line per person: class + mask polygon (normalized coords) | `1 0.214 0.337 0.219 0.301 ...` = "Man" + his mask outline; the polygon's extent is the box |
| **2. Crop builder** | Image + one detection's polygon/box | A 25%-padded crop around the person; if smaller than 320px it's upscaled first; the person's mask drawn as a **green outline** (nothing covered) | A 60×140px distant person becomes a ~340px crop with a green outline hugging his body |
| **3. Flash-Lite verdict** (one API call per detection) | The crop image + a fixed prompt ("judge ONLY the marked person"; posters count as people, dolls/statues don't; judge by face not clothing) | One JSON verdict | `{"verdict":"real_person","gender":"man","age_group":"adult","estimated_age":34,"box_correction":"ok","highlight_quality":"good","confidence":"high"}` |
| **4. Merge rules** | SAM3's class + Lite's verdict | The final decision per detection | `not_person` → **detection deleted** · Child only if Lite says child AND age ≤ 12 (>12 → adult) · Lite's committed gender wins · box corrections accepted only behind sanity checks |
| **5. Final labels** | All merged decisions | Clean training label file per image (+ 2–5% random human audit) | The statue SAM3 called "Woman" is gone; the man SAM3 called "Child" is now "Man" |

Visual walkthrough of real examples: **`trace.html`** (4 images, every stage's exact
input/output including the raw Gemini responses). Attach it with this brief.

## 2. Pilot findings (25 images per dataset; PASS 500 — its detections are rare)

**What worked immediately:**
- **LAGENDA (people-focused photos): gate kept 26/26 real people; gender 18/18 (100%) on
  committed adults.**
- **PASS (verified-empty scenes): killed 40/52 of SAM3's hallucinated detections (77%)** —
  each kill is a poisoned training label prevented. The 12 "survivors" mostly contain real
  tiny humans the benchmark missed (same pattern as EXP-2026-08).
- Zero API errors, zero parse failures across 738 calls.

**The blocker we caught (why pilots exist):** in crowd scenes the gate deleted **76 of 388
verified real people (kept only 80%, bar is 97%)**. Visual review showed why: the original
highlight was a semi-transparent green **tint**, and on small/distant people it covered the
few pixels that made them recognizable — the marker was hiding what it marked.

**The fix, chosen by measurement (re-ran exactly the 76 failed cases):**

| Variant | Recovered | Notes |
|---|---|---|
| Outline instead of tint | 45/76 (59%) | person fully visible, still not enough pixels |
| **Outline + upscale small crops to 320px** | **68/76 (89%)** | **adopted** — projects 97.9% total keep |

The upscale is free: Gemini bills small images the same, so more pixels per person costs
nothing extra.

**Also learned:**
- The 3 wrong labels on LAGENDA were all **Lite reading teens as ≤12** (the known industry-wide
  teen weakness — every model we've ever tested has it). The stored ages let us tighten the
  child cutoff policy without re-running anything.
- Lite **never proposed a box correction** (0 of 738) — our prompt made "ok" too easy. v1
  therefore ships as **class + existence correction only**; box refinement is a v1.1 question.

## 3. Accuracy summary (pilot scale — directional; bars are judged at the full 500 run)

| Metric | Bar | Pilot result |
|---|---|---|
| Keep real people (LAGENDA) | ≥ 97% | **100%** (26/26) |
| Keep real people (crowds) | ≥ 97% | 80% with tint → **projected 97.9% with adopted fix** (confirmation run next) |
| Kill hallucinations (empty scenes) | ≥ 65% | **77%** (40/52) |
| Kill hallucinations (crowds) | ≥ 60% | 63% (5/8, tiny sample) |
| Final label accuracy vs human GT | ≥ 93% | 88.5% (23/26 — the 3 misses are the teen cases above) |
| Gender on committed adults | ≥ 99% | **100%** (18/18) |

For context, raw SAM3 labels today: ~88.7% label accuracy with **~10% of adults leaking to
"Child"** (the error that lets an adult escape the blur) and a measured 62% false-person rate
on human-shaped objects. The pipeline's whole job is deleting those FPs and closing that leak.

## 4. Cost (measured tokens; $ at press-reported Lite rates $0.30/$2.50 per 1M — screenshot
verification still pending)

Measured per detection under the adopted config: ~1,475 input + ~66 output tokens →
**$0.61 per 1,000 detections** sequential, **~$0.30/1k on Batch API**.

**The full 500-images/dataset benchmark run (projected from measured pilot rates):**

| Dataset (500 imgs each) | Detections | Runtime (seq.) | Expected vs bar |
|---|---|---|---|
| LAGENDA | ~3,240 (6.5/img) | ~55 min | gate 100%* · 3-class 88.5%* vs ≥93 bar (misses = teens) · gender 100%* |
| CrowdHuman | ~10,480 (21/img) | ~2.9 h | TP-keep **projected 97.9%** vs 97 bar (pilot-v2 confirms) |
| PASS | ~52 (0.1/img) | ~1 min | FP-kill 76.9%* vs ≥65 bar |
| **Total** | **~13,800** | **~3.9 h** | *= measured at pilot scale |

**Spend ledger:**

| Item | Calls | Cost (est.) |
|---|---|---|
| Pilot v1 + variant recovery + traces (done) | 945 | **$0.57** |
| Pilot v2 confirmation (pending) | ~660 | $0.42 |
| Full 500×3 run (pending; reuses pilot verdicts) | ~13,000 | $7.90 |
| **Whole experiment end to end** | ~14,600 | **≈ $9** |

**Production, 500,000 images (cost scales with DETECTIONS, not images — audit the corpus's
persons-per-image before budgeting):**

| Corpus profile | Detections | Sequential | Batch API |
|---|---|---|---|
| Ordinary photos (~2.5 persons/img) | 1.25M | $760 | **~$380** |
| Multi-person, LAGENDA-like (~6.5/img) | 3.25M | $1,975 | **~$990** |
| Crowd-heavy (~21/img) | 10.5M | $6,380 | **~$3,190** |

For contrast (from the original proposal): VLM-label everything ≈ $4,200; human annotation
≈ $175,000 per 500k images.

## 5. What we're asking the team to confirm

1. **The adopted crop config** (green mask outline + upscale-to-320) — evidence:
   `failed_recovery_report.html` shows all 76 previously-failed people with before/after
   verdicts.
2. **The merge policy:** delete rejected detections; Child only when Lite says child AND
   age ≤ 12; Lite's committed gender overrides SAM3; person kept with unknown gender is
   tagged and droppable from training.
3. **v1 scope = existence + class correction only** (no box rewriting yet).
4. **Go/no-go for the full run** after the pilot-v2 confirmation shows crowds ≥ 97% keep
   (~$1 to confirm, then ~$10 for the full three-dataset run).

**Attachments to include:** `trace.html` (stage-by-stage walkthrough) ·
`pilotC_report.html` (scores + evidence crops, after the confirmation run) ·
`failed_recovery_report.html` (the blocker fix, before/after).

**Not covered by this pilot (known, tracked):** the Gulf/traditional-dress gender bias — none
of these datasets contains it; it gets its own hand-checked slice (the long-standing Track 2
item) before production rollout.
