# Experiment: Can a commercial VLM act as a full auto-labeler — and which one, at what cost?

**In one line:** we give GPT-5.6 Sol (OpenAI), Gemini 3.5 Flash & Gemini 3.1 Pro Preview
(Google), and Claude Sonnet 5 (Anthropic) whole images from all four staged benchmark datasets
(LAGENDA, CrowdHuman, PASS, object set) with one frozen prompt — find every person, box them,
gender + age group each — and score each dataset against whatever ground truth it supports,
tracking token cost throughout.

**Status:** Smoke test RUN COMPLETE 2026-07-16 · 3 images per dataset per model · Owner: Mostafa

> **n=3 per dataset is a plumbing/cost-ballpark smoke test, not an accuracy comparison.** Its
> job is to prove every engine runs end-to-end, returns parseable boxes, and to get a real
> $/1,000-images order of magnitude before spending money on a sample big enough to trust.
> No accuracy verdict — not even a tentative one — comes out of 3 images.

---

## The short version

- **All four engines passed plumbing** — 48/48 images produced scoreable output (after two
  harness fixes the smoke test existed to find: OpenAI's `max_completion_tokens` requirement,
  and the finding below).
- **Gemini ignores the prompt's box format and always emits its native y-first 0-1000
  convention** (`[ymin, xmin, ymax, xmax]`). First-pass scores showed both Geminis at 0% match;
  re-parsing the saved raw responses under `yxyx_1000` (no API re-run) took them to top-tier.
  Harness now handles this (`reparse_boxes.py`; scale-up runs parse Gemini with `yxyx_1000`).
- **On LAGENDA, GPT-5.6 Sol and both Geminis went 3/3 on detection, gender, and age; Claude
  Sonnet 5 went 2/3 found and produced the only adult→child leak** (the error direction that
  lets an adult escape the blur).
- **Opposite detection personalities on crowds:** Sol = highest recall (59.8%) but ~50%
  precision (loose/duplicate boxes); Gemini = conservative but ~96-97% precision. All four
  trail SAM3's 77% crowd recall; Gemini beats SAM3's 86.6% precision. Supports SAM3-detects /
  VLM-verifies hybrid, not VLM-as-sole-detector.
- **Negatives were near-perfect across the board** (vs SAM3's 62% object false-person rate):
  Claude 0 FPs anywhere; Sol 0 on objects, 1 low-conf on PASS; both Geminis' only miss was the
  same teddy bear.
- **Cost (est. rates): Gemini 3.5 Flash ~$1.30/1k persons labeled is ~5x cheaper than Sol
  (~$6.50)** with equal LAGENDA reads; Sol's reasoning tokens make it expensive even on empty
  images (~$11/1k vs ~$2/1k). At n=3 none of the accuracy numbers are verdicts.
- **Qwen3.7-Plus (added 2026-07-17): 3/3 LAGENDA, crowd recall 56.3%/precision 60.5% (#2 on
  both counts among VLMs), but the heaviest token bill (~5-6k output/image of visible
  reasoning) and two harness findings: x-first 0-1000 boxes + mid-response self-correction
  (multi-block JSON). Box-tightness sweep verdict across all five: Gemini = tight-but-blind,
  Sol/Qwen = see-more-box-worse, Claude = sees plenty (56% @IoU0.3) but boxes worst (0.657).
  After Flash's complete crowd rerun (first run lost an image to a 503): Flash recall 55.2%,
  precision 98.0%, best AP50 (0.545); Pro keeps best AP[.5:.95] (0.352). The rerun also
  measured Gemini's REAL output bill (thinking tokens): ~3.5x what the first pass recorded —
  Flash crowds cost ~$24.6/1k images, ~$1.50/1k persons (Qwen $0.40 and Sonnet $0.80 are the
  cheap crowd labelers); remaining Gemini runs are cost lower bounds pending rerun.

## Why we did this

EXP-2026-03/04 established that the production SAM3 labeler needs a person-verifier gate (62%
false-person rate on dolls/statues, 77% crowd recall) and EXP-2026-01/02 established that age is
the unsolved classification axis. A commercial VLM is a candidate for several of these jobs at
once — verifier gate, crowd detector, age/gender classifier — but nobody has measured any of
them on our benchmark datasets, and nobody knows what running one at labeling scale costs. This
experiment sets up that measurement and smoke-tests it.

Unlike EXP-2026-01 (which cropped an already-labeled person and asked the VLM to describe them),
this run gives the model the **whole image** and asks it to do the complete labeling job. That's
deliberately the harder, production-shaped task — the numbers here are NOT comparable to
EXP-2026-01's crop-based 99.3% gender figure, and the doc will not compare them.

## What we wanted to find out

**The question:** used as a whole-image person detector + classifier, how do the current
flagship VLMs — GPT-5.6 Sol, Gemini 3.5 Flash, Gemini 3.1 Pro Preview, Claude Sonnet 5 (IDs
confirmed against official vendor docs 2026-07-16) — behave on each of our four benchmark
datasets, and what does each cost per image?

**Components measured, per the framework (`docs/COMPONENT_FRAMEWORK.md`):** every dataset is
scored separately, never pooled:

| Dataset | Images | GT available | What gets scored |
|---|---|---|---|
| LAGENDA | 3 | 1 person/image, age + gender | C1 recall on the labeled person; C2 age group + C3 gender, scored ONLY on detected people |
| CrowdHuman | 3 | every person boxed (vbox) | C1: recall, precision, per-image counts |
| PASS | 3 | verified person-free | C1 negatives: any detection = hallucination |
| Object set | 3 | verified no real people | C1 negatives: false-person rate on dolls/statues/toys — the person-verifier-gate question |

**Pre-registered expectations (bars in the loose sense — at n=3 these are sanity checks, not
pass/fail gates):**

- **Plumbing:** every engine returns parseable JSON with usable boxes on ≥10 of its 12 images.
  A model that can't produce scoreable boxes at n=3 is disqualified from the scaled-up run.
- **PASS / object set:** the interesting read is directional — does an instructed VLM refuse
  dolls/statues where SAM3 (62.2% of images) failed? Zero false persons on 3+3 images is the
  hoped-for outcome; any false person on PASS at n=3 is a bad sign worth noting.
- **Cost — the production question this experiment exists to answer:** two unit prices per
  model, from each vendor's own usage field × hand-verified rates: **$/1,000 person-free images**
  (the fixed cost of "nothing here" — PASS measures it directly) and **$/1,000 persons labeled**
  (run cost ÷ persons detected; LAGENDA gives the sparse-image price, CrowdHuman the dense-crowd
  price, which is lower per person because the fixed image cost amortizes). Every run's
  `summary.json` carries a `labeling_economics` block with both. No pre-registered dollar bar —
  the deliverable is the ballpark itself.
- **No accuracy bar on LAGENDA/CrowdHuman at this n.** Detection/classification results are
  recorded and eyeballed, not judged.

## How we did it

1. **The answer keys.** All four datasets are already staged and registered
   (`docs/DATASET_REGISTRY.md`): LAGENDA (`/workspace/lagenda_eval/lagenda_yolo/`, human-checked
   age+gender), CrowdHuman seed-51 sample (odgt vbox GT, same visible-box convention as
   EXP-2026-03), PASS 3k (verified empty), object set (259 hand-verified doll/statue/toy images).
2. **One frozen prompt, four engines.** `vlm-cluster/vlm_detect_eval.py` sends the whole image +
   the detection prompt (quoted in full in the appendix — person ruling included: posters count,
   dolls don't, per the EXP-2026-03 owner's ruling) to each engine via the same
   `api_describers.py` classes as the original design. Only the model changes. 3 images per
   dataset, selected by seed-42 shuffle of the sorted file list — the same 3 images for every
   model.
3. **Scoring, per dataset.** PASS/objects: count detections (all + high-confidence-only).
   CrowdHuman: greedy mutual-exclusive IoU matching (`match_boxes`, reused from
   `run_model_children.py`, not reimplemented) against vbox GT at IoU 0.5; detections landing in
   odgt ignore regions don't count as FPs. LAGENDA: match the one GT box, then score gender and
   age group only on matches — Component 2/3 conditioned on Component 1, as always.
4. **Cost.** Token counts from each vendor's own response (`usage` / `usage_metadata`), rates
   from `model_pricing.json` — which ships `null` and must be hand-filled from vendor pricing
   pages on run day (see the caveat below).
5. **Definitions that matter.** "Child" = appears ≤12 (production cutoff, stated numerically in
   the prompt). `estimated_age` (a number) is also collected so implicit-boundary curves can be
   drawn at any cutoff post-hoc when this scales up. "Person" = real human or printed/photographic
   depiction of one; dolls/mannequins/statues/toys are not persons.

**Model-name vs. pricing caveat:** model IDs were confirmed 2026-07-16 by fetching
`developers.openai.com/api/docs/models` and `ai.google.dev/gemini-api/docs/models` directly —
not via search-result aggregators, which showed inconsistent numbers and suspicious tier names.
Preview IDs (`gemini-3.1-pro-preview`) can be renamed by the vendor; if a run 404s, check the
live model list first. Per-token **prices** were never trusted from any third-party source:
`model_pricing.json` ships with `null` rates and must be filled from each vendor's own pricing
page on the day of the run, with the rates + check date pasted into the appendix.

**Fairness caveats, pre-registered:**
- The person ruling (dolls don't count) is IN the prompt, so on the object set we compare
  *instructed* VLMs against *uninstructed* SAM3 — the right question for a verifier gate, but
  not apples-to-apples with EXP-2026-03's 62.2%; the doc must say so wherever the two appear
  together.
- Boxes are requested in pixels with image dimensions stated in the prompt (identical prompt
  across vendors trumps native conventions). If a model returns 0–1000-normalized coords anyway,
  the parser detects and rescales rather than punishing it (`parse_detections`, heuristic:
  image >1024px but all coords ≤1000) — rescale events are counted and reported per model.

## What we found

Run 2026-07-16 on a CPU pod. Per dataset, per model — never pooled. **All numbers are n=3
smoke-scale: plumbing proof + cost ballpark, not accuracy verdicts.**

**Five models final** (Qwen3.7-Plus added 2026-07-17; all crowd rows after box-convention
correction + the multi-block JSON fix; Flash's crowd row from the complete 2026-07-17 rerun —
its first run lost one image to a 503, which had understated its recall by ~14 points):

| n=3/dataset | GPT-5.6 Sol | Gemini 3.5 Flash | Gemini 3.1 Pro Preview | Claude Sonnet 5 | Qwen3.7-Plus |
|---|---|---|---|---|---|
| LAGENDA found / gender / age | 3/3 · 3/3 · 3/3 | 3/3 · 3/3 · 3/3 | 3/3 · 3/3 · 3/3 | 2/3¹ · 2/2 · 1/2² | 3/3 · 3/3 · 3/3 |
| CrowdHuman recall @IoU0.5 | **59.8%** | 55.2% | 50.6% | 31.0% | 56.3% |
| CrowdHuman precision | 49.5% | **98.0%** | 95.7% | 36.0% | 60.5% |
| CrowdHuman AP50 | 0.456 | **0.545** | 0.503 | 0.167 | 0.468 |
| CrowdHuman AP@[.50:.95] | 0.205 | 0.335 | **0.352** | 0.044 | 0.210 |
| Matched-IoU mean (tightness) | 0.750 | 0.818 | **0.849** | 0.657 | 0.752 |
| Recall @IoU0.3 → @0.7 | 73.6→37.9 | 55.2→47.1 | 51.7→48.3 | 56.3→10.3 | 69.0→35.6 |
| PASS false persons | 1 (low-conf) | 0 | 0 | 0 | 0 |
| Object-set false persons | 0 | 1 (teddy, high-conf) | 1 (teddy, low-conf) | 0 | 1 (teddy, high-conf) |
| Parse failures (final) | 0 | 0 (after crowd rerun) | 0 | 0 | 0 |

¹ Claude's "miss" was adjudicated from the per-person records: it detected the person and
labeled her woman/adult correctly, but its box (IoU 0.465) fell a hair under the 0.5 bar —
box tightness, not blindness, consistent with its crowd profile.
² The adult→child case is a 17-year-old read as child (est. 13) — the teen band, the same
architecture-independent failure zone SAM3 (36% of 15-19s → child) and Qwen2.5-VL-7B (~68% of
teens → child) show. Consequential direction for the product, but a shared failure band, not a
Claude-specific defect.

**What the box-tightness sweep revealed** (recall re-matched at IoU 0.3→0.7 decomposes
"missed at 0.5" into found-but-loose vs truly-undetected):

- **Gemini (both): flat curves + highest matched-IoU (0.818-0.849)** — what they box, they box
  tightly; their misses are genuinely undetected people. No threshold change helps them. After
  the complete rerun, Flash is no longer "blind": recall 55.2% is Qwen-level, at 98% precision.
- **Sol and Qwen: steep climbs (+14 and +13 pts relaxing 0.5→0.3)** — a meaningful share of
  their "misses" are real people with sloppy boxes. Coverage is better than the 0.5-recall
  suggests; box quality is the deficit.
- **Claude Sonnet 5 is the surprise reversal: at IoU 0.3 it finds 56.3%** — nearly Qwen-level
  coverage — collapsing to 10.3% at 0.7 (matched-IoU 0.657, the loosest). Its crowd problem is
  overwhelmingly BOX QUALITY, not blindness. Changes what "worst detector" means for it.
- **AP verdict: the Geminis own crowd detection — Flash takes AP50 (0.545), Pro takes
  AP@[.50:.95] (0.352, Flash 0.335)** — the only two models combining real recall with tight
  boxes. Given Flash costs a fraction of Pro, Flash is the better crowd detector per dollar;
  Pro's residual edge is marginally tighter boxes. AP is ranked on the 2-level VLM confidence,
  so not comparable to continuous-score detector mAPs.

**Qwen3.7-Plus engine findings** (both now handled in the harness):

- Emits **x-first 0-1000** boxes (third convention in the census: OpenAI/Anthropic honor pixel
  format, Gemini y-first 0-1000, Qwen x-first 0-1000) — and on one image emitted a full JSON
  block, wrote "Wait, I need to scale the coordinates...", and appended a corrected block.
  `extract_json` now takes the LAST balanced JSON object (the self-corrected answer); the
  rescale heuristic also gained a small-image trigger (coords exceeding image dims while ≤1000).
- **It's the token-heaviest model by far**: ~5,100-6,200 output tokens per people-image (vs
  Flash ~260) and ~480 even on empty scenes — visible deliberation you pay for; also slowest
  (~95-115s/image). Its $/image will hinge on DashScope's rate.

- **Plumbing:** every engine cleared the ≥10/12-scoreable bar (after the `max_completion_tokens`
  fix for OpenAI and the Gemini box-convention re-parse). Transient errors (OpenAI intermittent
  401s, Gemini 503 demand spikes) were absorbed by retry-with-backoff; one Gemini Flash crowd
  image was lost to a 503 that outlasted retries.
- **The Gemini box-convention finding:** raw first boxes like `[187, 238, 892, 552]` on a
  402×600 image — y-first, 0-1000-normalized, exactly Google's documented native detection
  format, despite the prompt specifying x-first pixels. Diagnosed by scoring saved raw text
  under all three candidate conventions (`reparse_boxes.py` prints the table); `yxyx_1000` took
  Flash's LAGENDA from 0/3 to 3/3 and crowd from 0 to 36/87 matched. Pro identical
  (0→3/3, 0→44/87).
- **Crowds:** the one dataset where every VLM clearly trails the production SAM3 baseline
  (77.1% recall, EXP-2026-03) — Sol 59.8%, Pro 50.6%, Flash 41.4%, Sonnet 31.0%. But precision
  inverts it: Gemini 95.7-97.3% vs SAM3's 86.6% (and Sol's 49.5%). Sol overdraws (106 boxes for
  87 people, half unmatched); Gemini underdraws but almost never wrongly.
- **Negatives:** across 24 verified person-free images (PASS + objects, all models), total false
  persons = 3, only one at high confidence (Flash's teddy bear). SAM3's baseline on the same
  object set was 62.2% of images gaining a false person. The instructed-VLM verifier-gate
  premise holds at smoke scale (instructed-vs-uninstructed caveat stands, §fairness).
- **Cost per 1,000 (measured tokens × estimated rates — Sol $5/$30, Flash $1.50/$9, Sonnet
  $2/$10 per M; Pro omitted, rate unverified):** empty images — Sol ~$11, Flash ~$2.40,
  Sonnet ~$1.70. Sparse people images — Sol ~$35, Flash ~$4.70, Sonnet ~$7.10. Per person
  labeled — sparse: Sol ~$6.50, Flash ~$1.30, Sonnet ~$1.50; crowds: Sol ~$2.70, Flash ~$0.60,
  Sonnet ~$0.80. Sol's reasoning tokens bill as output even on empty scenes (~260/image vs ~8).
  Exact figures pending hand-verified rates in `model_pricing.json`.

## Addendum 2026-07-19 — Gemini 3.6 Flash (sixth model, released post-run)

Same 12 seed-42 images, same frozen prompt, `--max-tokens 8000` (its thinking overruns the 4k
budget — see quirks). Verdict at smoke scale: **a regression vs 3.5 Flash on this workload — do
not swap it in for the scale-up.**

| n=3 | Gemini 3.5 Flash | Gemini 3.6 Flash |
|---|---|---|
| LAGENDA found / gender / age | 3/3 · 3/3 · 3/3 | 2/3 · 2/2 · **1 adult→child** (17yo → est. 12, teen band) |
| Crowd recall / precision | 55.2% / 98.0% | 41.4% / 81.8% |
| Crowd AP50 / AP@[.50:.95] | 0.545 / 0.335 | 0.414 / 0.237 |
| Matched-IoU mean (crowd / LAGENDA) | 0.818 / — | 0.801 / 0.924 |
| PASS / objects FPs | 0 / 1 (teddy) | 0 / 1 (teddy — victim #4, high conf) |
| Output tokens per crowd image | 2,469 | 3,515 (+42%) |
| Crowd $/1k images (reported rates) | ~$24.6 (verified $1.50/$9) | ~$28.7 ($1.50/$7.50 REPORTED, unverified) |

- The cheaper reported output rate is more than eaten by heavier thinking — Google's
  "fewer tokens" claim (made for agentic coding) does not transfer to this workload.
- **New quirks documented + handled in the harness:** (a) thinking overruns a 4,000-token
  budget and truncates the answer JSON (run with `--max-tokens 8000`); (b) occasional
  malformed JSON with orphan keys (`"label": "gender": "man"`) — `extract_json` now repairs
  key-followed-by-key sequences; (c) box convention still y-first 0-1000 (diagnose-confirmed).
- Run-to-run nondeterminism observed: crowd image 2 returned 21 detections on the first
  attempt, 14 on the rerun — at n=3 treat every 3.6 number as ±noisy even by smoke standards.

## Addendum 2026-07-22 — Gemini 3.5 Flash-Lite (seventh model, released 2026-07-21)

Same 12 seed-42 images, same frozen prompt, default 4k tokens (no truncation — see below).
**Verdict at smoke scale: the ideal verifier-gate candidate, and the cheapest model tested.**

| n=3 | Result |
|---|---|
| LAGENDA found / gender / age | **3/3 · 3/3 · 3/3** (perfect) |
| Crowd recall / precision | 42.5% / 61.7% (mid-pack — irrelevant to the gate seat) |
| PASS false persons | **0** |
| Object-set false persons | **0 — including the teddy bear** (first Gemini to reject it; only Sol and Claude had) |
| Box convention | stable y-first 0-1000 (clean diagnose, auto_axis agrees — no 3.6-style flipping) |
| Thinking | **OFF by default** — ~10 output tokens on empty scenes |
| Output tokens (sparse / crowd img) | 272 / 1,215 — a fraction of every sibling |
| Speed | fastest tested (~1-4 s/image) |

**Cost at reported rates ($0.30/$2.50, unverified):** empty ~$0.49/1k images · sparse
~$1.14/1k · crowds ~$3.51/1k · **~$0.18-0.29/1k persons — cheapest on every row, 2-5x under
Qwen.** A crop-verdict gate call runs ~$0.0005 → verifying EVERY detection of a 500k-image
corpus ≈ $50-100 total.

**Pipeline consequence:** Flash-Lite becomes the default candidate for the arbiter/verifier-gate
seat (statue rejection + crop classification are exactly its demonstrated strengths; weak crowd
detection doesn't matter because SAM3 owns detection). 3.5 Flash keeps the whole-image
benchmark crown; 3.6 remains the family's odd regression. Gate-experiment and scale-up should
both include Lite. All n=3 caveats apply — its perfect negatives are 6 images of evidence,
not a certification.

## What we can decide from this

The only decision n=3 supports is who advances to the scaled-up run, and what it will cost:

- **Advance: Gemini 3.5 Flash** (best AP50, best precision, 3/3 LAGENDA, near-best recall
  after the complete rerun — the strongest all-rounder) and **GPT-5.6 Sol** (still the raw
  recall leader; the "does paying 4-10x buy anything at scale?" control).
- **Advance conditionally: Gemini 3.1 Pro Preview** — after Flash's corrected numbers its edge
  shrank to marginally tighter boxes (AP[.5:.95] 0.352 vs 0.335) at ~1.3x the price; one
  scale-up look to confirm or kill. **Qwen3.7-Plus also earns a conditional slot**: cheapest
  per crowd person ($0.40/1k), #2 recall — if its slowness is tolerable at scale.
- **Deprioritize: Claude Sonnet 5 as detector** (weakest crowds, only adult→child leak) — but
  its perfect negatives slate keeps it a candidate for the verifier-gate-only role, which is a
  different, cheaper call pattern (one crop, yes/no) than whole-image labeling.
- **Architecture read:** no VLM replaces SAM3 as the crowd detector; Gemini-class precision +
  near-zero negatives is exactly the verifier profile the replacement pipeline's Stage-0 gate
  needs. The scale-up should measure the hybrid explicitly.
- **Scale-up cost (1,000 images/model at measured token profiles):** Flash ~$5, Sonnet ~$7,
  Sol ~$35-95 depending on crowd share. Trivial next to pod-GPU time.

## Final conclusions (2026-07-17, smoke-scale — directional, scale-up decides)

**Q1 — Can a VLM eliminate SAM3's false positives (the verifier-gate role)? Directionally yes,
and strongly.** SAM3's measured baseline (EXP-2026-03, full 259-image object set): **62.2% of
doll/statue/toy images gain a false person label** (upper bound pending the EXP-2026-04
re-verification) and 8.4 FPs/100 empty scenes. The five VLMs on samples of the same images:
**26 of 30 image-judgments perfectly clean; 4 false persons total, only 2 at high confidence —
and 3 of the 4 were the same teddy bear.** Claude Sonnet 5 was flawless everywhere. The
mechanism is what EXP-2026-03 predicted: SAM3 matches human FORM and never asks "real person?";
an instructed VLM asks exactly that. Caveats that keep this directional: instructed-VLM vs
uninstructed-SAM3; n=3/dataset; and we measured whole-image detection, not the actual gate call
pattern (SAM3 crop in → real-person yes/no out), which is cheaper and needs its own run.

**Q2 — Can a VLM be the MAIN labeling mechanism? Not alone.** Every VLM trails SAM3's 77% crowd
recall (Sol 59.8% best) — as sole labeler they silently drop roughly 1-in-2 people in crowded
scenes vs SAM3's 1-in-4. But label *quality* inverts the story: Gemini's 96-98% box precision
beats SAM3's 86.6%, and for training data the error directions aren't symmetric — a false label
poisons training, a missed person only shrinks it. On sparse/prominent-subject images (LAGENDA),
4 of 5 VLMs were perfect on detection+gender+age at n=3. The teen age band fails in every VLM
(Claude read a 17-year-old as 13) exactly as it does in SAM3 and Qwen2.5-VL-7B — so the
default-to-adult policy for the ambiguous band stays necessary under any labeler. Gender on the
Gulf-dress slice remains untested (the standing gap, unchanged).

**Recommended architecture — unchanged but now evidenced from both sides: SAM3 detects, a VLM
verifies and classifies.** SAM3 supplies the recall no VLM matched; the VLM supplies the
person-verification and label cleanliness SAM3 structurally lacks.

**Next steps, in order:** (1) the gate-specific experiment — feed SAM3's actual detections
(including its real false positives from EXP-2026-03's runs) as crops to the top VLMs and
measure kept-vs-rejected; (2) the 250-image/dataset scale-up (~$50-150 total) with
pre-registered bars — Flash + Sol as anchors, Pro and Qwen conditional.

## What this does NOT tell us

- **Nothing about accuracy, at n=3.** Twelve images per model. One odd image moves any rate by
  ~33 points. Every number in "What we found" is a smoke-test observation, not a measurement.
- **Not comparable to EXP-2026-01's crop-based numbers.** Whole-image detection is a different,
  harder task than describing a pre-cropped person. A lower gender accuracy here does not mean
  the model got worse than the EXP-2026-01 baseline — different measurement.
- **Object-set comparison to SAM3 is instructed-vs-uninstructed** (see fairness caveats).
- **Does not test the Gulf/traditional-dress bias** — same structural gap as every experiment so
  far; needs the still-unbuilt hand-checked slice.
- **Cost is a point-in-time snapshot** of pricing that changes; and per-image cost depends on
  image resolution (vision token counts scale with pixels), so the 12-image average may not
  transfer to a differently-sized dataset.
- **LAGENDA's ~1-label-per-image limit still applies:** its C1 read is recall-only; a model
  boxing extra real-but-unlabeled people is not penalized there (that's what CrowdHuman is for).

## What's next

- If ≥2 engines pass plumbing: scale the winners to ~250 images per dataset with the identical
  frozen prompt and pre-register real per-dataset bars (crowd recall vs SAM3's 77%, object-set
  false-person rate vs SAM3's 62%, LAGENDA leak direction) before that run.
- Feed the object-set result into the person-verifier-gate design for the replacement pipeline
  (Track 1 build) — this experiment is the gate's candidate-selection step.
- The Gulf-dress slice remains the standing gap; a VLM that survives the scaled run becomes a
  candidate for labeling that slice cheaply.

---

## Appendix — the details

- **Runbook:** `EXP-2026-06-POD-RUNBOOK.md` — copy-paste pod blocks for all 16 runs (4 models ×
  4 datasets) + what to paste back.
- **Sample selection:** 3 images per dataset, `sorted()` file list shuffled with
  `random.Random(42)`, first 3 taken — deterministic, identical across models
  (`vlm_detect_eval.py:select_images`).
- **The 16 runs:** see the runbook; one `vlm_detect_eval.py` invocation per (engine × dataset),
  each writing `detections.jsonl` + `summary.json` + `cost_report.json` under
  `/workspace/exp06/<engine>/<dataset>/`.
- **Filled-in pricing (paste before trusting cost numbers):** _model — rate checked [date] —
  $/1M input — $/1M output — vendor URL._ Blank until run day.

### Appendix — "verify it yourself"

- **The full frozen prompt** — `vlm-cluster/vlm_detect_eval.py:49-83` (`DETECT_PROMPT`), quoted
  here in full so reviewers don't need the repo:

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

- **How each model was called** — same request code as the original design, one image + the
  prompt above per call: OpenAI `vlm-cluster/api_describers.py:126-148` (Chat Completions,
  data-URI image), Gemini `:150-174` (`google-genai`, `Part.from_bytes`), Claude `:176-193`
  (Messages API, base64 image). Default models `gpt-5.6-sol` / `gemini-3.5-flash` /
  `claude-sonnet-5`; the Gemini Pro run passes `--model gemini-3.1-pro-preview`. Max output
  tokens 4000 per call (`--max-tokens`; raised from the describers' 300 default after the first
  smoke attempt — 300 truncates any crowd past ~4 people, and thinking models like Gemini 3.5
  Flash spend reasoning tokens from the same budget, which zeroed out its answers). Raw text is
  saved per image (`detections.jsonl:raw_text`) so truncation stays visible.
- **How output was parsed** — `parse_detections`, `vlm_detect_eval.py:89-146`: first `{...}`
  block via `describe.py:extract_json`; entries without a valid 4-number box dropped; boxes
  clamped to the image; the 0–1000 rescale heuristic quoted above; `parse_ok`/`rescaled` flags
  recorded per image.
- **How boxes were matched** — `match_boxes` imported from
  `vlm-cluster/run_model_children.py:44-66` (greedy, mutually-exclusive, best-IoU-first — the
  same function every prior experiment used; quoted in EXP-2026-02 Appendix B). Match IoU 0.5
  (`--match-iou`, default at `vlm_detect_eval.py` argparse).
- **Ground-truth loaders** — CrowdHuman: `load_odgt` (`vlm_detect_eval.py:156-174`) uses
  **vbox** (visible box), `tag=="person"`, ignore-tagged boxes excluded from FP counting —
  same convention as EXP-2026-03. LAGENDA: `load_lagenda_gt` (`:177-205`) joins `gt.jsonl`
  (`{"id": "<stem>_<n>", "gt_age", "gt_gender"}`) to line *n* of the YOLO label file.
- **Every threshold & configuration:**

  | Setting | Value | Source |
  |---|---|---|
  | Images per dataset per model | 3 (smoke test) | pre-registered above |
  | Random seed | 42 | `--seed` default |
  | Match IoU (CrowdHuman + LAGENDA) | 0.5 | `--match-iou` default |
  | Child/adult cutoff in prompt | ≤12 | `DETECT_PROMPT`, production cutoff |
  | Rescale heuristic trigger | image side >1024px AND all coords ≤1000 | `parse_detections` |
  | Max output tokens | 4000 | `vlm_detect_eval.py --max-tokens` default (300 would truncate crowds at ~4 people, and thinking models spend reasoning from the same budget) |
  | JPEG quality (API encoding) | 90 | `api_describers.py:_encode_jpeg` |
- **Box-convention handling (added after first pass):** Gemini responses are re-parsed under
  its native `yxyx_1000`; the diagnose-then-rewrite tool is `vlm-cluster/reparse_boxes.py`
  (scores saved raw text under all three conventions against GT so the choice is data-driven,
  not assumed — its selftest covers the transposition). Scale-up runs should parse Gemini
  output as `yxyx_1000` from the start.
- **What was NOT done:** no Qwen2.5-VL-7B baseline in this mode (dropped by decision — the
  crop-based EXP-2026-01 numbers stand as its record; a whole-image Qwen run would need GPU time
  and wasn't judged worth it at smoke-test scale); no per-vendor prompt tuning; no retry on API
  failure (one try, error logged, image scored as zero detections — `_safe_call`); no
  repeat-call variance; match-IoU 0.5 not sensitivity-swept at this n; the 0–1000 rescale
  heuristic is untested against a model that genuinely detects only in the top-left 1000px
  corner of a huge image (accepted risk, visible in `raw_text` if it ever happens).
