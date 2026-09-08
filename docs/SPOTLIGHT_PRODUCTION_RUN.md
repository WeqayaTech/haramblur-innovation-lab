# Spotlight production run — how we labeled 475k images, and why it's built the way it is

*2026-07-28 → 07-30. Written as the durable record: the next run like this is 6–9 months away,
and this document is what makes that run a repeat instead of a rediscovery.*

---

## 1. What was produced (the bottom line)

**Every person-detection in the Open Images v7 train split, verified individually — nothing
sampled, nothing estimated.**

| | |
|---|---|
| Images labeled | 475,207 (train) + 4,233 (val, same machinery, run first) |
| Detections found by SAM3 | 1,579,649 |
| Detections verified by Gemini | **1,579,649 — completion proven by counting files, not estimating** |
| Final training labels | **1,191,142** classed (Woman/Man/Child) + 196,119 class-3 Unknown |
| False detections removed | 192,388 (12.2%) — books, statues, backgrounds SAM3 called people |
| Classes corrected | 61,441 — incl. **14,229 adults rescued from Child labels** |
| API cost | **$507** measured ($0.321/1k detections, stable to 0.3% across 100+ jobs) |
| GPU cost | ~$150 across ~10 ephemeral pods |
| Response parse failures | 346 of 1.58M (**0.02%**) |
| Data duplicates | 0 (verified: 1,420,000+ unique keys, zero double-collected) |

Deliverables on the volume: `run/oiv7_train/labels_unk3/` (recommended, 4 classes),
`labels/` (3 classes), full per-detection provenance, and `DATA_README.md` as the
consumer-facing handoff. This doc is the producer-facing one.

---

## 2. The pipeline, stage by stage, with the reason each stage exists

```
images ──> [1] SAM3 detect ──> raw labels + RAW SIDECARS
                                     │
                  [2] build crop: 25%-padded, mask-OUTLINED, upscaled ≥320px
                                     │
                  [3] Gemini 3.5 Flash-Lite verdict per detection (Batch API)
                                     │
                  [4] merge rules: keep/delete, final class
                                     │
                  [5] emit YOLO labels + _audit.jsonl
```

**[1] SAM3 with text prompts ("woman", "man", "child"), frozen production config.**
Why not replace the detector? We measured it first (EXP-2026-02/03/04): recall is excellent on
prominent subjects (98.7%), weak in crowds (77%), and it hallucinates people onto human-shaped
objects (62% of doll/statue images). The economics said: keep the cheap high-recall detector,
kill its false positives downstream. Replacing the detector was a bigger project with less
measured upside than gating it.

**The raw sidecars are not optional.** Alongside each YOLO `.txt`, the labeler writes a `.json`
with per-part mask polygons (before YOLO's lossy single-polygon flattening), per-detection
confidence, and the NMS-suppressed detections. Reason: *log raw readings before lossy
post-processing so analysis never needs a re-run.* This paid for itself three times in one
week — the bridge-rendering fix, the confidence-gate analysis (EXP-2026-07), and two full
label-policy re-emits that cost 20 minutes instead of $500.

**[2] Mask-outlined crops — the "Spotlight" idea, and the single most load-bearing design
decision.** A crop of one person in a crowd contains parts of other people; a VLM asked "what
gender is this person?" will sometimes answer about a neighbor. Outlining the person's exact
mask (two-tone, so it reads on any background) tells the model *which* person. Two corrections
found during validation, both now defaults: a translucent tint *obscured* small people (crowd
TP-keep fell to 80% — outline only), and small crops upscale to ≥320px (free: Gemini bills
small images the same).

**[3] Gemini 3.5 Flash-Lite, one verdict per detection, via the Batch API.**
- *Why Flash-Lite:* benchmarked against GPT-5.6 Sol, Gemini Flash/Pro/3.6, Claude Sonnet,
  Qwen (EXP-2026-06/08). Lite was 3–30× cheaper at the gate task, thinking-off by default
  (~tiny outputs), and passed its pre-registered gate bars decisively (TP-keep 98.9%).
- *Why verify EVERY detection instead of routing only suspicious ones:* measured Batch cost
  ($0.32/1k) made full coverage ≈ $500 — cheap enough that the complexity of an escalation
  router wasn't worth its failure modes. EXP-2026-07 also proved SAM3's confidence cannot
  route: its false positives score as high as real people.
- *Why Batch API:* 50% discount and unbounded server-side parallelism. Jobs of 10–20k
  completed in ~7–10 min. The submit side (crop building, ~35 ms/det) is the real bottleneck —
  solved by sharding submits across machines by image-path hash.
- *Prompt:* frozen spotlight-e1 (9 fields: verdict/gender/age_group/estimated_age +
  analysis fields). 400-token output cap — measured usage is 80–180, the cap is cost
  containment, not a constraint.

**[4] Merge rules (v1 policy, deliberately simple):** Gemini decides *existence and class only*;
SAM3's geometry always wins (box-correction was tested and rejected — Lite's corrections were
relocations, not refinements). `not_person` → delete. `depiction` (photo/poster of a real
person) → **keep** — owner's religious ruling: a depiction of a real person is still a
gaze-lowering target. `child` + age ≤12 → Child. Committed gender → Man/Woman. Everything
else → Unknown.

**[5] Emit + audit.** Labels are a *projection* of the verdicts; `_audit.jsonl` records every
delete/relabel/drop with its reason. Two projections shipped: `labels/` (unknowns dropped) and
`labels_unk3/` (unknowns kept as class 3 so the trainer masks those regions at load time —
the trainer-owner's choice over grey-filling pixels, because it needs no image duplication and
stays reversible). Changing policy = re-emit (~20 min, zero API), never re-label.

---

## 3. Decisions that were argued, and how they were settled

| Decision | Chosen | Rejected | Why |
|---|---|---|---|
| Keep or replace SAM3 | Keep + gate | Detector bake-off first | Measured FP problem was fixable for $500; detector swap was unmeasured risk |
| Verifier model | Flash-Lite | Bigger VLMs | 3–30× cheaper, validated on pre-registered bars (EXP-2026-08) |
| Coverage | 100% of detections | Confidence-routed subset | SAM3 conf can't separate FPs from TPs (EXP-2026-07); full coverage was affordable |
| Target indication | Mask outline | Box outline; mask tint | Box mislabels neighbors (proven twice — OIV7 pilot AND crowd pilot); tint hides small people |
| Unknowns policy | Class 3, trainer masks | Drop silently; grey-fill pixels | Dropping trains "ignore this person" (false negatives); grey-fill duplicates storage; class-3 is reversible and storage-free |
| Child cutoff | ≤12, teen→adult default | ≤9 | Adult=blur is the safe error direction; both SAM3 and VLMs fail the teen band anyway (EXP-2026-01/02) |
| Depictions | Labeled like real people | Treated as FPs | Owner ruling: gaze-lowering applies to photos/posters of real people |
| Batch vs live API | Batch | Live | Half price; latency irrelevant for a corpus job |
| Crowd detection (parked track) | CrowdHuman human GT boxes | SAM3 detection | SAM3 misses 1-in-4 crowd people; every miss becomes a false-negative training signal — the worst error for this product |
| Crowd target indication | SAM-ViT-huge box-prompted masks | GT-box outline | Box outline failed the pilot (wrong-person labels); box-prompted SAM gives the person's mask without SAM3's recall problem ever entering |

---

## 4. Findings

**4.1 What verification actually catches at corpus scale.** Verdict mix was stable within ~1
point across every 100k slice: **82.0% real_person, 11.6% not_person, 6.4% depiction.** The
12% not-person rate is the trained-model "blurs books/backgrounds" bug measured at its source,
and removed.

**4.2 The error structure inverted vs. the benchmark.** On LAGENDA (prominent subjects),
SAM3's errors were 93% age-direction. On the real corpus, **gender corrections outnumber age
corrections 2.4:1** (42,669 vs 18,772). Benchmark subjects are large and easy; corpus people
are small and hard. Lesson: benchmark error *structure* does not transfer to production
distributions, even when headline accuracy does.

**4.3 A bias signal, free of charge.** Woman→Man corrections (30,567) outnumber Man→Woman
(12,102) by 2.5×. This is the same direction as the known men-in-Gulf-dress→Woman bias. It is
**unattributed** — nobody has eyeballed the crops yet — but it is the first corpus-scale,
zero-cost probe sample for the project's oldest open question. (Open item d.)

**4.4 The unknowns are a size phenomenon.** 196k detections (12.4%) got no committable gender:
median height **58px** vs 352px for committed calls. The verifier abstains on unreadable
people — correct behavior, but it means "unknown" is effectively a minimum-size policy, and it
should be owned as one.

**4.5 The adult-escape metric.** 14,229 adults were labeled Child by SAM3 and corrected —
each one would have escaped the blur. The reverse direction (4,543 adults→Child) is the
acceptable over-blur direction. This asymmetry is exactly what the product needs.

**4.6 Completion must be counted, never estimated.** The dashboard's sampled
detections-per-image (2.92) undercounted true density (3.324); "AWAITING 0" silently hid
159,655 unsubmitted detections. The only valid completion claim: count every sidecar
detection, set-subtract verified keys, require MISSING = 0. That check runs in ~4 minutes
with a 24-worker pool and caught what every estimate missed.

**4.7 Crowd pilot findings (track parked).** GT-box outline crops: FAILED — Gemini labels
neighbors inside the rectangle. Box-prompted SAM masks: fixed it. On the first 250 crowd
images (5,139 people): unknown-gender 15.5% (better than the general corpus — the mask
highlight communicates well), parse failures 0.02%, but **not_person 4.2% on human-annotated
boxes** — Gemini disputing human annotators 1-in-24, requiring adjudication before any emit
(mannequin? unreadable sliver? bad mask?). Kept honest by keeping the failed pilot's verdicts
on disk as the A/B record.

**4.8 Cost model, now measured.** $0.321/1k detections (Batch), ~1,480 input + ~80 output
tokens per detection. Full-corpus verification of ~1.6M detections ≈ $510. Crop-building
throughput ≈ 10k detections / 7 min / process; shard it. GPU labeling ≈ $0.06–0.09 per 1k
images on RTX 4090s (the best price/throughput of everything tried; A100s only win when
thread-pinned and even then marginally).

---

## 5. Operational lessons (the expensive kind)

1. **RunPod volume quota is invisible to `df`.** Full volume = silent write failures while
   processes "run": truncated `jobs.json` (0 bytes), frozen emit at 1% CPU, `.txt` files
   without sidecars. Check the console; probe with `dd`; after any interruption run the
   txt/json pair-check (once found 13,482 orphans that would have silently skipped forever).
2. **Gemini File API: 20 GiB per-project upload cap, files persist 48 h.** ~30 uncollected
   batch jobs fill it; then every submit 429s. Delete collected jobs' input files every cycle;
   roll back local reservations when upload fails (the leak once manufactured 650k phantom
   "in flight" detections).
3. **`nproc`, `uptime`, `vmstat` report the HOST inside RunPod containers.** Torch sizes its
   thread pool from nproc → 8 workers unpinned ran at 6% GPU. `OMP_NUM_THREADS` pinning was a
   10× throughput difference. Never size anything from host-reported numbers; measure
   throughput from the filesystem (`find -newermt`).
4. **The local ledger is not the truth; the API is.** `client.batches.list()` reconstructs the
   job ledger completely (display_name carries the local join key). After any corruption:
   rebuild from API, mark collected = "all keys have verdicts on disk".
5. **Serial walks over 475k network files take hours; a 24-worker Pool does ~2,100 files/s.**
   Parallelize every corpus walk, print progress every 50k, and prove equivalence against the
   serial reference (parallel_emit's selftest asserts byte-identical output).
6. **Verify every launch immediately** — `hostname` before pasting, one `pgrep` after, grep
   the log for the first real output line. The night's incidents were overwhelmingly
   "command ran on the wrong pod / with an unset variable / from the wrong directory," each
   invisible until minutes later. Absolute paths, hostname-stamped logs, per-shard claim
   locks, and unbuffered python (`-u`) are the standing defenses.
7. **Concurrency needs a single writer or disjoint outputs — nothing in between.** jobs.json
   append races lost records; two labelers on one output dir made orphan files; the fixes were
   per-shard files, per-shard locks, and rebuild-from-API. Hash-sharding by path
   (`md5(path) % N`) gives disjoint-and-complete work splits with zero coordination.

---

## 6. How to run this again in 6–9 months

1. Read this doc, then `DATA_README.md` (volume) for current data state, then
   `docs/AUTOLABEL_PIPELINE_V2_RUN.md` if it exists for the frozen runbook.
2. Production code: `/workspace/autolabel_pipeline_v2/` (pod) — diff against the dev copy in
   this repo's `vlm-cluster/` before trusting either. `pod_setup.sh` first on every pod.
3. Stage order: labeler (GPU pods, 2 workers/4090, hash-sharded, claim-locked) → batch
   submit/collect loops (CPU, sharded, storage-cleanup inside the loop) → completion check
   (count files, MISSING must be 0) → emit (parallel_emit, pick unknown policy) → pair-check →
   trace-report samples for eyeballs → DATA_README update.
4. Pre-register the verdict-mix expectations (82/11.6/6.4 ±2) and the cost ($0.32/1k ±10%) as
   the run's health bars; deviation means investigate, not continue.
5. Budget shape for a similar corpus: API ≈ $0.33/1k detections; GPU ≈ $0.08/1k images;
   density ≈ 3.3 det/img on Open-Images-like data (measure your own on the first 5%, from a
   *random* sample, and re-measure — ours drifted 2.92 → 3.324 and hid 11% of the corpus).

## 7. What this run does NOT establish

- **Gulf/traditional-dress gender accuracy** — still the project's biggest untested gap. The
  30k Woman→Man corrections are the ready-made probe sample.
- **Gemini's absolute accuracy at production scale.** The verifier was validated on benchmarks
  (LAGENDA, CrowdHuman, hand-verified object sets) with pre-registered bars; production
  quality control was verdict-mix stability + ~170 visually spot-checked images. No human
  ground truth exists for the 1.58M production verdicts themselves.
- **PASS negatives cleanliness** — flagged list exists, exclusion not yet applied.
- **Anything about the crowd set beyond its 250-image pilot** — parked, self-contained,
  adjudication pending.
