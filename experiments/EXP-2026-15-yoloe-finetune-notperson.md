# EXP-2026-15 — YOLOE fine-tuned on Spotlight labels + a trained `not-a-person` class from the 192k deleted boxes

**In one line:** the only version of the YOLOE bet with a capability nobody else in the
comparison has — fine-tune it on Spotlight labels with a 5th class, `not_person`, trained on
the **192,388 boxes the Spotlight run deleted as not-a-person** (SAM3 geometry, Gemini
verdicts — free hard-negative labels already on the volume), attacking the users'
blurs-books bug with real supervision instead of the Flash-Lite gate's $0.54/1k.

**Status: TRAINED 2026-08-07 — `yoloe_n_gradsupp` (yoloe-26n-seg, 30 epochs, 26.6 h on an
RTX 4090; portable ckpt `/workspace/exp15/train/yoloe_n_gradsupp/weights/best_plain.pt`;
W&B run yoloe_n_gradsupp). Evaluation NOT yet run — scored in the same upcoming dump
session as the EXP-2026-14 arms, against the W1/W2 bars below.** Re-scoped 2026-08-06
(owner decision), tooling built + selftested. The
not_person class (class 4) is **deferred** — this experiment is now a **4-class YOLOE
fine-tune with the Unknown handled by cls-gradient suppression** (EXP-2026-14's arm-C
mechanism, ported): `vlm-cluster/train_gradsuppress_yoloe.py`. Port verified against
pinned ultralytics 8.4.115 source — the PE fine-tune path has no visual prompts, so
`YOLOESegModel.loss` falls through to the standard seg criterion (`nn/tasks.py:1489`),
and `GSSegmentationLoss(GSDetectionLoss, v8SegmentationLoss)` composes by MRO. Selftest
(CPU, both plain and e2e seg paths, class-3 GT + masks, zero cls-grad on unknown anchors,
box loss on unknown-only, checkpoint pickle round-trip): **passed 2026-08-06.** The
PE-fuse trainer path (needs the CLIP text model) is pilot-verified on the pod, not in the
selftest. Data: **reuses EXP-2026-14's `tree_c` + `unk4_c.yaml` unchanged** (labels_unk3
polygons feed the seg branch directly). **Recorded deviation:** the GS design is adopted
here ahead of EXP-2026-14's arm-C verdict, by owner choice — if arm C fails its bars, this
run's unknown-handling premise inherits that result. Bars below unchanged; with class 4
deferred, W1 (halving object-set FPs) is the less likely win condition and W2 (crowd
recall) carries the case — recorded before running.

**Scale amendment (2026-08-06, pre-run, owner decision): primary arm = `yoloe-26n-seg`
(5.6M params / 11.7 GFLOPs), not the originally-frozen 26s.** Rationale: 26s (15.3M /
39.3 GFLOPs, ~110 ms measured as ONNX) can never ship; nano is the only scale with any
production adjacency (still ~2× the incumbent y26n's FLOPs — recorded expectation:
~60–70 ms CPU, a win must come from accuracy). 26s becomes a conditional follow-up
(labeling-seat/teacher ceiling) only if nano passes a win condition; 26l/distillation is
out of scope until a teacher is justified. Est. nano cost on RTX 4090: ~25–33 h, ~$20.
Training telemetry: W&B enabled for this experiment's runs (project `exp15/train`).

Original draft (not_person design, kept for possible revival) follows; it was blocked on
two inputs:
(a) EXP-2026-14's unknown-handling verdict (decides the unknown class design here), and
(b) the not_person label build + 100-crop curation sample (below). Bars pre-registered at
draft time; nothing runs before both inputs land. · Owner: Mostafa

---

## Why (and why only this version)

Zero-shot YOLOE was rejected for every seat (EXP-2026-13). A plain Spotlight fine-tune of
YOLOE would most likely tie the YOLO26 arms at 6× the size and ~3× the latency (EXP-2026-12:
labels dominate architecture) — a tie at that cost is a loss, so we don't run that version.
What justifies any run: the Spotlight audit trail (`run/oiv7_train/_audit.jsonl`) records
every deleted-not-person detection with recoverable geometry (join `(image_stem, det_index)`
→ raw sidecar polygon). That is a **192k-box labeled hard-negative dataset at $0** — dolls,
statues, posters-misjudged, books — i.e. supervision aimed exactly at the false-person
failure that SAM3 fails at 62% and users experience as blurred books. Only a
multi-class-flexible model can consume it without disturbing the production 0-2 mapping.

**Seat in play: labeling pipeline only** (detector seat vs SAM3, or distillation teacher).
14M params / ~110 ms disqualifies the extension seat regardless of accuracy. AGPL applies to
shipping; server-side labeling is a different conversation (owner call, flagged as in
EXP-2026-13).

## Design

**Classes:** `0 woman · 1 man · 2 child · 3 unknown · 4 not_person`. Classes 3 and 4 are
discarded at inference; 4 exists to absorb human-shaped non-people *with supervision* (the
trained version of the EXP-2026-13 distractor trick, which was safe but under-powered
zero-shot: −22.5% FP vs the ≥30% bar).

**Unknown handling (class 3):** whatever EXP-2026-14 adopts — unknown-as-class if arm B wins
or ties (zero extra code here), gradient-suppression port if arm C wins decisively (extra
work: the GS loss must be ported to YOLOE's trainer path; budgeted as +1 day, selftest
mandatory before any GPU).

**Training:** `yoloe-26s-seg` warm start, 30 epochs, same data tree/staging as EXP-2026-14,
labels = `labels_unk3` + class-4 lines merged in. Ultralytics YOLOE trainer
(`YOLOEPESegTrainer` family — exact trainer per docs at run time, not from memory). $100 cap
pre-registered (fine-tune ~$30–60 + label build + eval).

**not_person label build (new tool, `--selftest` required):** walk `_audit.jsonl` for
delete-verdict entries, pull each box's polygon from `raw/oiv7_train/<stem>.json` by
det_index, emit class-4 lines merged into a copy of the train labels. **Curation gate before
training:** a 100-crop seeded sample of the built class-4 boxes, eyeballed against the
EXP-2026-03 person ruling — the known risk is posters/photos-of-people sitting in the
deleted set (posters COUNT as people per the ruling; Gemini judged under a prompt carrying
it, but verify, don't assume). Bar: ≥90 of 100 sampled crops are genuinely not-people, else
filter (e.g. by Gemini's verdict reason field) and re-sample before any training.

## Pre-registered bars (written 2026-08-06, before the label build or any training)

Comparator = the best EXP-2026-14 YOLO26n arm (by its own adoption rule). YOLOE earns a
labeling-pipeline seat evaluation only if at least one of:

- **W1 (the not_person payoff):** object-set image-FP rate ≤ **half** the comparator's, OR
- **W2 (detection payoff):** CrowdHuman recall ≥ comparator **+3 pts**.

AND all no-regression bars hold: LAGENDA detection ≥ 94% · adult gender ≥ 90% ·
adult(≥20)→Child ≤ 0.3% · PASS ≤ 0.4 FPs/100. Fail both W1 and W2 → the YOLOE line is
closed with numbers and the money stops here.

Conditional ablation (runs ONLY if W1 passes): same fine-tune WITHOUT class 4, to attribute
the object-set gain to the not_person class rather than the YOLOE backbone. Skipped if W1
fails (nothing to attribute).

**Evaluation:** the standard protocol — 4 datasets + Spotlight-val mAP, conf 0.45, raw
sidecars at 0.05, classes 3–4 stripped from emitted labels (kept in sidecars). Results
appended to `EXP-2026-12-TEAM-BRIEF.md`.

## What we found — fine-tuned `yoloe_n_gradsupp`, scored 2026-08-08

Comparator = EXP-2026-14 arm C (`y26n_gradsupp`, the best YOLO26n arm), identical
protocol:

| metric | y26n_gradsupp (2.4M) | **yoloe_n_gradsupp (5.6M)** |
|---|---|---|
| CrowdHuman recall / precision | 36.5% / 94.0% | 36.3% / 93.5% |
| Object-set image-FP | 11.2% | 11.6% |
| PASS FPs/100 | 0.27 | 0.37 |
| LAGENDA detection | 93.9% | 93.3% |
| Adult gender | 91.1% | 91.5% |
| adult(≥20)→Child leak | 0.11% | 0.15% (4/2,693) |
| Spotlight-val mAP50 / 50-95 | 0.799 / 0.689 | 0.801 / 0.691 |

**W1 (object-FP ≤ half of comparator): FAIL** — 11.6% vs the ≤ 5.6% needed.
**W2 (crowd recall ≥ comparator +3): FAIL** — 36.3% vs the 39.5% needed; it ties.

**Verdict, per the pre-registered rule: both win conditions failed → the YOLOE line is
closed with numbers.** The pre-run prediction ("most likely outcome: ties y26n at bigger
size") is exactly what happened — statistical ties on every metric, at a real compute
penalty. Open-vocabulary pretraining bought nothing that Spotlight labels + a smaller
closed-vocab model didn't already have. The 26s follow-up does not run. One transferable
positive: the gradient-suppression port behaved identically on a second architecture and
training task (segmentation), strengthening EXP-2026-14's Q2 finding.

**Size/speed measured 2026-08-08 (correcting the pre-run assumption):** the deployable
fused model is **2.69M params — not the 5.6M in the architecture spec**, because the
text-prompt machinery is stripped during PE fine-tuning. So the penalty is compute, not
parameter count: **9.1 GFLOPs vs y26n's 5.3, and 49.1 ms vs 33.8 ms** (same session, EPYC
9254, ONNX CPU 4 threads, NMS baked into both). 45% slower for zero accuracy gain — the
verdict is unchanged, but the reason is now measured rather than assumed.

## What this will NOT tell us (written at draft time)

- Nothing about the Gulf-dress gender slice (unchanged biggest gap).
- Nothing about the extension seat — size/latency excludes it before accuracy is measured.
- The not_person class is trained on SAM3's *specific* false positives; generalization to a
  different detector's FPs is untested.
- If the curation gate fails badly (deleted set turns out poster-heavy), the W1 premise
  weakens and the experiment should be re-scoped, not pushed through.

## Run order

1. EXP-2026-14 verdict lands → fixes the class-3 design.
2. Build + curate not_person labels (CPU, the 100-crop gate).
3. Fine-tune (one A100/4090 pod, ~1–2 days incl. eval), bars applied as pre-registered.
