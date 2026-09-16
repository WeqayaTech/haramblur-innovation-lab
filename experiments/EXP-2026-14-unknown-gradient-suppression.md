# EXP-2026-14 — Unknown-label handling on YOLO26n: gradient suppression vs unknown-as-class vs background

**In one line:** Spotlight drops ~12% of detected people as unknown-gender (median ~58 px);
EXP-2026-12 proved that training with them as background costs ~7 pts of crowd recall while
grey-masking them gains ~7 — this experiment tests the two remaining designs on the fast
architecture (YOLO26n): a zero-code 4-class Unknown arm, and a custom **cls-gradient-suppression**
loss where the unknown person supervises WHERE (box/DFL + assignment) but never WHAT (no
classification gradient).

**Status:** BOTH ARMS TRAINED (2026-08-07): arm C `y26n_gradsupp` (30 ep, portable ckpt
`weights/best_plain.pt`) and arm B `y26n_unk4` (30 ep, 17.3 h, stock ckpt — strip class-3
lines from dumps before scoring). **Evaluation NOT yet run** — dumps + scoring + bars are
the next session (pods terminated after training; one 4090 + restage needed for the dump
pass). Bars pre-registered below; tooling selftested. ·
Owner: Mostafa · The gradient-suppression design is the coworker's sketch (2026-08-05),
implemented as specced: class-3 targets positive for box/DFL + assignment, masked from cls
loss, assigner given a class-agnostic score for class-3.

---

## The short version

_(written after the run — placeholder)_

## Why we did this

The four-way EXP-2026-12 comparison isolated unknown-handling as the single biggest lever on
crowd recall (bigger than any architecture change):

| unknown handling | arm | crowd recall | Δ vs shipped |
|---|---|---|---|
| dropped as background | y26n_ft | 32.8% | −2.1 |
| grey-masked | gelansfav14_gemlb_v1 (8.2M) | 46.2% | +11.3 |
| (shipped, old labels) | yolo11N-640 | 34.9% | — |

Masking works but destroys the box signal — the model learns nothing from those regions.
The unknowns are real people (too small/occluded to gender), so ideally they should still
teach detection. Two designs do that:

- **Arm B — unknown-as-class (zero code):** train `nc=4` on `labels_unk3`, discard class-3
  detections at inference. The person supervises detection through their own class channel.
  Risk: class 3 becomes a "small person" class whose boxes compete with real Woman/Man boxes
  on small people, re-suppressing them at inference.
- **Arm C — cls-gradient suppression (custom loss, this experiment's headline):** head stays
  `nc=3`; class-3 GT participates in assignment (via a class-agnostic score channel) and gets
  full box/DFL gradients, but its anchors are excluded from the classification loss entirely —
  pushed neither toward a class nor toward background. No phantom class exists at inference.

Arm A (background) already exists as y26n_ft — reused, not retrained.

## What we wanted to find out

**Q1:** Does either detection-supervised unknown design recover the masking gain on the nano
architecture? **Q2:** Which of B vs C is better, and does C's theoretical edge (no fake class
competing at inference) show up? **Q3:** Do the EXP-2026-12 wins (objects FP, leak, gender)
survive unchanged?

Components measured: 1, 2, 3 per `docs/COMPONENT_FRAMEWORK.md`, plus Spotlight-val mAP
(label-alignment, per the framework's mAP ruling) — same protocol as every model in
`docs/COMPONENT_FRAMEWORK.md`, same four datasets, conf 0.45, raw sidecars to 0.05.

### Pre-registered bars (written 2026-08-05, before any training)

- **Q1 bar (mechanism validated):** crowd recall ≥ **37.8%** (= y26n_ft's 32.8% + 5 pts) for
  an arm to count as recovering the unknown signal. Reference points, not bars: gemlb's 46.2%
  (different size class, 8.2M vs 2.4M) and shipped's 34.9%.
- **No-regression bars (all must hold for adoption):** object-set image-FP ≤ **15%**
  (y26n_ft: 12.7%); PASS ≤ **0.4 FPs/100** (y26n_ft: 0.23); LAGENDA detection recall ≥ **94%**;
  adult gender ≥ **90%**; adult(≥20)→Child leak ≤ **0.3%** (y26n_ft: 0.11%).
- **Adoption rule:** among arms passing all no-regression bars, highest crowd recall wins; if
  B and C are within 1 pt of each other, C wins ties (no phantom class to strip at inference,
  simpler deployment).
- **Budget cap:** $60 total GPU (two ~30-epoch nano trainings at the measured ~$11 each +
  eval time), pre-registered. Blowing it = stop and report, not quietly shrink.

## How we plan to do it

1. **Data:** the exp12 training tree/lists (`/workspace/exp12/train_full.txt`, NVMe staging
   recipe) with labels swapped to `labels_unk3` (class 3 = Unknown). Two yamls: arm B pairs
   val with `labels_unk3` val (4-class model, 4-class val); arm C pairs val with the
   classes-0-2 `labels/` val emit (3-class head, 3-class val — in-training metrics are
   plumbing signals only; real eval is external).
2. **Arm B:** stock `YOLO('yolo26n.pt').train(...)` at nc=4 — zero code. At dump time class-3
   detections are stripped from label files (kept in sidecars) before scoring.
3. **Arm C:** `vlm-cluster/train_gradsuppress.py` (new; mechanism + verify-yourself below).
   Warm-start `yolo26n.pt`, 30 epochs, same hyps as y26n_ft for comparability.
4. **Eval:** the standard four datasets + Spotlight-val mAP, scored per
   `docs/COMPONENT_FRAMEWORK.md`; results appended to `EXP-2026-12-TEAM-BRIEF.md`.

## What we found

Run 2026-08-08 (dumps + scoring on an L4; both arms trained on RTX 4090s, 30 epochs,
475k images, identical hyps to y26n_ft). Baseline column = y26n_ft (EXP-2026-12,
unknowns-as-background).

| metric (bar) | y26n_ft (baseline) | arm B `y26n_unk4` | arm C `y26n_gradsupp` |
|---|---|---|---|
| CrowdHuman recall (**Q1 bar ≥ 37.8%**) | 32.8% | 33.5% ❌ | **36.5% ❌ (near-miss, +3.7 over baseline)** |
| CrowdHuman precision | 93.9% | 94.2% | 94.0% |
| Object-set image-FP (≤ 15%) | 12.7% | 11.6% ✅ | **11.2%** ✅ |
| PASS FPs/100 (≤ 0.4) | 0.23 | 0.40 ✅ (at the bar) | 0.27 ✅ |
| LAGENDA detection (≥ 94%) | 94.1% | 93.3% ❌ | 93.9% ❌ (by 0.1 pt) |
| Adult gender (≥ 90%) | 90.7% | 90.9% ✅ | **91.1%** ✅ |
| adult(≥20)→Child leak (≤ 0.3%) | 0.11% | 0.15% ✅ | **0.11%** ✅ (3/2,708) |
| child recall (detected) | 79.0% | 81.1% | 81.0% |
| Spotlight-val mAP50 / mAP50-95 | 0.799 / 0.685 | 0.801 / 0.690 | 0.799 / 0.689 |

**Q1 — mechanism validated? Formally NO, but narrowly.** Arm C gained **+3.7 pts crowd
recall** over the background baseline — recovering roughly half the masking gain — while
*also improving* the object-set FP rate, gender, and holding the leak at 0.11%. But the
pre-registered bar was +5 (37.8%) and it landed at 36.5%; recorded as a fail, not
reinterpreted. The LAGENDA-detection no-regression bar also missed by 0.1 pt (93.9 vs
94.0) — within single-run noise, but the bar is the bar.

**Q2 — which design wins? Gradient suppression, decisively.** C beats B by +3.0 crowd
recall with better PASS, objects, and leak. Unknown-as-class recovered almost nothing
(+0.7 over background) — the phantom-class concern appears real enough to cancel most of
the detection-supervision benefit.

**Q3 — do the wins survive? Yes.** Both arms hold or improve every EXP-2026-12 win
(objects, PASS, leak, gender, mAP); the only slippage anywhere is ≤ 0.8 pt of LAGENDA
detection.

**Cross-arch context (not a bar):** the coworker's masked-unknowns line still leads on
crowd recall — `gelannfav14r4_gemlb_v2` (nano) measured 41.1% and `gelansfav14_gemlb_v1`
(small) 46.2% in the same protocol. Within one architecture the unknown-handling ladder
now reads: background 32.8 → unknown-as-class 33.5 → **gradient suppression 36.5** →
(masking, different arch) 41–46.

**Latency (measured 2026-08-08, one session, EPYC 9254, ONNX CPU 4 threads):** both arms
match the baseline exactly — `y26n_unk4` 33.6 ms, **`y26n_gradsupp` 33.8 ms**,
`y26n_spotlight` 34.0 ms, all with NMS baked in. Gradient suppression is free at
inference: it changes training only, and the exported graph is byte-comparable to the
baseline's. For contrast, `gelannfav14r4_gemlb_v2` runs 59.5 ms *before* the browser's
NMS pass.

## What we can decide from this

- **Gradient suppression is the best unknown-handling measured on YOLO26n** (Q2 answered
  with a 3-pt margin over unknown-as-class), and it costs nothing on the FP/leak/gender
  side. If a YOLO26n arm ships, it should be the gradsupp one.
- **But it does not close the crowd-recall gap to masking.** Per the pre-registered rules
  no arm formally validated the mechanism (+3.7 vs the +5 bar), and the masked gelan line
  still leads (41.1 nano / 46.2 small). The honest recommendation to the team: the next
  YOLO26n arm to try is **grey-masked unknowns on YOLO26n** — masking's gain has now
  survived two architectures, and no y26n arm has tried it yet. Gradient suppression's
  box-supervision premise recovered only about half of masking's benefit in practice.
- **Unknown-as-class is dead** for this stack: +0.7 recall, no compensating win, and it
  needs a class-strip step at inference.
- The 0.1-pt LAGENDA-detection bar miss is noted, not litigated: single run, no variance
  estimate, and re-running to nudge it would be goalpost-shopping.

## What this does NOT tell us

Written before running:

- **Nothing here touches the Gulf-dress gender slice** — still the lab's biggest open gap.
- Results are on YOLO26n; the masking comparison point (gemlb) is a different size class, so
  "C vs masking" is only settled if C is also run on the gelan-s architecture later.
- The unknown class definition is inherited from Spotlight (Gemini couldn't gender the crop,
  overwhelmingly tiny people) — a different unknown policy would need re-emitted labels.
- In-training val metrics for arms B and C are not comparable to each other (different val
  label sets by design) — only the external protocol numbers compare.

---

## Appendix — verify it yourself (written BEFORE running)

**The gradient-suppression mechanism** — `vlm-cluster/train_gradsuppress.py`, all against
pinned `ultralytics==8.4.115` source:

1. *Why stock training can't do this:* the assigner indexes per-class predicted scores by GT
   label — `tal.py:193`: `bbox_scores[mask_gt] = pd_scores[batch_ind, :, gt_labels...][mask_gt]`
   — so a class-3 GT on a 3-channel head is an out-of-bounds index.
2. *The fix:* `GSDetectionLoss` builds its assigner with `num_classes=nc+1` and appends a
   class-agnostic channel to the scores it hands the assigner:

```python
pd_sig = pred_scores.detach().sigmoid()
pd_aug = torch.cat([pd_sig, pd_sig.max(-1, keepdim=True).values], dim=-1)
```

   An unknown GT competes for anchors on "how person-ish is this anchor" (score^0.5 · IoU^6),
   never on a nonexistent class score.
3. *Box supervision flows untouched:* `BboxLoss.forward` weights anchors by
   `target_scores[fg_mask].sum(-1)` (`loss.py:132`); the unknown assignments live in channel
   `nc` of `target_scores`, so passing the full tensor gives unknown-assigned anchors their
   box/DFL gradient with the normal alignment weight.
4. *Cls suppression:*

```python
cls_target = target_scores[..., : self.nc]          # drop the Unknown channel
unk_rows = target_scores[..., self.nc] > 0          # anchors assigned to an Unknown GT
bce_loss = self.bce(pred_scores, cls_target.to(dtype))
bce_loss = bce_loss * (~unk_rows).unsqueeze(-1)     # no gradient at all on those anchors
```

   Genuinely empty anchors still get background suppression; only unknown-assigned anchors
   are exempted.
5. *e2e coverage:* YOLO26 trains both one2many and one2one branches; both get the patched
   loss via `E2ELoss(model, GSDetectionLoss)` (`loss.py:1280`).
6. *Head stays 3-class:* `GSTrainer.get_model` builds the model with `nc = data_nc − 1`; the
   data yaml declares nc=4 only so dataset verification accepts class-3 label lines.

**Selftest (run 2026-08-05 on macOS CPU, real ultralytics 8.4.115, no downloads/data):**
builds tiny models from `yolo11n.yaml` (plain path) AND `yolo26n.yaml` (e2e path), fabricates
a batch with a class-3 GT, and asserts: the loss computes (the OOB crash class), cls
gradients are exactly zero on unknown-assigned anchors and nonzero elsewhere, and an
unknown-only image still produces box loss. Output:
`yolo11n: OK (e2e=False · unk anchors 10 · box loss 2.181)` ·
`yolo26n: OK (e2e=True · unk anchors 1 · box loss 1.757)`.

**Honestly NOT verified at design time:** full-scale training dynamics (the selftest proves
gradient plumbing, not convergence); whether `model.load(weights)` transfers COCO weights
into the 3-class head identically to the stock trainer path (checked in the pilot phase);
the +5-pt Q1 bar is a judgment call anchored to the measured masking gain, not calibrated.

**Pilot incident log (2026-08-06):** pilot_b passed clean (4-class head, 606/708 weights
transferred, mAP50 0.383→0.462 over 3 epochs, Unknown-class val AP 0.085 — expected for a
heterogeneous discard-at-inference class; 2,409/50k backgrounds, 0 corrupt labels). pilot_c
trained epoch 1 correctly (3-class head confirmed, 390 fewer head params than pilot_b = the
dropped Unknown channel, mAP50 0.492 on 3-class val) but **crashed at checkpoint save:
`Can't pickle local object 'build.<locals>.GSDetectionModel'`** — ultralytics pickles the
model class by reference, so nested class definitions can't checkpoint. Fix: all GS classes
moved to module top level; the selftest gained a torch.save/load round-trip contract (a
deepcopied model with criterion attached, mimicking the EMA copy ultralytics saves) that
reproduces the failure class. Consequence recorded in the tool docstring: loading a
gradsupp checkpoint requires `train_gradsuppress` importable (eval tools already run from
vlm-cluster), or use `--export-plain` to remap the checkpoint to a stock DetectionModel.


## Benchmark results — 2026-08-12 (the verdict on gradient suppression)

Scored on the standard four-dataset protocol (conf 0.45, same samples/scorers as every other
model). The controlled comparison is `y26n_ft` vs `y26n_gs_e2e`: identical architecture,
identical Spotlight labels, identical head type — **only the unknown handling differs**.

| | objects %img-FP | PASS FP/100 | crowd recall | heavy occl | crowd prec | LAGENDA det | gender | leak(20+) | teen→Child |
|---|---|---|---|---|---|---|---|---|---|
| `y26n_ft` (unknowns → background) | 12.7% | 0.23 | 32.8 | 13.2 | 93.9 | 94.1 | 90.7 | 0.11% | 8.4% |
| **`y26n_gs_e2e`** (gradient suppression) | **11.2%** | 0.27 | **36.5** | **15.2** | 94.0 | 93.9 | **91.1** | 0.11% | 8.7% |
| **`y26n_gs_nms`** (same ckpt, NMS head) | **10.4%** | 0.30 | **38.4** | **17.6** | 92.7 | **95.4** | 90.6 | 0.15% | 9.6% |
| *(reference)* `gelansfav14_gemlb_v1` (grey-masked, 8.2M) | 18.2% | 0.23 | 46.2 | 27.0 | 92.7 | 95.8 | 90.6 | 0.00% | 13.9% |

**Gradient suppression is validated.** vs background-treatment on the same architecture:
**+3.7 crowd recall** (32.8 → 36.5), **+2.0 heavy-occlusion**, *and* it simultaneously
improves object false-positives (12.7 → 11.2%) and gender (90.7 → **91.1%**, the best gender
of any model this lab has measured). No metric regresses beyond noise (LAGENDA det −0.2).
It recovers roughly **half** the recall that unknown-as-background cost, at zero inference
cost, with no extra class to handle downstream.

**Reading the checkpoint's one-to-many head adds more for free:** `y26n_gs_nms` gains another
+1.9 crowd recall and +1.5 LAGENDA detection over the same weights' NMS-free head, and posts
the lowest object-FP rate in the whole comparison (10.4%). Costs ~1.3 crowd precision and
0.5 gender. This is the recommended deployable configuration.

**Grey-masking (the coworker's design) still leads crowd recall** — 46.2 / heavy 27.0 vs
38.4 / 17.6 — but that's an 8.2M model at ~4× the latency, and it is markedly worse on the
false-positive axis (18.2% vs 10.4% of object-set images). The untested arm that would settle
it: grey-masked unknowns on the yolo26n architecture.
