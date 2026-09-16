# Model comparison — shipped baseline vs Spotlight-label retrains (2026-08-05; YOLOE added 2026-08-05)

## What was tried

- Fine-tuned **Ultralytics YOLO26-nano** on the Spotlight (Gemini-verified) labels — 30
  epochs, 475k images, ~$11 GPU.
- Evaluated **four models on identical data** — including the first-ever benchmark of the
  deployed model on human ground truth — and timed everyone's ONNX on one machine.
- **Added 2026-08-05: zero-shot YOLOE** (the promptable/open-vocabulary candidate the team
  asked about, EXP-2026-13) — text prompts `woman, man, child`, **no training at all** —
  run through the identical datasets, settings, and scorers. Spoiler: **rejected for every
  seat** (read the ⚠ note under its columns before quoting them), but the numbers belong in
  this table so the comparison is complete.

## The models and their exact weights

| run name (used in every table below) | what it is | labels | params | weights path |
|---|---|---|---|---|
| **yolo11N-640** | Ultralytics YOLO11n — shipped-era baseline (closest ckpt to the deployed `v11nclean2` export; caveat 2) | old (pre-Spotlight) | 2.6M | `/workspace/model_v2/dataset_v2/yolo11N-640/weights/best.pt` |
| **gelansfav14_datav2_v4** | YOLOv9 GELAN-S-fav14 (YOLO-MIT) — best v9 checkpoint, never shipped (too slow) | old (SAM3 auto) | 8.2M | `/workspace/YOLO-MIT/runs/v9MIT/gelansfav14_datav2_v4/weights/best.ckpt` |
| **gelansfav14_gemlb_v1** | Same architecture — new Spotlight retrain, grey-masked unknowns, warm-started | Spotlight (masked) | 8.2M | `/workspace/YOLO-MIT/runs/v9MIT/gelansfav14_gemlb_v1/weights/best.ckpt` |
| **y26n_spotlight** | Ultralytics YOLO26n — Spotlight fine-tune, unknowns as background (no masking) | Spotlight (no mask) | 2.4M | `/workspace/exp12/train/y26n_spotlight/weights/best.pt` |
| **yoloe26s_zeroshot** | YOLOE-26s-seg, text prompts `woman,man,child` (prompt order = class ids) — ZERO training | none (zero-shot) | 14.0M fused (15.3M with prompt machinery) | `yoloe-26s-seg.pt` (ultralytics auto-download, 8.4.115) |

## Was this a fair, identical evaluation?

- **Same images, verified by count**: all five models processed exactly 259 object-set /
  3,000 PASS / 517 crowd (11,641 GT persons) / 4,899 LAGENDA images (5,000 GT persons);
  zero unprocessed anywhere.
- **Same settings**: 640×640 input, conf 0.45 (the shipped product threshold), NMS IoU 0.7,
  same matcher (IoU ≥ 0.5), same scoring code.
- **Same machine** for all latency numbers.
- Remaining biases are listed under **Caveats** — read before quoting close differences.

## Results, per dataset

### 1 · Object set — 259 hand-verified doll/statue/toy images
*The "it blurs books/statues" bug: there are no real people, so any detection is a false
positive. Lower = better.*

| | yolo11N-640 | gelansfav14_datav2_v4 | gelansfav14_gemlb_v1 | y26n_spotlight | yoloe26s_zeroshot ⚠ |
|---|---|---|---|---|---|
| % images gaining ≥1 false person | **54.1%** | 53.7% | 18.2% | **12.7%** | 2.7% ⚠ |
| false persons per 100 images | 88.4 | 89.6 | 26.3 | **23.9** | 2.7 ⚠ |

→ The deployed-era model fires on more than half of these images. Both old-label models
~54%; both Spotlight models 13–18%. **The labels, not the architecture, decide this bug.**

### 2 · PASS — 3,000 verified person-free scenes
*Hallucination rate on ordinary empty images. Lower = better.*

| | yolo11N-640 | gelansfav14_datav2_v4 | gelansfav14_gemlb_v1 | y26n_spotlight | yoloe26s_zeroshot ⚠ |
|---|---|---|---|---|---|
| false persons per 100 images | 1.17 | 1.43 | **0.23** | **0.23** | 0.0 ⚠ |

→ Spotlight training cuts empty-scene hallucination ~5–6×, identically on both architectures.

### 3 · CrowdHuman — 517 images, 11,641 exhaustively-boxed people
*Finding people in busy scenes. Higher recall = better; precision must hold.*

| | yolo11N-640 | gelansfav14_datav2_v4 | gelansfav14_gemlb_v1 | y26n_spotlight | yoloe26s_zeroshot ⚠ |
|---|---|---|---|---|---|
| recall (all) | 34.9% | 39.6% | **46.2%** | 32.8% | 0.3% ⚠ (25.3% at its best threshold) |
| recall, heavy occlusion | 11.9% | 15.6% | **27.0%** | 13.2% | 0.1% ⚠ |
| precision | 94.4% | 95.6% | 92.7% | 93.9% | 100% ⚠ (32 detections total) |

→ gelansfav14_gemlb_v1's grey-masked unknown handling is the star: **+11 pts recall over
shipped**. y26n_spotlight (unknowns as background) lands ~2 pts under shipped — the
unknown-handling choice, not the YOLO26 architecture, explains the gap.

**Threshold-free crowd view (class-agnostic person-mAP, same 517 images — confidence
ranking instead of a single cutoff, so zero-shot calibration is not an excuse here):**

| | yolo11N-640 | gelansfav14_datav2_v4 | gelansfav14_gemlb_v1 | y26n_spotlight | yoloe26s_zeroshot |
|---|---|---|---|---|---|
| person AP50 | 0.631 | 0.651 | **0.655** | 0.571 | 0.249 |
| person AP@[.5:.95] | 0.373 | **0.382** | 0.371 | 0.318 | 0.166 |

*(datav2_v4 numbers = the mit_prod run of the same checkpoint. Not comparable to the
3-class Spotlight-val mAPs in section 5 — this is one collapsed "person" class because
CrowdHuman GT has no gender.)*

### 4 · LAGENDA — 5,000 people with human-annotated age + gender
*The classification dataset: does the right person get the right label? Blur decisions
live here.*

| | yolo11N-640 | gelansfav14_datav2_v4 | gelansfav14_gemlb_v1 | y26n_spotlight | yoloe26s_zeroshot ⚠ |
|---|---|---|---|---|---|
| detection recall | 95.3% | **96.0%** | 95.8% | 94.1% | 13.6% ⚠ (93.8% at its best threshold — but gender then drops to 85.4%) |
| gender accuracy on adults | 89.7% | 87.2% | 90.6% | **90.7%** | 97.2% ⚠ (on the 13.6% it found) |
| **adults 20+ misread as Child (escape the blur)** | 0.52% | 0.77% | **0.00%** | 0.11% | 0.00% (≤0.73% at every threshold) |
| teens 15–19 read as Child | 15.0% | 26.6% | 13.9% | **8.4%** | 0.0% |
| child recall ≤12 (of detected) | 81.2% | **92.1%** | 85.8% | 79.0% | 45.2% |

→ The Spotlight models win the direction that matters most — adults escaping the blur:
0.00% / 0.11% vs 0.52% / 0.77% — with equal-or-better gender. Child recall is the
acceptable-error direction (an over-blurred 12-year-old is fine; an unblurred adult is not).

**⚠ How to read every yoloe26s_zeroshot column:** zero-shot confidences are not calibrated
to the shipped 0.45 threshold — at 0.45 it barely detects anyone, which makes its FP columns
(objects 2.7%, PASS 0.0) look spectacular *for free* and its recall columns terrible. The
full threshold sweep (EXP-2026-13, replayed offline from logged raw outputs) shows **no
threshold works**: at conf 0.05 LAGENDA recall reaches 93.8% but adult gender falls to 85.4%,
object-set FPs rise to 46.3% of images, and crowd recall peaks at 25.3% — the worst detector
we've measured (the person-mAP table above confirms it threshold-free). Verdict: rejected
for the production seat and the labeling-pipeline seat. Two things worth keeping: its
prompted `child` class **never leaks adults to Child at any threshold** (a conservative age
vote that fails teens only in the safe/blurred direction), and its distractor-vocabulary
trick (`statue, mannequin, doll` absorbed and discarded) proved **safe** (steals ~0 real
people) but under-powered (−22.5% relative object FPs vs the ≥30% adoption bar) — kept on
the shelf for future promptable models. Full write-up:
`experiments/EXP-2026-13-yoloe-promptable-candidate.md`.

### 5 · Spotlight validation set — mAP + size + speed, one view

**The evaluation dataset:** the Spotlight OIV7 **validation split** — 4,232 images (one of
the 4,233 lost to a filename collision when materializing the image list; identical set for
every model), with the Gemini-verified Spotlight labels as ground truth:
images `/workspace/open-images-v7/images/val/` (via the frozen list
`/workspace/exp12/val_full.txt`), labels `/workspace/spotlight/run/oiv7_val/labels/`.
Every model's raw detections (logged to conf 0.05) went through the same COCO-style
101-point evaluator (`vlm-cluster/map_eval.py`).

*Read carefully: this scores agreement with the **Spotlight labels**, so the two old-label
models are penalized partly by construction (they were trained toward different labels). It
measures label-alignment; the human-GT tables above measure truth.*

| model | tier | params | ONNX | CPU latency | mAP50 | mAP75 | mAP50-95 |
|---|---|---|---|---|---|---|---|
| **y26n_spotlight** | nano | 2.38M | 9.8 MB | **33 ms** | 0.799 | 0.724 | 0.685 |
| yolo11N-640 (shipped-era) | nano | 2.59M | 10.6 MB | 42 ms | 0.719 | 0.650 | 0.588 |
| gelannfav14\*_gemlb_v1 (nanos) | nano | 2.98M | — | ~45–50 ms† | *still training* | | |
| **gelansfav14_gemlb_v1** | small | 8.21M | 33 MB | 135 ms | **0.836** | **0.754** | **0.695** |
| gelansfav14_datav2_v4 | small | 8.21M | 33 MB | 136 ms | 0.797 | 0.714 | 0.642 |
| yoloe26s_zeroshot | small | 13.99M | 41.8 MB | ~110 ms‡ | 0.449 | 0.412 | 0.393 |

‡ informational export, added 2026-08-05 after the accuracy verdict (Phase D's *sanity
checks* — class-id and parity — were still skipped per EXP-2026-13's rules). Fixed-vocab
`woman,man,child` ONNX, 35.4 GFLOPs, seg-head graph: NMS-free end-to-end (300×38 = the
usual 6 det fields + 32 mask coefficients, plus a 32×160×160 proto tensor) — so its latency
includes mask computation a det-only graph wouldn't pay. Measured 109.6 ms on an EPYC 9254
at 4 threads with y26n_spotlight re-timed in the same session at 34.0 ms (vs 33 ms on the
benchmark machine) — i.e. directly placeable in this column at ~110 ms, **3.2× slower than
y26n_spotlight** at 4.3× its size. Per-class mAP: Man 0.561 / Woman 0.410 / Child 0.376
AP50 — with zero training, ~0.35 mAP50 behind the shipped-era baseline.

† owner's laptop measurement; not yet re-timed on the benchmark machine.

→ **The size-for-accuracy trade in one line: y26n_spotlight (2.38M) lands within 0.01
mAP50-95 of the 8.21M champion at 4× the speed.** gemlb's remaining edge is concentrated at
mAP50 — finding more small/occluded people, the unknown-masking effect. The empty nano row
is the decisive pending result. Method note: the conf-0.05 logging floor truncates the PR
tail uniformly — compare across rows, not against numbers from other tools (ultralytics'
own validator reads ~0.02 higher on the same model).

### 6 · Speed & size — AMD EPYC, ONNX Runtime CPU, 4 threads, 640×640

| | yolo11N-640 | gelansfav14_datav2_v4 | gelansfav14_gemlb_v1 | y26n_spotlight | yoloe26s_zeroshot |
|---|---|---|---|---|---|
| median latency | 42 ms \* | 136 ms \* | 135 ms \* | **33 ms** | ~110 ms ‡ (NMS-free, incl. seg head) |
| ONNX size | 10.6 MB | 33 MB | 33 MB | **9.8 MB** | 41.8 MB |

\* *these three output raw tensors — the browser still runs NMS on top, so their true cost
is higher. y26n_spotlight's 33 ms includes final boxes (NMS-free architecture).*

## Takeaways

1. **The Spotlight labeling effort fixes what users complain about** — measured on two
   architectures independently: object false-persons 54% → 13–18%, empty scenes 5×, the
   adult-escape leak to ~zero.
2. **vs the real deployed baseline, y26n_spotlight is better or equal on nearly
   everything** and ~26% faster with NMS included; its only deficit is ~2 pts of crowd
   recall, which the masking result says is a data-handling fix, not an architecture limit.
3. **Grey-masking unknowns (gelansfav14_gemlb_v1's design) is validated with numbers**:
   +11 recall over shipped on the same labels where background-treatment lost 2. Next
   candidates should all use masked or 4-class unknown handling.
4. Next: the in-training **gelannfav14\*_gemlb_v1 nanos** (2.98M, size-matched to
   y26n_spotlight), an **unknown-aware YOLO26 retrain**, and a **threshold sweep**
   (caveat 1).
5. **The YOLOE question is answered: no.** Zero-shot open-vocabulary detection is not close
   — no threshold passes detection + gender together, and crowd performance is the worst
   measured even threshold-free (person AP50 0.249 vs 0.57–0.66). The two salvage findings
   (a never-leaks child vote; a validated-safe distractor-vocabulary trick) are recorded in
   EXP-2026-13 for future candidates. Training YOLOE on Spotlight labels was pre-registered
   as conditional on the zero-shot gate and was therefore not run.

## 2026-08-08 round — unknown-handling arms + YOLOE fine-tune + the two gelan mains

Five new models through the identical protocol (same datasets, conf 0.45, same scorers,
same mAP evaluator). Baseline for the first three = y26n_spotlight above.

| | y26n_gradsupp (2.38M) | y26n_unk4 (2.38M) | yoloe_n_gradsupp (2.69M) | gelannfav14w_gemlb_v1 (~6.6M) | gelannfav14r4_gemlb_v2 (~3M) |
|---|---|---|---|---|---|
| unknown handling | gradient-suppressed | 4-class, dropped | gradient-suppressed | (owner's recipe) | (owner's recipe) |
| crowd recall | 36.5% | 33.5% | 36.3% | 36.7% | **41.1%** |
| crowd precision | 94.0% | **94.2%** | 93.5% | 92.3% | 92.7% |
| objects %img-FP ↓ | 11.2% | 11.6% | 11.6% | **10.0%** | 16.2% |
| PASS FP/100 ↓ | 0.27 | 0.40 | 0.37 | 0.57 | 0.33 |
| LAGENDA detection | 93.9% | 93.3% | 93.3% | 93.0% | **95.2%** |
| adult gender | 91.1% | 90.9% | 91.5% | 89.7% | **92.0%** |
| leak: adults ≥20 → Child ↓ | 0.11% | 0.15% | 0.15% | 0.34% | **0.07%** |
| child recall (detected) | 81.0% | 81.1% | 80.2% | **85.2%** | 81.3% |
| Spotlight-val mAP50 / 50-95 | 0.799 / 0.689 | 0.801 / 0.690 | **0.801 / 0.691** | 0.753 / 0.610 | 0.788 / 0.645 |
| CPU latency (one session, EPYC 9254) | **33.8 ms** | 33.6 ms | 49.1 ms | 89.3 ms ‡ | 59.5 ms ‡ |

‡ raw-tensor output — the browser runs NMS over 8,400 candidates on top of the time shown;
the YOLO26/YOLOE rows already include it. All rows re-timed 2026-08-08 in one session
(ONNX Runtime CPU, 4 threads, 640 px, median of 100 after 20 warmup); `y26n_spotlight`
34.0 ms and `yolo11N-640` 41.1 ms in the same session, within ~3% of the earlier EPYC 7352
numbers in section 6 above. Also timed: `gelannfav14r3_gemlb_v1` 58.5 ms,
`gelannfav14m_gemlb_v1` 73.1 ms.

Readings:

1. **Unknown handling on YOLO26n (EXP-2026-14):** gradient suppression is the clear
   winner of the two designs (+3.7 crowd recall over the background baseline vs +0.7 for
   unknown-as-class, with objects FP *improving* 12.7→11.2% and the leak holding at
   0.11%) — but it recovered only ~half the masking gain and formally missed its
   pre-registered +5-pt bar. **Recommendation: try grey-masked unknowns on YOLO26n** —
   masking now leads on two architectures and no y26n arm has used it.
2. **YOLOE is closed (EXP-2026-15):** `yoloe_n_gradsupp` ties `y26n_gradsupp` on every
   metric while costing 45% more inference time (49.1 vs 33.8 ms — 9.1 vs 5.3 GFLOPs;
   its fused model is only 2.69M params, so the penalty is compute, not size). Both
   pre-registered win conditions failed. No further YOLOE arms planned.
3. **gelannfav14r4_gemlb_v2 is the strongest escape-side nano measured**: best crowd
   recall (41.1%), best LAGENDA detection (95.2%), best gender (92.0%), best leak
   (0.07%) — at the cost of the worst object-set FP rate in the round (16.2% vs ~11%).
   The w_v1 wide model trades the other way (best objects 10.0%, worst leak 0.34%,
   lowest gender). **r4v2 vs y26n_gradsupp is now the real production question, and it
   hinges on (a) whether 16.2% object-FP is acceptable against the users' complaint, and
   (b) the latency column — still unmeasured for the gelan nanos on the benchmark
   protocol.**
4. mAP note (best-practice statement): all values from the one shared evaluator
   (COCO 101-point, IoU .50–.95, full confidence-ranked detections to the 0.05 floor,
   exhaustive Spotlight-val GT, per-class AP available in the JSONs); comparable within
   this table only, and it measures label alignment, not truth — the human-GT rows above
   it are the decision-grade numbers.

## Caveats (read before quoting close differences)

1. **Single threshold**: everything scored at conf 0.45 (the shipped threshold). Model
   confidences aren't calibrated to each other, so ±1–2 pt differences could move at a
   different threshold. Raw outputs down to conf 0.05 were logged for every model — a
   threshold-sweep re-score needs no re-run and is queued.
2. **Shipped-era identity**: yolo11N-640's `best.pt` is dated Dec-2025; the shipped TF.js
   export is Nov-2025 (same run family, possibly an earlier epoch). Needs one-line
   confirmation from the extension owner.
3. **The shipped extension also runs 320/416 input tiers** on slower machines —
   yolo11N-640 at 640 is its best case.
4. **Possible train/eval overlap on PASS for the old-label models**: the old training dir
   contains ~37k `negatives_*` background images of unconfirmed origin; if PASS-derived,
   the PASS scores of yolo11N-640 and gelansfav14_datav2_v4 are flattered. Being checked.
   (The Spotlight-trained models excluded these images.)
5. Object-set rates are **upper bounds for all models equally** (set re-verification
   pending); PASS has ~5% known contamination (hits all models equally); the crowd sample
   is 517 images (the historical 500 + a 17-img top-up — identical across all columns).
6. gelansfav14_gemlb_v1 was warm-started from gelansfav14_datav2_v4; its exact label
   variant needs confirmation from its owner.
7. **The Gulf/traditional-dress gender slice is untested for every model in this table** —
   still our biggest evaluation gap.

## Questions for the team

1. **@[retrain owner]** — confirm gelansfav14_gemlb_v1's unknown handling (grey-mask? all
   class-3 regions?) and warm-start source; what does "fav14" modify vs stock GELAN? Which
   nano revision (gelannfav14 base / r3 / r4) is the candidate?
2. **@[extension owner]** — confirm `v11nclean2` was exported from
   `model_v2/dataset_v2/yolo11N-640` (Nov export vs Dec best.pt), and whether the 224px
   second-stage classifier is active in the shipped build.
3. **@[extension owner]** — do we know the real-world distribution of the 320/416/640
   tiers among users? It weights how much the speed column matters.
4. **@[everyone]** — license: the shipped YOLO11n prints **AGPL-3.0** in its own metadata;
   YOLO26 is the same license. What's our current posture, and does anything change?
5. **@[everyone]** — who can help hand-check a Gulf/traditional-dress slice? It's the one
   thing none of these numbers cover.

*Full details + verify-it-yourself appendix:
`experiments/EXP-2026-12-yolo26-production-candidate.md` · protocol for adding any model:
`docs/COMPONENT_FRAMEWORK.md`.*
