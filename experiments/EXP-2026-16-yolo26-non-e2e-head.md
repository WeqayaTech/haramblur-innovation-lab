# EXP-2026-16 — Does training YOLO26n *without* the NMS-free head produce a better NMS detector?

**In one line:** YOLO26 trains two heads at once and de-weights the conventional one 8× over
training — so we train a pure one-to-many YOLO26n (no one2one branch, no loss schedule) and
ask whether the dense head we'd actually deploy comes out better than the one we already have.

**Status:** RUN COMPLETE 2026-08-12 — **bar MISSED (+0.3 vs +0.5 required); not adopted.**
The pure one-to-many model is marginally better, not meaningfully better. · Owner: Mostafa

---

## Why

The team wants a browser-deployable model in the **current production output format** (raw
`(1, 7, 8400)`, NMS in JS). YOLO26's default is NMS-free. Investigating how to get the raw
format surfaced a real mechanism in `ultralytics/utils/loss.py` (verified in installed
source, 8.4.115/8.4.118):

- YOLO26 trains with **`E2ELoss`**, which weights the two branches on a schedule:
  one-to-many **0.8 → 0.1** linearly over epochs (`final_o2m = 0.1`), one-to-one 0.2 → 0.9.
- But `Detect.forward` **detaches** the features feeding the one-to-one head
  (`# detach keeps one2one out of the backbone`), so o2o gradients never reach the backbone,
  neck, or dense head. The trunk is trained *solely* by the (decaying) o2m loss.

A colleague's hypothesis was that this leaves the dense head under-trained. **Measured first,
before assuming** — the existing `y26n_gradsupp` checkpoint, both branches, Spotlight val
(4,233 imgs):

| branch | mAP50 | mAP50-95 | P | R |
|---|---|---|---|---|
| e2e (NMS-free) | 0.827 | 0.711 | 0.811 | 0.740 |
| **o2m + NMS** | **0.833** | **0.715** | 0.803 | **0.761** |

So the dense head is *not* damaged — it's the better branch (+0.6 mAP50, and **+2.1 pts
recall**, the axis this product cares about), matching Ultralytics' published +0.6-0.8 gap.
The deployable model therefore already exists; **this experiment asks the narrower question:
can a pure one-to-many run do even better?**

Mechanisms that could make it better: (a) the o2m loss trains at weight 1.0 throughout
instead of decaying to 0.1; (b) `best.pt` is selected on **o2m fitness** rather than e2e
fitness. Reason for doubt: training used **AdamW**, which normalizes updates by gradient
magnitude and is therefore ~invariant to a constant loss scaling — so the decay may matter
far less than it looks.

## Pre-registered bar (written before the run finished)

**Adopt only if the non-e2e run beats `y26n_gradsupp`'s o2m branch — 0.833 mAP50 /
0.715 mAP50-95 on Spotlight val — by ≥ 0.5 pt.** Otherwise the existing checkpoint's raw
export is the deliverable and this arm is recorded as a negative result.

**Comparability caveat, stated up front:** this arm trains on the classes-0-2 labels with
**no unknown handling**, whereas `y26n_gradsupp` uses the gradient-suppression loss. A win
here is therefore not automatically a better production model — the clean head-to-head would
need the non-e2e YAML run through `train_gradsuppress.py` (verify `GSTrainer.get_model`
honors `end2end: False` first).

## Setup (verified before launch)

Config: `/workspace/exp16/yolo26n_noe2e.yaml` = stock `yolo26.yaml` with the single key
`end2end: True → False`. `parse_model` forwards it to the head; the head then skips building
`one2one_cv2/cv3`; `init_criterion` falls through to plain `v8DetectionLoss` (no `E2ELoss`,
no schedule).

Pre-flight assertions (all passed): head class `Detect`, `hasattr(head,'one2one_cv2')` False,
`head.end2end` False, criterion `v8DetectionLoss`, `Transferred 588/588` from COCO weights.
1-epoch smoke test on the 50k subset then confirmed the **trainer** preserves it —
`one2one present after training: False`, `names {0:Woman,1:Man,2:Child}`, model summary line
`Detect [3, 1, False, ...]`.

Training: `YOLO(yaml).load('yolo26n.pt')`, 30 epochs, imgsz 640, `batch=-1`, workers 12,
ultralytics pinned **8.4.115** (same as the comparison arm), data
`/workspace/exp12/spotlight_oiv7_local.yaml` (images staged to local NVMe — the network
volume is ~5× too slow).

Useful side-finding: **fused, the non-e2e model is 2,375,421 params — identical to the fused
e2e model**, because fusion strips whichever head is unused. Deployment size and latency are
therefore the same either way; only the output contract differs.

## Results — 2026-08-12

30 epochs, RTX 4090 + Ryzen 9 7950X (16 vCPU, workers 12), images on local NVMe:
**10.4 h** wall clock at 17.7 it/s (vs 23.1 h for `y26n_gradsupp` on 8 vCPU — the CPU upgrade
roughly halved training time; dataloading was the bottleneck, as suspected).

| model | mAP50 | mAP50-95 | P | R |
|---|---|---|---|---|
| `y26n_gradsupp`, o2m branch (the bar) | 0.833 | 0.715 | 0.803 | **0.761** |
| **`y26n_noe2e`** (pure one-to-many training) | **0.836** | **0.717** | **0.812** | 0.760 |
| delta | +0.3 | +0.2 | +0.9 | −0.1 |

Per class mAP50 — Woman 0.864→0.867, Man 0.881→0.880, Child 0.754→**0.761**.

**Verdict: bar MISSED (+0.3 / +0.2 vs the ≥0.5 pt requirement) → not adopted.** The
deliverable stays the `end2end=False` export of the existing `y26n_gradsupp` checkpoint.

**What this answers.** Training YOLO26 without the one-to-one head *is* mechanically possible
(no public precedent found) and *is* marginally better for the dense head — but by ~0.25 pt,
not by the amount the "dense head is starved by the 0.8→0.1 schedule" hypothesis predicts.
The most likely reason it's so small: training used **AdamW**, which normalizes updates by
gradient magnitude and is therefore ~invariant to a constant scaling of the loss, so
de-weighting the o2m term never shrank its effective step size the way it would under SGD.
The detach (o2o gradients never reach the trunk) already ruled out the "trunk optimized for
the wrong head" mechanism.

**Two honest confounds** (both push toward the non-e2e arm being flattered, which strengthens
the negative result):
1. This arm trained on classes-0-2 labels with **no unknown handling**, while the comparison
   model has the gradient-suppression loss. Despite lacking that, it still only tied — i.e.
   the non-e2e head advantage is roughly the same size as whatever gradsupp contributes here.
2. `batch=-1` auto-selected slightly different batch sizes on the two runs (19,801 vs 21,601
   iterations/epoch ⇒ ~24 vs ~22 images/batch), a small uncontrolled difference.

**Four-benchmark protocol — run anyway 2026-08-12, and it STRENGTHENS the negative verdict.**
Scored against the same LAGENDA / CrowdHuman / PASS / object-set protocol as every other model
(conf 0.45), `y26n_noe2e` loses to simply reading the NMS head of the existing
`y26n_gradsupp` checkpoint:

| | objects %img-FP | PASS FP/100 | crowd recall | heavy | crowd prec | LAGENDA det | gender | leak(20+) |
|---|---|---|---|---|---|---|---|---|
| `y26n_gs_nms` (free — same ckpt, other head) | **10.4%** | 0.30 | **38.4** | 17.6 | 92.7 | 95.4 | 90.6 | 0.15% |
| `y26n_noe2e` (10 GPU-hours) | 13.1% | 0.30 | 35.7 | 16.6 | 92.9 | 95.4 | 90.6 | 0.07% |

Worse on object false-positives (+2.7 pts) and crowd recall (−2.7 pts), tied elsewhere. The
dedicated retrain produced a model beaten by a zero-cost export flag on a checkpoint we
already had — because it lacks the gradient-suppression unknown handling, and that matters
more than the head configuration. **Conclusion: the non-e2e training path is closed. Use
`export(..., end2end=False)` on a gradsupp checkpoint instead.**

## Notes

- **No public precedent:** nothing found (docs, GitHub issues, papers) of anyone training
  YOLO26 non-e2e. `end2end` is documented as a predict/val/export argument only and is absent
  from the train-settings table; no non-e2e YAML ships. Either outcome is a new datapoint.
- `val(end2end=False)` on an e2e checkpoint fails with `KeyError: 'feats'` — `fuse()` strips
  the one2many head first. Set `model.model[-1].end2end = False` **before** calling val.
- Full mechanism notes + the TF.js export path live in `docs/TFJS_EXPORT_RECIPE.md`.
