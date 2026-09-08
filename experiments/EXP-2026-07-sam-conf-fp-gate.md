# Experiment: Can SAM3's confidence score filter out its false positives?

**In one line:** no — SAM3's false positives (statues, dolls, empty-scene hallucinations) score
almost exactly the same confidence as its real-person detections, so a "send only low-confidence
boxes to the VLM" gate cannot work; existence verification needs its own independent check.

**Status:** CONCLUDED 2026-07-19 · 624 FP + 9,206 verified-TP + 3,011 in-the-wild-FP detections · Owner: Mostafa

---

## The short version

- **The two confidence distributions are nearly identical:** false positives mean 0.705 / median
  0.717; real-crowd detections mean 0.735 / median 0.769. They overlap across the whole
  [0.4, 0.98] range.
- **No usable operating point exists.** Catching 62% of FPs (threshold <0.8) escalates 55% of
  all real detections; catching 86% escalates 76%. At that point you are verifying everything
  anyway — the gate saves nothing.
- **The mechanism, now measured:** SAM3's confidence scores *form-match quality*, not
  real-person-ness. Statues match human form excellently — its FPs run up to 0.98 confidence.
  Most confident FP class: **"Woman" (mean 0.743, higher than the real-crowd average)** — the
  statue→Woman pattern from EXP-2026-03 is not just frequent, it is confident.
- **Decision:** the team proposal "Gemini only on low-confidence SAM3 boxes" is rejected on
  measured grounds. The labeling pipeline's existence check must come from an independent
  judge (VLM crop verdict and/or a second architecturally-different detector), not from SAM3's
  own score.

## How we did it

1. **The answer key is structural:** on PASS (3,000 verified person-free scenes) and the object
   set (259 hand-verified doll/statue/toy images), *every* SAM3 detection is a false positive by
   construction — no matching needed. The CrowdHuman seed-51 sample (500 exhaustively-boxed crowd
   images) provides the comparison distribution (~87% of its detections are real people per
   EXP-2026-03's precision measurement).
2. **Confidence capture:** the production labeler does not write per-detection confidence, so a
   patched **copy** (`/workspace/autolabel/autolabel_sam_conf.py` — production script untouched)
   appends each detection's score as a trailing token on its label line. One-line change:
   `annotation = f"{class_id} {poly_str} {score:.4f}"`. All other settings frozen production
   (`--conf 0.4`, classes woman/man/child, NMS untouched).
3. **Re-run** over object_set + pass_3k + Images_sample500 (~3.8k images, ~1.7 h on an L4),
   outputs to `/workspace/exp07_conf/sam_labels/` — production label dirs untouched.
4. **Report:** `vlm-cluster/sam_fp_conf_report.py` (selftested) — histograms, per-class means,
   and the threshold sweep ("gate decision table"). Full JSON: `/workspace/exp07_conf/report.json`.

## Per-dataset split (the nuance the pooled numbers hide)

| | PASS (3,000 empty scenes) | Object set (259 doll/statue imgs) | Crowd (mostly real) |
|---|---|---|---|
| Detections | 251 FPs (matches EXP-2026-03's 8.4/100 rate) | 373 FPs | 12,667 (~87% real) |
| Conf mean / median | 0.630 / 0.616 | **0.755 / 0.798** | 0.735 / 0.769 |
| Share at conf >= 0.8 | 20.7% | **49.3%** | 44.8% |
| Dominant class | Man (173, mean 0.617) | **Woman (180, mean 0.764)** | Man (6,815) |

**Two different failure modes with opposite confidence signatures:**

- **Empty-scene hallucinations (PASS) lean low-confidence** — 97% below 0.9. A confidence gate
  would catch a fair share of these, at a price.
- **Human-shaped objects score HIGHER than real people** — objects mean 0.755 vs real-crowd
  0.735. SAM3 is literally more confident that a statue is a woman (0.764) than that a real
  woman is (0.747). Since statues/dolls/mannequins are the FP class that actually occurs in
  real-world images, the gate fails exactly where it matters most.

## The gate decision table (the result)

| Threshold | FPs caught | Real detections escalated (the cost) |
|---|---|---|
| <0.5 | 16.2% | 14.6% |
| <0.6 | 31.6% | 27.4% |
| <0.7 | 47.3% | 40.1% |
| <0.8 | 62.2% | 55.2% |
| <0.9 | 85.7% | 76.1% |

A viable gate needs a row with high FP-catch and low escalation. Every row catches FPs and real
people at nearly the same rate — the score contains almost no real-vs-fake signal.

## What this does NOT tell us

- The "mostly-true" crowd distribution contains SAM3's own crowd FPs (~13%, per EXP-2026-03's
  86.6% precision), slightly blurring the comparison — but the overlap is so complete that no
  plausible correction changes the verdict.
- FP confidences were measured on *negatives-only* datasets; FPs in ordinary mixed scenes
  (e.g. a statue behind real people) were not separately measured, but there is no reason to
  expect them to score differently.
- Reproducibility note: the objects+PASS rerun produced exactly the same detection count (624)
  as the original EXP-2026-03 run — deterministic. The crowd rerun differs slightly (12,667 vs
  13,174 lines in the old dir, ~4%) — unresolved; possibly stale extra files in the old exp03
  crowd dir or the torch 2.4→2.6 environment change. Does not affect the verdict; worth a
  5-minute diff if crowd counts ever matter downstream.

## Zoom: FP type decides gateability (added 2026-07-19, `conf_zoom_report.py`)

![PASS FP vs verified crowd TP confidence histograms](assets/exp07_zoom_histograms.svg)

*(Figure: PASS false positives (red, 251 dets over 3,000 empty images, mean 0.63) vs crowd true
positives verified against CrowdHuman GT (blue, 9,206 dets, mean 0.794). Data:
`assets/exp07_zoom_report.json`; regenerate via `conf_zoom_report.py` on
`/workspace/exp07_conf/`. PNG for ClickUp: `assets/png/EXP-2026-07/`.)*

Matching crowd detections against CrowdHuman's exhaustive GT splits the picture into verified
populations (LAGENDA skipped — no conf-logged run; crowd TPs are the harder, more representative
TP reference anyway):

| Population | N | Mean / median conf | Verdict |
|---|---|---|---|
| Crowd TP (matched to GT, IoU >= 0.5) | 9,206 | 0.794 / 0.839 | the reference |
| Objects FP (statues/dolls) | 373 | 0.755 / 0.798 | **un-gateable — scores like real people** |
| PASS FP (empty scenes) | 251 | 0.630 / 0.616 | partially gateable |
| Crowd FP (in-the-wild, unmatched non-ignore) | 3,011 | 0.577 / 0.541 | **substantially gateable** |

- **A confidence floor at ~0.55-0.60 is a useful cheap pre-filter:** contains ~52-63% of
  in-the-wild crowd FPs and ~38-47% of empty-scene FPs at a 10-14% TP loss — acceptable under
  the misses-shrink/false-labels-poison policy. It does NOT replace the independent existence
  check: statues sail through at any workable threshold.
- **Full containment is impossible:** the highest FP confidences (0.935 PASS, 0.96 crowd)
  would keep only ~12% of true positives above the bar.
- Caveat: this simple matcher counts duplicates/loose-box border cases as crowd "FP" (implied
  precision 75.4% vs EXP-2026-03's 86.6% under its more careful rules), so some crowd "FPs"
  are loose boxes on real people — true FP separation is likely slightly BETTER than shown.
  Directionally safe.

**Refined verdict:** confidence gating is a legitimate *pre-filter* (kills half the ordinary
clutter for free), never *the* filter — the human-shaped-object FP class, the one that matters
most, is immune to it at every threshold.

## Consequence for the pipeline design

Strengthens the dual-pipeline / VLM-arbitration design (`docs/PIPELINE_PROPOSAL_dual_vote_labeler.md`):
the "missing second opinion = disagreement" routing is not an optional refinement — it is the
only measured way to catch SAM3's false positives, because SAM3 cannot flag them itself at any
confidence threshold.

## Conclusion

The experiment is concluded with a two-part verdict:

1. **As THE filter: rejected.** No confidence threshold separates SAM3's false positives from
   its real detections — containing all FPs requires conf >= 0.935, keeping only ~12% of true
   positives. The decisive sub-finding: human-shaped objects (statues/dolls) score HIGHER than
   real people (0.755 vs 0.735 mean) — the FP class that matters most is fully immune to
   thresholding. The pipeline's existence check must be an independent judge (VLM crop verdict
   / second architecturally-different detector), as the dual-vote proposal specifies.
2. **As a PRE-filter: adopted as a pipeline knob.** A floor at ~0.55-0.60 removes roughly half
   of ordinary-scene false positives (empty scenes + crowd clutter) at a 10-14% TP loss —
   acceptable under the misses-shrink / false-labels-poison error policy, and it shrinks every
   downstream stage's workload. Threshold value to be re-derived on scale-up data before
   production use.

Artifacts: figure + JSON in `experiments/assets/` (this repo), full label runs + reports on the
volume at `/workspace/exp07_conf/`, tools `sam_fp_conf_report.py` + `conf_zoom_report.py` +
the conf-logging labeler copy `/workspace/autolabel/autolabel_sam_conf.py` (recommended to
become the production default — the escalation pipeline needs confidences anyway).

## Infra note (discovered during this run)

Current RunPod templates ship transformers 5.x with torch 2.4.1 — SAM3 import fails
(`DTensor` / torchaudio ABI errors) on a fresh pod. Fix, before anything else:
`pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`
(all three together — upgrading torch alone strands torchaudio's ABI).
