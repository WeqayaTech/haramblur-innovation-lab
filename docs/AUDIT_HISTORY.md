# Audit & design history (detailed)

Read `CLAUDE.md` first for orientation. This file is the detailed record of every audit and
design decision made before EXP-2026-01, organized by track. Read the section relevant to what
you're picking up — you don't need all of it.

**Important caveat:** most of the outputs described below live **only on the RunPod network
volume**, not in this local repo (pods are ephemeral, the volume persists — see `CLAUDE.md` →
Infrastructure). Paths are given from memory of prior sessions; **verify they still exist**
before assuming they do (`find /workspace -iname "<name>"`).

---

## Track 2 — Label verification audits (on Open Images val/train data)

### Val-set 3-way audit
Tools: `compare_classes.py`, `compare_gender.py`, `compare_child.py` (all in `vlm-cluster/historical/`).
Compare **Label vs production YOLO-MIT prediction vs VLM description** on Open Images val boxes.
`compare_gender.py` scopes to Woman/Man boxes (gender axis only); `compare_child.py` scopes to
Child boxes (age axis only); `compare_classes.py` does full 3-class. Each buckets outcomes into
`model_error` / `label_error` / `vlm_disagrees_both` / `uncertain` and generates montages.

**Finding:** confirmed the production model's known weakness — **men in Gulf/traditional dress
(ghutra, thobe) get misread as Woman.**

### Hard-negative deep dive
Tools: `describe_hardneg.py` (describes the model's *confident* misclassifications from a
`disagreements.jsonl` produced by the val eval) → `annotate_focus.py` (adds a red focus box to
old whole-image descriptions without re-running the VLM) → `analyze.py` / `cluster.py` (finds
what attributes characterize each confusion pair, e.g. `Man_as_Woman`).

**Finding:** `Man_as_Woman` errors correlate with long hair + no visible beard; separately,
ghutra/thobe + beard cases also misfire — i.e. **two different failure modes**, not one.

### VLM self-contradiction check
Tool: `vlm_contradictions.py`. Flags records where the VLM's own `apparent_gender` contradicts
its own attire/facial-hair read in the *same* response (e.g. `gender=woman` but `facial_hair=
full_beard`). Purpose: sanity-check whether a flagged "label error" is trustworthy, or whether
the VLM shares the model's own bias.

**Correction made:** `turban` was **removed** from `WOMAN_CONTRADICTORS` — on inspection, every
flagged case was a woman correctly wearing a turban (both genders wear turbans; it's not a
reliable masculine signal). Only `ghutra_keffiyeh` (specifically male Arab headwear) was kept.

### Training-set Child label audit
Pod location: `data_inspection_tools/vlm-cluster/child_train_audit/` — `manifest.jsonl` (4,957
randomly sampled Child-labeled training boxes), `descriptions.jsonl` (VLM blind description of
each), `report.py` (generates `disagreements.html`).

**Finding: training label quality is good.** 93.1% VLM agreement with the Child label, 5.5%
ambiguous (mostly teenagers — a real definitional edge, not VLM failure), only **1.3% (66 boxes)**
flagged as likely label errors. Conclusion: the Child class's 5.8% share of training data (vs
63.9% Man) — severe class imbalance — is the real problem, not label noise.

Then ran production YOLO-MIT against those same 4,957 boxes (`run_model_on_manifest.py`, built by
importing `MitModel`/`match_boxes`/`find_model_config` from `run_model_children.py` rather than
duplicating logic) for a full 3-way Label vs Model vs VLM comparison:
- Of the 66 flagged label errors: **49 cases** have model AND VLM both saying "adult" (strongest
  evidence — the model was trained on the Child label yet still predicts adult independently).
  **12 cases** only the VLM catches it (model also fooled — shares the label's bias).
- `report_3way.py` — HTML evidence report for the 49 vs 12 split, full image + crop + VLM field
  table per case.
- `report_ambiguous.py` — HTML report for the 274 ambiguous-age cases, with an age-band histogram
  to distinguish genuine teen ambiguity from the VLM simply refusing to answer.
- `verify_raw_labels.py` — draws boxes **directly from the raw YOLO `.txt` label files** (bypasses
  our manifest entirely) to confirm loose/off-target boxes are a genuine dataset issue, not a
  pipeline artifact. Confirmed: yes, some GT boxes are loose (e.g. only covering a hat).

### Missing images
Several training label files (Arabic/Middle-Eastern-named images: `khaleeji_woman`, `bedouin`,
`sheikh_galabiya`, etc.) have labels but **no corresponding image on disk** — flagged as a
separate dataset-integrity issue, not yet resolved.

---

## Track 1 — Next-gen auto-labeling pipeline (DESIGNED, NOT YET BUILT)

### Current production auto-labeler: `autolabel_sam.py`
Not in this repo (lives alongside the scraped image dirs, not yet located precisely in the local
checkout — `find` for it at pod start). Runs **Meta SAM3** with **one text prompt per class**
("woman" / "man" / "child"), keeps detections above `conf=0.4` (later runs used `0.7`), then does
**cross-class NMS on mask IoU > 0.7** — the highest-scoring class-prompt wins each person.

**Problems identified (root-cause hypothesis for several downstream issues):**
1. **Class assignment is a coin-flip.** SAM3's per-prompt confidence scores aren't a calibrated
   classifier — comparing "child: 0.62" vs "woman: 0.58" on a petite adult is arbitrary.
2. **Cross-class NMS can let the same person survive under two classes** if two prompts produce
   masks with IoU in the 0.5–0.7 range (common — hair/clothing edges differ per prompt).
3. **`mask_to_polygon` concatenates ALL contours from a fragmented mask into one polygon.** When
   a person is partially occluded, SAM3 returns multiple mask fragments; naive concatenation
   produces a self-crossing polygon whose bounding box can span a mostly-empty region — the likely
   cause of the loose/hat-only boxes found in the training audit above.
4. **Inconsistent settings across scrape runs**: `conf=0.4` default vs `0.7` for one directory;
   one invocation (`dataset_arab_traditional`) had a **broken `--output` flag**
   (`--output --"/workspace/.../dataset_arab_traditional/"`), which argparse likely rejects —
   meaning that directory's images may have **zero labels** (silently treated as background by
   YOLO training). **Partially verified 2026-07-09** (EXP-2026-02 Phase 1): the live line in
   `/workspace/autolabel/run_autolabel.sh` contains BOTH malformed flags — `--output
   --"/workspace/image_scraper/dataset_arab_traditional/"` and `--conf0.7` (no space) — so that
   command crashes argparse as written. Whether the run was later hand-corrected interactively,
   and whether that directory has labels on disk, is **still unchecked**. The three
   `dataset_clothing` runs (`arab_men`, `african_men`, `arab_women_abaya`) used the `conf=0.4`
   default, prompts `"woman" "man" "child"`, `--batch_size 1`, `--overwrite`.
5. **Unconfirmed hypothesis — possible direct bias injection:** the scraped bias-mitigation dirs
   (`dataset_clothing/arab_men`, `african_men`, `arab_women_abaya`, `dataset_arab_traditional`)
   were labeled with the SAME flawed demographic-prompt SAM3, on exactly the imagery where the
   Shaykh→Woman confusion is strongest. If SAM3's "woman" prompt fired on `arab_men` images, the
   production model would have **learned the bias from its own training labels**, not just
   inherited it from Open Images. Verification commands were drafted but **not yet run**:
   ```bash
   grep -rl "^0 " labels/arab_men | wc -l    # how many arab_men images got a 'woman' class-0 label
   ls labels/arab_men | wc -l                 # vs total labeled images
   ```

### Proposed replacement pipeline (design only — no code written yet)
1. **Stage 0 — detect only.** SAM3 with a single neutral prompt `"person"` — no class voting, so
   the coin-flip and cross-class-NMS problems disappear entirely. Fix `mask_to_polygon` to keep
   only the largest contour (or emit one polygon per contour) instead of concatenating.
2. **Stage 1 — build dual input.** For each detected person: full image with the box drawn +
   padded crop (matches the "blind read, full context" method later validated in EXP-2026-01).
3. **Stage 2 — classify via ensemble.** An on-pod open VLM (Gemma 4, see model notes below) +
   a specialist (MiVOLO V2 — numeric age + gender from face *and* body, works without a visible
   face) [+ optionally a frontier API model as a third vote / escalation tier].
4. **Stage 3 — fusion gate.** Ensemble agrees → auto-accept. Disagrees or abstains → escalate to
   a stronger model. Still unresolved → human review queue.
5. **Stage 4 — write labels.** YOLO format from the fused verdict + Stage-0 polygon.

**Deploy order suggested:** re-label the small scraped bias dirs first — they're where the
current labeler is most likely wrong and where the Shaykh-bias-injection hypothesis lives.

### Model landscape research (mid-2026, verified via live web search — training-data knowledge
of "current" models was stale and had to be corrected)
| Model | Type | Role considered |
|---|---|---|
| **Qwen3.7-Plus** | Alibaba, API-only, vision | cheap second VLM vote (~$0.40/$1.60 per 1M tok), different lab bias than Gemini |
| **Gemma 4** (12B/26B/31B) | Google, **open weights**, Apache 2.0 | on-pod, privacy-preserving anchor vote — direct upgrade path from the Qwen2.5-VL-7B describer |
| **Gemini 3.5 Flash / Pro** | Google API | accuracy anchor / escalation tier; tops Roboflow Vision Evals; pilot for child-image refusal rate before committing |
| **LocateAnything-3B** | NVIDIA, open, **non-commercial license** | text-prompted grounding (replaces GroundingDINO) — box verification / Stage-0 alternative, not a classifier |
| **MiVOLO V2** (WildChlamydia) | specialist, open | SOTA age/gender, has a dedicated **minor-vs-adult** head — directly on-target for our decision |

**Key ensemble-diversity lesson:** vendor diversity isn't enough — the val-set audit showed
Qwen2.5-VL-7B shares the Shaykh/attire bias with the production model (both trained on similar
web-scale data). **Architectural diversity matters more**: mixing a VLM with a face/body
specialist (MiVOLO) and a grounding model (LocateAnything) protects against correlated failure
in a way that three different VLM vendors might not.

### Supervisor feedback (shaped Track 3 / the three-pipeline split)
Core point: **can't select or validate a label-fixing model using labels we don't already
trust** (circular). Proposed fix: use **externally human-annotated benchmarks** — LAGENDA
(on-domain, from Open Images, numeric age+gender) and MSP60K / WIDER (OpenPAR framework — messy
long-tail: blur, occlusion, back-views, closer to real production input) — for **model
selection**, and reserve our own hand-checked slices for the two things public benchmarks
structurally can't cover: our specific attire bias, and our specific box quality. Also suggested
adding the MiVOLO minor/adult model to the ensemble. **EXP-2026-01 is the first executed instance
of this pipeline**, using LAGENDA + Qwen2.5-VL-7B (Gemma 4 / MSP60K / WIDER comparisons are not
yet run — candidate for EXP-2026-02).

### Three-pipeline framing (adopted, sent to supervisor)
1. **Auto-labeling pipeline** — image in, YOLO labels out. Cheap, high-volume. (Track 1 above.)
2. **Label verification pipeline** — image + existing label in, per-label accuracy report out.
   More expensive per item; run over *existing* data to find mislabels. (Track 2 above.)
3. **Verification-accuracy / calibration pipeline** — run pipeline 2's engine against trusted
   external GT to know how much to trust it, before trusting it on our own data. (Track 3 /
   EXP-2026-01.)

### Cost estimates discussed (order-of-magnitude, not measured)
- **Auto-labeling at scale**: ~$0.0015/image compute (~$1,400–1,800 per 1M images), dominated by
  the escalation-tier API calls (Gemini), not the on-pod GPU cost. Human-review cost is a policy
  choice, not a compute constraint — can equal or exceed compute cost if you review 100% of the
  disagreement queue instead of spot-checking it.
- **Full-dataset max-accuracy verification**: ~$6,000–12,000 for ~1M people — rarely justified.
  **Scoping to just the Child class + the scraped bias dirs** (where we already have evidence of
  problems) is ~$750–1,500 and captures most of the value. Always scope verification by risk, not
  run it wholesale.

---

## Track 4 — Data distribution / diversity analysis (parallel thread, not about label correctness)

Two independent, non-overlapping approaches — neither has been re-run recently as of EXP-2026-01:

### CV-only: `dataset-diversity-audit/diversity_audit.py`
One-time spike. No VLM needed (numpy/opencv core; optional CLIP zero-shot for two dimensions).
Crops every Person box and measures: **skin tone** (ITA° → Fitzpatrick-style band), **scale**
(bbox area ÷ image area, a distance proxy), **capture quality** (brightness/sharpness/resolution),
and optionally **view angle** + **dress code** via CLIP zero-shot prompts. Reports per-dimension
distribution + imbalance score (normalized entropy, Gini) → one rolled-up "diversity confidence"
number + ranked under-represented buckets. Saves **debug montages per bucket** so estimates can be
eyeballed (CLIP/skin-tone are not deterministic — always verify visually). Deliberately does NOT
infer ethnicity/nationality (unreliable + sensitive) — skin tone is the diversity proxy instead.
Full usage in `dataset-diversity-audit/README.md`.

### VLM-description-based: `vlm-cluster/field_report.py`, `analyze.py`, `collection_needs.py`
Run **after** `describe.py` has produced rich per-person JSON (age, gender, attire, skin tone,
pose, orientation, etc. — the same schema used throughout this project).
- `field_report.py` — per-field distribution + imbalance score, plus cross-tabs against `class`
  (e.g. class × facial_hair) ranked by sparsity, each linked to a montage.
- `analyze.py` — the general, no-hardcoded-slices version: every field, every pairwise
  combination, ranked sparsest-first. Broader but shallower than `field_report.py`.
- `collection_needs.py` — turns descriptions into a plain-English "shopping list": named,
  multi-attribute slices (e.g. "dark-skin children", "men in Gulf/Arab attire") each with a
  COLLECT/ok verdict and a verification montage. `TARGET_SLICES` is hand-edited to match what we
  care about — this is the most actionable of the three.

These three tools share `cluster.py`'s `flat()` / `load_records()` / `montage()` helpers with the
rest of `vlm-cluster/` — same JSONL format as `describe.py` output, so any existing description
run (child_train_audit, hardneg, or a fresh one) can be fed into any of them.
