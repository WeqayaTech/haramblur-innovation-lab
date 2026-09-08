#!/usr/bin/env python3
"""
EXP-2026-10 — end-to-end validation of Pipeline v1 (mask-highlight labeler).

The pipeline under test (docs/PIPELINE_V1_mask_highlight_labeler.md):
  SAM3 (frozen labels, already on disk) detects and provides mask polygons →
  each detection becomes a 25%-padded crop with ITS OWN mask highlighted
  (semi-transparent tint + outline, context preserved) → Gemini 3.5
  Flash-Lite returns one JSON verdict (real-person gate, gender, age,
  optional box correction) → merge rules produce the final label:
  rejected detections deleted, age > 12 → adult, sanity-checked box
  corrections applied.

Run per dataset, scored per dataset (never pooled):

  # Stage B: build highlighted crops + call Flash-Lite (resumable — rerun to continue)
  python3 pipeline_v1_eval.py run \
      --images /workspace/lagenda_eval/lagenda_yolo/images/val \
      --sam-labels /workspace/lagenda_eval/sam_autolabel/labels \
      --max-images 500 --seed 42 --out /workspace/exp10/lagenda

  # Stage C: merge + score against GT (no API calls; rerunnable)
  python3 pipeline_v1_eval.py score --run-dir /workspace/exp10/lagenda \
      --gt lagenda --gt-labels /workspace/lagenda_eval/lagenda_yolo/labels/val \
      --manifest /workspace/lagenda_eval/lagenda_yolo/gt.jsonl

  python3 pipeline_v1_eval.py score --run-dir /workspace/exp10/crowd \
      --gt crowdhuman --odgt /workspace/datasets/crowdhuman/annotation_val.odgt
  python3 pipeline_v1_eval.py score --run-dir /workspace/exp10/pass --gt none

  # QC arm: same crops WITHOUT highlighting (bias check), then compare
  python3 pipeline_v1_eval.py run ... --plain-crops --out /workspace/exp10/lagenda_plain
  python3 pipeline_v1_eval.py compare --run-a <highlighted>/verdicts.jsonl \
      --run-b <plain>/verdicts.jsonl

  python3 pipeline_v1_eval.py selftest      # no network, no data, stub engine

Reuses (does NOT duplicate): select_images + _safe_call from vlm_detect_eval,
extract_json from describe, match_boxes/iou from run_model_children,
load_odgt from eval_negatives_crowd, _rate + DEFAULT_CLASS_NAMES from
run_autolabel_on_manifest, engine construction from api_describers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from describe import extract_json
from eval_negatives_crowd import IGNORE_IOA, _ioa, load_odgt
from run_autolabel_on_manifest import DEFAULT_CLASS_NAMES, _rate
from run_model_children import iou, match_boxes

PAD = 0.25                    # crop padding — the validated EXP-2026-08 geometry
TINT_RGBA = (0, 200, 80, 90)  # highlight tint (green, ~35% alpha)
OUTLINE_RGB = (0, 220, 90)
OUTLINE_DARK = (0, 70, 20)    # underlay so the outline reads on any background
OUTLINE_W = 3
BOX_MIN_IOU = 0.3             # box_correction sanity: identity-swap guard
BOX_MAX_AREA_RATIO = 3.0      # box_correction sanity: bounded grow/shrink
# v1 POLICY (owner decision 2026-07-26): SAM3's geometry ALWAYS wins in
# production. Lite's box_correction is still logged verbatim on every verdict
# (raw-readings principle) but never applied. Measured basis: under the
# correct yxyx_1000 parse Lite wins 60% of head-to-heads yet regresses the
# mean (0.752 vs SAM3 0.787), and an agreement-gated variant nets only
# ~+0.002 mean IoU pipeline-wide. Set True only with fresh-sample validation.
APPLY_BOX_CORRECTIONS = False
# Final Child only if estimated_age <= this. Kept at 12 = the production class
# boundary (owner decision 2026-07-26). A sweep showed cutoff 9 would drive the
# strict leak metric (GT age > 12 labeled Child) to 0.00%, but that metric is
# the wrong instrument here: LAGENDA ages are HUMAN APPARENT-AGE annotations
# (±2-3 years), and every one of the 14 "leaks" at cutoff 12 was a GT age of
# 13-17 — i.e. annotator-vs-model disagreement inside the noise band, not an
# adult escaping the blur. ZERO GT-18+ people were labeled Child at cutoff 12.
# The product-meaningful metric is therefore "GT >= 18 labeled Child" (see
# leak_by_age_band in the experiment doc), and lowering the cutoff would only
# trade harmless teen disagreement for real children (8 of 32) being blurred.
CHILD_AGE_MAX = 12

PROMPT_VERSION = "spotlight-v2"   # v1 = lean verdict; v2 adds analysis fields

# Analysis fields: filter-only payload for dataset analysis — NEVER read by
# the merge rules. Vocabularies align with describe.py OBJECT_SCHEMA where
# they overlap so Track 4 tools consume this output directly.
ANALYSIS_KEYS = ["apparent_race", "occlusion", "occlusion_percent",
                 "face_visible", "orientation", "pose",
                 "exposed_body_parts", "clothing_fit",
                 "garment_type", "head_covering", "facial_hair",
                 "gender_cues", "skin_tone_mst", "skin_tone_confidence"]

PROMPT = """\
You are verifying ONE detection for a person-labeling dataset. In this image
exactly one candidate is MARKED with a green outline (and possibly a light
green tint). The outline may consist of several separate parts when the
person is partially hidden — all outlined parts mark the SAME single person.
Judge ONLY the marked candidate; ignore every other person or object.

Rules for what counts as a person:
- A real human being of any age counts, even partially visible or occluded.
- A printed or photographic depiction of a real human (poster, billboard,
  magazine, photo-in-photo) COUNTS as a person.
- A doll, mannequin, statue, sculpture, toy, cartoon, or drawing is NOT a
  person.
- Judge gender from the face and visible physical cues, not from clothing
  style or robes. Use "unknown" ONLY if no cue is visible at all.

Return EXACTLY this JSON, nothing else:
{{"verdict": "real_person" or "depiction" or "not_person",
"gender": "man" or "woman" or "unknown",
"age_group": "child" or "adult" or "unknown",
"estimated_age": <single best numeric age guess>,
"box_correction": [x1, y1, x2, y2] or "ok",
"highlight_quality": "good" or "covers_wrong_object" or "covers_multiple_people",
"verdict_confidence": "high" or "low",
"apparent_race": "white" or "black" or "east_asian" or "southeast_asian" or "south_asian" or "central_asian_turkic" or "middle_eastern_north_african" or "hispanic_latino" or "other" or "unknown",
"occlusion": "none" or "partial" or "heavy",
"occlusion_percent": <0-100, how much of the person is hidden>,
"face_visible": "yes" or "no",
"orientation": "frontal" or "three_quarter" or "profile" or "back",
"pose": "standing" or "sitting" or "walking" or "running" or "lying" or "other" or "unknown",
"exposed_body_parts": ["none"] or a list from: "face", "hair", "neck", "shoulders", "arms", "hands", "chest", "midriff", "back", "legs", "knees", "feet",
"clothing_fit": "loose" or "fitted" or "tight" or "unknown",
"garment_type": "thobe_robe" or "cloak_bisht" or "abaya" or "dress" or "shirt_trousers" or "suit" or "sportswear" or "swimwear" or "other" or "unknown",
"head_covering": "none" or "cap_hat" or "hijab" or "niqab" or "ghutra_keffiyeh" or "turban" or "helmet" or "other" or "unknown",
"facial_hair": "none" or "stubble" or "moustache" or "short_beard" or "full_beard" or "unknown",
"gender_cues": "<short text: which visible cues drove your gender answer>",
"skin_tone_mst": <Monk Skin Tone 1-10, 1 lightest .. 10 darkest> or "unknown",
"skin_tone_confidence": "reliable" or "uncertain_lighting"}}

Details: "child" means the marked person appears 12 years old or younger.
"box_correction" is the tight bounding box of the marked person in PIXEL
coordinates of THIS image (it is {W} pixels wide and {H} pixels tall) —
give it if the outline fits the person poorly, otherwise answer "ok".
The attribute fields describe the marked person for dataset analysis: give
your best estimate, using "unknown" only when truly not visible.
"apparent_race" is a coarse visual grouping judged from overall facial
features and skin tone — the 7 FairFace groups (white, black, east_asian,
southeast_asian, south_asian = FairFace "Indian"/Indian subcontinent,
middle_eastern_north_african = FairFace "Middle Eastern" incl. Arab and
North African, hispanic_latino) plus "central_asian_turkic" for Central
Asian and Turkic peoples (Kazakh, Uzbek, Kyrgyz, Turkmen, Uyghur, Turkish),
who otherwise fall between east_asian and middle_eastern_north_african. It
is an appearance guess from the pixels, never a claim about nationality,
culture, or identity — answer "unknown" when the face is not visible enough
to tell, "other" only if no group fits.
"exposed_body_parts" lists every body part from the given vocabulary that
shows visible SKIN (not covered by clothing); answer ["none"] only when no
listed part shows skin. This is an objective observation, not a judgment.
"clothing_fit" describes whether the clothing hangs loose or clings to the
body outline. If the person appears to be a minor, still fill these fields
factually but never describe their body beyond them.
Do not identify or name anyone.
"""


# --- production prompt (spotlight-e1) -------------------------------------
# The trimmed 9-field prompt that /workspace/autolabel_pipeline_v2 ships.
# Selectable with --prompt e1 so it can be A/B'd against the validated v2
# using the same scorers. Kept as an import so there is ONE definition.
def _load_e1():
    try:
        import spotlight_run
        return spotlight_run.PROMPT, spotlight_run.ANALYSIS_KEYS, "spotlight-e1"
    except Exception as exc:                    # noqa: BLE001
        raise SystemExit(f"--prompt e1 needs spotlight_run.py alongside: {exc}")


# ---------------------------------------------------------------------------
# SAM label parsing — polygons kept (seg_boxes only returns extents)
# ---------------------------------------------------------------------------

def seg_polys(label_file: Path, w: int, h: int):
    """[(cls, poly_px[(x,y),..] or None, box_xyxy_px)] — polygon lines keep
    their points for mask rendering; plain cxcywh lines get poly=None."""
    if not label_file.exists():
        return None
    out = []
    for line in label_file.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            vals = [float(x) for x in parts[1:]]
        except ValueError:
            continue
        if len(vals) == 4:
            cx, cy, bw, bh = vals
            box = [(cx - bw / 2) * w, (cy - bh / 2) * h,
                   (cx + bw / 2) * w, (cy + bh / 2) * h]
            out.append((cls, None, box))
        else:
            if len(vals) % 2 == 1:        # tolerate a trailing conf column
                vals = vals[:-1]
            xs, ys = vals[0::2], vals[1::2]
            if len(xs) < 3 or len(xs) != len(ys):
                continue
            poly = [(x * w, y * h) for x, y in zip(xs, ys)]
            box = [min(xs) * w, min(ys) * h, max(xs) * w, max(ys) * h]
            out.append((cls, poly, box))
    return out


# ---------------------------------------------------------------------------
# crop construction — the mask-highlight rendering
# ---------------------------------------------------------------------------

def outline_loops(size, shifted):
    """Split a (possibly bridge-stitched) YOLO polygon into one clean outline
    loop per CONNECTED PART of the person's mask.

    Why: SAM3 masks of occluded people have disconnected parts, but the YOLO
    segment format stores one polygon per person, so the labeler stitched the
    parts together with thin bridge retraces. Rasterizing the polygon and
    taking external contours after a 3x3 morphological opening recovers the
    parts (bridges are <=2px and vanish); each part gets its own loop.
    Falls back to the raw single loop if rasterization degenerates (tiny or
    malformed polygons)."""
    import cv2
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).polygon(shifted, fill=255)
    arr = np.array(m)
    # radius-3 opening: dissolves stitching bridges up to ~6px wide. (3x3
    # left 3-4px bridge slivers alive as standalone thin components that
    # drew as long diagonal lines — observed in trace3; and upscaled crops
    # scale the bridges up too.) Over-splitting is benign for rendering —
    # an extra loop around a limb still outlines the person.
    opened = cv2.morphologyEx(arr, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    if not opened.any():          # opening erased a very small person
        opened = arr
    cnts, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    scored = []
    for c in cnts:
        a = cv2.contourArea(c)
        p = cv2.arcLength(c, True)
        scored.append((a, p, c))
    loops = []
    if scored:
        a_max = max(a for a, _, _ in scored)
        for a, p, c in scored:
            if a < 9:
                continue
            # bridge-remnant filter (largest part always kept): drop
            # non-largest components that are negligible or long-and-thin —
            # mean width 2*area/perimeter under 2.5px is a sliver's
            # signature, never a real body part at crop scale
            if a < a_max and (a < 0.01 * a_max
                              or (p > 0 and 2 * a / p < 2.5)):
                continue
            pts = [(int(q[0][0]), int(q[0][1])) for q in c]
            if len(pts) >= 3:
                loops.append(pts + [pts[0]])
    if loops:
        return loops
    raw = [(int(x), int(y)) for x, y in shifted]
    return [raw + [raw[0]]]


def _draw_loops(crop, loops):
    """Two-tone loop drawing: dark underlay then bright green on top —
    visible on any background."""
    d = ImageDraw.Draw(crop)
    for lp in loops:
        d.line(lp, fill=OUTLINE_DARK, width=OUTLINE_W + 2)
    for lp in loops:
        d.line(lp, fill=OUTLINE_RGB, width=OUTLINE_W)


def _draw_outline(crop, shifted):
    """Legacy path (frozen YOLO labels): recover parts from the bridged
    single polygon heuristically, then draw. When RAW per-part polygons are
    available (autolabel_sam_raw.py sidecars) build_crop draws them directly
    and this function is not used."""
    _draw_loops(crop, outline_loops(crop.size, shifted))


def load_raw_dets(raw_dir: Path, stem: str):
    """Load a raw-mask sidecar written by autolabel_sam_raw.py:
    <stem>.json = {"image", "width", "height", "detections": [{"cls": int,
    "conf": float, "box": [x1,y1,x2,y2] px, "parts": [[[x,y],...], ...] px}]}.
    Returns (dets, parts_list) with dets shaped like seg_polys output
    ([(cls, None, box)]) or (None, None) if no sidecar."""
    f = raw_dir / f"{stem}.json"
    if not f.exists():
        return None, None
    rec = json.loads(f.read_text())
    dets, parts_list = [], []
    for d in rec.get("detections", []):
        dets.append((int(d["cls"]), None, [float(v) for v in d["box"]]))
        parts_list.append([[(float(x), float(y)) for x, y in part]
                           for part in d.get("parts", []) if len(part) >= 3])
    return dets, parts_list

def build_crop(img: Image.Image, box, poly, highlight=True, style="tint",
               min_side=0, parts=None):
    """Returns (crop_rgb_PIL, crop_origin_xy, sam_box_in_crop, scale).

    parts: RAW per-part mask polygons (from autolabel_sam_raw.py sidecars).
    When given, each part is drawn directly — exact, no rasterize/morphology
    heuristics — which is the production-runner rendering path. Without
    parts, falls back to the frozen-label path (single bridged YOLO polygon,
    heuristic part recovery).

    style: "tint" = semi-transparent fill + outline (pilot v1 — measured to
    obscure small/distant people); "outline" = mask outline only, zero pixels
    covered. min_side > 0 upscales small crops (LANCZOS) so the max side is
    at least min_side BEFORE drawing — distant people get more pixels at the
    same API cost. sam_box_crop is in final (scaled) crop coords; scale maps
    crop coords back to image coords (divide by it)."""
    x1, y1, x2, y2 = box
    pw, ph = (x2 - x1) * PAD, (y2 - y1) * PAD
    cx1, cy1 = int(max(0, x1 - pw)), int(max(0, y1 - ph))
    cx2, cy2 = int(min(img.width, x2 + pw)), int(min(img.height, y2 + ph))
    crop = img.crop((cx1, cy1, cx2, cy2)).convert("RGB")
    scale = 1.0
    if min_side and max(crop.size) < min_side:
        scale = min_side / max(crop.size)
        crop = crop.resize((max(1, int(crop.width * scale)),
                            max(1, int(crop.height * scale))),
                           Image.LANCZOS)
    sam_box_crop = [(x1 - cx1) * scale, (y1 - cy1) * scale,
                    (x2 - cx1) * scale, (y2 - cy1) * scale]
    if not highlight:
        return crop, (cx1, cy1), sam_box_crop, scale

    if parts:                     # RAW per-part masks — draw exactly
        shifted_parts = [[((px - cx1) * scale, (py - cy1) * scale)
                          for px, py in part] for part in parts]
        if style == "tint":
            overlay = Image.new("RGBA", crop.size, (0, 0, 0, 0))
            d = ImageDraw.Draw(overlay)
            for sp in shifted_parts:
                d.polygon(sp, fill=TINT_RGBA)
            crop = Image.alpha_composite(crop.convert("RGBA"),
                                         overlay).convert("RGB")
        loops = []
        for sp in shifted_parts:
            lp = [(int(x), int(y)) for x, y in sp]
            loops.append(lp + [lp[0]])
        _draw_loops(crop, loops)
        return crop, (cx1, cy1), sam_box_crop, scale

    shifted = ([((px - cx1) * scale, (py - cy1) * scale) for px, py in poly]
               if poly
               else [(sam_box_crop[0], sam_box_crop[1]),
                     (sam_box_crop[2], sam_box_crop[1]),
                     (sam_box_crop[2], sam_box_crop[3]),
                     (sam_box_crop[0], sam_box_crop[3])])
    if style == "tint":
        overlay = Image.new("RGBA", crop.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        d.polygon(shifted, fill=TINT_RGBA)
        crop = Image.alpha_composite(crop.convert("RGBA"),
                                     overlay).convert("RGB")
    _draw_outline(crop, shifted)
    return crop, (cx1, cy1), sam_box_crop, scale


# ---------------------------------------------------------------------------
# verdict parsing + merge rules
# ---------------------------------------------------------------------------

def blurriness(img: Image.Image, box):
    """Variance-of-Laplacian sharpness of the person's native-resolution
    region (computed BEFORE any highlight drawing or upscaling — those would
    contaminate/soften the measurement). Higher = sharper. Local + free +
    deterministic, which is why it is not asked of the model."""
    import cv2
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.width, x2), min(img.height, y2)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    gray = np.array(img.crop((x1, y1, x2, y2)).convert("L"))
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def parse_verdict(text: str):
    obj = extract_json(text)
    if not isinstance(obj, dict) or "verdict" not in obj:
        return None
    v = {
        "verdict": str(obj.get("verdict", "")).strip().lower(),
        "gender": str(obj.get("gender", "unknown")).strip().lower(),
        "age_group": str(obj.get("age_group", "unknown")).strip().lower(),
        "highlight_quality": str(obj.get("highlight_quality", "good")).strip().lower(),
        # team rename; legacy "confidence" accepted from pre-v2 records
        "verdict_confidence": str(obj.get("verdict_confidence",
                                          obj.get("confidence",
                                                  "unknown"))).strip().lower(),
    }
    for k in ANALYSIS_KEYS:               # filter-only payload, passed through
        val = obj.get(k)
        if isinstance(val, str):
            val = val.strip()
        elif isinstance(val, list):       # exposed_body_parts
            val = [str(x).strip().lower() for x in val if str(x).strip()]
        if k == "occlusion_percent" and val is not None:
            try:
                val = float(val)
            except (TypeError, ValueError):
                val = None
        v[k] = val
    try:
        v["estimated_age"] = float(obj.get("estimated_age"))
    except (TypeError, ValueError):
        v["estimated_age"] = None
    bc = obj.get("box_correction")
    if (isinstance(bc, (list, tuple)) and len(bc) == 4):
        try:
            v["box_correction"] = [float(x) for x in bc]
        except (TypeError, ValueError):
            v["box_correction"] = None
    else:
        v["box_correction"] = None       # "ok" or malformed -> keep SAM box
    return v


def merge(rec):
    """Apply the pipeline's merge rules to one verdict record. Returns
    {keep, final_class, box_final_img, box_corrected(bool)}. Policy
    (pre-registered): keep iff real_person/depiction; Child iff
    age_group==child AND estimated_age<=12 (>12 defaults to adult);
    committed VLM gender wins; gender unknown -> class 'Unknown'
    (droppable); box_correction accepted only behind sanity checks."""
    v = rec["v"]
    out = {"keep": v is not None and v["verdict"] in ("real_person", "depiction"),
           "box_final_img": rec["box_img"], "box_corrected": False}
    if not out["keep"]:
        out["final_class"] = None
        return out
    crop_scale = rec.get("crop_scale", 1.0)
    age = v["estimated_age"]
    if v["age_group"] == "child" and (age is None or age <= CHILD_AGE_MAX):
        out["final_class"] = "Child"
    elif v["gender"] == "man":
        out["final_class"] = "Man"
    elif v["gender"] == "woman":
        out["final_class"] = "Woman"
    else:
        out["final_class"] = "Unknown"
    bc = v["box_correction"] if APPLY_BOX_CORRECTIONS else None
    if bc is not None:
        cw, ch = rec["crop_wh"]
        x1, y1, x2, y2 = bc
        if x2 > x1 and y2 > y1 and 0 <= x1 and 0 <= y1 and x2 <= cw and y2 <= ch:
            sb = rec["sam_box_crop"]
            a_new = (x2 - x1) * (y2 - y1)
            a_old = max(1.0, (sb[2] - sb[0]) * (sb[3] - sb[1]))
            ratio = max(a_new / a_old, a_old / a_new)
            if iou(bc, sb) >= BOX_MIN_IOU and ratio <= BOX_MAX_AREA_RATIO:
                ox, oy = rec["crop_origin"]
                s = crop_scale
                out["box_final_img"] = [x1 / s + ox, y1 / s + oy,
                                        x2 / s + ox, y2 / s + oy]
                out["box_corrected"] = True
    return out


# ---------------------------------------------------------------------------
# Stage B runner
# ---------------------------------------------------------------------------

def run_stage_b(engine, images_dir: Path, sam_dir: Path, out_dir: Path,
                max_images: int, seed: int, highlight: bool,
                style: str = "tint", min_side: int = 0,
                only_ids: set | None = None, raw_dir: Path | None = None,
                max_spend: float | None = None, rate_in: float = 0.30,
                rate_out: float = 2.50, report_every: int = 50):
    from vlm_detect_eval import _safe_call, select_images
    out_dir.mkdir(parents=True, exist_ok=True)
    # provenance: the prompt is edited often, so version the CONTENT, not a
    # hand-maintained string. run_meta.json freezes everything needed to
    # reproduce/interpret this run; prompt_sha also lands on every record so
    # a mixed-prompt verdicts.jsonl is detectable after the fact.
    prompt_sha = hashlib.sha256(PROMPT.encode()).hexdigest()[:12]
    meta = {"prompt_version": PROMPT_VERSION, "prompt_sha": prompt_sha,
            "prompt_text": PROMPT, "model": engine.cost.model_id,
            "crop": {"pad": PAD, "style": style if highlight else None,
                     "min_crop_side": min_side, "highlight": highlight},
            "merge_thresholds": {"child_age_max": CHILD_AGE_MAX,
                                 "box_min_iou": BOX_MIN_IOU,
                                 "box_max_area_ratio": BOX_MAX_AREA_RATIO},
            "labels_source": str(raw_dir or sam_dir),
            "raw_masks": raw_dir is not None,
            "images": str(images_dir), "seed": seed,
            "max_images": max_images}
    mpath = out_dir / "run_meta.json"
    if mpath.exists():                     # resumed run: flag prompt drift
        prev = json.loads(mpath.read_text())
        if prev.get("prompt_sha") != prompt_sha:
            print(f"[run] *** WARNING: this run dir was started with prompt "
                  f"{prev.get('prompt_sha')} but the current prompt is "
                  f"{prompt_sha} — verdicts.jsonl will MIX prompts. Use a "
                  f"fresh --out dir unless you mean this. ***")
            meta["prompt_sha_previous"] = prev.get("prompt_sha")
    mpath.write_text(json.dumps(meta, indent=2))
    verdicts_path = out_dir / "verdicts.jsonl"
    done = set()
    if verdicts_path.exists():
        for line in verdicts_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
    paths = select_images(images_dir, max_images, seed)
    n_img = n_det = n_skip = 0
    stop_for_budget = False
    import time as _time
    t_start = _time.time()

    def _spend():
        """USD spent so far in THIS process (rates are assumptions until
        vendor-confirmed; token counts are exact)."""
        c = engine.cost
        return (c.input_tokens * rate_in + c.output_tokens * rate_out) / 1e6

    def _status(done):
        el = max(1e-6, _time.time() - t_start)
        rate = done / el * 60
        spent = _spend()
        st = {"detections_done": done, "images_done": n_img,
              "resumed": n_skip, "elapsed_min": round(el / 60, 1),
              "dets_per_min": round(rate, 1),
              "usd_spent": round(spent, 4),
              "usd_per_1k_dets": round(spent / done * 1000, 4) if done else None,
              "max_spend_usd": max_spend,
              "pct_of_budget": (round(100 * spent / max_spend, 1)
                                if max_spend else None),
              "input_tokens": engine.cost.input_tokens,
              "output_tokens": engine.cost.output_tokens,
              "api_errors": engine.cost.errors,
              "prompt_sha": prompt_sha, "model": engine.cost.model_id}
        (out_dir / "status.json").write_text(json.dumps(st, indent=2))
        return st
    with verdicts_path.open("a") as fh:
        for p in paths:
            confs = None
            if raw_dir is not None:      # raw sidecars: exact per-part masks
                dets, parts_list = load_raw_dets(raw_dir, p.stem)
                sc = raw_dir / f"{p.stem}.json"
                if dets is not None:
                    confs = [d.get("conf") for d in
                             json.loads(sc.read_text())["detections"]]
            else:
                dets = seg_polys(sam_dir / f"{p.stem}.txt", 1, 1)
                parts_list = None
            if dets is None:
                continue
            img = None
            n_img += 1
            for li, _ in enumerate(dets):
                det_id = f"{p.stem}_{li}"
                if only_ids is not None and det_id not in only_ids:
                    continue
                if det_id in done:
                    n_skip += 1
                    continue
                if img is None:
                    img = Image.open(p).convert("RGB")
                    if raw_dir is None:
                        dets = seg_polys(sam_dir / f"{p.stem}.txt",
                                         img.width, img.height)
                cls, poly, box = dets[li]
                parts = parts_list[li] if parts_list else None
                blur = blurriness(img, box)
                crop, origin, sam_box_crop, scale = build_crop(
                    img, box, poly, highlight, style, min_side, parts=parts)
                prompt = PROMPT.format(W=crop.width, H=crop.height)
                bgr = np.array(crop)[:, :, ::-1].copy()
                text, _, _ = _safe_call(engine, bgr, prompt)
                v = parse_verdict(text)
                rec = {"id": det_id, "image": p.name,
                       "img_wh": [img.width, img.height],
                       "sam_class": DEFAULT_CLASS_NAMES.get(cls, str(cls)),
                       "box_img": [round(x, 1) for x in box],
                       "crop_origin": list(origin),
                       "crop_wh": [crop.width, crop.height],
                       "sam_box_crop": [round(x, 1) for x in sam_box_crop],
                       "crop_scale": round(scale, 4),
                       "has_poly": poly is not None,
                       "raw_parts": len(parts) if parts else None,
                       "sam_conf": confs[li] if confs else None,
                       "blurriness": round(blur, 1) if blur is not None else None,
                       "person_px_height": round(box[3] - box[1], 1),
                       "prompt_version": PROMPT_VERSION,
                       "prompt_sha": prompt_sha,
                       "highlighted": highlight,
                       "highlight_style": style if highlight else None,
                       "v": v, "parse_ok": v is not None, "raw_text": text}
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                n_det += 1

                if n_det % report_every == 0:
                    st = _status(n_det)
                    budget = (f" · {st['pct_of_budget']}% of ${max_spend:g} budget"
                              if max_spend else "")
                    print(f"[run] {n_det} dets · {n_img} imgs · "
                          f"${st['usd_spent']:.2f} spent "
                          f"(${st['usd_per_1k_dets']:.2f}/1k) · "
                          f"{st['dets_per_min']:.0f} dets/min{budget}",
                          flush=True)
                if max_spend is not None and _spend() >= max_spend:
                    _status(n_det)
                    print(f"\n[run] *** SPEND LIMIT REACHED: "
                          f"${_spend():.2f} >= ${max_spend:g}. Stopping. ***\n"
                          f"[run] {n_det} detections completed and saved. "
                          f"Re-run the same command (raise --max-spend) to "
                          f"continue where it stopped.", flush=True)
                    if img is not None:
                        img.close()
                    stop_for_budget = True
                    break
            if img is not None:
                img.close()
            if stop_for_budget:
                break
    cost = engine.cost.report()
    (out_dir / "cost_report.json").write_text(json.dumps(cost, indent=2))
    print(f"[run] done: {n_img} images, {n_det} new verdicts, {n_skip} "
          f"already present -> {verdicts_path}")
    if n_img == 0:
        print("[run] *** WARNING: 0 images processed — no labels/sidecars "
              "found for any image. Check that --raw-labels/--sam-labels "
              "points at an existing dir covering --images (did Stage A "
              "run?). ***")
    print(json.dumps(cost, indent=2))


def load_verdicts(run_dir: Path):
    recs = []
    for line in (run_dir / "verdicts.jsonl").read_text().splitlines():
        if line.strip():
            recs.append(json.loads(line))
    return recs


# ---------------------------------------------------------------------------
# Stage C scorers — per dataset, never pooled
# ---------------------------------------------------------------------------

def _box_stats(pairs):
    """pairs = [(iou_sam, iou_final, corrected)] on GT-matched kept dets."""
    corrected = [(a, b) for a, b, c in pairs if c]
    improved = sum(1 for a, b in corrected if b > a)
    return {
        "gt_matched_kept": len(pairs),
        "corrections_accepted": len(corrected),
        "corrections_improved_iou": improved,
        "pct_improved": _rate(improved, len(corrected)) if corrected else None,
        "mean_iou_sam": (round(sum(a for a, _, _ in pairs) / len(pairs), 3)
                         if pairs else None),
        "mean_iou_final": (round(sum(b for _, b, _ in pairs) / len(pairs), 3)
                           if pairs else None),
    }


def score_pass(recs):
    """PASS: every SAM detection is a false positive; the pipeline should
    delete them. Survivors listed for the owner skim (PASS crop-level
    contamination caveat, EXP-2026-08)."""
    merged = [(r, merge(r)) for r in recs]
    killed = [r for r, m in merged if not m["keep"]]
    survivors = [r for r, m in merged if m["keep"]]
    return {
        "mode": "pass_negatives", "detections": len(recs),
        "parse_failures": sum(1 for r in recs if not r["parse_ok"]),
        "fp_killed": len(killed), "fp_survived": len(survivors),
        "kill_rate": _rate(len(killed), len(recs)),
        "survivors": [{"id": r["id"], "sam_class": r["sam_class"],
                       "verdict": r["v"]["verdict"],
                       "final": merge(r)["final_class"]} for r in survivors],
    }


def score_crowd(recs, odgt_path: Path, match_iou_thr: float):
    """CrowdHuman: GT-matched dets are verified real people -> TP-keep;
    clear-FP dets (no GT overlap, outside ignores) -> kill rate; box
    refinement scored on matched-kept dets."""
    gt = load_odgt(odgt_path)
    by_img = {}
    for r in recs:
        by_img.setdefault(Path(r["image"]).stem, []).append(r)
    tp = tp_kept = fp = fp_killed = ignored = 0
    label_dist = Counter()
    box_pairs = []
    for stem, rows in by_img.items():
        entry = gt.get(stem)
        if entry is None:
            continue
        vboxes = [p["vbox"] for p in entry["persons"]]
        dets = [("d", *r["box_img"], r) for r in rows]
        kept_dets = [d for d in dets
                     if not any(_ioa(list(d[1:5]), ig) > IGNORE_IOA
                                for ig in entry["ignores"])]
        ignored += len(dets) - len(kept_dets)
        matched = match_boxes(vboxes, kept_dets, match_iou_thr)
        matched_ids = set()
        for gi, (d, j) in matched.items():
            r = d[5]
            matched_ids.add(id(r))
            m = merge(r)
            tp += 1
            if m["keep"]:
                tp_kept += 1
                label_dist[m["final_class"]] += 1
                box_pairs.append((j, iou(m["box_final_img"], vboxes[gi]),
                                  m["box_corrected"]))
        for d in kept_dets:
            r = d[5]
            if id(r) in matched_ids:
                continue
            best = max((iou(v, d[1:5]) for v in vboxes), default=0.0)
            if best < 0.1:               # clear FP (same rule as EXP-2026-03/09)
                fp += 1
                if not merge(r)["keep"]:
                    fp_killed += 1
    return {
        "mode": "crowdhuman", "detections": len(recs),
        "parse_failures": sum(1 for r in recs if not r["parse_ok"]),
        "dets_in_ignore_regions_excluded": ignored,
        "gt_verified_tp": tp, "tp_kept": tp_kept,
        "tp_keep_rate": _rate(tp_kept, tp),
        "clear_fp": fp, "clear_fp_killed": fp_killed,
        "clear_fp_kill_rate": _rate(fp_killed, fp) if fp else None,
        "final_label_distribution_on_kept_tp": dict(label_dist),
        "box_refinement": _box_stats(box_pairs),
    }


def score_lagenda(recs, gt_labels: Path, manifest: Path, match_iou_thr: float):
    """LAGENDA: the one human-labeled person per image gives gender+age GT.
    Scores the FINAL pipeline label (post-merge) on GT-matched dets."""
    from vlm_detect_eval import load_lagenda_gt
    gt = load_lagenda_gt(gt_labels, manifest)
    by_img = {}
    for r in recs:
        by_img.setdefault(Path(r["image"]).stem, []).append(r)
    matched_total = kept = 0
    cls_ok = cls_bad = unknown_cls = 0
    adult_as_child = child_as_adult = 0
    gender_ok = gender_bad = 0
    box_pairs = []
    for stem, rows in by_img.items():
        entry = gt.get(stem)
        if not entry:
            continue
        wh = rows[0].get("img_wh")
        if not wh:
            continue                     # pre-fix records lack img_wh
        w, h = wh
        gt_boxes = [[g["box_norm"][0] * w, g["box_norm"][1] * h,
                     g["box_norm"][2] * w, g["box_norm"][3] * h]
                    for g in entry]
        dets = [("d", *r["box_img"], r) for r in rows]
        matched = match_boxes(gt_boxes, dets, match_iou_thr)
        for gi, (d, j) in matched.items():
            g = entry[gi]
            r = d[5]
            m = merge(r)
            matched_total += 1
            if not m["keep"]:
                continue                  # counted via tp_keep_rate
            kept += 1
            gt_child = g["gt_age"] <= 12
            gt_cls = ("Child" if gt_child
                      else ("Man" if g["gt_gender"] == "M" else "Woman"))
            fc = m["final_class"]
            if fc == "Unknown":
                unknown_cls += 1
            elif fc == gt_cls:
                cls_ok += 1
            else:
                cls_bad += 1
                if not gt_child and fc == "Child":
                    adult_as_child += 1
                if gt_child and fc in ("Man", "Woman"):
                    child_as_adult += 1
            if not gt_child and fc in ("Man", "Woman"):
                if fc == ("Man" if g["gt_gender"] == "M" else "Woman"):
                    gender_ok += 1
                else:
                    gender_bad += 1
            box_pairs.append((j, iou(m["box_final_img"], gt_boxes[gi]),
                              m["box_corrected"]))
    return {
        "mode": "lagenda", "detections": len(recs),
        "parse_failures": sum(1 for r in recs if not r["parse_ok"]),
        "gt_matched": matched_total,
        "tp_keep_rate": _rate(kept, matched_total),
        "final_3class_accuracy_on_kept": _rate(cls_ok, cls_ok + cls_bad),
        "final_class_unknown_droppable": unknown_cls,
        "adult_labeled_child": adult_as_child,
        "adult_to_child_leak_rate": _rate(adult_as_child, kept),
        "child_labeled_adult_blur_safe": child_as_adult,
        "gender_accuracy_committed_adults": _rate(gender_ok,
                                                  gender_ok + gender_bad),
        "box_refinement": _box_stats(box_pairs),
    }


def compare_runs(path_a: Path, path_b: Path):
    """QC arm: highlighted vs plain verdicts on the same detections."""
    def index(p):
        return {r["id"]: r for r in
                (json.loads(x) for x in p.read_text().splitlines() if x.strip())
                if r["parse_ok"]}
    a, b = index(path_a), index(path_b)
    shared = sorted(set(a) & set(b))
    keep_flip = label_flip = committed = 0
    for i in shared:
        ma, mb = merge(a[i]), merge(b[i])
        if ma["keep"] != mb["keep"]:
            keep_flip += 1
        elif ma["keep"] and ma["final_class"] != "Unknown" \
                and mb["final_class"] != "Unknown":
            committed += 1
            if ma["final_class"] != mb["final_class"]:
                label_flip += 1
    return {
        "mode": "highlight_vs_plain", "shared_parsed": len(shared),
        "keep_verdict_flips": keep_flip,
        "keep_flip_rate": _rate(keep_flip, len(shared)),
        "committed_label_pairs": committed,
        "label_flips": label_flip,
        "label_flip_rate": _rate(label_flip, committed) if committed else None,
    }


# ---------------------------------------------------------------------------
# trace mode — a 3-5 image walkthrough showing every stage's input/output
# ---------------------------------------------------------------------------

CLASS_RGB = {"Woman": (200, 40, 160), "Man": (40, 90, 220),
             "Child": (0, 170, 200), "Unknown": (120, 120, 120)}
SAM3_CONFIG_NOTE = """\
SAM3 stage = the FROZEN production labels (not re-run here).
Producer: /workspace/autolabel/autolabel_sam.py
Config:   --classes "woman" "man" "child"  (three text prompts)
          conf 0.4, NMS IoU 0.7, batch 1   (script defaults, EXP-2026-02/03)
Output:   YOLO segment polygons per detection: "<class> x1 y1 x2 y2 ..." (normalized)
          class 0=Woman 1=Man 2=Child; the polygon IS the mask."""


def run_trace(engine, images_dir: Path, sam_dir: Path, out_dir: Path,
              n_images: int, max_dets: int, seed: int, gt_spec,
              style: str = "outline", min_side: int = 320,
              raw_dir: Path | None = None):
    import html as H
    from crowd_headtohead import _annotated, _b64_jpeg
    from vlm_detect_eval import _safe_call, select_images

    out_dir.mkdir(parents=True, exist_ok=True)
    # sample images that actually have >=1 SAM detection
    picked = []
    for p in select_images(images_dir, 0, seed):
        if raw_dir is not None:
            dets, _ = load_raw_dets(raw_dir, p.stem)
        else:
            dets = seg_polys(sam_dir / f"{p.stem}.txt", 1, 1)
        if dets:
            picked.append(p)
        if len(picked) >= n_images:
            break

    gt_boxes_for = lambda stem, w, h: []          # noqa: E731
    gt_kind = None
    if gt_spec and gt_spec[0] == "lagenda":
        from vlm_detect_eval import load_lagenda_gt
        gt = load_lagenda_gt(gt_spec[1], gt_spec[2])
        gt_kind = "lagenda"

        def gt_boxes_for(stem, w, h):
            return [([g["box_norm"][0] * w, g["box_norm"][1] * h,
                      g["box_norm"][2] * w, g["box_norm"][3] * h],
                     f"GT: {g['gt_gender']} age {g['gt_age']}")
                    for g in gt.get(stem, [])]
    elif gt_spec and gt_spec[0] == "crowdhuman":
        gt = load_odgt(gt_spec[1])
        gt_kind = "crowdhuman"

        def gt_boxes_for(stem, w, h):
            e = gt.get(stem)
            return ([(p["vbox"], "GT person") for p in e["persons"]]
                    if e else [])

    css = """body{font-family:-apple-system,Segoe UI,sans-serif;background:#fafafa;
    color:#222;margin:24px;max-width:1240px}
    h1{margin-bottom:2px}.sub{color:#666;margin-top:0}
    h2{border-bottom:2px solid #ccc;padding-bottom:4px;margin-top:44px}
    h3{margin:18px 0 6px;color:#333}
    .stage{background:#fff;border:1px solid #ddd;border-radius:6px;
           padding:12px;margin:10px 0}
    .stage img{max-width:560px;display:block;margin:6px 0}
    .det{display:inline-block;vertical-align:top;background:#fdfdfd;
         border:1px solid #e2e2e2;border-radius:6px;padding:10px;margin:6px;
         max-width:400px}
    .det img{max-width:360px}
    pre{background:#f0f0f0;padding:10px;border-radius:6px;font-size:11.5px;
        overflow-x:auto;white-space:pre-wrap}
    .keep{color:#0a0;font-weight:700}.del{color:#d00;font-weight:700}
    .tag{font-size:12px;color:#666}"""
    doc = [f"<title>EXP-2026-10 pipeline trace</title><style>{css}</style>",
           "<h1>Pipeline v1 — stage-by-stage trace</h1>",
           f"<p class='sub'>{len(picked)} sample images · every stage's exact "
           f"input and output · crop config: mask <b>{style}</b>, 25% pad, "
           f"small crops upscaled to {min_side}px (the config adopted after "
           "the pilot variant test) · box colors: "
           "<b style='color:#00be00'>green=GT</b>, "
           "<b style='color:#2858dc'>blue=Man</b>, "
           "<b style='color:#c828a0'>magenta=Woman</b>, "
           "<b style='color:#00aac8'>cyan=Child</b>, "
           "<b style='color:#d00'>red=deleted by the gate</b>; corrected boxes "
           "drawn solid with the original SAM3 box in gray</p>",
           "<details><summary><b>The exact Gemini prompt</b> (identical for every "
           "crop; only the W/H pixel numbers change)</summary>"
           f"<pre>{H.escape(PROMPT)}</pre></details>",
           f"<div class='stage'><h3>SAM3 stage config (frozen)</h3>"
           f"<pre>{H.escape(SAM3_CONFIG_NOTE)}</pre></div>"]

    for p in picked:
        img = Image.open(p).convert("RGB")
        w, h = img.size
        if raw_dir is not None:
            dets, parts_list = load_raw_dets(raw_dir, p.stem)
            raw_label_text = json.dumps(
                json.loads((raw_dir / f"{p.stem}.json").read_text()),
                indent=1)[:3000]
        else:
            dets = seg_polys(sam_dir / f"{p.stem}.txt", w, h)
            parts_list = [None] * len(dets)
            raw_label_text = (sam_dir / f"{p.stem}.txt").read_text()
        shown = dets[:max_dets]
        doc.append(f"<h2>{p.name} <span class='tag'>({w}x{h}, "
                   f"{len(dets)} SAM3 detections"
                   + (f", showing first {max_dets}" if len(dets) > max_dets
                      else "") + ")</span></h2>")

        # Stage 0 — raw input (+ GT overlay when available)
        gtb = gt_boxes_for(p.stem, w, h)
        doc.append("<div class='stage'><h3>Stage 0 — raw image"
                   + (f" with {gt_kind} ground truth" if gtb else
                      " (no GT overlay for this dataset)") + "</h3>"
                   + f"<img src='data:image/jpeg;base64,"
                   f"{_annotated(img, [(b, lab, (0, 190, 0)) for b, lab in gtb], 620)}'>"
                   "</div>")

        # Stage 1 — SAM3 raw output
        truncated = "\n".join(raw_label_text.splitlines()[:max_dets])
        sam_overlay = [(list(d[2]),
                        f"{DEFAULT_CLASS_NAMES.get(d[0], d[0])}",
                        CLASS_RGB.get(DEFAULT_CLASS_NAMES.get(d[0]), (90,) * 3))
                       for d in dets]
        doc.append("<div class='stage'><h3>Stage 1 — SAM3 raw output "
                   "(verbatim label lines + rendered)</h3>"
                   f"<pre>{H.escape(truncated)}"
                   + (f"\n... ({len(dets) - max_dets} more lines)" if
                      len(dets) > max_dets else "") + "</pre>"
                   f"<img src='data:image/jpeg;base64,"
                   f"{_annotated(img, sam_overlay, 620)}'></div>")

        # Stage 2+3 — per detection: exact Gemini input + raw output
        doc.append("<div class='stage'><h3>Stages 2–3 — per detection: the "
                   "EXACT highlighted crop sent to Gemini, and its raw "
                   "response</h3>")
        merged_final = []
        for li, (cls, poly, box) in enumerate(shown):
            crop, origin, sam_box_crop, scale = build_crop(
                img, box, poly, True, style, min_side,
                parts=parts_list[li])
            prompt = PROMPT.format(W=crop.width, H=crop.height)
            bgr = np.array(crop)[:, :, ::-1].copy()
            text, _, _ = _safe_call(engine, bgr, prompt)
            v = parse_verdict(text)
            rec = {"box_img": box, "crop_origin": list(origin),
                   "crop_wh": [crop.width, crop.height],
                   "sam_box_crop": sam_box_crop, "crop_scale": scale, "v": v}
            m = merge(rec)
            merged_final.append((box, m, v))
            status = (f"<span class='keep'>KEEP → {m['final_class']}</span>"
                      if m["keep"] else
                      f"<span class='del'>DELETE — Gemini verdict: "
                      f"{(v or {}).get('verdict', 'unparseable response')}"
                      f"</span>")
            corr = (" · box corrected" if m["box_corrected"] else "")
            doc.append(
                "<div class='det'>"
                f"<b>det {li}</b> — SAM3 says "
                f"{DEFAULT_CLASS_NAMES.get(cls, cls)} · crop "
                f"{crop.width}x{crop.height}px (25% pad, mask {style}"
                + (f", upscaled {scale:.1f}x" if scale > 1.0 else "")
                + (f", RAW mask: {len(parts_list[li])} part(s)"
                   if parts_list[li] else "") + ")"
                f"<img src='data:image/jpeg;base64,{_b64_jpeg(crop)}'>"
                f"<b>Gemini raw response:</b><pre>{H.escape(text or '(empty)')}</pre>"
                f"<b>After merge rules:</b> {status}{corr}</div>")
        doc.append("</div>")

        # Stage 4 — final merged result. UNTRACED detections (beyond the
        # --max-dets sample cap) are drawn gray so the cap can never be
        # mistaken for the pipeline deleting people — a full run processes
        # every detection identically.
        final_overlay = []
        for box, m, v in merged_final:
            if not m["keep"]:
                reason = (v or {}).get("verdict", "no-parse")
                final_overlay.append((list(box), f"DEL:{reason}", (221, 0, 0)))
                continue
            col = CLASS_RGB.get(m["final_class"], (90,) * 3)
            if m["box_corrected"]:
                final_overlay.append((list(box), "", (150, 150, 150)))
            final_overlay.append((list(m["box_final_img"]),
                                  m["final_class"], col))
        n_untraced = len(dets) - len(shown)
        for cls, _, box in dets[len(shown):]:
            final_overlay.append((list(box), "not traced", (160, 160, 160)))
        doc.append("<div class='stage'><h3>Stage 4 — final training labels "
                   "for the traced detections</h3>"
                   "<p class='tag'>red = deleted, with Gemini's verdict as "
                   "the reason (DEL:not_person etc.); "
                   + (f"<b>gray = the {n_untraced} detections beyond this "
                      "trace's per-image sample cap — NOT deleted; a full "
                      "run sends every one of them to Gemini identically.</b> "
                      if n_untraced else "")
                   + "</p>"
                   f"<img src='data:image/jpeg;base64,"
                   f"{_annotated(img, final_overlay, 620)}'></div>")
        img.close()

    (out_dir / "trace.html").write_text("\n".join(doc))
    cost = engine.cost.report()
    (out_dir / "cost_report.json").write_text(json.dumps(cost, indent=2))
    print(f"[trace] {len(picked)} images -> {out_dir/'trace.html'}")
    print(json.dumps(cost, indent=2))


# ---------------------------------------------------------------------------
# selftest — stub engine, no network
# ---------------------------------------------------------------------------

class _StubEngine:
    """Returns canned verdicts keyed by call order."""
    def __init__(self, answers):
        self.answers = list(answers)
        import api_describers
        self.cost = api_describers.CostTracker("stub-model")

    def _call(self, image_bgr, prompt):
        return self.answers.pop(0), 100, 40


def _selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "imgs").mkdir(); (root / "sam").mkdir()
        Image.new("RGB", (100, 100), (120,) * 3).save(root / "imgs/x.jpg")
        # det0: polygon person (Man), det1: box-only FP (Woman on nothing)
        (root / "sam/x.txt").write_text(
            "1 0.10 0.10 0.40 0.10 0.40 0.90 0.10 0.90\n"
            "0 0.70 0.70 0.10 0.10\n")
        answers = [
            # det0: real man, corrected box inside crop, tight
            json.dumps({"verdict": "real_person", "gender": "man",
                        "age_group": "adult", "estimated_age": 34,
                        "box_correction": [8, 8, 38, 78],
                        "highlight_quality": "good", "confidence": "high"}),
            # det1: statue -> deleted
            json.dumps({"verdict": "not_person", "gender": "unknown",
                        "age_group": "unknown", "estimated_age": 30,
                        "box_correction": "ok",
                        "highlight_quality": "good", "confidence": "high"}),
        ]
        run_stage_b(_StubEngine(answers), root / "imgs", root / "sam",
                    root / "out", 0, 42, highlight=True)
        recs = load_verdicts(root / "out")
        assert len(recs) == 2 and all(r["parse_ok"] for r in recs), recs
        # resume: second run adds nothing
        run_stage_b(_StubEngine([]), root / "imgs", root / "sam",
                    root / "out", 0, 42, highlight=True)
        assert len(load_verdicts(root / "out")) == 2
        # only_ids: restrict to one detection
        run_stage_b(_StubEngine([answers2()[0]]), root / "imgs", root / "sam",
                    root / "out_only", 0, 42, highlight=True,
                    only_ids={"x_0"})
        assert len(load_verdicts(root / "out_only")) == 1

        m0, m1 = merge(recs[0]), merge(recs[1])
        # v1 policy: geometry never overwritten (APPLY_BOX_CORRECTIONS False)
        assert m0["keep"] and m0["final_class"] == "Man", m0
        assert not m0["box_corrected"] and m0["box_final_img"] == recs[0]["box_img"], m0
        assert recs[0]["v"]["box_correction"] is not None, "correction must still be LOGGED"
        assert not m1["keep"], m1

        # teen policy: age 15 "child" claim -> adult (gender wins)
        teen = dict(recs[0])
        teen["v"] = dict(recs[0]["v"],
                         age_group="child", estimated_age=15, gender="woman",
                         box_correction=None)
        assert merge(teen)["final_class"] == "Woman"
        kid = dict(teen)
        kid["v"] = dict(teen["v"], estimated_age=8)
        assert merge(kid)["final_class"] == "Child"
        # box sanity: far-away correction rejected (identity-swap guard)
        far = dict(recs[0])
        far["v"] = dict(recs[0]["v"], box_correction=[60, 60, 90, 90])
        assert not merge(far)["box_corrected"]

        # bridge-stitched polygon (two squares joined by a zero-width
        # retrace, the YOLO multi-part flattening) -> TWO clean loops
        bridged = [(10, 10), (40, 10), (40, 40), (10, 40),   # part 1
                   (10, 10), (10, 70),                        # bridge out
                   (10, 70), (40, 70), (40, 95), (10, 95),    # part 2
                   (10, 70), (10, 10)]                        # bridge back
        loops = outline_loops((100, 100), bridged)
        assert len(loops) == 2, f"expected 2 part-loops, got {len(loops)}"
        # a WIDE (5px) bridge — the trace3 failure mode: must also split to 2
        wide_bridge = [(10, 10), (40, 10), (40, 40), (22, 40),
                       (22, 70), (40, 70), (40, 95), (10, 95),
                       (10, 70), (18, 70), (18, 40), (10, 40)]
        loops = outline_loops((100, 100), wide_bridge)
        assert len(loops) == 2, f"wide bridge: expected 2 loops, got {len(loops)}"
        # a plain square stays a single loop
        assert len(outline_loops((100, 100),
                                 [(10, 10), (60, 10), (60, 60), (10, 60)])) == 1

        # outline style: no tint compositing path, still returns 4-tuple
        with Image.open(root / "imgs/x.jpg") as im:
            c, o, sb, sc = build_crop(im, [10, 10, 40, 90],
                                      [(10, 10), (40, 10), (40, 90)],
                                      style="outline")
            assert sc == 1.0 and c.size[0] > 0
            # upscale: 30x80 box + pad -> small crop, min_side 200 doubles+
            c2, o2, sb2, sc2 = build_crop(im, [10, 10, 40, 90], None,
                                          style="outline", min_side=200)
            assert sc2 > 1.0 and max(c2.size) >= 200, (c2.size, sc2)
        # scale-aware box mapping: correction in scaled crop coords maps back
        up = dict(recs[0])
        up["crop_scale"] = 2.0
        up["sam_box_crop"] = [v * 2 for v in recs[0]["sam_box_crop"]]
        up["v"] = dict(recs[0]["v"], box_correction=[16, 16, 76, 156])
        up["crop_wh"] = [recs[0]["crop_wh"][0] * 2, recs[0]["crop_wh"][1] * 2]
        assert not merge(up)["box_corrected"], "v1 policy: corrections not applied"

        # PASS scorer: 1 killed, 1 survivor
        s = score_pass(recs)
        assert s["fp_killed"] == 1 and s["fp_survived"] == 1, s

        # crowd scorer: person A matches det0 (kept TP), det1 clear FP killed
        odgt = {"ID": "x", "gtboxes": [
            {"tag": "person", "vbox": [10, 10, 30, 80],
             "fbox": [10, 10, 30, 80], "extra": {}}]}
        (root / "anno.odgt").write_text(json.dumps(odgt) + "\n")
        s = score_crowd(recs, root / "anno.odgt", 0.5)
        assert s["gt_verified_tp"] == 1 and s["tp_kept"] == 1, s
        assert s["clear_fp"] == 1 and s["clear_fp_killed"] == 1, s
        assert s["box_refinement"]["corrections_accepted"] == 0, s

        # compare: identical runs -> zero flips
        c = compare_runs(root / "out/verdicts.jsonl",
                         root / "out/verdicts.jsonl")
        assert c["keep_verdict_flips"] == 0 and c["label_flips"] == 0, c

        # trace mode: full walkthrough HTML with stub verdicts
        run_trace(_StubEngine(answers2()), root / "imgs", root / "sam",
                  root / "trace_out", n_images=1, max_dets=8, seed=42,
                  gt_spec=("crowdhuman", root / "anno.odgt"))
        html = (root / "trace_out/trace.html").read_text()
        for marker in ("Stage 0", "Stage 1", "Stages 2–3", "Stage 4",
                       "Gemini raw response", "DELETE", "KEEP"):
            assert marker in html, f"trace missing: {marker}"

        # RAW-sidecar path: 2-part mask drawn exactly, no heuristics
        (root / "raw").mkdir()
        (root / "raw/x.json").write_text(json.dumps({
            "image": "x.jpg", "width": 100, "height": 100,
            "detections": [{"cls": 1, "conf": 0.91, "box": [10, 10, 40, 90],
                            "parts": [[[10, 10], [40, 10], [40, 40], [10, 40]],
                                      [[10, 60], [40, 60], [40, 90], [10, 90]]]}]}))
        dets, pl = load_raw_dets(root / "raw", "x")
        assert len(dets) == 1 and len(pl[0]) == 2, (dets, pl)
        with Image.open(root / "imgs/x.jpg") as im:
            c, _, _, _ = build_crop(im, dets[0][2], None, parts=pl[0],
                                    style="outline", min_side=320)
            assert c.size[0] > 0
        run_trace(_StubEngine([answers2()[0]]), root / "imgs", root / "raw",
                  root / "trace_raw", n_images=1, max_dets=8, seed=42,
                  gt_spec=None, raw_dir=root / "raw")
        html = (root / "trace_raw/trace.html").read_text()
        assert "RAW mask: 2 part(s)" in html, "raw parts caption missing"

        # run cmd over raw sidecars: exact-part rendering + raw_parts logged,
        # v2 prompt fields passed through, computed fields present
        v2_answer = json.dumps({
            "verdict": "real_person", "gender": "man", "age_group": "adult",
            "estimated_age": 34, "box_correction": "ok",
            "highlight_quality": "good", "verdict_confidence": "high",
            "apparent_race": "east_asian", "occlusion": "partial",
            "occlusion_percent": 40, "face_visible": "yes",
            "orientation": "back", "pose": "standing",
            "exposed_body_parts": ["Face", "hands"], "clothing_fit": "loose",
            "garment_type": "thobe_robe",
            "head_covering": "ghutra_keffiyeh", "facial_hair": "none",
            "gender_cues": "face shape", "skin_tone_mst": 5,
            "skin_tone_confidence": "reliable"})
        run_stage_b(_StubEngine([v2_answer]), root / "imgs", root / "raw",
                    root / "out_rawrun", 0, 42, highlight=True,
                    style="outline", min_side=320, raw_dir=root / "raw")
        rr = load_verdicts(root / "out_rawrun")
        assert len(rr) == 1 and rr[0]["raw_parts"] == 2, rr
        # provenance: run_meta.json freezes prompt text/hash + all params,
        # and the same hash lands on every record
        rm = json.loads((root / "out_rawrun/run_meta.json").read_text())
        assert rm["prompt_sha"] == rr[0]["prompt_sha"], (rm, rr[0])
        assert rm["prompt_text"] == PROMPT and rm["raw_masks"] is True, rm
        assert rm["crop"]["min_crop_side"] == 320, rm
        assert rm["merge_thresholds"]["child_age_max"] == CHILD_AGE_MAX, rm
        assert merge(rr[0])["keep"] and merge(rr[0])["final_class"] == "Man"
        rv = rr[0]["v"]
        assert rv["verdict_confidence"] == "high", rv
        assert rv["apparent_race"] == "east_asian", rv
        assert rv["occlusion_percent"] == 40.0, rv
        assert rv["orientation"] == "back" and rv["skin_tone_mst"] == 5, rv
        assert rv["exposed_body_parts"] == ["face", "hands"], rv
        assert rv["clothing_fit"] == "loose", rv
        assert rr[0]["person_px_height"] == 80.0, rr[0]
        assert rr[0]["blurriness"] is not None, rr[0]   # uniform gray -> 0.0
        assert rr[0]["prompt_version"] == PROMPT_VERSION
        # legacy fallback: pre-v2 records used "confidence"
        assert recs[0]["v"]["verdict_confidence"] == "high", recs[0]["v"]
    print("pipeline_v1_eval.py self-test passed")


def answers2():
    return [
        json.dumps({"verdict": "real_person", "gender": "man",
                    "age_group": "adult", "estimated_age": 34,
                    "box_correction": "ok",
                    "highlight_quality": "good", "confidence": "high"}),
        json.dumps({"verdict": "not_person", "gender": "unknown",
                    "age_group": "unknown", "estimated_age": 30,
                    "box_correction": "ok",
                    "highlight_quality": "good", "confidence": "high"}),
    ]


def main():
    ap = argparse.ArgumentParser(description="EXP-2026-10 pipeline v1 eval")
    ap.add_argument("cmd", choices=["run", "score", "compare", "trace",
                                    "selftest"])
    ap.add_argument("--n", type=int, default=4,
                    help="trace: number of sample images")
    ap.add_argument("--max-dets", type=int, default=8,
                    help="trace: max detections traced per image")
    ap.add_argument("--images"); ap.add_argument("--sam-labels")
    ap.add_argument("--out"); ap.add_argument("--run-dir")
    ap.add_argument("--model", default="gemini-3.5-flash-lite")
    ap.add_argument("--max-images", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--plain-crops", action="store_true",
                    help="QC arm: no highlighting (bias check)")
    ap.add_argument("--highlight-style", choices=["tint", "outline"],
                    default="outline",
                    help="default 'outline' = the ADOPTED config (pilot "
                         "variant test 2026-07-23: tint obscured small "
                         "people, 80.4%% TP-keep); 'tint' kept only for "
                         "comparison arms")
    ap.add_argument("--min-crop-side", type=int, default=320,
                    help="upscale crops so max side >= this before marking "
                         "(ADOPTED default 320 — recovered 68/76 failed "
                         "small-person cases at zero token cost); 0 = off")
    ap.add_argument("--only-ids", default=None,
                    help="file with one detection id per line; run ONLY "
                         "those (e.g. re-testing prior failures)")
    ap.add_argument("--prompt", choices=["v2", "e1"], default="v2",
                    help="v2 = the 21-field validated prompt; e1 = the trimmed "
                         "9-field production prompt (spotlight_run.py)")
    ap.add_argument("--max-spend", type=float, default=None,
                    help="HARD STOP once estimated spend reaches this many "
                         "USD. Work already done is saved; re-run to continue.")
    ap.add_argument("--rate-in", type=float, default=0.30,
                    help="$/1M input tokens for the spend meter")
    ap.add_argument("--rate-out", type=float, default=2.50,
                    help="$/1M output tokens for the spend meter")
    ap.add_argument("--report-every", type=int, default=50,
                    help="print progress + write status.json every N detections")
    ap.add_argument("--raw-labels", default=None,
                    help="dir of autolabel_sam_raw.py per-image .json "
                         "sidecars (RAW per-part mask polygons) — rendering "
                         "uses the exact parts, no heuristics (trace cmd)")
    ap.add_argument("--gt", choices=["lagenda", "crowdhuman", "none"])
    ap.add_argument("--odgt"); ap.add_argument("--gt-labels")
    ap.add_argument("--manifest")
    ap.add_argument("--match-iou", type=float, default=0.5)
    ap.add_argument("--run-a"); ap.add_argument("--run-b")
    args = ap.parse_args()

    if args.cmd == "selftest":
        _selftest()
        return
    if getattr(args, "prompt", "v2") == "e1":
        global PROMPT, ANALYSIS_KEYS, PROMPT_VERSION
        PROMPT, ANALYSIS_KEYS, PROMPT_VERSION = _load_e1()
        print(f"[eval] using production prompt {PROMPT_VERSION} "
              f"({len(ANALYSIS_KEYS)} analysis fields)")
    if args.cmd == "run":
        if not (args.images and args.out
                and (args.sam_labels or args.raw_labels)):
            ap.error("run needs --images --out and one of "
                     "--sam-labels / --raw-labels")
        import api_describers
        engine = api_describers.build_engine("gemini", args.model,
                                             max_tokens=args.max_tokens)
        run_stage_b(engine, Path(args.images),
                    Path(args.sam_labels or args.raw_labels),
                    Path(args.out), args.max_images, args.seed,
                    highlight=not args.plain_crops,
                    style=args.highlight_style,
                    min_side=args.min_crop_side,
                    only_ids=(set(Path(args.only_ids).read_text().split())
                              if args.only_ids else None),
                    raw_dir=Path(args.raw_labels) if args.raw_labels else None,
                    max_spend=args.max_spend, rate_in=args.rate_in,
                    rate_out=args.rate_out, report_every=args.report_every)
        return
    if args.cmd == "trace":
        if not (args.images and args.out
                and (args.sam_labels or args.raw_labels)):
            ap.error("trace needs --images --out and one of "
                     "--sam-labels / --raw-labels")
        args.sam_labels = args.sam_labels or args.raw_labels
        gt_spec = None
        if args.gt == "lagenda":
            if not (args.gt_labels and args.manifest):
                ap.error("--gt lagenda needs --gt-labels --manifest")
            gt_spec = ("lagenda", Path(args.gt_labels), Path(args.manifest))
        elif args.gt == "crowdhuman":
            if not args.odgt:
                ap.error("--gt crowdhuman needs --odgt")
            gt_spec = ("crowdhuman", Path(args.odgt))
        import api_describers
        engine = api_describers.build_engine("gemini", args.model,
                                             max_tokens=args.max_tokens)
        run_trace(engine, Path(args.images), Path(args.sam_labels),
                  Path(args.out), args.n, args.max_dets, args.seed, gt_spec,
                  style=args.highlight_style, min_side=args.min_crop_side,
                  raw_dir=Path(args.raw_labels) if args.raw_labels else None)
        return
    if args.cmd == "compare":
        if not (args.run_a and args.run_b):
            ap.error("compare needs --run-a --run-b")
        out = compare_runs(Path(args.run_a), Path(args.run_b))
        print(json.dumps(out, indent=2))
        return
    # score
    if not (args.run_dir and args.gt):
        ap.error("score needs --run-dir --gt")
    recs = load_verdicts(Path(args.run_dir))
    if args.gt == "none":
        out = score_pass(recs)
    elif args.gt == "crowdhuman":
        if not args.odgt:
            ap.error("--gt crowdhuman needs --odgt")
        out = score_crowd(recs, Path(args.odgt), args.match_iou)
    else:
        if not (args.gt_labels and args.manifest):
            ap.error("--gt lagenda needs --gt-labels --manifest")
        out = score_lagenda(recs, Path(args.gt_labels), Path(args.manifest),
                            args.match_iou)
    (Path(args.run_dir) / f"score_{args.gt}.json").write_text(
        json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "survivors"},
                     indent=2))
    if "survivors" in out:
        print(f"[score] {len(out['survivors'])} survivors listed in "
              f"score_{args.gt}.json (owner skim needed)")


if __name__ == "__main__":
    main()
