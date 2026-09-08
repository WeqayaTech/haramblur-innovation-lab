#!/usr/bin/env python3
"""
Deployment ship-table comparison — the customer-experience metrics.

Scores candidate models against a labeled dataset the way a USER experiences
them, per simulated audience, instead of box-level detection metrics:

  M-1  exposure pass rate    % of images where unblurred target-class area
                             (union-coverage, area-weighted) stays under tau.
                             Subsumes every escape path: no box, wrong gender,
                             called Child, partial blur.
  M-2  false-blur rate       % of images whose wrongly-blurred area exceeds
                             phi, plus false-blur boxes per 100 images.
                             Computed per mode: only the blurred class's
                             predictions can false-blur (a false "Man" box is
                             invisible to a male user).
  B-1  child clarity         % of child-containing images with no GT Child
                             blurred beyond kappa. Teen-band-tagged images
                             (est. age 10-17 anywhere in frame) are excluded.

Modes: male (blur target = class 0 Woman), female (target = class 1 Man).
Class 2 (Child) is never a blur target.

Comparisons are made at a MATCHED OPERATING POINT, never a shared confidence:
the first --pred model is the incumbent, anchored at --anchor-conf; every
other model is evaluated at the confidence (from --grid) whose false-blur
image rate is closest to the incumbent's. Comparing checkpoints at the same
raw threshold scores calibration drift and calls it accuracy.

Slices: every metric is recomputed per slice tag (see `curate`). Slices
overlap; they are views, not a partition. An aggregate win with a slice
regression is not a win.

Rules inherited from the project (do not relax):
  - predictions overlapping an --ignore region are neither exposure credit
    nor false blur (absence of a label is not evidence of absence);
  - GT boxes come from seg_boxes() (segment polygons parse as extents; a
    plain 5-field reader would drop every holdout row);
  - glob *.txt in label dirs, never * (audit sidecars live beside labels);
  - bootstrap resamples IMAGES, not boxes.

Subcommands
-----------
curate   Build slices.json from a Spotlight verdicts_batch.jsonl + GT labels:
         collection tag (stem prefix before "__"), per-audience exposure
         tiers from `exposed_body_parts` (t0_covered / t1_modest /
         t2_ordinary / t3_revealing / t4_high — the agreed 5-tier taxonomy,
         see the constants below), size bands from GT box height in
         model-input pixels, and the teen-band exclusion tag. Gemini labels SELECT slices; they never score them.

           python3 deploy_compare.py curate \\
             --verdicts .../run/verdicts_batch.jsonl \\
             --gt-labels .../labels_eval --out slices.json

materialize  Build per-slice subset FOLDERS as symlinks (no data copied —
         zero extra storage; the holdout's images are themselves symlinks,
         and links-to-links resolve fine on the same volume):

           python3 deploy_compare.py materialize \\
             --slices slices.json --images .../images \\
             --labels .../labels_eval --out /workspace/deploycmp/subsets

         creates <out>/<tag>/{images,labels}/ with absolute symlinks, one
         folder per slice tag (":" becomes "_"). Point any existing tool at
         a subset folder as if it were a dataset.

score    Run the comparison. Predictions are run_ultralytics_labels.py raw
         sidecars (log-raw, floor <= min grid conf).

           python3 deploy_compare.py score \\
             --gt-labels .../labels_eval --ignore .../ignore \\
             --slices slices.json --out /workspace/deploycmp/holdout \\
             --pred yolo11N-640=/workspace/.../v11n/raw \\
             --pred y26n_gradsupp=/workspace/.../y26n_gs/raw

selftest No data, no GPU, no network:
           python3 deploy_compare.py selftest

Outputs: printed ship table, <out>/summary.json (all models x modes x confs,
full curves, matched points, CIs), <out>/per_image.jsonl (one record per
image x model x mode at the matched point — the diagnostic tier and any
re-bootstrap are pure post-processing).

Thresholds (pre-register before a bake-off; defaults are the spec's draft
values): --tau 0.005 --phi 0.01 --kappa 0.2 --dilate 0.1 --anchor-conf 0.45.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    from run_autolabel_on_manifest import seg_boxes
except ImportError:                     # cv2-less dev machine: identical copy
    def seg_boxes(label_file: Path, w: int, h: int):
        """Fallback mirror of run_autolabel_on_manifest.seg_boxes (which
        needs cv2 via its import chain). Keep byte-identical in behavior."""
        if not label_file.exists():
            return None
        boxes = []
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
                boxes.append((cls, (cx - bw / 2) * w, (cy - bh / 2) * h,
                              (cx + bw / 2) * w, (cy + bh / 2) * h))
            else:
                xs, ys = vals[0::2], vals[1::2]
                if len(xs) < 3 or len(xs) != len(ys):
                    continue
                boxes.append((cls, min(xs) * w, min(ys) * h,
                              max(xs) * w, max(ys) * h))
        return boxes

WOMAN, MAN, CHILD = 0, 1, 2
MODES = {"male": WOMAN, "female": MAN}          # audience -> blur-target class

# ------------------------------------------------------------------ curate
# Exposure tiers — AGREED DEFINITION (owner, 2026-08-25). Ordered, exhaustive,
# a pure function of the observable `exposed_body_parts` list:
#   t0_covered    exposed ⊆ {face, hands}          (hijab-level coverage;
#                 corroborated: 69/78 holdout hijab/niqab wearers land here)
#   t1_modest     + hair / neck / ears             (bare head, body covered)
#   t2_ordinary   + arms / shoulders / feet / knees (t-shirt / everyday)
#   t3_revealing  chest without midriff (neckline — 90% of all `chest`
#                 occurrences), or legs without chest (shorts/skirt), or back
#   t4_high       midriff/stomach, or chest AND legs together (swimwear /
#                 minimal). Owner decision: t4 GATES ALONE despite n≈229 on
#                 the holdout — report its wide CI, don't merge it away.
# An UNRECOGNIZED part is tiered t4 and warned about — new vocabulary must
# never silently land in a covered tier.
T0_SET = {"face", "hands", "eyes"}
T1_EXTRA = {"hair", "neck", "ears"}
T2_EXTRA = {"arms", "shoulders", "feet", "knees", "wrists", "head",
            "upper_arms", "forearms"}
T4_PARTS = {"midriff", "stomach", "torso", "thighs"}
KNOWN_PARTS = T0_SET | T1_EXTRA | T2_EXTRA | T4_PARTS | {"chest", "legs",
                                                         "back"}
EXPOSURE_ORDER = {"t0_covered": 0, "t1_modest": 1, "t2_ordinary": 2,
                  "t3_revealing": 3, "t4_high": 4}

TEEN_LO, TEEN_HI = 10, 17          # est. age band excluded from child clarity
SIZE_BANDS = ((0, 64, "small"), (64, 160, "med"), (160, 10 ** 9, "large"))
MODEL_INPUT = 640                  # size bands measured in model-input px


def exposure_bucket(parts, unknown_seen: set) -> str:
    core = {p for p in (parts or []) if p != "none"}
    if core <= T0_SET:
        return "t0_covered"
    if core <= T0_SET | T1_EXTRA:
        return "t1_modest"
    if core <= T0_SET | T1_EXTRA | T2_EXTRA:
        return "t2_ordinary"
    unrec = core - KNOWN_PARTS
    if unrec:
        unknown_seen.update(unrec)
        return "t4_high"           # conservative: unknown vocab -> top tier
    if core & T4_PARTS or ("chest" in core and "legs" in core):
        return "t4_high"
    return "t3_revealing"          # chest-only / legs-only / back


def size_band(h_px: float) -> str:
    for lo, hi, name in SIZE_BANDS:
        if lo <= h_px < hi:
            return name
    return "large"


def curate(args) -> dict:
    """verdicts + GT labels -> {stem: [tags]} with a _meta census."""
    by_stem: dict[str, dict] = defaultdict(
        lambda: {"w": None, "m": None, "teen": False, "wh": None})
    unknown_parts: set = set()
    n_rows = 0
    with open(args.verdicts) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            v = d.get("v") or {}
            n_rows += 1
            s = by_stem[d["image_stem"]]
            if s["wh"] is None and d.get("img_wh"):
                s["wh"] = d["img_wh"]
            age = v.get("estimated_age")
            if isinstance(age, (int, float)) and TEEN_LO <= age <= TEEN_HI:
                s["teen"] = True
            if v.get("verdict") != "real_person":
                continue
            if v.get("age_group") == "child":
                continue                       # adults drive exposure slices
            g = v.get("gender")
            if g not in ("woman", "man"):
                continue
            b = exposure_bucket(v.get("exposed_body_parts"), unknown_parts)
            key = "w" if g == "woman" else "m"
            cur = s[key]
            if cur is None or EXPOSURE_ORDER[b] > EXPOSURE_ORDER[cur]:
                s[key] = b                     # image takes its MAX bucket

    slices: dict[str, list] = {}
    census: Counter = Counter()
    label_dir = Path(args.gt_labels)
    # detection-free images have no verdict rows but still belong to their
    # collection (the negatives arm lives there) — tag every GT stem.
    for lf in label_dir.glob("*.txt"):
        if lf.stem not in by_stem:
            by_stem[lf.stem]                    # default entry, no wh
    for stem, s in sorted(by_stem.items()):
        tags = []
        if "__" in stem:
            tags.append("collection:" + stem.split("__", 1)[0])
        if s["w"]:
            tags.append("w_exposure:" + s["w"])
        if s["m"]:
            tags.append("m_exposure:" + s["m"])
        if s["teen"]:
            tags.append("teen_band")
        lf = label_dir / f"{stem}.txt"
        wh = s["wh"]
        if wh and lf.exists():
            boxes = seg_boxes(lf, wh[0], wh[1]) or []
            scale = MODEL_INPUT / max(wh[0], wh[1], 1)
            bands = defaultdict(set)
            for cls, x1, y1, x2, y2 in boxes:
                if cls == WOMAN:
                    bands["w_size"].add(size_band((y2 - y1) * scale))
                elif cls == MAN:
                    bands["m_size"].add(size_band((y2 - y1) * scale))
            for pfx, names in bands.items():
                tags += [f"{pfx}:{n}" for n in sorted(names)]
        slices[stem] = tags
        for t in tags:
            census[t] += 1

    meta = {
        "verdicts": str(args.verdicts), "gt_labels": str(args.gt_labels),
        "n_verdict_rows": n_rows, "n_images": len(slices),
        "rules": {"tiers": {
                      "t0_covered": "exposed subset of " + str(sorted(T0_SET)),
                      "t1_modest": "+ " + str(sorted(T1_EXTRA)),
                      "t2_ordinary": "+ " + str(sorted(T2_EXTRA)),
                      "t3_revealing": "chest w/o midriff, or legs w/o chest,"
                                      " or back",
                      "t4_high": str(sorted(T4_PARTS)) + " or chest+legs; "
                                 "also any unrecognized part"},
                  "teen_band_est_age": [TEEN_LO, TEEN_HI],
                  "size_bands_input_px": [list(b[:2]) + [b[2]]
                                          for b in SIZE_BANDS],
                  "model_input_px": MODEL_INPUT,
                  "unknown_parts_bucketed_high": sorted(unknown_parts)},
        "census": dict(sorted(census.items())),
    }
    out = {"_meta": meta, "slices": slices}
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))
        print(f"[curate] wrote {args.out}")
    print(f"[curate] {len(slices)} images, {n_rows} verdict rows")
    if unknown_parts:
        print(f"[curate] WARNING unrecognized parts bucketed HIGH: "
              f"{sorted(unknown_parts)}")
    for tag, n in sorted(census.items()):
        print(f"  {tag:28s} {n}")
    return out


# -------------------------------------------------------------- materialize
def materialize(args):
    """slices.json -> per-tag subset folders of SYMLINKS (no data copied)."""
    sl = json.loads(Path(args.slices).read_text())
    tag_of = sl["slices"]
    img_dir, lbl_dir = Path(args.images), Path(args.labels)
    out = Path(args.out)
    want = set(args.tags.split(",")) if args.tags else None
    # stem -> image filename index (images carry mixed extensions)
    img_of = {}
    for p in img_dir.iterdir():
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            img_of.setdefault(p.stem, p.name)
    by_tag = defaultdict(list)
    for stem, tags in tag_of.items():
        for t in tags:
            if want is None or t in want:
                by_tag[t].append(stem)
    counts = {}
    for tag, stems in sorted(by_tag.items()):
        d = out / tag.replace(":", "_")
        (d / "images").mkdir(parents=True, exist_ok=True)
        (d / "labels").mkdir(parents=True, exist_ok=True)
        n_i = n_l = 0
        for s in stems:
            iname = img_of.get(s)
            if iname:
                dst = d / "images" / iname
                if not dst.is_symlink():
                    dst.symlink_to((img_dir / iname).resolve())
                n_i += 1
            lsrc = lbl_dir / f"{s}.txt"
            if lsrc.exists():
                dst = d / "labels" / f"{s}.txt"
                if not dst.is_symlink():
                    dst.symlink_to(lsrc.resolve())
                n_l += 1
        counts[tag] = {"images": n_i, "labels": n_l}
        print(f"  {tag:28s} -> {d}  ({n_i} img links, {n_l} label links)")
    (out / "_meta.json").write_text(json.dumps(
        {"slices_file": str(args.slices), "images": str(img_dir),
         "labels": str(lbl_dir), "note": "all files are symlinks; "
         "dereference with cp -rL if moving off-volume",
         "counts": counts}, indent=1))
    print(f"[materialize] {len(by_tag)} subset folders under {out} "
          f"(symlinks only, no data copied)")


# --------------------------------------------------- rectangle-union geometry
def _union_area_within(win, rects) -> float:
    """Exact area of union(rects) clipped to window `win` (both xyxy)."""
    clipped = []
    for x1, y1, x2, y2 in rects:
        a, b = max(x1, win[0]), max(y1, win[1])
        c, d = min(x2, win[2]), min(y2, win[3])
        if c > a and d > b:
            clipped.append((a, b, c, d))
    if not clipped:
        return 0.0
    xs = sorted({v for r in clipped for v in (r[0], r[2])})
    ys = sorted({v for r in clipped for v in (r[1], r[3])})
    area = 0.0
    for i in range(len(xs) - 1):
        cx = (xs[i] + xs[i + 1]) / 2
        cols = [r for r in clipped if r[0] <= cx < r[2]]
        if not cols:
            continue
        w = xs[i + 1] - xs[i]
        for j in range(len(ys) - 1):
            cy = (ys[j] + ys[j + 1]) / 2
            if any(r[1] <= cy < r[3] for r in cols):
                area += w * (ys[j + 1] - ys[j])
    return area


def covered_frac(gt_box, blur_rects) -> float:
    area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    if area <= 0:
        return 1.0
    return _union_area_within(gt_box, blur_rects) / area


def dilate(box, frac):
    x1, y1, x2, y2 = box
    d = frac * ((x2 - x1) * (y2 - y1)) ** 0.5
    return (x1 - d, y1 - d, x2 + d, y2 + d)


# ------------------------------------------------------------------- scoring
def image_metrics(gt_boxes, ignore_boxes, preds, target, img_w, img_h, thr):
    """One image, one mode, one confidence cut -> per-image record.

    gt_boxes:      [(cls, x1, y1, x2, y2)]
    ignore_boxes:  [(x1, y1, x2, y2)]
    preds:         [(cls, x1, y1, x2, y2, conf)]  (log-raw, any conf)
    target:        blur-target class for this mode
    thr:           dict(tau, phi, kappa, dilate, conf)
    """
    img_area = float(img_w * img_h)
    blur = [(x1, y1, x2, y2) for c, x1, y1, x2, y2, cf in preds
            if c == target and cf >= thr["conf"]]
    gt_t = [(x1, y1, x2, y2) for c, x1, y1, x2, y2 in gt_boxes if c == target]
    gt_child = [(x1, y1, x2, y2) for c, *r in gt_boxes if c == CHILD
                for x1, y1, x2, y2 in [r]]

    # M-1: exposure (area-weighted uncovered target pixels / image area),
    # plus the person-level view: uncovered fraction per GT person
    # (size-independent — every person counts equally).
    exposure = 0.0
    person_uncov = []
    for g in gt_t:
        area = (g[2] - g[0]) * (g[3] - g[1])
        uncov = 1.0 - covered_frac(g, blur)
        person_uncov.append(round(uncov, 4))
        exposure += uncov * area / img_area
    m1_pass = exposure <= thr["tau"] if gt_t else None

    # M-2: false blur = predicted-blur area outside dilated GT-target boxes
    # and outside ignore regions.
    allowed = [dilate(g, thr["dilate"]) for g in gt_t] + list(ignore_boxes)
    false_area = 0.0
    fp_boxes = 0
    if blur:
        win = (0.0, 0.0, float(img_w), float(img_h))
        blur_a = _union_area_within(win, blur)
        # covered-by-allowed part of the blur union: union(blur) ∩ union(allowed)
        inter = 0.0
        if allowed:
            xs = sorted({v for r in blur + allowed for v in (r[0], r[2])})
            ys = sorted({v for r in blur + allowed for v in (r[1], r[3])})
            for i in range(len(xs) - 1):
                cx = (xs[i] + xs[i + 1]) / 2
                bcols = [r for r in blur if r[0] <= cx < r[2]]
                if not bcols:
                    continue
                acols = [r for r in allowed if r[0] <= cx < r[2]]
                if not acols:
                    continue
                w = xs[i + 1] - xs[i]
                for j in range(len(ys) - 1):
                    cy = (ys[j] + ys[j + 1]) / 2
                    if (any(r[1] <= cy < r[3] for r in bcols)
                            and any(r[1] <= cy < r[3] for r in acols)):
                        inter += w * (ys[j + 1] - ys[j])
        false_area = max(0.0, blur_a - inter)
        for b in blur:
            barea = (b[2] - b[0]) * (b[3] - b[1])
            if barea <= 0:
                continue
            inside = _union_area_within(b, allowed) if allowed else 0.0
            if inside / barea < 0.5:
                fp_boxes += 1
    fb_fail = (false_area / img_area) > thr["phi"]

    # B-1: child clarity (blurred child = covered beyond kappa) + the
    # continuous view (covered fraction per child)
    child_cov = [round(covered_frac(g, blur), 4) for g in gt_child]
    child_blurred = any(c > thr["kappa"] for c in child_cov)
    child_pass = (not child_blurred) if gt_child else None

    return {"exposure": round(exposure, 6), "m1_pass": m1_pass,
            "person_uncov": person_uncov,
            "false_area_frac": round(false_area / img_area, 6),
            "fb_fail": fb_fail, "fp_boxes": fp_boxes,
            "child_pass": child_pass, "child_cov": child_cov,
            "n_gt_target": len(gt_t), "n_gt_child": len(gt_child)}


def aggregate(records, tag_of, tags=("all",)):
    """records: {stem: per-image record}. Returns {tag: metric dict}."""
    out = {}
    for tag in tags:
        stems = [s for s in records
                 if tag == "all" or tag in tag_of.get(s, ())]
        tgt = [records[s] for s in stems if records[s]["m1_pass"] is not None]
        chd = [records[s] for s in stems
               if records[s]["child_pass"] is not None
               and "teen_band" not in tag_of.get(s, ())]
        n = len(stems)
        expos = sorted(r["exposure"] for r in tgt)
        uncov = [u for r in tgt for u in r.get("person_uncov", ())]
        ccov = [c for s in stems for c in records[s].get("child_cov", ())
                if "teen_band" not in tag_of.get(s, ())]
        out[tag] = {
            "n_images": n,
            "n_target_images": len(tgt),
            "m1_pass_rate": (sum(r["m1_pass"] for r in tgt) / len(tgt)
                             if tgt else None),
            # image view: mean exposed screen fraction + tail (threshold-free)
            "mean_exposure": (sum(expos) / len(expos) if expos else None),
            "exposure_p90": (expos[int(0.9 * (len(expos) - 1))]
                             if expos else None),
            # person view: mean uncovered fraction + share fully covered
            "n_persons": len(uncov),
            "person_uncov_mean": (sum(uncov) / len(uncov)
                                  if uncov else None),
            "person_covered90_rate": (sum(1 for u in uncov if u <= 0.1)
                                      / len(uncov) if uncov else None),
            "falseblur_img_rate": (sum(r["fb_fail"] for r in
                                       (records[s] for s in stems)) / n
                                   if n else None),
            # continuous false-blur: mean wrongly-blurred screen fraction
            "fb_area_mean": (sum(records[s]["false_area_frac"]
                                 for s in stems) / n if n else None),
            "fp_boxes_per_100": (100.0 * sum(records[s]["fp_boxes"]
                                             for s in stems) / n
                                 if n else None),
            "n_child_images": len(chd),
            "child_clarity": (sum(r["child_pass"] for r in chd) / len(chd)
                              if chd else None),
            "child_blur_mean": (sum(ccov) / len(ccov) if ccov else None),
        }
    return out


# ------------------------------------------------------------------ data IO
def load_sidecars(raw_dir: Path, workers: int = 16):
    """{stem: (w, h, [(cls,x1,y1,x2,y2,conf)])} from log-raw sidecars.
    Reads are THREADED (network-volume small files are latency-bound; the
    serial form cost ~10 min per model on a cold volume). Volume-corrupt
    files (ENXIO etc.) are skipped with a warning — the stems-intersection
    in score() then drops that image for ALL models, so the comparison
    stays like-for-like."""
    from concurrent.futures import ThreadPoolExecutor

    def _read(p):
        try:
            return p, json.loads(p.read_text())
        except OSError as e:
            print(f"[load] WARNING unreadable sidecar skipped: {p} ({e})")
            return p, None
    out = {}
    with ThreadPoolExecutor(workers) as ex:
        for p, d in ex.map(_read, sorted(raw_dir.glob("*.json"))):
            if d is None:
                continue
            dets = [(r["cls"], *r["box_xyxy"], r["conf"])
                    for r in d.get("detections", []) if not r.get("excluded")]
            out[p.stem] = (d["width"], d["height"], dets)
    return out


def load_ignores(ignore_dir, stem, w, h):
    if not ignore_dir:
        return []
    rows = seg_boxes(Path(ignore_dir) / f"{stem}.txt", w, h)
    return [(x1, y1, x2, y2) for _c, x1, y1, x2, y2 in rows or []]


def bootstrap_ci(passes_a, passes_b=None, n_boot=5000, seed=20260825):
    """95% CI on mean(passes_a) and, if passes_b given (paired, same images),
    on mean(a) - mean(b). Resamples images."""
    try:
        import numpy as np
    except ImportError:
        return None
    a = np.asarray(passes_a, dtype=float)
    if a.size == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    means = a[idx].mean(axis=1)
    res = {"mean": float(a.mean()),
           "ci95": [float(np.percentile(means, 2.5)),
                    float(np.percentile(means, 97.5))]}
    if passes_b is not None:
        b = np.asarray(passes_b, dtype=float)
        diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
        res["diff_mean"] = float(a.mean() - b.mean())
        res["diff_ci95"] = [float(np.percentile(diffs, 2.5)),
                            float(np.percentile(diffs, 97.5))]
    return res


# -------------------------------------------------------------------- score
def score(args):
    thr_base = {"tau": args.tau, "phi": args.phi, "kappa": args.kappa,
                "dilate": args.dilate}
    grid = sorted(set(args.grid_values + [args.anchor_conf]))
    models = []
    for spec in args.pred:
        name, _, raw = spec.partition("=")
        if not raw:
            sys.exit(f"--pred wants name=rawdir, got: {spec}")
        models.append((name, Path(raw)))

    slices, tag_of = ["all"], {}
    if args.slices:
        sl = json.loads(Path(args.slices).read_text())
        tag_of = {s: set(t) for s, t in sl["slices"].items()}
        slices += sorted({t for ts in tag_of.values() for t in ts})

    gt_dir = Path(args.gt_labels)
    sidecars = {name: load_sidecars(raw) for name, raw in models}
    stems = None
    for name, sc in sidecars.items():
        print(f"[score] {name}: {len(sc)} sidecars")
        stems = set(sc) if stems is None else stems & set(sc)
    gt_stems = {p.stem for p in gt_dir.glob("*.txt")}
    dropped = len(stems) - len(stems & gt_stems)
    stems = sorted(stems & gt_stems)
    print(f"[score] scoring {len(stems)} images present in ALL models + GT"
          + (f" (dropped {dropped} without GT)" if dropped else ""))
    if not stems:
        sys.exit("no overlapping images between models and GT")

    # cache GT + ignore per stem using the first model's sidecar dims
    gt_cache, ign_cache = {}, {}
    ref = sidecars[models[0][0]]
    for s in stems:
        w, h, _ = ref[s]
        gt_cache[s] = seg_boxes(gt_dir / f"{s}.txt", w, h) or []
        ign_cache[s] = load_ignores(args.ignore, s, w, h)

    modes = ["male", "female"] if args.mode == "both" else [args.mode]
    summary = {"protocol": {**thr_base, "anchor_conf": args.anchor_conf,
                            "grid": grid, "incumbent": models[0][0],
                            "gt_labels": str(gt_dir),
                            "ignore": str(args.ignore) if args.ignore else None,
                            "slices_file": str(args.slices) if args.slices
                            else None, "n_images": len(stems)},
               "modes": {}}
    out_dir = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    per_image_f = (open(out_dir / "per_image.jsonl", "w") if out_dir else None)

    def run_model(name, target, cf):
        thr = {**thr_base, "conf": cf}
        recs = {}
        for s in stems:
            w, h, dets = sidecars[name][s]
            recs[s] = image_metrics(gt_cache[s], ign_cache[s], dets,
                                    target, w, h, thr)
        return recs

    for mode in modes:
        target = MODES[mode]
        # pass 1: sweep the grid, keep only aggregates (memory: the full
        # per-image record set for every model x conf does not fit).
        # ALL slices are aggregated at every conf so per-slice matched
        # operating points come free afterwards.
        curves_full = {}     # model -> conf -> {tag: aggregate}
        for name, _ in models:
            curves_full[name] = {}
            for cf in grid:
                curves_full[name][cf] = aggregate(
                    run_model(name, target, cf), tag_of, tags=slices)
        curves = {n: {cf: curves_full[n][cf]["all"] for cf in grid}
                  for n, _ in models}

        inc_name = models[0][0]
        # match on the CONTINUOUS false-blur area (threshold-free), not the
        # phi-thresholded image rate — smoother, no denominator games
        fb0 = curves[inc_name][args.anchor_conf]["fb_area_mean"]
        matched = {inc_name: args.anchor_conf}
        for name, _ in models[1:]:
            matched[name] = min(
                grid, key=lambda cf:
                (abs(curves[name][cf]["fb_area_mean"] - fb0), cf))
        # per-slice matched operating points: each slice gets its own conf
        # per model, matched to the incumbent's false-blur area ON THAT SLICE
        # (global matching over-fires small-image slices — this is the
        # honest per-category comparison)
        slice_matched = {}
        for tag in slices:
            fb0_t = curves_full[inc_name][args.anchor_conf][tag]["fb_area_mean"]
            if fb0_t is None:
                continue
            row = {}
            for name, _ in models:
                cf_t = args.anchor_conf if name == inc_name else min(
                    grid, key=lambda cf: (
                        abs((curves_full[name][cf][tag]["fb_area_mean"]
                             or 0.0) - fb0_t), cf))
                row[name] = {"conf": cf_t, **curves_full[name][cf_t][tag]}
            slice_matched[tag] = row

        # pass 2: per-image records only at each model's matched point
        recs_at = {name: {matched[name]: run_model(name, target,
                                                   matched[name])}
                   for name, _ in models}

        mode_out = {"target_class": target, "matched_conf": matched,
                    "slice_matched": slice_matched,
                    "incumbent_falseblur_at_anchor": fb0,
                    "models": {}, "curves": {
                        n: {str(cf): curves[n][cf] for cf in grid}
                        for n, _ in models}}
        inc_recs = recs_at[inc_name][matched[inc_name]]
        inc_tstems = [s for s in stems if inc_recs[s]["m1_pass"] is not None]
        for name, _ in models:
            cf = matched[name]
            recs = recs_at[name][cf]
            agg = aggregate(recs, tag_of, tags=slices)
            ci = bootstrap_ci(
                [float(recs[s]["m1_pass"]) for s in inc_tstems],
                None if name == inc_name else
                [float(inc_recs[s]["m1_pass"]) for s in inc_tstems],
                n_boot=args.bootstrap)
            # continuous headline: mean exposure (lower = better; a NEGATIVE
            # diff vs the incumbent is an improvement)
            ci_exp = bootstrap_ci(
                [recs[s]["exposure"] for s in inc_tstems],
                None if name == inc_name else
                [inc_recs[s]["exposure"] for s in inc_tstems],
                n_boot=args.bootstrap)
            mode_out["models"][name] = {"conf": cf, "slices": agg,
                                        "m1_bootstrap": ci,
                                        "exposure_bootstrap": ci_exp}
            if per_image_f:
                for s in stems:
                    per_image_f.write(json.dumps(
                        {"image": s, "mode": mode, "model": name, "conf": cf,
                         **recs[s], "tags": sorted(tag_of.get(s, ()))}) + "\n")
        summary["modes"][mode] = mode_out

        # ---- printed ship table
        print(f"\n=== {mode.upper()} audience (blur target: "
              f"{'Woman' if target == WOMAN else 'Man'}) — matched false-blur"
              f" (incumbent {inc_name} @ {args.anchor_conf}) ===")
        hdr = f"{'model':28s} {'conf':>5s} {'Eimg%':>7s} {'P90%':>6s} " \
              f"{'Epers%':>7s} {'cov90%':>7s} {'FBarea%':>8s} " \
              f"{'chblur%':>8s} {'M1@tau':>7s}"
        print(hdr)
        for name, _ in models:
            a = mode_out["models"][name]["slices"]["all"]
            print(f"{name:28s} {matched[name]:>5.2f} "
                  f"{_pct(a['mean_exposure'], 2):>7s} "
                  f"{_pct(a['exposure_p90'], 1):>6s} "
                  f"{_pct(a['person_uncov_mean'], 1):>7s} "
                  f"{_pct(a['person_covered90_rate']):>7s} "
                  f"{_pct(a['fb_area_mean'], 3):>8s} "
                  f"{_pct(a['child_blur_mean'], 1):>8s} "
                  f"{_pct(a['m1_pass_rate']):>7s}")
        interesting = [t for t in slices if t != "all"
                       and (t.startswith(("collection:",
                                          "w_" if mode == "male" else "m_"))
                            )]
        for tag in interesting:
            row = " | ".join(
                f"{n}: {_pct(mode_out['models'][n]['slices'][tag]['person_uncov_mean'])}"
                f"/{_pct(mode_out['models'][n]['slices'][tag]['fb_area_mean'], 2)}"
                for n, _ in models)
            n_img = mode_out["models"][inc_name]["slices"][tag]["n_images"]
            print(f"  slice {tag:24s} (n={n_img:5d})  Epers/FBarea:  {row}")

    if per_image_f:
        per_image_f.close()
    if out_dir:
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
        print(f"\n[score] wrote {out_dir}/summary.json and per_image.jsonl")
    return summary


def _pct(v, nd=1):
    return "-" if v is None else f"{100 * v:.{nd}f}"


def _f(v, nd):
    return "-" if v is None else f"{v:.{nd}f}"


# ----------------------------------------------------------------- selftest
def selftest():
    import tempfile
    ok = 0

    def check(cond, msg):
        nonlocal ok
        assert cond, f"SELFTEST FAIL: {msg}"
        ok += 1

    thr = {"tau": 0.005, "phi": 0.01, "kappa": 0.2, "dilate": 0.1,
           "conf": 0.45}
    W = H = 1000
    gt_w = [(WOMAN, 100, 100, 500, 900)]

    # geometry: two half-boxes that jointly cover the woman -> exposure 0
    halves = [(WOMAN, 100, 100, 500, 500, 0.9), (WOMAN, 100, 500, 500, 900, 0.9)]
    r = image_metrics(gt_w, [], halves, WOMAN, W, H, thr)
    check(abs(r["exposure"]) < 1e-9 and r["m1_pass"],
          "union of partial boxes covers the person")
    check(not r["fb_fail"], "covering boxes are not false blur")

    # over-blur is free; leak is proportional
    big = [(WOMAN, 90, 90, 510, 910, 0.9)]
    r = image_metrics(gt_w, [], big, WOMAN, W, H, thr)
    check(r["exposure"] == 0 and not r["fb_fail"] and r["fp_boxes"] == 0,
          "generous box inside dilation: free")
    clipped = [(WOMAN, 100, 100, 500, 580, 0.9)]     # covers 60% of height
    r = image_metrics(gt_w, [], clipped, WOMAN, W, H, thr)
    exp_expected = 0.4 * (400 * 800) / (W * H)
    check(abs(r["exposure"] - exp_expected) < 1e-6 and not r["m1_pass"],
          "partial blur charges exactly the uncovered pixels")
    check(r["person_uncov"] == [0.4],
          "person view: uncovered fraction is size-independent (40%)")

    # wrong gender = same failure as a miss (the M-1-subsumes-M-2 property)
    man_pred = [(MAN, 100, 100, 500, 900, 0.9)]
    r = image_metrics(gt_w, [], man_pred, WOMAN, W, H, thr)
    check(not r["m1_pass"], "woman called Man escapes in male mode")
    check(not r["fb_fail"], "a Man box cannot false-blur a male user")
    # ...but for the FEMALE audience that same Man box IS a false blur
    r = image_metrics(gt_w, [], man_pred, MAN, W, H, thr)
    check(r["fb_fail"] and r["fp_boxes"] == 1,
          "Man box on a GT woman is a false blur for the female audience")

    # woman called Child escapes too (Child never blurs)
    r = image_metrics(gt_w, [], [(CHILD, 100, 100, 500, 900, 0.9)],
                      WOMAN, W, H, thr)
    check(not r["m1_pass"], "woman called Child escapes")

    # phantom blur on empty image; ignore region absolves it
    phantom = [(WOMAN, 200, 200, 600, 600, 0.9)]
    r = image_metrics([], [], phantom, WOMAN, W, H, thr)
    check(r["fb_fail"] and r["fp_boxes"] == 1, "phantom woman-blur fails M-2")
    r = image_metrics([], [(190, 190, 610, 610)], phantom, WOMAN, W, H, thr)
    check(not r["fb_fail"] and r["fp_boxes"] == 0,
          "prediction on an ignore region is neither credit nor false blur")

    # child clarity
    gt_c = [(CHILD, 300, 300, 600, 800)]
    r = image_metrics(gt_c, [], [(WOMAN, 300, 300, 600, 800, 0.9)],
                      WOMAN, W, H, thr)
    check(r["child_pass"] is False and r["fb_fail"],
          "blurred child fails B-1 and counts as false blur")
    r = image_metrics(gt_c, [], [], WOMAN, W, H, thr)
    check(r["child_pass"] is True and r["m1_pass"] is None,
          "no target GT -> M-1 undefined, child untouched -> B-1 pass")

    # conf gating: sub-threshold prediction does not blur
    r = image_metrics(gt_w, [], [(WOMAN, 100, 100, 500, 900, 0.30)],
                      WOMAN, W, H, thr)
    check(not r["m1_pass"], "sub-conf box does not blur")

    # curate: buckets, teen tag, collection, unknown-part conservatism
    seen: set = set()
    check(exposure_bucket(["face", "hands"], seen) == "t0_covered"
          and exposure_bucket(["face", "hair", "neck"], seen) == "t1_modest"
          and exposure_bucket(["face", "shoulders", "arms"], seen)
          == "t2_ordinary", "lower exposure tiers")
    check(exposure_bucket(["face", "chest"], seen) == "t3_revealing"
          and exposure_bucket(["face", "legs", "feet"], seen) == "t3_revealing"
          and exposure_bucket(["face", "back"], seen) == "t3_revealing",
          "neckline/legs/back are t3, not t4 (the agreed definition)")
    check(exposure_bucket(["face", "midriff"], seen) == "t4_high"
          and exposure_bucket(["chest", "legs", "face"], seen) == "t4_high",
          "midriff or chest+legs are t4")
    check(exposure_bucket(["face", "tentacles"], seen) == "t4_high"
          and "tentacles" in seen, "unknown part tiers t4")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "labels").mkdir()
        # polygon row (seg format) + box row; image 200x400
        (td / "labels" / "women__a.txt").write_text(
            "0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n"   # polygon woman
            "1 0.5 0.5 0.2 0.1\n")                          # box man
        (td / "labels" / "randoms__z.txt").write_text("")   # detection-free
        vd = td / "verdicts.jsonl"
        rows = [
            {"image_stem": "women__a", "img_wh": [200, 400],
             "v": {"verdict": "real_person", "gender": "woman",
                   "age_group": "adult", "estimated_age": 30,
                   "exposed_body_parts": ["face", "chest"]}},
            {"image_stem": "women__a", "img_wh": [200, 400],
             "v": {"verdict": "real_person", "gender": "man",
                   "age_group": "adult", "estimated_age": 14,
                   "exposed_body_parts": ["face"]}},
            {"image_stem": "child__b", "img_wh": [100, 100],
             "v": {"verdict": "not_person", "gender": "unknown",
                   "age_group": "child", "estimated_age": 5,
                   "exposed_body_parts": []}},
        ]
        vd.write_text("\n".join(json.dumps(r) for r in rows))
        ns = argparse.Namespace(verdicts=vd, gt_labels=td / "labels", out=None)
        cur = curate(ns)
        t = cur["slices"]["women__a"]
        check("collection:women" in t and "w_exposure:t3_revealing" in t
              and "m_exposure:t0_covered" in t and "teen_band" in t,
              f"curate tags: {t}")
        # polygon woman: 100x200 px native, image long side 400 -> x1.6 scale
        check("w_size:large" in t and "m_size:small" not in t,
              f"size bands from seg polygon: {t}")
        check("collection:child" in cur["slices"]["child__b"]
              and not any(x.startswith("w_exposure") for x in cur["slices"]["child__b"]),
              "not_person rows create no exposure slice")
        check(cur["slices"].get("randoms__z") == ["collection:randoms"],
              "detection-free GT stems still get their collection tag")

        # materialize: symlink folders, zero copies
        (td / "imgs").mkdir()
        (td / "imgs" / "women__a.jpg").write_bytes(b"\xff\xd8fake")
        slf = td / "sl.json"
        slf.write_text(json.dumps(cur))
        materialize(argparse.Namespace(
            slices=slf, images=td / "imgs", labels=td / "labels",
            out=td / "subs", tags="collection:women"))
        link = td / "subs" / "collection_women" / "images" / "women__a.jpg"
        check(link.is_symlink()
              and link.resolve() == (td / "imgs" / "women__a.jpg").resolve(),
              "materialized subset is a symlink, not a copy")
        check((td / "subs" / "collection_women" / "labels"
               / "women__a.txt").is_symlink(), "label symlinked too")
        check(not (td / "subs" / "collection_child").exists(),
              "--tags filter respected")

        # end-to-end score: incumbent + candidate, matched operating point
        for m, boxes in {
            "inc": [("a", WOMAN, 50, 100, 150, 300, 0.60)],
            "cand": [("a", WOMAN, 50, 100, 150, 300, 0.20)],  # low-calibrated
        }.items():
            raw = td / m / "raw"
            raw.mkdir(parents=True)
            dets = [{"det_index": 0, "cls": c, "cls_name": "w",
                     "box_xyxy": [x1, y1, x2, y2], "conf": cf,
                     "kept": True, "excluded": None}
                    for _s, c, x1, y1, x2, y2, cf in boxes]
            (raw / "women__a.json").write_text(json.dumps(
                {"image": "women__a.jpg", "width": 200, "height": 400,
                 "detections": dets}))
        sl = td / "slices.json"
        sl.write_text(json.dumps(cur))
        # an unreadable "sidecar" (a dir raises OSError on read) is skipped
        (td / "inc" / "raw" / "zzz_corrupt.json").mkdir()
        check(len(load_sidecars(td / "inc" / "raw")) == 1,
              "corrupt sidecar skipped, not fatal")
        args2 = argparse.Namespace(
            gt_labels=td / "labels", ignore=None, slices=sl, out=None,
            pred=[f"inc={td}/inc/raw", f"cand={td}/cand/raw"],
            tau=0.005, phi=0.01, kappa=0.2, dilate=0.1, anchor_conf=0.45,
            grid_values=[0.15, 0.45, 0.75], mode="male", bootstrap=200)
        s = score(args2)
        mm = s["modes"]["male"]
        check(mm["matched_conf"]["inc"] == 0.45, "incumbent at anchor")
        # candidate's 0.20-conf box only blurs at grid 0.15; matched-FP must
        # pick a conf where its false-blur rate matches (0 everywhere here,
        # ties resolve to the LOWEST conf -> 0.15, where it actually detects)
        check(mm["matched_conf"]["cand"] == 0.15,
              f"matched-FP tie resolves low: {mm['matched_conf']}")
        check(mm["slice_matched"]["all"]["inc"]["conf"] == 0.45
              and "person_uncov_mean" in mm["slice_matched"]["all"]["cand"],
              "per-slice matched operating points emitted")
        inc_m1 = mm["models"]["inc"]["slices"]["all"]["m1_pass_rate"]
        cand_m1 = mm["models"]["cand"]["slices"]["all"]["m1_pass_rate"]
        check(inc_m1 == 1.0 and cand_m1 == 1.0,
              "both pass at their matched points (calibration neutralized)")
        check("w_exposure:t3_revealing" in mm["models"]["inc"]["slices"],
              "slice metrics computed")

    print(f"selftest OK ({ok} checks)")


# --------------------------------------------------------------------- main
def _grid(spec: str):
    lo, hi, step = (float(x) for x in spec.split(":"))
    vals, v = [], lo
    while v <= hi + 1e-9:
        vals.append(round(v, 4))
        v += step
    return vals


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("curate", help="build slices.json from verdicts + GT")
    c.add_argument("--verdicts", required=True)
    c.add_argument("--gt-labels", required=True)
    c.add_argument("--out")

    m = sub.add_parser("materialize",
                       help="build per-slice subset folders (symlinks only)")
    m.add_argument("--slices", required=True)
    m.add_argument("--images", required=True)
    m.add_argument("--labels", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--tags", help="comma-separated tag filter (default: all)")

    s = sub.add_parser("score", help="run the ship-table comparison")
    s.add_argument("--gt-labels", required=True)
    s.add_argument("--ignore", help="ignore-region label dir")
    s.add_argument("--slices", help="slices.json from `curate`")
    s.add_argument("--pred", action="append", required=True,
                   metavar="NAME=RAWDIR",
                   help="model sidecar dir; FIRST one is the incumbent")
    s.add_argument("--mode", choices=["male", "female", "both"],
                   default="both")
    s.add_argument("--tau", type=float, default=0.005)
    s.add_argument("--phi", type=float, default=0.01)
    s.add_argument("--kappa", type=float, default=0.2)
    s.add_argument("--dilate", type=float, default=0.1)
    s.add_argument("--anchor-conf", type=float, default=0.45)
    s.add_argument("--grid", default="0.05:0.90:0.05",
                   help="conf grid lo:hi:step (anchor conf always added)")
    s.add_argument("--bootstrap", type=int, default=5000)
    s.add_argument("--out", help="output dir for summary.json + per_image.jsonl")

    sub.add_parser("selftest", help="no data, no GPU, no network")

    args = ap.parse_args()
    if args.cmd == "selftest":
        selftest()
    elif args.cmd == "curate":
        curate(args)
    elif args.cmd == "materialize":
        materialize(args)
    else:
        args.grid_values = _grid(args.grid)
        score(args)


if __name__ == "__main__":
    main()
