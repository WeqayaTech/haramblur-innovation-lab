#!/usr/bin/env python3
"""
Build a small-person benchmark set — the regime the production complaint is
about ("yolo26n misses small/distant people") and the one no staged dataset
currently isolates.

Three subcommands, three different jobs:

  census   How much small-person data do we already have? Reads any GT source
           and prints the person-height distribution in BOTH native pixels and
           model-input pixels, plus how many images qualify at each cutoff.
           Run this FIRST — it decides the cutoffs, no staging, no GPU.

  mine     Stage the real arm: pick images that actually contain small people,
           emit a dataset where people OUTSIDE the size band become IGNORE
           regions rather than being deleted or scored. That makes the number
           a pure small-person recall — a model is neither credited for the
           big person in the foreground nor penalised for it.

  synth    Stage the controlled arm: take images whose people are LARGE and
           confidently labelled, shrink-and-pad them so the tallest person
           lands on a target pixel height, and keep the class labels. This is
           the only arm that can measure gender/age on small people, because
           the labels were earned at a size where they are trustworthy, and it
           isolates apparent size as the single changing variable.

Why both arms: the real arm is realistic but its GT is only as complete as
whoever annotated it (a machine-labelled corpus cannot measure small-person
recall — the labeler's own misses become "no label", and absence of a label is
not evidence of absence). The synth arm has trustworthy labels but clean
downscaled pixels — no motion blur, no haze, correct focus — so it is an upper
bound on real-world small-person performance, not a substitute for the real arm.

Everything emitted is scored by the EXISTING tools with no changes:
    run_ultralytics_labels.py  -> labels + log-raw sidecars
    eval_negatives_crowd.py --mode crowd   (odgt arms, ignores honoured)
    run_autolabel_on_manifest.py / conf_sweep.py / map_eval.py --ignore-labels

    python3 build_small_person_set.py census --source yolo \
        --images /workspace/datasets/haramblur_holdout/labeling/full/images \
        --labels /workspace/datasets/haramblur_holdout/labeling/full/labels_eval
    python3 build_small_person_set.py census --source odgt \
        --images /workspace/datasets/crowdhuman/Images \
        --odgt   /workspace/datasets/crowdhuman/annotation_val.odgt
    python3 build_small_person_set.py --selftest      # no data, no GPU, no network
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Reused rather than reimplemented (repo convention):
#   seg_boxes      — the ONLY correct YOLO reader here; a plain 5-field parser
#                    silently misreads the holdout's segment polygons.
#   load_odgt      — CrowdHuman persons + ignore regions.
#   shrink_and_pad / scale_boxes — the shrink geometry, already selftested.
from run_autolabel_on_manifest import (seg_boxes, load_gt,   # noqa: E402
                                       gt_class)
from eval_negatives_crowd import load_odgt, _area         # noqa: E402
from scale_robustness import shrink_and_pad, scale_boxes  # noqa: E402

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp",
              ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP")
# ^ deliberately includes .webp and uppercase: autolabel_sam_raw.py:174 skips
#   both silently (open production bug recorded in CLAUDE.md). Do not copy that.

DEFAULT_CLASS_NAMES = {0: "Woman", 1: "Man", 2: "Child", 3: "Unknown"}
# production class ids 0/1/2 — the order `gt_class` returns names in
DEFAULT_CLASS_NAMES_ORDER = ("Woman", "Man", "Child")

# Reporting bands, in whichever space the run selects on. The 96 px edge is
# the product question ("people spanning <=96 px"); it also happens to be
# COCO's medium/large area edge, so numbers stay comparable to mAP_medium.
BANDS = ((0, 16, "tiny"), (16, 32, "verysmall"), (32, 64, "small"),
         (64, 96, "medium"), (96, 10 ** 9, "large"))


def load_exclude(path) -> set:
    """Stems to keep OUT of the new set — e.g. the seed-51 `Images_sample500`
    that every previous crowd number was measured on. Keeping the new set
    disjoint from it means small-person results are not read off images the
    protocol has already been tuned against. Accepts a file of stems/paths or
    a directory of images."""
    if not path:
        return set()
    p = Path(path)
    if p.is_dir():
        return {q.stem for q in list_images(p)}
    return {Path(line.strip()).stem for line in p.read_text().splitlines()
            if line.strip()}


def band_of(h: float) -> str:
    for lo, hi, name in BANDS:
        if lo <= h < hi:
            return name
    return "large"


def input_height(h_px: float, w: int, h: int, imgsz: int) -> float:
    """Person height as the MODEL sees it, after aspect-preserving letterbox
    to `imgsz` on the long side. A 96 px person in a 4000 px-wide photo is
    15 px to the network; in a 640 px photo it is 96 px. Native pixels alone
    are not a difficulty measure across datasets of different resolutions."""
    long_side = max(w, h)
    if long_side <= 0:
        return 0.0
    return h_px * imgsz / long_side


def image_size(path: Path):
    """(w, h) from the header only — no decode."""
    from PIL import Image
    with Image.open(path) as im:
        return im.size


def list_images(images_dir: Path):
    return sorted(p for p in Path(images_dir).iterdir()
                  if p.suffix in IMAGE_EXTS)


# ---------------------------------------------------------------------------
# sources -> a uniform record per image
#   {stem, path, w, h, persons: [{box(xyxy px), cls, h_px, h_in}], ignores: [box]}
# ---------------------------------------------------------------------------
def _record(stem, path, w, h, boxes, ignores, imgsz):
    persons = []
    for cls, x1, y1, x2, y2 in boxes:
        hp = y2 - y1
        persons.append({"box": [x1, y1, x2, y2], "cls": cls, "h_px": hp,
                        "h_in": input_height(hp, w, h, imgsz)})
    return {"stem": stem, "path": str(path), "w": w, "h": h,
            "persons": persons, "ignores": [list(b) for b in ignores]}


def load_yolo_source(images_dir: Path, labels_dir: Path, imgsz: int,
                     ignore_dir: Path | None = None, limit: int = 0,
                     log=print):
    """YOLO labels (boxes OR segment polygons). Works for the holdout
    (labels_eval + ignore), Spotlight OIV7 val, LAGENDA v2 labels_3class."""
    out, skipped = [], 0
    for i, img in enumerate(list_images(images_dir)):
        if limit and len(out) >= limit:
            break
        boxes = seg_boxes(Path(labels_dir) / f"{img.stem}.txt", 1, 1)
        if boxes is None:                 # unprocessed != empty
            skipped += 1
            continue
        try:
            w, h = image_size(img)
        except Exception:
            skipped += 1
            continue
        boxes = [(c, x1 * w, y1 * h, x2 * w, y2 * h) for c, x1, y1, x2, y2 in boxes]
        ign = []
        if ignore_dir:
            ib = seg_boxes(Path(ignore_dir) / f"{img.stem}.txt", 1, 1) or []
            ign = [[x1 * w, y1 * h, x2 * w, y2 * h] for _c, x1, y1, x2, y2 in ib]
        out.append(_record(img.stem, img, w, h, boxes, ign, imgsz))
        if log and (i + 1) % 2000 == 0:
            log(f"  ...{i + 1} images read")
    if skipped and log:
        log(f"  ({skipped} images had no label file or were unreadable)")
    return out


def person_height(src: dict, size_box: str) -> float:
    """How tall this person is, for the purpose of calling them small.

    `vbox` is the visible extent, `fbox` the estimated full body. CrowdHuman is
    not self-consistent about the two — in the mined set 548 people carry a
    vbox TALLER than their fbox, the worst being 622 px of visible person on a
    "<=96 px" full-body box. A person is never smaller than the part of them
    you can see, so full-body sizing takes the larger of the two."""
    h_f = src["fbox"][3] - src["fbox"][1]
    h_v = src["vbox"][3] - src["vbox"][1]
    return max(h_f, h_v) if size_box == "fbox" else h_v


def load_odgt_source(images_dir: Path, odgt_path: Path, imgsz: int,
                     limit: int = 0, size_box: str = "fbox", log=print):
    """CrowdHuman: human-exhaustive person boxes. Class is unknown (person
    only), so cls is None and only Component 1 is measurable here.

    Two boxes exist per person and they mean different things. `vbox` is the
    VISIBLE extent, `fbox` the estimated FULL body. Matching always uses vbox
    (what the protocol scores against), but the SIZE axis is chosen by
    `size_box`, and the choice matters: selecting on vbox pulls in large people
    who happen to be 90% hidden, so "small" would silently mean "small OR
    occluded" and the occlusion breakdown could not separate them. Default
    `fbox` = genuinely small people, with occlusion left as its own axis."""
    recs = load_odgt(Path(odgt_path))
    by_stem = {p.stem: p for p in list_images(images_dir)}
    out, missing = [], 0
    for i, (image_id, r) in enumerate(sorted(recs.items())):
        if limit and len(out) >= limit:
            break
        img = by_stem.get(image_id)
        if img is None:
            missing += 1
            continue
        try:
            w, h = image_size(img)
        except Exception:
            missing += 1
            continue
        boxes = [(None, *p["vbox"]) for p in recs[image_id]["persons"]]
        rec = _record(image_id, img, w, h, boxes, r["ignores"], imgsz)
        for person, src in zip(rec["persons"], recs[image_id]["persons"]):
            person["occ_ratio"] = src["occ_ratio"]
            person["fbox"] = src["fbox"]
            person["h_px"] = person_height(src, size_box)
            person["h_in"] = input_height(person["h_px"], w, h, imgsz)
        out.append(rec)
        if log and (i + 1) % 2000 == 0:
            log(f"  ...{i + 1} annotations read")
    if missing and log:
        log(f"  ({missing} annotated ids had no image on disk)")
    return out


def load_source(args, log=print):
    if args.source == "yolo":
        if not args.labels:
            raise SystemExit("--source yolo needs --labels")
        return load_yolo_source(Path(args.images), Path(args.labels), args.imgsz,
                                Path(args.ignore_labels) if args.ignore_labels else None,
                                args.limit, log)
    if not args.odgt:
        raise SystemExit("--source odgt needs --odgt")
    return load_odgt_source(Path(args.images), Path(args.odgt), args.imgsz,
                            args.limit, args.size_box, log)


# ---------------------------------------------------------------------------
# census
# ---------------------------------------------------------------------------
def heights(recs, space):
    key = "h_px" if space == "native" else "h_in"
    return [p[key] for r in recs for p in r["persons"]]


def percentiles(vals, ps=(5, 10, 25, 50, 75, 90, 95)):
    if not vals:
        return {}
    s = sorted(vals)
    return {f"p{p}": round(s[min(len(s) - 1, int(len(s) * p / 100))], 1) for p in ps}


def census(recs, space, cutoffs, imgsz, log=print):
    key = "h_px" if space == "native" else "h_in"
    hs = heights(recs, space)
    band_counts = Counter(band_of(h) for h in hs)
    rows = []
    for c in cutoffs:
        n_any = sum(1 for r in recs if any(p[key] <= c for p in r["persons"]))
        n_all = sum(1 for r in recs
                    if r["persons"] and all(p[key] <= c for p in r["persons"]))
        n_people = sum(1 for h in hs if h <= c)
        rows.append({"cutoff_px": c, "images_with_any": n_any,
                     "images_all_small": n_all, "persons_at_or_below": n_people})
    summary = {"space": space, "imgsz": imgsz, "n_images": len(recs),
               "n_persons": len(hs), "percentiles_px": percentiles(hs),
               "bands": dict(band_counts), "cutoffs": rows}
    if log:
        log(f"\n{len(recs)} images, {len(hs)} persons — heights in "
            f"{'NATIVE pixels' if space == 'native' else f'MODEL-INPUT pixels (letterbox {imgsz})'}")
        log(f"  percentiles: {summary['percentiles_px']}")
        log("  bands: " + "  ".join(f"{n}={band_counts.get(n, 0)}"
                                    for _lo, _hi, n in BANDS))
        log(f"\n  {'cutoff':>8} {'persons<=':>10} {'imgs with any':>14} {'imgs all-small':>15}")
        for r in rows:
            log(f"  {r['cutoff_px']:>8} {r['persons_at_or_below']:>10} "
                f"{r['images_with_any']:>14} {r['images_all_small']:>15}")
    return summary


# ---------------------------------------------------------------------------
# mine — real arm
# ---------------------------------------------------------------------------
def select(recs, space, max_h, min_small, require_all, seed, n, exclude=frozenset()):
    """Images qualifying for the small-person band, plus per-image split of
    persons into in-band (scored) and out-of-band (-> ignore)."""
    key = "h_px" if space == "native" else "h_in"
    picked = []
    for r in recs:
        if r["stem"] in exclude:
            continue
        small = [p for p in r["persons"] if p[key] <= max_h]
        big = [p for p in r["persons"] if p[key] > max_h]
        if len(small) < min_small:
            continue
        if require_all and big:
            continue
        picked.append({**r, "small": small, "big": big})
    picked.sort(key=lambda r: r["stem"])
    if n and len(picked) > n:
        random.Random(seed).shuffle(picked)
        picked = sorted(picked[:n], key=lambda r: r["stem"])
    return picked


def to_yolo_line(cls, box, w, h):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
    bw, bh = (x2 - x1) / w, (y2 - y1) / h
    return (f"{int(cls)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")


def link_or_copy(src: Path, dst: Path, copy: bool):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        import shutil
        shutil.copy2(src, dst)
    else:
        os.symlink(os.path.realpath(src), dst)


def _xywh(b):
    """xyxy -> odgt's xywh."""
    return [b[0], b[1], b[2] - b[0], b[3] - b[1]]


def emit_mine(picked, out_dir: Path, source_kind: str, copy: bool, meta: dict,
              log=print):
    """images/ + labels/ (in-band people) + ignore/ (out-of-band people and
    any inherited ignore regions) + persons.odgt for the crowd scorer."""
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    (out_dir / "ignore").mkdir(parents=True, exist_ok=True)
    (out_dir / "annotations").mkdir(parents=True, exist_ok=True)

    odgt_lines, n_small, n_ign = [], 0, 0
    for r in picked:
        src = Path(r["path"])
        link_or_copy(src, out_dir / "images" / src.name, copy)
        w, h = r["w"], r["h"]
        lab = [to_yolo_line(p["cls"] if p["cls"] is not None else 0, p["box"], w, h)
               for p in r["small"]]
        ign = [to_yolo_line(0, p["box"], w, h) for p in r["big"]]
        ign += [to_yolo_line(0, b, w, h) for b in r["ignores"]]
        n_small += len(lab)
        n_ign += len(ign)
        # An empty label file means "processed, found nothing" — never absent.
        (out_dir / "labels" / f"{src.stem}.txt").write_text("\n".join(lab) + ("\n" if lab else ""))
        (out_dir / "ignore" / f"{src.stem}.txt").write_text("\n".join(ign) + ("\n" if ign else ""))
        gtboxes = [{"tag": "person",
                    "vbox": _xywh(p["box"]),
                    # the person's REAL full-body box, not a copy of vbox:
                    # occ_ratio = area(vbox)/area(fbox), so copying it here
                    # would silently report every person as unoccluded and
                    # collapse the light/partial/heavy breakdown.
                    "fbox": _xywh(p.get("fbox") or p["box"]),
                    "extra": {}}
                   for p in r["small"]]
        gtboxes += [{"tag": "person", "extra": {"ignore": 1},
                     "fbox": _xywh(p["box"])} for p in r["big"]]
        gtboxes += [{"tag": "person", "extra": {"ignore": 1},
                     "fbox": _xywh(b)} for b in r["ignores"]]
        odgt_lines.append(json.dumps({"ID": src.stem, "gtboxes": gtboxes}))

    (out_dir / "annotations" / "persons.odgt").write_text("\n".join(odgt_lines) + "\n")
    occ = Counter()
    for r in picked:
        for p in r["small"]:
            if "occ_ratio" in p:
                occ["light" if p["occ_ratio"] >= 0.9
                    else "partial" if p["occ_ratio"] >= 0.5 else "heavy"] += 1
    meta = {**meta, "arm": "mined_real", "source_kind": source_kind,
            "n_images": len(picked), "n_persons_scored": n_small,
            "n_ignore_regions": n_ign, "occlusion_mix": dict(occ)}
    (out_dir / "manifest.json").write_text(json.dumps(meta, indent=2))
    if log:
        log(f"emitted {len(picked)} images, {n_small} in-band persons, "
            f"{n_ign} ignore regions -> {out_dir}")
    return meta


# ---------------------------------------------------------------------------
# synth — controlled arm
# ---------------------------------------------------------------------------
def synth_scale(rec, target_h, space, imgsz):
    """Scale that puts the TALLEST person on `target_h`, so after shrinking no
    person in the image exceeds the target. Returns None if the image is
    already at or below the target (nothing to learn from upscaling it)."""
    key = "h_px" if space == "native" else "h_in"
    tallest = max((p[key] for p in rec["persons"]), default=0.0)
    if tallest <= target_h:
        return None
    return target_h / tallest


def emit_synth(recs, out_dir: Path, targets, space, imgsz, min_source_h,
               n, seed, meta, exclude=frozenset(), log=print):
    from PIL import Image
    key = "h_px" if space == "native" else "h_in"
    pool = [r for r in recs
            if r["stem"] not in exclude
            and r["persons"] and max(p[key] for p in r["persons"]) >= min_source_h]
    pool.sort(key=lambda r: r["stem"])
    if n and len(pool) > n:
        random.Random(seed).shuffle(pool)
        pool = sorted(pool[:n], key=lambda r: r["stem"])

    out_dir = Path(out_dir)
    rows, made = [], 0
    for target in targets:
        # "h96", never "h96.0": a stray ".0" makes every downstream path wrong,
        # and a runner pointed at a non-existent dir silently reports
        # processed=0 rather than failing.
        d = out_dir / f"h{target:g}"
        (d / "images").mkdir(parents=True, exist_ok=True)
        (d / "labels").mkdir(parents=True, exist_ok=True)
        (d / "ignore").mkdir(parents=True, exist_ok=True)
        for r in pool:
            s = synth_scale(r, target, space, imgsz)
            if s is None:
                continue
            img = Image.open(r["path"]).convert("RGB")
            canvas, ox, oy = shrink_and_pad(img, s)
            boxes = [(p["cls"] if p["cls"] is not None else 0, *p["box"])
                     for p in r["persons"]]
            scaled = scale_boxes(boxes, s, ox, oy)
            name = f"{r['stem']}_h{target:g}.jpg"
            canvas.save(d / "images" / name, quality=95)
            w, h = canvas.size
            (d / "labels" / f"{r['stem']}_h{target:g}.txt").write_text(
                "\n".join(to_yolo_line(c, (x1, y1, x2, y2), w, h)
                          for c, x1, y1, x2, y2 in scaled) + "\n")
            (d / "ignore" / f"{r['stem']}_h{target:g}.txt").write_text("")
            ign = [list(b) for b in r["ignores"]]
            if ign:
                ig_scaled = scale_boxes([(0, *b) for b in ign], s, ox, oy)
                (d / "ignore" / f"{r['stem']}_h{target:g}.txt").write_text(
                    "\n".join(to_yolo_line(0, (x1, y1, x2, y2), w, h)
                              for _c, x1, y1, x2, y2 in ig_scaled) + "\n")
            made += 1
            rows.append({"target_h": target, "stem": r["stem"], "scale": round(s, 5),
                         "n_persons": len(scaled)})
        # each height dir is its own self-describing set, so any scorer or
        # verifier can be pointed straight at it
        n_here = sum(1 for x in rows if x["target_h"] == target)
        (d / "manifest.json").write_text(json.dumps(
            {**meta, "arm": "synth_shrunk", "target_h": target,
             "max_height_px": target, "space": space, "imgsz": imgsz,
             "n_images": n_here,
             "images": [x for x in rows if x["target_h"] == target]}, indent=2))
        if log:
            log(f"  h{target:g}: {n_here} images")
    meta = {**meta, "arm": "synth_shrunk", "targets": list(targets),
            "space": space, "min_source_height_px": min_source_h,
            "n_source_images": len(pool), "n_images_emitted": made,
            "caveat": ("LANCZOS-downscaled people are cleaner than real distant "
                       "people (no motion blur, haze, or focus loss), so this arm "
                       "is an upper bound on real small-person performance.")}
    (out_dir / "manifest.json").write_text(json.dumps(
        {**meta, "images": rows}, indent=2))
    if log:
        log(f"emitted {made} synthetic images across {len(targets)} target heights "
            f"-> {out_dir}")
    return meta



# ---------------------------------------------------------------------------
# paste — isolated people on a flat canvas
# ---------------------------------------------------------------------------
def seg_polygons(label_file: Path, w: int, h: int):
    """[(cls, [(x, y), ...]), ...] in pixels. `seg_boxes` deliberately returns
    only the extent; here we need the outline itself, to cut the person out of
    their background. Plain cxcywh rows have no outline and are skipped."""
    if not label_file.exists():
        return None
    out = []
    for line in label_file.read_text().splitlines():
        parts = line.split()
        if len(parts) < 7:                # cls + >=3 points
            continue
        try:
            cls = int(float(parts[0]))
            vals = [float(x) for x in parts[1:]]
        except ValueError:
            continue
        if len(vals) % 2 or len(vals) < 6:
            continue
        pts = [(vals[i] * w, vals[i + 1] * h) for i in range(0, len(vals), 2)]
        out.append((cls, pts))
    return out


def poly_bbox(pts):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def load_paste_people(images_dir: Path, gt_jsonl: Path, gt_labels_dir: Path,
                      polygons_dir: Path, min_source_h: float, match_iou: float,
                      limit: int = 0, log=print):
    """People who carry BOTH a human class and an outline.

    Class comes from LAGENDA's human age+gender (via the project's own
    `gt_class`), geometry from the SAM3 polygon that matches that person's
    human box. Matching is by IoU against the human box, so a polygon that
    drifted onto someone else is dropped rather than silently mislabelled."""
    from run_model_children import match_boxes

    gt_rows = load_gt(gt_jsonl, 0)
    people, stats = [], Counter()
    for i, img in enumerate(list_images(images_dir)):
        if limit and len(people) >= limit:
            break
        rows = gt_rows.get(img.name)
        if not rows:
            stats["no_human_rows"] += 1
            continue
        boxes = seg_boxes(gt_labels_dir / f"{img.stem}.txt", 1, 1)
        polys = seg_polygons(polygons_dir / f"{img.stem}.txt", 1, 1)
        if boxes is None or not polys:
            stats["no_boxes_or_polys"] += 1
            continue
        try:
            w, h = image_size(img)
        except Exception:
            stats["unreadable"] += 1
            continue

        wanted, classes = [], []
        for r in rows:
            if r["box_index"] >= len(boxes):
                continue
            cname = gt_class(r)
            if cname is None:
                stats["undecidable_class"] += 1
                continue
            _c, x1, y1, x2, y2 = boxes[r["box_index"]]
            wanted.append((x1 * w, y1 * h, x2 * w, y2 * h))
            # carry the human age/gender through, not just the collapsed class:
            # it is what lets the emitted set be scored by the project's own
            # gt.jsonl-based scorer with no new scoring code.
            classes.append((cname, r.get("gt_age"), r.get("gt_gender")))
        if not wanted:
            continue

        dets = []
        for _c, pts in polys:
            pp = [(x * w, y * h) for x, y in pts]
            dets.append((0, *poly_bbox(pp), 1.0, pp))
        # match_boxes returns {gt_index: (det, iou)} and only reads d[1:5],
        # so the polygon rides along in the tuple and comes back attached to
        # the detection that actually won the match.
        m = match_boxes(wanted, dets, match_iou)
        for gi, (det, _iou) in m.items():
            pts = det[6]
            bx = poly_bbox(pts)
            if bx[3] - bx[1] < min_source_h:
                stats["too_small_at_source"] += 1
                continue
            people.append({"image": str(img), "pts": pts, "bbox": bx,
                           "cls_name": classes[gi][0],
                           "gt_age": classes[gi][1], "gt_gender": classes[gi][2],
                           "src_h": bx[3] - bx[1]})
        stats["matched"] += len(m)
        if log and (i + 1) % 500 == 0:
            log(f"  ...{i + 1} images scanned, {len(people)} usable people")
    if log:
        log(f"  usable people: {len(people)}   {dict(stats)}")
    return people


def cut_out(person, target_h, min_fill=0.12, max_fill=0.92):
    """Person on transparent background, scaled to `target_h`.

    SAM3 stores one polygon per person, so a multi-part mask (an occluded
    person) is bridge-stitched and the bridge shows as a spike. The fill ratio
    (mask area / bbox area) is the cheap tell: a human silhouette fills roughly
    a fifth to a half of its box, so anything outside the band is rejected
    rather than pasted as a visibly broken cut-out."""
    from PIL import Image, ImageDraw

    x1, y1, x2, y2 = person["bbox"]
    bw, bh = int(round(x2 - x1)), int(round(y2 - y1))
    if bw < 2 or bh < 2:
        return None, "degenerate"
    with Image.open(person["image"]) as im:
        src = im.convert("RGB")
    crop = src.crop((int(x1), int(y1), int(x1) + bw, int(y1) + bh))
    mask = Image.new("L", (bw, bh), 0)
    ImageDraw.Draw(mask).polygon([(px - int(x1), py - int(y1))
                                  for px, py in person["pts"]], fill=255)
    fill = sum(mask.getdata()) / (255.0 * bw * bh)
    if not (min_fill <= fill <= max_fill):
        return None, "implausible_mask"
    scale = target_h / bh
    nw, nh = max(1, int(round(bw * scale))), max(1, int(round(bh * scale)))
    rgba = Image.new("RGBA", (bw, bh))
    rgba.paste(crop, (0, 0), mask)
    return rgba.resize((nw, nh), Image.LANCZOS), None


# ---------------------------------------------------------------------------
# avatars — social-media profile pictures (Facebook / LinkedIn style)
# ---------------------------------------------------------------------------
# Why this arm exists: a feed is full of head-and-shoulders portrait crops, and
# that is a different distribution from the full-body people these detectors
# were trained on. A profile picture is also SHARP — it is a deliberate portrait,
# not a distant figure — so this arm separates "small" from "low quality", which
# every other arm confounds.
HEAD_FRAC = 0.34          # of a person's box height, taken from the top
HEAD_LIFT = 0.04          # nudge the crop above the box top; boxes clip hair


def crop_sharpness(crop_rgb):
    """Detail score: variance of the Laplacian, measured on the crop resized to
    a FIXED 128x128 so the number is comparable across crop sizes (a raw VoL
    grows with resolution and would just rank big crops first).

    Same metric the Spotlight pipeline uses for its `blurriness` field."""
    import cv2
    import numpy as np
    a = np.asarray(crop_rgb.convert("RGB"))[:, :, ::-1]
    g = cv2.cvtColor(cv2.resize(a, (128, 128), interpolation=cv2.INTER_AREA),
                     cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def avatar_crop(person, out_px, shape, rng, min_src_px=96,
                min_scale=1.5, min_sharpness=300.0):
    """A head-and-shoulders portrait of this person, `out_px` square.

    The crop is a square anchored on the top of the person's box and centred on
    it horizontally — for a standing person that is head plus shoulders. It is
    taken from the ORIGINAL photo, not from a cut-out silhouette, because a
    profile picture is a photographic crop.

    Returns (RGBA, None) or (None, reason). Never upscales: if the source region
    is smaller than the requested avatar the person is skipped, so every avatar
    emitted is genuinely sharp rather than an interpolated blur.
    """
    from PIL import Image, ImageDraw

    x1, y1, x2, y2 = person["bbox"]
    bw, bh = x2 - x1, y2 - y1
    if bw < 2 or bh < 2:
        return None, "degenerate"
    side = min(bw, bh * HEAD_FRAC)
    if side < min_src_px or side < out_px:
        return None, "source_too_small"          # would have to upscale
    # A 1:1 crop is only as crisp as the original pixels happened to be. Real
    # profile pictures are downscaled from something larger, which is what makes
    # them look sharp, so require genuine headroom rather than mere no-upscaling.
    if side < out_px * min_scale:
        return None, "insufficient_downscale"
    cx = (x1 + x2) / 2.0
    top = y1 - bh * HEAD_LIFT
    left = cx - side / 2.0
    with Image.open(person["image"]) as im:
        src = im.convert("RGB")
    L, T = int(round(left)), int(round(top))
    R, B = L + int(round(side)), T + int(round(side))
    if L < 0 or T < 0 or R > src.width or B > src.height:
        return None, "crop_out_of_frame"         # never pad — that is not a portrait
    raw = src.crop((L, T, R, B))
    # Reject sources that were already soft — motion blur, missed focus, heavy
    # JPEG. Downscaling cannot rescue detail that was never recorded.
    if min_sharpness and crop_sharpness(raw) < min_sharpness:
        return None, "soft_source"
    crop = raw.resize((out_px, out_px), Image.LANCZOS)

    rgba = Image.new("RGBA", (out_px, out_px), (0, 0, 0, 0))
    mask = Image.new("L", (out_px, out_px), 0)
    d = ImageDraw.Draw(mask)
    if shape == "circle":
        d.ellipse([0, 0, out_px - 1, out_px - 1], fill=255)
    else:                                        # rounded square, the other common style
        d.rounded_rectangle([0, 0, out_px - 1, out_px - 1],
                            radius=max(2, out_px // 8), fill=255)
    rgba.paste(crop, (0, 0), mask)
    return rgba, None


def emit_avatars(people, out_dir: Path, targets, imgs_per_target, per_image,
                 canvas, bg, seed, class_names, meta, shapes=("circle", "rounded"),
                 min_scale=1.5, min_sharpness=300.0, log=print):
    """Grid-ish pages of profile pictures on a light canvas, one page per size.

    Same PAIRED design as `paste`: the cast and their anchors are chosen once at
    the largest avatar size and reused at every smaller one, so a score change
    across sizes cannot be blamed on different people or a different layout.
    """
    from PIL import Image

    rng = random.Random(seed)
    out_dir = Path(out_dir)
    for sub in ("images", "labels", "ignore"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    name_to_id = {n: i for i, n in enumerate(class_names)}

    by_class = {n: [p for p in people if p["cls_name"] == n] for n in class_names}
    for v in by_class.values():
        rng.shuffle(v)
    cursor = {n: 0 for n in class_names}

    def next_person():
        order = sorted(class_names, key=lambda n: cursor[n] / max(1, len(by_class[n])))
        for n in order:
            if cursor[n] < len(by_class[n]):
                cursor[n] += 1
                return by_class[n][cursor[n] - 1]
        for n in class_names:
            if by_class[n]:
                cursor[n] += 1
                return by_class[n][(cursor[n] - 1) % len(by_class[n])]
        return None

    skipped, rows, gt_lines = Counter(), [], []
    biggest = int(max(targets))
    casts = []
    for k in range(imgs_per_target):
        cast, boxes, guard = [], [], 0
        while len(cast) < per_image and guard < per_image * 12:
            guard += 1
            person = next_person()
            if person is None:
                break
            shape = shapes[len(cast) % len(shapes)]
            av, why = avatar_crop(person, biggest, shape, rng,
                                  min_scale=min_scale, min_sharpness=min_sharpness)
            if av is None:
                skipped[why] += 1
                continue
            spot = place(boxes, biggest, biggest, canvas, rng, margin=10)
            if spot is None:
                skipped["no_room"] += 1
                continue
            boxes.append(spot)
            cast.append((person, spot[0], spot[1], shape))
        casts.append(cast)

    for target in targets:
        target = int(target)
        for k, cast in enumerate(casts):
            stem = f"a{target}_{k:04d}"
            img = Image.new("RGB", (canvas, canvas), (bg, bg, bg))
            placed = []
            for person, ax, ay, shape in cast:
                # the cast already passed the gates at the LARGEST size, so a
                # smaller render is strictly easier — only the scale rule is
                # re-checked, never the sharpness one
                av, why = avatar_crop(person, target, shape, rng,
                                      min_scale=min_scale, min_sharpness=0.0)
                if av is None:
                    skipped[why] += 1
                    continue
                # keep each avatar centred on the anchor it had at the biggest
                # size, so layout is stable across the whole size sweep
                cx = ax + biggest / 2.0
                cy = ay + biggest / 2.0
                x = int(round(min(max(0, cx - target / 2.0), canvas - target)))
                y = int(round(min(max(0, cy - target / 2.0), canvas - target)))
                img.paste(av, (x, y), av)
                placed.append((name_to_id[person["cls_name"]],
                               (x, y, x + target, y + target), person))
            img.save(out_dir / "images" / f"{stem}.jpg", quality=95)
            (out_dir / "labels" / f"{stem}.txt").write_text(
                "\n".join(to_yolo_line(cid, b, canvas, canvas)
                           for cid, b, _p in placed) + ("\n" if placed else ""))
            (out_dir / "ignore" / f"{stem}.txt").write_text("")
            for idx, (_cid, _b, person) in enumerate(placed):
                gt_lines.append(json.dumps({
                    "id": f"{stem}_{idx}", "image": f"{stem}.jpg",
                    "gt_age": person.get("gt_age"),
                    "gt_gender": person.get("gt_gender")}))
            rows.append({"target": target, "stem": stem, "n": len(placed)})
        if log:
            log(f"  a{target}: {sum(1 for r in rows if r['target'] == target)} images")

    (out_dir / "gt.jsonl").write_text("\n".join(gt_lines) + "\n")
    meta = {**meta, "arm": "avatars", "targets": [int(t) for t in targets],
            # all sizes share one directory, so the band check means "no avatar
            # is bigger than the largest target it was asked for"
            "max_height_px": float(max(targets)),
            "n_persons_scored": sum(r["n"] for r in rows),
            "canvas": canvas, "bg": bg, "per_image": per_image,
            "shapes": list(shapes), "head_frac": HEAD_FRAC,
            "min_scale": min_scale, "min_sharpness_vol128": min_sharpness,
            "n_images": len(rows), "n_people": sum(r["n"] for r in rows),
            "skipped": dict(skipped),
            "caveat": ("Head-and-shoulders crops taken from the original photo and "
                       "never upscaled, so every avatar is sharp. The crop is a "
                       "geometric approximation from the person's box (top "
                       f"{HEAD_FRAC:.0%} of its height), not a face detection. "
                       f"Every avatar is at least a {min_scale:g}x downscale of a "
                       f"source scoring >= {min_sharpness:g} variance-of-Laplacian "
                       "at 128px, so soft and out-of-focus people are excluded.")}
    (out_dir / "manifest.json").write_text(json.dumps({**meta, "images": rows}, indent=2))
    if log:
        log(f"emitted {len(rows)} images, {sum(r['n'] for r in rows)} avatars "
            f"-> {out_dir}   skipped: {dict(skipped)}")
    return meta


def place(boxes, pw, ph, canvas, rng, margin=4, tries=60):
    """A free spot with no overlap. Returns None if the canvas is too full."""
    for _ in range(tries):
        x = rng.randint(margin, max(margin, canvas - pw - margin))
        y = rng.randint(margin, max(margin, canvas - ph - margin))
        box = (x, y, x + pw, y + ph)
        if all(box[0] >= b[2] + margin or box[2] + margin <= b[0] or
               box[1] >= b[3] + margin or box[3] + margin <= b[1]
               for b in boxes):
            return box
    return None


def emit_paste(people, out_dir: Path, targets, imgs_per_target, per_image,
               canvas, grey, seed, class_names, meta, log=print):
    from PIL import Image

    rng = random.Random(seed)
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    (out_dir / "ignore").mkdir(parents=True, exist_ok=True)
    name_to_id = {n: i for i, n in enumerate(class_names)}

    # round-robin the classes so a size band cannot end up all-Man by luck
    by_class = {n: [p for p in people if p["cls_name"] == n] for n in class_names}
    for v in by_class.values():
        rng.shuffle(v)
    cursor = {n: 0 for n in class_names}

    def next_person():
        order = sorted(class_names, key=lambda n: cursor[n] / max(1, len(by_class[n])))
        for n in order:
            if cursor[n] < len(by_class[n]):
                cursor[n] += 1
                return by_class[n][cursor[n] - 1]
        for n in class_names:                     # wrap around; reuse is fine
            if by_class[n]:
                cursor[n] += 1
                return by_class[n][(cursor[n] - 1) % len(by_class[n])]
        return None

    rows, skipped, gt_lines = [], Counter(), []
    # PAIRED design: each image's cast and their positions are chosen ONCE, at
    # the largest target, then re-rendered at every smaller one. So h96_0007 and
    # h24_0007 hold the same six people at the same anchors, and any difference
    # in the score across sizes cannot be blamed on different people or a
    # different layout — apparent size is the only variable that moves.
    biggest = max(targets)
    casts = []
    for k in range(imgs_per_target):
        cast, boxes, guard = [], [], 0
        while len(cast) < per_image and guard < per_image * 8:
            guard += 1
            person = next_person()
            if person is None:
                break
            cut, why = cut_out(person, biggest)
            if cut is None:
                skipped[why] += 1
                continue
            spot = place(boxes, cut.width, cut.height, canvas, rng)
            if spot is None:
                skipped["no_room"] += 1
                continue
            boxes.append(spot)
            cast.append((person, spot[0], spot[1]))
        casts.append(cast)

    for target in targets:
        for k, cast in enumerate(casts):
            stem = f"h{int(target)}_{k:04d}"
            img = Image.new("RGB", (canvas, canvas), (grey, grey, grey))
            placed = []
            for person, ax, ay in cast:
                cut, why = cut_out(person, target)
                if cut is None:
                    skipped[why] += 1
                    continue
                x = min(ax, canvas - cut.width)
                y = min(ay, canvas - cut.height)
                img.paste(cut, (x, y), cut)
                placed.append((name_to_id[person["cls_name"]],
                               (x, y, x + cut.width, y + cut.height), person))
            img.save(out_dir / "images" / f"{stem}.jpg", quality=95)
            lines = [to_yolo_line(cid, b, canvas, canvas) for cid, b, _p in placed]
            (out_dir / "labels" / f"{stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""))
            (out_dir / "ignore" / f"{stem}.txt").write_text("")
            for line_idx, (cid, b, person) in enumerate(placed):
                rows.append({"image": stem, "cls": cid,
                             "cls_name": person["cls_name"],
                             "target_h": target, "box": list(b),
                             "src_image": Path(person["image"]).name,
                             "src_h": round(person["src_h"], 1)})
                # id = "<stem>_<label line index>" — the join key the project's
                # scorer expects, so run_autolabel_on_manifest.py reads this set
                # exactly as it reads LAGENDA.
                gt_lines.append(json.dumps({
                    "id": f"{stem}_{line_idx}", "image": f"{stem}.jpg",
                    "gt_age": person.get("gt_age"),
                    "gt_gender": person.get("gt_gender"),
                    "target_h": target}))
        if log:
            n = sum(1 for r in rows if r["target_h"] == target)
            log(f"  h{int(target)}: {imgs_per_target} images, {n} people")

    mix = Counter(r["cls_name"] for r in rows)
    meta = {**meta, "arm": "paste_grey", "targets": list(targets),
            "canvas": canvas, "grey": grey, "per_image": per_image,
            "n_images": len(targets) * imgs_per_target, "n_persons": len(rows),
            "class_names": list(class_names), "class_mix": dict(mix),
            "paired": ("same cast and anchors at every target height — size is "
                       "the only variable across the size ladder"),
            "skipped": dict(skipped), "n_source_people": len(people),
            "caveat": ("People are cut out and pasted on flat grey: boxes are "
                       "exact and never contain a neighbour, and apparent size "
                       "is the only variable. But there is no scene context and "
                       "no motion blur/haze/focus loss, and cut edges are sharp "
                       "— treat as a controlled probe, not a realism estimate.")}
    (out_dir / "gt.jsonl").write_text("\n".join(gt_lines) + "\n")
    (out_dir / "manifest.json").write_text(json.dumps({**meta, "persons": rows},
                                                      indent=2))
    if log:
        log(f"emitted {meta['n_images']} images, {len(rows)} people "
            f"{dict(mix)} -> {out_dir}   skipped {dict(skipped)}")
    return meta

# ---------------------------------------------------------------------------
def _selftest():
    have_pil = True
    try:
        import PIL  # noqa: F401
    except ImportError:
        have_pil = False

    # letterbox height: a 96px person in a 4000px-wide photo is 15px to a 640 net
    assert abs(input_height(96, 4000, 3000, 640) - 15.36) < 1e-6
    assert abs(input_height(96, 640, 480, 640) - 96.0) < 1e-6, "no rescale when long side == imgsz"
    assert input_height(96, 0, 0, 640) == 0.0

    assert band_of(8) == "tiny" and band_of(16) == "verysmall"
    assert band_of(31.9) == "verysmall" and band_of(32) == "small"
    assert band_of(95.9) == "medium" and band_of(96) == "large" and band_of(1e9) == "large"

    # seg_boxes is the real parser: polygons must survive, 5-field boxes too
    tmp = Path("/tmp/_smallperson_selftest")
    (tmp / "labels").mkdir(parents=True, exist_ok=True)
    (tmp / "labels" / "poly.txt").write_text(
        "0 0.10 0.10 0.30 0.10 0.30 0.50 0.10 0.50\n1 0.5 0.5 0.2 0.4\n")
    got = seg_boxes(tmp / "labels" / "poly.txt", 100, 100)
    assert got[0] == (0, 10.0, 10.0, 30.0, 50.0), got
    assert got[1] == (1, 40.0, 30.0, 60.0, 70.0), got
    assert seg_boxes(tmp / "labels" / "nope.txt", 10, 10) is None, "absent != empty"

    # selection + ignore split
    recs = [
        _record("a", "/x/a.jpg", 640, 640, [(0, 0, 0, 10, 40), (1, 0, 0, 10, 300)], [], 640),
        _record("b", "/x/b.jpg", 640, 640, [(2, 0, 0, 10, 50)], [[0, 0, 5, 5]], 640),
        _record("c", "/x/c.jpg", 640, 640, [(1, 0, 0, 10, 400)], [], 640),
    ]
    picked = select(recs, "native", 96, 1, False, 0, 0)
    assert [r["stem"] for r in picked] == ["a", "b"], picked
    assert len(picked[0]["small"]) == 1 and len(picked[0]["big"]) == 1
    only = select(recs, "native", 96, 1, True, 0, 0)
    assert [r["stem"] for r in only] == ["b"], "require_all must drop image a"
    assert select(recs, "native", 96, 2, False, 0, 0) == [], "min_small=2 unmet"

    assert [r["stem"] for r in select(recs, "native", 96, 1, False, 0, 0, {"a"})] == ["b"]
    ex = tmp / "ex.txt"
    ex.write_text("/some/dir/a.jpg\nb\n\n")
    assert load_exclude(ex) == {"a", "b"}, load_exclude(ex)
    assert load_exclude(None) == set()

    # a native-px selection is NOT an input-px selection
    big_img = [_record("d", "/x/d.jpg", 4000, 3000, [(0, 0, 0, 50, 300)], [], 640)]
    assert select(big_img, "native", 96, 1, False, 0, 0) == [], "300px native is not small"
    assert len(select(big_img, "input", 96, 1, False, 0, 0)) == 1, "48px at input IS small"

    # YOLO line round-trip through the real parser
    line = to_yolo_line(2, (10, 20, 30, 60), 100, 200)
    (tmp / "labels" / "rt.txt").write_text(line + "\n")
    rt = seg_boxes(tmp / "labels" / "rt.txt", 100, 200)[0]
    assert rt[0] == 2 and max(abs(a - b) for a, b in zip(rt[1:], (10, 20, 30, 60))) < 1e-3, rt

    # emit: labels hold in-band people, ignore holds the rest, odgt reloads
    out = tmp / "arm"
    emit_mine(picked, out, "selftest", copy=False, meta={}, log=None)
    assert (out / "labels" / "a.txt").read_text().startswith("0 ")
    assert len(seg_boxes(out / "labels" / "a.txt", 640, 640)) == 1
    assert len(seg_boxes(out / "ignore" / "a.txt", 640, 640)) == 1, "the 300px person"
    assert len(seg_boxes(out / "ignore" / "b.txt", 640, 640)) == 1, "inherited ignore region"
    # CrowdHuman is not self-consistent — some vboxes are taller than their
    # fbox. Sizing must take the larger, or a 200px-visible person is admitted
    # to a "<=96px" set on the strength of a 50px full-body box.
    assert person_height({"fbox": [0.0, 0.0, 10.0, 50.0],
                          "vbox": [0.0, 0.0, 10.0, 200.0]}, "fbox") == 200.0
    assert person_height({"fbox": [0.0, 0.0, 10.0, 300.0],
                          "vbox": [0.0, 0.0, 10.0, 40.0]}, "fbox") == 300.0
    assert person_height({"fbox": [0.0, 0.0, 10.0, 300.0],
                          "vbox": [0.0, 0.0, 10.0, 40.0]}, "vbox") == 40.0

    # a large person seen through a keyhole must NOT count as small, and the
    # occlusion ratio must survive the emit round-trip
    occl = _record("o", "/x/o.jpg", 640, 640, [(None, 0, 0, 10, 40)], [], 640)
    occl["persons"][0].update(occ_ratio=0.1, fbox=[0.0, 0.0, 100.0, 400.0],
                              h_px=400.0, h_in=400.0)
    only_small = select([occl], "native", 96, 1, False, 0, 0)
    assert only_small == [], "fbox-sized 400px person must not be 'small'"
    occl["persons"][0].update(h_px=40.0, h_in=40.0)      # as if size_box=vbox
    assert len(select([occl], "native", 96, 1, False, 0, 0)) == 1
    emit_mine(select([occl], "native", 96, 1, False, 0, 0), tmp / "occ",
              "selftest", copy=False, meta={}, log=None)
    ro = load_odgt(tmp / "occ" / "annotations" / "persons.odgt")["o"]["persons"][0]
    assert abs(ro["occ_ratio"] - 0.01) < 1e-6, (
        f"occ_ratio must come from the REAL fbox, got {ro['occ_ratio']}")

    back = load_odgt(out / "annotations" / "persons.odgt")
    assert len(back["a"]["persons"]) == 1 and len(back["a"]["ignores"]) == 1, back["a"]
    assert abs(_area(back["a"]["persons"][0]["vbox"]) - 400.0) < 1e-6, "vbox xywh round-trip"

    # synth geometry: tallest person lands ON the target, none exceed it
    r = _record("s", "/x/s.jpg", 640, 640, [(0, 0, 0, 100, 400), (1, 0, 0, 50, 200)], [], 640)
    s = synth_scale(r, 64, "native", 640)
    assert abs(s - 0.16) < 1e-9, s
    canvas_h = [(p["box"][3] - p["box"][1]) * s for p in r["persons"]]
    assert abs(max(canvas_h) - 64) < 1e-6 and max(canvas_h) <= 64 + 1e-6, canvas_h
    assert synth_scale(r, 500, "native", 640) is None, "never upscale"

    # avatars: the crop must sit at the TOP of the person box (head+shoulders),
    # be square, and never upscale
    from PIL import Image as _I
    av_img = tmp / "av.jpg"
    _I.new("RGB", (600, 900), (10, 120, 200)).save(av_img)
    tall = {"image": str(av_img), "bbox": (100.0, 100.0, 340.0, 800.0),
            "cls_name": "Woman", "gt_age": 30, "gt_gender": "F"}
    # flat colour has no detail at all, so the sharpness gate is switched off
    # for the pure-geometry assertions below
    out, why = avatar_crop(tall, 128, "circle", random.Random(0),
                           min_scale=1.0, min_sharpness=0.0)
    assert out is not None and out.size == (128, 128), (out, why)
    side = min(340.0 - 100.0, 700.0 * HEAD_FRAC)
    assert abs(side - 238.0) < 1.0, side
    small_req = avatar_crop(tall, int(side) + 40, "circle", random.Random(0),
                            min_scale=1.0, min_sharpness=0.0)
    assert small_req[0] is None and small_req[1] == "source_too_small", small_req
    edge = {**tall, "bbox": (0.0, 0.0, 40.0, 120.0)}
    assert avatar_crop(edge, 128, "circle", random.Random(0),
                       min_scale=1.0, min_sharpness=0.0)[1] in (
        "source_too_small", "crop_out_of_frame")
    # a blurred source must be REJECTED where the same image sharp is accepted
    from PIL import ImageFilter as _IF
    chk = _I.new("RGB", (600, 900))
    for yy in range(0, 900, 6):
        for xx in range(0, 600, 6):
            if (xx // 6 + yy // 6) % 2 == 0:
                chk.paste((255, 255, 255), (xx, yy, xx + 6, yy + 6))
    chk.save(av_img)
    sharp = avatar_crop(tall, 96, "circle", random.Random(0), min_sharpness=50.0)
    assert sharp[0] is not None, sharp
    chk.filter(_IF.GaussianBlur(9)).save(av_img)
    soft = avatar_crop(tall, 96, "circle", random.Random(0), min_sharpness=50.0)
    assert soft[0] is None and soft[1] == "soft_source", soft
    # and the downscale-headroom rule must bite independently of sharpness
    tight = avatar_crop(tall, int(side) - 2, "circle", random.Random(0),
                        min_scale=1.5, min_sharpness=0.0)
    assert tight[0] is None and tight[1] == "insufficient_downscale", tight


    # circular mask must leave the corners transparent
    px = out.getpixel((1, 1))
    assert px[3] == 0, f"circle corner should be transparent, got {px}"
    assert out.getpixel((64, 64))[3] == 255, "centre must be opaque"

    # census arithmetic
    c = census(recs, "native", [96], 640, log=None)
    assert c["n_images"] == 3 and c["n_persons"] == 4
    assert c["cutoffs"][0] == {"cutoff_px": 96, "images_with_any": 2,
                               "images_all_small": 1, "persons_at_or_below": 2}, c["cutoffs"]

    # polygon reader: outlines survive, plain boxes are (correctly) skipped
    (tmp / "labels" / "poly2.txt").write_text(
        "0 0.1 0.1 0.3 0.1 0.3 0.5 0.1 0.5\n1 0.5 0.5 0.2 0.4\n")
    pg = seg_polygons(tmp / "labels" / "poly2.txt", 100, 100)
    assert len(pg) == 1 and len(pg[0][1]) == 4, pg
    assert poly_bbox(pg[0][1]) == (10.0, 10.0, 30.0, 50.0), poly_bbox(pg[0][1])
    assert seg_polygons(tmp / "labels" / "absent.txt", 10, 10) is None

    # placement rejects overlap and reports failure rather than stacking
    rng = random.Random(0)
    assert place([], 10, 10, 640, rng) is not None
    assert place([(0, 0, 640, 640)], 100, 100, 640, rng) is None

    if have_pil:
        from PIL import Image
        img = Image.new("RGB", (200, 100))
        canvas, ox, oy = shrink_and_pad(img, 0.5)
        assert canvas.size == (200, 100) and (ox, oy) == (50, 25)

        # cut_out: a full-bbox mask is implausible for a person and is rejected
        src = tmp / "src.jpg"
        Image.new("RGB", (100, 400), (200, 30, 30)).save(src)
        full = {"image": str(src), "pts": [(0, 0), (100, 0), (100, 400), (0, 400)],
                "bbox": (0.0, 0.0, 100.0, 400.0)}
        cut, why = cut_out(full, 64)
        assert cut is None and why == "implausible_mask", (cut, why)
        # a plausible silhouette scales to exactly the target height
        slim = {"image": str(src), "pts": [(40, 0), (60, 0), (60, 400), (40, 400)],
                "bbox": (0.0, 0.0, 100.0, 400.0)}
        cut, why = cut_out(slim, 64)
        assert why is None and cut.height == 64, (why, cut and cut.size)
        assert cut.mode == "RGBA" and cut.getpixel((0, 0))[3] == 0, "outside mask must be transparent"

        # end-to-end emit: labels are exact, classes balanced, boxes disjoint
        people = [dict(slim, cls_name=n, src_h=400.0)
                  for n in ("Woman", "Man", "Child")] * 2
        pm = emit_paste(people, tmp / "paste", [64.0, 32.0], 2, 3, 640, 114, 0,
                        DEFAULT_CLASS_NAMES_ORDER, {}, log=None)
        assert pm["n_images"] == 4 and pm["n_persons"] == 12, pm
        gtl = [json.loads(x) for x in
               (tmp / "paste" / "gt.jsonl").read_text().splitlines() if x.strip()]
        assert len(gtl) == 12, len(gtl)
        n0 = len(seg_boxes(tmp / "paste" / "labels" / "h64_0000.txt", 640, 640))
        ids = sorted(int(g["id"].rsplit("_", 1)[1]) for g in gtl
                     if g["image"] == "h64_0000.jpg")
        assert ids == list(range(n0)), (ids, n0)

        by = {}
        for r in json.loads((tmp / "paste" / "manifest.json").read_text())["persons"]:
            by.setdefault((r["image"][-4:], r["target_h"]), []).append(r)
        assert [r["src_image"] for r in by[("0000", 64.0)]] == \
               [r["src_image"] for r in by[("0000", 32.0)]], "cast must be paired"
        for a, b in zip(by[("0000", 64.0)], by[("0000", 32.0)]):
            assert a["box"][0] == b["box"][0] and a["box"][1] == b["box"][1], \
                "anchors must be identical across heights"
            assert abs((a["box"][3] - a["box"][1]) - 64) < 1.5
            assert abs((b["box"][3] - b["box"][1]) - 32) < 1.5
        assert set(pm["class_mix"]) == {"Woman", "Man", "Child"}, pm["class_mix"]
        got = seg_boxes(tmp / "paste" / "labels" / "h64_0000.txt", 640, 640)
        assert len(got) == 3, got
        for _c, x1, y1, x2, y2 in got:
            assert abs((y2 - y1) - 64) < 1.5, (y2 - y1)
        for a in range(len(got)):
            for b in range(a + 1, len(got)):
                ax, ay, ax2, ay2 = got[a][1:]
                bx, by, bx2, by2 = got[b][1:]
                assert (ax >= bx2 or ax2 <= bx or ay >= by2 or ay2 <= by), \
                    "pasted people must never overlap"
        print("build_small_person_set selftest OK (incl. PIL paths)")
    else:
        print("build_small_person_set selftest OK "
              "(SKIPPED: PIL not installed — image reads, shrink_and_pad and "
              "synth emission are unexercised here; run on the pod to cover them)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?",
                    choices=["census", "mine", "synth", "paste", "avatars"])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--source", choices=["yolo", "odgt"], default="yolo")
    ap.add_argument("--images")
    ap.add_argument("--labels", help="--source yolo: YOLO label dir (boxes or polygons)")
    ap.add_argument("--ignore-labels", help="--source yolo: existing ignore-region dir")
    ap.add_argument("--odgt", help="--source odgt: CrowdHuman .odgt")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="letterbox long side used for input-space heights (default 640)")
    ap.add_argument("--size-box", choices=["fbox", "vbox"], default="fbox",
                    help="--source odgt: measure a person's size by full-body "
                         "box (default — genuinely small people) or visible "
                         "extent (mixes in large but heavily occluded people)")
    ap.add_argument("--space", choices=["native", "input"], default="native",
                    help="measure person height in native pixels (what QA sees) "
                         "or model-input pixels (what decides detectability)")
    ap.add_argument("--max-height", type=float, default=96.0,
                    help="a person at or below this height is in-band (default 96)")
    ap.add_argument("--min-small", type=int, default=1,
                    help="minimum in-band people for an image to qualify")
    ap.add_argument("--all-small", action="store_true",
                    help="keep only images where EVERY person is in-band")
    ap.add_argument("--cutoffs", default="16,24,32,48,64,96,128",
                    help="census only: comma-separated heights to tabulate")
    ap.add_argument("--targets", default="96,64,48,32,24,16",
                    help="synth only: target heights to emit, one dir each")
    ap.add_argument("--min-source-height", type=float, default=200.0,
                    help="synth only: only shrink people at least this tall")
    ap.add_argument("-n", type=int, default=0, help="cap images (0 = all)")
    ap.add_argument("--limit", type=int, default=0, help="cap images READ from source")
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument("--copy", action="store_true",
                    help="copy images instead of symlinking (needed off-volume)")
    ap.add_argument("--gt-jsonl", help="paste: LAGENDA gt.jsonl (human age+gender)")
    ap.add_argument("--polygons", help="paste: dir of YOLO segment-polygon labels")
    ap.add_argument("--match-iou", type=float, default=0.5,
                    help="paste: IoU needed to trust a polygon as that person")
    ap.add_argument("--images-per-target", type=int, default=50)
    ap.add_argument("--per-image", type=int, default=6)
    ap.add_argument("--canvas", type=int, default=640,
                    help="paste: canvas side; at 640 native px == model-input px")
    ap.add_argument("--bg", type=int, default=240,
                    help="avatars: canvas grey level (240 = light, like a feed)")
    ap.add_argument("--min-scale", type=float, default=1.5,
                    help="avatars: source crop must be at least this multiple of "
                         "the avatar size, so every avatar is a real downscale")
    ap.add_argument("--min-sharpness", type=float, default=300.0,
                    help="avatars: reject source crops below this "
                         "variance-of-Laplacian at 128px (calibrated on LAGENDA: "
                         "p25=190, p50=392, so 300 keeps the sharper ~60%%)")
    ap.add_argument("--shapes", default="circle,rounded",
                    help="avatars: mask styles to alternate between")
    ap.add_argument("--grey", type=int, default=114,
                    help="paste: background grey (114 = the letterbox pad value)")
    ap.add_argument("--exclude-stems",
                    help="file of stems/paths, or an image dir, to keep OUT "
                         "(e.g. the seed-51 crowd benchmark sample)")
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return
    if not args.cmd or not args.images:
        ap.error("need a subcommand and --images (or --selftest)")

    recs = load_source(args)
    exclude = load_exclude(args.exclude_stems)
    if exclude:
        print(f"[exclude] {len(exclude)} stems held out ({args.exclude_stems})")
    meta = {"exclude_stems": args.exclude_stems, "n_excluded": len(exclude),
            "size_box": args.size_box if args.source == "odgt" else None,
            "source": {"images": args.images, "labels": args.labels,
                       "odgt": args.odgt, "kind": args.source},
            "space": args.space, "imgsz": args.imgsz,
            "max_height_px": args.max_height, "min_small": args.min_small,
            "all_small": args.all_small, "seed": args.seed, "n_cap": args.n}

    if args.cmd == "census":
        cuts = [float(x) for x in args.cutoffs.split(",")]
        s = census(recs, args.space, cuts, args.imgsz)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps({**meta, **s}, indent=2))
        return

    if not args.out:
        ap.error("--out is required for mine/synth")

    if args.cmd == "avatars":
        for req in ("gt_jsonl", "labels", "polygons", "out"):
            if not getattr(args, req):
                ap.error(f"avatars needs --{req.replace('_', '-')}")
        people = load_paste_people(
            Path(args.images), Path(args.gt_jsonl), Path(args.labels),
            Path(args.polygons), args.min_source_height, args.match_iou,
            args.limit)
        if not people:
            raise SystemExit("no usable people — check --gt-jsonl/--labels/--polygons")
        emit_avatars(people, Path(args.out),
                     [float(x) for x in args.targets.split(",")],
                     args.images_per_target, args.per_image, args.canvas,
                     args.bg, args.seed, DEFAULT_CLASS_NAMES_ORDER, meta,
                     shapes=tuple(args.shapes.split(",")),
                     min_scale=args.min_scale, min_sharpness=args.min_sharpness)
        return

    if args.cmd == "paste":
        for req in ("gt_jsonl", "polygons", "labels"):
            if not getattr(args, req):
                ap.error(f"paste needs --{req.replace('_', '-')}")
        people = load_paste_people(Path(args.images), Path(args.gt_jsonl),
                                   Path(args.labels), Path(args.polygons),
                                   args.min_source_height, args.match_iou,
                                   args.limit)
        if not people:
            raise SystemExit("no usable people — check --polygons / --gt-jsonl")
        emit_paste(people, Path(args.out),
                   [float(x) for x in args.targets.split(",")],
                   args.images_per_target, args.per_image, args.canvas,
                   args.grey, args.seed, DEFAULT_CLASS_NAMES_ORDER, meta)
        return

    if args.cmd == "mine":
        picked = select(recs, args.space, args.max_height, args.min_small,
                        args.all_small, args.seed, args.n, exclude)
        if not picked:
            raise SystemExit("no images qualified — run `census` and loosen the band")
        emit_mine(picked, Path(args.out), args.source, args.copy, meta)
    else:
        targets = [float(x) for x in args.targets.split(",")]
        emit_synth(recs, Path(args.out), targets, args.space, args.imgsz,
                   args.min_source_height, args.n, args.seed, meta, exclude)


if __name__ == "__main__":
    main()
