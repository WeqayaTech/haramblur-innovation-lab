#!/usr/bin/env python3
"""
Rebuild the LAGENDA evaluation set from the FULL official annotation CSV.

Why (found 2026-08-10): the eval set we had been using sampled ~ONE person per
image (5,000 persons / 4,899 images = 1.02/img). The official
`lagenda_annotation.csv` carries 347,509 rows for 67,159 images:

  * 333,321 rows have a person box   (~4.96/img)  -> DETECTION ground truth
  *  84,192 rows also have age+gender (~1.25/img) -> CLASSIFICATION ground truth

So our detection-recall numbers were "recall on the single most prominent
subject", not recall on people. This rebuild keeps every person box.

Emits three artifacts so BOTH existing scorers run unchanged:

  <out>/labels/<stem>.txt   YOLO boxes, one line per person box, class 0.
                            Line index == the id suffix used by the manifest.
  <out>/gt.jsonl            {id, image, gt_age, gt_gender} for rows that have
                            age+gender -> `run_autolabel_on_manifest.py`
                            (classification: gender / age / leak, conditioned
                            on detection, exactly as before).
  <out>/persons.odgt        CrowdHuman-style records over ALL person boxes ->
                            `eval_negatives_crowd.py --mode crowd`
                            (detection recall / precision / duplicates over the
                            full ~5.8 boxes per image).

Contamination: LAGENDA is sampled from Open Images and our models trained on
the OIV7 train split — 7.4% of LAGENDA images (4,983 of 67,159) are in it.
`--exclude-list` drops those images entirely; the count is reported and written
to <out>/BUILD_README.md.

HONEST CAVEAT, carried into the README: the person BOXES are the dataset
authors' detector output (they trained a YOLOv8 for the public release), while
age/gender got 10 crowdsourced votes per sample. So detection metrics scored
against these boxes measure agreement with THEIR detector, not human truth —
a tier below CrowdHuman's human-drawn exhaustive boxes. Classification metrics
are human-labeled and unaffected.

Usage:
    python3 build_lagenda_full.py \
        --csv /workspace/datasets/lagenda_full/annotations/lagenda_annotation.csv \
        --images /workspace/lagenda_eval/lagenda_yolo/images/val \
        --exclude-list /workspace/exp12/train_full.txt \
        --out /workspace/datasets/lagenda_full/eval_v2

    python3 build_lagenda_full.py --selftest    # synthetic, no data needed
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# production taxonomy, same translation both experiments use
try:
    import translation as T
    _PROD3 = T.get("production_3class").heads[0]
    _NAME2ID = {"Woman": 0, "Man": 1, "Child": 2}
except Exception:                                   # selftest without the module
    T = _PROD3 = None
    _NAME2ID = {"Woman": 0, "Man": 1, "Child": 2}


def prod_class(age, gender):
    """(age, gender) -> 0/1/2 via translation.py, or None if untranslatable."""
    if _PROD3 is None:
        return 2 if float(age) <= 12 else (0 if str(gender).upper().startswith("F") else 1)
    name = _PROD3.label(T.norm_gender(gender), T.age_bucket_from_years(float(age)))
    return _NAME2ID.get(name)


def stem_of(img_name: str) -> str:
    return img_name.split("/")[-1].rsplit(".", 1)[0]


def load_rows(csv_path: Path):
    """-> {stem: [row, ...]} preserving CSV order (order defines line index)."""
    by = defaultdict(list)
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            by[stem_of(r["img_name"])].append(r)
    return by


def load_exclude(path: Path | None):
    if not path:
        return set()
    return {stem_of(l.strip()) for l in open(path) if l.strip()}


def has_box(r) -> bool:
    return all(r[k] != "-1" for k in
               ("person_x0", "person_y0", "person_x1", "person_y1"))


def has_label(r) -> bool:
    return r["age"] not in ("-1", "") and r["gender"] not in ("-1", "")


def build(csv_path: Path, images_dir: Path, out_dir: Path,
          exclude: set, image_size, log=print):
    """image_size(stem) -> (w, h) | None. Injected so the selftest needs no files."""
    by = load_rows(csv_path)
    labels_dir = out_dir / "labels"
    cls_dir = out_dir / "labels_3class"
    ign_dir = out_dir / "ignore"
    for d in (labels_dir, cls_dir, ign_dir):
        d.mkdir(parents=True, exist_ok=True)

    manifest, odgt_lines = [], []
    n_img = n_box = n_lab = n_excluded = n_missing = n_degenerate = 0
    n_cls = n_ign = n_untranslatable = 0

    for stem in sorted(by):
        if stem in exclude:
            n_excluded += 1
            continue
        wh = image_size(stem)
        if wh is None:               # image not present locally -> skip entirely
            n_missing += 1
            continue
        w, h = wh
        lines, gtboxes, cls_lines, ign_lines = [], [], [], []
        for r in by[stem]:
            if not has_box(r):
                continue             # face-only row, no person box
            x0, y0, x1, y1 = (float(r["person_x0"]), float(r["person_y0"]),
                              float(r["person_x1"]), float(r["person_y1"]))
            x0, y0 = max(0.0, x0), max(0.0, y0)
            x1, y1 = min(float(w), x1), min(float(h), y1)
            if x1 <= x0 or y1 <= y0:
                n_degenerate += 1
                continue
            idx = len(lines)         # line index == id suffix, assigned AFTER filtering
            lines.append(f"0 {((x0+x1)/2)/w:.6f} {((y0+y1)/2)/h:.6f} "
                         f"{(x1-x0)/w:.6f} {(y1-y0)/h:.6f}")
            gtboxes.append({"tag": "person",
                            "vbox": [x0, y0, x1 - x0, y1 - y0],
                            "extra": {}})
            yolo = (f"{((x0+x1)/2)/w:.6f} {((y0+y1)/2)/h:.6f} "
                    f"{(x1-x0)/w:.6f} {(y1-y0)/h:.6f}")
            if has_label(r):
                manifest.append({"id": f"{stem}_{idx}", "image": f"{stem}.jpg",
                                 "gt_age": float(r["age"]),
                                 "gt_gender": r["gender"]})
                n_lab += 1
                cid = prod_class(r["age"], r["gender"])
                if cid is None:
                    n_untranslatable += 1
                    ign_lines.append("0 " + yolo)      # unusable label -> ignore
                    n_ign += 1
                else:
                    cls_lines.append(f"{cid} " + yolo)
                    n_cls += 1
            else:
                ign_lines.append("0 " + yolo)          # detected but unlabeled
                n_ign += 1
        if not lines:
            continue
        (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        (cls_dir / f"{stem}.txt").write_text("\n".join(cls_lines) + ("\n" if cls_lines else ""))
        (ign_dir / f"{stem}.txt").write_text("\n".join(ign_lines) + ("\n" if ign_lines else ""))
        odgt_lines.append(json.dumps({"ID": stem, "gtboxes": gtboxes}))
        n_img += 1
        n_box += len(lines)

    (out_dir / "gt.jsonl").write_text(
        "".join(json.dumps(m) + "\n" for m in manifest))
    (out_dir / "persons.odgt").write_text("\n".join(odgt_lines) + "\n")

    stats = {"images": n_img, "person_boxes": n_box, "labeled_persons": n_lab,
             "excluded_contaminated": n_excluded, "images_not_found": n_missing,
             "degenerate_boxes_dropped": n_degenerate,
             "boxes_per_image": round(n_box / n_img, 2) if n_img else 0,
             "labeled_per_image": round(n_lab / n_img, 2) if n_img else 0,
             "class3_boxes": n_cls, "ignore_boxes": n_ign,
             "untranslatable_labels": n_untranslatable}
    (out_dir / "BUILD_README.md").write_text(f"""# LAGENDA full-annotation eval set

Built by `vlm-cluster/build_lagenda_full.py` from `{csv_path.name}`.

{json.dumps(stats, indent=2)}

## What each file is for

- `labels/<stem>.txt` — every person box, class 0, YOLO normalized cxcywh.
  Line index is the `_N` suffix in `gt.jsonl` ids.
- `gt.jsonl` — `{{id, image, gt_age, gt_gender}}` for rows carrying human
  age+gender. Feed to `run_autolabel_on_manifest.py --gt-manifest`.
- `persons.odgt` — CrowdHuman-format records over ALL person boxes. Feed to
  `eval_negatives_crowd.py --mode crowd --gt-odgt`.
- `labels_3class/` + `ignore/` — the VALID human-labeled 3-class mAP set:
  `labels_3class` holds only persons with human age+gender, mapped to
  {{0: Woman, 1: Man, 2: Child}} via `translation.py`; `ignore` holds every
  person box WITHOUT usable labels. Feed both to
  `map_eval.py --gt-labels .../labels_3class --ignore-labels .../ignore`, so a
  detection landing on a real-but-unlabeled person is neither a true nor a
  false positive (COCO's iscrowd/ignore semantics). Without the ignore set the
  3-class mAP would be badly wrong — 4 of every 6 real people are unlabeled.

## Caveats (read before quoting numbers)

- **Person boxes are detector-derived**, not human-drawn (the authors trained a
  YOLOv8 for the public release); age/gender had 10 crowdsourced votes each.
  Detection metrics here measure agreement with THEIR detector — one tier below
  CrowdHuman's human-drawn exhaustive GT. Classification metrics are human.
- **Contaminated images excluded**: LAGENDA is sampled from Open Images and our
  models trained on the OIV7 train split; images present in the exclude list
  were dropped ({n_excluded} of them).
- `occ_ratio` is 1.0 for every person (LAGENDA has no full-vs-visible box
  distinction), so `recall_by_occlusion` will report everything as "light" —
  ignore that breakdown here; it is meaningful only on CrowdHuman.
""")
    log(json.dumps(stats, indent=2))
    return stats


# ---------------------------------------------------------------- selftest
def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        csv_path = td / "a.csv"
        csv_path.write_text(
            "img_name,age,gender,face_x0,face_y0,face_x1,face_y1,"
            "person_x0,person_y0,person_x1,person_y1\n"
            # img A: 3 rows -> 1 face-only (no box), 1 labeled, 1 box-only
            "lag/aaa.jpg,30,M,1,1,2,2,10,10,50,90\n"
            "lag/aaa.jpg,-1,-1,3,3,4,4,60,10,90,90\n"
            "lag/aaa.jpg,20,F,5,5,6,6,-1,-1,-1,-1\n"
            # img B: contaminated, must vanish entirely
            "lag/bbb.jpg,40,F,1,1,2,2,10,10,50,90\n"
            # img C: degenerate box (x1<=x0) must be dropped, one good row after
            "lag/ccc.jpg,50,M,1,1,2,2,80,10,20,90\n"
            "lag/ccc.jpg,12,F,1,1,2,2,10,10,40,80\n")
        out = td / "out"
        stats = build(csv_path, td, out, exclude={"bbb"},
                      image_size=lambda s: (100, 100), log=lambda *_: None)
        assert stats["images"] == 2 and stats["person_boxes"] == 3, stats
        assert stats["labeled_persons"] == 2 and stats["excluded_contaminated"] == 1, stats
        assert stats["degenerate_boxes_dropped"] == 1, stats
        assert stats["class3_boxes"] == 2 and stats["ignore_boxes"] == 1, stats
        c = (out / "labels_3class" / "aaa.txt").read_text().split()
        assert c[0] == "1", c            # age 30 male -> Man
        assert (out / "ignore" / "aaa.txt").read_text().startswith("0 "), "unlabeled -> ignore"
        assert (out / "labels_3class" / "ccc.txt").read_text().split()[0] == "2", "age 12 -> Child"
        assert not (out / "labels" / "bbb.txt").exists(), "contaminated leaked"

        # ids must index the FILTERED line order, not the raw CSV order
        man = [json.loads(l) for l in (out / "gt.jsonl").read_text().splitlines()]
        ids = {m["id"] for m in man}
        assert ids == {"aaa_0", "ccc_0"}, ids
        lines = (out / "labels" / "aaa.txt").read_text().splitlines()
        assert len(lines) == 2 and lines[0].startswith("0 0.3"), lines
        # the face-only row (no person box) must not appear anywhere
        assert all(m["id"] != "aaa_2" for m in man)

        # odgt round-trips through the real loader
        sys.path.insert(0, str(Path(__file__).parent))
        from eval_negatives_crowd import load_odgt
        recs = load_odgt(out / "persons.odgt")
        assert set(recs) == {"aaa", "ccc"}, recs.keys()
        assert len(recs["aaa"]["persons"]) == 2
        assert recs["ccc"]["persons"][0]["vbox"] == [10.0, 10.0, 40.0, 80.0]
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv")
    ap.add_argument("--images", help="dir of images to build for (others skipped)")
    ap.add_argument("--exclude-list", help="file of paths/stems to drop (train split)")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not (a.csv and a.images and a.out):
        ap.error("--csv, --images and --out required (or --selftest)")

    from PIL import Image
    images_dir = Path(a.images)
    sizes = {}
    for p in images_dir.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            sizes[p.stem] = p

    def image_size(stem):
        p = sizes.get(stem)
        if p is None:
            return None
        with Image.open(p) as im:
            return im.size

    build(Path(a.csv), images_dir, Path(a.out),
          load_exclude(Path(a.exclude_list) if a.exclude_list else None),
          image_size)


if __name__ == "__main__":
    sys.exit(main())
