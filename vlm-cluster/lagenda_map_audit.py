#!/usr/bin/env python3
"""
Audit + demonstrate the LAGENDA v2 mAP setup. Two jobs, both about TRUST:

  --verify  Independently re-derive the box counts straight from the official
            CSV (this file does NOT import build_lagenda_full.py — it is a
            second implementation, so agreement means something) and check them
            against what is actually on disk. Prints a reconciliation table and
            asserts the arithmetic:  scoreable + ignore == total person boxes.

  --trace   Run ONE model against the eval set on a handful of images and show,
            detection by detection, which of the three outcomes it got:
              MATCH   -> landed on a human-labeled person of the right class
              WRONG   -> landed on a human-labeled person, wrong class
              IGNORED -> landed on a real person the dataset never labeled
                         (neither credited nor penalized — the whole point)
              FP      -> landed on nothing
            Renders overlays so the IGNORED boxes can be eyeballed: if those
            grey boxes are obviously real people, the ignore list is doing
            exactly what it should.

Usage:
    python3 lagenda_map_audit.py --verify \
        --csv /workspace/datasets/lagenda_full/annotations/lagenda_annotation.csv \
        --eval-dir /workspace/datasets/lagenda_full/eval_v2 \
        --images /workspace/lagenda_eval/lagenda_yolo/images/val \
        --exclude-list /workspace/exp12/train_full.txt

    python3 lagenda_map_audit.py --trace \
        --eval-dir /workspace/datasets/lagenda_full/eval_v2 \
        --raw /workspace/mapdump/y26n_gradsupp/lagenda/raw \
        --images /workspace/lagenda_eval/lagenda_yolo/images/val \
        --n 6 --out /workspace/lagenda_audit_overlays

    python3 lagenda_map_audit.py --selftest
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

MATCH_IOU = 0.5          # same as map_eval / match_boxes
IGNORE_IOA = 0.5         # same as map_eval's ignore rule
CLASS_NAMES = {0: "Woman", 1: "Man", 2: "Child"}


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def ioa(box, region):
    ix1, iy1 = max(box[0], region[0]), max(box[1], region[1])
    ix2, iy2 = min(box[2], region[2]), min(box[3], region[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    a = (box[2] - box[0]) * (box[3] - box[1])
    return inter / a if a > 0 else 0.0


def read_yolo(path: Path):
    """-> [(cls, xyxy_norm)] from cxcywh lines; missing file -> []."""
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        c = int(float(p[0]))
        cx, cy, w, h = [float(x) for x in p[1:5]]
        out.append((c, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
    return out


# ------------------------------------------------------------------ verify
def verify(csv_path: Path, eval_dir: Path, images_dir: Path,
           exclude_path: Path | None, log=print):
    """Recount from the CSV independently, then reconcile against disk."""
    have = {p.stem for p in images_dir.rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}} \
        if images_dir.exists() else None
    excl = set()
    if exclude_path and exclude_path.exists():
        excl = {l.strip().split("/")[-1].rsplit(".", 1)[0]
                for l in open(exclude_path) if l.strip()}

    csv_total = csv_boxed = csv_labeled = 0
    csv_imgs = set()
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            stem = r["img_name"].split("/")[-1].rsplit(".", 1)[0]
            if have is not None and stem not in have:
                continue
            if stem in excl:
                continue
            csv_total += 1
            boxed = all(r[k] != "-1" for k in
                        ("person_x0", "person_y0", "person_x1", "person_y1"))
            if not boxed:
                continue
            csv_boxed += 1
            csv_imgs.add(stem)
            if r["age"] not in ("-1", "") and r["gender"] not in ("-1", ""):
                csv_labeled += 1
    csv_ignore = csv_boxed - csv_labeled

    disk_all = sum(len(read_yolo(p)) for p in (eval_dir / "labels").glob("*.txt"))
    disk_cls = sum(len(read_yolo(p)) for p in (eval_dir / "labels_3class").glob("*.txt"))
    disk_ign = sum(len(read_yolo(p)) for p in (eval_dir / "ignore").glob("*.txt"))
    disk_man = sum(1 for _ in open(eval_dir / "gt.jsonl")) \
        if (eval_dir / "gt.jsonl").exists() else 0
    disk_imgs = len(list((eval_dir / "labels").glob("*.txt")))

    per_cls = Counter()
    for p in (eval_dir / "labels_3class").glob("*.txt"):
        for c, _ in read_yolo(p):
            per_cls[CLASS_NAMES.get(c, c)] += 1

    log(f"{'quantity':<38} {'from CSV':>10} {'on disk':>10}  {'':<6}")
    rows = [("images with >=1 person box", len(csv_imgs), disk_imgs),
            ("person boxes (total)", csv_boxed, disk_all),
            ("  scoreable (human age+gender)", csv_labeled, disk_cls),
            ("  IGNORE (box, no human label)", csv_ignore, disk_ign),
            ("manifest rows (gt.jsonl)", csv_labeled, disk_man)]
    ok = True
    for name, a, b in rows:
        good = (a == b)
        ok &= good
        log(f"{name:<38} {a:>10,} {b:>10,}  {'OK' if good else 'MISMATCH':<8}")
    log("")
    log(f"arithmetic: {disk_cls:,} scoreable + {disk_ign:,} ignore "
        f"= {disk_cls + disk_ign:,} vs {disk_all:,} total person boxes  "
        f"{'OK' if disk_cls + disk_ign == disk_all else 'MISMATCH'}")
    ok &= (disk_cls + disk_ign == disk_all)
    log(f"share of people that are scoreable: "
        f"{100*disk_cls/max(disk_all,1):.1f}%  (the other "
        f"{100*disk_ign/max(disk_all,1):.1f}% are IGNORED, not counted as errors)")
    log(f"scoreable classes: " + ", ".join(f"{k}={v:,}" for k, v in sorted(per_cls.items())))
    log("")
    log("VERIFIED" if ok else "FAILED — numbers do not reconcile")
    return ok


# ------------------------------------------------------------------- trace
def trace(eval_dir: Path, raw_dir: Path, images_dir: Path, n: int,
          out_dir: Path | None, conf: float, log=print):
    """Show the three outcomes per detection on n sample images."""
    stems = sorted(p.stem for p in (eval_dir / "labels_3class").glob("*.txt"))
    # prefer images that actually exercise the ignore path
    ranked = sorted(stems, key=lambda s: -len(read_yolo(eval_dir / "ignore" / f"{s}.txt")))
    picked = ranked[:n]

    totals = Counter()
    for stem in picked:
        gt = read_yolo(eval_dir / "labels_3class" / f"{stem}.txt")
        ign = [b for _, b in read_yolo(eval_dir / "ignore" / f"{stem}.txt")]
        sc = raw_dir / f"{stem}.json"
        if not sc.exists():
            log(f"[{stem}] no sidecar, skipped")
            continue
        d = json.loads(sc.read_text())
        w, h = d["width"], d["height"]
        dets = sorted(((x["conf"], x["cls"],
                        (x["box_xyxy"][0]/w, x["box_xyxy"][1]/h,
                         x["box_xyxy"][2]/w, x["box_xyxy"][3]/h))
                       for x in d["detections"]
                       if not x.get("excluded") and x["conf"] >= conf),
                      reverse=True)

        used = [False] * len(gt)
        outcomes = []
        for cf, cls, box in dets:
            best, bi = 0.0, -1
            for i, (gc, gb) in enumerate(gt):
                if used[i]:
                    continue
                v = iou(box, gb)
                if v > best:
                    best, bi = v, i
            if best >= MATCH_IOU and bi >= 0:
                used[bi] = True
                same = gt[bi][0] == cls
                outcomes.append(("MATCH" if same else "WRONG", cf, cls, gt[bi][0], best, box))
            elif any(ioa(box, r) >= IGNORE_IOA for r in ign):
                outcomes.append(("IGNORED", cf, cls, None, 0.0, box))
            else:
                outcomes.append(("FP", cf, cls, None, 0.0, box))
        misses = sum(1 for u in used if not u)

        c = Counter(o[0] for o in outcomes)
        totals.update(c)
        totals["MISS"] += misses
        totals["scoreable"] += len(gt)
        totals["ignore_boxes"] += len(ign)
        log(f"\n[{stem}]  scoreable GT={len(gt)}  ignore GT={len(ign)}  "
            f"detections>={conf}={len(dets)}")
        for kind, cf, cls, gcls, ov, _ in outcomes:
            extra = (f" as {CLASS_NAMES.get(cls,cls)} vs GT {CLASS_NAMES.get(gcls,gcls)}"
                     f" (IoU {ov:.2f})") if gcls is not None else \
                    f" as {CLASS_NAMES.get(cls,cls)}"
            log(f"   {kind:<8} conf {cf:.2f}{extra}")
        if misses:
            log(f"   MISS     {misses} labeled person(s) nobody detected")

        if out_dir:
            _render(images_dir, stem, gt, ign, outcomes, out_dir)

    log("\n=== totals over sampled images ===")
    log(f"  scoreable GT people        : {totals['scoreable']}")
    log(f"  ignore (unlabeled) people  : {totals['ignore_boxes']}")
    log(f"  MATCH  (correct)           : {totals['MATCH']}")
    log(f"  WRONG  (right person, wrong class): {totals['WRONG']}")
    log(f"  IGNORED (hit unlabeled person, DISCARDED): {totals['IGNORED']}")
    log(f"  FP     (hit nothing)       : {totals['FP']}")
    log(f"  MISS   (labeled person not found): {totals['MISS']}")
    log("\nNote how many detections land in IGNORED — under the old 1-person-per-image")
    log("ground truth every one of those would have been scored as a FALSE POSITIVE.")
    if out_dir:
        log(f"\noverlays -> {out_dir}   green=scoreable GT  grey=ignored GT")
        log("            yellow=MATCH  orange=IGNORED  red=FP")
    return totals


def _render(images_dir: Path, stem, gt, ign, outcomes, out_dir: Path):
    from PIL import Image, ImageDraw
    p = next((images_dir / f"{stem}{e}" for e in (".jpg", ".jpeg", ".png")
              if (images_dir / f"{stem}{e}").exists()), None)
    if not p:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    im = Image.open(p).convert("RGB")
    W, H = im.size
    dr = ImageDraw.Draw(im)
    def box(b, col, wd):
        dr.rectangle([b[0]*W, b[1]*H, b[2]*W, b[3]*H], outline=col, width=wd)
    for b in ign:
        box(b, (150, 150, 150), 3)                    # unlabeled real people
    for _, b in gt:
        box(b, (0, 210, 0), 5)                        # scoreable GT
    colors = {"MATCH": (255, 220, 0), "WRONG": (255, 120, 0),
              "IGNORED": (255, 150, 40), "FP": (230, 0, 0)}
    for kind, _, _, _, _, b in outcomes:
        box(b, colors[kind], 2)
    im.save(out_dir / f"{stem}.jpg", quality=80)


# ---------------------------------------------------------------- selftest
def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ev = td / "ev"
        (ev / "labels").mkdir(parents=True)
        (ev / "labels_3class").mkdir()
        (ev / "ignore").mkdir()
        # image x: 1 labeled Woman at centre-left, 2 unlabeled people right
        (ev / "labels" / "x.txt").write_text(
            "0 0.2 0.5 0.2 0.4\n0 0.6 0.5 0.1 0.3\n0 0.8 0.5 0.1 0.3\n")
        (ev / "labels_3class" / "x.txt").write_text("0 0.2 0.5 0.2 0.4\n")
        (ev / "ignore" / "x.txt").write_text("0 0.6 0.5 0.1 0.3\n0 0.8 0.5 0.1 0.3\n")
        (ev / "gt.jsonl").write_text('{"id":"x_0"}\n')
        raw = td / "raw"; raw.mkdir()
        raw.joinpath("x.json").write_text(json.dumps({
            "image": "x.jpg", "width": 100, "height": 100, "detections": [
                # exactly on the labeled Woman -> MATCH
                {"cls": 0, "box_xyxy": [10, 30, 30, 70], "conf": 0.9, "excluded": None},
                # exactly on an unlabeled person -> IGNORED
                {"cls": 1, "box_xyxy": [55, 35, 65, 65], "conf": 0.8, "excluded": None},
                # empty corner -> FP
                {"cls": 1, "box_xyxy": [0, 0, 5, 5], "conf": 0.7, "excluded": None}]}))
        t = trace(ev, raw, td / "noimgs", 1, None, 0.45, log=lambda *_: None)
        assert t["MATCH"] == 1 and t["IGNORED"] == 1 and t["FP"] == 1, t
        assert t["MISS"] == 0, t

        # verify(): build a CSV that reconciles with those files
        csvp = td / "a.csv"
        csvp.write_text(
            "img_name,age,gender,face_x0,face_y0,face_x1,face_y1,"
            "person_x0,person_y0,person_x1,person_y1\n"
            "l/x.jpg,30,F,1,1,2,2,10,30,30,70\n"
            "l/x.jpg,-1,-1,1,1,2,2,55,35,65,65\n"
            "l/x.jpg,-1,-1,1,1,2,2,75,35,85,65\n")
        class _P(type(td)):
            pass
        imgs = td / "imgs"; imgs.mkdir(); (imgs / "x.jpg").write_bytes(b"")
        assert verify(csvp, ev, imgs, None, log=lambda *_: None), "verify failed"
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--csv")
    ap.add_argument("--eval-dir")
    ap.add_argument("--images")
    ap.add_argument("--exclude-list")
    ap.add_argument("--raw", help="sidecar dir of the model to trace")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--conf", type=float, default=0.45)
    ap.add_argument("--out", help="overlay output dir")
    a = ap.parse_args()

    if a.selftest:
        selftest(); return
    if a.verify:
        if not (a.csv and a.eval_dir and a.images):
            ap.error("--verify needs --csv --eval-dir --images")
        ok = verify(Path(a.csv), Path(a.eval_dir), Path(a.images),
                    Path(a.exclude_list) if a.exclude_list else None)
        sys.exit(0 if ok else 1)
    if a.trace:
        if not (a.eval_dir and a.raw and a.images):
            ap.error("--trace needs --eval-dir --raw --images")
        trace(Path(a.eval_dir), Path(a.raw), Path(a.images), a.n,
              Path(a.out) if a.out else None, a.conf)
        return
    ap.error("pick --verify, --trace or --selftest")


if __name__ == "__main__":
    sys.exit(main())
