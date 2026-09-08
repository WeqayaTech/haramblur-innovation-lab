#!/usr/bin/env python3
"""
HARAMBLUR — confidence separation zoom: FP vs verified-TP distributions.

Builds four confidence populations from conf-logged SAM3 label runs
(autolabel_sam_conf.py output — trailing conf token per line):

  PASS FP     — every detection on the 3,000 verified-empty images is an FP
  crowd TP    — detections MATCHED to CrowdHuman vbox GT (IoU >= 0.5)
  crowd FP    — unmatched detections not on an ignore region (CrowdHuman is
                exhaustively labeled, so these are real in-the-wild FPs)
  LAGENDA TP  — detections matched to the labeled person's YOLO GT box
                (unmatched LAGENDA dets are EXCLUDED — they are usually real
                unlabeled people, neither TP nor FP)

Then answers the separation question directly: is there a threshold below
which all/most FPs live while TPs survive?

    python3 conf_zoom_report.py \
        --pass-dir /workspace/exp07_conf/sam_labels/pass --pass-images 3000 \
        --crowd-dir /workspace/exp07_conf/sam_labels/crowd \
        --crowd-images-dir /workspace/datasets/crowdhuman/Images_sample500 \
        --odgt /workspace/datasets/crowdhuman/annotation_val.odgt \
        --lagenda-dir /workspace/exp07_conf/sam_labels/lagenda \
        --lagenda-gt-labels /workspace/lagenda_eval/lagenda_yolo/labels/val \
        --out /workspace/exp07_conf/zoom

    python3 conf_zoom_report.py --selftest
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_model_children import iou, match_boxes

MATCH_IOU = 0.5
THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
BUCKETS = [(0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]


def parse_labels(label_dir: Path):
    """{stem: [(conf, box_norm_xyxy), ...]} from conf-logged polygon labels."""
    out = {}
    for lf in sorted(Path(label_dir).rglob("*.txt")):
        dets = []
        for line in lf.read_text().splitlines():
            t = line.split()
            if len(t) < 6 or len(t) % 2 == 1:   # need trailing conf (even count)
                continue
            conf = float(t[-1])
            xs = [float(v) for v in t[1:-1:2]]
            ys = [float(v) for v in t[2:-1:2]]
            if not xs or not ys:
                continue
            dets.append((conf, [min(xs), min(ys), max(xs), max(ys)]))
        out[lf.stem] = dets
    return out


def image_size(path: Path):
    from PIL import Image
    with Image.open(path) as im:
        return im.size                      # (w, h)


def crowd_split(label_dir: Path, images_dir: Path, odgt_path: Path):
    """Match against odgt vbox GT -> (tp_confs, fp_confs, n_images_used)."""
    gt = {}
    with open(odgt_path) as fh:
        for line in fh:
            rec = json.loads(line)
            persons, ignores = [], []
            for g in rec.get("gtboxes", []):
                x, y, bw, bh = g.get("vbox", g.get("fbox", [0, 0, 0, 0]))
                box = [x, y, x + bw, y + bh]
                if g.get("tag") == "person" and not g.get("extra", {}).get("ignore", 0):
                    persons.append(box)
                else:
                    ignores.append(box)
            gt[rec["ID"]] = (persons, ignores)

    img_index = {p.stem: p for p in Path(images_dir).rglob("*")
                 if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
    tps, fps, used = [], [], 0
    for stem, dets in parse_labels(label_dir).items():
        if stem not in gt or stem not in img_index or not dets:
            continue
        w, h = image_size(img_index[stem])
        persons, ignores = gt[stem]
        det_tuples = [("d", b[0] * w, b[1] * h, b[2] * w, b[3] * h, conf)
                      for conf, b in dets]
        matched = match_boxes(persons, det_tuples, MATCH_IOU)
        matched_ids = {id(m[0]) for m in matched.values()}
        used += 1
        for dt in det_tuples:
            if id(dt) in matched_ids:
                tps.append(dt[5])
            else:
                box = list(dt[1:5])
                if any(iou(box, ib) >= MATCH_IOU for ib in ignores):
                    continue                 # ignore region: neither TP nor FP
                fps.append(dt[5])
    return tps, fps, used


def lagenda_tps(label_dir: Path, gt_labels_dir: Path):
    """Match in normalized space against the LAGENDA YOLO GT boxes -> TP confs.
    Unmatched detections are excluded (usually real unlabeled people)."""
    tps, used = [], 0
    for stem, dets in parse_labels(label_dir).items():
        gtf = Path(gt_labels_dir) / f"{stem}.txt"
        if not gtf.exists() or not dets:
            continue
        gt_boxes = []
        for line in gtf.read_text().splitlines():
            p = line.split()
            if len(p) >= 5:
                cx, cy, bw, bh = [float(v) for v in p[1:5]]
                gt_boxes.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
        if not gt_boxes:
            continue
        det_tuples = [("d", *b, conf) for conf, b in dets]
        matched = match_boxes(gt_boxes, det_tuples, MATCH_IOU)
        used += 1
        tps.extend(m[0][5] for m in matched.values())
    return tps, used


def stats(name, confs, n_images, kind):
    confs = sorted(confs)
    n = len(confs)
    b = {"name": name, "kind": kind, "n": n, "images": n_images,
         "per_100_images": round(100 * n / n_images, 1) if n_images else None}
    if n:
        b.update(mean=round(sum(confs) / n, 3), median=round(confs[n // 2], 3),
                 min=round(confs[0], 3), max=round(confs[-1], 3),
                 histogram={f"{lo:.1f}-{min(hi, 1.0):.1f}":
                            sum(1 for c in confs if lo <= c < hi)
                            for lo, hi in BUCKETS},
                 pct_le={f"{t:.2f}": round(100 * sum(1 for c in confs if c <= t) / n, 1)
                         for t in THRESHOLDS})
    return b


def print_report(blocks):
    print(f"{'population':<14}{'N':>7}{'images':>8}{'per100img':>11}"
          f"{'mean':>7}{'median':>8}{'min':>6}{'max':>6}")
    for b in blocks:
        if b["n"]:
            print(f"{b['name']:<14}{b['n']:>7}{b['images']:>8}"
                  f"{b['per_100_images'] if b['per_100_images'] is not None else '-':>11}"
                  f"{b['mean']:>7}{b['median']:>8}{b['min']:>6}{b['max']:>6}")

    print("\n=== SEPARATION TABLE — % of each population at conf <= t ===")
    hdr = "  t     " + "".join(f"{b['name']:>12}" for b in blocks if b["n"])
    print(hdr)
    for t in THRESHOLDS:
        k = f"{t:.2f}"
        row = f"  {k}  " + "".join(f"{b['pct_le'][k]:>11}%" for b in blocks if b["n"])
        print(row)

    fps = [b for b in blocks if b["kind"] == "FP" and b["n"]]
    tps = [b for b in blocks if b["kind"] == "TP" and b["n"]]
    if fps and tps:
        print("\n=== CLEAN-THRESHOLD CHECK ===")
        for f in fps:
            print(f"  max {f['name']} conf = {f['max']} — a gate must reach this "
                  f"to contain ALL its FPs")
            for tb in tps:
                lost = tb["pct_le"].get(f"{min(THRESHOLDS, key=lambda t: abs(t - f['max'])):.2f}")
                kept = round(100 - (lost or 0), 1)
                print(f"     at that level, {tb['name']} keeps only ~{kept}% "
                      f"of its TPs above the bar")


def main():
    ap = argparse.ArgumentParser(description="FP vs TP confidence separation")
    ap.add_argument("--pass-dir")
    ap.add_argument("--pass-images", type=int, default=3000)
    ap.add_argument("--crowd-dir")
    ap.add_argument("--crowd-images-dir")
    ap.add_argument("--odgt")
    ap.add_argument("--lagenda-dir")
    ap.add_argument("--lagenda-gt-labels")
    ap.add_argument("--out")
    ap.add_argument("--populations", default=None,
                    help='comma-separated filter, e.g. "PASS FP,crowd TP" — '
                         "other populations are computed but not shown/drawn")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    blocks = []
    if args.pass_dir:
        confs = [c for dets in parse_labels(Path(args.pass_dir)).values()
                 for c, _ in dets]
        blocks.append(stats("PASS FP", confs, args.pass_images, "FP"))
    if args.crowd_dir and args.odgt and args.crowd_images_dir:
        tp, fp, used = crowd_split(Path(args.crowd_dir),
                                   Path(args.crowd_images_dir), Path(args.odgt))
        blocks.append(stats("crowd TP", tp, used, "TP"))
        blocks.append(stats("crowd FP", fp, used, "FP"))
    if args.lagenda_dir and args.lagenda_gt_labels:
        tp, used = lagenda_tps(Path(args.lagenda_dir), Path(args.lagenda_gt_labels))
        blocks.append(stats("LAGENDA TP", tp, used, "TP"))

    if args.populations:
        keep = [n.strip() for n in args.populations.split(",")]
        blocks = [b for b in blocks if b["name"] in keep]
    print_report(blocks)
    if args.out:
        outp = Path(args.out)
        outp.mkdir(parents=True, exist_ok=True)
        (outp / "zoom_report.json").write_text(json.dumps(blocks, indent=2))
        (outp / "zoom_histograms.svg").write_text(make_svg(blocks))
        print(f"\n[report] -> {outp/'zoom_report.json'} + zoom_histograms.svg")


def make_svg(blocks: list) -> str:
    """Self-contained SVG (house style): one bar histogram per population,
    FP populations red, TP populations blue, counts on the bars."""
    plots = [b for b in blocks if b.get("histogram")]
    W, PH, PAD = 640, 200, 50
    H = PAD + len(plots) * PH + 6
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="sans-serif">',
             f'<rect width="{W}" height="{H}" fill="white"/>',
             f'<text x="{W/2}" y="22" text-anchor="middle" font-size="15" '
             f'font-weight="bold" fill="#222">SAM3 confidence — false positives vs verified true positives</text>',
             f'<text x="{W/2}" y="38" text-anchor="middle" font-size="11" fill="#666">'
             f'production settings (threshold 0.4) · crowd TP = detections matched to CrowdHuman GT at IoU&#8805;0.5</text>']
    for i, b in enumerate(plots):
        top = PAD + i * PH
        hist = b["histogram"]
        mx = max(hist.values()) or 1
        color = "#c0392b" if b["kind"] == "FP" else "#2471a3"
        sub = (f'{b["n"]} detections over {b["images"]} images '
               f'({b["per_100_images"]}/100 img)' if b.get("per_100_images") is not None
               else f'{b["n"]} detections over {b["images"]} images')
        parts.append(f'<text x="16" y="{top+14}" font-size="13" font-weight="bold" '
                     f'fill="#222">{b["name"]} — mean conf {b["mean"]}</text>')
        parts.append(f'<text x="16" y="{top+28}" font-size="11" fill="#666">{sub}</text>')
        bw = (W - 90) / len(hist)
        for j, (label, cnt) in enumerate(hist.items()):
            bh = (cnt / mx) * (PH - 84)
            x = 50 + j * bw
            y = top + 40 + (PH - 84 - bh)
            parts.append(f'<rect x="{x:.0f}" y="{y:.0f}" width="{bw-8:.0f}" '
                         f'height="{bh:.0f}" fill="{color}" opacity="0.85"/>')
            parts.append(f'<text x="{x+(bw-8)/2:.0f}" y="{y-4:.0f}" text-anchor="middle" '
                         f'font-size="10.5" fill="#222">{cnt}</text>')
            parts.append(f'<text x="{x+(bw-8)/2:.0f}" y="{top+PH-26}" text-anchor="middle" '
                         f'font-size="10" fill="#555">{label}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # lagenda: one GT person, one matching det (TP, 0.9), one far det (excluded)
        (td / "lab").mkdir(); (td / "gt").mkdir()
        (td / "lab" / "img.txt").write_text(
            "0 0.30 0.30 0.30 0.70 0.50 0.70 0.50 0.30 0.90\n"     # matches GT
            "0 0.05 0.05 0.05 0.10 0.10 0.10 0.10 0.05 0.45\n")    # far away
        (td / "gt" / "img.txt").write_text("0 0.4 0.5 0.2 0.4\n")  # cx cy w h
        tps, used = lagenda_tps(td / "lab", td / "gt")
        assert used == 1 and tps == [0.9], (used, tps)
        b = stats("LAGENDA TP", tps, used, "TP")
        assert b["pct_le"]["0.85"] == 0.0 and b["pct_le"]["0.90"] == 100.0
        # pass parsing
        confs = [c for dets in parse_labels(td / "lab").values() for c, _ in dets]
        assert sorted(confs) == [0.45, 0.9]
    print("[conf_zoom_report] selftest OK")


if __name__ == "__main__":
    main()
