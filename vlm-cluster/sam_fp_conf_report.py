#!/usr/bin/env python3
"""
HARAMBLUR — quick check: are SAM3's false positives low-confidence?

Decides whether a "send only low-confidence boxes to Gemini" gate can work.
On PASS and the object set, EVERY detection is a false positive by
construction, so their label files ARE the FP set — no matching needed.
CrowdHuman labels (mostly real people) serve as the true-positive-ish
comparison distribution.

Handles the format question automatically: standard YOLO labels carry no
confidence column. Lines are `cls` + 2N polygon coords (odd token count) or
`cls x y w h` (5 tokens). An EVEN token count means one extra value — a
confidence — was appended; the script detects which position it sits in and
sanity-checks it against the production threshold (conf >= 0.4).

    python3 sam_fp_conf_report.py \
        --fp-dirs /workspace/exp03/sam_labels/objects /workspace/exp03/sam_labels/pass \
        --true-dirs /workspace/exp03/sam_labels/crowd \
        --out /workspace/exp07_conf

    python3 sam_fp_conf_report.py --inspect /workspace/exp03/sam_labels/objects
    python3 sam_fp_conf_report.py --selftest
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CLASS_NAMES = {0: "Woman", 1: "Man", 2: "Child", }
BUCKETS = [(0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
SWEEP = [0.5, 0.6, 0.7, 0.8, 0.9]


def parse_line(tokens: list[str]):
    """Return (cls, conf_or_None). Even token count => a conf was appended."""
    cls = int(float(tokens[0]))
    if len(tokens) % 2 == 1 or len(tokens) == 5:
        return cls, None                      # cls + 2N coords / cls x y w h
    # even count: extra value is conf — Ultralytics-style puts it LAST,
    # some custom writers put it right after the class. Prefer whichever
    # looks like a confidence given the production threshold (>= 0.4).
    last, second = float(tokens[-1]), float(tokens[1])
    for cand in (last, second):
        if 0.0 <= cand <= 1.0:
            return cls, cand
    return cls, None


def collect(dirs: list[str]):
    """files_seen counts label files with >=1 parsed detection (the labeler
    writes empty files for empty images, so raw file count is meaningless)."""
    dets, no_conf_lines, files_seen = [], 0, 0
    for d in dirs:
        for lf in sorted(Path(d).rglob("*.txt")):
            n_before = len(dets)
            for line in lf.read_text().splitlines():
                tokens = line.split()
                if len(tokens) < 5:
                    continue
                cls, conf = parse_line(tokens)
                if conf is None:
                    no_conf_lines += 1
                dets.append({"cls": cls, "conf": conf, "file": lf.name})
            if len(dets) > n_before:
                files_seen += 1
    return dets, no_conf_lines, files_seen


def report_block(name: str, dets: list[dict]) -> dict:
    confs = [d["conf"] for d in dets if d["conf"] is not None]
    out = {"name": name, "detections": len(dets), "with_conf": len(confs)}
    if not confs:
        return out
    confs.sort()
    n = len(confs)
    out["conf_mean"] = round(sum(confs) / n, 3)
    out["conf_median"] = round(confs[n // 2], 3)
    out["conf_min"] = round(confs[0], 3)
    out["conf_max"] = round(confs[-1], 3)
    out["histogram"] = {f"{lo:.1f}-{min(hi,1.0):.1f}":
                        sum(1 for c in confs if lo <= c < hi)
                        for lo, hi in BUCKETS}
    out["pct_below"] = {f"{t:.1f}": round(100 * sum(1 for c in confs if c < t) / n, 1)
                        for t in SWEEP}
    by_cls = {}
    for d in dets:
        if d["conf"] is not None:
            by_cls.setdefault(CLASS_NAMES.get(d["cls"], str(d["cls"])), []).append(d["conf"])
    out["by_class"] = {k: {"n": len(v), "mean": round(sum(v) / len(v), 3)}
                       for k, v in sorted(by_cls.items())}
    return out


def print_block(b: dict):
    print(f"\n=== {b['name']} ===")
    extra = (f"  | label files with >=1 detection: {b['images_with_detections']}"
             if "images_with_detections" in b else "")
    print(f"  detections: {b['detections']}  (with conf: {b['with_conf']}){extra}")
    if b.get("conf_mean") is None:
        return
    print(f"  conf mean {b['conf_mean']}  median {b['conf_median']}  "
          f"range [{b['conf_min']}, {b['conf_max']}]")
    print("  histogram:", "  ".join(f"{k}:{v}" for k, v in b["histogram"].items()))
    print("  % below threshold:", "  ".join(f"<{k}: {v}%" for k, v in b["pct_below"].items()))
    print("  by class:", b["by_class"])


def main():
    ap = argparse.ArgumentParser(description="SAM3 FP confidence histogram")
    ap.add_argument("--fp-dirs", nargs="+", default=[],
                    help="label dirs where EVERY detection is a false positive "
                         "(PASS, object_set runs)")
    ap.add_argument("--true-dirs", nargs="+", default=[],
                    help="label dirs that are mostly real people (crowd run) "
                         "— the comparison distribution")
    ap.add_argument("--out", help="write report.json here")
    ap.add_argument("--inspect", help="print raw sample lines from one dir and exit")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    if args.inspect:
        shown = 0
        for lf in sorted(Path(args.inspect).rglob("*.txt")):
            for line in lf.read_text().splitlines()[:2]:
                t = line.split()
                print(f"{lf.name}: {len(t)} tokens | first 3: {t[:3]} | last 2: {t[-2:]}")
                shown += 1
            if shown >= 8:
                return
        if shown == 0:
            print("no label files found")
        return

    if not args.fp_dirs:
        raise SystemExit("--fp-dirs required (or --inspect / --selftest)")

    blocks = []
    # one block PER dataset dir, so PASS and objects read separately
    for d in args.fp_dirs:
        dets_d, _, files_d = collect([d])
        b = report_block(f"FP dataset: {Path(d).name}", dets_d)
        b["images_with_detections"] = files_d
        blocks.append(b)
    fp_dets, fp_noconf, fp_files = collect(args.fp_dirs)
    fp_combined = report_block("ALL FALSE POSITIVES combined", fp_dets)
    if len(args.fp_dirs) > 1:
        blocks.append(fp_combined)
    true_block = None
    if args.true_dirs:
        tp_dets, tp_noconf, _ = collect(args.true_dirs)
        true_block = report_block("MOSTLY-TRUE (crowd) — comparison distribution", tp_dets)
        blocks.append(true_block)

    for b in blocks:
        print_block(b)

    if fp_noconf and fp_noconf == len(fp_dets):
        print("\n[!] NO CONFIDENCE COLUMN in the stored labels — the histogram "
              "cannot be built from these files.")
        print("    Next step: check how autolabel_sam.py writes labels and re-run "
              "it on object_set + pass_3k with per-detection confidence saved:")
        print("      grep -n 'write\\|conf\\|f.write\\|savetxt' /workspace/autolabel/autolabel_sam.py | head -20")
        print("    (a one-line change appending conf to each label line, run on "
              "~3.3k images, is ~1 GPU-hour)")
        return

    # the decision table: if we gate at threshold t, what % of FPs are caught,
    # and what % of (mostly) real detections get escalated too (= Gemini cost)?
    fpb = fp_combined
    if true_block and fpb.get("pct_below") and true_block.get("pct_below"):
        tpb = true_block
        print("\n=== GATE DECISION TABLE ===")
        print("  thr   FPs caught   real dets escalated (cost)")
        for t in SWEEP:
            k = f"{t:.1f}"
            print(f"  <{k}   {fpb['pct_below'][k]:>6}%      {tpb['pct_below'][k]:>6}%")
        print("  Read: a viable gate needs a row with HIGH fp-caught and LOW "
              "escalation. If FPs sit at high conf (low caught-%), the "
              "confidence gate does not work and existence needs its own check.")

    if args.out:
        outp = Path(args.out)
        outp.mkdir(parents=True, exist_ok=True)
        (outp / "report.json").write_text(json.dumps(blocks, indent=2))
        svg = make_svg(blocks)
        (outp / "conf_histograms.svg").write_text(svg)
        print(f"\n[report] -> {outp/'report.json'} + conf_histograms.svg")


def make_svg(blocks: list) -> str:
    """One self-contained SVG: a bar histogram per dataset block (house chart
    style — explicit colors, white background, renders in GitHub/ClickUp)."""
    plots = [b for b in blocks if b.get("histogram")]
    W, PH, PAD = 640, 190, 46
    H = PAD + len(plots) * PH + 8
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="sans-serif">',
             f'<rect width="{W}" height="{H}" fill="white"/>',
             f'<text x="{W/2}" y="24" text-anchor="middle" font-size="15" '
             f'font-weight="bold" fill="#222">SAM3 detection confidence by dataset</text>',
             f'<text x="{W/2}" y="40" text-anchor="middle" font-size="11" fill="#666">'
             f'production settings (conf threshold 0.4) — FP datasets: every detection is a false positive</text>']
    for i, b in enumerate(plots):
        top = PAD + i * PH
        hist = b["histogram"]
        mx = max(hist.values()) or 1
        is_fp = "FP" in b["name"] or "FALSE" in b["name"]
        color = "#c0392b" if is_fp else "#2471a3"
        parts.append(f'<text x="16" y="{top+16}" font-size="12.5" font-weight="bold" '
                     f'fill="#222">{b["name"]} — {b["detections"]} detections, '
                     f'mean conf {b.get("conf_mean","?")}</text>')
        bw = (W - 90) / len(hist)
        for j, (label, cnt) in enumerate(hist.items()):
            bh = 0 if mx == 0 else (cnt / mx) * (PH - 70)
            x = 50 + j * bw
            y = top + 30 + (PH - 70 - bh)
            parts.append(f'<rect x="{x:.0f}" y="{y:.0f}" width="{bw-8:.0f}" '
                         f'height="{bh:.0f}" fill="{color}" opacity="0.85"/>')
            parts.append(f'<text x="{x+(bw-8)/2:.0f}" y="{y-4:.0f}" text-anchor="middle" '
                         f'font-size="10.5" fill="#222">{cnt}</text>')
            parts.append(f'<text x="{x+(bw-8)/2:.0f}" y="{top+PH-24}" text-anchor="middle" '
                         f'font-size="10" fill="#555">{label}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fp = td / "fp"; tp = td / "tp"
        fp.mkdir(); tp.mkdir()
        # polygon + trailing conf (even token count): cls + 4 coords + conf
        (fp / "a.txt").write_text("0 0.1 0.1 0.2 0.2 0.45\n2 0.1 0.1 0.2 0.2 0.55\n")
        (tp / "b.txt").write_text("1 0.1 0.1 0.2 0.2 0.95\n1 0.3 0.3 0.4 0.4 0.88\n")
        dets, noconf, _ = collect([str(fp)])
        assert noconf == 0 and dets[0]["conf"] == 0.45, dets
        b = report_block("t", dets)
        assert b["pct_below"]["0.6"] == 100.0 and b["pct_below"]["0.5"] == 50.0, b
        assert b["by_class"]["Woman"]["n"] == 1
        # no-conf format (odd token count polygon): detected as conf-less
        (fp / "c.txt").write_text("0 0.1 0.1 0.2 0.2 0.3 0.3\n")
        dets2, noconf2, _ = collect([str(fp)])
        assert noconf2 == 1, (noconf2, dets2)
    print("[sam_fp_conf_report] selftest OK")


if __name__ == "__main__":
    main()
