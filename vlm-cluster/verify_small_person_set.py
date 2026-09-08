#!/usr/bin/env python3
"""
Verify a small-person set built by `build_small_person_set.py` — and produce
the evidence, not just a verdict.

Two outputs:
  * hard checks, printed and exit-coded: counts line up, every scored person is
    actually inside the size band, everyone outside it is an ignore region, the
    held-out stems really are absent, the occlusion ratios survived the emit.
  * `VERIFY.html` — sampled images with the emitted boxes drawn on them, each
    accompanied by ZOOMED crops of that image's smallest people. The zoom panel
    is the point: a 24 px box is unreadable at page scale, so "is that really a
    person?" can only be answered by magnifying it.
  * `size_hist.svg` — standalone chart of the person-height distribution.

    python3 verify_small_person_set.py --set /workspace/datasets/smallperson_v1/crowd_small
    python3 verify_small_person_set.py --selftest
"""
from __future__ import annotations

import argparse
import html
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from run_autolabel_on_manifest import seg_boxes                    # noqa: E402
from eval_negatives_crowd import load_odgt                         # noqa: E402
from build_small_person_set import (BANDS, band_of, input_height,  # noqa: E402
                                    list_images, load_exclude)


BAND_EPS_PX = 0.01      # float round-trip slack, not a real tolerance


class Checks:
    def __init__(self):
        self.rows = []

    def add(self, name, ok, detail=""):
        self.rows.append((name, bool(ok), detail))
        return ok

    @property
    def failed(self):
        return [r for r in self.rows if not r[1]]

    def report(self, log=print):
        for name, ok, detail in self.rows:
            log(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        n_ok = sum(1 for _n, ok, _d in self.rows if ok)
        log(f"\n{n_ok}/{len(self.rows)} checks passed")
        return not self.failed


def collect(set_dir: Path, imgsz: int):
    """Re-read everything from the emitted files — never from the manifest."""
    from PIL import Image
    man = json.loads((set_dir / "manifest.json").read_text())
    imgs = list_images(set_dir / "images")
    per_image = []
    for p in imgs:
        with Image.open(p) as im:
            w, h = im.size
        scored = seg_boxes(set_dir / "labels" / f"{p.stem}.txt", w, h)
        ign = seg_boxes(set_dir / "ignore" / f"{p.stem}.txt", w, h)
        per_image.append({"path": p, "stem": p.stem, "w": w, "h": h,
                          "scored": scored, "ignore": ign})
    return man, per_image


def run_checks(set_dir: Path, man, per_image, imgsz: int, log=print):
    c = Checks()
    n_img = len(per_image)
    cutoff = man.get("max_height_px", 96.0)
    space = man.get("space", "native")
    # arms that paste onto a blank canvas have no odgt and no ignore regions —
    # every person is in-band by construction
    # only the mined arm carries an odgt; every GENERATED arm (synth, paste,
    # avatars, and anything added later) is scored from YOLO labels + gt.jsonl
    is_synth = man.get("arm") != "mined_real"

    c.add("images present", n_img > 0, f"{n_img} images")
    c.add("one label file per image",
          all(r["scored"] is not None for r in per_image),
          f"{sum(1 for r in per_image if r['scored'] is None)} missing "
          "(absent != empty — an unprocessed image must never look like 'found nothing')")
    c.add("one ignore file per image",
          all(r["ignore"] is not None for r in per_image))

    n_scored = sum(len(r["scored"] or []) for r in per_image)
    n_ign = sum(len(r["ignore"] or []) for r in per_image)
    c.add("manifest person count matches files on disk",
          man.get("n_persons_scored") in (None, n_scored),
          f"manifest {man.get('n_persons_scored')} vs files {n_scored}")

    # every scored box inside the band, measured the way the set says it was
    over, heights = [], []
    for r in per_image:
        for cls, x1, y1, x2, y2 in (r["scored"] or []):
            hp = y2 - y1
            hh = hp if space == "native" else input_height(hp, r["w"], r["h"], imgsz)
            heights.append(hh)
            # Labels are stored as 6-decimal normalized values, so a box sitting
            # exactly ON the cutoff round-trips back a few ten-thousandths of a
            # pixel over it. Tolerate sub-hundredth-pixel overshoot; anything
            # larger is a real out-of-band person and must fail.
            if hh > cutoff + BAND_EPS_PX:
                over.append((r["stem"], round(hh, 4)))
    # For odgt sets sized by full-body box, the VISIBLE box drawn in the label
    # is <= the full-body height, so this is a necessary condition, not the
    # definition; for vbox/yolo sets it is exactly the selection rule.
    c.add(f"every scored person <= {cutoff:g}px ({space})", not over,
          f"{len(over)} over-band, e.g. {over[:3]}" if over else
          f"max {max(heights):.1f}px over {len(heights)} people")

    c.add("normalized label values within [0,1]", True)  # replaced below
    c.rows.pop()
    bad = []
    for r in per_image:
        txt = (set_dir / "labels" / f"{r['stem']}.txt").read_text()
        for line in txt.splitlines():
            vals = [float(v) for v in line.split()[1:]] if line.strip() else []
            if any(v < -1e-6 or v > 1 + 1e-6 for v in vals):
                bad.append(r["stem"])
                break
    c.add("normalized label values within [0,1]", not bad, f"{len(bad)} bad files")

    # held-out stems really absent
    ex_path = man.get("exclude_stems")
    if ex_path and Path(ex_path).exists():
        ex = load_exclude(ex_path)
        leaked = {r["stem"] for r in per_image} & ex
        c.add("held-out benchmark stems absent", not leaked,
              f"{len(ex)} held out, {len(leaked)} leaked")

    if is_synth:
        c.add("synth: ignore files present (empty is valid)",
              all(r["ignore"] is not None for r in per_image))
    else:
        odgt = set_dir / "annotations" / "persons.odgt"
        c.add("persons.odgt exists", odgt.exists())
        if odgt.exists():
            od = load_odgt(odgt)
            c.add("odgt record count == image count", len(od) == n_img,
                  f"{len(od)} vs {n_img}")
            c.add("odgt person count == label person count",
                  sum(len(v["persons"]) for v in od.values()) == n_scored,
                  f"{sum(len(v['persons']) for v in od.values())} vs {n_scored}")
            c.add("odgt ignore count == ignore label count",
                  sum(len(v["ignores"]) for v in od.values()) == n_ign,
                  f"{sum(len(v['ignores']) for v in od.values())} vs {n_ign}")
            occ = [p["occ_ratio"] for v in od.values() for p in v["persons"]]
            # If every ratio is exactly 1.0 the emit copied vbox into fbox and
            # the light/partial/heavy breakdown is silently dead.
            c.add("occlusion ratios survived the emit",
                  bool(occ) and any(o < 0.999 for o in occ),
                  f"{sum(1 for o in occ if o < 0.999)}/{len(occ)} occluded, "
                  f"mean {sum(occ) / len(occ):.3f}" if occ else "no persons")

    cls_mix = Counter(cls for r in per_image for cls, *_ in (r["scored"] or []))
    log(f"\nclass mix: {dict(cls_mix)}   scored {n_scored} · ignore {n_ign}")
    log(f"height bands ({space}): " + "  ".join(
        f"{n}={sum(1 for h in heights if band_of(h) == n)}" for _lo, _hi, n in BANDS))
    return c, heights


# ---------------------------------------------------------------------------
def size_hist_svg(heights, out_path: Path, cutoff, space):
    """Standalone SVG — explicit colours, white background, no external refs."""
    if not heights:
        return
    nb = 24
    hi = max(cutoff, max(heights))
    step = hi / nb
    counts = [0] * nb
    for h in heights:
        counts[min(nb - 1, int(h / step))] += 1
    W, H, pad = 720, 340, 52
    top = max(counts) or 1
    bw = (W - 2 * pad) / nb
    bars = []
    for i, n in enumerate(counts):
        bh = (H - 2 * pad) * n / top
        x, y = pad + i * bw, H - pad - bh
        bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw - 2:.1f}" '
                    f'height="{bh:.1f}" fill="#2f6fb5"/>')
    ticks = []
    for i in range(0, nb + 1, 4):
        x = pad + i * bw
        ticks.append(f'<line x1="{x:.1f}" y1="{H - pad}" x2="{x:.1f}" y2="{H - pad + 5}" '
                     f'stroke="#333"/><text x="{x:.1f}" y="{H - pad + 20}" '
                     f'font-family="sans-serif" font-size="12" fill="#333" '
                     f'text-anchor="middle">{i * step:.0f}</text>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<rect width="{W}" height="{H}" fill="#ffffff"/>
<text x="{W / 2}" y="26" font-family="sans-serif" font-size="15" fill="#111" text-anchor="middle">Person height distribution ({space} px) — n={len(heights)}</text>
{''.join(bars)}
<line x1="{pad}" y1="{H - pad}" x2="{W - pad}" y2="{H - pad}" stroke="#333"/>
<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{H - pad}" stroke="#333"/>
{''.join(ticks)}
<text x="{W / 2}" y="{H - 8}" font-family="sans-serif" font-size="12" fill="#333" text-anchor="middle">height (px)</text>
<text x="14" y="{H / 2}" font-family="sans-serif" font-size="12" fill="#333" transform="rotate(-90 14 {H / 2})" text-anchor="middle">people</text>
</svg>'''
    out_path.write_text(svg)


def render_html(set_dir: Path, man, per_image, sample, out_path: Path,
                imgsz: int, n_zoom=4):
    from PIL import Image, ImageDraw
    from build_error_gallery import _b64_jpeg, _resize_max

    space = man.get("space", "native")
    cards = []
    for r in sample:
        with Image.open(r["path"]) as im:
            img = im.convert("RGB")
        full = img.copy()
        d = ImageDraw.Draw(full)
        for _c, x1, y1, x2, y2 in (r["ignore"] or []):
            d.rectangle([x1, y1, x2, y2], outline=(150, 150, 150), width=3)
        for _c, x1, y1, x2, y2 in (r["scored"] or []):
            d.rectangle([x1, y1, x2, y2], outline=(0, 230, 60), width=3)
        frame = _b64_jpeg(_resize_max(full, 620), quality=72)

        smallest = sorted(r["scored"] or [], key=lambda b: b[4] - b[3])[:n_zoom]
        zooms = []
        for _c, x1, y1, x2, y2 in smallest:
            pad = 0.25 * max(x2 - x1, y2 - y1)
            crop = img.crop((max(0, int(x1 - pad)), max(0, int(y1 - pad)),
                             min(r["w"], int(x2 + pad)), min(r["h"], int(y2 + pad))))
            if crop.width < 4 or crop.height < 4:
                continue
            k = max(1, int(150 / max(crop.width, crop.height)))
            big = crop.resize((crop.width * k, crop.height * k), Image.NEAREST)
            hp = y2 - y1
            hin = input_height(hp, r["w"], r["h"], imgsz)
            zooms.append(
                f"<figure><img src='data:image/jpeg;base64,{_b64_jpeg(big, 80)}'>"
                f"<figcaption>{hp:.0f}px native · {hin:.0f}px @{imgsz}</figcaption></figure>")

        cards.append(
            f"<div class='card'><h3>{html.escape(r['stem'])}</h3>"
            f"<div class='meta'>{r['w']}×{r['h']} · "
            f"{len(r['scored'] or [])} scored · {len(r['ignore'] or [])} ignored</div>"
            f"<img class='frame' src='data:image/jpeg;base64,{frame}'>"
            f"<div class='zooms'>{''.join(zooms)}</div></div>")

    out_path.write_text(f"""<!doctype html><meta charset="utf-8">
<title>small-person set — verification</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px;background:#fff;color:#111;max-width:1400px}}
.card{{border:1px solid #ddd;border-radius:8px;padding:14px;margin:18px 0}}
h3{{margin:0 0 4px;font-size:15px}} .meta{{color:#666;font-size:12px;margin-bottom:8px}}
.frame{{max-width:100%;border:1px solid #ccc}}
.zooms{{display:flex;flex-wrap:wrap;gap:12px;margin-top:12px}}
figure{{margin:0;text-align:center}} figure img{{border:2px solid #0e6;image-rendering:pixelated}}
figcaption{{font-size:11px;color:#444;margin-top:3px}}
.key{{background:#f6f6f6;padding:10px 14px;border-radius:6px;font-size:13px}}
</style>
<h1>Small-person set — verification</h1>
<div class="key"><b>{html.escape(str(set_dir))}</b><br>
green = person scored as small · grey = ignore region (not scored either way)<br>
Zoom panels are nearest-neighbour magnifications of that image's smallest people —
they are the only way to judge whether a 24&nbsp;px box really contains a person.<br>
Size axis: {html.escape(space)} px, cutoff {man.get('max_height_px')},
size box {man.get('size_box')}.</div>
<p><img src="size_hist.svg" style="max-width:760px"></p>
{''.join(cards)}""")


def _selftest():
    from PIL import Image
    tmp = Path("/tmp/_verify_smallperson")
    for sub in ("images", "labels", "ignore", "annotations"):
        (tmp / sub).mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (200, 100), (30, 30, 30)).save(tmp / "images" / "a.jpg")
    # one 40px person (in band), one 80px ignore region
    (tmp / "labels" / "a.txt").write_text("0 0.25 0.5 0.1 0.4\n")
    (tmp / "ignore" / "a.txt").write_text("0 0.75 0.5 0.2 0.8\n")
    (tmp / "annotations" / "persons.odgt").write_text(json.dumps({
        "ID": "a", "gtboxes": [
            {"tag": "person", "vbox": [40, 30, 20, 40], "fbox": [40, 30, 20, 80],
             "extra": {}},
            {"tag": "person", "extra": {"ignore": 1}, "fbox": [140, 10, 40, 80]}]}) + "\n")
    (tmp / "manifest.json").write_text(json.dumps(
        {"n_persons_scored": 1, "max_height_px": 96.0, "space": "native",
         "size_box": "fbox", "arm": "mined_real"}))

    man, per_image = collect(tmp, 640)
    assert len(per_image) == 1 and len(per_image[0]["scored"]) == 1
    c, heights = run_checks(tmp, man, per_image, 640, log=lambda *a: None)
    assert not c.failed, c.failed
    assert abs(heights[0] - 40.0) < 1e-6, heights

    # a box exactly ON the cutoff must PASS despite normalization round-trip
    (tmp / "labels" / "a.txt").write_text("0 0.25 0.5 0.1 0.960000\n")   # 96.0px of 100
    man, per_image = collect(tmp, 640)
    c0, _ = run_checks(tmp, man, per_image, 640, log=lambda *a: None)
    assert not any("<= 96px" in n for n, ok, _d in c0.rows if not ok), c0.rows
    # ...but half a pixel over must still FAIL
    (tmp / "labels" / "a.txt").write_text("0 0.25 0.5 0.1 0.965\n")      # 96.5px
    man, per_image = collect(tmp, 640)
    ch, _ = run_checks(tmp, man, per_image, 640, log=lambda *a: None)
    assert any("<= 96px" in n for n, ok, _d in ch.rows if not ok), ch.rows

    # an over-band person must FAIL the band check
    (tmp / "labels" / "a.txt").write_text("0 0.25 0.5 0.1 0.99\n")
    man, per_image = collect(tmp, 640)
    c2, _ = run_checks(tmp, man, per_image, 640, log=lambda *a: None)
    assert any("<= 96px" in n for n, ok, _d in c2.rows if not ok), c2.rows

    # a copied fbox must FAIL the occlusion check (the bug this guards)
    (tmp / "labels" / "a.txt").write_text("0 0.25 0.5 0.1 0.4\n")
    (tmp / "annotations" / "persons.odgt").write_text(json.dumps({
        "ID": "a", "gtboxes": [{"tag": "person", "vbox": [40, 30, 20, 40],
                                "fbox": [40, 30, 20, 40], "extra": {}},
                               {"tag": "person", "extra": {"ignore": 1},
                                "fbox": [140, 10, 40, 80]}]}) + "\n")
    man, per_image = collect(tmp, 640)
    c3, _ = run_checks(tmp, man, per_image, 640, log=lambda *a: None)
    assert any("occlusion" in n for n, ok, _d in c3.rows if not ok), c3.rows

    # a generated arm must NOT be asked for an odgt it never has
    gen = json.loads((tmp / "manifest.json").read_text())
    gen["arm"] = "avatars"
    (tmp / "manifest.json").write_text(json.dumps(gen))
    (tmp / "annotations" / "persons.odgt").unlink()
    man_g, per_g = collect(tmp, 640)
    cg, _ = run_checks(tmp, man_g, per_g, 640, log=lambda *a: None)
    assert not any("odgt" in n for n, _ok, _d in cg.rows), cg.rows

    size_hist_svg([10, 20, 30, 96], tmp / "size_hist.svg", 96, "native")
    svg = (tmp / "size_hist.svg").read_text()
    assert svg.startswith("<svg") and "<rect" in svg and "http" not in svg.replace(
        "http://www.w3.org/2000/svg", ""), "SVG must be self-contained"
    print("verify_small_person_set selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", dest="set_dir")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-html", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    if not args.set_dir:
        ap.error("--set is required (or --selftest)")

    set_dir = Path(args.set_dir)
    print(f"verifying {set_dir}")
    man, per_image = collect(set_dir, args.imgsz)
    c, heights = run_checks(set_dir, man, per_image, args.imgsz)
    print()
    ok = c.report()

    size_hist_svg(heights, set_dir / "size_hist.svg",
                  man.get("max_height_px", 96), man.get("space", "native"))
    if not args.no_html:
        # prefer images that actually contain the smallest people
        ranked = sorted((r for r in per_image if r["scored"]),
                        key=lambda r: min(b[4] - b[3] for b in r["scored"]))
        pool = ranked[:len(ranked) // 2] or ranked
        sample = random.Random(args.seed).sample(pool, min(args.samples, len(pool)))
        render_html(set_dir, man, per_image, sample, set_dir / "VERIFY.html", args.imgsz)
        print(f"wrote {set_dir / 'VERIFY.html'} and {set_dir / 'size_hist.svg'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
