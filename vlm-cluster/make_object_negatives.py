#!/usr/bin/env python3
"""
EXP-2026-03 — build the hand-verified human-shaped-object negatives set
(toys / dolls / mannequins / statues) from Open Images candidates.

Open Images is NON-exhaustive, so its labels alone can't guarantee "no person"
— that's why this set is small and HAND-VERIFIED. Workflow (three steps):

  1. --download    pull candidate images per class via FiftyOne's Open Images
                   V7 loader (image-level labels), copy into --out with the
                   class name prefixed to each filename (doll_xxx.jpg).
  2. --make-sheet  write contact_sheet.html in --out. A human reviews EVERY
                   image against the pre-registered ruling (EXP-2026-03 §2.2):
                   REJECT if it contains a real person OR a photo/poster/
                   printed depiction of a real person (those count as people —
                   SAM firing there would be CORRECT, not a false positive).
                   Write rejected filenames, one per line, into rejects.txt.
  3. --apply-rejects rejects.txt   move rejects to --out/rejected/ and print
                   the final counts per class.

    pip install fiftyone
    python3 make_object_negatives.py --download --out /workspace/datasets/object_set \
        --classes Doll "Teddy bear" Sculpture --per-class 100
    python3 make_object_negatives.py --make-sheet --out /workspace/datasets/object_set
    # ... human review -> rejects.txt ...
    python3 make_object_negatives.py --apply-rejects rejects.txt --out /workspace/datasets/object_set

  --list-classes <substring> searches Open Images' class list (e.g. to check
  whether a "Mannequin" class exists before assuming it does).

NOTE: folder name must NOT contain the string "negative" — the production
autolabel_sam.py skips such paths (`if "negative" in img_path: continue`).
"""
from __future__ import annotations

import argparse
import html
import shutil
import sys
from pathlib import Path

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def _require_fiftyone():
    try:
        import fiftyone  # noqa: F401
        import fiftyone.zoo  # noqa: F401
    except ImportError:
        sys.exit("fiftyone is required for --download/--list-classes: pip install fiftyone")


def _oi_boxable_class_names():
    """Open Images boxable class display names — WITHOUT downloading images.
    Tries FiftyOne's helper first, else fetches Google's class CSV directly."""
    try:
        import fiftyone.utils.openimages as fouo
        return list(fouo.get_classes())
    except Exception:
        import csv
        import io
        import urllib.request
        for url in (
            "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv",
            "https://storage.googleapis.com/openimages/2018_04/class-descriptions-boxable.csv",
        ):
            try:
                data = urllib.request.urlopen(url, timeout=30).read().decode()
                return [r[1] for r in csv.reader(io.StringIO(data)) if len(r) >= 2]
            except Exception:
                continue
        raise


def list_classes(substring: str):
    names = _oi_boxable_class_names()
    hits = sorted(c for c in names if substring.lower() in c.lower())
    print(f"{len(hits)} Open Images boxable classes matching {substring!r} "
          f"(of {len(names)} total):")
    for c in hits:
        print("  ", c)


def download(out_dir: Path, classes, per_class: int, split: str):
    _require_fiftyone()
    import fiftyone.zoo as foz
    out_dir.mkdir(parents=True, exist_ok=True)
    if "negative" in str(out_dir).lower():
        sys.exit(f"refusing: output path {out_dir} contains 'negative' — "
                 "autolabel_sam.py would silently skip every image in it")
    total = 0
    for cls in classes:
        slug = cls.lower().replace(" ", "")
        print(f"== downloading up to {per_class} candidates for {cls!r} ({split}) ==")
        ds = foz.load_zoo_dataset(
            "open-images-v7", split=split,
            label_types=["classifications"], classes=[cls],
            max_samples=per_class, shuffle=True, seed=51,
            dataset_name=f"exp03_{slug}_{split}", drop_existing_dataset=True,
        )
        n = 0
        for sample in ds:
            src = Path(sample.filepath)
            dst = out_dir / f"{slug}_{src.name}"
            if not dst.exists():
                shutil.copy2(src, dst)
            n += 1
        total += n
        print(f"   copied {n} images as {slug}_*")
    print(f"done: {total} candidate images in {out_dir}")
    print("next: --make-sheet, then hand-review per the EXP-2026-03 §2.2 ruling")


def make_sheet(out_dir: Path):
    imgs = sorted(p for p in out_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not imgs:
        sys.exit(f"no images found in {out_dir}")
    parts = ["""<!doctype html><meta charset='utf-8'><title>hand-verify object negatives</title>
<style>body{font-family:sans-serif;background:#111;color:#eee;margin:16px}
.g{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:10px}
.c{background:#1c1c22;padding:6px;border-radius:6px;font-size:11px;word-break:break-all}
img{width:100%;border-radius:4px}</style>
<h2>Hand-verification — EXP-2026-03 object negatives</h2>
<p><b>REJECT</b> (add filename to rejects.txt) if the image contains a real person
<b>or a photo/poster/printed depiction of a real person</b> (per the pre-registered
ruling those count as people — SAM firing there would be correct, not a false
positive). Mannequins/dolls/statues/toys themselves are KEEP.</p><div class='g'>"""]
    for p in imgs:
        parts.append(f"<div class='c'><img src='{html.escape(p.name)}' loading='lazy'>"
                     f"{html.escape(p.name)}</div>")
    parts.append("</div>")
    (out_dir / "contact_sheet.html").write_text("\n".join(parts))
    print(f"wrote {out_dir/'contact_sheet.html'} ({len(imgs)} images) — open in a browser")


def apply_rejects(out_dir: Path, rejects_file: Path):
    rej_dir = out_dir / "rejected"
    rej_dir.mkdir(exist_ok=True)
    names = [l.strip() for l in open(rejects_file) if l.strip()]
    moved = missing = 0
    for name in names:
        src = out_dir / name
        if src.exists():
            shutil.move(str(src), str(rej_dir / name))
            moved += 1
        else:
            missing += 1
    kept = [p for p in out_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    by_class = {}
    for p in kept:
        by_class[p.name.split("_")[0]] = by_class.get(p.name.split("_")[0], 0) + 1
    print(f"moved {moved} rejects to {rej_dir} ({missing} names not found)")
    print(f"final verified set: {len(kept)} images — {by_class}")
    print("record these counts in EXP-2026-03 Appendix B (how the set was built)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--out", type=Path, help="the object-negatives folder")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--classes", nargs="+",
                    default=["Doll", "Teddy bear", "Sculpture"])
    ap.add_argument("--per-class", type=int, default=100)
    ap.add_argument("--split", default="validation",
                    help="open-images split to pull from (validation/test)")
    ap.add_argument("--make-sheet", action="store_true")
    ap.add_argument("--apply-rejects", type=Path, metavar="REJECTS_TXT")
    ap.add_argument("--list-classes", metavar="SUBSTRING")
    args = ap.parse_args()

    if args.list_classes:
        list_classes(args.list_classes)
        return
    if not args.out:
        ap.error("--out is required")
    if args.download:
        download(args.out, args.classes, args.per_class, args.split)
    if args.make_sheet:
        make_sheet(args.out)
    if args.apply_rejects:
        apply_rejects(args.out, args.apply_rejects)
    if not (args.download or args.make_sheet or args.apply_rejects):
        ap.error("nothing to do: pass --download / --make-sheet / --apply-rejects")


if __name__ == "__main__":
    main()
