#!/usr/bin/env python3
"""
HARAMBLUR — EXP-2026-11 Stage B: quantify temporal instability from the raw
per-frame sidecars written by video_flicker_probe.py. CPU-only, no model.

Key idea: people don't blink in and out of existence between frames, so on
continuous video every high-frequency inconsistency in the detections IS the
flicker problem — no ground truth needed. Detections are linked into tracks by
IoU (class-agnostic: identity is spatial, class is an attribute of the track),
then each instability gets its own number:

  existence   gaps per track, gap-length histogram, blur on/off toggles per min
  gap cause   "threshold gap" (a sub-threshold detection still overlapped the
              person) vs "miss gap" (nothing there at all) — decides whether a
              dual-threshold fix would work
  confidence  per-track mean/std + fraction of frames within the straddle band
              just above the production threshold
  box jitter  median consecutive-frame IoU + center drift / box diagonal
  class flip  transitions per track-minute + full transition matrix
  exposure    for each track whose majority class is the viewer's blur target:
              seconds visibly UNBLURRED inside the track's lifetime (gap or
              wrong class), and how many distinct exposure events
  phantoms    short-lived tracks (blip < ~0.3 s) that pop a blur onto nothing

    python3 flicker_metrics.py --in /workspace/innovation-lab/exp11_flicker/probe \
        --conf 0.1 --out /workspace/innovation-lab/exp11_flicker/report_conf10
    python3 flicker_metrics.py --in ./probe --render \
        --videos /workspace/innovation-lab/exp11_flicker/videos --out ./report_conf10
    python3 flicker_metrics.py --selftest

Outputs: report.json + a per-video stdout summary; with --render also
<stem>_annotated.mp4 (boxes + the blur a male viewer would see; magenta=Woman,
blue=Man, orange=Child, thin gray=sub-threshold raw detection).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

CLASS_NAMES = {0: "Woman", 1: "Man", 2: "Child"}
BLUR_TARGET = {"male": 0, "female": 1}   # class a viewer of that gender must not see


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def derive_fps(frames, meta):
    """Effective probed fps from the sidecar's decoder timestamps. Container fps
    metadata can be bogus (1ms-timebase WebM reports 1000 fps), so timestamps
    are the authority; meta fps is only the fallback for degenerate spans."""
    if len(frames) >= 2:
        span_ms = frames[-1][1] - frames[0][1]
        if span_ms > 0:
            fps = (len(frames) - 1) / (span_ms / 1000.0)
            if 0.5 <= fps <= 240.0:
                return fps
    base = meta.get("fps_sane") or meta.get("fps", 30.0)
    return base / max(meta.get("every", 1), 1)


def load_sidecar(jsonl_path: Path):
    frames = []
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            frames.append((r["f"], r["t"], [tuple(d) for d in r["dets"]]))
    meta_path = jsonl_path.with_name(jsonl_path.stem + ".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return frames, meta


# ---------------------------------------------------------------------------
# track building
# ---------------------------------------------------------------------------

def build_tracks(frames, conf_thr, track_iou=0.3, max_gap=15):
    """Greedy frame-to-frame IoU association of thresholded detections.
    frames: [(fidx, t_ms, dets)] with dets = [(cls, x1, y1, x2, y2, conf)].
    Returns list of tracks: {"entries": [(seq_idx, det)], "first", "last"}
    where seq_idx is the index into `frames` (uniform probed-frame steps)."""
    open_tracks, closed = [], []
    for si, (_fidx, _t, dets) in enumerate(frames):
        cands = [d for d in dets if d[5] >= conf_thr]
        pairs = []
        for ti, tr in enumerate(open_tracks):
            for di, d in enumerate(cands):
                j = iou(tr["entries"][-1][1][1:5], d[1:5])
                if j >= track_iou:
                    pairs.append((j, ti, di))
        pairs.sort(reverse=True)
        used_t, used_d = set(), set()
        for j, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti); used_d.add(di)
            open_tracks[ti]["entries"].append((si, cands[di]))
        for di, d in enumerate(cands):
            if di not in used_d:
                open_tracks.append({"entries": [(si, d)]})
        still = []
        for tr in open_tracks:
            if si - tr["entries"][-1][0] > max_gap:
                closed.append(tr)
            else:
                still.append(tr)
        open_tracks = still
    closed.extend(open_tracks)
    for tr in closed:
        tr["first"] = tr["entries"][0][0]
        tr["last"] = tr["entries"][-1][0]
    closed.sort(key=lambda t: t["first"])
    return closed


# ---------------------------------------------------------------------------
# per-track metrics
# ---------------------------------------------------------------------------

def analyze_track(tr, frames, conf_thr, straddle_band=0.05, gap_iou=0.3):
    present = {si: d for si, d in tr["entries"]}
    span = range(tr["first"], tr["last"] + 1)
    gaps, cur = [], []
    for si in span:
        if si in present:
            if cur:
                gaps.append(cur)
                cur = []
        else:
            cur.append(si)
    gap_kinds = []
    for g in gaps:
        sub_hits = 0
        for si in g:
            prev = max(s for s in present if s < g[0])
            ref_box = present[prev][1:5]
            raw = frames[si][2]
            if any(d[5] < conf_thr and iou(d[1:5], ref_box) >= gap_iou
                   for d in raw):
                sub_hits += 1
        gap_kinds.append("threshold_gap" if sub_hits >= (len(g) + 1) // 2
                         else "miss_gap")
    confs = [d[5] for _, d in tr["entries"]]
    mean_c = sum(confs) / len(confs)
    std_c = (sum((c - mean_c) ** 2 for c in confs) / len(confs)) ** 0.5
    straddle = sum(1 for c in confs if c < conf_thr + straddle_band) / len(confs)

    step_ious, drifts = [], []
    ent = tr["entries"]
    for (sa, da), (sb, db) in zip(ent, ent[1:]):
        if sb - sa == 1:
            step_ious.append(iou(da[1:5], db[1:5]))
            cxa, cya = (da[1] + da[3]) / 2, (da[2] + da[4]) / 2
            cxb, cyb = (db[1] + db[3]) / 2, (db[2] + db[4]) / 2
            diag = ((da[3] - da[1]) ** 2 + (da[4] - da[2]) ** 2) ** 0.5
            drifts.append((((cxb - cxa) ** 2 + (cyb - cya) ** 2) ** 0.5)
                          / max(diag, 1e-6))
    cls_seq = [d[0] for _, d in tr["entries"]]
    majority = Counter(cls_seq).most_common(1)[0][0]
    flips = [(a, b) for a, b in zip(cls_seq, cls_seq[1:]) if a != b]

    return {
        "first": tr["first"], "last": tr["last"],
        "present_frames": len(ent), "span_frames": len(span),
        "majority_class": majority,
        "gaps": [len(g) for g in gaps], "gap_kinds": gap_kinds,
        "conf_mean": round(mean_c, 3), "conf_std": round(std_c, 3),
        "straddle_frac": round(straddle, 3),
        "jitter_iou_median": round(median(step_ious), 3) if step_ious else None,
        "jitter_iou_p10": round(pctile(step_ious, 10), 3) if step_ious else None,
        "drift_median": round(median(drifts), 4) if drifts else None,
        "class_flips": len(flips), "flip_pairs": flips,
        "cls_seq": cls_seq, "present_set": set(present),
    }


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def pctile(xs, p):
    s = sorted(xs)
    if not s:
        return None
    k = min(len(s) - 1, max(0, int(round(p / 100 * (len(s) - 1)))))
    return s[k]


# ---------------------------------------------------------------------------
# per-video rollup
# ---------------------------------------------------------------------------

def analyze_video(frames, meta, conf_thr, track_iou, max_gap,
                  min_track_s=0.5, phantom_max_s=0.3):
    fps = derive_fps(frames, meta)
    minutes = len(frames) / fps / 60.0
    tracks = build_tracks(frames, conf_thr, track_iou, max_gap)
    infos = [analyze_track(t, frames, conf_thr) for t in tracks]

    min_frames = max(3, int(round(min_track_s * fps)))
    phantom_frames = max(2, int(round(phantom_max_s * fps)))
    real = [i for i in infos if i["present_frames"] >= min_frames]
    phantoms = [i for i in infos if i["present_frames"] <= phantom_frames]

    gap_hist = Counter()
    gap_kind = Counter()
    toggles = 0
    flip_matrix = Counter()
    flips = 0
    for i in real:
        for g in i["gaps"]:
            gap_hist[g] += 1
        for k in i["gap_kinds"]:
            gap_kind[k] += 1
        toggles += 2 * len(i["gaps"])
        flips += i["class_flips"]
        for a, b in i["flip_pairs"]:
            flip_matrix[f"{CLASS_NAMES.get(a, a)}->{CLASS_NAMES.get(b, b)}"] += 1

    exposure = {}
    for viewer, target in BLUR_TARGET.items():
        exp_frames = exp_events = n_target = 0
        for i in real:
            if i["majority_class"] != target:
                continue
            n_target += 1
            covered_prev = True
            for si in range(i["first"], i["last"] + 1):
                if si in i["present_set"]:
                    k = si - i["first"]
                    seq_pos = sorted(i["present_set"]).index(si)
                    covered = i["cls_seq"][seq_pos] == target
                else:
                    covered = False
                if not covered:
                    exp_frames += 1
                    if covered_prev:
                        exp_events += 1
                covered_prev = covered
        exposure[viewer] = {
            "target_tracks": n_target,
            "exposed_seconds": round(exp_frames / fps, 2),
            "exposure_events": exp_events,
        }

    jm = [i["jitter_iou_median"] for i in real if i["jitter_iou_median"] is not None]
    dm = [i["drift_median"] for i in real if i["drift_median"] is not None]
    return {
        "frames": len(frames), "fps_probed": round(fps, 2),
        "minutes": round(minutes, 2), "conf_thr": conf_thr,
        "tracks_total": len(infos), "tracks_real": len(real),
        "phantom_blips": len(phantoms),
        "phantom_blips_per_min": round(len(phantoms) / max(minutes, 1e-6), 1),
        "toggles_per_min": round(toggles / max(minutes, 1e-6), 1),
        "gap_len_hist": dict(sorted(gap_hist.items())),
        "gap_kinds": dict(gap_kind),
        "class_flips_per_min": round(flips / max(minutes, 1e-6), 1),
        "flip_matrix": dict(flip_matrix),
        "jitter_iou_median": round(median(jm), 3) if jm else None,
        "drift_median": round(median(dm), 4) if dm else None,
        "conf_straddle_frac_mean": round(
            sum(i["straddle_frac"] for i in real) / len(real), 3) if real else None,
        "exposure": exposure,
        "tracks": [{k: v for k, v in i.items()
                    if k not in ("present_set", "cls_seq")} for i in infos],
    }


# ---------------------------------------------------------------------------
# render — annotated video with the blur a viewer would actually see
# ---------------------------------------------------------------------------

def render_video(video_path: Path, frames, meta, conf_thr, out_path: Path,
                 viewer="male"):
    import cv2
    target = BLUR_TARGET[viewer]
    colors = {0: (255, 0, 255), 1: (255, 128, 0), 2: (0, 165, 255)}  # BGR
    cap = cv2.VideoCapture(str(video_path))
    fps = min(max(derive_fps(frames, meta), 1.0), 120.0)
    w, h = meta.get("width"), meta.get("height")
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                         fps, (w, h))
    by_fidx = {f: dets for f, _t, dets in frames}
    fidx = -1
    while True:
        ok, im = cap.read()
        if not ok:
            break
        fidx += 1
        if fidx not in by_fidx:
            continue
        dets = by_fidx[fidx]
        for c, x1, y1, x2, y2, cf in dets:
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            if x2 <= x1 or y2 <= y1:
                continue
            if cf < conf_thr:
                cv2.rectangle(im, (x1, y1), (x2, y2), (128, 128, 128), 1)
                continue
            if int(c) == target:
                roi = im[y1:y2, x1:x2]
                small = cv2.resize(roi, (max(1, (x2 - x1) // 16),
                                         max(1, (y2 - y1) // 16)))
                im[y1:y2, x1:x2] = cv2.resize(
                    small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
            col = colors.get(int(c), (200, 200, 200))
            cv2.rectangle(im, (x1, y1), (x2, y2), col, 2)
            cv2.putText(im, f"{CLASS_NAMES.get(int(c), c)} {cf:.2f}",
                        (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, col, 1, cv2.LINE_AA)
        cv2.putText(im, f"f={fidx} dets={sum(1 for d in dets if d[5] >= conf_thr)}",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1,
                    cv2.LINE_AA)
        vw.write(im)
    cap.release()
    vw.release()


# ---------------------------------------------------------------------------
# selftest — synthetic sidecar with hand-computable ground truth
# ---------------------------------------------------------------------------

def selftest():
    fps = 30.0
    frames = []
    for f in range(100):
        dets = []
        if f in (50, 51):                       # dropout, sub-threshold det remains
            dets.append((0, 100.0, 80.0, 300.0, 480.0, 0.2))
        else:
            cls = 1 if f in (70, 71, 72) else 0
            j = (f % 3) * 2.0
            dets.append((cls, 100.0 + j, 80.0, 300.0 + j, 480.0, 0.7))
        if f in (10, 11, 12):                    # phantom blip above threshold
            dets.append((0, 500.0, 90.0, 560.0, 200.0, 0.55))
        frames.append((f, f / fps * 1000, dets))
    meta = {"fps": fps, "every": 1, "width": 640, "height": 480}
    r = analyze_video(frames, meta, conf_thr=0.5, track_iou=0.3, max_gap=15)

    assert r["tracks_total"] == 2, r["tracks_total"]
    assert r["tracks_real"] == 1
    assert r["phantom_blips"] == 1
    assert r["gap_len_hist"] == {2: 1}, r["gap_len_hist"]
    assert r["gap_kinds"] == {"threshold_gap": 1}, r["gap_kinds"]
    assert r["toggles_per_min"] > 0
    main_tr = [t for t in r["tracks"] if t["present_frames"] >= 15][0]
    assert main_tr["class_flips"] == 2, main_tr["class_flips"]
    assert r["flip_matrix"] == {"Woman->Man": 1, "Man->Woman": 1}
    exp = r["exposure"]["male"]
    assert exp["target_tracks"] == 1
    # exposed: 2 gap frames + 3 flipped frames = 5 -> 5/30 s; events: gap + flip
    assert abs(exp["exposed_seconds"] - 5 / 30) < 0.02, exp
    assert exp["exposure_events"] == 2, exp
    assert r["exposure"]["female"]["target_tracks"] == 0
    assert main_tr["jitter_iou_median"] is not None
    print("selftest OK: tracks/gaps/gap-cause/flips/exposure/phantoms all "
          "match hand-computed values")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", help="probe output dir (sidecars)")
    ap.add_argument("--conf", type=float, default=0.1,
                    help="production confidence threshold to simulate")
    ap.add_argument("--track-iou", type=float, default=0.3)
    ap.add_argument("--max-gap-s", type=float, default=0.5,
                    help="max unseen seconds before a track is closed")
    ap.add_argument("--viewer", choices=["male", "female"], default="male",
                    help="whose blur to render in --render mode")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--videos", help="dir with the original videos (for --render)")
    ap.add_argument("--out", default="./exp11_report")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.indir:
        ap.error("--in is required (or --selftest)")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for jsonl in sorted(Path(args.indir).glob("*.jsonl")):
        if jsonl.name == "probe_summary.json":
            continue
        frames, meta = load_sidecar(jsonl)
        if not frames:
            print(f"[warn] {jsonl.name}: empty, skipped")
            continue
        fps_probed = derive_fps(frames, meta)
        max_gap = max(3, int(round(args.max_gap_s * fps_probed)))
        r = analyze_video(frames, meta, args.conf, args.track_iou, max_gap)
        report[jsonl.stem] = r
        print(f"\n=== {jsonl.stem} ({r['minutes']} min @ {r['fps_probed']} fps, "
              f"thr={args.conf}) ===")
        print(f"  tracks: {r['tracks_real']} real / {r['tracks_total']} total · "
              f"phantom blips {r['phantom_blips']} ({r['phantom_blips_per_min']}/min)")
        print(f"  blur toggles/min {r['toggles_per_min']} · gaps {r['gap_len_hist']} "
              f"· cause {r['gap_kinds']}")
        print(f"  class flips/min {r['class_flips_per_min']} · {r['flip_matrix']}")
        print(f"  jitter IoU median {r['jitter_iou_median']} · center drift "
              f"{r['drift_median']} · conf straddle {r['conf_straddle_frac_mean']}")
        for viewer, e in r["exposure"].items():
            print(f"  exposure[{viewer}]: {e['exposed_seconds']}s unblurred over "
                  f"{e['target_tracks']} target tracks, {e['exposure_events']} events")
        if args.render:
            if not args.videos:
                raise SystemExit("--render needs --videos")
            vids = [p for p in Path(args.videos).iterdir()
                    if p.stem == jsonl.stem]
            if not vids:
                print(f"[warn] no source video for {jsonl.stem}, render skipped")
            else:
                out_mp4 = out_dir / f"{jsonl.stem}_annotated.mp4"
                render_video(vids[0], frames, meta, args.conf, out_mp4,
                             args.viewer)
                print(f"  rendered -> {out_mp4}")

    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\n[done] report.json -> {out_dir}")


if __name__ == "__main__":
    main()
