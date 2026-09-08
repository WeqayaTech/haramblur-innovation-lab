#!/usr/bin/env python3
"""
HARAMBLUR — EXP-2026-11 Stage A: run the production YOLO-MIT model over every
frame of a set of videos and log RAW detections to a per-video JSONL sidecar.

One expensive GPU pass per video; every temporal-smoothing question afterwards
(thresholds, hysteresis, EMA, class votes, detect-every-N) is an offline replay
over the sidecar via flicker_metrics.py — no model re-run, ever.

The confidence floor here is deliberately LOW (0.05, well under any plausible
production threshold) so sub-threshold detections are preserved: they are what
distinguishes "person dropped below the cutoff" from "person truly not detected"
when analyzing blur flicker.

    # pod (GPU) — EXP-2026-11 lives self-contained under /workspace/innovation-lab/:
    python3 video_flicker_probe.py \
        --videos /workspace/innovation-lab/exp11_flicker/videos \
        --run-dir /workspace/YOLO-MIT/runs/v9MIT/gelansfav14_datav2_v4 \
        --repo /workspace/YOLO-MIT --imgsz 640 \
        --out /workspace/innovation-lab/exp11_flicker/probe

    # local plumbing test (no GPU, no cv2 video, no data):
    python3 video_flicker_probe.py --selftest

Outputs (in --out):
  <stem>.jsonl      one line per processed frame:
                      {"f": idx, "t": ms, "dets": [[cls, x1, y1, x2, y2, conf], ...]}
                    boxes in ORIGINAL pixel coords, conf >= --floor-conf
  <stem>.meta.json  video fps/size/frame count + every model/probe setting

Resumable: re-running the same command skips fully-processed videos and
continues partially-processed ones from the last logged frame.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

VIDEO_EXTS = {".mp4", ".webm", ".mkv", ".avi", ".mov", ".m4v"}


def list_videos(spec: str) -> list[Path]:
    p = Path(spec)
    if p.is_dir():
        vids = sorted(q for q in p.iterdir() if q.suffix.lower() in VIDEO_EXTS)
    else:
        vids = [Path(s) for s in spec.split(",")]
    missing = [v for v in vids if not v.exists()]
    if missing:
        raise SystemExit(f"missing video(s): {missing}")
    if not vids:
        raise SystemExit(f"no videos found under {spec}")
    return vids


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def probe_video(video: Path, model, out_dir: Path, every: int,
                max_seconds: float, settings: dict) -> dict:
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"[ERROR] cv2 cannot open {video} — skipping this video")
        return {"video": video.name, "error": "cannot open"}
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # Some containers (e.g. 1ms-timebase WebM) report a bogus fps like 1000.
    # It is only used for the --max-seconds cutoff and progress estimates here;
    # per-frame timestamps come from the decoder (CAP_PROP_POS_MSEC), and
    # flicker_metrics derives the effective fps from those timestamps.
    fps_sane = fps if 1.0 <= fps <= 120.0 else 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_jsonl = out_dir / f"{video.stem}.jsonl"
    out_meta = out_dir / f"{video.stem}.meta.json"

    limit = n_frames
    if max_seconds > 0:
        limit = min(limit, int(max_seconds * fps_sane))
    todo = list(range(0, limit, every))

    done = count_lines(out_jsonl)
    if done >= len(todo):
        cap.release()
        print(f"[skip] {video.name}: already complete ({done} frames)")
        return {"video": video.name, "frames": done, "skipped": True}
    if done:
        print(f"[resume] {video.name}: {done}/{len(todo)} frames already logged")

    meta = {
        "video": video.name, "fps": fps, "fps_sane": fps_sane,
        "width": w, "height": h,
        "n_frames_total": n_frames, "n_frames_probed": len(todo),
        "every": every, "max_seconds": max_seconds, **settings,
    }
    out_meta.write_text(json.dumps(meta, indent=2))

    t0 = time.time()
    written = 0
    todo_set = set(todo)
    with open(out_jsonl, "a") as f:
        i = -1  # index within `todo` of the current probed frame
        for fidx in range(limit):
            pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)  # decoder timestamp of the
            ok, frame = cap.read()                   # frame about to be read
            if not ok:
                break
            if fidx not in todo_set:
                continue
            i += 1
            if i < done:
                continue
            t_ms = pos_ms if (pos_ms > 0 or fidx == 0) else fidx / fps_sane * 1000.0
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            dets = [[c, round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1),
                     round(cf, 4)] for (c, x1, y1, x2, y2, cf) in model.detect(img)]
            f.write(json.dumps({"f": fidx, "t": round(t_ms, 1),
                                "dets": dets}) + "\n")
            written += 1
            if written % 200 == 0:
                rate = written / max(time.time() - t0, 1e-6)
                left = (len(todo) - done - written) / max(rate, 1e-6)
                print(f"  {video.name}: {done + written}/{len(todo)} "
                      f"({rate:.1f} fps, ~{left/60:.1f} min left)", flush=True)
    cap.release()
    dt = time.time() - t0
    print(f"[done] {video.name}: {done + written}/{len(todo)} frames "
          f"in {dt/60:.1f} min")
    return {"video": video.name, "frames": done + written, "seconds": round(dt, 1)}


# ---------------------------------------------------------------------------
# selftest — plumbing only, no GPU / cv2-video / data
# ---------------------------------------------------------------------------

class FlickerStub:
    """Deterministic detections that mimic the failure modes we measure:
    a stable person whose conf oscillates, drops out briefly, and class-flips.
    Frame index is passed via set_frame() since detect() only sees the image."""

    def __init__(self):
        self.fidx = 0

    def set_frame(self, fidx):
        self.fidx = fidx

    def detect(self, img):
        f = self.fidx
        dets = []
        if f not in (50, 51):                       # 2-frame dropout
            conf = 0.6 + 0.3 * ((f % 7) / 6.0)      # oscillating confidence
            cls = 1 if f in (70, 71, 72) else 0      # brief class flip
            jitter = (f % 3) * 2.0                   # small box jitter
            dets.append((cls, 100.0 + jitter, 80.0, 300.0 + jitter, 480.0, conf))
        if f in (10, 11, 12):                        # phantom 3-frame blip
            dets.append((0, 500.0, 90.0, 560.0, 200.0, 0.35))
        return dets


def selftest():
    import tempfile
    from PIL import Image
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        stub = FlickerStub()
        img = Image.new("RGB", (640, 480), (60, 60, 60))
        path = out / "synthetic.jsonl"
        with open(path, "w") as f:
            for fidx in range(100):
                stub.set_frame(fidx)
                dets = [[c, x1, y1, x2, y2, round(cf, 4)]
                        for (c, x1, y1, x2, y2, cf) in stub.detect(img)]
                f.write(json.dumps({"f": fidx, "t": fidx / 30 * 1000,
                                    "dets": dets}) + "\n")
        (out / "synthetic.meta.json").write_text(json.dumps(
            {"video": "synthetic.mp4", "fps": 30.0, "width": 640, "height": 480,
             "n_frames_total": 100, "n_frames_probed": 100, "every": 1,
             "max_seconds": 0, "floor_conf": 0.05}))
        lines = [json.loads(l) for l in open(path)]
        assert len(lines) == 100
        assert lines[50]["dets"] == [] and lines[51]["dets"] == []
        assert len(lines[10]["dets"]) == 2
        assert lines[70]["dets"][0][0] == 1 and lines[69]["dets"][0][0] == 0
        n_resume = count_lines(path)
        assert n_resume == 100
    print("selftest OK: sidecar written, dropout/flip/phantom frames present, "
          "resume line-count works")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", help="directory of videos, or comma-separated files")
    ap.add_argument("--engine", default="mit", choices=["mit", "ultralytics"],
                    help="mit = production YOLO-MIT checkpoint (default); "
                         "ultralytics = any YOLO26/YOLOE model via "
                         "run_ultralytics_labels.UltralyticsModel "
                         "(EXP-2026-12/13 candidate arms)")
    ap.add_argument("--model-path", default=None,
                    help="ultralytics engine: weights path or hub name "
                         "(e.g. yolo26n.pt, a Spotlight fine-tune best.pt)")
    ap.add_argument("--prompts", default=None,
                    help="ultralytics engine, YOLOE only: comma-separated text "
                         "vocabulary; order = class ids ('woman,man,child' "
                         "reproduces production 0,1,2)")
    ap.add_argument("--run-dir", default="/workspace/YOLO-MIT/runs/v9MIT/gelansfav14_datav2_v4")
    ap.add_argument("--repo", default="/workspace/YOLO-MIT")
    ap.add_argument("--model-config", default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--floor-conf", type=float, default=0.05,
                    help="LOGGING floor, not the production threshold — keep low")
    ap.add_argument("--iou-nms", type=float, default=0.5)
    ap.add_argument("--max-dets", type=int, default=300)
    ap.add_argument("--class-num", type=int, default=3)
    ap.add_argument("--every", type=int, default=1, help="probe every Nth frame")
    ap.add_argument("--max-seconds", type=float, default=0, help="0 = whole video")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="/workspace/innovation-lab/exp11_flicker/probe")
    ap.add_argument("--mock", action="store_true",
                    help="use the deterministic FlickerStub instead of the real "
                         "model — end-to-end plumbing test, no GPU/weights")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    if not args.videos:
        ap.error("--videos is required (or --selftest)")

    if args.mock:
        stub = FlickerStub()

        class _Counting:
            def __init__(self):
                self.n = -1

            def detect(self, img):
                self.n += 1
                stub.set_frame(self.n)
                return stub.detect(img)

        model = _Counting()
        mc = "MOCK"
        device = "cpu"
    elif args.engine == "ultralytics":
        if not args.model_path:
            ap.error("--engine ultralytics requires --model-path")
        sys.path.insert(0, str(Path(__file__).parent))
        from run_ultralytics_labels import UltralyticsModel

        import torch
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        prompts = ([s.strip() for s in args.prompts.split(",")]
                   if args.prompts else None)
        mc = f"ultralytics:{args.model_path}"
        print(f"[model] {mc} · imgsz={args.imgsz} "
              f"· floor={args.floor_conf} · device={device}"
              + (f" · prompts={prompts}" if prompts else ""))
        model = UltralyticsModel(args.model_path, args.imgsz, args.floor_conf,
                                 args.iou_nms, device, prompts)
    else:
        sys.path.insert(0, str(Path(__file__).parent))
        sys.path.insert(0, args.repo)
        from run_model_children import MitModel, find_model_config

        import torch
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        mc = find_model_config(Path(args.run_dir), Path(args.repo),
                               args.model_config)
        print(f"[model] config={mc} · imgsz={args.imgsz} "
              f"· floor={args.floor_conf} · device={device}")
        model = MitModel(args.run_dir, args.repo, mc, args.imgsz,
                         args.floor_conf, args.iou_nms, args.max_dets,
                         args.class_num, device)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = {"engine": args.engine, "model_path": args.model_path,
                "prompts": args.prompts,
                "run_dir": args.run_dir, "model_config": str(mc),
                "imgsz": args.imgsz, "floor_conf": args.floor_conf,
                "iou_nms": args.iou_nms, "max_dets": args.max_dets,
                "class_num": args.class_num, "device": device}

    results = [probe_video(v, model, out_dir, args.every, args.max_seconds,
                           settings) for v in list_videos(args.videos)]
    (out_dir / "probe_summary.json").write_text(json.dumps(results, indent=2))
    print(f"[all done] {len(results)} videos -> {out_dir}")


if __name__ == "__main__":
    main()
