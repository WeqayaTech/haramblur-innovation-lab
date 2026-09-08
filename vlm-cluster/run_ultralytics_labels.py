#!/usr/bin/env python3
"""
EXP-2026-12 / EXP-2026-13 — Stage A runner for Ultralytics-family candidate
models (YOLO26, YOLOE) over an image directory.

Runs any model loadable via `ultralytics.YOLO` and writes, per image:

  <out>/labels/<stem>.txt      YOLO bbox lines "cls cx cy w h" (normalized),
                               thresholded at --conf. NO trailing confidence —
                               seg_boxes() in run_autolabel_on_manifest.py
                               silently drops 6-field lines (it reads them as a
                               malformed polygon), so the scorer-facing labels
                               stay 5-field. An EMPTY file is written when a
                               processed image has zero detections — "processed,
                               found nothing" must stay distinguishable from
                               "not processed" (the EXP-2026-04 lesson).
  <out>/raw/<stem>.json        every detection above --floor (default 0.05)
                               with conf + class name + distractor verdicts —
                               the log-raw sidecar, so conf sweeps and
                               distractor policies replay offline, no re-run.

Class handling (choose one):
  --map identity               model's own class ids pass through (a model
                               fine-tuned on Spotlight labels already speaks
                               {0 Woman, 1 Man, 2 Child}).
  --map coco-person            keep only COCO class 0 (person), emit as class 0.
                               Component 1 scoring is class-agnostic
                               (match_boxes ignores class), so the id carries
                               no meaning — do NOT feed these labels to a
                               Component 2/3 scorer.
  --prompts "woman,man,child"  YOLOE only: set the text vocabulary. Prompt
                               order IS the class id order, so
                               "woman,man,child" reproduces the production
                               {0,1,2} mapping. Requires a YOLOE checkpoint.
  --distractors "statue,mannequin,doll"
                               YOLOE only, appended after --prompts: extra
                               vocabulary entries whose detections are EXCLUDED
                               from the label files (recorded in the raw
                               sidecar with excluded=true). The pre-registered
                               EXP-2026-13 anti-FP arm.

Also exports `UltralyticsModel` with .detect(img) -> [(cls,x1,y1,x2,y2,conf)]
(pixel xyxy), duck-typed to MitModel so video_flicker_probe.py can drive a
candidate model over the EXP-2026-11 clips with --engine ultralytics.

Resumable: an image whose label file already exists is skipped (delete the
labels dir to force a re-run). Zero-detection images write an empty label file
so resume never re-processes them.

    # YOLO26 pretrained, person-only, Component 1 datasets
    python3 run_ultralytics_labels.py --model yolo26n.pt --map coco-person \
        --images /workspace/datasets/crowdhuman/Images \
        --out /workspace/exp12/y26n_coco/crowd --conf 0.45 --floor 0.05

    # YOLOE zero-shot, production vocabulary + distractor arm
    python3 run_ultralytics_labels.py --model yoloe-11s-seg.pt \
        --prompts "woman,man,child" --distractors "statue,mannequin,doll" \
        --images /workspace/datasets/object_set/images \
        --out /workspace/exp13/yoloe_s_distract/object_set

    python3 run_ultralytics_labels.py --selftest   # no GPU, no weights, no net
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ---------------------------------------------------------------- model wrap
import two_axis as TA


class UltralyticsModel:
    """Wraps ultralytics.YOLO with the same .detect(img) contract as
    run_model_children.MitModel: returns [(cls, x1, y1, x2, y2, conf)] in
    pixel xyxy, already NMS'd/thresholded at `floor` (log-raw floor — callers
    apply their own working threshold downstream)."""

    def __init__(self, model_path: str, imgsz: int, floor: float, iou: float,
                 device: str, prompts=None, distractors=None, no_e2e: bool = False):
        self.imgsz, self.floor, self.iou, self.device = imgsz, floor, iou, device
        self.names = None          # id -> name, filled after first predict
        self.n_prompt = 0          # how many leading classes are real prompts
        if prompts:
            from ultralytics import YOLOE
            self.model = YOLOE(model_path)
            vocab = list(prompts) + list(distractors or [])
            # one-arg set_classes is the documented YOLOE path (ultralytics
            # 8.4.x); older builds need the explicit text-embedding form.
            try:
                self.model.set_classes(vocab)
            except TypeError:
                self.model.set_classes(vocab, self.model.get_text_pe(vocab))
            self.names = {i: n for i, n in enumerate(vocab)}
            self.n_prompt = len(prompts)
        else:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
        if no_e2e:
            # Benchmark the one-to-many (NMS) head — what an `end2end=False`
            # export ships. MUST be set before the first predict: fuse() strips
            # whichever head is unused, and afterwards the flag is too late
            # (KeyError: 'feats'). No-op on models that never had an e2e head.
            head = self.model.model.model[-1]
            if hasattr(head, "one2one_cv2"):
                head.end2end = False
                self.model.model.end2end = False

    def detect(self, img):
        res = self.model.predict(img, imgsz=self.imgsz, conf=self.floor,
                                 iou=self.iou, device=self.device,
                                 verbose=False)[0]
        if self.names is None:
            self.names = dict(res.names)
        out = []
        for b in res.boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            out.append((int(b.cls[0]), x1, y1, x2, y2, float(b.conf[0])))
        return out


class TwoLabelModel:
    """EXP-2026-17 decode: a two-axis (gender x age) head read as TWO argmax
    groups instead of one.

    Stock `predict()` cannot express this — its NMS collapses each box to a
    single argmax class and `DetectionPredictor` never passes NMS's
    `multi_label` flag — so this reads the raw `(1, 4+6, N)` tensor and does
    its own decode. That decode is also the reference for the browser, which
    has to do exactly the same three steps in JS.

    Contract: `.detect()` returns the SAME 6-tuple every other model here
    returns, with `cls` already COLLAPSED to two_axis.COLLAPSED_NAMES
    (0 Woman, 1 Man, 2 Child, 3 UnknownGender) so `run()` and every existing
    scorer work unchanged. The per-axis detail is kept in `.last_axes`,
    aligned index-for-index, and lands in the raw sidecar.

    Box selection is deliberately identical to stock single-label inference —
    confidence is the max over ALL six channels and NMS is class-agnostic — so
    the geometry can be parity-checked against `predict(agnostic_nms=True)`.
    See `selftest`, which asserts that parity on a real forward pass.
    """

    def __init__(self, model_path: str, imgsz: int, floor: float, iou: float,
                 device: str, max_det: int = 300):
        import torch
        from ultralytics import YOLO
        self.torch = torch
        self.imgsz, self.floor, self.iou = imgsz, floor, iou
        self.device, self.max_det = device, max_det
        self.model = YOLO(model_path)
        head = self.model.model.model[-1]
        if hasattr(head, "one2one_cv2"):
            # Read the one-to-many head: the e2e (1,300,6) form has a single
            # `cls` scalar per row and structurally cannot carry two axes.
            # Must be set before the first forward — fuse() strips the unused
            # head and afterwards the flag is too late (KeyError: 'feats').
            head.end2end = False
            self.model.model.end2end = False
        nc = getattr(head, "nc", None)
        if nc != TA.NC:
            raise ValueError(
                f"--map two-axis needs a {TA.NC}-channel head "
                f"({list(TA.CLASS_NAMES)}), got nc={nc}")
        # eval mode is what makes Detect return the inference tensor; in
        # train mode it returns the raw feature dict and the decode KeyErrors
        self.model.model.eval().to(device)
        self.names = {i: n for i, n in enumerate(TA.COLLAPSED_NAMES)}
        self.n_prompt = 0
        self.last_axes = []

    def _preprocess(self, img):
        """Letterbox exactly as the stock predictor does, so parity holds."""
        import numpy as np
        from ultralytics.data.augment import LetterBox
        arr = np.asarray(img)[:, :, ::-1]                    # RGB -> BGR
        lb = LetterBox((self.imgsz, self.imgsz), auto=False, stride=32)
        t = lb(image=arr).transpose(2, 0, 1)[::-1]           # HWC BGR -> CHW RGB
        t = self.torch.from_numpy(np.ascontiguousarray(t)).float().div(255)[None]
        return t.to(self.model.device)

    def detect(self, img):
        t = self._preprocess(img)
        with self.torch.no_grad():
            raw = self.model.model(t)
        pred = raw[0] if isinstance(raw, (list, tuple)) else raw   # (1, 4+nc, N)
        dets, self.last_axes = decode_two_axis(
            pred, t.shape[2:], (img.size[1], img.size[0]),
            self.floor, self.iou, self.max_det)
        return dets


def decode_two_axis(pred, in_shape, orig_shape, floor, iou, max_det=300):
    """Raw `(1, 4+6, N)` head output -> ([(cls,x1,y1,x2,y2,conf)], [axes]).

    The three steps the browser must also implement, kept as a pure function
    so they can be parity-checked against stock NMS without a network in the
    way (see selftest_two_axis_parity):

      1. conf = max over ALL six channels; drop anything below `floor`.
      2. class-agnostic NMS at `iou` — one box is one person, and staying
         class-agnostic is what makes this comparable with stock inference.
      3. one argmax per channel group -> (gender, age), then collapse to a
         two_axis.COLLAPSED_NAMES id for the label file.
    """
    import torch
    import torchvision
    from ultralytics.utils import ops

    p = pred[0].transpose(0, 1).float()          # (N, 4+nc)
    scores = p[:, 4:]                            # already sigmoid
    conf = scores.amax(1)
    keep = conf >= floor
    if not bool(keep.any()):
        return [], []
    xyxy = ops.xywh2xyxy(p[keep, :4])
    conf, scores = conf[keep], scores[keep]
    idx = torchvision.ops.nms(xyxy, conf, iou)[:max_det]
    xyxy = ops.scale_boxes(in_shape, xyxy[idx].clone(), orig_shape)
    conf, scores = conf[idx], scores[idx]

    dets, axes = [], []
    for k in range(xyxy.shape[0]):
        gid, gconf, aid, aconf = TA.split_scores(scores[k].tolist())
        x1, y1, x2, y2 = (float(v) for v in xyxy[k].tolist())
        dets.append((TA.collapse(gid, aid), x1, y1, x2, y2, float(conf[k])))
        axes.append({"gender_cls": gid, "gender_name": TA.CLASS_NAMES[gid],
                     "gender_conf": round(gconf, 4),
                     "age_cls": aid, "age_name": TA.CLASS_NAMES[aid],
                     "age_conf": round(aconf, 4)})
    return dets, axes


class MitAdapter:
    """Same .detect contract, backed by the production YOLO-MIT checkpoint
    (run_model_children.MitModel) — lets EXP-2026-12 Phase A dump the
    production model's labels over the Component 1 datasets with the exact
    machinery used for the candidates."""

    def __init__(self, run_dir: str, repo: str, model_config, imgsz: int,
                 floor: float, iou_nms: float, device: str):
        sys.path.insert(0, str(Path(__file__).parent))
        sys.path.insert(0, repo)
        from run_model_children import MitModel, find_model_config
        mc = find_model_config(Path(run_dir), Path(repo), model_config)
        self.inner = MitModel(run_dir, repo, mc, imgsz, floor, iou_nms,
                              300, 3, device)
        self.names = {0: "Woman", 1: "Man", 2: "Child"}
        self.n_prompt = 0

    def detect(self, img):
        return [(int(c), x1, y1, x2, y2, float(cf))
                for (c, x1, y1, x2, y2, cf) in self.inner.detect(img)]


class _StubModel:
    """Selftest stand-in: fixed detections, no ultralytics import."""

    def __init__(self, dets_by_stem, names, n_prompt=0):
        self._d, self.names, self.n_prompt = dets_by_stem, names, n_prompt

    def detect(self, img):
        return self._d.get(getattr(img, "_stem", None), [])


# ---------------------------------------------------------------- the runner
def map_detection(det, mode: str, n_prompt: int):
    """-> (emit_cls | None, excluded_reason | None). None cls = drop from
    label file (still logged raw)."""
    cls = det[0]
    if mode == "identity":
        return cls, None
    if mode == "coco-person":
        return (0, None) if cls == 0 else (None, "non_person_class")
    if mode == "prompts":
        return (cls, None) if cls < n_prompt else (None, "distractor")
    if mode == "two-axis":
        # TwoLabelModel already collapsed the two axes to a COLLAPSED_NAMES id;
        # unknown-gender people stay IN the file so class-agnostic Component 1
        # recall is unaffected and Component 3 can score them as an abstention
        return cls, None
    raise ValueError(mode)


def run(model, images_dir: Path, out_dir: Path, conf: float, mode: str,
        limit: int = 0, log=print):
    labels, raw = out_dir / "labels", out_dir / "raw"
    labels.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    imgs = sorted(p for p in images_dir.rglob("*")
                  if p.suffix.lower() in IMG_EXTS)
    if limit:
        imgs = imgs[:limit]
    done = skipped = 0
    t0 = time.time()
    from PIL import Image
    for p in imgs:
        lf = labels / f"{p.stem}.txt"
        if lf.exists():
            skipped += 1
            continue
        img = Image.open(p).convert("RGB")
        img._stem = p.stem                      # selftest hook, harmless live
        w, h = img.size
        dets = model.detect(img)
        lines, raw_rows = [], []
        for i, (cls, x1, y1, x2, y2, cf) in enumerate(dets):
            emit, why = map_detection((cls, x1, y1, x2, y2, cf), mode,
                                      model.n_prompt)
            name = (model.names or {}).get(cls, str(cls))
            kept = emit is not None and cf >= conf
            row = {"det_index": i, "cls": cls, "cls_name": name,
                   "box_xyxy": [round(v, 2) for v in (x1, y1, x2, y2)],
                   "conf": round(cf, 4), "kept": kept, "excluded": why}
            # two-axis runs add per-axis keys; `cls`/`conf` keep their meaning
            # so conf_sweep.py and the scorers read the sidecar unchanged
            axes = getattr(model, "last_axes", None)
            if axes:
                row.update(axes[i])
            raw_rows.append(row)
            if kept:
                cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                bw, bh = (x2 - x1) / w, (y2 - y1) / h
                lines.append(f"{emit} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        lf.write_text("\n".join(lines) + ("\n" if lines else ""))
        (raw / f"{p.stem}.json").write_text(json.dumps(
            {"image": p.name, "width": w, "height": h,
             "detections": raw_rows}, indent=None))
        done += 1
        if done % 100 == 0:
            rate = done / max(time.time() - t0, 1e-6)
            log(f"  {done}/{len(imgs) - skipped} imgs · {rate:.1f} img/s")
    log(f"[done] processed={done} resumed-skip={skipped} "
        f"out={out_dir}")
    return done, skipped


def selftest_two_axis_parity(n=500, floor=0.05, iou=0.7, max_det=300):
    """Prove the two-axis decode selects exactly the boxes stock NMS does.

    The decode exists because stock `predict()` collapses each box to one
    argmax class and `DetectionPredictor` never passes NMS's `multi_label`
    flag — so nothing downstream would catch it silently disagreeing on
    geometry or confidence, and a silent confidence bug in an export has
    already cost this project once.

    This runs `decode_two_axis` and ultralytics' own `non_max_suppression`
    over the SAME synthetic `(1, 4+6, N)` head output and asserts identical
    boxes and confidences. Synthetic rather than a real forward pass on
    purpose: a randomly initialized network emits a near-CONSTANT score field,
    and a field of ties makes NMS keep an arbitrary subset, so the comparison
    would fail on tie-breaking rather than on arithmetic. What is under test
    here is the arithmetic — the network contributes nothing to it.
    """
    try:
        import torch
        import torchvision  # noqa: F401
        from ultralytics.utils import ops
        from ultralytics.utils.nms import non_max_suppression
    except ImportError as e:
        print(f"  two-axis parity: SKIPPED ({e})")
        return False

    g = torch.Generator().manual_seed(7)
    in_shape, orig_shape = (640, 640), (1080, 810)
    cxcy = torch.rand(n, 2, generator=g) * 600 + 20
    wh = torch.rand(n, 2, generator=g) * 180 + 20
    scores = torch.rand(n, TA.NC, generator=g)
    pred = torch.cat([cxcy, wh, scores], 1).transpose(0, 1)[None]   # (1, 4+nc, N)

    mine, axes = decode_two_axis(pred, in_shape, orig_shape, floor, iou, max_det)

    # the reference: ultralytics' own NMS at the same operating point
    ref = non_max_suppression(pred.clone(), floor, iou, agnostic=True,
                              max_det=max_det, nc=TA.NC)[0]
    ref_box = ops.scale_boxes(in_shape, ref[:, :4].clone(), orig_shape)

    assert len(mine) == len(ref), (
        f"decode returned {len(mine)} boxes, stock NMS returned {len(ref)} — "
        "box selection has diverged")
    assert mine, "parity fixture produced no detections; the check proves nothing"
    assert len({round(d[5], 6) for d in mine}) > 10, \
        "fixture confidences are degenerate; parity would only test tie-breaking"
    dbox = max(max(abs(a_ - b_) for a_, b_ in zip(d[1:5], ref_box[k].tolist()))
               for k, d in enumerate(mine))
    dconf = max(abs(d[5] - float(ref[k, 4])) for k, d in enumerate(mine))
    assert dbox < 1e-3, f"box mismatch vs stock NMS: max delta {dbox}"
    assert dconf < 1e-5, f"conf mismatch vs stock NMS: max delta {dconf}"

    # and the part stock cannot do: both axes on every detection
    assert len(axes) == len(mine)
    for ax, (cid, *_r) in zip(axes, mine):
        assert ax["gender_cls"] in range(*TA.GROUPS[0])
        assert ax["age_cls"] in range(*TA.GROUPS[1])
        assert cid == TA.collapse(ax["gender_cls"], ax["age_cls"])
        assert 0 <= cid < len(TA.COLLAPSED_NAMES)
    # the collapse must be exercised, not vacuously satisfied
    assert len({ax["age_cls"] for ax in axes}) > 1, "age axis never varied"
    assert len({ax["gender_cls"] for ax in axes}) > 1, "gender axis never varied"
    print(f"  two-axis parity: {len(mine)} boxes match ultralytics NMS "
          f"(max box delta {dbox:.2e} px, max conf delta {dconf:.2e}); "
          f"both axes decoded and collapsed")
    return True


def selftest_two_axis_wiring():
    """End-to-end smoke for the parts the arithmetic parity check does not
    touch: building a 6-channel model, forcing the one-to-many head, letterbox
    preprocess, and `run()` merging the per-axis keys into the sidecar while
    leaving `cls`/`conf` alone (conf_sweep.py and the scorers read those).

    Degenerate random-init scores are fine here — this asserts SHAPE and
    PLUMBING, never values. Never downloads: built from the installed yaml.
    """
    import json
    import tempfile
    try:
        from ultralytics.utils import ROOT, YAML
        from PIL import Image
        import numpy as np
    except ImportError as e:
        print(f"  two-axis wiring: SKIPPED ({e})")
        return False
    src_yaml = next(Path(ROOT).glob("cfg/models/*/yolo26.yaml"), None)
    if src_yaml is None:
        print("  two-axis wiring: SKIPPED (yolo26.yaml not in this install)")
        return False

    cfg = YAML.load(src_yaml)
    cfg["nc"] = TA.NC
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # the scale suffix must be in the FILENAME — ultralytics derives the
        # model scale from the name, not from the file contents
        y = td / "yolo26n.yaml"
        YAML.save(y, cfg)
        m = TwoLabelModel(str(y), 320, 1e-6, 0.7, "cpu")
        assert m.names == {i: n for i, n in enumerate(TA.COLLAPSED_NAMES)}

        imgs = td / "imgs"
        imgs.mkdir()
        rng = np.random.default_rng(0)
        Image.fromarray(rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
                        ).save(imgs / "a.jpg")
        out = td / "out"
        done, _ = run(m, imgs, out, conf=1e-6, mode="two-axis", log=lambda *a: None)
        assert done == 1, done

        rows = json.loads((out / "raw" / "a.json").read_text())["detections"]
        assert rows, "no detections logged"
        for r in rows:
            # the keys every existing scorer reads keep their meaning...
            assert 0 <= r["cls"] < len(TA.COLLAPSED_NAMES)
            assert 0.0 <= r["conf"] <= 1.0
            # ...and the two-axis detail is additive
            assert r["gender_cls"] in range(*TA.GROUPS[0])
            assert r["age_cls"] in range(*TA.GROUPS[1])
            assert r["cls"] == TA.collapse(r["gender_cls"], r["age_cls"])
        for line in (out / "labels" / "a.txt").read_text().splitlines():
            f = line.split()
            assert len(f) == 5, f"label lines must stay 5-field, got {len(f)}"
            assert 0 <= int(f[0]) < len(TA.COLLAPSED_NAMES)
    print(f"  two-axis wiring: {len(rows)} detections, sidecar carries both "
          f"axes, labels stayed 5-field")
    return True


# ---------------------------------------------------------------- selftest
def selftest():
    import tempfile
    from PIL import Image
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "imgs").mkdir()
        for stem in ("a", "b", "c"):
            Image.new("RGB", (100, 80)).save(td / "imgs" / f"{stem}.jpg")

        # prompts mode: 3 real classes + 1 distractor; b has only a distractor
        # (must yield an EMPTY label file); c has nothing.
        dets = {"a": [(0, 10, 10, 50, 70, 0.9),    # woman, kept
                      (3, 60, 10, 90, 70, 0.8),    # distractor, excluded
                      (1, 0, 0, 20, 20, 0.10)],    # man, under conf 0.45
                "b": [(3, 5, 5, 40, 60, 0.95)],
                "c": []}
        names = {0: "woman", 1: "man", 2: "child", 3: "statue"}
        m = _StubModel(dets, names, n_prompt=3)
        out = td / "out"
        done, _ = run(m, td / "imgs", out, conf=0.45, mode="prompts",
                      log=lambda *a: None)
        assert done == 3
        a = (out / "labels" / "a.txt").read_text().strip().splitlines()
        assert len(a) == 1 and a[0].startswith("0 "), a
        assert len(a[0].split()) == 5, "label lines must stay 5-field"
        assert (out / "labels" / "b.txt").read_text() == "", \
            "distractor-only image must write an EMPTY label file"
        assert (out / "labels" / "c.txt").exists(), \
            "zero-detection image must still write a label file"
        raw_a = json.loads((out / "raw" / "a.json").read_text())
        assert len(raw_a["detections"]) == 3, "raw sidecar logs everything"
        excl = [d for d in raw_a["detections"] if d["excluded"] == "distractor"]
        assert len(excl) == 1 and excl[0]["cls_name"] == "statue"
        sub = [d for d in raw_a["detections"] if d["conf"] < 0.45]
        assert sub and not sub[0]["kept"], "sub-conf logged raw, not emitted"

        # coco-person mode: non-person classes drop, person becomes class 0
        m2 = _StubModel({"a": [(0, 10, 10, 50, 70, 0.9),
                               (16, 0, 0, 30, 30, 0.9)]},   # a dog
                        {0: "person", 16: "dog"})
        out2 = td / "out2"
        run(m2, td / "imgs", out2, conf=0.45, mode="coco-person",
            log=lambda *a: None)
        a2 = (out2 / "labels" / "a.txt").read_text().strip().splitlines()
        assert len(a2) == 1 and a2[0].startswith("0 "), a2

        # resume: second run skips everything
        _, skipped = run(m, td / "imgs", out, conf=0.45, mode="prompts",
                         log=lambda *a: None)
        assert skipped == 3

        # round-trip through the real scorer parser
        sys.path.insert(0, str(Path(__file__).parent))
        from run_autolabel_on_manifest import seg_boxes
        boxes = seg_boxes(out / "labels" / "a.txt", 100, 80)
        assert len(boxes) == 1 and boxes[0][0] == 0, boxes
        x1, y1, x2, y2 = boxes[0][1:]
        assert abs(x1 - 10) < 1 and abs(y2 - 70) < 1, boxes
    selftest_two_axis_parity()
    selftest_two_axis_wiring()
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--engine", default="ultralytics",
                    choices=["ultralytics", "mit"],
                    help="mit = production YOLO-MIT checkpoint via "
                         "run_model_children.MitModel (EXP-2026-12 Phase A "
                         "anchor; needs --run-dir/--repo, ignores --model)")
    ap.add_argument("--run-dir",
                    default="/workspace/YOLO-MIT/runs/v9MIT/gelansfav14_datav2_v4")
    ap.add_argument("--repo", default="/workspace/YOLO-MIT")
    ap.add_argument("--model-config", default=None)
    ap.add_argument("--model")
    ap.add_argument("--images")
    ap.add_argument("--out")
    ap.add_argument("--conf", type=float, default=0.45,
                    help="working threshold for the label files "
                         "(production extension threshold = 0.45)")
    ap.add_argument("--floor", type=float, default=0.05,
                    help="log-raw floor for the sidecars")
    ap.add_argument("--iou", type=float, default=0.7,
                    help="NMS IoU (production extension = 0.7; "
                         "NMS-free models ignore it)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--map", default="identity",
                    choices=["identity", "coco-person", "two-axis"],
                    help="two-axis: EXP-2026-17 6-channel gender x age head, "
                         "read as two argmax groups and collapsed to "
                         f"{list(TA.COLLAPSED_NAMES)} for the label files "
                         "(both axes + both confidences go to the sidecar). "
                         "Score with --classes/--class-names "
                         + ",".join(TA.COLLAPSED_NAMES))
    ap.add_argument("--prompts", default=None,
                    help="YOLOE text vocabulary, comma-separated; order = "
                         "class ids ('woman,man,child' = production 0,1,2)")
    ap.add_argument("--distractors", default=None,
                    help="YOLOE extra vocabulary excluded from label files")
    ap.add_argument("--no-e2e", action="store_true",
                    help="YOLO26/YOLOv10: predict via the one-to-many (NMS) head "
                         "instead of the NMS-free one — i.e. benchmark exactly "
                         "what an `end2end=False` export deploys")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if not (args.images and args.out):
        ap.error("--images and --out required (or --selftest)")
    if args.engine == "ultralytics" and not args.model:
        ap.error("--model required for --engine ultralytics")
    if args.distractors and not args.prompts:
        ap.error("--distractors requires --prompts")

    prompts = [s.strip() for s in args.prompts.split(",")] if args.prompts \
        else None
    distract = [s.strip() for s in args.distractors.split(",")] \
        if args.distractors else None
    if args.engine == "mit":
        mode = "identity"
        model = MitAdapter(args.run_dir, args.repo, args.model_config,
                           args.imgsz, args.floor, args.iou, args.device)
        print(f"[model] YOLO-MIT {args.run_dir} imgsz={args.imgsz} "
              f"conf={args.conf} floor={args.floor}")
    elif args.map == "two-axis":
        if prompts:
            ap.error("--map two-axis is a trained 6-channel head, not a prompted one")
        mode = "two-axis"
        model = TwoLabelModel(args.model, args.imgsz, args.floor, args.iou,
                              args.device)
        print(f"[model] {args.model} TWO-AXIS imgsz={args.imgsz} "
              f"conf={args.conf} floor={args.floor} "
              f"-> {list(TA.COLLAPSED_NAMES)}")
    else:
        mode = "prompts" if prompts else args.map
        model = UltralyticsModel(args.model, args.imgsz, args.floor, args.iou,
                                 args.device, prompts, distract, args.no_e2e)
        print(f"[model] {args.model} imgsz={args.imgsz} conf={args.conf} "
              f"floor={args.floor} mode={mode}"
              + (f" prompts={prompts} distractors={distract}" if prompts
                 else ""))
    run(model, Path(args.images), Path(args.out), args.conf, mode,
        args.limit)


if __name__ == "__main__":
    main()
