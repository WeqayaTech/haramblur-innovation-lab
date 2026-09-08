#!/usr/bin/env python3
"""
EXP-2026-15 — YOLOE fine-tune with cls-gradient-suppressed Unknown labels.

Ports EXP-2026-14's arm-C mechanism (train_gradsuppress.py — the unknown
person supervises WHERE via box/DFL/assignment but never WHAT: zero
classification gradient) to the YOLOE segmentation fine-tune path.

Why the port is small (verified against pinned ultralytics==8.4.115 source):
the PE fine-tune path has NO visual prompts, so `YOLOESegModel.loss` falls
through to the standard criterion (`nn/tasks.py:1489-1491` — TVPSegmentLoss
only when `visual_prompt`), i.e. `E2ELoss(self, v8SegmentationLoss)`. And
`v8SegmentationLoss.loss` computes its det losses via
`get_assigned_targets_and_loss` (loss.py:511) — the exact method
GSDetectionLoss overrides. So:

    GSSegmentationLoss(GSDetectionLoss, v8SegmentationLoss)   # MRO does it all

MRO check: __init__ = GSDetectionLoss's (super() -> v8SegmentationLoss ->
v8DetectionLoss, then replaces the assigner with the nc+1 class-agnostic one);
loss() = v8SegmentationLoss's (seg branch intact — unknown-assigned anchors
also get MASK supervision through fg_mask/target_gt_idx, which is desired:
the unknown teaches where, including shape); get_assigned_targets_and_loss =
the GS override (cls gradient zeroed on unknown-assigned anchors).

Trainer: subclass of the stock `YOLOEPESegTrainer` (train_seg.py:60) with two
changes — the model head is built with nc = data_nc − 1 (Unknown never exists
at inference) and the model class is GSYOLOESegModel (GS criterion). The
PE-fuse steps (get_text_pe -> set_classes -> fuse -> unfreeze head convs) are
copied verbatim from the stock trainer. Text embeddings for
['Woman','Man','Child'] only. NOTE: get_text_pe needs the mobileclip text
model — already cached on the volume from EXP-2026-13 Phase 0.

Data: same yamls as EXP-2026-14 arm C (nc=4, labels_unk3 train, classes-0-2
val). Default = FULL fine-tune (all params trainable); pass --freeze N for
the linear-probe recipe.

Checkpoint caveat (same as train_gradsuppress.py): classes are top-level so
they pickle; loading a checkpoint needs this module importable, or convert
with train_gradsuppress.py --export-plain (works for these too — it remaps
by class NAME, see note below).

Usage (pod):
    python3 train_gradsuppress_yoloe.py --data /workspace/exp15/unk4_c.yaml \
        --weights yoloe-26s-seg.pt --epochs 30 --imgsz 640 --batch -1 \
        --project /workspace/exp15/train --name yoloe_gradsupp

    python3 train_gradsuppress_yoloe.py --selftest   # CPU, no downloads/data
"""
from __future__ import annotations

import argparse
import sys
from copy import deepcopy

import torch
from ultralytics.models.yolo.yoloe.train_seg import YOLOEPESegTrainer
from ultralytics.nn.tasks import YOLOESegModel
from ultralytics.utils.loss import E2ELoss, v8SegmentationLoss

from train_gradsuppress import GSDetectionLoss


class GSSegmentationLoss(GSDetectionLoss, v8SegmentationLoss):
    """Segmentation loss with cls-gradient suppression for gt class == nc.

    Everything comes from the MRO: GS __init__/assigner + GS
    get_assigned_targets_and_loss, v8SegmentationLoss.loss for the seg branch.
    """


class GSYOLOESegModel(YOLOESegModel):
    def init_criterion(self):
        return (
            E2ELoss(self, GSSegmentationLoss)
            if getattr(self, "end2end", False)
            else GSSegmentationLoss(self)
        )


class GSYOLOEPESegTrainer(YOLOEPESegTrainer):
    """YOLOEPESegTrainer with nc = data_nc - 1 and the GS criterion.

    Body mirrors the stock get_model (ultralytics 8.4.115 train_seg.py:70-113)
    with the two GS changes marked."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        nc = self.data["nc"] - 1  # GS: drop the Unknown slot from the head
        model = GSYOLOESegModel(  # GS: model class carries the GS criterion
            cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
            ch=self.data["channels"],
            nc=nc,
            verbose=verbose,
        )

        del model.model[-1].savpe

        assert weights is not None, "Pretrained weights must be provided."
        model.load(weights)

        model.eval()
        names = [self.data["names"][i] for i in range(nc)]  # GS: real classes only
        tpe = model.get_text_pe(names)
        model.set_classes(names, tpe)
        model.model[-1].fuse(model.pe)
        model.model[-1].cv3[0][2] = deepcopy(model.model[-1].cv3[0][2]).requires_grad_(True)
        model.model[-1].cv3[1][2] = deepcopy(model.model[-1].cv3[1][2]).requires_grad_(True)
        model.model[-1].cv3[2][2] = deepcopy(model.model[-1].cv3[2][2]).requires_grad_(True)
        if getattr(model.model[-1], "one2one_cv3", None) is not None:
            model.model[-1].one2one_cv3[0][2] = deepcopy(model.model[-1].cv3[0][2]).requires_grad_(True)
            model.model[-1].one2one_cv3[1][2] = deepcopy(model.model[-1].cv3[1][2]).requires_grad_(True)
            model.model[-1].one2one_cv3[2][2] = deepcopy(model.model[-1].cv3[2][2]).requires_grad_(True)

        model.train()
        return model

    def set_model_attributes(self):
        # the stock hook overwrites model.nc/names from the data yaml (nc=4) —
        # this is how the pilot checkpoint ended up carrying 4 names on a
        # 3-channel head; re-assert the real head identity afterwards
        super().set_model_attributes()
        nc = self.data["nc"] - 1
        self.model.nc = nc
        self.model.names = {i: self.data["names"][i] for i in range(nc)}


def export_plain(ckpt_path: str):
    """Re-save a checkpoint with GS classes remapped to stock ultralytics
    classes, so it loads anywhere without this module on sys.path. Also
    handles legacy __main__-pickled checkpoints when run as
    `python3 train_gradsuppress_yoloe.py --export-plain <ckpt>` (this file's
    top-level definitions populate __main__ for the unpickler)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    remap = {"GSYOLOESegModel": YOLOESegModel, "GSPlainSegModel": SegmentationModel}
    n = 0
    for key in ("model", "ema"):
        m = ckpt.get(key)
        if m is not None and m.__class__.__name__ in remap:
            m.__class__ = remap[m.__class__.__name__]
            if hasattr(m, "criterion"):
                del m.criterion
            n += 1
    out = ckpt_path.replace(".pt", "_plain.pt")
    torch.save(ckpt, out)
    print(f"remapped {n} model object(s) -> {out}")


# ---------------------------------------------------------------- selftest
def selftest():
    """CPU-only, no downloads, no dataset, no CLIP. Exercises the GS SEG loss
    (the part this file adds) on plain seg models from yaml — one non-e2e, one
    e2e — with a class-3 GT + fabricated masks. Contracts:
      1. loss computes (box, seg, cls, dfl all finite) with a class-3 GT,
      2. cls gradients are ZERO on anchors assigned to the Unknown GT,
      3. box loss NONZERO for an Unknown-only image,
      4. GSYOLOESegModel + criterion survive a torch.save/load round-trip.
    The PE-fuse trainer path (needs CLIP weights) is pilot-verified on the
    pod, not here — recorded in the experiment doc.
    """
    assert GSYOLOESegModel.__module__ == "train_gradsuppress_yoloe", (
        f"classes would pickle as {GSYOLOESegModel.__module__}.* — checkpoints "
        "unloadable outside the training process (run via the canonical module)")
    for cfg in ("yolo11n-seg.yaml", "yolo26n-seg.yaml"):
        _selftest_one(cfg)
    print("selftest OK (both seg paths)")


from ultralytics.nn.tasks import SegmentationModel


class GSPlainSegModel(SegmentationModel):
    """Selftest stand-in: plain seg model with the GS criterion (top-level so
    the pickle contract can be exercised without CLIP downloads)."""

    def init_criterion(self):
        return (
            E2ELoss(self, GSSegmentationLoss)
            if getattr(self, "end2end", False)
            else GSSegmentationLoss(self)
        )


def _make_seg_model(cfg):
    from ultralytics.cfg import get_cfg

    m = GSPlainSegModel(cfg, nc=3, ch=3, verbose=False)
    m.args = get_cfg()
    return m


def _selftest_one(cfg):
    import copy
    import os
    import tempfile

    torch.manual_seed(0)
    model = _make_seg_model(cfg)
    model.train()

    imgsz = 64
    im = torch.rand(2, 3, imgsz, imgsz)
    preds = model(im)

    # masks at imgsz/4 (mask_ratio), overlap encoding: instance k -> value k+1
    mh = mw = imgsz // 4
    masks = torch.zeros(2, mh, mw)
    masks[0, 5:11, 5:11] = 1.0   # img 0, instance 0 (Woman)
    masks[1, 4:12, 4:12] = 1.0   # img 1, instance 0 (Unknown)
    batch = {
        "batch_idx": torch.tensor([0.0, 1.0]),
        "cls": torch.tensor([[0.0], [3.0]]),   # Woman · Unknown
        "bboxes": torch.tensor([[0.5, 0.5, 0.4, 0.4], [0.5, 0.5, 0.5, 0.5]]),
        "masks": masks,
        "sem_masks": torch.zeros(2, mh, mw),
    }

    crit = model.init_criterion()
    model.criterion = crit
    inner = crit.one2one if hasattr(crit, "one2one") else crit
    assert isinstance(inner, GSSegmentationLoss), type(inner)
    assert type(inner).loss.__qualname__.startswith("v8SegmentationLoss"), \
        "MRO wrong: seg loss branch not active"

    loss, items = crit(preds, batch)
    total = loss.sum() if loss.numel() > 1 else loss
    total.backward()
    assert torch.isfinite(total), "loss not finite"

    # contract 2: cls grads zero on unknown-assigned anchors
    model.zero_grad(set_to_none=True)
    preds2 = model(im)
    parsed = inner.parse_output(preds2)
    branch = parsed["one2one"] if isinstance(parsed, dict) and "one2one" in parsed else parsed
    scores = branch["scores"]
    scores.retain_grad()
    _, det_loss, _ = inner.get_assigned_targets_and_loss(branch, batch)
    det_loss[1].backward(retain_graph=True)  # cls only
    unk_rows = inner.last_unk_rows
    assert unk_rows is not None and unk_rows.any(), "no anchors assigned to the Unknown GT"
    g = scores.grad.permute(0, 2, 1)
    assert g[unk_rows].abs().max() == 0.0, "cls gradient leaked into Unknown-assigned anchors"
    assert g[~unk_rows].abs().max() > 0.0, "cls gradient missing everywhere else"

    # contract 3: unknown-only image still yields box loss
    batch_unk = {
        "batch_idx": torch.tensor([0.0]),
        "cls": torch.tensor([[3.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]]),
        "masks": masks[1:],
        "sem_masks": torch.zeros(1, mh, mw),
    }
    preds3 = model(im[:1])
    parsed3 = inner.parse_output(preds3)
    branch3 = parsed3["one2one"] if isinstance(parsed3, dict) and "one2one" in parsed3 else parsed3
    _, l3, _ = inner.get_assigned_targets_and_loss(branch3, batch_unk)
    assert l3[0].detach() > 0, "box loss zero for Unknown-only image"

    # contract 4: checkpoint pickle round-trip (top-level classes)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "ckpt.pt")
        torch.save({"model": copy.deepcopy(model), "epoch": 0}, p)
        torch.load(p, map_location="cpu", weights_only=False)
        gm = GSYOLOESegModel.__new__(GSYOLOESegModel)  # class importability check
        torch.save({"cls": gm.__class__}, os.path.join(td, "c.pt"))
        torch.load(os.path.join(td, "c.pt"), map_location="cpu", weights_only=False)

    print(f"  {cfg}: OK (e2e={getattr(model, 'end2end', False)} · unk anchors {int(unk_rows.sum())} "
          f"· box loss on unk-only {float(l3[0].detach()):.3f} · seg loss {float(items.get('seg_loss', items.get('seg', 0))):.3f} "
          f"· pickle ok)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="data yaml (nc=4, last class = Unknown; val = classes-0-2 labels)")
    ap.add_argument("--weights", default="yoloe-26s-seg.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=-1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--project", default="runs/gradsupp_yoloe")
    ap.add_argument("--name", default="gs")
    ap.add_argument("--freeze", type=int, default=None, help="linear-probe style; default = full fine-tune")
    ap.add_argument("--export-plain", metavar="CKPT", help="re-save a GS ckpt as stock classes (portable)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if args.export_plain:
        export_plain(args.export_plain)
        return
    if not args.data:
        ap.error("--data required (or --selftest / --export-plain)")

    from ultralytics import YOLOE

    kw = dict(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        workers=args.workers, project=args.project, name=args.name, device=0,
    )
    if args.freeze is not None:
        kw["freeze"] = args.freeze
    YOLOE(args.weights).train(trainer=GSYOLOEPESegTrainer, **kw)


if __name__ == "__main__":
    # Re-import self under the canonical module name so checkpoints pickle
    # classes as train_gradsuppress_yoloe.* (importable anywhere), NOT
    # __main__.* (loadable only inside the training process).
    import train_gradsuppress_yoloe as _canonical

    sys.exit(_canonical.main())
