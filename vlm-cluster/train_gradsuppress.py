#!/usr/bin/env python3
"""
EXP-2026-14 — cls-gradient-suppression training for unknown-person labels.

The unknown-handling problem: Spotlight drops ~12% of detections as
unknown-gender (median ~58 px — "too small to judge"). Training with those
people absent teaches the model that small people are background (EXP-2026-12:
y26n lost ~7 pts crowd recall). Grey-masking recovers recall (gemlb +6.6) but
throws away the box signal entirely. This trainer implements the third option
(coworker's design, 2026-08-05): the unknown person supervises WHERE
(box/DFL + assignment) but never WHAT (no classification gradient) —
"somebody is here, don't force a gender".

Mechanism (all against pinned ultralytics==8.4.115 source, file:line refs):

  * labels = `labels_unk3` (class 3 = Unknown). The data yaml declares nc=4;
    the MODEL head is built with nc=3 (`GSTrainer.get_model`) — class ids 0-2
    stay the production mapping and there is no phantom 4th channel at
    inference.
  * assignment: `v8DetectionLoss` feeds the assigner per-class pred scores and
    indexes them by gt label (tal.py:193), which would crash for label 3 on a
    3-channel head. `GSDetectionLoss` builds the assigner with num_classes=nc+1
    and appends a CLASS-AGNOSTIC channel (max over the 3 real channels) to the
    scores it hands the assigner — an unknown GT competes for anchors on "how
    person-ish is this anchor", never on a nonexistent class score.
  * box/DFL loss: unchanged. The assigner's target_scores carry unknown
    assignments in channel nc; `BboxLoss.forward` weights by
    `target_scores[fg_mask].sum(-1)` (loss.py:132), so unknown-assigned
    anchors get box supervision with their alignment weight, for free.
  * cls loss: BCE target = target_scores[..., :nc] (channel nc dropped), and
    every anchor row assigned to an unknown GT is zeroed out of the BCE —
    pushed neither toward a class NOR toward background. Background
    suppression for genuinely empty anchors is untouched.
  * end-to-end (YOLO26): both one2many and one2one branches get the patched
    loss via `E2ELoss(model, GSDetectionLoss)` (loss.py:1280).

All classes are TOP-LEVEL on purpose: ultralytics checkpoints pickle the model
class by reference, so `GSDetectionModel` must be importable as
`train_gradsuppress.GSDetectionModel` (a nested class crashed pilot_c at save
time). Consequence: `torch.load` of a gradsupp checkpoint needs this module
importable — run eval tools from the vlm-cluster dir (they already are), or
convert the checkpoint to a plain DetectionModel with `--export-plain`.

In-training validator caveat: pair the data yaml's val split with the
classes-0-2 `labels/` emit (NOT labels_unk3) so the 3-class head is validated
against 3-class GT. Real evaluation is the external protocol
(docs/MODEL_EVAL_PROTOCOL.md) — in-training metrics are plumbing signals.

Usage (pod):
    python3 train_gradsuppress.py --data /workspace/exp14/unk4_c.yaml \
        --weights yolo26n.pt --epochs 30 --imgsz 640 --batch -1 \
        --project /workspace/exp14/train --name y26n_gradsupp

    python3 train_gradsuppress.py --export-plain <ckpt.pt>   # portable ckpt
    python3 train_gradsuppress.py --selftest    # CPU, no downloads, no data
"""
from __future__ import annotations

import argparse
import sys

import torch
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import E2ELoss, v8DetectionLoss
from ultralytics.utils.tal import TaskAlignedAssigner, make_anchors


class GSDetectionLoss(v8DetectionLoss):
    """v8DetectionLoss with cls-gradient suppression for gt class == nc."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2=None):
        super().__init__(model, tal_topk, tal_topk2)
        # replace the assigner: one extra class slot for Unknown, whose
        # "pred score" will be the class-agnostic max channel
        self.assigner = TaskAlignedAssigner(
            topk=tal_topk,
            num_classes=self.nc + 1,
            alpha=0.5,
            beta=6.0,
            stride=self.stride.tolist(),
            topk2=tal_topk2,
        )
        self.last_unk_rows = None  # exposed for the selftest

    # copy of v8DetectionLoss.get_assigned_targets_and_loss
    # (ultralytics 8.4.115 loss.py:403-467) with the three GS changes
    # marked  # GS:
    def get_assigned_targets_and_loss(self, preds, batch):
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        # GS: append a class-agnostic channel so gt label == nc indexes it
        pd_sig = pred_scores.detach().sigmoid()
        pd_aug = torch.cat([pd_sig, pd_sig.max(-1, keepdim=True).values], dim=-1)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pd_aug,
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        # full sum (incl. unknown channel) normalizes the box loss;
        # unknown-assigned anchors keep their box weight
        target_scores_sum = max(target_scores.sum(), 1)

        # GS: cls loss on the real channels only, with unknown-assigned
        # anchor rows removed from the BCE entirely
        cls_target = target_scores[..., : self.nc]
        unk_rows = target_scores[..., self.nc] > 0  # (b, h*w)
        self.last_unk_rows = unk_rows
        bce_loss = self.bce(pred_scores, cls_target.to(dtype))  # (b, h*w, nc)
        if self.class_weights is not None:
            bce_loss *= self.class_weights
        bce_loss = bce_loss * (~unk_rows).unsqueeze(-1)
        cls_norm = max(cls_target.sum(), 1)
        loss[1] = bce_loss.sum() / cls_norm

        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            dict(zip(self.loss_names, loss.detach())),
        )


class GSDetectionModel(DetectionModel):
    """DetectionModel whose criterion is the gradient-suppression loss."""

    def init_criterion(self):
        return (
            E2ELoss(self, GSDetectionLoss)
            if getattr(self, "end2end", False)
            else GSDetectionLoss(self)
        )


class GSTrainer(DetectionTrainer):
    """DetectionTrainer whose model has nc = data_nc - 1 (last class =
    Unknown, gradient-suppressed) and the GS criterion."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        nc = self.data["nc"] - 1  # drop the Unknown slot from the head
        model = GSDetectionModel(cfg, nc=nc, ch=self.data["channels"], verbose=verbose)
        model.names = {i: self.data["names"][i] for i in range(nc)}
        if weights:
            model.load(weights)
        return model

    def set_model_attributes(self):
        # the stock hook overwrites model.nc/names from the data yaml (nc=4);
        # re-assert the 3-class head identity afterwards
        super().set_model_attributes()
        nc = self.data["nc"] - 1
        self.model.nc = nc
        self.model.names = {i: self.data["names"][i] for i in range(nc)}


def export_plain(ckpt_path: str):
    """Re-save a gradsupp checkpoint with the model class remapped to plain
    DetectionModel, so it loads anywhere without this module on sys.path."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    n = 0
    for key in ("model", "ema"):
        m = ckpt.get(key)
        if m is not None and m.__class__.__name__ == "GSDetectionModel":
            m.__class__ = DetectionModel
            if hasattr(m, "criterion"):
                del m.criterion
            n += 1
    out = ckpt_path.replace(".pt", "_plain.pt")
    torch.save(ckpt, out)
    print(f"remapped {n} model object(s) -> {out}")


# ---------------------------------------------------------------- selftest
def selftest():
    """CPU-only, no downloads, no dataset. Asserts the four contract points:
      1. loss computes with a class-3 GT (the OOB-index crash class),
      2. cls gradients are ZERO on anchors assigned to the Unknown GT,
      3. box loss is NONZERO for an Unknown-only image,
      4. the model (with criterion attached) survives a torch.save/load
         round-trip — the checkpoint-pickling failure that killed pilot_c.
    """
    assert GSDetectionModel.__module__ == "train_gradsuppress", (
        f"classes would pickle as {GSDetectionModel.__module__}.* — checkpoints "
        "unloadable outside the training process (run via the canonical module)")
    for cfg in ("yolo11n.yaml", "yolo26n.yaml"):  # plain path AND e2e path
        _selftest_one(cfg)
    print("selftest OK (both paths)")


def _selftest_one(cfg):
    import copy
    import os
    import tempfile

    from ultralytics.cfg import get_cfg

    torch.manual_seed(0)
    model = GSDetectionModel(cfg, nc=3, ch=3, verbose=False)
    model.args = get_cfg()  # default hyps (box/cls/dfl gains)
    model.train()

    imgsz = 64
    im = torch.rand(2, 3, imgsz, imgsz)
    preds = model(im)

    # image 0: one Woman (class 0); image 1: one Unknown (class 3)
    batch = {
        "batch_idx": torch.tensor([0.0, 1.0]),
        "cls": torch.tensor([[0.0], [3.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.4, 0.4], [0.5, 0.5, 0.5, 0.5]]),
    }

    crit = model.init_criterion()
    model.criterion = crit  # mimic training state for the pickle test
    inner = crit.one2one if hasattr(crit, "one2one") else crit
    assert isinstance(inner, GSDetectionLoss), type(inner)

    loss, items = crit(preds, batch)
    total = loss.sum() if loss.numel() > 1 else loss
    total.backward()
    assert torch.isfinite(total), "loss not finite"

    # contract 2: re-run the inner loss standalone to inspect unk rows + grads
    model.zero_grad(set_to_none=True)
    preds2 = model(im)
    parsed = inner.parse_output(preds2)
    branch = parsed["one2one"] if isinstance(parsed, dict) and "one2one" in parsed else parsed
    scores = branch["scores"]
    scores.retain_grad()
    _, l, _ = inner.get_assigned_targets_and_loss(branch, batch)
    l[1].backward(retain_graph=True)  # cls loss only
    unk_rows = inner.last_unk_rows
    assert unk_rows is not None and unk_rows.any(), "no anchors assigned to the Unknown GT"
    g = scores.grad.permute(0, 2, 1)  # -> (b, h*w, nc) to match unk_rows
    assert g[unk_rows].abs().max() == 0.0, "cls gradient leaked into Unknown-assigned anchors"
    assert g[~unk_rows].abs().max() > 0.0, "cls gradient missing everywhere else"

    # contract 3: Unknown-only batch still produces box loss
    batch_unk = {
        "batch_idx": torch.tensor([0.0]),
        "cls": torch.tensor([[3.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]]),
    }
    preds3 = model(im[:1])
    parsed3 = inner.parse_output(preds3)
    branch3 = parsed3["one2one"] if isinstance(parsed3, dict) and "one2one" in parsed3 else parsed3
    _, l3, _ = inner.get_assigned_targets_and_loss(branch3, batch_unk)
    assert l3[0].detach() > 0, "box loss is zero for an Unknown-only image — box supervision not flowing"

    # contract 4: checkpoint pickling round-trip (what killed pilot_c) —
    # deepcopy mimics the EMA copy ultralytics saves
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "ckpt.pt")
        torch.save({"model": copy.deepcopy(model), "epoch": 0}, p)
        back = torch.load(p, map_location="cpu", weights_only=False)
        assert back["model"].__class__ is GSDetectionModel

    print(f"  {cfg}: OK (e2e={getattr(model, 'end2end', False)} · unk anchors {int(unk_rows.sum())} "
          f"· box loss on unk-only {float(l3[0].detach()):.3f} · pickle round-trip ok)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="data yaml (nc=4, last class = Unknown; val paired with classes-0-2 labels)")
    ap.add_argument("--weights", default="yolo26n.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=-1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--project", default="runs/gradsupp")
    ap.add_argument("--name", default="gs")
    ap.add_argument("--freeze", type=int, default=None)
    ap.add_argument("--optimizer", default=None,
                    help="e.g. MuSGD / SGD / AdamW. REQUIRED for --lr0 to take effect: "
                         "with the default optimizer=auto, ultralytics overrides lr0 "
                         "and momentum and picks its own (logs 'ignoring lr0').")
    ap.add_argument("--lr0", type=float, default=None,
                    help="initial LR; only honored when --optimizer is set explicitly")
    ap.add_argument("--lrf", type=float, default=None,
                    help="final LR factor (final_lr = lr0 * lrf); ultralytics default 0.01")
    ap.add_argument("--warmup-epochs", type=float, default=None,
                    help="shorten for warm restarts from an already-trained ckpt")
    ap.add_argument("--export-plain", metavar="CKPT", help="re-save a gradsupp ckpt as plain DetectionModel")
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

    from ultralytics import YOLO

    kw = dict(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        workers=args.workers, project=args.project, name=args.name, device=0,
    )
    if args.freeze is not None:
        kw["freeze"] = args.freeze
    for k, v in (("optimizer", args.optimizer), ("lr0", args.lr0),
                 ("lrf", args.lrf), ("warmup_epochs", args.warmup_epochs)):
        if v is not None:
            kw[k] = v
    if args.lr0 is not None and args.optimizer is None:
        ap.error("--lr0 without --optimizer: optimizer=auto ignores lr0 (see --help)")
    YOLO(args.weights).train(trainer=GSTrainer, **kw)


if __name__ == "__main__":
    # Re-import self under the canonical module name so checkpoints pickle
    # classes as train_gradsuppress.* (importable anywhere), NOT __main__.*
    # (loadable only inside the training process — the trap hit 2026-08-06).
    import train_gradsuppress as _canonical

    sys.exit(_canonical.main())
