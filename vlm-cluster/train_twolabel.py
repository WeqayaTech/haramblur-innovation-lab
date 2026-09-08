#!/usr/bin/env python3
"""
EXP-2026-17 — two-axis (gender x age) multi-label detection training.

The 3-class head {Woman, Man, Child} entangles two questions in one channel, so
the model cannot say "certain this is a woman, uncertain of the age". This
trainer splits them into two independent groups of channels:

    nc = 6   channels 0-2 = GENDER {Woman, Man, GenderUnknown}
             channels 3-5 = AGE    {Adult, Child,  AgeUnknown}

The label file carries TWO rows per person with IDENTICAL geometry — one naming
the gender class, one naming the age class:

    0 0.412 0.533 0.104 0.288     # gender: Woman
    3 0.412 0.533 0.104 0.288     # age:    Adult

Why a custom loss is needed (all against pinned ultralytics==8.4.115):

  * The file format is legal and the dataset loader, augmentation, training
    loop and export all handle it untouched. `v8DetectionLoss` even uses
    BCEWithLogitsLoss per channel already, so multi-label needs no new loss
    TYPE — detection classification in YOLO is sigmoid, not softmax.
  * What does NOT work is assignment. `TaskAlignedAssigner` gives each anchor
    exactly ONE GT and a one-hot target, so under stock training no anchor is
    ever supervised on both axes — the two rows split the person's anchor
    budget instead of sharing it. Which row wins an anchor is decided by noise:
    the topk candidate sets differ because the alignment metric indexes each
    GT's own class score, and anchors both rows want are broken by
    `select_highest_overlaps` -> `overlaps.argmax(1)` (tal.py:330), which for
    two identical boxes has identical IoU and falls to the first-listed row.
    `_demo_stock_assigner_drops_an_axis` in the selftest asserts this rather
    than asserting it in prose (stock: 1 hot channel per anchor; ours: 2).

Mechanism (the three changes, marked `# 2AX:` in the loss body):

  1. Pair the duplicate-geometry rows back into one PERSON carrying a multi-hot
     label (one hot channel per group). Pairing is by exact (image, box)
     equality — augmentation applies the same transform to both rows of a pair,
     so they stay bit-identical. A broken pair is a hard error, never a silent
     half-supervised person.
  2. Assign ONCE per person, class-agnostically: the assigner is built with
     num_classes=nc+1 and every gt label is set to `nc`, which indexes an
     appended CLASS-AGNOSTIC channel (max over the real channels). Anchors
     compete on "how person-ish is this anchor", never on one axis's score —
     the same mechanism `train_gradsuppress.py:108-119` already runs in
     production for unknown-gender people.
  3. Scatter a MULTI-HOT cls target instead of a one-hot: the assigner's
     agnostic channel holds the per-anchor alignment weight, which is written
     into both of the person's hot channels. Box/DFL are untouched and keep
     weighting by that same alignment weight (loss.py:132).

Unknowns need no masking on either axis: an unreadable gender writes channel 2
and an unreadable age writes channel 5, so every person always contributes
exactly two hot channels and the box signal is never thrown away.

All classes are TOP-LEVEL on purpose: ultralytics checkpoints pickle the model
class by reference, so `TwoLabelDetectionModel` must be importable as
`train_twolabel.TwoLabelDetectionModel` (a nested class crashed EXP-2026-14's
pilot_c at save time). Consequence: `torch.load` of a two-axis checkpoint needs
this module importable — run eval tools from the vlm-cluster dir (they already
are), or convert with `--export-plain`.

In-training validator caveat: `DetectionValidator` scores the 6 channels as if
they were mutually exclusive, so training-time mAP is NOT interpretable across
the two groups and `best.pt` is selected on that fitness. Treat in-training
metrics as plumbing signals and evaluate BOTH best.pt and last.pt through the
external protocol (docs/MODEL_EVAL_PROTOCOL.md).

Usage (pod):
    python3 train_twolabel.py --data /workspace/exp17/twoaxis.yaml \
        --weights yolo26n.pt --epochs 30 --imgsz 640 --batch -1 \
        --project /workspace/exp17/train --name y26n_twoaxis

    python3 train_twolabel.py --export-plain <ckpt.pt>   # portable ckpt
    python3 train_twolabel.py --selftest    # CPU, no downloads, no data
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import E2ELoss, v8DetectionLoss
from ultralytics.utils.tal import TaskAlignedAssigner, make_anchors

# The taxonomy lives in two_axis.py so the trainer, the label emit and the
# eval decode cannot drift apart. GROUPS are half-open [start, end) channel
# ranges; every person must have exactly one hot channel inside each group.
from two_axis import (AGE_UNKNOWN, CLASS_NAMES, GENDER_UNKNOWN, GROUP_NAMES,
                      GROUPS, NC)  # noqa: E402

# An orphaned row (one axis present, the other missing) is repaired to that
# axis's Unknown rather than killing the run -- but only while it stays rare.
# Measured 2026-08-15: 2 of 475,204 OIV7 images, both scrape duplicates,
# orphaned by ultralytics' own duplicate-row removal in verify_image_label
# (two polygons reduce to one box; an odd removal orphans the partner).
# Invisible on disk -- see exp17_check_pairs.py.
#
# The guard is CUMULATIVE, deliberately. A per-batch fraction cannot separate
# "rare artifact" from "systematic breakage": one orphan among a 16-image
# batch's ~96 people is already 1%, so any small per-batch threshold trips on
# a single benign event. Across thousands of people the two are far apart --
# an artifact rate stays near 1e-5, a broken emit sits near 100%.
MAX_ORPHAN_FRAC = 0.01        # cumulative, judged after MIN_PERSONS_FOR_RATE
MIN_PERSONS_FOR_RATE = 5000   # before this, only a catastrophic batch raises
CATASTROPHIC_BATCH = 0.10     # >10% of one batch (and >3 people) = broken now


class TwoLabelDetectionLoss(v8DetectionLoss):
    """v8DetectionLoss with per-person multi-hot targets over two class groups."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2=None):
        super().__init__(model, tal_topk, tal_topk2)
        if self.nc != NC:
            raise ValueError(
                f"two-axis head expects nc={NC} ({CLASS_NAMES}), got nc={self.nc}. "
                "Check the data yaml's nc/names.")
        # replace the assigner: one extra class slot whose "pred score" is the
        # class-agnostic max channel, so assignment never depends on an axis
        self.assigner = TaskAlignedAssigner(
            topk=tal_topk,
            num_classes=self.nc + 1,
            alpha=0.5,
            beta=6.0,
            stride=self.stride.tolist(),
            topk2=tal_topk2,
        )
        # exposed for the selftest
        self.last_n_persons = None
        self.last_cls_target = None
        self.last_n_collisions = 0
        self.n_orphans_repaired = 0
        self.n_persons_seen = 0

    # 2AX: pair the duplicate-geometry rows back into one person each.
    @staticmethod
    def _within_rank(key):
        """For each element, its 0-based occurrence index among equal keys,
        in original row order. Vectorized; no python loop over rows."""
        srt = torch.argsort(key, stable=True)
        ks = key[srt]
        idx = torch.arange(ks.numel(), device=key.device)
        new_run = torch.ones_like(ks, dtype=torch.bool)
        new_run[1:] = ks[1:] != ks[:-1]
        run_start = torch.cummax(torch.where(new_run, idx, torch.zeros_like(idx)), 0).values
        rank = torch.empty_like(idx)
        rank[srt] = idx - run_start
        return rank

    def pair_targets(self, batch):
        """batch rows -> (targets, multihot).

        `targets` is (n_persons, 6+nc): [batch_idx, nc, x, y, w, h, *multihot].
        The dummy class is `nc` so the assigner indexes the appended
        class-agnostic channel. The multi-hot rides along as extra columns so
        `preprocess` pads it in lockstep with the boxes (it copies every column
        after the first and only touches cols 1:5).

        Pairing is by (image, box) AND occurrence order, not by box alone.
        Box alone is not enough: this corpus carries near-duplicate detections
        (EXP-2026-03 measured 1.1% duplicates; EXP-2026-02 found the
        cross-class NMS leak where one person is boxed under two prompts), and
        two boxes differing in the 6th decimal on disk can land on exactly
        equal float32 values after mosaic scaling and clipping. Those four rows
        would collapse into one "person" carrying two genders. Ranking within
        (box, axis) recovers them as two people instead, and it is correct
        because the emit writes each person's rows adjacently in file order,
        which augmentation preserves (it filters rows, never reorders them).
        """
        bi = batch["batch_idx"].view(-1).to(self.device)
        cls = batch["cls"].view(-1).long().to(self.device)
        box = batch["bboxes"].to(self.device)
        if bi.numel() == 0:
            return torch.zeros(0, 6 + self.nc, device=self.device), None

        key = torch.cat([bi[:, None], box], 1)
        grp = torch.unique(key, dim=0, return_inverse=True)[1]   # group id per row
        axis = (cls >= GROUPS[1][0]).long()                      # 0 = gender, 1 = age
        rank = self._within_rank(grp * 2 + axis)                 # nth person on this box
        pkey = grp * (int(rank.max()) + 1) + rank

        puniq, pinv = torch.unique(pkey, return_inverse=True)
        n_persons = puniq.numel()
        multihot = torch.zeros(n_persons, self.nc, device=self.device, dtype=box.dtype)
        multihot[pinv, cls] = 1.0
        pbox = torch.zeros(n_persons, 4, device=self.device, dtype=box.dtype)
        pbox[pinv] = box                     # rows of a person share their box
        pbi = torch.zeros(n_persons, device=self.device, dtype=box.dtype)
        pbi[pinv] = bi
        self.last_n_collisions = int(n_persons - grp.max() - 1)  # boxes shared by 2+ people

        # every person must land exactly one hot channel in each group
        bad = torch.zeros(n_persons, dtype=torch.bool, device=self.device)
        for lo, hi in GROUPS:
            bad |= multihot[:, lo:hi].sum(1) != 1
        n_bad = int(bad.sum())
        self.n_persons_seen += n_persons
        self.n_orphans_repaired += n_bad
        if n_bad:
            rate = self.n_orphans_repaired / max(self.n_persons_seen, 1)
            catastrophic = n_bad > 3 and n_bad > CATASTROPHIC_BATCH * n_persons
            systematic = (self.n_persons_seen >= MIN_PERSONS_FOR_RATE
                          and rate > MAX_ORPHAN_FRAC)
            if catastrophic or systematic:
                i = int(torch.nonzero(bad)[0])
                raise RuntimeError(
                    f"{n_bad} of {n_persons} people in this batch, and "
                    f"{self.n_orphans_repaired} of {self.n_persons_seen} "
                    f"({100 * rate:.3f}%) so far, do not carry exactly one label per "
                    f"axis -- this is systematic, not a data artifact.\n"
                    f"  first offender: multihot={multihot[i].tolist()} "
                    f"box={pbox[i].tolist()} img_in_batch={int(pbi[i])}\n"
                    f"  rows={int(cls.numel())} people={n_persons} "
                    f"shared-box people={self.last_n_collisions}\n"
                    "Run exp17_check_pairs.py --labels <dir> --data <yaml> to see "
                    "whether the break is on disk (L1) or in the label cache (L2).")
            # rare artifact: fill the missing axis with its Unknown. Honest --
            # that axis's information really was lost -- and it keeps the box
            # and the surviving axis supervising instead of dropping a person.
            for (lo, hi), unk in zip(GROUPS, (GENDER_UNKNOWN, AGE_UNKNOWN)):
                fix = bad & (multihot[:, lo:hi].sum(1) != 1)
                if fix.any():
                    multihot[fix, lo:hi] = 0.0
                    multihot[fix, unk] = 1.0

        targets = torch.cat([pbi[:, None],
                             torch.full((n_persons, 1), float(self.nc),
                                        device=self.device, dtype=box.dtype),
                             pbox,
                             multihot], 1)
        self.last_n_persons = n_persons
        return targets, multihot

    # copy of v8DetectionLoss.get_assigned_targets_and_loss
    # (ultralytics 8.4.115 loss.py:403-467) with the three changes marked # 2AX:
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

        # 2AX: one row per PERSON with a multi-hot label, instead of one row per
        # label line (which would make the two axes fight for the same anchors)
        targets, _ = self.pair_targets(batch)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        if targets.shape[-1] != 5 + self.nc:
            raise RuntimeError(
                f"preprocess returned width {targets.shape[-1]}, expected {5 + self.nc}. "
                "This loss rides the multi-hot along as extra target columns; the pinned "
                "ultralytics==8.4.115 preprocess passes them through. Re-check the pin.")
        gt_labels, gt_bboxes, gt_multihot = targets.split((1, 4, self.nc), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        # 2AX: append a class-agnostic channel so gt label == nc indexes it
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

        # target_scores holds the alignment weight in the agnostic channel only;
        # its sum is the same per-person normalizer stock uses for the box loss
        target_scores_sum = max(target_scores.sum(), 1)

        # 2AX: multi-hot cls target — the anchor's alignment weight written into
        # BOTH of its person's hot channels (one per group)
        align_w = target_scores[..., self.nc]  # (b, h*w)
        mh_sel = gt_multihot.gather(1, target_gt_idx.unsqueeze(-1).expand(-1, -1, self.nc))
        cls_target = align_w.unsqueeze(-1) * mh_sel * fg_mask.unsqueeze(-1)
        self.last_cls_target = cls_target
        bce_loss = self.bce(pred_scores, cls_target.to(dtype))  # (b, h*w, nc)
        if self.class_weights is not None:
            bce_loss *= self.class_weights
        # ~2x target_scores_sum (two hot channels), which keeps the loss scale
        # per positive channel the same as stock so the default cls gain holds
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


class TwoLabelDetectionModel(DetectionModel):
    """DetectionModel whose criterion is the two-axis multi-label loss."""

    def init_criterion(self):
        return (
            E2ELoss(self, TwoLabelDetectionLoss)
            if getattr(self, "end2end", False)
            else TwoLabelDetectionLoss(self)
        )


class TwoLabelTrainer(DetectionTrainer):
    """DetectionTrainer that builds a TwoLabelDetectionModel.

    Unlike GSTrainer there is no nc-1 trick: every one of the 6 channels is a
    real predicted class, so the head width matches the data yaml's nc and the
    stock `set_model_attributes` already sets the right nc/names.
    """

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = TwoLabelDetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose)
        if weights:
            model.load(weights)
        return model


def resolve_resume(ap, args):
    """Return the checkpoint the YOLO object must be BUILT from to resume.

    `Model.train()` rewrites `resume=True` into "the path this YOLO object was
    loaded from", and if THAT file carries no epoch/optimizer state it drops
    the resume and starts a fresh run behind one WARNING line
    (ultralytics 8.4.115 engine/model.py:805-814). Passing `--weights
    yolo26n.pt --resume` -- this module's own defaults -- lands exactly there:
    the released weights have epoch=-1 and no optimizer, so a resume meant to
    continue a 50-epoch run would instead silently throw it away and retrain
    from scratch into a new `<name>2` directory. That happened to EXP-2026-17
    the first time and cost nothing only because it was caught before launch.

    So resolve the checkpoint HERE, prove it is resumable, and hand it back as
    the model to build from.
    """
    ckpt = Path(args.resume if isinstance(args.resume, str)
                else Path(args.project) / args.name / "weights" / "last.pt")
    if not ckpt.is_file():
        ap.error(f"--resume: no checkpoint at {ckpt}")
    c = torch.load(ckpt, map_location="cpu", weights_only=False)
    if c.get("epoch", -1) < 0 or c.get("optimizer") is None:
        ap.error(
            f"--resume: {ckpt} has epoch={c.get('epoch')} and "
            f"optimizer={'present' if c.get('optimizer') else 'missing'} -- that is "
            "not a resumable training checkpoint. Resuming from it would silently "
            "start a NEW run instead of continuing one.")
    total = (c.get("train_args") or {}).get("epochs", args.epochs)
    print(f"resuming {ckpt}: {c['epoch'] + 1} of {total} epochs already done")
    return str(ckpt)


def export_plain(ckpt_path: str):
    """Re-save a two-axis checkpoint with the model class remapped to plain
    DetectionModel, so it loads anywhere without this module on sys.path."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    n = 0
    for key in ("model", "ema"):
        m = ckpt.get(key)
        if m is not None and m.__class__.__name__ == "TwoLabelDetectionModel":
            m.__class__ = DetectionModel
            if hasattr(m, "criterion"):
                del m.criterion
            n += 1
    out = ckpt_path.replace(".pt", "_plain.pt")
    torch.save(ckpt, out)
    print(f"remapped {n} model object(s) -> {out}")


# ---------------------------------------------------------------- selftest
def selftest():
    """CPU-only, no downloads, no dataset. Asserts the contract points:
      1. the loss computes on paired two-row GT,
      2. two rows with identical geometry collapse to exactly ONE person,
      3. every foreground anchor's cls target has exactly two hot channels,
         one per group,
      4. cls gradient reaches BOTH groups on the SAME anchor (the thing stock
         assignment cannot do),
      5. a broken pair raises instead of training a half-supervised person,
      6. a person that is GenderUnknown AND AgeUnknown still produces box loss,
      7. the model survives a torch.save/load round-trip (the pickling failure
         that killed EXP-2026-14's pilot_c),
      8. and the premise itself: the STOCK assigner gives these same paired
         labels only ONE hot channel per anchor.
    """
    assert TwoLabelDetectionModel.__module__ == "train_twolabel", (
        f"classes would pickle as {TwoLabelDetectionModel.__module__}.* — checkpoints "
        "unloadable outside the training process (run via the canonical module)")
    _demo_stock_assigner_drops_an_axis()
    _selftest_resume_guard()
    for cfg in ("yolo11n.yaml", "yolo26n.yaml"):  # plain path AND e2e path
        _selftest_one(cfg)
    print("selftest OK (both paths)")


def _selftest_resume_guard():
    """Contract 9: a non-resumable checkpoint must be REJECTED.

    Ultralytics turns that case into a fresh training run behind one warning
    line, which is indistinguishable from a successful resume until you notice
    the epoch counter restarted -- see resolve_resume.
    """
    import contextlib
    import io
    import os
    import shutil
    import tempfile

    p = argparse.ArgumentParser()
    with tempfile.TemporaryDirectory() as td:
        bad = os.path.join(td, "pretrained.pt")
        torch.save({"epoch": -1, "optimizer": None}, bad)   # the shape of yolo26n.pt
        args = argparse.Namespace(resume=bad, project=td, name="run", epochs=80)
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                resolve_resume(p, args)
            except SystemExit:
                pass
            else:
                raise AssertionError(
                    "a checkpoint with no epoch/optimizer state must be rejected, "
                    "or --resume silently retrains from scratch")

        good = os.path.join(td, "last.pt")
        torch.save({"epoch": 49, "optimizer": {"state": {}},
                    "train_args": {"epochs": 80}}, good)
        args.resume = good
        assert resolve_resume(p, args) == good

        # bare --resume resolves <project>/<name>/weights/last.pt
        os.makedirs(os.path.join(td, "run", "weights"))
        shutil.copy(good, os.path.join(td, "run", "weights", "last.pt"))
        args.resume = True
        assert resolve_resume(p, args).endswith(os.path.join("run", "weights", "last.pt"))

        args.resume = os.path.join(td, "nope.pt")
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                resolve_resume(p, args)
            except SystemExit:
                pass
            else:
                raise AssertionError("a missing checkpoint path must be rejected")
    print("  resume guard: non-resumable and missing checkpoints rejected, "
          "bare --resume resolves last.pt")


def _demo_stock_assigner_drops_an_axis():
    """Assert the premise this whole module rests on.

    Runs the STOCK assigner on one person written as two identical-geometry
    rows and checks that every foreground anchor comes back with exactly ONE
    hot channel — i.e. no anchor is ever supervised on both axes, so the two
    rows split the person's anchor budget rather than sharing it. Our loss
    gives the same anchors TWO hot channels.

    This is an assertion rather than a comment so that if a future ultralytics
    ever assigns multi-label targets natively, the selftest says so loudly and
    the patch can be retired instead of quietly duplicating upstream.
    """
    torch.manual_seed(0)
    anc = torch.stack(torch.meshgrid(torch.arange(8) + 0.5, torch.arange(8) + 0.5,
                                     indexing="ij"), -1).view(-1, 2).float()
    n = anc.shape[0]
    box = torch.tensor([[1.0, 1.0, 7.0, 7.0]])            # xyxy
    gt_bboxes = box.expand(2, 4).unsqueeze(0).clone()     # (1, 2, 4) — identical
    gt_labels = torch.tensor([[[0.0], [3.0]]])            # gender row, age row
    mask_gt = torch.ones(1, 2, 1)
    pd_bboxes = box.expand(n, 4).unsqueeze(0).clone()
    pd_scores = torch.rand(1, n, NC)

    a = TaskAlignedAssigner(topk=10, num_classes=NC, alpha=0.5, beta=6.0, stride=[1])
    _, _, target_scores, fg_mask, _ = a(pd_scores, pd_bboxes, anc, gt_labels, gt_bboxes, mask_gt)
    fg = fg_mask.bool()
    assert fg.any(), "synthetic case assigned nothing — the demo itself is broken"
    hot = (target_scores > 0).sum(-1)[fg]
    assert torch.all(hot == 1), (
        f"stock assigner produced {sorted(set(hot.tolist()))} hot channels per anchor; "
        "if this is now 2, ultralytics assigns multi-label targets natively and "
        "train_twolabel.py is redundant")
    print(f"  stock assigner on paired GT: {int(fg.sum())} fg anchors, "
          f"{sorted(set(hot.tolist()))} hot channel(s) each — one axis unsupervised per anchor")


def _paired_batch(rows):
    """rows = [(img_idx, gender_cls, age_cls, (x, y, w, h)), ...] -> a batch
    with the two-rows-per-person layout the label files use."""
    bi, cls, box = [], [], []
    for img, g, a, b in rows:
        for c in (g, a):
            bi.append(float(img))
            cls.append([float(c)])
            box.append(list(b))
    return {"batch_idx": torch.tensor(bi),
            "cls": torch.tensor(cls),
            "bboxes": torch.tensor(box)}


def _selftest_one(cfg):
    import copy
    import os
    import tempfile

    from ultralytics.cfg import get_cfg

    torch.manual_seed(0)
    model = TwoLabelDetectionModel(cfg, nc=NC, ch=3, verbose=False)
    model.args = get_cfg()  # default hyps (box/cls/dfl gains)
    model.train()

    imgsz = 64
    im = torch.rand(2, 3, imgsz, imgsz)
    preds = model(im)

    # image 0: an adult woman; image 1: a child of unreadable gender
    batch = _paired_batch([
        (0, 0, 3, (0.5, 0.5, 0.4, 0.4)),   # Woman  + Adult
        (1, 2, 4, (0.5, 0.5, 0.5, 0.5)),   # GenderUnknown + Child
    ])

    crit = model.init_criterion()
    model.criterion = crit  # mimic training state for the pickle test
    inner = crit.one2one if hasattr(crit, "one2one") else crit
    assert isinstance(inner, TwoLabelDetectionLoss), type(inner)

    loss, items = crit(preds, batch)
    total = loss.sum() if loss.numel() > 1 else loss
    total.backward()
    assert torch.isfinite(total), "loss not finite"

    # contract 2+3+4: re-run the inner loss standalone to inspect targets/grads
    model.zero_grad(set_to_none=True)
    preds2 = model(im)
    parsed = inner.parse_output(preds2)
    branch = parsed["one2one"] if isinstance(parsed, dict) and "one2one" in parsed else parsed
    scores = branch["scores"]
    scores.retain_grad()
    _, l, _ = inner.get_assigned_targets_and_loss(branch, batch)

    n_persons = inner.last_n_persons
    assert n_persons == 2, f"4 label rows should pair into 2 people, got {n_persons}"

    ct = inner.last_cls_target                      # (b, h*w, nc)
    fg = ct.sum(-1) > 0
    assert fg.any(), "no foreground anchors assigned"
    hot = (ct > 0).sum(-1)[fg]
    assert torch.all(hot == 2), (
        f"foreground anchors must carry exactly 2 hot channels, saw {sorted(set(hot.tolist()))}")
    for lo, hi in GROUPS:
        per_group = (ct[..., lo:hi] > 0).sum(-1)[fg]
        assert torch.all(per_group == 1), (
            f"channels {lo}..{hi - 1} must have exactly one hot per anchor")

    l[1].backward(retain_graph=True)  # cls loss only
    g = scores.grad.permute(0, 2, 1)  # -> (b, h*w, nc) to match fg
    for (lo, hi), gname in zip(GROUPS, GROUP_NAMES):
        assert g[fg][:, lo:hi].abs().max() > 0.0, (
            f"no cls gradient reached the '{gname}' group — the axes are not both supervised")

    # contract 5b: two DIFFERENT people whose boxes coincide (near-duplicate
    # detections that augmentation rounds to the same float32 box) must resolve
    # to TWO people, not one person carrying two genders -- the failure the
    # 1-epoch smoke hit on 2026-08-15
    collide = _paired_batch([
        (0, 0, 3, (0.5, 0.5, 0.4, 0.4)),   # Woman + Adult
        (0, 1, 4, (0.5, 0.5, 0.4, 0.4)),   # Man   + Child, identical box
    ])
    tgt, mh = inner.pair_targets(collide)
    assert inner.last_n_persons == 2, (
        f"two people sharing a box must stay two people, got {inner.last_n_persons}")
    assert inner.last_n_collisions == 1, inner.last_n_collisions
    for lo, hi in GROUPS:
        assert torch.all(mh[:, lo:hi].sum(1) == 1), mh
    # and the pairing is the file order, not an arbitrary cross-product
    assert sorted(tuple(r.nonzero().flatten().tolist()) for r in mh) == [(0, 3), (1, 4)], mh

    # contract 5c: a RARE orphan (one axis lost to ultralytics' duplicate-row
    # removal) is repaired to Unknown, not fatal -- the failure mode that would
    # otherwise kill a 26 h run at hour 20 over 2 junk images in 475,204
    many = [(0, i % 2, 3 + (i % 2), (0.1 + 0.005 * i, 0.5, 0.02, 0.02))
            for i in range(120)]
    b = _paired_batch(many)
    b["batch_idx"] = torch.cat([b["batch_idx"], torch.tensor([0.0])])
    b["cls"] = torch.cat([b["cls"], torch.tensor([[0.0]])])          # lone gender row
    b["bboxes"] = torch.cat([b["bboxes"], torch.tensor([[0.9, 0.9, 0.02, 0.02]])])
    before = inner.n_orphans_repaired
    _, mh = inner.pair_targets(b)
    assert inner.n_orphans_repaired == before + 1, inner.n_orphans_repaired
    # the exact shape that broke smoke3: ONE orphan among ~96 people is 1.04%,
    # which any small per-batch fraction would reject. It must be repaired.
    assert inner.n_persons_seen >= 121
    for lo, hi in GROUPS:
        assert torch.all(mh[:, lo:hi].sum(1) == 1), "repair left an invalid person"
    orphan = mh[-1] if mh[-1, 0] > 0 else mh[mh[:, AGE_UNKNOWN] > 0][0]
    assert orphan[AGE_UNKNOWN] == 1, "the lost axis must become its Unknown"

    # contract 5: SYSTEMATIC breakage is still a hard error. A lone orphan is
    # repaired (contract 5c) because at batch scale one artifact is already ~1%
    # of a batch's people -- so the guard has to be cumulative, and the thing
    # that must still kill the run is a batch that is mostly orphans.
    broken = {"batch_idx": torch.zeros(10),
              "cls": torch.zeros(10, 1),                       # 10 gender rows, no ages
              "bboxes": torch.tensor([[0.1 + 0.05 * i, 0.5, 0.02, 0.02] for i in range(10)])}
    try:
        inner.pair_targets(broken)
    except RuntimeError as e:
        assert "systematic" in str(e), e
    else:
        raise AssertionError("an all-orphan batch should have raised")

    # contract 6: a doubly-unknown person still supervises localization
    batch_unk = _paired_batch([(0, 2, 5, (0.5, 0.5, 0.5, 0.5))])
    preds3 = model(im[:1])
    parsed3 = inner.parse_output(preds3)
    branch3 = parsed3["one2one"] if isinstance(parsed3, dict) and "one2one" in parsed3 else parsed3
    _, l3, _ = inner.get_assigned_targets_and_loss(branch3, batch_unk)
    assert l3[0].detach() > 0, "box loss is zero for a doubly-unknown person — box supervision lost"
    ct3 = inner.last_cls_target
    fg3 = ct3.sum(-1) > 0
    assert torch.all((ct3[..., 2] > 0)[fg3]) and torch.all((ct3[..., 5] > 0)[fg3]), \
        "doubly-unknown person should be hot on GenderUnknown and AgeUnknown"

    # contract 7: checkpoint pickling round-trip — deepcopy mimics the EMA copy
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "ckpt.pt")
        torch.save({"model": copy.deepcopy(model), "epoch": 0}, p)
        back = torch.load(p, map_location="cpu", weights_only=False)
        assert back["model"].__class__ is TwoLabelDetectionModel

    print(f"  {cfg}: OK (e2e={getattr(model, 'end2end', False)} · persons {n_persons} "
          f"· fg anchors {int(fg.sum())} · box loss on unknown-only {float(l3[0].detach()):.3f} "
          f"· pickle round-trip ok)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help=f"data yaml (nc=6, names = {list(CLASS_NAMES)})")
    ap.add_argument("--weights", default="yolo26n.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=-1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--project", default="runs/twoaxis")
    ap.add_argument("--name", default="twoaxis")
    ap.add_argument("--freeze", type=int, default=None)
    ap.add_argument("--save-period", type=int, default=None,
                    help="checkpoint every N epochs. Cheap insurance on a long "
                         "run: local disk dies with the pod, and best.pt is "
                         "selected on a fitness number that is not meaningful "
                         "for a two-axis head")
    ap.add_argument("--resume", nargs="?", const=True, default=None, metavar="CKPT",
                    help="resume after an interruption or an OOM. Bare --resume "
                         "uses <project>/<name>/weights/last.pt; pass a path to "
                         "name the checkpoint explicitly. Either way the run is "
                         "REBUILT from that checkpoint -- see resolve_resume")
    ap.add_argument("--fraction", type=float, default=None,
                    help="train on this fraction of the dataset -- for a fast "
                         "smoke that still reaches the validation pass "
                         "(0.03 ~ 3 min vs ~95 min for a full epoch at batch 16)")
    ap.add_argument("--export-plain", metavar="CKPT", help="re-save a two-axis ckpt as plain DetectionModel")
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
    if args.fraction is not None:
        kw["fraction"] = args.fraction
    if args.save_period is not None:
        kw["save_period"] = args.save_period
    weights = args.weights
    if args.resume:
        # the YOLO object must BE the checkpoint, not the pretrained weights
        weights, kw["resume"] = resolve_resume(ap, args), True
    YOLO(weights).train(trainer=TwoLabelTrainer, **kw)


if __name__ == "__main__":
    # Re-import self under the canonical module name so checkpoints pickle
    # classes as train_twolabel.* (importable anywhere), NOT __main__.*
    # (loadable only inside the training process — the trap hit 2026-08-06).
    import train_twolabel as _canonical

    sys.exit(_canonical.main())
