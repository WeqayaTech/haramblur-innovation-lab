# EXP-2026-14 — pod runbook (unknown-handling arms on YOLO26n)

Copy-paste, phased. Training needs a big pod (**A100 or RTX 4090, 200 GB container disk** —
the NVMe staging recipe from EXP-2026-12 is mandatory: the network volume is 5× too slow to
train from). Eval phases run on the same pod. Bars are pre-registered in the experiment doc;
nothing here may change them. House rules: nohup + hostname-stamped logs on the volume, OMP
pinning, empty-label-file semantics, count-don't-estimate.

## Phase 0 — setup + sync + selftest

```bash
cd /workspace/data_inspection_tools/vlm-cluster
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
pip install "ultralytics==8.4.115"
```

Sync `train_gradsuppress.py` from local (new file — the pod copy does not have it), then:

```bash
python3 train_gradsuppress.py --selftest    # expect: both paths OK
python3 run_ultralytics_labels.py --selftest
```

Verify inputs:

```bash
ls -d /workspace/spotlight/run/oiv7_train/labels_unk3 /workspace/spotlight/run/oiv7_val/labels_unk3
ls -d /workspace/spotlight/run/oiv7_val/labels
ls /workspace/exp12/train_full.txt /workspace/exp12/val_full.txt /workspace/exp12/spotlight_oiv7.yaml
wc -l /workspace/spotlight/run/oiv7_train/labels_unk3/* 2>/dev/null | tail -1   # sanity: files exist
```

⚠ also confirm the val `labels_unk3` emit actually ran (CLAUDE.md open item f) — count files:
`ls /workspace/spotlight/run/oiv7_val/labels_unk3 | wc -l` (expect ≈ 4,233). If missing,
re-emit first (parallel_emit.py, ~minutes for val).

## Phase 1 — stage images to NVMe + build the two trees

Stage exactly as EXP-2026-12 Phase B did (parallel cp of the train/val image trees to local
NVMe, ~8 min for 116 GB). Then build both label trees (dir symlinks — images shared, labels
differ):

```bash
mkdir -p /workspace/exp14
for ARM in b c; do
  mkdir -p /local_nvme/exp14/tree_$ARM/images /local_nvme/exp14/tree_$ARM/labels
  ln -sfn /local_nvme/oiv7/images/train /local_nvme/exp14/tree_$ARM/images/train
  ln -sfn /local_nvme/oiv7/images/val   /local_nvme/exp14/tree_$ARM/images/val
  ln -sfn /workspace/spotlight/run/oiv7_train/labels_unk3 /local_nvme/exp14/tree_$ARM/labels/train
done
ln -sfn /workspace/spotlight/run/oiv7_val/labels_unk3 /local_nvme/exp14/tree_b/labels/val
ln -sfn /workspace/spotlight/run/oiv7_val/labels      /local_nvme/exp14/tree_c/labels/val
```

(⚠ adjust `/local_nvme/oiv7` to wherever the staging recipe actually put the images; label
symlinks point at the volume — labels are small, only images need NVMe.)

Two yamls on the volume (`/workspace/exp14/unk4_b.yaml`, `/workspace/exp14/unk4_c.yaml`),
both `nc: 4`, `names: {0: Woman, 1: Man, 2: Child, 3: Unknown}`, `path:` at the respective
tree, `train: images/train`, `val: images/val`.

## Phase 2 — pilots FIRST (~1-2 h, plumbing only, never scoreboard numbers)

50k-subset yaml (same recipe as exp12's `spotlight_oiv7_50k.yaml`), 3 epochs each:

```bash
HOST=$(hostname)
# Arm B pilot — stock 4-class
nohup python3 -u -c "
from ultralytics import YOLO
YOLO('yolo26n.pt').train(data='/workspace/exp14/unk4_b_50k.yaml',
    epochs=3, imgsz=640, batch=-1, device=0, workers=8,
    project='/workspace/exp14/train', name='pilot_b')
" > /workspace/exp14/log_pilot_b_$HOST.log 2>&1 &
wait
# Arm C pilot — gradient suppression
cd /workspace/data_inspection_tools/vlm-cluster
nohup python3 -u train_gradsuppress.py --data /workspace/exp14/unk4_c_50k.yaml \
    --weights yolo26n.pt --epochs 3 --imgsz 640 --batch -1 \
    --project /workspace/exp14/train --name pilot_c \
    > /workspace/exp14/log_pilot_c_$HOST.log 2>&1 &
```

Pilot pass criteria (before spending on full runs): both train without error; arm B ckpt
`YOLO(best).names` has 4 names, arm C ckpt has **3**; arm C's COCO warm-start transferred
(non-random boxes on a few LAGENDA images); loss curves decrease. Watch first-epoch img/s —
project cost against the $60 cap; if it blows, stop and report.

## Phase 3 — full trainings (30 epochs each, sequential)

Same two commands with the full yamls, `epochs=30`, names `y26n_unk4` / `y26n_gradsupp`.
RunPod traps: container RAM limit is the pod card's number (not `free`); on CUDA OOM
resume with the workers patch + expandable_segments (see EXP-2026-12 doc, ops traps).

## Phase 4 — dump + score (the standard protocol, all four datasets + val mAP)

Per arm, the EXP-2026-12 Phase A/C command pattern with
`--model /workspace/exp14/train/<name>/weights/best.pt --map identity`, out
`/workspace/exp14/<arm>/<dataset>`. **Arm B extra step:** strip class-3 lines from the
emitted label files before scoring (sidecars keep everything):

```bash
find /workspace/exp14/y26n_unk4 -path '*/labels/*.txt' -exec sed -i '/^3 /d' {} +
```

Then the three `eval_negatives_crowd.py` runs + `run_autolabel_on_manifest.py` + the OIV7-val
dump + `map_eval.py` per arm — identical commands to the EXP-2026-13 runbook Phase A scoring,
paths swapped. Score, then apply the pre-registered bars (experiment doc) and the adoption
rule. Append both arms to the tables in `EXP-2026-12-TEAM-BRIEF.md`.

## Phase 5 — wrap-up

Fill the experiment doc per arm; scoreboard rows in `docs/COMPONENT_FRAMEWORK.md`
(Component 1 crowd rows especially); flicker replay (EXP-2026-11 clips) for the winning arm
only; the unknown-handling recommendation goes to the coworker for the gelan line.
