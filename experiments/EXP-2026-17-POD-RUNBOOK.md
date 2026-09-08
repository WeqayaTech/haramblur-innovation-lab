# EXP-2026-17 — pod runbook (two-axis gender × age head)

Copy-paste, one step at a time, checking the expected outcome before moving on.
Long jobs use `nohup` with a log on the volume — **not tmux** (not installed on
fresh pods).

Doc: `EXP-2026-17-two-axis-head.md`. Bars are pre-registered there; fill the
scorecard, do not move them.

**Pod spec: ≥200 GB container disk.** Budget below. A 150 GB pod does not fit
(measured 2026-08-15).

| | |
|---|---:|
| OIV7 train images | ~116–134 GB |
| OIV7 val images | ~1.2 GB |
| Two-axis labels (train) | 5.7 GB |
| Ultralytics label cache | 3–4 GB |
| Checkpoints, plots | < 1 GB |
| **Total** | **~127–145 GB** |

`df` inside a RunPod container lists the HOST's disk (`/dev/nvme0n1 1.9T`) —
that is not yours. **The only real number is the `overlay` row.** `/dev/sda2`
is the NVIDIA driver bind-mount; `/dev/shm` is RAM.

What survives a pod reset: everything under `/workspace` (code, raw labels,
verdicts, exp12 images, Phase 0 output). What is lost: pip installs and
everything on local disk. Steps 2–7 rebuild the local side in ~45 min.

---

## Step 0 — Pod setup

**Why.** Fresh RunPod templates ship transformers 5.x against torch 2.4.1, and
the trainer's loss is copied from `ultralytics==8.4.115` by line number — an
unpinned install silently diverges from the copied method bodies.

**Pick an L4, A100 or 4090.** Not a Blackwell (sm_120): its PyTorch is too old
and fails with a CUDA kernel error.

```bash
pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install "ultralytics==8.4.115"
cd /workspace/data_inspection_tools/vlm-cluster
python3 two_axis.py && python3 translation.py && python3 exp17_phase0_counts.py --selftest
python3 parallel_emit.py --selftest && python3 run_ultralytics_labels.py --selftest
python3 train_twolabel.py --selftest
```

**Expected.** Six pass lines. `train_twolabel.py` additionally prints
`stock assigner on paired GT: … [1] hot channel(s) each` — that is the
assertion that stock ultralytics cannot learn this label format, and it is why
the custom trainer exists.

## Step 1 — Name `$TREE`

**Why.** An unset `$TREE` does not fail: `$TREE/labels/train` silently resolves
to `/labels/train` and the emit writes 475k files to the filesystem root. This
happened on 2026-08-15, and it is the same class of bug CLAUDE.md records from
the production run.

```bash
export TREE=/root/exp17/tree
echo 'export TREE=/root/exp17/tree' >> ~/.bashrc
mkdir -p $TREE/images/{train,val} $TREE/labels
echo "TREE=$TREE" && df -h / | tail -1
```

**Expected.** `TREE=/root/exp17/tree`, and the `overlay` row shows ~195 GB
available. Paste `: "${TREE:?TREE is unset}"` ahead of any later block if you
open a new shell.

## Step 2 — Transform the labels, straight onto local disk

**Why.** The two-axis labels are a deterministic, $0 function of verdicts that
already live on the volume, so materializing them there would only add a second
copy of 475k files to a quota `df` cannot see. `--out` was always a free
parameter; this is a path change, not a code change. Ultralytics finds labels by
swapping `/images/` → `/labels/` in each image path, so this layout is the one
that works without symlinking.

```bash
: "${TREE:?TREE is unset}"
cd /workspace/data_inspection_tools/vlm-cluster
python3 parallel_emit.py --two-axis \
    --raw-labels /workspace/spotlight/raw/oiv7_val \
    --run        /workspace/spotlight/run/oiv7_val \
    --out        $TREE/labels/val --workers 24

nohup python3 parallel_emit.py --two-axis \
    --raw-labels /workspace/spotlight/raw/oiv7_train \
    --run        /workspace/spotlight/run/oiv7_train \
    --out        $TREE/labels/train --workers 24 \
    > /workspace/exp17/emit_train_$(hostname).log 2>&1 &
```

Val finishes in seconds and proves the invocation before the ~12-minute train
run. **Expected (measured 2026-08-15, these must reproduce exactly):**

| | train | val |
|---|---:|---:|
| images | 475,207 | 4,233 |
| kept | 1,387,261 | 8,744 |
| deleted | 192,388 | 1,200 |
| gender_unknown | 222,795 | 1,129 |
| age_unknown | 178,672 | 679 |
| child | 61,532 | 1,190 |
| adult | 1,147,057 | 6,875 |
| no_verdict · skipped | 0 · 0 | 0 · 0 |

## Step 3 — Verify the emit against the production run

**Why.** Last cheap check before GPU spend: prove these are the production
labels plus the restored unknowns, not something that drifted.

```bash
: "${TREE:?TREE is unset}"
cat $TREE/labels/train/_emit_stats.json
du -sh $TREE/labels/train && ls $TREE/labels/train | wc -l
```

**Expected.** The table above. Three identities must hold:

- `kept + deleted = 1,579,649` — every SAM3 detection accounted for.
- `child + adult + age_unknown = kept` — every person got exactly one age label.
- `gender_unknown = 222,795 = 196,119 + 26,676` — the unknown-gender adults
  Spotlight dropped, **plus** the children whose gender the 3-class emit
  discarded when it collapsed them to `Child`.

Size ~5.7 GB, 475,209 files (475,207 `.txt` + `_audit.jsonl` +
`_emit_stats.json`). If `kept` is not 1,387,261, stop.

## Step 4 — Pair check on disk

**Why.** Guarantee #2 of 3 that every person carries both axes. (#1 is the
emit's own selftest; #3 is the trainer, which raises rather than
half-training.)

```bash
: "${TREE:?TREE is unset}"
cd /workspace/data_inspection_tools/vlm-cluster && python3 - <<'EOF'
from pathlib import Path
import os, random
for split in ("train", "val"):
    d = Path(os.environ["TREE"]) / "labels" / split
    fs = list(d.glob("*.txt"))
    bad = pairs = empty = 0
    for p in random.Random(17).sample(fs, min(5000, len(fs))):
        rows = [l.split() for l in p.read_text().splitlines() if l.strip()]
        if not rows:
            empty += 1; continue
        if len(rows) % 2:
            bad += 1; continue
        for g, a in zip(rows[0::2], rows[1::2]):
            pairs += 1
            if not (0 <= int(g[0]) <= 2 and 3 <= int(a[0]) <= 5 and g[1:] == a[1:]):
                bad += 1
    print(f"{split}: files={len(fs)} sampled_pairs={pairs} empty={empty} broken={bad}")
    assert bad == 0, "broken pairs -- do NOT train on this"
EOF
```

**Expected.** `broken=0` on both splits. A nonzero `empty` is correct — those
are images where SAM3 found nobody (54,369 of them), and they train as
background.

## Step 5 — Put the audit record on the volume

**Why.** The labels live on disposable local disk. These two files are the only
record of what they were once the pod dies, and they are small.

```bash
: "${TREE:?TREE is unset}"
mkdir -p /workspace/exp17/emit_record
for T in train val; do
  cp $TREE/labels/$T/_emit_stats.json /workspace/exp17/emit_record/${T}_stats.json
  cp $TREE/labels/$T/_audit.jsonl     /workspace/exp17/emit_record/${T}_audit.jsonl
done
ls -la /workspace/exp17/emit_record/
```

## Step 6 — Stage the images to local disk

**Why.** Training reads images every epoch and the network volume is ~5× too
slow to train from (EXP-2026-12). `train_full.txt` already holds absolute paths
into `/workspace/exp12/train_tree/images/`, so it drives the copy directly.
`-n 200` batches 200 files per `cp` across 24 workers; one process per file is
dramatically slower over MooseFS.

```bash
: "${TREE:?TREE is unset}"
head -1 /workspace/exp12/val_full.txt | xargs ls -l      # pre-flight: resolves?

time xargs -a /workspace/exp12/val_full.txt   -P 24 -n 200 cp -n -t $TREE/images/val
time xargs -a /workspace/exp12/train_full.txt -P 24 -n 200 cp -n -t $TREE/images/train
```

Val first: 4,233 files in seconds, proving the invocation before the long one.
**Expected.** Train takes ~10–30 min.

## Step 7 — Verify staging by count AND bytes

**Why.** Count alone will not catch a file truncated by an interrupted copy,
and `cp -n` would skip it on a re-run, leaving a corrupt image in the training
set forever.

```bash
: "${TREE:?TREE is unset}"
ls $TREE/images/train | wc -l && ls $TREE/images/val | wc -l
du -sb $TREE/images/train /workspace/exp12/train_tree/images/train
df -h / | tail -1
```

**Expected.** 475,207 and 4,233. The two `du -sb` numbers **must match**; if
they do not, re-run the copy **without `-n`** so it overwrites rather than
skipping. Overlay should still show ≥40 GB available.

## Step 8 — Data yaml

**Why.** Declares the 6-channel taxonomy. The trainer raises if `nc != 6`, so a
typo here surfaces instantly rather than after an epoch.

```bash
: "${TREE:?TREE is unset}"
cat > /workspace/exp17/twoaxis.yaml <<EOF
path: $TREE
train: images/train
val: images/val
nc: 6
names:
  0: Woman
  1: Man
  2: GenderUnknown
  3: Adult
  4: Child
  5: AgeUnknown
EOF
cat /workspace/exp17/twoaxis.yaml
```

**Expected.** `path:` shows the expanded absolute path, not the literal
`$TREE`. Keep the yaml on the volume; the tree it points at is disposable.

Val uses two-axis labels too — the head has 6 real channels, so there is no
3-class GT to validate against. In-training mAP is not interpretable either way
(the validator treats the 6 channels as mutually exclusive); it is a plumbing
signal only, which is why Step 11 evaluates both `best.pt` and `last.pt`.

## Step 9 — One-epoch smoke

**Why.** The only test that exercises pairing **after augmentation**.
Everything before this checked the labels at rest; this checks them as the
model sees them.

```bash
: "${TREE:?TREE is unset}"
cd /workspace/data_inspection_tools/vlm-cluster
export OMP_NUM_THREADS=8        # nproc reports HOST cores inside RunPod
python3 train_twolabel.py --data /workspace/exp17/twoaxis.yaml \
    --weights yolo26n.pt --epochs 1 --imgsz 640 --batch 16 \
    --project /workspace/exp17/train --name smoke 2>&1 | tail -40
```

**Expected.** A 10–20 minute pause while ultralytics scans and caches 475k
label files — that is not a hang — then one epoch runs to completion. Ignore
the mAP numbers.

If it dies with *"do not carry exactly one 'gender'/'age' label"*, an
augmentation broke a pair. **Stop and report the traceback**; do not work
around it.

## Step 10 — The real run

**Why.** The experiment. Pre-registered GPU cap: **$25**.

```bash
: "${TREE:?TREE is unset}"
cd /workspace/data_inspection_tools/vlm-cluster
export OMP_NUM_THREADS=8
nohup python3 train_twolabel.py --data /workspace/exp17/twoaxis.yaml \
    --weights yolo26n.pt --epochs 30 --imgsz 640 --batch -1 \
    --project /workspace/exp17/train --name y26n_twoaxis \
    > /workspace/exp17/train_$(hostname).log 2>&1 &
sleep 180 && tail -20 /workspace/exp17/train_$(hostname).log
```

**Expected.** 10–26 h, ~$11. Container RAM is the pod card's number, not what
`free` shows. On a CUDA OOM, resume and patch `workers` in the checkpoint's
`train_args` with `expandable_segments`.

**Copy the weights to the volume when it finishes** — local disk dies with the
pod:

```bash
mkdir -p /workspace/exp17/weights
cp /workspace/exp17/train/y26n_twoaxis/weights/{best,last}.pt /workspace/exp17/weights/
```

## Step 11 — Dump the four datasets

**Why.** Components 1–3 on the frozen protocol, at the production operating
point, with the same matcher every model in `docs/MODEL_COMPARISON.md` used.
Evaluate **both** checkpoints: `best.pt` was selected on a fitness number that
does not mean anything for this head.

```bash
cd /workspace/data_inspection_tools/vlm-cluster
for CK in best last; do
for DS in lagenda crowd objects pass; do
  case $DS in
    lagenda) IMGS=/workspace/lagenda_eval/images ;;
    crowd)   IMGS=/workspace/exp12/crowd_sample_imgs ;;
    # object_set images live in the dataset ROOT, not an images/ subdir --
    # pointing at images/ silently processes 0 images and `set -e` misses it
    objects) IMGS=/workspace/datasets/object_set ;;
    pass)    IMGS=/workspace/datasets/pass_3k/images ;;
  esac
  python3 run_ultralytics_labels.py --map two-axis \
      --model /workspace/exp17/weights/$CK.pt \
      --images $IMGS --out /workspace/exp17/y26n_twoaxis_$CK/$DS \
      --conf 0.45 --floor 0.05 --iou 0.7 --imgsz 640 --device cuda:0
done; done
```

**Expected.** `processed=N` per dataset matching LAGENDA 4,899 · crowd 517 ·
objects 259 · PASS 3,000, with `not_processed: 0`.

## Step 12 — Score

**Why.** The decode already collapsed the two axes into the label space the
existing scorers expect, so they run unmodified. The class-name ORDER matters:
it is what keeps `conf_sweep.py`'s positional metrics meaning what they say
(`classes[0]`=Woman, `classes[:2]`=adults, `classes[2]`=Child).

```bash
CLS=Woman,Man,Child,UnknownGender
python3 run_autolabel_on_manifest.py --class-names $CLS ...   # LAGENDA, Components 2+3
python3 eval_negatives_crowd.py      --class-names $CLS ...   # crowd, objects, pass
python3 conf_sweep.py                --classes     $CLS ...   # per-axis thresholds
python3 eval_taxonomy.py --schemes two_axis_full ...          # native two-axis view
```

`two_axis_full` is the only way to see gender accuracy **on children** and age
accuracy independent of gender.

## Step 13 — Write up

Fill the scorecard in `EXP-2026-17-two-axis-head.md` against the pre-registered
bars, then update `docs/MODEL_COMPARISON.md` and `docs/COMPONENT_FRAMEWORK.md`.
Use the exact run name (`y26n_twoaxis`) and weights path in every table — no
invented aliases. Report gender **and** age coverage next to every accuracy
number. Differences of ≤ 1 point are ties: single run, no variance estimate.
