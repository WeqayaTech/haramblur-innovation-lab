# EXP-2026-03 — Pod runbook (you run, Claude reads outputs)

Same pattern as EXP-2026-02: each phase is copy-paste blocks for the pod terminal; blocks marked
**→ paste back** produce output Claude needs before the next phase. Pod: **L4 or A100** (not
Blackwell/sm_120).

**Volume layout (convention as of this experiment):** reusable benchmark data lives under
**`/workspace/datasets/<name>/`** — these outlive any one experiment (CrowdHuman, the PASS
sample, and the hand-verified object set will be reused by the MiVOLO and replacement-pipeline
work). Experiment-specific outputs (SAM label runs, evals) live under **`/workspace/exp03/`**.
(LAGENDA stays at its legacy `/workspace/lagenda_eval/` path — existing scripts reference it.)

**Folder-name rule for this whole experiment:** the production labeler skips any path containing
the string `"negative"` — every folder below deliberately avoids that word. Don't rename anything
to `negatives/`.

---

## Phase 0 — sync scripts + self-tests (CPU, minutes)

From your **local** machine:

```bash
scp -P <POD_PORT> \
    ~/Desktop/HaramBlur/Innovation-lab/vlm-cluster/eval_negatives_crowd.py \
    ~/Desktop/HaramBlur/Innovation-lab/vlm-cluster/make_object_negatives.py \
    root@<POD_IP>:/workspace/data_inspection_tools/vlm-cluster/
```

On the pod:

```bash
mkdir -p /workspace/datasets/{pass_3k,object_set,crowdhuman}
mkdir -p /workspace/exp03/{sam_labels/{pass,objects,crowd},eval}
cd /workspace/data_inspection_tools/vlm-cluster
python3 eval_negatives_crowd.py --selftest          # → paste back if it fails
ls /workspace/autolabel/autolabel_sam.py            # production labeler still there?
```

---

## Phase 1 — datasets

### 1a. CrowdHuman val (~4,370 images + annotation_val.odgt)

Hosted on Hugging Face by the dataset's author (`sshao0516/CrowdHuman`). List the files first —
exact archive names can change:

```bash
pip install -U "huggingface_hub[cli]"
export HF_HOME=/workspace/.cache/huggingface   # reuse the cached token (avoids anon rate limits)
# list the repo's actual files first — this hf CLI version wants EXPLICIT filenames,
# not --include patterns (patterns get treated as literal names and 404)
python3 - <<'EOF'
from huggingface_hub import list_repo_files
for f in list_repo_files('sshao0516/CrowdHuman', repo_type='dataset'):
    print(f)
EOF
# → paste back, then download the val zip + val odgt by their exact names, e.g.:
hf download sshao0516/CrowdHuman CrowdHuman_val.zip annotation_val.odgt \
    --repo-type dataset --local-dir /workspace/datasets/ch_dl
```

Then (adapt names to what the listing showed):

```bash
cd /workspace/datasets/crowdhuman
# `unzip` isn't on the pod image — use Python's zipfile
python3 -c "import zipfile; zipfile.ZipFile('/workspace/datasets/ch_dl/CrowdHuman_val.zip').extractall('.')"
cp /workspace/datasets/ch_dl/annotation_val.odgt .
find . -name '*.jpg' | head -3                              # → paste back (confirms the image dir)
echo "jpgs: $(find . -name '*.jpg' | wc -l)  gt lines: $(wc -l < annotation_val.odgt)"  # → paste back
```

Point `--images` at wherever the `find` shows the jpgs landed (usually `Images/`; may be flat).

(If the HF repo is gated/unavailable, fallback: the official form at www.crowdhuman.org emails
Google Drive links — same files.)

### 1b. PASS subset (3,000 person-free images)

PASS (~1.4M images) is hosted as ten `PASS.N.tar` files (~18.7 GB each) on **Zenodo record
6615455** (verified 2026-07-13; the VGG page just links here). We download ONE tar, seed-51
sample 3,000, and delete it. Needs ~20 GB free (`df -h /workspace`).

```bash
cd /workspace/datasets
wget -q --show-progress \
  "https://zenodo.org/records/6615455/files/PASS.0.tar?download=1" -O PASS.0.tar

# seed-51 random sample of 3,000 — reads the tar index and extracts ONLY those 3,000
python3 - <<'EOF'
import tarfile, random
from pathlib import Path
dst = Path('/workspace/datasets/pass_3k'); dst.mkdir(exist_ok=True)
tf = tarfile.open('/workspace/datasets/PASS.0.tar')
members = [m for m in tf.getmembers()
           if m.isfile() and m.name.lower().endswith(('.jpg', '.jpeg', '.png'))]
random.Random(51).shuffle(members)
for m in members[:3000]:
    m.name = Path(m.name).name           # flatten any subdirs
    tf.extract(m, dst)
tf.close()
print('extracted', len(list(dst.iterdir())), 'PASS images to', dst)
EOF

rm -f PASS.0.tar    # free the 18.7 GB
```

### 1c. Toy/mannequin/statue candidates (Open Images → hand-verified)

```bash
pip install fiftyone
cd /workspace/data_inspection_tools/vlm-cluster
# does a Mannequin-like class exist? (don't assume)
python3 make_object_negatives.py --list-classes mannequin     # → paste back
python3 make_object_negatives.py --list-classes statue        # → paste back

python3 make_object_negatives.py --download \
    --out /workspace/datasets/object_set \
    --classes Doll "Teddy bear" Sculpture \
    --per-class 100                       # add Mannequin/Statue if the listing found them
python3 make_object_negatives.py --make-sheet --out /workspace/datasets/object_set
```

**Hand-verification (the human step, ~30–45 min):** download `/workspace/datasets/object_set/`
to your machine (VS Code Remote: right-click → Download), open `contact_sheet.html` in a browser,
and apply the pre-registered ruling — **REJECT any image containing a real person OR a
photo/poster/printed depiction of one** (those count as people; SAM firing there would be correct).
Write rejected filenames one-per-line into `rejects.txt`, upload it back, then:

```bash
python3 make_object_negatives.py --apply-rejects rejects.txt --out /workspace/datasets/object_set
# → paste back the final per-class counts (they go in the experiment doc's Appendix B)
```

---

## Phase 2 — Stage A: the frozen production labeler, once per dataset (GPU)

Settings identical to EXP-2026-02 (conf 0.4, nms_iou 0.7 — the script's defaults). Time at the
L4's measured ~1.08 it/s: objects ~3 min · PASS ~46 min · CrowdHuman ~68 min (its images are
larger/denser, may run slower).

```bash
export HF_HOME=/workspace/.cache/huggingface     # SAM3 weights + token cached here
cd /workspace/autolabel

python3 autolabel_sam.py --input /workspace/datasets/object_set \
    --output /workspace/exp03/sam_labels/objects \
    --classes "woman" "man" "child" --batch_size 1

python3 autolabel_sam.py --input /workspace/datasets/pass_3k \
    --output /workspace/exp03/sam_labels/pass \
    --classes "woman" "man" "child" --batch_size 1

python3 autolabel_sam.py --input /workspace/datasets/crowdhuman/Images \
    --output /workspace/exp03/sam_labels/crowd \
    --classes "woman" "man" "child" --batch_size 1
# → paste back each run's final tqdm line
```

(Each run is resumable — rerun the same command to continue; don't pass `--overwrite`.)

## Phase 3 — Stage B: score each dataset separately (CPU, minutes)

```bash
cd /workspace/data_inspection_tools/vlm-cluster

python3 eval_negatives_crowd.py --mode negatives \
    --images /workspace/datasets/object_set \
    --pred-labels /workspace/exp03/sam_labels/objects \
    --out /workspace/exp03/eval/objects --overlays 60
cat /workspace/exp03/eval/objects/summary.json          # → paste back

python3 eval_negatives_crowd.py --mode negatives \
    --images /workspace/datasets/pass_3k \
    --pred-labels /workspace/exp03/sam_labels/pass \
    --out /workspace/exp03/eval/pass --overlays 60
cat /workspace/exp03/eval/pass/summary.json             # → paste back

python3 eval_negatives_crowd.py --mode crowd \
    --images /workspace/datasets/crowdhuman/Images \
    --gt-odgt /workspace/datasets/crowdhuman/annotation_val.odgt \
    --pred-labels /workspace/exp03/sam_labels/crowd \
    --out /workspace/exp03/eval/crowd --overlays 60
cat /workspace/exp03/eval/crowd/summary.json            # → paste back
```

## Phase 4 — visual proof + the poster hand-check

Download the three `eval/*/overlays/` folders (small annotated JPEGs):
- `eval/objects/overlays/` — SAM boxing mannequins/dolls as people: the money shots.
- `eval/pass/overlays/` — what (if anything) it hallucinates in empty scenes.
- `eval/crowd/overlays/` — duplicates (purple) and clear FPs (red) against GT (green).

**CrowdHuman poster check (per the §2.2 ruling):** eyeball the red clear-FP boxes and tally
three buckets — real unlabeled person / poster-statue-of-a-person (correct per our ruling!) /
genuine hallucination. → paste back the three counts; they turn CrowdHuman precision from a
lower bound into an honest estimate.

## Phase 5 — write-up

Claude fills the per-dataset results into `EXP-2026-03-sam-negatives-and-crowds.md` (§2.6),
scores each against its pre-registered bar, and generates charts.
