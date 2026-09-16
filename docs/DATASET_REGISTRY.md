# Dataset registry — what lives on the RunPod volume and how

**In one line:** every reusable benchmark dataset on the volume gets one folder under
`/workspace/datasets/<name>/`, one `README.md` manifest inside it, and one row in the inventory
table below. This repo file is the source of truth (the volume isn't in git).

## Why a filesystem convention and not a database

The question came up ("can I store it like a database?"). Deliberate answer: **no database.**

- **Images must stay as plain files.** Every consumer we have — `autolabel_sam.py`,
  `describe.py`, the eval scripts, any future training run — reads image files from a directory.
  Putting images in a DB (or even tar shards) would mean unpacking them right back out.
- **Annotations are small and already script-friendly.** All our annotation data together is a
  few hundred MB of JSON/JSONL/odgt against multi-GB image dirs. Our tools consume these formats
  directly; a DB adds a server, a schema, and a failure mode, and removes nothing.
- **The benefits people want from "a database" are actually these,** and we get them with files:
  - *knowing what exists* → the inventory table below;
  - *provenance / re-downloadability* → per-dataset `README.md` with source URL, date, checksums;
  - *integrity* → `sha256` of the original archives recorded at staging time;
  - *not re-doing expensive work* → hand-verification artifacts (e.g. `rejects.txt`) kept inside
    the dataset folder, never in an `expNN/` folder that might get cleaned up.
- **If cross-dataset queries ever become a real need** (e.g. "all persons with gender labels
  across LAGENDA + WIDER"), the right tool is DuckDB/pandas reading the JSONL manifests in place
  — still no server, no migration.

## The per-dataset layout

```
/workspace/datasets/<name>/          # no "negative" in the name — the labeler skips such paths
├── README.md          # manifest: source URL(s), download date, archive sha256s, counts,
│                      #   license/terms, sampling seeds, known quirks, staged-by
├── images/            # the images, original filenames, flat or original structure
├── annotations/       # original annotation files, byte-for-byte as downloaded
└── (derived/)         # anything we computed (samples, verified subsets) — never edits originals
```

Rules (carried over from the EXP-2026-03 convention, now formalized):

1. **Reusable data under `/workspace/datasets/`, experiment outputs under `/workspace/expNN/`.**
   A dataset folder is append-only; experiment runs never write into it.
2. **Originals are immutable.** Sampling/filtering produces new files under `derived/` (or a new
   sibling folder like `Images_sample500/`) with the seed recorded in the README.
3. **Record the archive sha256 before deleting the archive.** Volume space is finite; archives
   get deleted after extraction, so the checksum + source URL in the README is what makes the
   dataset re-creatable and disputes resolvable.
4. **Hand-made artifacts are the most expensive thing we own** (object_set's verification,
   future Gulf-dress labels). They live inside the dataset folder and get backed up to this repo
   when small (e.g. rejects lists, label CSVs).
5. **Never put the string "negative" in a folder name** — `autolabel_sam.py` silently skips it.

## Inventory (update this table whenever the volume changes)

| Dataset | Volume path | What it is | Components served | Source / how re-created | Status |
|---|---|---|---|---|---|
| LAGENDA sample | `/workspace/lagenda_eval/` (legacy path — don't move, scripts reference it) | 5,000 people, human-checked age+gender | 1 (prominent-subject recall), 2, 3 | LAGENDA public release; sampling in EXP-2026-01 appendix | staged, heavily used |
| CrowdHuman val | `/workspace/datasets/crowdhuman/` (`Images/` 4,370 + `Images_sample500/` seed-51 + `annotation_val.odgt`) | exhaustively-boxed crowds | 1 | HF `sshao0516/CrowdHuman`; sample seed 51 | staged |
| PASS 3k | `/workspace/datasets/pass_3k/` | 3,000 verified person-free scenes | 1 (negatives) | Zenodo 6615455, PASS.0.tar, seed-51 sample of 3,000 | staged |
| Object set | `/workspace/datasets/object_set/` **(images in the ROOT, no `images/` subdir — breaks the convention; a runner pointed at `.../object_set/images` silently processes 0 files)** | 259 hand-verified doll/statue/toy images | 1 (negatives) | Open Images candidates via `make_object_negatives.py` + human rejects pass — **expensive to rebuild, never delete** | staged |
| WIDER Attribute | `/workspace/datasets/wider_attribute/` | 13,789 full-scene images, 57,524 person boxes, 14 binary attributes incl. gender (`male`); 30 scene categories; 5,509 train / 1,362 val / 6,918 test | 1 (multi-person scenes with real boxes), 3 (gender attribute) — **no age labels**, so NOT a Component-2 set | annotations: CUHK server (alive, verified 2026-07-15); images: official Google Drive (see staging block below) | **staging in progress** |
| Small-person benchmark | `/workspace/datasets/smallperson_v1/` (`crowd_small/` 800 imgs + `synth_shrunk/{h96,h64,h48,h32,h24}/` 1,500 imgs + `paste_grey/` 250 imgs / 1,500 people + `avatars/` 240 imgs / 1,336 profile pictures) | people ≤96 px: 16,314 real (CrowdHuman val, human boxes, Component 1 only) + 1,500 synthetically shrunk LAGENDA images carrying human gender/age (Components 2-3) | 1, 2, 3 | `vlm-cluster/build_small_person_set.py`; seed 20260820; `Images_sample500` excluded | **built + scored 2026-08-20**; crowd arm 12/12, synth + paste + avatar arms 7/7 each; all three arms scored for `yolo11N-640` / `y26n_gradsupp` / `y26n_noe2e_warm50-2` (see EXP-2026-19) |
| LAGENDA fl1199 (3-way set) | `/workspace/datasets/lagenda_full/fl1199/` (`images/` symlinks, `raw/*.txt` + `*.json` sidecars, `manifest.jsonl`, `SOL_HANDOFF.md`); human GT in `../eval_v2/` | **The only set with HUMAN age+gender where EVERY person is labelled** — 1,199 LAGENDA images / 1,514 human-labelled people, selected 3 independent ways. Supports a true **Gemini vs Sol vs hand-labelled** comparison; elsewhere in LAGENDA 4 of 6 people are unlabelled so FPs are unmeasurable. Fresh SAM3 run 2026-08-23: 1,884 detections **with sidecars** (29.1% multi-part masks — the older `sam_autolabel` labels have no `parts`). **Gemini not yet run on it** (only 3.3%, and with a different prompt). **Contains ZERO people <64px at model input** and is 6x sparser than the rest of LAGENDA — prominent-subject only. | 2 (age), 3 (gender) with human GT; **not** 1 for crowd/small-person | `build_lagenda_full.py` v2 → fully-labelled filter (`_holdout_review/lagenda/`) → `autolabel_sam_raw_allext.py` | staged + SAM3-labelled 2026-08-23 |
| HaramBlur holdout | `/workspace/datasets/haramblur_holdout/` (`images/<collection>/` raw + `clean/<collection>.txt` path lists + `_archives/` with SHA256SUMS) | **HOLDOUT TEST SET — NEVER TRAIN ON IT** (carries a `DO_NOT_TRAIN` marker). 14,531 scraped images in 6 query collections: women 3,589 · shiekhs 3,739 · child 1,556 · men 1,776 · randoms 2,850 (person-free, query `baseball bat`) · women_hd 1,021 (Pexels). 12,736 pass the clean filter (decodable, non-UI-asset, min side ≥160px, exact-dupe) and **11,494 survive near-duplicate removal — `clean/FINAL_unique.txt` is the labeling input** (mirror-augmented PDQ ≤31 OR SSCD ≥0.80, leader-clustered; thresholds calibrated on this data, see the folder README). **No per-image ground truth** — folder names are query labels, not verified labels; needs adjudication or a verifier pass before any accuracy number. `shiekhs` is the project's first Gulf/traditional-dress slice. | 1 (negatives via `randoms`), 3 (Gulf-dress probe via `shiekhs`) — pending annotation | Google Drive folder `1ipx4bawh-nHi-S9Z47dH9ScP0EVe6HYD` (owner info@weqayatech.com), downloaded 2026-08-18 via `gdown`; archive sha256s recorded before extraction; `census.py` + `build_clean.py` regenerate every number | staged 2026-08-18; **fully Spotlight-labeled 2026-08-18** — all 11,494 unique images through SAM3 + Gemini 3.5 Flash-Lite (Batch API): 21,004 detections, 15.4% deleted, 607 relabeled, **$6.78**. Two variants: `labeling/full/labels_std/` (classes 0-2) and **`labels_unk3/` (class 3 = Unknown — USE THIS for eval, treat class 3 as an ignore region)**. Full run record, findings and traces: `labeling/FULL_RUN.md`; pilot: `labeling/README.md`. Key results: Woman→Man relabels run **2.48:1** over Man→Woman (independently reproducing Spotlight OIV7's unattributed 2.5:1), and the **Gulf-dress bias is quantified for the first time — SAM3 misreads 4.94% of shaykh detections as Woman, while the Gemini verifier calls 2,661/2,689 ghutra wearers `man` and 0 `woman`**. Machine labels, NOT human-verified; the 254 `randoms` gate-survivors need adjudication before any FP rate is quoted. **Verified complete 2026-08-20** (`labeling/tools/verify_eval_ready.py`, 10/10 checks; report at `EVAL_VERIFICATION.html`). Paths + benchmark commands: `docs/HOLDOUT_BENCHMARK_HANDOFF.md`. **Bug found: `autolabel_sam_raw.py:174` silently skips `.webp`** (patched copy used here). |
| Flicker videos | `/workspace/innovation-lab/exp11_flicker/videos/` (exception to the `/workspace/datasets/` convention — EXP-2026-11 is self-contained under `/workspace/innovation-lab/` by owner decision 2026-08-03) | 4 Wikimedia Commons video clips (talking-head interview, 2 street/crowd scenes, Dubai souk walking tour = Gulf-dress slice) for temporal-stability probing | none of the 3 — measures a new axis (temporal stability, EXP-2026-11); no GT, self-supervised via track consistency | direct Commons URLs in `EXP-2026-11-POD-RUNBOOK.md` Phase 2 (verified 2026-08-03); sha256s in the folder's SHA256SUMS | **pending staging** (runbook ready) |
| not_person hard negatives | `/workspace/datasets/notperson_hardneg_oiv7/` (`images/` + `labels/` symlinks only, ~25MB; ready-made `train_exp{12,14}_oversample{5,10}x.txt` lists) | OIV7 train images where Spotlight verified a SAM3 detection `not_person` (toy/statue/doll/painting) but Gemini still committed a gender cue — the training signal for the toy/statue false-positive bug. 12,814 gendered `not_person` detections / **9,254 unique images** (9,966 man-cued, 2,848 woman-cued); 6,504 pure background, 2,750 mixed with a real labeled person. Val: 82 dets / 62 images (too small to be its own eval arm) | 1 (negatives, targeted) | Mined from `/workspace/spotlight/run/oiv7_train/verdicts_batch.jsonl` (already-run Spotlight production labeling — no new API calls); reproducible via `notperson_gendered_oiv7_{train,val}.jsonl` on the volume | built 2026-08-31; **strategy = oversample existing paths in the train list, not a new label class** (unlike `labels_unk3`'s Unknown class, these are non-people — "no box here" is already the correct signal, just diluted at 1.9% of the corpus); untested — needs an ablation run against `object_set` before adoption |

## Staging block — WIDER Attribute (run on the pod)

We want the **full scenes + original JSON boxes** (CUHK source), NOT OpenPAR's Google Drive
mirror, which is pre-cropped per person (`split_image/`) and would destroy the Component-1 use
and the box-conditioning for Component 3. The OpenPAR mirror
(`https://drive.google.com/drive/folders/1f0CH-H5V_Ej-rJu_pRQJJiktdbJLv58M`) is fallback-only,
and only for attribute work on crops.

```bash
mkdir -p /workspace/datasets/wider_attribute/{images,annotations,archives}
cd /workspace/datasets/wider_attribute/archives

# 1. annotations — direct from the CUHK server (1.6 MB, verified alive 2026-07-15)
wget "http://mmlab.ie.cuhk.edu.hk/projects/WIDERAttribute_files/wider_attribute_annotation.zip"

# 2. images — official Google Drive from the project page (~4 GB)
pip install -q gdown
gdown "https://drive.google.com/uc?id=0B-PXtfvNMLanWEVCaHZnR0RHSlE" -O wider_attribute_image.zip
# if Drive quota blocks it, retry later or use a cookies-based gdown; LAST resort is the
# OpenPAR mirror (pre-cropped — record in the README that full scenes are then missing)

# 3. checksums BEFORE anything else (goes into README.md)
sha256sum *.zip                                        # → paste back

# 4. extract (unzip isn't on the pod image — use Python)
cd /workspace/datasets/wider_attribute
python3 - <<'EOF'
import zipfile
zipfile.ZipFile('archives/wider_attribute_annotation.zip').extractall('annotations')
zipfile.ZipFile('archives/wider_attribute_image.zip').extractall('images')
EOF

# 5. verify counts against the paper's numbers
find images -name '*.jpg' | wc -l                      # expect 13,789   → paste back
ls annotations/                                        # expect trainval + test JSONs → paste back
python3 - <<'EOF'
import json, glob
total = 0
for f in glob.glob('annotations/**/*.json', recursive=True):
    d = json.load(open(f))
    n = sum(len(img.get('targets', [])) for img in d.get('images', []))
    print(f, len(d.get('images', [])), 'images,', n, 'person boxes')
    total += n
print('total boxes:', total, '(expect 57,524 across trainval+test)')
EOF
# → paste back

# 6. free the archive space ONLY after checksums are recorded and counts verified
# rm -rf archives/

# 7. write the manifest
cat > README.md <<'EOF'
# WIDER Attribute (full scenes + original JSON)
- Source page: http://mmlab.ie.cuhk.edu.hk/projects/WIDERAttribute.html
- Annotations: wider_attribute_annotation.zip from the CUHK server (sha256: FILL_IN)
- Images: official Google Drive id 0B-PXtfvNMLanWEVCaHZnR0RHSlE (sha256: FILL_IN)
- Downloaded: 2026-07-15 · Staged by: Mostafa
- 13,789 full-scene images / 57,524 person boxes / 14 binary attributes (incl. male);
  splits 5,509 train / 1,362 val / 6,918 test; 30 scene categories.
- Attribute encoding: 1 = positive, -1 = negative, 0 = unspecified.
- NOT pre-cropped (we deliberately avoided the OpenPAR split_image mirror).
- No age labels — Component 1 & 3 use only.
- License/terms: public, non-commercial research use.
EOF
```

## For developers — using and testing the datasets

Two access paths, in order of usefulness:

1. **GPU work: attach the network volume to your pod** (any pod in the volume's datacenter, via
   the RunPod console — SSH over exposed TCP for VS Code Remote-SSH, or the Web Terminal; pick
   an L4/A100, not Blackwell, or apply `vlm-cluster/SETUP_BLACKWELL.md`). Everything is at
   `/workspace/datasets/<name>/` plus LAGENDA at `/workspace/lagenda_eval/`. There is no local
   mount tool in this repo — all volume access is over SSH on the pod.
2. **No access needed at all for tool development:** the scorers self-test synthetically —
   `python3 eval_negatives_crowd.py --selftest`, `run_autolabel_on_manifest.py --selftest`,
   `translation.py` — so a developer can modify eval code and test it with zero data.

**First command on any fresh pod** (CPU-only, seconds):

```bash
python3 vlm-cluster/verify_datasets.py                 # on the pod (root /workspace)
```

It checks every staged dataset against the expected counts in this registry and prints
PASS / FAIL / MISSING per dataset. If it fails, fix the volume (or this registry) before running
anything that depends on the data. When you stage a new dataset, add a check block to
`verify_datasets.py` in the same commit as its inventory row here.

**Reproduce-something-real smoke test** (~1 min, CPU): re-score an existing experiment's frozen
outputs and diff against the committed numbers —

```bash
cd /workspace/data_inspection_tools/vlm-cluster
python3 eval_negatives_crowd.py --mode negatives \
    --images /workspace/datasets/pass_3k \
    --pred-labels /workspace/exp04/sam_labels/pass \
    --out /tmp/smoke_pass --overlays 0 --class-names Person
# expect: fp_per_100_images == 27.37, pct 12.7% — the EXP-2026-04 PASS row
```

**Etiquette:** dataset folders are append-only (experiment outputs go to `/workspace/expNN/`);
never rename dataset folders (scripts and runbooks reference the paths, and adding "negative"
to a name silently breaks the labeler); the hand-verified sets (object_set, future Gulf-dress
labels) are the most expensive artifacts on the volume — treat them as irreplaceable.

## Adding the next dataset

Copy the staging pattern: `mkdir` the four-part layout → download into `archives/` → `sha256sum`
→ extract → verify counts against the source's published numbers → README.md → only then delete
archives → add the inventory row here.
