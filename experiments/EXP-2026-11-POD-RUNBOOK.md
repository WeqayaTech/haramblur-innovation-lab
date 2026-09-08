# EXP-2026-11 — pod runbook (flicker baseline)

Copy-paste phases; paste each phase's output back before moving on.
GPU need: small — ~10k frames of single-image inference, ~10-20 min on an L4.
Everything after Phase 3 is CPU.

**Layout note (owner decision 2026-08-03): this experiment is fully self-contained under
`/workspace/innovation-lab/exp11_flicker/`** — code, videos, sidecars, and reports all live
there. The shared `/workspace/data_inspection_tools/vlm-cluster/` pod copy and
`/workspace/datasets/` are deliberately NOT touched. The probe's two imports
(`run_model_children.py`, `dataset_utils.py`) are shipped into the folder so it runs
standalone.

```
/workspace/innovation-lab/exp11_flicker/
├── code/       the 4 synced .py files
├── videos/     the 4 clips + SHA256SUMS + README.md
├── probe/      raw per-frame sidecars (<stem>.jsonl + .meta.json)  ← the asset
└── report_conf35|45|55/   metrics + annotated mp4s (0.45 = production threshold)
```

## Phase 0 — pod setup

Pick an **L4 or A100** (not Blackwell). Then:

```bash
pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install opencv-python-headless
mkdir -p /workspace/innovation-lab/exp11_flicker/{code,videos}
nproc; python3 -c "import torch, cv2; print(torch.cuda.is_available(), cv2.__version__)"
```

Expect `True` and a cv2 version. (The torch trio is the standard fresh-pod SAM3-era fix;
harmless if already current.)

## Phase 1 — sync the code (self-contained copy)

From the **local** machine:

```bash
scp vlm-cluster/video_flicker_probe.py vlm-cluster/flicker_metrics.py \
    vlm-cluster/run_model_children.py vlm-cluster/dataset_utils.py \
    <pod>:/workspace/innovation-lab/exp11_flicker/code/
```

Then on the pod, prove the plumbing before any GPU time:

```bash
cd /workspace/innovation-lab/exp11_flicker/code
python3 video_flicker_probe.py --selftest && python3 flicker_metrics.py --selftest
```

Both must print `selftest OK`.

## Phase 2 — stage the videos

~0.6 GB total — glance at the RunPod console volume quota first (df lies).

```bash
cd /workspace/innovation-lab/exp11_flicker/videos

wget -O interview.webm "https://upload.wikimedia.org/wikipedia/commons/9/9b/Voices_from_Wikimania_Wikimedian_Netha_Hussain.webm"
wget -O pedestrian_area.webm "https://upload.wikimedia.org/wikipedia/commons/a/ae/Video_Codec_Test_pedestrian_area_1080p25.y4m.webm"
wget -O street_crossing.webm "https://upload.wikimedia.org/wikipedia/commons/d/d2/People_waiting_to_cross_the_street.webm"
wget -O dubai_souk.webm "https://upload.wikimedia.org/wikipedia/commons/transcoded/8/85/Dubai_4K_Naif_Deira_%26_Gold_Souk_Day_Walking_Tour_2026_Old_Dubai.webm/Dubai_4K_Naif_Deira_%26_Gold_Souk_Day_Walking_Tour_2026_Old_Dubai.webm.480p.vp9.webm"

sha256sum *.webm | tee SHA256SUMS
cat > README.md <<'EOF'
# exp11_flicker/videos — EXP-2026-11 temporal-stability probe clips
4 Wikimedia Commons clips, downloaded 2026-08-03. See SHA256SUMS.
- interview.webm      — Voices from Wikimania, Wikimedian Netha Hussain (CC BY 4.0), 2:09 1080p, talking head
- pedestrian_area.webm — Video Codec Test pedestrian area (CC0), 15s 1080p25, crowd walking
- street_crossing.webm — People waiting to cross the street (CC BY-SA 3.0), 15s 720p
- dubai_souk.webm     — Dubai Naif Deira & Gold Souk walking tour, 480p transcode (CC BY 4.0),
                        37 min; only the first 180 s are probed. Gulf-dress slice.
Source pages: search each filename on commons.wikimedia.org.
EOF
```

Expected sizes: interview ~105 MB, pedestrian ~11 MB, crossing ~30 MB, dubai ~448 MB.

## Phase 3 — the probe run (GPU)

`--max-seconds 180` only caps `dubai_souk` (the others are shorter). Floor conf 0.05 is
the *logging* floor, deliberate — production threshold is applied later, offline.

```bash
cd /workspace/innovation-lab/exp11_flicker/code
LOG=/workspace/innovation-lab/exp11_flicker/probe_$(hostname).log
nohup python3 -u video_flicker_probe.py \
    --videos /workspace/innovation-lab/exp11_flicker/videos \
    --run-dir /workspace/YOLO-MIT/runs/v9MIT/gelansfav14_datav2_v4 \
    --repo /workspace/YOLO-MIT --imgsz 640 --max-seconds 180 \
    --iou-nms 0.7 --max-dets 70 \
    --out /workspace/innovation-lab/exp11_flicker/probe > "$LOG" 2>&1 &
echo "log: $LOG"; sleep 60; tail -5 "$LOG"
```

Resumable — if it dies, rerun the same command. Done when `probe_summary.json` exists:

```bash
tail -3 /workspace/innovation-lab/exp11_flicker/probe_*.log
ls -la /workspace/innovation-lab/exp11_flicker/probe/
```

Paste back the tail + file listing (4 `.jsonl` + 4 `.meta.json` expected).

## Phase 4 — metrics + renders (CPU, pod)

Main pass at **conf 0.45 — the confirmed production scoreThreshold** (extension
`src/constants.mjs` MODELS config, read 2026-08-03), plus two sensitivity passes:

```bash
cd /workspace/innovation-lab/exp11_flicker/code
E=/workspace/innovation-lab/exp11_flicker
python3 flicker_metrics.py --in $E/probe --conf 0.45 --render \
    --videos $E/videos --out $E/report_conf45
python3 flicker_metrics.py --in $E/probe --conf 0.35 --out $E/report_conf35
python3 flicker_metrics.py --in $E/probe --conf 0.55 --out $E/report_conf55
```

Paste back the full stdout of the conf 0.45 pass (the per-video summaries are the
baseline numbers).

## Phase 5 — watch the renders (local)

```bash
bash mount.sh   # then open ./mnt/runpod/innovation-lab/exp11_flicker/report_conf45/*_annotated.mp4
```

Magenta box = Woman (pixelated for a male viewer), blue = Man, orange = Child, thin gray
= sub-threshold detection. What to look for: blur popping on/off on a steady person
(existence flicker), gray boxes during pop-offs (threshold gaps — dual-threshold would fix
those), Woman/Man labels alternating on men in kanduras in `dubai_souk` (the bias as
flicker), momentary blurs on background objects (phantom blips).

Then: fold numbers into `EXP-2026-11-video-flicker-baseline.md` §results, and only after
that pre-register bars for the first smoothing-policy replay experiment.
