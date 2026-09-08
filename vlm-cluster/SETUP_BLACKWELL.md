# Fresh-pod bring-up (RTX PRO Blackwell / sm_120)

RunPod pods are ephemeral — every new pod resets the environment. The default
PyTorch on these images is built for CUDA ≤12.4 (max compute capability sm_90),
but the **RTX PRO 4000 Blackwell is sm_120**, so you get:

    RuntimeError: CUDA error: no kernel image is available for execution on the device

Run these steps once per new pod.

## 1. Swap to a Blackwell-ready PyTorch (CUDA 12.8)
```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```
`torchaudio` is removed on purpose — the stale cu124 build throws
`libtorchaudio.so: undefined symbol` against the new torch, and we don't use audio.

Verify (must print capability `(12, 0)` and `True`):
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability())"
nvidia-smi   # confirm 2x Blackwell, driver >= 570
```

## 2. Stage-1 (describe) deps — usually already on the image
```bash
pip install transformers>=4.49 qwen-vl-utils accelerate opencv-python PyYAML Pillow
```

## 3. Stage-2 (cluster / field_report) deps — installs without touching torch
```bash
pip install sentence-transformers scikit-learn hdbscan umap-learn matplotlib
```

## 4. Point the HF cache at the network volume (avoid re-downloading 16GB)
```bash
export HF_HOME=/workspace/hf-cache
```
Add it to `~/.bashrc` if you want it to stick for the session's shells.

## 5. Smoke test (20 crops) before any big run
```bash
python describe.py --yaml /workspace/open-images-v7/dataset.yaml \
  --split train --max-crops 20 --out ./smoke
python3 -c "import json,pprint; pprint.pprint(json.loads(open('smoke/descriptions.jsonl').readline())['object'])"
```

## 6. Full 2-GPU describe run (one process per GPU, disjoint shards)
```bash
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python describe.py --yaml /workspace/open-images-v7/dataset.yaml \
  --split train --num-shards 2 --shard-id 0 --max-crops 750 --out ./run > s0.log 2>&1 &
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 python describe.py --yaml /workspace/open-images-v7/dataset.yaml \
  --split train --num-shards 2 --shard-id 1 --max-crops 750 --out ./run > s1.log 2>&1 &
wait
```
Progress: `wc -l run/descriptions.shard*.jsonl` (the reliable meter) or `tail -f s0.log`.

## 7. Reports (CPU only — keep off the busy GPUs)
```bash
CUDA_VISIBLE_DEVICES="" python field_report.py --in ./run --out ./run/field_report
CUDA_VISIBLE_DEVICES="" python cluster.py     --in ./run --out ./run/clusters
```

## Notes
- Long runs die if the SSH session closes (`&` jobs get SIGHUP). Use `tmux` /
  `nohup` for overnight runs.
- `describe.py` is resumable — re-run the same command to continue; raise
  `--max-crops` to go deeper into the same shuffled list.
- To make this permanent, bake steps 1–3 into a custom RunPod template image.
