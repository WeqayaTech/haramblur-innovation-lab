#!/usr/bin/env bash
# One-shot bring-up for a fresh RTX PRO Blackwell (sm_120) RunPod pod.
# Run once per new pod:  bash setup.sh
set -e

echo "[setup] swapping to CUDA 12.8 (Blackwell-ready) torch ..."
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

echo "[setup] stage-1 (describe) deps ..."
pip install "transformers>=4.49" qwen-vl-utils accelerate opencv-python PyYAML Pillow

echo "[setup] stage-2 (cluster/report) deps ..."
pip install sentence-transformers scikit-learn hdbscan umap-learn matplotlib

python -c "import torch; print('[setup] torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'cap', torch.cuda.get_device_capability())"

echo "[setup] done. Run:  export HF_HOME=/workspace/hf-cache"
