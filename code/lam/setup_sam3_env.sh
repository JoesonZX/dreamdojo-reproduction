#!/usr/bin/env bash
# Setup script for SAM3 conda environment.
# Run in a tmux session (takes ~10-15 min):
#   bash /home/xuan/embodied-ai/code/lam/setup_sam3_env.sh
set -e

echo "=== Step 1: Create conda env sam3 (Python 3.12) ==="
conda create -y -n sam3 python=3.12

echo "=== Step 2: Activate + install PyTorch 2.10 (CUDA 12.8) ==="
# Run sub-commands in conda env using 'conda run'
conda run -n sam3 pip install torch==2.10.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

echo "=== Step 3: Clone SAM3 ==="
git clone https://huggingface.co/facebook/sam3 /home/xuan/embodied-ai/code/sam3 2>/dev/null || \
git clone https://github.com/facebookresearch/sam3 /home/xuan/embodied-ai/code/sam3

echo "=== Step 4: Install SAM3 ==="
conda run -n sam3 pip install -e /home/xuan/embodied-ai/code/sam3

echo "=== Step 5: Install visualization deps ==="
conda run -n sam3 pip install opencv-python-headless h5py tqdm

echo "=== Done! ==="
echo "Test with:"
echo "  conda activate sam3"
echo "  python -c 'import sam3; print(sam3.__version__)'"
