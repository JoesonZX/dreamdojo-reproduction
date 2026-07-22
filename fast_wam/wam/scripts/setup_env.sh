#!/usr/bin/env bash
# Create the python environment for the multi-view LAM on a fresh machine.
#
#   bash fast_wam/wam/scripts/setup_env.sh [VENV_DIR]
#
# Default VENV_DIR is ~/.venv-wam. Verified against python 3.10 + CUDA 12.8.
# Run scripts/smoke_test.py afterwards -- it is the acceptance test for this step.
set -euo pipefail

VENV="${1:-$HOME/.venv-wam}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"     # <root>/fast_wam/wam/scripts -> <root>

echo "== system deps =="
command -v ffmpeg  >/dev/null || { echo "ffmpeg missing: sudo apt install -y ffmpeg"; exit 1; }
command -v ffprobe >/dev/null || { echo "ffprobe missing: sudo apt install -y ffmpeg"; exit 1; }
python3 --version

echo "== venv: $VENV =="
if [[ ! -d "$VENV" ]]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip wheel

echo "== torch (cu128) =="
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128

echo "== rest =="
pip install \
  accelerate==1.13.0 numpy==2.2.6 opencv-python-headless==4.13.0.92 \
  decord==0.6.0 einops==0.8.2 safetensors==0.7.0 PyYAML==6.0.3 \
  h5py==3.16.0 pillow==12.2.0 ego4d

echo "== sanity =="
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda available:", torch.cuda.is_available(),
      "devices:", torch.cuda.device_count())
for m in ("yaml","cv2","decord","einops","safetensors","accelerate","numpy","h5py"):
    __import__(m); print("  ok", m)
PY

echo
echo "== NOTE =="
echo "The vendored AdaWorld blocks (code/adaworld/lam) hardcode .cuda(), so the"
echo "model only runs on GPU. Make sure code/adaworld exists at: $REPO_ROOT/code/adaworld"
[[ -d "$REPO_ROOT/code/adaworld/lam" ]] && echo "  found: OK" || echo "  MISSING -- transfer it (see scripts/transfer.sh)"

echo
echo "Next:  source $VENV/bin/activate && python fast_wam/wam/scripts/smoke_test.py"
