#!/usr/bin/env bash
# Ship the code (NOT the data, NOT the checkpoints) to the training machine.
#
#   bash fast_wam/wam/scripts/transfer.sh user@a6000-host [REMOTE_ROOT]
#
# Sends two directories:
#   fast_wam/wam    -- this project
#   code/adaworld  -- the vendored transformer blocks model.py imports read-only
#
# Pull results back with the --pull flag (only model.safetensors; the optimizer
# state is several GB and is not needed off-machine).
set -euo pipefail

REMOTE="${1:?usage: transfer.sh user@host [REMOTE_ROOT] [--pull]}"
REMOTE_ROOT="${2:-/home/xuan/embodied-ai}"
MODE="${3:-push}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

RSYNC_OPTS=(-avz --progress
  --exclude '__pycache__' --exclude '*.pyc' --exclude '.git'
  --exclude 'results/' --exclude 'checkpoints/')

if [[ "$MODE" == "--pull" ]]; then
  echo "== pulling checkpoints (model weights only) =="
  mkdir -p "$LOCAL_ROOT/fast_wam/checkpoints" "$LOCAL_ROOT/fast_wam/results"
  rsync -avz --progress \
    --include '*/' --include 'model.safetensors' --include 'config.yaml' \
    --exclude '*' \
    "$REMOTE:$REMOTE_ROOT/fast_wam/checkpoints/" "$LOCAL_ROOT/fast_wam/checkpoints/"
  rsync -avz --progress "$REMOTE:$REMOTE_ROOT/fast_wam/results/" "$LOCAL_ROOT/fast_wam/results/"
  echo "done."
  exit 0
fi

echo "== pushing code to $REMOTE:$REMOTE_ROOT =="
ssh "$REMOTE" "mkdir -p $REMOTE_ROOT/fast_wam $REMOTE_ROOT/code"
rsync "${RSYNC_OPTS[@]}" "$LOCAL_ROOT/fast_wam/wam"   "$REMOTE:$REMOTE_ROOT/fast_wam/"
rsync "${RSYNC_OPTS[@]}" "$LOCAL_ROOT/code/adaworld" "$REMOTE:$REMOTE_ROOT/code/"

echo
echo "== remote paths in the configs =="
if [[ "$REMOTE_ROOT" != "/home/xuan/embodied-ai" ]]; then
  echo "REMOTE_ROOT differs from the paths baked into config/*.yaml."
  echo "Run this on the remote to fix them in one shot:"
  echo "  sed -i 's|/home/xuan/embodied-ai|$REMOTE_ROOT|g' $REMOTE_ROOT/fast_wam/wam/config/*.yaml"
else
  echo "paths match, nothing to rewrite."
fi

echo
echo "Next on the remote:"
echo "  cd $REMOTE_ROOT"
echo "  bash fast_wam/wam/scripts/setup_env.sh"
echo "  source ~/.venv-wam/bin/activate"
echo "  python fast_wam/wam/scripts/smoke_test.py"
