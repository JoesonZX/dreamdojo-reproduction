#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   CUDA_VISIBLE_DEVICES=0,1 bash run_ours.sh a
#   CUDA_VISIBLE_DEVICES=0,1 bash run_ours.sh b
#   CUDA_VISIBLE_DEVICES=0,1 bash run_ours.sh b2
#   CUDA_VISIBLE_DEVICES=0,1 bash run_ours.sh c
#   CUDA_VISIBLE_DEVICES=0,1 bash run_ours.sh c-fg
#   CUDA_VISIBLE_DEVICES=0,1 bash run_ours.sh all

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_COUNT="${GPU_COUNT:-2}"
PYTHON_BIN="${PYTHON_BIN:-/home/xuan/.venv/bin/python}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

run_one() {
  local name="$1"
  local cfg="$ROOT/code/config/$2"
  echo "[run] $name -> $cfg"
  "$PYTHON_BIN" -m accelerate.commands.launch --num_processes "$GPU_COUNT" --mixed_precision bf16 \
    "$ROOT/code/train.py" --config "$cfg"
}

case "${1:-all}" in
  a)
    run_one "ours-a-zero" "ft_l40_ours_a_zero_5k.yaml"
    ;;
  b)
    run_one "ours-b-zero-reverse" "ft_l40_ours_b_zero_reverse_5k.yaml"
    ;;
  b2)
    run_one "ours-b2-zero-action-reverse" "ft_l40_ours_b2_zero_action_reverse_5k.yaml"
    ;;
  c)
    run_one "ours-c-zero-reverse-hardneg" "ft_l40_ours_c_zero_reverse_hardneg_5k.yaml"
    ;;
  c-fg|cfg)
    run_one "ours-c-zero-reverse-hardneg-fg" "ft_l40_ours_c_zero_reverse_hardneg_fg_5k.yaml"
    ;;
  all)
    run_one "ours-a-zero" "ft_l40_ours_a_zero_5k.yaml"
    run_one "ours-b-zero-reverse" "ft_l40_ours_b_zero_reverse_5k.yaml"
    run_one "ours-c-zero-reverse-hardneg" "ft_l40_ours_c_zero_reverse_hardneg_5k.yaml"
    ;;
  *)
    echo "Usage: CUDA_VISIBLE_DEVICES=0,1 bash run_ours.sh {a|b|b2|c|c-fg|all}" >&2
    exit 2
    ;;
esac
