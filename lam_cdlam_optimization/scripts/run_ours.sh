#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   CUDA_VISIBLE_DEVICES=0,1 bash run_ours.sh a
#   CUDA_VISIBLE_DEVICES=0,1 bash run_ours.sh b
#   CUDA_VISIBLE_DEVICES=0,1 bash run_ours.sh b2
#   CUDA_VISIBLE_DEVICES=0,1 bash run_ours.sh r
#   CUDA_VISIBLE_DEVICES=0,1 bash run_ours.sh c
#   CUDA_VISIBLE_DEVICES=0,1 bash run_ours.sh c-fg
#   CUDA_VISIBLE_DEVICES=0,1 bash run_ours.sh all

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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
  r)
    run_one "ours-r-zero-opposite" "ft_l40_ours_r_zero_opposite_5k.yaml"
    ;;
  r2-l01)
    run_one "ours-r2-opp-l01" "ft_l40_ours_r2_opp_l01_5k.yaml"
    ;;
  r2-l03)
    run_one "ours-r2-opp-l03" "ft_l40_ours_r2_opp_l03_5k.yaml"
    ;;
  r2-l05)
    run_one "ours-r2-opp-l05" "ft_l40_ours_r2_opp_l05_5k.yaml"
    ;;
  r2-l08)
    run_one "ours-r2-opp-l08" "ft_l40_ours_r2_opp_l08_5k.yaml"
    ;;
  c)
    run_one "ours-c-zero-reverse-hardneg" "ft_l40_ours_c_zero_reverse_hardneg_5k.yaml"
    ;;
  c-fg|cfg)
    run_one "ours-c-zero-reverse-hardneg-fg" "ft_l40_ours_c_zero_reverse_hardneg_fg_5k.yaml"
    ;;
  d)
    run_one "ours-d-r2-fg" "ft_l40_ours_d_r2_fg_5k.yaml"
    ;;
  dir)
    run_one "ours-e-dir" "ft_l40_ours_e_dir_5k.yaml"
    ;;
  dirsame)
    run_one "ours-e-dir-samedir-control" "ft_l40_ours_e_dirsame_5k.yaml"
    ;;
  dirctrl)
    run_one "ours-e-dir-control" "ft_l40_ours_e_dirctrl_5k.yaml"
    ;;
  dir-both)
    # real arm + matched control; the control is what makes the direction claim falsifiable
    run_one "ours-e-dir" "ft_l40_ours_e_dir_5k.yaml"
    run_one "ours-e-dir-control" "ft_l40_ours_e_dirctrl_5k.yaml"
    ;;
  cdlam)
    run_one "cdlam-repro" "ft_l40_cdlam_repro_5k.yaml"
    ;;
  idm)
    run_one "idm" "ft_l40_idm_5k.yaml"
    ;;
  lf)
    # Label-free twins (story b): trunk + best model without the 18-D action loss.
    run_one "kl-full-lf" "ft_l40_full_5k_lf.yaml"
    run_one "ours-r2-l01-lf" "ft_l40_ours_r2_opp_l01_5k_lf.yaml"
    ;;
  seeds)
    # Seed-variance runs (training.seed=1/2; split stays data.seed=42) for the three
    # headline models — sizes the noise floor before any headline claim.
    run_one "kl-full-s1" "ft_l40_full_5k_s1.yaml"
    run_one "kl-full-s2" "ft_l40_full_5k_s2.yaml"
    run_one "ours-a-s1" "ft_l40_ours_a_zero_5k_s1.yaml"
    run_one "ours-a-s2" "ft_l40_ours_a_zero_5k_s2.yaml"
    run_one "ours-r2-l01-s1" "ft_l40_ours_r2_opp_l01_5k_s1.yaml"
    run_one "ours-r2-l01-s2" "ft_l40_ours_r2_opp_l01_5k_s2.yaml"
    ;;
  seeds-local)
    # Local share of the seed runs (r2-s1/s2 + a-s1 run on the A6000 box).
    run_one "kl-full-s1" "ft_l40_full_5k_s1.yaml"
    run_one "kl-full-s2" "ft_l40_full_5k_s2.yaml"
    run_one "ours-a-s2" "ft_l40_ours_a_zero_5k_s2.yaml"
    ;;
  all)
    run_one "ours-a-zero" "ft_l40_ours_a_zero_5k.yaml"
    run_one "ours-b-zero-reverse" "ft_l40_ours_b_zero_reverse_5k.yaml"
    run_one "ours-c-zero-reverse-hardneg" "ft_l40_ours_c_zero_reverse_hardneg_5k.yaml"
    ;;
  *)
    echo "Usage: CUDA_VISIBLE_DEVICES=0,1 bash run_ours.sh {a|b|b2|r|r2-l01|r2-l03|r2-l05|r2-l08|c|c-fg|d|seeds|all}" >&2
    exit 2
    ;;
esac
