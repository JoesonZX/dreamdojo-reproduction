#!/usr/bin/env bash
# Launch the 4 disentanglement runs, one per GPU, each in its own tmux session.
#
#   bash run_all.sh            # parallel: baseline->GPU0 exp1->GPU1 exp2->GPU2 exp3->GPU3
#   bash run_all.sh --status   # show sessions + tail each log
#   bash run_all.sh --kill     # kill all lamdis_* sessions
#
# Attach to a run:   tmux attach -t lamdis_baseline      (detach: Ctrl-b then d)
set -euo pipefail

REPO=/home/xuan/embodied-ai
PY=${PY:-/home/xuan/.venv/bin/python}   # use the venv python explicitly (don't rely on PATH)
TRAIN=$REPO/lam_disentangle/code/train.py
CFG=$REPO/lam_disentangle/code/config
RUNS=(baseline exp1_action exp2_split exp3_indep)   # index == GPU id

if [[ "${1:-}" == "--status" ]]; then
  tmux ls 2>/dev/null | grep lamdis_ || echo "no lamdis_ sessions"
  for name in "${RUNS[@]}"; do
    echo "=== $name (last line) ==="
    tail -n 2 "$REPO/checkpoints/lam-dis/$name/train.log" 2>/dev/null || echo "  (no log yet)"
  done
  exit 0
fi

if [[ "${1:-}" == "--kill" ]]; then
  for name in "${RUNS[@]}"; do tmux kill-session -t "lamdis_$name" 2>/dev/null && echo "killed lamdis_$name" || true; done
  exit 0
fi

for gpu in "${!RUNS[@]}"; do
  name="${RUNS[$gpu]}"
  out="$REPO/checkpoints/lam-dis/$name"
  mkdir -p "$out"
  tmux new-session -d -s "lamdis_$name" \
    "CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 $PY $TRAIN --config $CFG/$name.yaml 2>&1 | tee $out/train.log"
  echo "launched lamdis_$name on GPU $gpu  ->  $out/train.log"
done
echo
echo "Watch all:   bash run_all.sh --status"
echo "Attach one:  tmux attach -t lamdis_baseline   (detach: Ctrl-b d)"
