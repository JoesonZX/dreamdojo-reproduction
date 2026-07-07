#!/usr/bin/env bash
# Fine-tune the official DreamDojo LAM_400k.ckpt under our 4 settings, in parallel
# (one per GPU), then auto-evaluate. Survives SSH disconnect (each run in its own tmux).
#
#   bash run_finetune.sh           # launch 4 FT runs (GPU 0..3) + eval waiter
#   bash run_finetune.sh --status  # show sessions + last log line
#   bash run_finetune.sh --kill    # kill all ftlam_* sessions
set -uo pipefail

REPO=/home/xuan/embodied-ai
PY=${PY:-/home/xuan/.venv/bin/python}
TRAIN=$REPO/lam_disentangle/code/train.py
EVAL=$REPO/lam_disentangle/code/eval.py
CFG=$REPO/lam_disentangle/code/config
OUT=$REPO/checkpoints/lam-dis
RES=$REPO/lam_disentangle/results
RUNS=(ft_l32_full ft_l32_frozen ft_l40_full ft_l40_frozen)   # index == GPU id

if [[ "${1:-}" == "--status" ]]; then
  tmux ls 2>/dev/null | grep ftlam_ || echo "no ftlam_ sessions"
  for n in "${RUNS[@]}"; do printf "%-16s " "$n"; tail -n1 "$OUT/$n/train.log" 2>/dev/null || echo "(no log)"; done
  exit 0
fi
if [[ "${1:-}" == "--kill" ]]; then
  for n in "${RUNS[@]}"; do tmux kill-session -t "ftlam_$n" 2>/dev/null && echo "killed ftlam_$n" || true; done
  tmux kill-session -t ftlam_eval 2>/dev/null || true
  exit 0
fi

for gpu in "${!RUNS[@]}"; do
  n="${RUNS[$gpu]}"; out="$OUT/$n"; mkdir -p "$out"; rm -rf "$out"/step_* "$out"/train.log
  tmux new-session -d -s "ftlam_$n" \
    "CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 $PY $TRAIN --config $CFG/$n.yaml 2>&1 | tee $out/train.log"
  echo "launched ftlam_$n on GPU $gpu"
done

# Auto-eval: 0-step anchor (eval the raw LAM_400k once via ft_l32_full arch) + each FT run.
cat > /tmp/eval_ftlam.sh <<EOF
set -uo pipefail
PY=$PY
for n in ${RUNS[@]}; do
  while ! grep -q "Training complete" "$OUT/\$n/train.log" 2>/dev/null; do sleep 60; done
done
for n in ${RUNS[@]}; do
  ck=\$(ls -d "$OUT/\$n"/step_* | sort | tail -1); mkdir -p "$RES/\$n"
  CUDA_VISIBLE_DEVICES=0 \$PY "$EVAL" --checkpoint "\$ck" --config "$CFG/\$n.yaml" \
    --mode action_probe --n_samples 4000 --out_path "$RES/\$n/probe.txt" >"$RES/\$n/probe_stdout.txt" 2>&1
  CUDA_VISIBLE_DEVICES=0 \$PY "$EVAL" --checkpoint "\$ck" --config "$CFG/\$n.yaml" \
    --mode reconstruction --n_samples 1000 >"$RES/\$n/recon.txt" 2>&1
done
echo "ftlam evals done \$(date)" > "$RES/ftlam_DONE.txt"
EOF
tmux new-session -d -s ftlam_eval 'bash /tmp/eval_ftlam.sh'
echo; echo "watch:  bash run_finetune.sh --status   |   results -> $RES/{run}/probe.txt"
