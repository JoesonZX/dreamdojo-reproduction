#!/usr/bin/env bash
# Wait until all 4 trainings finish, then run all 4 evaluations for each run and
# write a summary. Designed to run UNATTENDED inside its own tmux session, so it
# survives SSH disconnects and produces results without anyone watching.
#
#   tmux new-session -d -s lamdis_eval 'bash lam_disentangle/run_eval_all.sh'
#   tmux attach -t lamdis_eval        # check on it (Ctrl-b d to detach)
#
# Results land in lam_disentangle/results/.
set -uo pipefail

REPO=/home/xuan/embodied-ai
CODE=$REPO/lam_disentangle/code
OUT=$REPO/checkpoints/lam-dis
RES=$REPO/lam_disentangle/results
RUNS=(baseline exp1_action exp2_split exp3_indep)
GPU=${EVAL_GPU:-0}
PY=${PY:-/home/xuan/.venv/bin/python}
mkdir -p "$RES"
SUMMARY="$RES/SUMMARY.txt"
: > "$SUMMARY"

log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$RES/eval.log"; }

# ── 1. Wait for all trainings to finish ────────────────────────────────────────
log "waiting for 4 trainings to complete..."
while true; do
  done_n=0
  for n in "${RUNS[@]}"; do grep -q "Training complete" "$OUT/$n/train.log" 2>/dev/null && done_n=$((done_n+1)); done
  for n in "${RUNS[@]}"; do
    if grep -qE "Traceback|CUDA error|Killed|OOM" "$OUT/$n/train.log" 2>/dev/null; then
      log "ERROR detected in $n — proceeding to eval whatever checkpoints exist"; break
    fi
  done
  [ $done_n -eq 4 ] && { log "all 4 trainings complete"; break; }
  sleep 30
done

# ── 2. Evaluate each run (sequential, single GPU) ──────────────────────────────
for n in "${RUNS[@]}"; do
  cfg="$CODE/config/$n.yaml"
  ckpt=$(ls -d "$OUT/$n"/step_* 2>/dev/null | sort | tail -1)
  rdir="$RES/$n"; mkdir -p "$rdir"
  if [ -z "$ckpt" ]; then log "[$n] no checkpoint found, skipping"; continue; fi
  log "[$n] evaluating $ckpt"

  CUDA_VISIBLE_DEVICES=$GPU "$PY" "$CODE/eval.py" --checkpoint "$ckpt" --config "$cfg" \
      --mode action_probe   --n_samples 4000 --out_path "$rdir/probe.txt"  >>"$rdir/probe_stdout.txt" 2>&1
  CUDA_VISIBLE_DEVICES=$GPU "$PY" "$CODE/eval.py" --checkpoint "$ckpt" --config "$cfg" \
      --mode reconstruction --n_samples 1000                                >"$rdir/recon.txt" 2>&1
  CUDA_VISIBLE_DEVICES=$GPU "$PY" "$CODE/eval.py" --checkpoint "$ckpt" --config "$cfg" \
      --mode visualize      --n_samples 2000 --out_path "$rdir/tsne.png"   >>"$rdir/viz_stdout.txt" 2>&1
  CUDA_VISIBLE_DEVICES=$GPU "$PY" "$CODE/eval.py" --checkpoint "$ckpt" --config "$cfg" \
      --mode perturb                          --out_path "$rdir/perturb.png" >>"$rdir/perturb_stdout.txt" 2>&1
  log "[$n] done -> $rdir"
done

# ── 3. Build a compact summary table ───────────────────────────────────────────
{
  echo "==== LAM disentanglement — evaluation summary ===="
  echo "generated: $(date)"
  for n in "${RUNS[@]}"; do
    echo; echo "### $n"
    echo "-- real-action regression R^2 (action-sub should be >> env-sub) --"
    grep -E "full z_mu|action-sub|env-sub" "$RES/$n/probe.txt" 2>/dev/null
    echo "-- classification (action-sub vs env-sub) --"
    grep -E "action-sub:|env-sub   :" "$RES/$n/probe.txt" 2>/dev/null
    echo "-- reconstruction (world-model-quality proxy) --"
    grep -E "PSNR|MSE" "$RES/$n/recon.txt" 2>/dev/null | head -1
  done
} | tee "$SUMMARY"

log "ALL EVALS COMPLETE -> $SUMMARY"
