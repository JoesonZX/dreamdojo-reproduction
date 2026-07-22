#!/usr/bin/env bash
# Train every arm, one after another, on a 3-GPU machine.
#
#   bash fast_wam/wam/scripts/train_all.sh                 # all arms, seed 42
#   bash fast_wam/wam/scripts/train_all.sh "sv mv_cross"   # a subset
#   SEEDS="42 1 2" bash fast_wam/wam/scripts/train_all.sh  # seed replicates
#
# Layout: GPUs 0+1 run the heavy cross-view arms two-way data-parallel; GPU 2
# runs a single-GPU arm concurrently (gradient_accumulation doubled so the
# effective batch matches). Adjust GPU_PAIR / GPU_SOLO for your machine.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
CFG="$ROOT/fast_wam/wam/config"
PY="${PY:-python}"

ARMS="${1:-sv mv_data mv_cross view_prompt}"
SEEDS="${SEEDS:-42}"
GPU_PAIR="${GPU_PAIR:-0,1}"
PORT="${PORT:-29540}"

command -v nvidia-smi >/dev/null && {
  echo "== GPU state before starting (shared machine: check nothing else is running) =="
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
  echo
}

for seed in $SEEDS; do
  for arm in $ARMS; do
    cfg="$CFG/$arm.yaml"
    [[ -f "$cfg" ]] || { echo "!! no config $cfg -- skipping"; continue; }
    out="$ROOT/fast_wam/checkpoints/${arm}_s${seed}"
    log="$ROOT/fast_wam/results/${arm}_s${seed}.log"
    mkdir -p "$(dirname "$log")"

    echo "===================================================================="
    echo "ARM $arm  seed $seed  -> $out"
    echo "===================================================================="
    CUDA_VISIBLE_DEVICES="$GPU_PAIR" "$PY" -m accelerate.commands.launch \
      --num_processes 2 --mixed_precision bf16 --main_process_port "$PORT" \
      "$ROOT/fast_wam/wam/train.py" --config "$cfg" \
      --override "training.seed=$seed" "training.output_dir=$out" \
      2>&1 | tee "$log"

    if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
      echo "!! $arm seed $seed FAILED -- see $log"
      continue
    fi

    last=$(ls -d "$out"/step_* 2>/dev/null | sort | tail -1)
    if [[ -n "$last" ]]; then
      echo "-- evaluating $last"
      CUDA_VISIBLE_DEVICES="${GPU_PAIR%%,*}" "$PY" "$ROOT/fast_wam/wam/eval_crossview.py" \
        --checkpoint "$last" --config "$cfg" --n_samples 4000 \
        --out "$ROOT/fast_wam/results/${arm}_s${seed}.json" 2>&1 | tee -a "$log"
    fi
  done
done

echo
echo "== summary =="
"$PY" - <<PY
import json, glob, os
rows = []
for p in sorted(glob.glob("$ROOT/fast_wam/results/*.json")):
    d = json.load(open(p))
    rows.append((os.path.basename(p)[:-5], d.get("crossview_cos_median"),
                 d.get("null_cos_median"), d.get("crossview_gain"),
                 d.get("action_r2_in_view"), d.get("action_r2_heldout_view")))
if rows:
    print(f"{'arm':22s} {'cos':>8s} {'null':>8s} {'gain':>8s} {'R2_in':>8s} {'R2_held':>8s}")
    for r in rows:
        f = lambda v: f"{v:8.4f}" if isinstance(v, float) else f"{'--':>8s}"
        print(f"{r[0]:22s} {f(r[1])} {f(r[2])} {f(r[3])} {f(r[4])} {f(r[5])}")
    print()
    print("Read 'gain' (= cos - null), NOT 'cos'. Raw cosine is ~1 even for an")
    print("untrained model because the latent has a dominant mean direction.")
PY
