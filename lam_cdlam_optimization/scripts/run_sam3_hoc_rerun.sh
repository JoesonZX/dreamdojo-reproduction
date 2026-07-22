#!/bin/bash
# Re-run ONLY the object channel for tasks whose noun was fixed after the main
# run (diagnostics 1 & 2). --object_only reuses the stored, verified hand
# channel and recomputes just object -> half the time, hand never at risk.
#
#   GPUS="0 1" bash run_sam3_hoc_rerun.sh
set -u
cd "$(dirname "$0")/.."
ROOT="${ROOT:-/home/xuan/embodied-ai/data/egodex/test_240p}"
MAP="${MAP:-code/config/sam3_object_prompts.yaml}"
GPUS="${GPUS:-0 1}"
LOGDIR="${LOGDIR:-logs/sam3_hoc_rerun}"
mkdir -p "$LOGDIR"

# Tasks with a noun changed after the main run (verified + partial fixes).
TASKS=(
  play_reset_connect_four
  open_close_insert_remove_tupperware
  stack_unstack_tupperware
  scoop_dump_ice
  vertical_pick_place
  insert_remove_shirt_in_tube
  boil_serve_egg
  insert_remove_utensils
  color
  wrap_unwrap_food
  insert_remove_furniture_bench_cabinet
  insert_remove_plug_socket
  setup_cleanup_table
)

N=$(echo $GPUS | wc -w)
echo "[$(date '+%F %T')] object-only re-run of ${#TASKS[@]} tasks across $N GPU(s): $GPUS"

i=0
for g in $GPUS; do
  CUDA_VISIBLE_DEVICES=$g conda run --no-capture-output -n sam3 \
    python -u code/precompute_sam3_hoc_masks.py \
      --data_root "$ROOT" \
      --object_prompt_map "$MAP" \
      --object_only \
      --task "${TASKS[@]}" \
      --num_shards "$N" --shard "$i" \
      > "$LOGDIR/shard${i}_gpu${g}.log" 2>&1 &
  echo "  shard $i -> GPU $g"
  i=$((i+1))
done
wait

echo "[$(date '+%F %T')] re-run done."
echo "=== residual NO-OBJECT per task after re-run:"
grep -h NO-OBJECT $LOGDIR/shard*.log 2>/dev/null | grep -oP 'NO-OBJECT \[\K[^/]+' | sort | uniq -c | sort -rn
