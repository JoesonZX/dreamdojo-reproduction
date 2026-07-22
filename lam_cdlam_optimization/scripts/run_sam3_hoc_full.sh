#!/bin/bash
# Full SAM3 hand/object precompute over EgoDex, sharded across GPUs.
#
# Safe to re-run: writes are atomic (tmp + rename) and an existing
# {episode}.npy is skipped BEFORE SAM3 runs, so a resumed job costs a stat
# per finished episode instead of re-propagating it.
#
#   bash run_sam3_hoc_full.sh                 # GPUs 0,1
#   GPUS="1 2 3" bash run_sam3_hoc_full.sh    # GPUs 1,2,3
set -u
cd "$(dirname "$0")/.."

ROOT="${ROOT:-/home/xuan/embodied-ai/data/egodex/test_240p}"
GPUS="${GPUS:-0 1}"
OUT_SUBDIR="${OUT_SUBDIR:-sam3_hoc_frames}"
OBJ_MAP="${OBJ_MAP:-code/config/sam3_object_prompts.yaml}"
LOGDIR="${LOGDIR:-logs/sam3_hoc}"
mkdir -p "$LOGDIR"

N=$(echo $GPUS | wc -w)
echo "[$(date '+%F %T')] full precompute -> $OUT_SUBDIR  across $N GPU(s): $GPUS"
echo "  videos: $(find "$ROOT" -name '*.mp4' | wc -l)   already done: $(find "$ROOT" -path "*$OUT_SUBDIR*" -name '*.npy' | wc -l)"

i=0
for g in $GPUS; do
  CUDA_VISIBLE_DEVICES=$g conda run --no-capture-output -n sam3 \
    python -u code/precompute_sam3_hoc_masks.py \
      --data_root "$ROOT" \
      --out_subdir "$OUT_SUBDIR" \
      --object_prompt_map "$OBJ_MAP" \
      --num_shards "$N" --shard "$i" \
      > "$LOGDIR/shard${i}_gpu${g}.log" 2>&1 &
  echo "  shard $i -> GPU $g  (log: $LOGDIR/shard${i}_gpu${g}.log)"
  i=$((i+1))
done
wait

echo "[$(date '+%F %T')] done. episodes: $(find "$ROOT" -path "*$OUT_SUBDIR*" -name '*.npy' | wc -l)"
echo "  size: $(du -sh --total $ROOT/*/$OUT_SUBDIR 2>/dev/null | tail -1)"
echo "  NO-OBJECT episodes: $(grep -ch NO-OBJECT $LOGDIR/shard*.log | paste -sd+ | bc)"
echo "  FAILED   episodes: $(grep -ch FAILED    $LOGDIR/shard*.log | paste -sd+ | bc)"

# Per-task object-grounding failure rate: the tasks whose noun needs fixing.
echo "=== tasks by NO-OBJECT count (fix these nouns, then re-run with --task X --overwrite):"
grep -h "NO-OBJECT" $LOGDIR/shard*.log | grep -oP 'NO-OBJECT \[\K[^/]+' | sort | uniq -c | sort -rn | head -40
