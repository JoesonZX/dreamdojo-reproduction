"""
Precompute SAM3 hand/object masks for EgoDex videos.

One file per episode, holding PER-FRAME masks:
  {task_name}/sam3_hoc_frames/{episode_id}.npy   uint8 [2, n_frames, H, W//8]
    channel 0: hands
    channel 1: manipulated objects
Bits are packed along the width axis (np.packbits), so H and W are both
recoverable from the shape and the array mmaps cleanly for random access.

The pair-level tensor the trainer consumes ([3,H,W] hand/object/contact for a
given (t, skip)) is derived on the fly in EgoDexDataset._load_fg_mask. Storing
per-frame instead of per-pair keeps `downsample_factors` and `contact_radius`
out of the precompute, so changing either no longer costs a full SAM3 rerun.

Run in the SAM3 environment, sharded across GPUs:
  for i in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$i conda run --no-capture-output -n sam3 \
      python -u code/precompute_sam3_hoc_masks.py \
        --data_root /home/xuan/embodied-ai/data/egodex/test_240p \
        --num_shards 3 --shard $i &
  done
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


def get_all_videos(data_root: Path):
    return sorted(data_root.glob("*/*.mp4"))


def _resize_mask(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    if mask.shape != (target_h, target_w):
        mask = cv2.resize(
            mask.astype(np.uint8),
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    return mask.astype(bool)


def _response_has_mask(response) -> bool:
    outputs = (response or {}).get("outputs", response) or {}
    binary_masks = outputs.get("out_binary_masks", None)
    if binary_masks is None or len(binary_masks) == 0:
        return False
    if isinstance(binary_masks, torch.Tensor):
        return bool(binary_masks.any())
    return any(np.asarray(m).any() for m in binary_masks)


def run_sam3_on_video(video_path: Path, prompt: str, predictor, target_h: int, target_w: int,
                      n_frames: int, anchor_fracs=(0.5,), output_prob_thresh: float = 0.5):
    """Ground `prompt` on an anchor frame, then propagate both ways from it.

    SAM3 grounds the text prompt exactly once and propagates the resulting
    tracks. Grounding on frame 0 loses the whole episode whenever the hand has
    not reached the object yet -- common in EgoDex, where a clip often opens on
    an untouched scene. So try anchors mid-clip (where a manipulation is
    actually in progress) and keep the first that grounds anything, then
    propagate in BOTH directions so the pre-contact frames are still covered.
    """
    response = predictor.handle_request({
        "type": "start_session",
        "resource_path": str(video_path),
    })
    session_id = response["session_id"]
    try:
        anchor = None
        for frac in anchor_fracs:
            cand = min(max(int(n_frames * frac), 0), max(n_frames - 1, 0))
            resp = predictor.handle_request({
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": cand,
                "text": prompt,
                "output_prob_thresh": output_prob_thresh,
            })
            if _response_has_mask(resp):
                anchor = cand
                break
        if anchor is None:
            return {}, None

        masks = {}
        for out in predictor.handle_stream_request({
            "type": "propagate_in_video",
            "session_id": session_id,
            "start_frame_index": anchor,
            "propagation_direction": "both",
            "output_prob_thresh": output_prob_thresh,
        }):
            frame_idx = out.get("frame_index", len(masks))
            outputs = out.get("outputs", out)
            binary_masks = outputs.get("out_binary_masks", None)
            if binary_masks is None or len(binary_masks) == 0:
                continue
            if isinstance(binary_masks, torch.Tensor):
                binary_masks = binary_masks.cpu().numpy()
            merged = binary_masks[0].astype(bool)
            for i in range(1, len(binary_masks)):
                merged |= binary_masks[i].astype(bool)
            masks[int(frame_idx)] = _resize_mask(merged, target_h, target_w)
    finally:
        # clear_cache_threshold=0 forces empty_cache() on EVERY close, not just
        # when the card is >80% full (SAM3's default gate). A single cluttered
        # video (a pile of foam legos) transiently balloons the caching
        # allocator's *reserved* pool to ~50G -- live `alloc` stays ~5G, so it's
        # fragmentation, not a leak -- and draining it every close lets the next
        # video start clean instead of OOMing on the poisoned pool. Verified:
        # soft_legos OOM -> reserved 50G -> next video back to 5.24G. The extra
        # CUDA sync (~100ms) is nothing against ~20s/video.
        try:
            predictor.handle_request({
                "type": "close_session",
                "session_id": session_id,
                "clear_cache_threshold": 0,
            })
        except Exception:
            pass
    return masks, anchor


def _to_array(masks: dict, n_frames: int, h: int, w: int) -> np.ndarray:
    arr = np.zeros((n_frames, h, w), dtype=bool)
    for idx, m in masks.items():
        if 0 <= idx < n_frames:
            arr[idx] = m
    return arr


def _run_channel(video_path, prompt, predictor, args, n_frames):
    """One SAM3 pass, returning ({}, None) on OOM instead of retrying.

    Retrying at higher output_prob_thresh was verified useless here: a genuinely
    cluttered video (soft legos) OOMs at 0.5/0.8/0.95 alike, because the blowup
    is in detection/propagation, not the prob gate. Worse, each retry re-loads
    the whole clip onto the GPU, compounding the reserved pool. So fail fast:
    empty the cache and yield an empty channel for this video (logged upstream),
    keeping the caching allocator clean for the next video.
    """
    try:
        return run_sam3_on_video(
            video_path, prompt, predictor, args.img_h, args.img_w,
            n_frames, args.anchor_fracs, args.output_prob_thresh,
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"  OOM [{video_path.parent.name}/{video_path.stem}] '{prompt}' "
              f"-> empty channel", flush=True)
        return {}, None


def resolve_object_prompt(video_path: Path, args) -> str:
    return args.object_prompt_map.get(video_path.parent.name,
                                      args.object_prompt_map.get("default", args.object_prompt))


def process_video(video_path: Path, predictor, args):
    out_dir = video_path.parent / args.out_subdir
    out_path = out_dir / f"{video_path.stem}.npy"
    # Writes are atomic (tmp + rename), so an existing file is always complete.
    # --object_only intentionally reprocesses existing files (to graft a new
    # object channel onto the stored hand), so it must not be skipped here.
    if out_path.exists() and not args.overwrite and not args.object_only:
        return 0

    cap = cv2.VideoCapture(str(video_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if n_frames < 2:
        return 0

    object_prompt = resolve_object_prompt(video_path, args)

    # --object_only: the hand channel is already computed and correct for every
    # video (99.6% coverage), so a noun-fix re-run should recompute ONLY object
    # and reuse the stored hand -- halving the re-run and never risking the good
    # hand masks. Needs an existing file to graft onto; falls back to a full run.
    prev_hand = None
    if args.object_only and out_path.exists():
        try:
            prev = np.load(out_path, mmap_mode="r")     # [2, n, H, W//8] uint8
            prev_hand = np.unpackbits(prev[0], axis=-1).astype(bool)  # [n,H,W]
        except Exception:
            prev_hand = None

    try:
        if prev_hand is None:
            hand_masks, hand_anchor = _run_channel(
                video_path, args.hand_prompt, predictor, args, n_frames)
        object_masks, obj_anchor = _run_channel(
            video_path, object_prompt, predictor, args, n_frames)
    except Exception as exc:
        print(f"FAILED [{video_path.parent.name}/{video_path.stem}]: "
              f"{str(exc)[:200]}", flush=True)
        torch.cuda.empty_cache()
        return 0

    if prev_hand is None and hand_anchor is None:
        print(f"NO-HAND [{video_path.parent.name}/{video_path.stem}]: "
              f"'{args.hand_prompt}' grounded nothing", flush=True)
    if obj_anchor is None:
        print(f"NO-OBJECT [{video_path.parent.name}/{video_path.stem}]: "
              f"'{object_prompt}' grounded nothing at any anchor", flush=True)

    # CAP_PROP_FRAME_COUNT can disagree with what the decoder actually yielded;
    # trust whichever is larger so no propagated frame is silently dropped.
    seen = max([*object_masks.keys(), n_frames - 1])
    if prev_hand is None:
        seen = max(seen, *hand_masks.keys())
    n_frames = max(n_frames, seen + 1)

    if prev_hand is not None:
        # Align the stored hand array to n_frames (pad/truncate along time).
        hand_arr = np.zeros((n_frames, args.img_h, args.img_w), dtype=bool)
        m = min(n_frames, prev_hand.shape[0])
        hand_arr[:m] = prev_hand[:m]
    else:
        hand_arr = _to_array(hand_masks, n_frames, args.img_h, args.img_w)

    stacked = np.stack([
        hand_arr,
        _to_array(object_masks, n_frames, args.img_h, args.img_w),
    ])                                              # [2, n, H, W] bool
    packed = np.packbits(stacked, axis=-1)          # [2, n, H, W//8] uint8

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(f".tmp{os.getpid()}.npy")
    np.save(tmp_path, packed)
    os.replace(tmp_path, out_path)
    return n_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/home/xuan/embodied-ai/data/egodex/test_240p")
    parser.add_argument("--out_subdir", default="sam3_hoc_frames")
    parser.add_argument("--hand_prompt", default="hands")
    parser.add_argument("--object_prompt", default="objects being manipulated",
                        help="fallback object prompt when no map / no map entry")
    parser.add_argument("--object_prompt_map", default=None,
                        help="YAML of {task_name: noun}; SAM3 needs a concrete "
                             "category noun to ground on this footage")
    parser.add_argument("--img_h", type=int, default=240)
    parser.add_argument("--img_w", type=int, default=320)
    parser.add_argument("--anchor_fracs", nargs="+", type=float, default=[0.5, 0.35, 0.65],
                        help="fractions of the clip to try grounding on, in order; "
                             "the first one that grounds anything wins")
    parser.add_argument("--output_prob_thresh", type=float, default=0.5,
                        help="raise to admit fewer instances in cluttered scenes")
    parser.add_argument("--task", nargs="+", default=None,
                        help="restrict to one or more task names")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None,
                        help="process at most N videos (pilot runs)")
    parser.add_argument("--stratified", action="store_true",
                        help="with --limit, round-robin across tasks instead of "
                             "taking an alphabetical prefix")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--object_only", action="store_true",
                        help="recompute ONLY the object channel and reuse the "
                             "stored hand channel (for noun-fix re-runs)")
    args = parser.parse_args()

    if args.img_w % 8 != 0:
        raise ValueError(f"img_w must be a multiple of 8 for bit packing, got {args.img_w}")
    if not 0 <= args.shard < args.num_shards:
        raise ValueError(f"shard must be in [0,{args.num_shards}), got {args.shard}")

    map_path = args.object_prompt_map
    if map_path:
        import yaml
        with open(map_path) as f:
            args.object_prompt_map = yaml.safe_load(f) or {}
        print(f"Loaded {len(args.object_prompt_map)} object prompts from {map_path}", flush=True)
    else:
        args.object_prompt_map = {}

    from sam3.model_builder import build_sam3_predictor

    print("Loading SAM3 predictor...", flush=True)
    predictor = build_sam3_predictor(version="sam3", use_fa3=False)
    print("Model loaded.", flush=True)

    videos = get_all_videos(Path(args.data_root))
    if args.task:
        keep = set(args.task)
        videos = [v for v in videos if v.parent.name in keep]

    if not videos:
        raise SystemExit(f"No videos under {args.data_root}"
                         + (f" for task={args.task}" if args.task else ""))

    if args.limit is not None:
        if args.stratified:
            # Alphabetical order buries whole task families past any prefix cut,
            # so a pilot subset must interleave tasks and sample within them.
            rng = np.random.default_rng(args.seed)
            by_task = {}
            for v in videos:
                by_task.setdefault(v.parent.name, []).append(v)
            for group in by_task.values():
                rng.shuffle(group)
            ordered, rounds = [], max(len(g) for g in by_task.values())
            for i in range(rounds):
                for task in sorted(by_task):
                    if i < len(by_task[task]):
                        ordered.append(by_task[task][i])
            videos = ordered[:args.limit]
        else:
            videos = videos[:args.limit]

    # Shard AFTER subsetting so every shard sees the same candidate pool.
    videos = videos[args.shard::args.num_shards]
    print(f"[shard {args.shard}/{args.num_shards}] {len(videos)} videos "
          f"-> {args.out_subdir}", flush=True)

    total_frames = 0
    done = 0
    for video_path in tqdm(videos, desc=f"shard{args.shard}"):
        n = process_video(video_path, predictor, args)
        if n:
            done += 1
            total_frames += n
    print(f"Done. Wrote {done} episodes / {total_frames} frames.", flush=True)


if __name__ == "__main__":
    main()
