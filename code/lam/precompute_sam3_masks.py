"""
Precompute SAM3 foreground masks for EgoDex videos.

For each episode, SAM3 segments "hands" (or custom prompt) on every frame,
then saves the binary mask for each frame pair as a .pt file:
  {task_name}/sam3_masks/{episode_idx}/{t:06d}_skip{skip}.pt

This is the same storage format as precompute_flow_masks.py (RAFT), so the
EgoDexDataset can load them with use_fg_mask=True by switching mask_dir.

Usage (sam3 conda env, GPU 0):
  conda activate sam3
  CUDA_VISIBLE_DEVICES=0 python precompute_sam3_masks.py \\
      --data_root /home/xuan/embodied-ai/data/egodex/test \\
      --prompt "hands" \\
      --downsample_factors 1 2

Requirements:
  conda activate sam3  (Python 3.12, PyTorch 2.10, SAM3 installed)
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


def get_all_videos(data_root: Path) -> list:
    return sorted(data_root.glob("*/*.mp4"))


def run_sam3_on_video(video_path: Path, prompt: str, predictor, target_h: int, target_w: int):
    """
    Run SAM3 on a single video.
    Returns dict {frame_idx: binary mask np.ndarray [H, W] bool} for all frames.

    SAM3 API:
      1. start_session  → session_id
      2. add_prompt     (text prompt on frame 0)
      3. handle_stream_request(propagate_in_video) → yields per-frame dicts
         each dict: {"frame_index": int, "out_binary_masks": {obj_id: ndarray}}
    """
    response = predictor.handle_request({
        "type": "start_session",
        "resource_path": str(video_path),
    })
    session_id = response["session_id"]

    predictor.handle_request({
        "type": "add_prompt",
        "session_id": session_id,
        "frame_index": 0,
        "text": prompt,
    })

    masks = {}
    for out in predictor.handle_stream_request({
        "type": "propagate_in_video",
        "session_id": session_id,
    }):
        frame_idx = out.get("frame_index", len(masks))
        # out = {"frame_index": int, "outputs": {"out_binary_masks": ndarray [N, H, W], ...}}
        outputs = out.get("outputs", out)
        binary_masks = outputs.get("out_binary_masks", None)
        if binary_masks is not None and len(binary_masks) > 0:
            if isinstance(binary_masks, torch.Tensor):
                binary_masks = binary_masks.cpu().numpy()
            merged = binary_masks[0].astype(bool)
            for i in range(1, len(binary_masks)):
                merged = merged | binary_masks[i].astype(bool)
            if merged.shape != (target_h, target_w):
                merged = cv2.resize(merged.astype(np.uint8), (target_w, target_h),
                                    interpolation=cv2.INTER_NEAREST).astype(bool)
            masks[int(frame_idx)] = merged

    try:
        predictor.handle_request({"type": "close_session", "session_id": session_id})
    except Exception:
        pass

    return masks


def process_video(video_path: Path, prompt: str, predictor, downsample_factors: list,
                  img_h: int, img_w: int, dry_run: bool = False):
    """
    Process one video: run SAM3, save per-frame-pair masks.
    """
    task_name = video_path.parent.name
    ep_idx    = video_path.stem

    out_dir = video_path.parent / "sam3_masks" / ep_idx
    out_dir.mkdir(parents=True, exist_ok=True)

    # Count frames
    cap = cv2.VideoCapture(str(video_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if n_frames < 2:
        return 0

    try:
        masks = run_sam3_on_video(video_path, prompt, predictor, img_h, img_w)
    except Exception as e:
        print(f"  FAILED [{task_name}/{ep_idx}]: {e}")
        return 0

    saved = 0
    max_skip = max(downsample_factors)
    for t in range(n_frames - max_skip):
        for skip in downsample_factors:
            if t + skip >= n_frames:
                continue
            out_path = out_dir / f"{t:06d}_skip{skip}.pt"
            if out_path.exists():
                continue
            # For frame-pair (t, t+skip), use the mask at frame t as foreground indicator
            if t in masks:
                mask_tensor = torch.from_numpy(masks[t])
                torch.save(mask_tensor, out_path)
                saved += 1
        if dry_run and saved >= 5:
            break

    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",          default="/home/xuan/embodied-ai/data/egodex/test")
    parser.add_argument("--prompt",             default="hands")
    parser.add_argument("--downsample_factors", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--img_h",              type=int, default=240)
    parser.add_argument("--img_w",              type=int, default=320)
    parser.add_argument("--dry_run",            action="store_true",
                        help="Process first 3 videos only for testing")
    parser.add_argument("--task",               default=None,
                        help="Process only this task (folder name)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    from sam3.model_builder import build_sam3_predictor
    print("Loading SAM3 predictor (downloads checkpoint on first run)...")
    predictor = build_sam3_predictor(version="sam3", use_fa3=False)
    print("Model loaded.")

    data_root = Path(args.data_root)
    all_videos = get_all_videos(data_root)
    if args.task:
        all_videos = [v for v in all_videos if v.parent.name == args.task]
    if args.dry_run:
        all_videos = all_videos[:3]
        print(f"Dry run: processing {len(all_videos)} videos")
    else:
        print(f"Processing {len(all_videos)} videos, prompt='{args.prompt}'")

    total_saved = 0
    for vp in tqdm(all_videos, desc="videos"):
        n = process_video(vp, args.prompt, predictor,
                          args.downsample_factors, args.img_h, args.img_w, args.dry_run)
        total_saved += n

    print(f"\nDone. Saved {total_saved} mask files.")
    print(f"Masks stored at: {{task}}/sam3_masks/{{ep_idx}}/{{t:06d}}_skip{{skip}}.pt")


if __name__ == "__main__":
    main()
