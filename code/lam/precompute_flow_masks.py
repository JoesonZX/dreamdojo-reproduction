"""
Precompute RAFT optical flow masks for all AgiBot-World episodes.

Masks are saved as bool tensors at:
  <ep_dir>/flow_masks/{t:06d}_skip{skip}.pt

Usage:
  # All episodes, head_color, skip=1
  CUDA_VISIBLE_DEVICES=1 python precompute_flow_masks.py \
      --data_root /home/xuan/embodied-ai/data/agibotworld/extracted \
      --camera head_color \
      --skips 1 2 3 4 \
      --threshold 1.5

  # Dry run (process only first 2 episodes)
  CUDA_VISIBLE_DEVICES=1 python precompute_flow_masks.py \
      --data_root /home/xuan/embodied-ai/data/agibotworld/extracted \
      --dry_run
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    import decord
    decord.bridge.set_bridge("torch")
    _HAVE_DECORD = True
except ImportError:
    _HAVE_DECORD = False
    import cv2


def load_frames_cpu(video_path: Path, raft_h: int, raft_w: int):
    """Load all frames as float32 [N, 3, H, W] on CPU, resized for RAFT.

    Keeps frames on CPU to avoid OOM on long 1080p videos.
    Only 2 frames are moved to GPU at inference time.
    """
    if _HAVE_DECORD:
        vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
        frames = vr.get_batch(list(range(len(vr))))  # [N, H, W, 3] uint8
        frames = frames.float() / 255.0
        frames = frames.permute(0, 3, 1, 2)          # [N, 3, H, W]
    else:
        cap = cv2.VideoCapture(str(video_path))
        frames_list = []
        while True:
            ret, f = cap.read()
            if not ret:
                break
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            frames_list.append(torch.from_numpy(f).float() / 255.0)
        cap.release()
        if not frames_list:
            return torch.zeros(0, 3, 1, 1)
        frames = torch.stack(frames_list).permute(0, 3, 1, 2)  # [N, 3, H, W]

    # Resize on CPU (avoid allocating large GPU tensor for whole video)
    h, w = frames.shape[2], frames.shape[3]
    if h != raft_h or w != raft_w:
        frames = F.interpolate(frames, size=(raft_h, raft_w),
                               mode="bilinear", align_corners=False)
    return frames  # stays on CPU


def compute_masks_for_episode(
    video_path: Path,
    raft,
    skips: list,
    threshold: float,
    target_h: int,
    target_w: int,
    device: torch.device,
    dataset_type: str = "agibot",
):
    """Compute and save flow masks for one episode video.

    AgiBot layout: {task}/{ep_id}/videos/{camera}.mp4
      → masks saved at: {task}/{ep_id}/flow_masks/{t:06d}_skip{skip}.pt

    EgoDex layout: {task_name}/{idx}.mp4
      → masks saved at: {task_name}/flow_masks/{idx}/{t:06d}_skip{skip}.pt
    """
    if dataset_type == "egodex":
        mask_dir = video_path.parent / "flow_masks" / video_path.stem
    else:
        mask_dir = video_path.parent.parent / "flow_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    # Determine RAFT input size (must be divisible by 8) from first frame
    try:
        if _HAVE_DECORD:
            vr_probe = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
            h, w = vr_probe[0].shape[:2]
        else:
            cap = cv2.VideoCapture(str(video_path))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cap.release()
    except Exception as e:
        print(f"  [skip] Failed to probe {video_path}: {e}")
        return 0

    raft_h = (h // 8) * 8
    raft_w = (w // 8) * 8

    try:
        # All frames on CPU; only 2 frames sent to GPU per RAFT call
        frames = load_frames_cpu(video_path, raft_h, raft_w)
    except Exception as e:
        print(f"  [skip] Failed to load {video_path}: {e}")
        return 0

    n = frames.shape[0]
    if n < 2:
        return 0

    saved = 0
    for skip in skips:
        for t in range(n - skip):
            mask_path = mask_dir / f"{t:06d}_skip{skip}.pt"
            if mask_path.exists():
                continue  # Already computed, skip

            # Send only 2 frames to GPU
            f0 = frames[t:t+1].to(device)
            f1 = frames[t+skip:t+skip+1].to(device)

            with torch.no_grad():
                # RAFT expects images in [0, 1] float, returns flow [1, 2, H, W]
                flow_preds = raft(f0, f1)
                flow = flow_preds[-1]  # Use final prediction [1, 2, H, W]

            flow_mag = flow.norm(dim=1).squeeze(0)  # [H, W]

            # Dilate slightly to catch edges of moving objects
            mask = (flow_mag > threshold)
            mask = F.max_pool2d(
                mask.float().unsqueeze(0).unsqueeze(0), 3, stride=1, padding=1
            ).squeeze().bool()

            # Resize to target resolution
            if mask.shape != (target_h, target_w):
                mask = F.interpolate(
                    mask.float().unsqueeze(0).unsqueeze(0),
                    size=(target_h, target_w),
                    mode="nearest",
                ).squeeze().bool()

            torch.save(mask.cpu(), mask_path)
            saved += 1

    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/home/xuan/embodied-ai/data/agibotworld/extracted")
    parser.add_argument("--camera",       default="head_color",
                        help="AgiBot camera name (ignored for egodex)")
    parser.add_argument("--skips",        nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--threshold",    type=float, default=1.5)
    parser.add_argument("--target_h",     type=int, default=240)
    parser.add_argument("--target_w",     type=int, default=320)
    parser.add_argument("--raft_model",   default="small", choices=["small", "large"])
    parser.add_argument("--dataset_type", default="agibot", choices=["agibot", "egodex"],
                        help="Dataset layout: agibot (task/ep/videos/cam.mp4) or egodex (task/idx.mp4)")
    parser.add_argument("--dry_run",      action="store_true",
                        help="Process only the first 2 episodes")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    try:
        from torchvision.models.optical_flow import raft_small, raft_large, Raft_Small_Weights, Raft_Large_Weights
        if args.raft_model == "small":
            raft = raft_small(weights=Raft_Small_Weights.DEFAULT).to(device).eval()
        else:
            raft = raft_large(weights=Raft_Large_Weights.DEFAULT).to(device).eval()
        print(f"Loaded RAFT-{args.raft_model} from torchvision")
    except ImportError:
        print("ERROR: torchvision >= 0.13 required for RAFT.")
        sys.exit(1)

    data_root = Path(args.data_root)

    if args.dataset_type == "egodex":
        # EgoDex: {task_name}/{idx}.mp4
        video_paths = sorted(data_root.glob("*/*.mp4"))
    else:
        # AgiBot: {task_id}/{ep_id}/videos/{camera}.mp4
        video_paths = sorted(data_root.glob(f"*/*/videos/{args.camera}.mp4"))
        if not video_paths:
            video_paths = sorted(data_root.glob(f"*/videos/{args.camera}.mp4"))

    if not video_paths:
        print(f"No videos found under {args.data_root} (dataset_type={args.dataset_type})")
        sys.exit(1)

    print(f"Found {len(video_paths)} episodes | dataset={args.dataset_type} "
          f"skips={args.skips} threshold={args.threshold}")

    if args.dry_run:
        video_paths = video_paths[:2]
        print("Dry run: processing first 2 episodes only")

    total_saved = 0
    for i, vp in enumerate(video_paths):
        ep_label = "/".join(vp.parts[-2:])
        print(f"[{i+1}/{len(video_paths)}] {ep_label} … ", end="", flush=True)
        n = compute_masks_for_episode(
            vp, raft, args.skips, args.threshold,
            args.target_h, args.target_w, device,
            dataset_type=args.dataset_type,
        )
        print(f"{n} masks saved")
        total_saved += n

    print(f"\nDone. Total masks saved: {total_saved:,}")


if __name__ == "__main__":
    main()
