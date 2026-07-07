"""
Visualize SAM3 segmentation masks on EgoDex videos.

For each sampled episode:
  - Runs SAM3 video predictor with text prompt "hands"
  - Saves a comparison image: [frame | mask | overlay] × N frames

Usage:
  conda activate sam3
  python visualize_sam3_masks.py \\
      --data_root /home/xuan/embodied-ai/data/egodex/test \\
      --out_dir   /home/xuan/embodied-ai/results/sam3_masks \\
      --n_tasks   6 \\
      --prompt    "hands"

  # Single episode:
  python visualize_sam3_masks.py \\
      --video /home/xuan/embodied-ai/data/egodex/test/basic_pick_place/0.mp4
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch


def overlay_mask(frame: np.ndarray, mask: np.ndarray, color=(0, 255, 0), alpha=0.5) -> np.ndarray:
    """Overlay a binary mask on a BGR frame."""
    out = frame.copy()
    out[mask > 0] = (
        (1 - alpha) * out[mask > 0].astype(float)
        + alpha * np.array(color, dtype=float)
    ).astype(np.uint8)
    # Draw mask contour
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 2)
    return out


def load_predictor():
    """Build SAM3 predictor once (downloads checkpoint on first call)."""
    from sam3.model_builder import build_sam3_predictor
    # version="sam3" uses Sam3VideoPredictorMultiGPU (auto-manages CUDA)
    return build_sam3_predictor(version="sam3", use_fa3=False)


def run_sam3_on_video(video_path: Path, prompt: str, predictor):
    """
    Run SAM3 on a single video with a text prompt.

    SAM3 API (correct order):
      1. start_session  → session_id
      2. add_prompt     (registers text prompt on frame 0)
      3. handle_stream_request(propagate_in_video)  → yields per-frame dicts
         each dict has "out_binary_masks": {obj_id: np.ndarray [H, W] bool}

    Returns (frames_dict, masks_dict) where keys are frame indices.
    """
    # 1. Start session
    response = predictor.handle_request({
        "type": "start_session",
        "resource_path": str(video_path),
    })
    session_id = response["session_id"]

    # 2. Register text prompt on frame 0
    predictor.handle_request({
        "type": "add_prompt",
        "session_id": session_id,
        "frame_index": 0,
        "text": prompt,
    })

    # 3. Propagate through video; stream yields one dict per frame
    masks_per_frame = {}
    for out in predictor.handle_stream_request({
        "type": "propagate_in_video",
        "session_id": session_id,
    }):
        frame_idx = out.get("frame_index", len(masks_per_frame))
        # out = {"frame_index": int, "outputs": {"out_binary_masks": ndarray [N, H, W], ...}}
        outputs = out.get("outputs", out)  # fallback: some versions use flat dict
        binary_masks = outputs.get("out_binary_masks", None)
        if binary_masks is not None and len(binary_masks) > 0:
            if isinstance(binary_masks, torch.Tensor):
                binary_masks = binary_masks.cpu().numpy()
            # binary_masks: [N_objects, H, W] — merge all objects with OR
            merged = binary_masks[0].astype(bool)
            for i in range(1, len(binary_masks)):
                merged = merged | binary_masks[i].astype(bool)
            masks_per_frame[frame_idx] = merged

    # Close session to free memory
    try:
        predictor.handle_request({"type": "close_session", "session_id": session_id})
    except Exception:
        pass

    # Load video frames with OpenCV for visualization
    cap = cv2.VideoCapture(str(video_path))
    frames = {}
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames[idx] = frame
        idx += 1
    cap.release()

    return frames, masks_per_frame


def make_comparison_grid(frames: dict, masks: dict, n_sample: int = 8) -> np.ndarray:
    """
    Build a comparison grid image.
    Each row: [original frame | mask binary | overlay]
    """
    total = len(frames)
    if total == 0:
        return np.zeros((100, 300, 3), dtype=np.uint8)

    sample_idxs = sorted(random.sample(list(frames.keys()), min(n_sample, total)))
    rows = []
    for i in sample_idxs:
        frame = frames[i]
        h, w = frame.shape[:2]

        if i in masks and masks[i] is not None:
            mask = masks[i]
            if isinstance(mask, torch.Tensor):
                mask = mask.squeeze().cpu().numpy()
            mask = (mask > 0.5).astype(np.uint8)
            # Resize mask to frame size if needed
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_vis = (mask * 255).astype(np.uint8)
            mask_bgr = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
            overlay  = overlay_mask(frame, mask, color=(0, 200, 0), alpha=0.45)
        else:
            mask_bgr = np.zeros_like(frame)
            overlay  = frame.copy()
            cv2.putText(overlay, "no mask", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Add frame index label
        for img, label in [(frame, f"frame {i}"), (mask_bgr, "mask"), (overlay, "overlay")]:
            cv2.putText(img, label, (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        row = np.concatenate([frame, mask_bgr, overlay], axis=1)
        rows.append(row)

    return np.concatenate(rows, axis=0)


def process_video(video_path: Path, prompt: str, out_dir: Path,
                  predictor, n_sample: int = 8):
    task = video_path.parent.name
    ep   = video_path.stem
    print(f"  [{task}/{ep}] running SAM3 with prompt='{prompt}' ...", end="", flush=True)

    try:
        frames, masks = run_sam3_on_video(video_path, prompt, predictor)
        grid = make_comparison_grid(frames, masks, n_sample=n_sample)
        out_path = out_dir / task / f"{ep}_sam3.jpg"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), grid)
        n_masked = sum(1 for m in masks.values() if m is not None)
        print(f" {n_masked}/{len(frames)} frames masked → {out_path.name}")
    except Exception as e:
        print(f" FAILED: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/home/xuan/embodied-ai/data/egodex/test")
    parser.add_argument("--out_dir",   default="/home/xuan/embodied-ai/results/sam3_masks")
    parser.add_argument("--prompt",    default="hands",
                        help="Text prompt for SAM3 (e.g. 'hands', 'robot arm')")
    parser.add_argument("--n_tasks",   type=int, default=6,
                        help="Number of tasks to sample (1 episode each)")
    parser.add_argument("--n_sample",  type=int, default=8,
                        help="Number of frames to show per episode in the grid")
    parser.add_argument("--video",     default=None,
                        help="Run on a single video path instead of sampling")
    args = parser.parse_args()

    print("Loading SAM3 predictor (downloads checkpoint on first run)...")
    predictor = load_predictor()
    print("Model ready.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.video:
        process_video(Path(args.video), args.prompt, out_dir, predictor, args.n_sample)
        return

    data_root = Path(args.data_root)
    all_tasks = sorted(t for t in data_root.iterdir() if t.is_dir() and t.name != "flow_masks")
    sampled_tasks = random.sample(all_tasks, min(args.n_tasks, len(all_tasks)))

    print(f"Sampling {len(sampled_tasks)} tasks, prompt='{args.prompt}'")
    for task_dir in sampled_tasks:
        videos = sorted(task_dir.glob("*.mp4"))
        if not videos:
            continue
        video = random.choice(videos)
        process_video(video, args.prompt, out_dir, predictor, args.n_sample)

    print(f"\nDone. Results saved to {out_dir}")


if __name__ == "__main__":
    main()
