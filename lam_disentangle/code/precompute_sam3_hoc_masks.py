"""
Precompute SAM3 hand/object/contact masks for EgoDex frame pairs.

Each saved file is a bool tensor [3,H,W]:
  channel 0: hands
  channel 1: manipulated objects
  channel 2: derived contact band = dilated(hand) & dilated(object)

The output layout matches EgoDexDataset's frame-pair lookup:
  {task_name}/sam3_hoc_masks/{episode_id}/{t:06d}_skip{skip}.pt

Run in the SAM3 environment:
  conda run -n sam3 python code/precompute_sam3_hoc_masks.py \
      --data_root /home/xuan/embodied-ai/data/egodex/test_240p \
      --downsample_factors 1 2
"""

import argparse
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


def run_sam3_on_video(video_path: Path, prompt: str, predictor, target_h: int, target_w: int):
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

    try:
        predictor.handle_request({"type": "close_session", "session_id": session_id})
    except Exception:
        pass
    return masks


def _frame_mask(masks: dict, frame_idx: int, shape):
    m = masks.get(frame_idx)
    if m is None:
        return np.zeros(shape, dtype=bool)
    return m


def _contact_mask(hand: np.ndarray, obj: np.ndarray, radius: int):
    if radius <= 0:
        return hand & obj
    k = 2 * radius + 1
    kernel = np.ones((k, k), dtype=np.uint8)
    hand_d = cv2.dilate(hand.astype(np.uint8), kernel, iterations=1).astype(bool)
    obj_d = cv2.dilate(obj.astype(np.uint8), kernel, iterations=1).astype(bool)
    return hand_d & obj_d


def process_video(video_path: Path, predictor, args):
    out_dir = video_path.parent / args.out_subdir / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if n_frames < 2:
        return 0

    try:
        hand_masks = run_sam3_on_video(
            video_path, args.hand_prompt, predictor, args.img_h, args.img_w,
        )
        object_masks = run_sam3_on_video(
            video_path, args.object_prompt, predictor, args.img_h, args.img_w,
        )
    except Exception as exc:
        print(f"FAILED [{video_path.parent.name}/{video_path.stem}]: {exc}")
        return 0

    saved = 0
    shape = (args.img_h, args.img_w)
    max_skip = max(args.downsample_factors)
    for t in range(n_frames - max_skip):
        for skip in args.downsample_factors:
            if t + skip >= n_frames:
                continue
            out_path = out_dir / f"{t:06d}_skip{skip}.pt"
            if out_path.exists() and not args.overwrite:
                continue

            # Weight both endpoints because the reconstruction target is t+skip,
            # while the action evidence is the transition between the two frames.
            hand = _frame_mask(hand_masks, t, shape) | _frame_mask(hand_masks, t + skip, shape)
            obj = _frame_mask(object_masks, t, shape) | _frame_mask(object_masks, t + skip, shape)
            contact = _contact_mask(hand, obj, args.contact_radius)
            hoc = torch.from_numpy(np.stack([hand, obj, contact], axis=0))
            torch.save(hoc, out_path)
            saved += 1

        if args.dry_run and saved >= 8:
            break
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/home/xuan/embodied-ai/data/egodex/test_240p")
    parser.add_argument("--out_subdir", default="sam3_hoc_masks")
    parser.add_argument("--hand_prompt", default="hands")
    parser.add_argument("--object_prompt", default="objects being manipulated")
    parser.add_argument("--downsample_factors", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--img_h", type=int, default=240)
    parser.add_argument("--img_w", type=int, default=320)
    parser.add_argument("--contact_radius", type=int, default=5)
    parser.add_argument("--task", default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    from sam3.model_builder import build_sam3_predictor

    print("Loading SAM3 predictor...")
    predictor = build_sam3_predictor(version="sam3", use_fa3=False)
    print("Model loaded.")

    videos = get_all_videos(Path(args.data_root))
    if args.task:
        videos = [v for v in videos if v.parent.name == args.task]
    if args.dry_run:
        videos = videos[:3]
    print(f"Processing {len(videos)} videos -> {args.out_subdir}")

    total = 0
    for video_path in tqdm(videos, desc="videos"):
        total += process_video(video_path, predictor, args)
    print(f"Done. Saved {total} H/O/C mask files.")


if __name__ == "__main__":
    main()
