"""
Sanity-check SAM3 hand/object/contact masks before committing to a full run.

Answers one question: does the open-vocabulary prompt "objects being
manipulated" actually work on egocentric EgoDex footage? Two outputs:

  1. Overlay grids (PNG) — hand=red, object=green, contact=blue.
  2. Per-task coverage stats, plus the failure rates that decide it:
       obj_empty  — prompt found nothing (mask is unusable)
       obj_full   — prompt degenerated to segmenting the whole scene
     Both are the ways an open-vocab prompt fails silently; plain coverage
     means little on its own.

Reads either mask layout, so it runs on the existing per-pair .pt tree today
without waiting for the per-frame rerun:
  sam3_hoc_frames/{episode}.npy       uint8 [2,n,H,W//8]   (per-frame, new)
  sam3_hoc_masks/{episode}/*.pt       bool  [3,H,W]        (per-pair, legacy)

  python code/inspect_sam3_hoc.py \
      --data_root /home/xuan/embodied-ai/data/egodex/test_240p \
      --n_videos 24 --frames_per_video 4 --out /tmp/sam3_check
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch


def _contact_band(hand: np.ndarray, obj: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return (hand & obj).astype(bool)
    k = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    return cv2.dilate(hand, k).astype(bool) & cv2.dilate(obj, k).astype(bool)


def load_masks_for_frames(video_path: Path, frames, contact_radius: int,
                          frames_subdir: str = "sam3_hoc_frames",
                          allow_legacy: bool = True):
    """Return {frame_idx: [3,H,W] bool} from whichever layout exists.

    allow_legacy must be False when comparing two per-frame variants: otherwise
    an episode missing from `frames_subdir` silently falls back to the shared
    legacy tree, and every variant reports identical numbers.
    """
    npy = video_path.parent / frames_subdir / f"{video_path.stem}.npy"
    if npy.exists():
        arr = np.load(npy, mmap_mode="r")
        n = arr.shape[1]
        out = {}
        for t in frames:
            if t >= n:
                continue
            hand = np.unpackbits(arr[0, t], axis=-1)
            obj = np.unpackbits(arr[1, t], axis=-1)
            contact = _contact_band(hand, obj, contact_radius)
            out[t] = np.stack([hand.astype(bool), obj.astype(bool), contact])
        return out, "frames"

    pair_dir = video_path.parent / "sam3_hoc_masks" / video_path.stem
    if allow_legacy and pair_dir.is_dir():
        out = {}
        for t in frames:
            p = pair_dir / f"{t:06d}_skip1.pt"
            if p.exists():
                out[t] = torch.load(p, map_location="cpu", weights_only=True).numpy()
        return out, "pairs"

    return {}, None


def read_frames(video_path: Path, frames):
    cap = cv2.VideoCapture(str(video_path))
    out = {}
    for t in sorted(frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t))
        ok, img = cap.read()
        if ok:
            out[t] = img
    cap.release()
    return out


def overlay(img: np.ndarray, hoc: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """hand=red, object=green, contact=blue, drawn over a dimmed frame."""
    h, w = hoc.shape[-2:]
    if img.shape[:2] != (h, w):
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    out = img.astype(np.float32)
    # BGR: hand -> red channel 2, object -> green 1, contact -> blue 0
    for ch, bgr_idx in ((0, 2), (1, 1), (2, 0)):
        m = hoc[ch].astype(bool)
        if m.any():
            out[m] = (1 - alpha) * out[m]
            out[m, bgr_idx] = np.minimum(255.0, out[m, bgr_idx] + alpha * 255.0)
    return out.astype(np.uint8)


def pick_videos(data_root: Path, n_videos: int, seed: int,
                frames_subdir: str = "sam3_hoc_frames",
                allow_legacy: bool = True):
    """Round-robin across tasks so the sample is not an alphabetical prefix."""
    by_task = defaultdict(list)
    for v in sorted(data_root.glob("*/*.mp4")):
        # The legacy precompute mkdir'd an episode dir before running SAM3, so
        # "directory exists" is not evidence of data — require a real file.
        legacy = v.parent / "sam3_hoc_masks" / v.stem
        has = (v.parent / frames_subdir / f"{v.stem}.npy").exists()
        if allow_legacy and not has:
            has = legacy.is_dir() and next(legacy.glob("*.pt"), None) is not None
        if has:
            by_task[v.parent.name].append(v)
    if not by_task:
        return []
    rng = np.random.default_rng(seed)
    for g in by_task.values():
        rng.shuffle(g)
    ordered = []
    for i in range(max(len(g) for g in by_task.values())):
        for task in sorted(by_task):
            if i < len(by_task[task]):
                ordered.append(by_task[task][i])
    return ordered[:n_videos]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/home/xuan/embodied-ai/data/egodex/test_240p")
    ap.add_argument("--out", default="/tmp/sam3_check")
    ap.add_argument("--n_videos", type=int, default=24)
    ap.add_argument("--frames_per_video", type=int, default=4)
    ap.add_argument("--contact_radius", type=int, default=5)
    ap.add_argument("--full_thresh", type=float, default=0.5,
                    help="object coverage above this counts as a degenerate "
                         "whole-scene segmentation")
    ap.add_argument("--frames_subdir", default="sam3_hoc_frames",
                    help="per-frame layout to read; vary it to compare prompts")
    ap.add_argument("--allow_legacy", choices=["auto", "yes", "no"], default="auto",
                    help="fall back to the legacy sam3_hoc_masks tree. 'auto' "
                         "disables it whenever --frames_subdir is non-default, "
                         "so variant-vs-variant comparisons stay honest.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    allow_legacy = (args.allow_legacy == "yes" or
                    (args.allow_legacy == "auto" and args.frames_subdir == "sam3_hoc_frames"))

    videos = pick_videos(data_root, args.n_videos, args.seed, args.frames_subdir, allow_legacy)
    if not videos:
        print(f"No masks found under {data_root}/*/{args.frames_subdir}"
              + (" (nor the legacy tree)" if allow_legacy else " (legacy fallback off)"))
        return
    print(f"Sampling {len(videos)} episodes across "
          f"{len({v.parent.name for v in videos})} tasks "
          f"[subdir={args.frames_subdir} legacy_fallback={allow_legacy}]")

    rng = np.random.default_rng(args.seed)
    per_task = defaultdict(list)
    rows = []

    for v in videos:
        cap = cv2.VideoCapture(str(v))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if n < 4:
            continue
        # Skip the very start/end: the first frames are often pre-contact.
        frames = sorted(rng.choice(np.arange(int(n * 0.1), int(n * 0.9)),
                                   size=min(args.frames_per_video, max(1, int(n * 0.8))),
                                   replace=False).tolist())
        masks, layout = load_masks_for_frames(v, frames, args.contact_radius,
                                              args.frames_subdir, allow_legacy)
        if not masks:
            continue
        imgs = read_frames(v, list(masks.keys()))

        tiles = []
        for t in sorted(masks):
            hoc = masks[t]
            npix = float(hoc.shape[-1] * hoc.shape[-2])
            stat = {
                "hand": hoc[0].sum() / npix,
                "obj": hoc[1].sum() / npix,
                "contact": hoc[2].sum() / npix,
            }
            per_task[v.parent.name].append(stat)
            rows.append(stat)
            if t in imgs:
                tile = overlay(imgs[t], hoc)
                cv2.putText(tile, f"{t} h{stat['hand']:.2f} o{stat['obj']:.2f}",
                            (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                tiles.append(tile)

        if tiles:
            grid = np.concatenate(tiles, axis=1)
            cv2.imwrite(str(out_dir / f"{v.parent.name}__{v.stem}.png"), grid)

    if not rows:
        print("Masks located but no frames decoded — check the videos.")
        return

    def agg(items, key):
        return float(np.mean([r[key] for r in items]))

    print(f"\nlayout={layout}  frames={len(rows)}  overlays -> {out_dir}\n")
    print(f"{'task':<52} {'hand':>6} {'obj':>6} {'cont':>6} {'o_empty':>8} {'o_full':>7}")
    print("-" * 90)
    for task in sorted(per_task):
        it = per_task[task]
        print(f"{task[:52]:<52} {agg(it,'hand'):>6.3f} {agg(it,'obj'):>6.3f} "
              f"{agg(it,'contact'):>6.3f} "
              f"{np.mean([r['obj'] < 1e-4 for r in it]):>8.2f} "
              f"{np.mean([r['obj'] > args.full_thresh for r in it]):>7.2f}")

    o_empty = float(np.mean([r["obj"] < 1e-4 for r in rows]))
    o_full = float(np.mean([r["obj"] > args.full_thresh for r in rows]))
    h_empty = float(np.mean([r["hand"] < 1e-4 for r in rows]))
    summary = {
        "n_frames": len(rows), "layout": layout,
        "hand_cov": agg(rows, "hand"), "obj_cov": agg(rows, "obj"),
        "contact_cov": agg(rows, "contact"),
        "hand_empty_rate": h_empty, "obj_empty_rate": o_empty, "obj_full_rate": o_full,
    }
    print("-" * 90)
    print(f"{'OVERALL':<52} {summary['hand_cov']:>6.3f} {summary['obj_cov']:>6.3f} "
          f"{summary['contact_cov']:>6.3f} {o_empty:>8.2f} {o_full:>7.2f}")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\nVerdict:")
    bad = []
    if o_empty > 0.3:
        bad.append(f"object prompt empty on {o_empty:.0%} of frames")
    if o_full > 0.3:
        bad.append(f"object prompt covers >{args.full_thresh:.0%} of frame on {o_full:.0%}")
    if h_empty > 0.3:
        bad.append(f"hand prompt empty on {h_empty:.0%} of frames")
    if summary["contact_cov"] < 1e-3:
        bad.append("contact band is essentially empty -> fg weighting is a no-op")
    if bad:
        print("  SAM3 masks look unreliable here:")
        for b in bad:
            print(f"    - {b}")
        print("  Look at the overlays before spending GPU-hours on the full run.")
    else:
        print("  Stats look sane. Confirm by eye on the overlays, then run the pilot.")


if __name__ == "__main__":
    main()
