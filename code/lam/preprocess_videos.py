"""
AgiBot-World Alpha preprocessing: extract tar archives + transcode AV1 → h264.

AgiBot distributes data as .tar files containing AV1-encoded MP4s.
OpenCV/decord cannot decode AV1 on this server (no hw accel, outdated ffmpeg bindings).
This script:
  1. Extracts each .tar to <out_root>/{task_id}/{ep_id}/
  2. Transcodes every AV1 MP4 to h264 in-place (overwrites the AV1 file)

Output layout:
  <out_root>/{task_id}/{ep_id}/videos/head_color.mp4        ← h264, readable by OpenCV
  <out_root>/{task_id}/{ep_id}/videos/hand_left_color.mp4   ← h264
  ...

Usage:
  # Extract + transcode task 410 (the small 198MB test tar):
  python preprocess_videos.py \\
      --tar_root  /home/xuan/embodied-ai/data/agibotworld/observations \\
      --out_root  /home/xuan/embodied-ai/data/agibotworld/extracted \\
      --camera    head_color \\
      --n_workers 4

  # Include specific tasks only:
  python preprocess_videos.py \\
      --tar_root  /home/xuan/embodied-ai/data/agibotworld/observations \\
      --out_root  /home/xuan/embodied-ai/data/agibotworld/extracted \\
      --tasks 410 327 352

  # Skip already-done tasks (safe to re-run):
  python preprocess_videos.py ...   # same command, already-done eps are skipped
"""

import argparse
import os
import subprocess
import tarfile
from pathlib import Path

from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────

def transcode_to_h264(src: Path) -> bool:
    """Transcode AV1 MP4 to h264 in-place. Returns True on success."""
    tmp = src.with_suffix(".tmp.mp4")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src),
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-an",          # drop audio (not needed for LAM)
        str(tmp)
    ]
    ret = subprocess.run(cmd, capture_output=True)
    if ret.returncode != 0:
        tmp.unlink(missing_ok=True)
        return False
    tmp.rename(src)
    return True


def process_tar(tar_path: Path, out_root: Path, camera: str) -> dict:
    """
    Extract one .tar and transcode the target camera video.
    Returns {"ok": n_ok, "skip": n_skip, "fail": n_fail}.
    """
    task_id = tar_path.parent.name   # e.g. "410"
    ep_out_base = out_root / task_id

    counts = {"ok": 0, "skip": 0, "fail": 0}

    with tarfile.open(tar_path) as tf:
        # List episodes in this tar (top-level dirs)
        ep_ids = {m.name.split("/")[0] for m in tf.getmembers()
                  if "/" in m.name and not m.name.startswith(".")}

        for ep_id in sorted(ep_ids):
            ep_out = ep_out_base / ep_id
            marker = ep_out / ".done"

            if marker.exists():
                counts["skip"] += 1
                continue

            # Extract just this episode
            ep_out.mkdir(parents=True, exist_ok=True)
            members = [m for m in tf.getmembers()
                       if m.name.startswith(f"{ep_id}/")]
            tf.extractall(path=ep_out_base, members=members)

            # Transcode target camera video
            vid = ep_out / "videos" / f"{camera}.mp4"
            if vid.exists():
                ok = transcode_to_h264(vid)
                if not ok:
                    counts["fail"] += 1
                    continue
            else:
                # Camera not present in this episode — still mark done
                pass

            marker.touch()
            counts["ok"] += 1

    return counts


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tar_root",  required=True,
                        help="Directory containing observations/{task_id}/*.tar")
    parser.add_argument("--out_root",  required=True,
                        help="Output root for extracted + transcoded episodes")
    parser.add_argument("--camera",    default="head_color",
                        help="Which camera to transcode (others are left as-is)")
    parser.add_argument("--tasks",     nargs="*",
                        help="Limit to specific task IDs (e.g. --tasks 410 327)")
    parser.add_argument("--n_workers", type=int, default=2,
                        help="Parallel tar workers (each spawns ffmpeg subprocesses)")
    args = parser.parse_args()

    tar_root = Path(args.tar_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Collect all tar files
    all_tars = sorted(tar_root.glob("*/*.tar"))
    if args.tasks:
        all_tars = [t for t in all_tars if t.parent.name in args.tasks]

    if not all_tars:
        print(f"No .tar files found under {tar_root}. Check the path.")
        return

    print(f"Found {len(all_tars)} tar files to process.")

    total = {"ok": 0, "skip": 0, "fail": 0}
    for tar_path in tqdm(all_tars, desc="Processing tars"):
        c = process_tar(tar_path, out_root, args.camera)
        for k in total:
            total[k] += c[k]

    print(f"\nDone.  OK: {total['ok']}  Skipped: {total['skip']}  Failed: {total['fail']}")
    print(f"Extracted episodes → {out_root}")


if __name__ == "__main__":
    main()
