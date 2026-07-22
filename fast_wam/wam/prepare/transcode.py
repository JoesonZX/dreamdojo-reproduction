#!/usr/bin/env python
"""
Transcode the Ego-Exo4D streams listed in ``raw_index.json`` into the compact,
random-access-friendly tree the LAM trains on, then write ``index.json``.

Each camera becomes a 320x240 **all-intra** h264 file (``-g 1``), which is what
makes ``decord``'s random frame access cheap -- the training loop pulls two
arbitrary frames per sample, so a normal GOP structure would dominate runtime.
Frames are centre-cropped to 4:3 before scaling so ego (square-ish Aria) and exo
(16:9 GoPro) end up in the same geometry.

    python fast_wam/wam/prepare/transcode.py \
        --raw_index /data/egoexo4d/raw_index.json \
        --out       /data/egoexo4d_mv240 \
        --jobs 16

Held-out camera: the highest exo slot of each take is reserved for evaluation
and recorded as ``heldout_cam``. It is still transcoded (eval needs it) but the
training dataset filters it out.
"""

import argparse
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

FFMPEG_VF = (
    "crop=w='if(gte(iw/ih,4/3),ih*4/3,iw)':h='if(gte(iw/ih,4/3),ih,iw*3/4)',"
    "scale=320:240"
)


def probe_frames(path: Path) -> int:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
             "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=600,
        ).stdout.strip()
        return int(out)
    except Exception:
        return 0


def transcode_one(job) -> tuple:
    uid, cam, src, dst = job
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        r = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
             "-vf", FFMPEG_VF, "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "20", "-g", "1", "-an", "-sn", "-vsync", "0", str(dst)],
            capture_output=True, text=True,
        )
        if r.returncode != 0 or not dst.exists():
            return uid, cam, 0, r.stderr[-200:]
    return uid, cam, probe_frames(dst), ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--skip_ego", action="store_true",
                    help="do not transcode the Aria stream (Step A: exo-exo only)")
    args = ap.parse_args()

    with open(args.raw_index) as f:
        raw = json.load(f)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    jobs = []
    for t in raw["takes"]:
        for cam, src in t["cams"].items():
            if args.skip_ego and cam == "ego":
                continue
            jobs.append((t["uid"], cam, src, str(out_root / t["uid"] / f"{cam}.mp4")))
    print(f"{len(jobs)} streams to transcode -> {out_root} (jobs={args.jobs})")

    counts: dict = {}
    failures = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(transcode_one, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs)):
            uid, cam, n, err = fut.result()
            counts.setdefault(uid, {})[cam] = n
            if n == 0:
                failures.append((uid, cam, err))
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(jobs)} done, {len(failures)} failed")

    takes_out = []
    for t in raw["takes"]:
        cams = counts.get(t["uid"], {})
        good = {c: n for c, n in cams.items() if n > 0}
        if not good:
            continue
        # All views of a take are frame-aligned; use the common length so a
        # frame index is valid in every stream.
        n_frames = min(good.values())
        exo = sorted([c for c in good if c.startswith("exo")],
                     key=lambda c: int(c[3:]))
        # Reserve the last exo slot for evaluation only if >=2 remain for training.
        heldout = exo[-1] if len(exo) >= 3 else None
        train_cams = [c for c in good if c != heldout]
        takes_out.append({
            "uid": t["uid"],
            "take_name": t["take_name"],
            "scenario": t["scenario"],
            "n_frames": int(n_frames),
            "cams": sorted(good),
            "train_cams": sorted(train_cams),
            "heldout_cam": heldout,
            "has_hand": False,          # set by hand_actions.py
        })

    index = {"img_h": 240, "img_w": 320, "takes": takes_out}
    with open(out_root / "index.json", "w") as f:
        json.dump(index, f, indent=1)

    n_held = sum(1 for t in takes_out if t["heldout_cam"])
    print(f"\nwrote {out_root/'index.json'}: {len(takes_out)} takes, "
          f"{n_held} with a held-out camera")
    if failures:
        print(f"[warn] {len(failures)} streams failed, first 5:")
        for f_ in failures[:5]:
            print("   ", f_)


if __name__ == "__main__":
    main()
