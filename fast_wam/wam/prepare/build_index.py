#!/usr/bin/env python
"""
Scan a downloaded Ego-Exo4D tree and emit ``raw_index.json``.

Ego-Exo4D layout (after ``egoexo -o <root> --parts downscaled_takes/448``):

    <root>/takes.json
    <root>/takes/<take_name>/frame_aligned_videos/downscaled/448/
            aria01_214-1.mp4      <- ego (Project Aria RGB)
            cam01.mp4 cam02.mp4 …  <- exo (static GoPro)

Camera file names vary across universities, so classification is by pattern:
anything containing "aria" is ego, everything else is treated as exo and sorted
by name to give stable slot indices.

    python fast_wam/wam/prepare/build_index.py \
        --egoexo_root /data/egoexo4d \
        --out         /data/egoexo4d/raw_index.json \
        --scenarios cooking "bike repair" health \
        --min_exo 3

Run with ``--inspect`` first: it prints the directory layout it actually found
for a few takes, so you can confirm the globbing before committing to a batch.
"""

import argparse
import json
import re
from pathlib import Path

VIDEO_EXTS = (".mp4", ".MP4")


def find_video_dir(take_dir: Path) -> Path | None:
    """Prefer the 448px downscaled variant; fall back to full-res."""
    for rel in ("frame_aligned_videos/downscaled/448",
                "frame_aligned_videos/downscaled",
                "frame_aligned_videos"):
        d = take_dir / rel
        if d.is_dir() and any(p.suffix in VIDEO_EXTS for p in d.iterdir()):
            return d
    return None


def classify(video_paths: list[Path]) -> tuple[Path | None, list[Path]]:
    ego, exo = None, []
    for p in sorted(video_paths, key=lambda x: x.name):
        name = p.name.lower()
        if "aria" in name:
            # Some takes ship several Aria streams (rgb + slam). Keep the RGB one.
            if ego is None or "214-1" in name:
                ego = p
        elif re.match(r"^(cam|gp|gopro)", name):
            exo.append(p)
    return ego, exo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--egoexo_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenarios", nargs="*", default=None,
                    help="substring filter on the take's task/scenario name")
    ap.add_argument("--min_exo", type=int, default=3,
                    help="takes with fewer exo cameras are dropped (need >=3 so "
                         "one can be held out and >=2 remain for cross-view)")
    ap.add_argument("--max_takes", type=int, default=0, help="0 = no limit")
    ap.add_argument("--inspect", action="store_true")
    args = ap.parse_args()

    root = Path(args.egoexo_root)
    takes_dir = root / "takes"
    if not takes_dir.is_dir():
        raise SystemExit(f"{takes_dir} not found -- is --egoexo_root correct?")

    # takes.json gives uid + task name; fall back to directory names if absent.
    meta_by_name = {}
    takes_json = root / "takes.json"
    if takes_json.exists():
        with open(takes_json) as f:
            for t in json.load(f):
                meta_by_name[t.get("take_name", t.get("root_dir", ""))] = t
    else:
        print(f"[warn] {takes_json} missing -- uid will fall back to the dir name")

    if args.inspect:
        for d in sorted(takes_dir.iterdir())[:3]:
            print(f"\n=== {d.name} ===")
            vd = find_video_dir(d)
            print(f"  video dir: {vd}")
            if vd:
                for p in sorted(vd.iterdir())[:12]:
                    print(f"    {p.name}")
            m = meta_by_name.get(d.name, {})
            print(f"  meta keys: {sorted(m)[:12]}")
            print(f"  task_name: {m.get('task_name')}  parent: {m.get('parent_task_name')}")
        return

    entries, skipped = [], {"no_video_dir": 0, "too_few_exo": 0, "scenario": 0}
    for take_dir in sorted(takes_dir.iterdir()):
        if not take_dir.is_dir():
            continue
        meta = meta_by_name.get(take_dir.name, {})
        scenario = str(meta.get("parent_task_name") or meta.get("task_name") or take_dir.name)

        if args.scenarios:
            if not any(s.lower() in scenario.lower() for s in args.scenarios):
                skipped["scenario"] += 1
                continue

        vd = find_video_dir(take_dir)
        if vd is None:
            skipped["no_video_dir"] += 1
            continue
        ego, exo = classify([p for p in vd.iterdir() if p.suffix in VIDEO_EXTS])
        if len(exo) < args.min_exo:
            skipped["too_few_exo"] += 1
            continue

        cams = {}
        if ego is not None:
            cams["ego"] = str(ego)
        for i, p in enumerate(exo):
            cams[f"exo{i}"] = str(p)

        entries.append({
            "uid": meta.get("take_uid", take_dir.name),
            "take_name": take_dir.name,
            "scenario": scenario,
            "cams": cams,
            "n_exo": len(exo),
            "has_ego": ego is not None,
        })
        if args.max_takes and len(entries) >= args.max_takes:
            break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"egoexo_root": str(root), "takes": entries}, f, indent=1)

    n_exo = sum(e["n_exo"] for e in entries)
    print(f"kept {len(entries)} takes ({n_exo} exo streams, "
          f"{sum(e['has_ego'] for e in entries)} with ego)")
    print(f"skipped: {skipped}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
