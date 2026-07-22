#!/usr/bin/env python
"""
Turn Ego-Exo4D 3D hand-pose annotations into per-frame SE(3) wrist poses, so the
dataset can compute the same 18-D two-hand delta action used elsewhere.

This is the "egocentric latent-action annotation" step: **no human labelling** --
Ego-Exo4D already ships 3D hand keypoints (340K manual + 21M automatic 3D
annotations); this script only converts keypoints -> a wrist frame -> a motion
vector.

Per hand we build an SE(3) frame from three keypoints:
    origin = wrist
    x      = normalize(index_mcp - wrist)
    z      = normalize(x  x  (pinky_mcp - wrist))
    y      = z  x  x
and write ``<processed>/<take_uid>/hand_poses.npz`` with ``left``/``right``
arrays of shape (n_frames, 4, 4) plus a ``valid`` mask. Gaps are forward/backward
filled so every frame index is addressable; the mask records which frames were
actually measured.

⚠️ The exact annotation JSON schema differs between the manual and automatic
releases. RUN ``--inspect`` FIRST on a real file and confirm the printed key
names before batch-processing:

    python fast_wam/wam/prepare/hand_actions.py --inspect \
        --annotations /data/egoexo4d/annotations --processed /data/egoexo4d_mv240

Then:

    python fast_wam/wam/prepare/hand_actions.py \
        --annotations /data/egoexo4d/annotations \
        --processed   /data/egoexo4d_mv240 \
        --prefer automatic
"""

import argparse
import json
from pathlib import Path

import numpy as np

# Candidate keypoint names, in preference order, for each role.
KEY_ALIASES = {
    "wrist":     ["{h}_wrist", "{h}_hand_wrist", "wrist"],
    "index_mcp": ["{h}_index_1", "{h}_index_mcp", "{h}_index_finger_mcp", "index_1"],
    "pinky_mcp": ["{h}_pinky_1", "{h}_pinky_mcp", "{h}_pinky_finger_mcp", "pinky_1"],
}


def _find_ann_files(ann_root: Path, prefer: str) -> dict:
    """take_uid -> json path, searching ego_pose/*/hand/{prefer,...}."""
    order = [prefer] + [s for s in ("annotation", "automatic") if s != prefer]
    found = {}
    for sub in order:
        for p in ann_root.glob(f"ego_pose/*/hand/{sub}/*.json"):
            found.setdefault(p.stem, p)
    return found


def _get_xyz(node) -> np.ndarray | None:
    if node is None:
        return None
    if isinstance(node, dict):
        if all(k in node for k in ("x", "y", "z")):
            return np.array([node["x"], node["y"], node["z"]], dtype=np.float64)
        return None
    if isinstance(node, (list, tuple)) and len(node) >= 3:
        return np.array(node[:3], dtype=np.float64)
    return None


def _lookup(kps: dict, role: str, hand: str) -> np.ndarray | None:
    for pat in KEY_ALIASES[role]:
        v = _get_xyz(kps.get(pat.format(h=hand)))
        if v is not None:
            return v
    return None


def _frame_kps(entry) -> dict:
    """Normalise one frame's annotation payload into {keypoint_name: node}."""
    if isinstance(entry, list):
        entry = entry[0] if entry else {}
    if not isinstance(entry, dict):
        return {}
    for key in ("annotation3D", "annotation_3d", "3d", "keypoints_3d"):
        if key in entry and isinstance(entry[key], dict):
            return entry[key]
    # Already flat?
    if any(isinstance(v, dict) and "x" in v for v in entry.values()):
        return entry
    return {}


def _frame_to_se3(kps: dict, hand: str) -> np.ndarray | None:
    w = _lookup(kps, "wrist", hand)
    i = _lookup(kps, "index_mcp", hand)
    p = _lookup(kps, "pinky_mcp", hand)
    if w is None:
        return None
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = w
    if i is None or p is None:
        return T                                    # translation only
    x = i - w
    nx = np.linalg.norm(x)
    if nx < 1e-8:
        return T
    x /= nx
    z = np.cross(x, p - w)
    nz = np.linalg.norm(z)
    if nz < 1e-8:
        return T
    z /= nz
    y = np.cross(z, x)
    T[:3, :3] = np.stack([x, y, z], axis=1)
    return T


def _fill_gaps(poses: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Forward then backward fill invalid frames so every index is addressable."""
    if not valid.any():
        return poses
    idx = np.where(valid)[0]
    first, last = idx[0], idx[-1]
    for t in range(first + 1, len(poses)):
        if not valid[t]:
            poses[t] = poses[t - 1]
    for t in range(first - 1, -1, -1):
        poses[t] = poses[t + 1]
    for t in range(last + 1, len(poses)):
        poses[t] = poses[last]
    return poses


def process_take(ann_path: Path, n_frames: int):
    with open(ann_path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "frames" in data:
        data = data["frames"]
    if not isinstance(data, dict):
        return None

    out = {}
    for hand in ("left", "right"):
        poses = np.tile(np.eye(4, dtype=np.float64), (n_frames, 1, 1))
        valid = np.zeros(n_frames, dtype=bool)
        for k, entry in data.items():
            try:
                t = int(k)
            except (TypeError, ValueError):
                continue
            if not (0 <= t < n_frames):
                continue
            T = _frame_to_se3(_frame_kps(entry), hand)
            if T is not None:
                poses[t] = T
                valid[t] = True
        out[hand] = (_fill_gaps(poses, valid), valid)
    coverage = float((out["left"][1] | out["right"][1]).mean())
    return out, coverage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True, help="<egoexo_root>/annotations")
    ap.add_argument("--processed", required=True, help="output of transcode.py")
    ap.add_argument("--prefer", default="automatic", choices=["automatic", "annotation"])
    ap.add_argument("--min_coverage", type=float, default=0.3,
                    help="takes with less annotated-frame coverage are not marked has_hand")
    ap.add_argument("--inspect", action="store_true")
    args = ap.parse_args()

    ann_root = Path(args.annotations)
    proc = Path(args.processed)
    files = _find_ann_files(ann_root, args.prefer)
    print(f"found {len(files)} hand-annotation files under {ann_root}")

    if args.inspect:
        for uid, p in list(files.items())[:2]:
            print(f"\n=== {p} ===")
            with open(p) as f:
                d = json.load(f)
            if isinstance(d, dict) and "frames" in d:
                d = d["frames"]
            keys = list(d)[:3] if isinstance(d, dict) else []
            print(f"  top-level type={type(d).__name__} n={len(d)} first keys={keys}")
            for k in keys[:1]:
                kps = _frame_kps(d[k])
                print(f"  frame {k}: {len(kps)} keypoints")
                for name in list(kps)[:24]:
                    print(f"     {name}")
                for hand in ("left", "right"):
                    print(f"   -> {hand} SE(3) resolvable: "
                          f"{_frame_to_se3(kps, hand) is not None}")
        return

    with open(proc / "index.json") as f:
        index = json.load(f)

    n_ok = 0
    for t in index["takes"]:
        p = files.get(t["uid"])
        if p is None:
            continue
        res = process_take(p, t["n_frames"])
        if res is None:
            continue
        hands, coverage = res
        if coverage < args.min_coverage:
            continue
        np.savez_compressed(
            proc / t["uid"] / "hand_poses.npz",
            left=hands["left"][0].astype(np.float32),
            right=hands["right"][0].astype(np.float32),
            valid_left=hands["left"][1],
            valid_right=hands["right"][1],
        )
        t["has_hand"] = True
        t["hand_coverage"] = round(coverage, 3)
        n_ok += 1

    with open(proc / "index.json", "w") as f:
        json.dump(index, f, indent=1)
    print(f"wrote hand poses for {n_ok}/{len(index['takes'])} takes; index.json updated")


if __name__ == "__main__":
    main()
