"""Audit whether reversible EgoDex episodes contain mixed local motion.

EgoDex often stores multiple candidate verbs for a task/episode, for example
["insert", "remove"], plus `which_llm_description` to select the performed verb.
That is good enough for episode-level labels, but contrastive training uses
frame pairs. If many local frame-pair actions go opposite to the episode's main
motion direction, video-level verb labels become noisy.

This script computes a lightweight local-motion purity diagnostic from the raw
18D hand-pose action sequence. It does not assign semantic verbs per frame; it
only flags episodes whose local actions are directionally mixed.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np

from dataset import _pair_action


OPPOSITE_VERBS = {
    "add": "remove",
    "assemble": "disassemble",
    "charge": "uncharge",
    "close": "open",
    "disassemble": "assemble",
    "dump": "scoop",
    "fold": "unfold",
    "insert": "remove",
    "lock": "unlock",
    "open": "close",
    "pick": "put",
    "pull": "push",
    "push": "pull",
    "remove": "insert",
    "scoop": "dump",
    "screw": "unscrew",
    "stack": "unstack",
    "stock": "unstock",
    "tie": "untie",
    "uncharge": "charge",
    "unfold": "fold",
    "unlock": "lock",
    "unscrew": "screw",
    "unstack": "stack",
    "unstock": "stock",
    "untie": "tie",
    "unzip": "zip",
    "zip": "unzip",
}


def _resolve_verb(verbs: list[str], which) -> str:
    if not verbs:
        return "unknown"
    if which is not None:
        try:
            idx = int(which) - 1
            if 0 <= idx < len(verbs):
                return verbs[idx]
        except Exception:
            pass
    return verbs[0] if len(verbs) == 1 else "unknown"


def _cosine_rows(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    denom = (np.linalg.norm(x, axis=1) * np.linalg.norm(ref) + 1e-8)
    return (x @ ref) / denom


def _select_motion_dims(actions: np.ndarray, motion_part: str) -> np.ndarray:
    if motion_part == "translation":
        # left dp[0:3] + right dp[9:12]. The 6D rotation representation carries
        # near-identity components, so raw 18D cosine can hide local reversals.
        return actions[:, [0, 1, 2, 9, 10, 11]]
    if motion_part == "left_translation":
        return actions[:, 0:3]
    if motion_part == "right_translation":
        return actions[:, 9:12]
    return actions


def audit_episode(
    path: Path,
    skip: int,
    stride: int,
    min_motion_norm: float,
    motion_part: str,
) -> dict | None:
    with h5py.File(path, "r") as hf:
        verbs = [str(x).strip().lower().replace("_", " ") for x in hf.attrs.get("llm_verbs", [])]
        which = hf.attrs.get("which_llm_description", None)
        left = hf["transforms"]["leftHand"][:].astype(np.float32)
        right = hf["transforms"]["rightHand"][:].astype(np.float32)

    resolved = _resolve_verb(verbs, which)
    has_opposite_candidates = any(OPPOSITE_VERBS.get(v) in verbs for v in verbs)
    if not has_opposite_candidates:
        return None

    T = min(len(left), len(right))
    acts = []
    for t in range(0, T - skip, stride):
        try:
            acts.append(_pair_action(left, right, t, skip))
        except Exception:
            continue
    if len(acts) < 5:
        return None
    acts = np.stack(acts, axis=0)
    acts = _select_motion_dims(acts, motion_part)
    norms = np.linalg.norm(acts, axis=1)
    keep = norms > min_motion_norm
    if keep.sum() < 5:
        return None
    acts = acts[keep]
    norms = norms[keep]
    mean = acts.mean(axis=0)
    mean_norm = float(np.linalg.norm(mean))
    if mean_norm < 1e-8:
        return None
    cos = _cosine_rows(acts, mean)
    return {
        "episode": f"{path.parent.name}/{path.stem}",
        "task": path.parent.name,
        "verbs": ",".join(verbs),
        "resolved_verb": resolved,
        "which_llm_description": "" if which is None else str(which),
        "motion_part": motion_part,
        "n_pairs": int(len(acts)),
        "mean_action_norm": mean_norm,
        "median_pair_norm": float(np.median(norms)),
        "opposite_local_frac": float((cos < -0.2).mean()),
        "weak_or_opposite_local_frac": float((cos < 0.2).mean()),
        "median_cos_to_episode_mean": float(np.median(cos)),
        "p10_cos_to_episode_mean": float(np.quantile(cos, 0.10)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/home/xuan/embodied-ai/data/egodex/test_240p")
    parser.add_argument("--out", default="results/reversible_episode_audit.csv")
    parser.add_argument("--skip", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--min_motion_norm", type=float, default=1e-4)
    parser.add_argument("--motion_part", default="translation",
                        choices=["translation", "left_translation", "right_translation", "all18"])
    args = parser.parse_args()

    rows = []
    for path in sorted(Path(args.data_root).glob("*/*.hdf5")):
        row = audit_episode(path, args.skip, args.stride, args.min_motion_norm, args.motion_part)
        if row is not None:
            rows.append(row)

    rows.sort(key=lambda r: (r["opposite_local_frac"], r["weak_or_opposite_local_frac"]), reverse=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "episode", "task", "verbs", "resolved_verb", "which_llm_description", "motion_part",
        "n_pairs", "mean_action_norm", "median_pair_norm",
        "opposite_local_frac", "weak_or_opposite_local_frac",
        "median_cos_to_episode_mean", "p10_cos_to_episode_mean",
    ]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"audited reversible episodes: {len(rows)}")
    print(f"wrote {out}")
    print("top ambiguous episodes:")
    for row in rows[:20]:
        print(
            f"{row['episode']:60s} resolved={row['resolved_verb']:12s} "
            f"opp={row['opposite_local_frac']:.3f} weak={row['weak_or_opposite_local_frac']:.3f} "
            f"p10cos={row['p10_cos_to_episode_mean']:.3f}"
        )


if __name__ == "__main__":
    main()
