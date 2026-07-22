#!/usr/bin/env python
"""
Evaluate a trained multi-view LAM.

Two questions, both answered on the **held-out camera** that no arm ever
trained on, so the comparison is fair to every arm:

  1. Cross-view agreement -- does the same physical transition map to the same
     latent action when seen from a camera the model never trained on?
       * median cosine( z(v), z(v*) )
       * the same statistic over MISMATCHED transitions = the null. Agreement is
         meaningless without it: high-dimensional Gaussians are not orthogonal
         once the latent has any dominant direction.

  2. Probe transfer -- fit a ridge z -> wrist-delta action on the training view,
     then read it off the held-out view. The R^2 drop is how much of the action
     information is view-specific.

Usage:
    python fast_wam/wam/eval_crossview.py \
        --checkpoint fast_wam/checkpoints/mv_cross/step_0010000 \
        --config     fast_wam/wam/config/mv_cross.yaml \
        --n_samples  4000 \
        --out        fast_wam/results/mv_cross.json
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import (  # noqa: E402
    EgoExoMultiViewDataset, pair_action, split_take_uids, view_slot_id,
)
from model import MultiViewLAM  # noqa: E402


def load_model(checkpoint: str, cfg: dict, device: str) -> MultiViewLAM:
    m = cfg["model"]
    model = MultiViewLAM(
        in_dim=m.get("image_channels", 3),
        model_dim=m["model_dim"],
        latent_dim=m["latent_dim"],
        patch_size=m["patch_size"],
        enc_blocks=m["enc_blocks"],
        dec_blocks=m["dec_blocks"],
        num_heads=m["num_heads"],
        dropout=0.0,
        num_views=m.get("num_views", 0),
    )
    ckpt = Path(checkpoint)
    sd_path = ckpt / "model.safetensors"
    if sd_path.exists():
        from safetensors.torch import load_file
        state = load_file(str(sd_path))
    else:
        state = torch.load(ckpt / "pytorch_model.bin", map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"checkpoint is missing {len(missing)} tensors, e.g. {missing[:5]}")
    if unexpected:
        print(f"[load] ignored {len(unexpected)} extra tensors, e.g. {unexpected[:5]}")
    return model.to(device).eval()


def ridge_fit(X: np.ndarray, Y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Closed-form ridge with intercept. X:(N,D) Y:(N,K) -> W:(D+1,K)."""
    Xb = np.concatenate([X, np.ones((len(X), 1), dtype=X.dtype)], axis=1)
    A = Xb.T @ Xb
    A[:-1, :-1] += alpha * np.eye(X.shape[1], dtype=X.dtype)   # don't penalise intercept
    return np.linalg.solve(A, Xb.T @ Y)


def r2_score(Y: np.ndarray, Yhat: np.ndarray) -> float:
    ss_res = float(((Y - Yhat) ** 2).sum())
    ss_tot = float(((Y - Y.mean(0)) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def apply_ridge(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    Xb = np.concatenate([X, np.ones((len(X), 1), dtype=X.dtype)], axis=1)
    return Xb @ W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--n_samples", type=int, default=4000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    dcfg = cfg["data"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, cfg, device)

    root = Path(dcfg["processed_root"])
    with open(root / "index.json") as f:
        index = json.load(f)

    # Same split as training -- one shared function, no drift.
    max_skip = max(dcfg.get("downsample_factors", [1, 2]))
    takes = [t for t in index["takes"] if t["n_frames"] >= max(16, max_skip + 2)]
    _, val_uids = split_take_uids(
        [t["uid"] for t in takes], dcfg.get("val_ratio", 0.05), int(dcfg.get("split_seed", 42))
    )
    val_takes = [t for t in takes if t["uid"] in val_uids and t.get("heldout_cam")]
    if not val_takes:
        raise RuntimeError("no val take has a heldout_cam; re-run prepare/build_index.py")

    # A throwaway dataset instance only to reuse its frame IO + action stats.
    io = EgoExoMultiViewDataset(
        processed_root=str(root), mode="single",
        view_pairs=dcfg.get("view_pairs", "exo_exo"), split="val",
        val_ratio=dcfg.get("val_ratio", 0.05), seed=int(dcfg.get("split_seed", 42)),
        img_h=dcfg.get("img_h", 240), img_w=dcfg.get("img_w", 320),
        downsample_factors=tuple(dcfg.get("downsample_factors", [1, 2])),
        load_actions=False,
    )

    rng = random.Random(args.seed)
    # Build the sample list: each entry is the SAME transition seen from the
    # training view and from the held-out view.
    plan = []
    for _ in range(args.n_samples):
        t_meta = rng.choice(val_takes)
        cams = [c for c in t_meta["train_cams"] if c != t_meta["heldout_cam"]]
        if dcfg.get("view_pairs", "exo_exo") == "exo_exo":
            cams = [c for c in cams if c != "ego"]
        if not cams:
            continue
        skip = rng.choice(tuple(dcfg.get("downsample_factors", [1, 2])))
        t = rng.randint(0, max(0, t_meta["n_frames"] - skip - 2))
        plan.append((t_meta, rng.choice(cams), t_meta["heldout_cam"], t, skip))

    z_train, z_held, actions = [], [], []
    have_actions = True
    buf_a, buf_b, buf_va, buf_vb = [], [], [], []

    def flush():
        if not buf_a:
            return
        with torch.no_grad():
            va = torch.stack(buf_va).to(device)
            vb = torch.stack(buf_vb).to(device)
            za = model.encode(torch.stack(buf_a).to(device))["z_mu"]
            zb = model.encode(torch.stack(buf_b).to(device))["z_mu"]
        z_train.append(za.float().cpu().numpy())
        z_held.append(zb.float().cpu().numpy())
        buf_a.clear(); buf_b.clear(); buf_va.clear(); buf_vb.clear()

    for i, (t_meta, cam_v, cam_star, t, skip) in enumerate(plan):
        d = root / t_meta["uid"]
        try:
            fa = io._load_pair(d / f"{cam_v}.mp4", t, skip)
            fb = io._load_pair(d / f"{cam_star}.mp4", t, skip)
        except Exception:
            continue
        buf_a.append(fa); buf_b.append(fb)
        buf_va.append(torch.tensor(view_slot_id(cam_v)))
        buf_vb.append(torch.tensor(view_slot_id(cam_star)))

        if t_meta.get("has_hand") and have_actions:
            try:
                left, right = io._load_poses(d)
                T = min(len(left), len(right))
                actions.append(pair_action(left, right, min(t, T - skip - 1), skip))
            except Exception:
                have_actions = False
        else:
            have_actions = have_actions and False

        if len(buf_a) >= args.batch_size:
            flush()
        if (i + 1) % 500 == 0:
            print(f"  encoded {i+1}/{len(plan)}")
    flush()

    Za = np.concatenate(z_train, 0)
    Zb = np.concatenate(z_held, 0)
    n = min(len(Za), len(Zb))
    Za, Zb = Za[:n], Zb[:n]
    print(f"[eval] {n} paired transitions, latent dim {Za.shape[1]}")

    # ── 1. cross-view agreement vs mismatched null ────────────────────────────
    def cos_rows(A, B):
        A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
        B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
        return (A * B).sum(1)

    matched = cos_rows(Za, Zb)
    perm = np.random.RandomState(args.seed).permutation(n)
    null = cos_rows(Za, Zb[perm])

    res = {
        "checkpoint": str(args.checkpoint),
        "n_paired": int(n),
        "crossview_cos_median": float(np.median(matched)),
        "crossview_cos_mean": float(matched.mean()),
        "null_cos_median": float(np.median(null)),
        "null_cos_mean": float(null.mean()),
        "crossview_gain": float(np.median(matched) - np.median(null)),
    }

    # ── 2. probe transfer (needs hand actions) ────────────────────────────────
    if have_actions and len(actions) >= n:
        A = np.stack(actions[:n], 0).astype(np.float32)
        A = (A - A.mean(0)) / (A.std(0) + 1e-6)
        idx = np.random.RandomState(args.seed + 1).permutation(n)
        cut = int(n * 0.7)
        tr, te = idx[:cut], idx[cut:]
        W = ridge_fit(Za[tr], A[tr], alpha=1.0)
        res["action_r2_in_view"] = r2_score(A[te], apply_ridge(W, Za[te]))
        res["action_r2_heldout_view"] = r2_score(A[te], apply_ridge(W, Zb[te]))
        res["action_r2_drop"] = res["action_r2_in_view"] - res["action_r2_heldout_view"]
    else:
        res["action_r2_in_view"] = None
        res["action_r2_heldout_view"] = None
        res["action_r2_drop"] = None
        print("[eval] no hand poses available -- skipping the action probe")

    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"[eval] wrote {args.out}")


if __name__ == "__main__":
    main()
