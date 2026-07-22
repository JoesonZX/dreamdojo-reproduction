"""Phase 1 — Tier-1.5 编排(prereg §3;within-LAM,Eval-A,Eval-B 密封)。

对 A0/A1 两 seed 的原 LAM decoder,在同一批 Eval-A clip 上做 K_max=16 步自回归 own-latent
rollout(latent 变体 own/zero/neg),读出 horizon={4,8,16} 的 fidelity / 生成-use / 符号-use /
运动比。**within-LAM functional validation,不作下游 WM 主张。**

    CUDA_VISIBLE_DEVICES=<free> python -u code/eval_tier15.py --device cuda:0 --seeds 42 1 --n_clips 200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np                                                          # noqa: E402
import torch                                                                # noqa: E402

import _apath  # noqa: F401  # shared A (lam_cdlam_optimization) substrate on sys.path
from benchmark_lam import RUN_SPECS, _load_cfg, _resolve                    # noqa: E402
from dataset import EgoDexDataset                                           # noqa: E402
from eval import load_model                                                 # noqa: E402
from rollout_tier15 import sample_clips, load_clip, encode_seq, rollout, rollout_metrics  # noqa: E402

MANIFEST = "/home/xuan/embodied-ai/lam_condition_utilization/data/eval_manifest.json"
SEED_ARMS = {"42": ("sb_A0", "sb_A1"), "1": ("sb_A0_s1", "sb_A1_s1")}
HORIZONS = [4, 8, 16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", default=["42", "1"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_clips", type=int, default=200)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--skip", type=int, default=2)
    ap.add_argument("--clip_seed", type=int, default=42)
    ap.add_argument("--rollout_bs", type=int, default=16)
    ap.add_argument("--out", default="results/stage_b_tier15.json")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    spec_map = {s.name: s for s in RUN_SPECS}
    horizons = [h for h in HORIZONS if h <= args.K]

    # dataset on the frozen Eval-A manifest (encoder never trained on it)
    cfg = _load_cfg(_resolve(spec_map["sb_A0"].config))
    d = cfg["data"]
    ds = EgoDexDataset(data_root=d["data_root"], img_h=d.get("img_h", 240), img_w=d.get("img_w", 320),
                       downsample_factors=(1, 2), split="train", val_ratio=d.get("val_ratio", 0.1),
                       seed=d.get("seed", 42), load_verbs=True, load_actions=True, action_dim=18,
                       exclude_manifest=MANIFEST, only_manifest=True)

    clips = sample_clips(ds, args.n_clips, args.K, args.skip, seed=args.clip_seed)
    n_rev = sum(c["reversible"] for c in clips)
    print(f"clips={len(clips)}  reversible={n_rev}  K={args.K} skip={args.skip}")

    # load all clip frames once (model-independent), shared across arms
    frames_list, actions_list = [], []
    for j, c in enumerate(clips):
        cl = load_clip(ds, c)
        frames_list.append(cl["frames"]); actions_list.append(cl["actions"])
        if (j + 1) % 50 == 0:
            print(f"  loaded {j+1}/{len(clips)} clips")
    frames_all = torch.stack(frames_list, dim=0)                            # [N,K+1,H,W,C]
    f0 = frames_all[:, 0]                                                   # [N,H,W,C]
    print(f"frames_all {tuple(frames_all.shape)}")

    out = {"config": {"n_clips": len(clips), "n_reversible": int(n_rev), "K": args.K,
                      "skip": args.skip, "clip_seed": args.clip_seed, "horizons": horizons,
                      "manifest": "Eval-A (within-LAM functional validation; Eval-B sealed)"},
           "seeds": {}}

    for seed in args.seeds:
        a0_name, a1_name = SEED_ARMS[seed]
        print(f"\n{'='*66}\nSEED {seed}:  A0={a0_name}  A1={a1_name}\n{'='*66}")
        out["seeds"][seed] = {}
        for tag, arm in [("A0", a0_name), ("A1", a1_name)]:
            spec = spec_map[arm]
            acfg = _load_cfg(_resolve(spec.config))
            acfg["data"]["exclude_manifest"] = MANIFEST; acfg["data"]["only_manifest"] = True
            model = load_model(str(_resolve(spec.checkpoint)), acfg, device); model.eval()
            ap_dim = model.action_part

            # per-clip pairwise latent sequence
            zseq = torch.stack([encode_seq(model, fr, device) for fr in frames_list], dim=0)  # [N,K,lat]
            z_zero = zseq.clone(); z_zero[..., :ap_dim] = 0.0
            z_neg = zseq.clone(); z_neg[..., :ap_dim] = -z_neg[..., :ap_dim]

            r_own = rollout(model, f0, zseq, device, args.rollout_bs)
            r_zero = rollout(model, f0, z_zero, device, args.rollout_bs)
            r_neg = rollout(model, f0, z_neg, device, args.rollout_bs)
            m = rollout_metrics(frames_all, r_own, r_zero, r_neg)
            out["seeds"][seed][tag] = m

            def at(name):
                return {k: round(m[name][k - 1], 4) for k in horizons}
            print(f"  {tag} ({arm}):")
            print(f"    PSNR_own   @{horizons} = {at('psnr_own')}")
            print(f"    gen_use    @{horizons} = {at('gen_use_psnr')}   (own−zero, dB)")
            print(f"    sign_use   @{horizons} = {at('sign_use_psnr')}   (own−neg, dB)")
            print(f"    motion_rat @{horizons} = {at('motion_ratio')}   (out/GT motion)")
            del model
            torch.cuda.empty_cache()

        # A0-vs-A1 compact contrast
        a0, a1 = out["seeds"][seed]["A0"], out["seeds"][seed]["A1"]
        print(f"\n  [A0−A1 horizon contrast]")
        for name, lab in [("psnr_own", "ΔPSNR_own"), ("gen_use_psnr", "Δgen_use"),
                          ("sign_use_psnr", "Δsign_use"), ("motion_ratio", "Δmotion_ratio")]:
            vals = {k: round(a0[name][k - 1] - a1[name][k - 1], 4) for k in horizons}
            print(f"    {lab:16} @{horizons} = {vals}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
