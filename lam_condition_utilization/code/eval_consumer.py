"""Phase 2 route-C 评测:把训练好的 consumer 在给定 manifest 上做 K 步 rollout + 光流 following。

⚠️ **Eval-B 解封是单独一步,需用户显式 go。** 本脚本本身不区分 dev / Eval-B——由 --manifest 决定;
默认给 dev(consumer-train 池的 held-out clip),**不要**在未获授权时传 data/eval_manifest_B.json。

A0-vs-A1 对比:分别对两个 arm 的 consumer 跑,比 target 相对 shuffle/antiparallel 的 following gap。

    python -u code/eval_consumer.py --consumer_ckpt results/consumer_zmu_sb_A0_s0.pt --cond zmu --arm sb_A0 \
        --manifest <dev-or-sealed> --K 8
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
from consumer_model import ConditionalUNet                                  # noqa: E402
from rollout_tier15 import sample_clips, encode_seq                        # noqa: E402
from train_consumer import cache_clips, dev_gate                           # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--consumer_ckpt", required=True)
    ap.add_argument("--cond", choices=["gt_action", "zmu"], required=True)
    ap.add_argument("--arm", default=None, help="encoder arm for zmu cond")
    ap.add_argument("--manifest", required=True, help="episodes to eval on (dev subset or, once unsealed, Eval-B)")
    ap.add_argument("--only_manifest", action="store_true", default=True)
    ap.add_argument("--n_clips", type=int, default=120)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--skip", type=int, default=2)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--sample_seed", type=int, default=999)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if "eval_manifest_B" in args.manifest:
        print("⚠️  You are evaluating on the SEALED Eval-B manifest. This is the one-time route-C unseal.")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    spec_map = {s.name: s for s in RUN_SPECS}
    cfg = _load_cfg(_resolve(spec_map["sb_A0"].config)); d = cfg["data"]
    ds = EgoDexDataset(data_root=d["data_root"], img_h=d.get("img_h", 240), img_w=d.get("img_w", 320),
                       downsample_factors=(1, 2), split="train", val_ratio=d.get("val_ratio", 0.1),
                       seed=d.get("seed", 42), load_verbs=True, load_actions=True, action_dim=18,
                       exclude_manifest=args.manifest, only_manifest=True)
    specs = sample_clips(ds, args.n_clips, args.K, args.skip, seed=args.sample_seed)
    fr, act = cache_clips(ds, specs)
    print(f"eval clips={len(specs)}  manifest={args.manifest}")

    if args.cond == "gt_action":
        cond, cond_dim = act.clone(), 18
    else:
        spec = spec_map[args.arm]; acfg = _load_cfg(_resolve(spec.config))
        acfg["data"]["exclude_manifest"] = args.manifest; acfg["data"]["only_manifest"] = True
        enc = load_model(str(_resolve(spec.checkpoint)), acfg, device); enc.eval()
        apd = enc.action_part
        cond = torch.stack([encode_seq(enc, f, device)[:, :apd] for f in fr])
        mu = cond.reshape(-1, apd).mean(0); sd = cond.reshape(-1, apd).std(0) + 1e-6
        cond = (cond - mu) / sd; cond_dim = apd
        del enc; torch.cuda.empty_cache()

    model = ConditionalUNet(cond_dim=cond_dim, base=args.base).to(device)
    model.load_state_dict(torch.load(args.consumer_ckpt, map_location=device)); model.eval()
    gate = dev_gate(model, fr, cond, device)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"ckpt": args.consumer_ckpt, "manifest": args.manifest,
                                          "cond": args.cond, "arm": args.arm, "gate": gate}, indent=2))
    print(f"GATE on {Path(args.manifest).name}:")
    for k, v in gate.items():
        print(f"  {k:12} endpoint_err={v['endpoint_err']:.4f}  dir_cos={v['dir_cos']:+.4f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
