"""α=0 生成/OOD 鲁棒性编排。见 code/ood_robustness.py 头注。

对 A0/A1(两 seed)在冻结 manifest 上跑 sampling-robustness + do(z_a)-extrapolation 曲线。
判读:α=0 若 utilization 恢复但 OOD 脆弱,会表现为 mse_s 随 s 陡升(比 α=1 陡)或 tv_k
在大 k 处远超真实帧 TV。

    CUDA_VISIBLE_DEVICES=<free> python code/eval_ood_robustness.py --arms sb_A0 sb_A1 sb_A0_s1 sb_A1_s1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402

import _apath  # noqa: F401  # shared A (lam_cdlam_optimization) substrate on sys.path
from benchmark_lam import RUN_SPECS, _load_cfg, _make_loader, _resolve  # noqa: E402
from eval import load_model  # noqa: E402
from ood_robustness import ood_curves  # noqa: E402

MANIFEST = "/home/xuan/embodied-ai/lam_condition_utilization/data/eval_manifest.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=["sb_A0", "sb_A1", "sb_A0_s1", "sb_A1_s1"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--n_pool", type=int, default=1000)
    ap.add_argument("--n_eps", type=int, default=3)
    ap.add_argument("--out", default="results/stage_b_ood_robustness.json")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    spec_map = {s.name: s for s in RUN_SPECS}
    res = {}

    for name in args.arms:
        spec = spec_map.get(name)
        if spec is None:
            print(f"[skip] {name}: no RunSpec"); continue
        ck = _resolve(spec.checkpoint)
        if not any(ck.glob("*.safetensors")):
            print(f"[skip] {name}: no checkpoint at {ck}"); continue
        cfg = _load_cfg(_resolve(spec.config))
        cfg["data"]["exclude_manifest"] = MANIFEST
        cfg["data"]["only_manifest"] = True
        model = load_model(str(ck), cfg, device); model.eval()

        r = ood_curves(model, _make_loader(cfg, args.num_workers, args.batch_size, split="train"),
                       device, n_pool=args.n_pool, n_eps=args.n_eps)
        res[name] = r
        scales, ks = r["_scales"], r["_ks"]
        print(f"\n[{name}] n={r['n']}  (tv_ref: gt={r['tv_gt']:.4f} o_t={r['tv_ot']:.4f} μ-decode={r['tv_mu']:.4f})")
        print("  sampling  MSE(s): " + "  ".join(f"s={s}:{r[f'mse_s{s}']:.5f}" for s in scales)
              + f"   ΔMSE(2−0)={r[f'mse_s{scales[-1]}']-r['mse_s0.0']:+.5f}")
        print("  sampling  dev(s): " + "  ".join(f"s={s}:{r[f'dev_s{s}']:.4f}" for s in scales))
        print("  extrap  motion(k): " + "  ".join(f"k={k}:{r[f'motion_k{k}']:.4f}" for k in ks))
        print("  extrap      TV(k): " + "  ".join(f"k={k}:{r[f'tv_k{k}']:.4f}" for k in ks))
        del model
        torch.cuda.empty_cache()

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
