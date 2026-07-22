"""阶段 B 的 confirmatory 评估:在**冻结评测 manifest** 上做配对比较。

这是本项目第一批有资格叫 confirmatory 的数字 —— 此前所有 probe 结果都因
encoder 训练时见过评测 episode 而只能算 exploratory。

H1(representation amplification):
    P_m  = G_m − E_perm[G_m]        (null-calibrated 条件方向 probe)
    ΔP   = P_D − P_R                (**主对比是方法 vs 匹配对照**,不是 vs 基线)
H2(direction-specific decoder utilization,阶段 B 的主 Tier-1 指标):
    C_m  = E[(MSE_opp − MSE_same) / (motion_energy + eps)]
    ΔC   = C_D − C_R
守门:S(同向 adverse)不得复制 D 的改善;D 相对 B 也须改善。

    CUDA_VISIBLE_DEVICES=0 python code/eval_stage_b.py --n_perm 999
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch

import _apath  # noqa: F401  # shared A (lam_cdlam_optimization) substrate on sys.path
from benchmark_lam import (RUN_SPECS, _collect_opposite, _make_loader,  # noqa: E402
                           _load_cfg, _resolve)
from dirprobe import conditional_direction_probe, paired_delta  # noqa: E402
from h2_utilization import h2_swap_utilization, paired_c_delta  # noqa: E402
from eval import load_model  # noqa: E402

MANIFEST = "/home/xuan/embodied-ai/lam_condition_utilization/data/eval_manifest.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=["sb_B", "sb_R", "sb_D", "sb_U", "sb_S"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--n_perm", type=int, default=999)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--h2_pool", type=int, default=1500)
    ap.add_argument("--skip_h1", action="store_true", help="H2 only (H1 already computed)")
    ap.add_argument("--skip_h2", action="store_true")
    ap.add_argument("--out", default="results/stage_b_eval.json")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    spec_map = {s.name: s for s in RUN_SPECS}
    results, h2_res = {}, {}

    for name in args.arms:
        spec = spec_map.get(name)
        if spec is None:
            print(f"[skip] {name}: no RunSpec")
            continue
        cfg = _load_cfg(_resolve(spec.config))
        # evaluate ONLY on the frozen manifest the encoder never trained on
        cfg["data"]["exclude_manifest"] = MANIFEST
        cfg["data"]["only_manifest"] = True
        ck = _resolve(spec.checkpoint)
        if not any(ck.glob("*.safetensors")):
            print(f"[skip] {name}: no checkpoint at {ck}")
            continue
        model = load_model(str(ck), cfg, device)
        model.eval()
        ap_ = model.action_part
        print(f"[{name}] {ck.parent.name}/{ck.name}", flush=True)

        d = _collect_opposite(model, cfg, device, args.num_workers, args.batch_size,
                              split="train", min_eps_per_verb=2)
        if d is None:
            print(f"[skip] {name}: empty opposite collection on the manifest pool")
            del model
            torch.cuda.empty_cache()
            continue
        if not args.skip_h1:
            r = conditional_direction_probe(
                d["mu"][:, :ap_], d["static_mu"][:, :ap_], d["task"], d["verb"], d["ep"],
                n_perm=args.n_perm, n_boot=args.n_boot, seed=42)
            results[name] = r
            print(f"    P={r['dirprobe_P']:+.4f} "
                  f"CI=[{r['dirprobe_P_ci_lo']:+.4f},{r['dirprobe_P_ci_hi']:+.4f}] "
                  f"p={r['dirprobe_perm_p']:.4f} "
                  f"tasks={r['dirprobe_n_tasks']:.0f} eps={r['dirprobe_n_episodes']:.0f}", flush=True)

        # H2: decoder-side utilization on the SAME frozen manifest pool
        if not args.skip_h2:
            h2_loader = _make_loader(cfg, args.num_workers, args.batch_size, split="train")
            h2 = h2_swap_utilization(model, h2_loader, device, n_pool=args.h2_pool)
            h2_res[name] = h2
        else:
            h2 = None
        if h2 is not None:
            print(f"  H2: C={h2['c'].mean():+.4f} n={len(h2['c'])} "
                  f"anchors_tasks={len(np.unique(h2['task']))}", flush=True)
        del model
        torch.cuda.empty_cache()

    pairs = [("sb_D", "sb_R"), ("sb_D", "sb_B"), ("sb_R", "sb_B"),
             ("sb_U", "sb_B"), ("sb_S", "sb_R")]
    deltas = {}
    if not args.skip_h1:
        print("\n=== H1 配对对比(ΔP,主对比是 D vs R) ===")
    for a, b in pairs:
        if a in results and b in results:
            dd = paired_delta(results[a], results[b], n_boot=args.n_boot)
            deltas[f"{a}_vs_{b}"] = dd
            sig = "显著" if dd["delta_ci_lo"] > 0 else ("负向" if dd["delta_ci_hi"] < 0 else "n.s.")
            print(f"  ΔP({a} − {b}) = {dd['delta_P']:+.4f} "
                  f"CI=[{dd['delta_ci_lo']:+.4f},{dd['delta_ci_hi']:+.4f}]  {sig}")

    print("\n=== H2 配对对比(ΔC,主对比 D vs R) ===")
    h2_deltas = {}
    for a, b in pairs:
        if h2_res.get(a) is not None and h2_res.get(b) is not None:
            dd = paired_c_delta(h2_res[a], h2_res[b], n_boot=args.n_boot)
            h2_deltas[f"{a}_vs_{b}"] = dd
            sig = "显著>0" if dd["delta_ci_lo"] > 0 else ("显著<0" if dd["delta_ci_hi"] < 0 else "n.s.")
            print(f"  ΔC({a} − {b}) = {dd['delta_C']:+.4f} "
                  f"CI=[{dd['delta_ci_lo']:+.4f},{dd['delta_ci_hi']:+.4f}] "
                  f"(C_a={dd['C_a']:+.4f} C_b={dd['C_b']:+.4f} anchors={dd['n_anchors']:.0f}) {sig}")

    h2_slim = {k: ({"C_mean": float(v["c"].mean()), "n": int(len(v["c"])),
                    "n_tasks": int(len(np.unique(v["task"])))} if v else None)
               for k, v in h2_res.items()}
    payload = {}
    out = Path(args.out)
    if out.exists():
        try:
            payload = json.loads(out.read_text())
        except Exception:
            payload = {}
    if not args.skip_h1:
        payload["per_arm"] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                              for k, v in results.items()}
        payload["paired"] = deltas
    if not args.skip_h2:
        payload["h2_per_arm"] = h2_slim
        payload["h2_paired"] = h2_deltas
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out} (skip_h1={args.skip_h1} skip_h2={args.skip_h2})")


if __name__ == "__main__":
    main()
