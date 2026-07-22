"""H2 分解 + posterior SNR 的编排(零训练诊断,顾问第三轮回复 §1/§3)。

对每个阶段 B checkpoint,在**冻结评测 manifest** 上:
  1. h2_decompose  —— own/same/opp/zero/shuffle 误差 + R_out + 同量纲归一化 +
                       前景/背景 + GT-action donor 距离前提校验;
  2. posterior_snr —— 逐维 signal/noise/SNR/KL、active dims、same/opp 分离 vs 噪声。

判读矩阵(顾问 §1):
  R_out≈0 且正对照 C>0            → decoder 对该方向近似不变(utilization deficit)
  R_out>0 但 c_persist≈0          → 有响应但方向错/donor-GT 不对齐
  E_own < E_same≈E_opp            → 用了 episode latent,但 swap 没隔离出方向
  E_own≈E_same≈E_opp 且 zero/shuf 同 → 更强的 latent bypass
  donor_frac_same_lt_opp ≈ 0.5    → donor 语义前提不成立,C≈0 不可解读

⚠️ 本步不含 H2 正对照(硬门,另做);它只做现有 checkpoint 的分解。

    CUDA_VISIBLE_DEVICES=<free> python code/eval_h2_decompose.py --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np                                                    # noqa: E402
import torch                                                          # noqa: E402

import _apath  # noqa: F401  # shared A (lam_cdlam_optimization) substrate on sys.path
from benchmark_lam import RUN_SPECS, _load_cfg, _make_loader, _resolve  # noqa: E402
from eval import load_model                                           # noqa: E402
from h2_decompose import h2_decompose, posterior_snr, summarize_decompose  # noqa: E402

MANIFEST = "/home/xuan/embodied-ai/lam_condition_utilization/data/eval_manifest.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=["sb_B", "sb_R", "sb_D", "sb_U", "sb_S"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--h2_pool", type=int, default=1500)
    ap.add_argument("--snr_pool", type=int, default=3000)
    ap.add_argument("--fg_q", type=float, default=0.90)
    ap.add_argument("--out", default="results/stage_b_h2_decompose.json")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    spec_map = {s.name: s for s in RUN_SPECS}
    dec_summ, snr_summ, perdim, persample = {}, {}, {}, {}

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

        model = load_model(str(ck), cfg, device)
        model.eval()
        print(f"\n[{name}] {ck.parent.name}/{ck.name}", flush=True)

        dec = h2_decompose(model, _make_loader(cfg, args.num_workers, args.batch_size, split="train"),
                           device, n_pool=args.h2_pool, fg_q=args.fg_q)
        if dec is not None:
            s = summarize_decompose(dec)
            dec_summ[name] = s
            raw_num = float(np.mean(dec["E_opp_all"] - dec["E_same_all"]))
            print(f"  anchors={s['n_anchors']} tasks={s['n_tasks']} fg_frac={s['fg_frac_mean']:.3f}")
            print(f"  E_own={s['E_own']['mean']:.5f} E_same={s['E_same']['mean']:.5f} "
                  f"E_opp={s['E_opp']['mean']:.5f} E_zero={s['E_zero']['mean']:.5f} "
                  f"E_shuf={s['E_shuf']['mean']:.5f}")
            print(f"  raw(E_opp-E_same)={raw_num:+.2e}  c_persist={s['c_persist']['mean']:+.4f}  "
                  f"Rout={s['Rout']['mean']:.5f} (fg={s['Rout_fg']['mean']:.5f})")
            print(f"  usage_zero={s['usage_zero']['mean']:+.4f} usage_shuf={s['usage_shuf']['mean']:+.4f}  "
                  f"donor d_same<d_opp={s['donor_frac_same_lt_opp']:.3f} "
                  f"(d_same={s['d_same']['mean']:.3f} d_opp={s['d_opp']['mean']:.3f})")
            persample[name] = {k: dec[k] for k in dec}

        snr = posterior_snr(model, _make_loader(cfg, args.num_workers, args.batch_size, split="train"),
                            device, n_pool=args.snr_pool)
        perdim[name] = {k[1:]: snr[k] for k in snr if k.startswith("_")}
        snr_summ[name] = {k: v for k, v in snr.items() if not k.startswith("_")}
        print(f"  SNR: KL_tot={snr['kl_total']:.2f} (act={snr['kl_total_action']:.2f} "
              f"env={snr['kl_total_env']:.2f})  active_dims={snr['active_dims']} "
              f"(act={snr['active_dims_action']}/{model.action_part})  "
              f"eff_rank_a={snr['eff_rank_action']:.1f}")
        print(f"       sep(same,opp)/noise={snr['sep_over_noise']:.3f} "
              f"(sep={snr['sep_same_opp_action']:.3f} noise={snr['noise_scale_action']:.3f}) "
              f"snr_act_mean={snr['snr_action_mean']:.3f}  n_rev_tasks={snr['n_reversible_tasks']}")

        del model
        torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"h2_decompose": dec_summ, "posterior_snr": snr_summ}, indent=2))
    # per-dim + per-sample arrays for later paired bootstrap / plotting
    npz_pd = {f"{a}__{k}": v for a, d in perdim.items() for k, v in d.items()}
    if npz_pd:
        np.savez(str(out.with_name("stage_b_h2_decompose_perdim.npz")), **npz_pd)
    npz_ps = {f"{a}__{k}": v for a, d in persample.items() for k, v in d.items()}
    if npz_ps:
        np.savez(str(out.with_name("stage_b_h2_decompose_persample.npz")), **npz_ps)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
