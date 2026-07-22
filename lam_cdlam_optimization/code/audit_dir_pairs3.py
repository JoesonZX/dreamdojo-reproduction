"""L_dir 配对审计第三轮:存不存在**能富集真实反转**的挖掘规则?

第二轮结论:全 episode 范围内"反平行"仅比各向同性 null 高 1.24×,且时间差分布与
全体对**完全一致**(Δt 中位 5.47s vs 5.60s)⇒ 朴素规则挑出的主要是巧合。

本轮扫描两个旋钮,看有没有窗口能把富集度推上去:
  - 时间差 Δt 分箱(真实的伸手↔收手反转应发生在特定时间尺度上)
  - 幅度门 p60/p80/p90(滤掉抖动)
每格都与"保幅度、随机化方向"的 null 对照,报告 lift。

    /home/xuan/.venv/bin/python code/audit_dir_pairs3.py --n_episodes 150
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audit_dir_pairs import DATA_ROOT, _load, motion_vectors

DT_BINS = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 1e9)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_episodes", type=int, default=150)
    ap.add_argument("--skip", type=int, default=1)
    ap.add_argument("--cos_thresh", type=float, default=-0.5)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/dir_pair_audit3.json")
    args = ap.parse_args()

    eps = sorted(DATA_ROOT.glob("*/*.hdf5"))
    rng = np.random.default_rng(args.seed)
    if len(eps) > args.n_episodes:
        eps = [eps[i] for i in rng.choice(len(eps), args.n_episodes, replace=False)]
    mag_pcts = [60.0, 80.0, 90.0]
    print(f"auditing {len(eps)} episodes | cos<{args.cos_thresh}")

    # acc[mag_pct][bin] = [n_anti, n_pairs, n_anti_null]
    acc = {m: np.zeros((len(DT_BINS), 3)) for m in mag_pcts}
    n_ok = 0

    for i, p in enumerate(eps):
        try:
            left, right, cam = _load(p)
        except Exception:
            continue
        dl = np.linalg.norm(np.diff(left[:, :3, 3], axis=0), axis=1).sum()
        dr = np.linalg.norm(np.diff(right[:, :3, 3], axis=0), axis=1).sum()
        mv = motion_vectors(left if dl >= dr else right, cam, args.skip)
        if mv is None or "cam" not in mv:
            continue
        v = mv["cam"]
        mag = np.linalg.norm(v, axis=1)
        n_ok += 1
        for m in mag_pcts:
            keep = mag > max(np.percentile(mag, m), 1e-9)
            if keep.sum() < 8:
                continue
            idx = np.flatnonzero(keep)
            u = v[keep] / mag[keep][:, None]
            iu = np.triu_indices(len(u), k=1)
            cos = (u @ u.T)[iu]
            g = rng.normal(size=u.shape)
            g /= np.linalg.norm(g, axis=1, keepdims=True)
            cos_null = (g @ g.T)[iu]
            dt = np.abs(idx[iu[1]] - idx[iu[0]]) / args.fps
            for b, (lo, hi) in enumerate(DT_BINS):
                sel = (dt >= lo) & (dt < hi)
                if not sel.any():
                    continue
                acc[m][b, 0] += (cos[sel] < args.cos_thresh).sum()
                acc[m][b, 1] += sel.sum()
                acc[m][b, 2] += (cos_null[sel] < args.cos_thresh).sum()
        if (i + 1) % 50 == 0:
            print(f"  ...{i+1}/{len(eps)}")

    print(f"\n=== 反平行率 vs Δt 分箱({n_ok} episodes);lift = 真实/null ===")
    out = {}
    for m in mag_pcts:
        print(f"\n幅度门 p{m:.0f}:")
        print(f"  {'Δt 区间':>12s} {'真实率':>8s} {'null率':>8s} {'lift':>6s} {'对数':>10s}")
        rows = []
        for b, (lo, hi) in enumerate(DT_BINS):
            n_a, n_p, n_n = acc[m][b]
            if n_p < 100:
                continue
            r, rn = n_a / n_p, n_n / n_p
            lift = r / max(rn, 1e-9)
            label = f"{lo:.1f}-{hi:.1f}s" if hi < 1e8 else f">{lo:.0f}s"
            flag = "  <<<" if lift >= 1.5 else ""
            print(f"  {label:>12s} {r:8.2%} {rn:8.2%} {lift:6.2f} {n_p:10.0f}{flag}")
            rows.append({"bin": label, "real": float(r), "null": float(rn),
                         "lift": float(lift), "n_pairs": float(n_p)})
        out[f"p{m:.0f}"] = rows

    best = max((r["lift"], f"p{m:.0f}", r["bin"]) for m in mag_pcts for r in out[f"p{m:.0f}"])
    print(f"\n最佳格子: lift={best[0]:.2f} @ {best[1]} / Δt {best[2]}")
    print("判读: lift<1.3 ⇒ 无论怎么切,反平行都接近巧合,朴素配对规则不成立;")
    print("      lift>=1.5 ⇒ 该窗口存在真实富集,可作为挖掘规则的起点。")
    out["_best"] = {"lift": best[0], "mag_gate": best[1], "dt_bin": best[2]}
    out["_config"] = vars(args)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
