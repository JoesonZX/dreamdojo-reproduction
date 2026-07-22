"""L_dir 配对审计第二轮:反平行对是"真实反转"还是"巧合"?

第一轮(audit_dir_pairs.py)发现反平行对极其丰富(99% episode 有 ≥6 对,占比 ~30%)。
但**各向同性随机方向本身就给 25%**(3D 单位向量的 cos 均匀分布于 [-1,1])——
所以丰富 ≠ 有意义。本轮回答三个决定 L_dir 成败的问题:

1. **null 对照**:episode 内打乱方向后反平行率是多少?真实数据高出多少?
2. **时间结构**:真实的伸手↔收手反转应当**时间上相邻**(同一子动作的两个相位);
   巧合的反平行对则在时间差上均匀分布。看 Δt 分布是判别关键。
3. **代理一致性**:own(18D 标签口径)与 cam(光流代理口径)是否标记**同一批**对?
   聚合率相近不代表逐对一致。

    /home/xuan/.venv/bin/python code/audit_dir_pairs2.py --n_episodes 150
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audit_dir_pairs import DATA_ROOT, _load, motion_vectors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_episodes", type=int, default=150)
    ap.add_argument("--skip", type=int, default=1)
    ap.add_argument("--mag_pct", type=float, default=60.0)
    ap.add_argument("--cos_thresh", type=float, default=-0.5)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/dir_pair_audit2.json")
    args = ap.parse_args()

    eps = sorted(DATA_ROOT.glob("*/*.hdf5"))
    rng = np.random.default_rng(args.seed)
    if len(eps) > args.n_episodes:
        eps = [eps[i] for i in rng.choice(len(eps), args.n_episodes, replace=False)]
    print(f"auditing {len(eps)} episodes | cos<{args.cos_thresh} | mag>p{args.mag_pct}")

    real_rate, null_rate = [], []
    dt_anti, dt_all = [], []
    agree_own_cam, jaccard = [], []
    mag_ratio_anti, mag_ratio_all = [], []
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
        keep = mag > max(np.percentile(mag, args.mag_pct), 1e-9)
        if keep.sum() < 8:
            continue
        n_ok += 1
        idx = np.flatnonzero(keep)
        u = v[keep] / mag[keep][:, None]
        C = u @ u.T
        iu = np.triu_indices(len(u), k=1)
        cos = C[iu]
        anti = cos < args.cos_thresh
        real_rate.append(float(anti.mean()))

        # (1) null: 保持幅度、随机化方向(各向同性)
        g = rng.normal(size=u.shape)
        g /= np.linalg.norm(g, axis=1, keepdims=True)
        cos_null = (g @ g.T)[iu]
        null_rate.append(float((cos_null < args.cos_thresh).mean()))

        # (2) 时间结构:反平行对 vs 全部对的时间差(秒)
        ti, tj = idx[iu[0]], idx[iu[1]]
        dt = np.abs(tj - ti) / args.fps
        dt_anti.append(dt[anti])
        dt_all.append(dt)

        # 幅度相似性(真实反转应当幅度相当)
        mi, mj = mag[keep][iu[0]], mag[keep][iu[1]]
        r = np.minimum(mi, mj) / np.maximum(mi, mj)
        mag_ratio_anti.append(r[anti])
        mag_ratio_all.append(r)

        # (3) own vs cam 逐对一致性
        if "own" in mv:
            vo = mv["own"][keep]
            uo = vo / (np.linalg.norm(vo, axis=1, keepdims=True) + 1e-12)
            anti_o = (uo @ uo.T)[iu] < args.cos_thresh
            both = float((anti & anti_o).sum())
            union = float((anti | anti_o).sum())
            agree_own_cam.append(float((anti == anti_o).mean()))
            jaccard.append(both / union if union > 0 else np.nan)

        if (i + 1) % 50 == 0:
            print(f"  ...{i+1}/{len(eps)}")

    R, N = np.array(real_rate), np.array(null_rate)
    dtA = np.concatenate(dt_anti)
    dtL = np.concatenate(dt_all)
    mrA = np.concatenate(mag_ratio_anti)
    mrL = np.concatenate(mag_ratio_all)
    J = np.array([x for x in jaccard if x == x])
    A = np.array(agree_own_cam)

    print(f"\n=== 1. 真实 vs 各向同性 null({n_ok} episodes) ===")
    print(f"真实反平行率 : {R.mean():.2%}  (中位 {np.median(R):.2%})")
    print(f"随机化 null   : {N.mean():.2%}  (中位 {np.median(N):.2%})")
    lift = R.mean() / max(N.mean(), 1e-9)
    print(f"提升倍数     : {lift:.2f}×   ⇒ " +
          ("反平行主要是巧合,配对规则缺乏特异性" if lift < 1.3 else "真实数据显著富集"))

    print(f"\n=== 2. 时间结构(反平行对是否时间相邻) ===")
    for name, d in (("反平行对", dtA), ("全部对", dtL)):
        q = np.percentile(d, [25, 50, 75])
        print(f"{name:8s} Δt 中位 {q[1]:6.2f}s  (p25 {q[0]:.2f} / p75 {q[2]:.2f})  "
              f"≤2s 占比 {np.mean(d <= 2.0):.1%}")
    print("  ⇒ 若两行接近,说明反平行对**不是**时间局部的真实反转,而是全 episode 的巧合配对")

    print(f"\n=== 3. 幅度相似性(真实反转应幅度相当) ===")
    print(f"反平行对 min/max 幅度比 中位 {np.median(mrA):.3f} | 全部对 {np.median(mrL):.3f}")

    print(f"\n=== 4. own(18D 口径) vs cam(代理口径) 逐对一致性 ===")
    print(f"逐对标签一致率 : {A.mean():.1%}")
    print(f"Jaccard(交/并) : {np.nanmean(J):.3f}  ⇒ " +
          ("两种口径挑出的是**不同**的对,代理需谨慎" if np.nanmean(J) < 0.6
           else "两种口径高度一致,代理可行"))

    summary = {
        "episodes": n_ok,
        "real_rate_mean": float(R.mean()), "null_rate_mean": float(N.mean()),
        "lift": float(lift),
        "dt_anti_median_s": float(np.median(dtA)), "dt_all_median_s": float(np.median(dtL)),
        "dt_anti_frac_le2s": float(np.mean(dtA <= 2.0)),
        "dt_all_frac_le2s": float(np.mean(dtL <= 2.0)),
        "magratio_anti_median": float(np.median(mrA)),
        "magratio_all_median": float(np.median(mrL)),
        "own_cam_agreement": float(A.mean()), "own_cam_jaccard": float(np.nanmean(J)),
        "_config": vars(args),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
