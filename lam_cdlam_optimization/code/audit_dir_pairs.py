"""L_dir 配对挖掘的离线 oracle 审计(stage-1 前的风险闸门)。

问题:episode 内到底存不存在足够多的"运动反平行"帧对?若 oracle(GT 位姿)下通过率
就很低,L_dir 的设计有根本问题,不该启动训练。

三种运动表示(它们之间的分歧本身就是结论):
  own    : inv(T_t) @ T_{t+skip} 的平移 —— 18D 标签用的就是这个(手自身坐标系)
  world  : p_{t+skip} - p_t —— 世界系位移
  cam    : inv(C_t) 旋转到 t 时刻相机系的世界位移 —— ego-compensated,
           最接近"光流/关键点代理去掉相机运动后应该看到的东西"
另外量化 **相机运动残差**:未补偿的相机系位移 vs 补偿后的差异,
用来回答顾问的警告"光流同时含相机和背景运动"。

纯 CPU,只读 HDF5,不解码视频。
    /home/xuan/.venv/bin/python code/audit_dir_pairs.py --n_episodes 200
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

DATA_ROOT = Path("/home/xuan/embodied-ai/data/egodex/test_240p")


def _load(hdf5_path: Path):
    with h5py.File(hdf5_path, "r") as hf:
        tf = hf["transforms"]
        left = tf["leftHand"][:].astype(np.float64)
        right = tf["rightHand"][:].astype(np.float64)
        cam = tf["camera"][:].astype(np.float64) if "camera" in tf else None
    return left, right, cam


def motion_vectors(T: np.ndarray, cam: np.ndarray | None, skip: int):
    """Return dict of [N,3] per-frame-pair motion vectors under three conventions."""
    n = len(T) - skip
    if n <= 1:
        return None
    A, B = T[:n], T[skip:skip + n]
    p_a, p_b = A[:, :3, 3], B[:, :3, 3]

    # own frame: translation part of inv(T_t) @ T_{t+skip}
    R_a = A[:, :3, :3]
    own = np.einsum("nji,nj->ni", R_a, p_b - p_a)          # R_a^T @ (p_b - p_a)

    world = p_b - p_a

    out = {"own": own, "world": world}
    if cam is not None and len(cam) >= skip + n:
        C_a = cam[:n]
        Rc_a = C_a[:, :3, :3]
        c_a, c_b = C_a[:, :3, 3], cam[skip:skip + n][:, :3, 3]
        # ego-compensated: world hand displacement expressed in camera frame at t
        out["cam"] = np.einsum("nji,nj->ni", Rc_a, world)
        # what an UNcompensated image-space proxy would see: hand motion relative
        # to the moving camera (includes camera translation)
        out["cam_raw"] = np.einsum("nji,nj->ni", Rc_a, world - (c_b - c_a))
    return out


def antiparallel_stats(v: np.ndarray, mag_pct: float, cos_thresh: float):
    """Count anti-parallel pairs among frame pairs passing the magnitude gate."""
    mag = np.linalg.norm(v, axis=1)
    if len(mag) < 2:
        return None
    floor = np.percentile(mag, mag_pct)
    keep = mag > max(floor, 1e-9)
    n_keep = int(keep.sum())
    if n_keep < 2:
        return {"n_pairs": len(mag), "n_gated": n_keep, "n_anti": 0,
                "frac_anti": 0.0, "min_cos": float("nan"), "median_mag": float(np.median(mag))}
    u = v[keep] / np.linalg.norm(v[keep], axis=1, keepdims=True)
    C = u @ u.T
    iu = np.triu_indices(n_keep, k=1)
    cos = C[iu]
    return {
        "n_pairs": len(mag),
        "n_gated": n_keep,
        "n_anti": int((cos < cos_thresh).sum()),
        "frac_anti": float((cos < cos_thresh).mean()),
        "min_cos": float(cos.min()),
        "median_mag": float(np.median(mag)),
        "cos": cos,                       # kept for the global histogram
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_episodes", type=int, default=200)
    ap.add_argument("--skip", type=int, default=1)
    ap.add_argument("--mag_pct", type=float, default=60.0,
                    help="magnitude gate: keep frame pairs above this percentile within the episode")
    ap.add_argument("--cos_thresh", type=float, default=-0.5, help="anti-parallel iff cos < this (>120 deg)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/dir_pair_audit.json")
    args = ap.parse_args()

    eps = sorted(DATA_ROOT.glob("*/*.hdf5"))
    rng = np.random.default_rng(args.seed)
    if len(eps) > args.n_episodes:
        eps = [eps[i] for i in rng.choice(len(eps), args.n_episodes, replace=False)]
    print(f"auditing {len(eps)} episodes | skip={args.skip} mag_pct={args.mag_pct} "
          f"cos<{args.cos_thresh}")

    conventions = ["own", "world", "cam", "cam_raw"]
    per_conv = {c: defaultdict(list) for c in conventions}
    all_cos = {c: [] for c in conventions}
    per_task = defaultdict(lambda: defaultdict(list))
    cam_residual = []
    n_ok = 0

    for i, p in enumerate(eps):
        try:
            left, right, cam = _load(p)
        except Exception:
            continue
        task = p.parent.name
        # dominant hand = the one that moves more over the episode
        dl = np.linalg.norm(np.diff(left[:, :3, 3], axis=0), axis=1).sum()
        dr = np.linalg.norm(np.diff(right[:, :3, 3], axis=0), axis=1).sum()
        T = left if dl >= dr else right
        mv = motion_vectors(T, cam, args.skip)
        if mv is None:
            continue
        n_ok += 1
        for c in conventions:
            if c not in mv:
                continue
            st = antiparallel_stats(mv[c], args.mag_pct, args.cos_thresh)
            if st is None:
                continue
            all_cos[c].append(st.pop("cos"))
            for k, val in st.items():
                per_conv[c][k].append(val)
            per_conv[c]["has_any"].append(1.0 if st["n_anti"] > 0 else 0.0)
            per_conv[c]["has_ge6"].append(1.0 if st["n_anti"] >= 6 else 0.0)
            if c == "cam":
                per_task[task]["frac_anti"].append(st["frac_anti"])
                per_task[task]["n_anti"].append(st["n_anti"])
        if "cam" in mv and "cam_raw" in mv:
            a, b = mv["cam"], mv["cam_raw"]
            denom = np.linalg.norm(a, axis=1) + 1e-9
            cam_residual.append(float(np.median(np.linalg.norm(a - b, axis=1) / denom)))
        if (i + 1) % 50 == 0:
            print(f"  ...{i+1}/{len(eps)}")

    print(f"\n=== 通过率(每 episode,{n_ok} 个可用) ===")
    hdr = f"{'convention':10s} {'有≥1对':>8s} {'有≥6对':>8s} {'反平行占比':>10s} {'中位对数':>9s} {'最负cos':>9s}"
    print(hdr)
    summary = {}
    for c in conventions:
        d = per_conv[c]
        if not d:
            continue
        summary[c] = {
            "episodes": len(d["n_anti"]),
            "has_any": float(np.mean(d["has_any"])),
            "has_ge6": float(np.mean(d["has_ge6"])),
            "frac_anti_mean": float(np.mean(d["frac_anti"])),
            "n_anti_median": float(np.median(d["n_anti"])),
            "min_cos_median": float(np.median(d["min_cos"])),
            "n_gated_median": float(np.median(d["n_gated"])),
        }
        s = summary[c]
        print(f"{c:10s} {s['has_any']:8.1%} {s['has_ge6']:8.1%} {s['frac_anti_mean']:10.3%} "
              f"{s['n_anti_median']:9.0f} {s['min_cos_median']:9.3f}")

    print(f"\n=== 余弦分布(gated 帧对之间) ===")
    for c in conventions:
        if not all_cos[c]:
            continue
        cos = np.concatenate(all_cos[c])
        qs = np.percentile(cos, [1, 5, 25, 50])
        summary[c]["cos_pct"] = {"p1": float(qs[0]), "p5": float(qs[1]),
                                 "p25": float(qs[2]), "p50": float(qs[3])}
        print(f"{c:10s} p1={qs[0]:+.3f} p5={qs[1]:+.3f} p25={qs[2]:+.3f} median={qs[3]:+.3f}  "
              f"(<-0.5: {(cos < -0.5).mean():.2%}, <-0.8: {(cos < -0.8).mean():.2%})")

    if cam_residual:
        cr = np.array(cam_residual)
        summary["camera_residual_rel_median"] = float(np.median(cr))
        print(f"\n=== 相机运动残差 ===")
        print(f"未补偿 vs 补偿的相对差异(中位数): {np.median(cr):.2%}  "
              f"p25={np.percentile(cr,25):.2%} p75={np.percentile(cr,75):.2%}")
        print("  = 光流类代理若不做 ego-motion 补偿,方向向量会有这么大的偏差")

    tasks = sorted(per_task.items(), key=lambda kv: -np.mean(kv[1]["frac_anti"]))
    print(f"\n=== 按 task(cam 口径,反平行占比 top/bottom 5,共 {len(tasks)}) ===")
    for name, d in tasks[:5]:
        print(f"  {name:38s} {np.mean(d['frac_anti']):7.2%}  (中位 {np.median(d['n_anti']):.0f} 对)")
    print("  ...")
    for name, d in tasks[-5:]:
        print(f"  {name:38s} {np.mean(d['frac_anti']):7.2%}  (中位 {np.median(d['n_anti']):.0f} 对)")

    summary["_config"] = vars(args)
    summary["_per_task_cam"] = {k: {"frac_anti_mean": float(np.mean(v["frac_anti"])),
                                    "n_anti_median": float(np.median(v["n_anti"]))}
                                for k, v in per_task.items()}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
