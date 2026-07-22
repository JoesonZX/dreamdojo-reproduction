"""L_dir 的离线配对挖掘(阶段 B 前置)。

按 `notes/dir_pair_audit.md` 定稿的规则挖 episode 内的运动反平行帧对:

    同 episode ∧ Δt∈[dt_lo, dt_hi] ∧ 两者幅度 > episode 内 p{mag_pct}
              ∧ cos(v_i, v_j) < cos_thresh,v = ego-compensated 运动向量

同时产出**两个对照臂**(每 episode 与方法臂等量,同样的 loss 形式,只有配对选择不同
——这正是能证伪"方向"主张的设计):

  kind=0 反平行(方法):cos < cos_thresh,100% 反平行
  kind=1 匹配非方向排斥:同 Δt/幅度门,**排除反平行**并优先取 |cos|<ortho_max 的
                      近正交对。这是 H1 的**主对照**——同样的排斥预算施加在没有方向
                      关系的对上。(v1 只是"不看 cos",实测 52% 仍是反平行,证伪力太弱。)
  kind=2 同向对照    :cos > +|cos_thresh|,**同向**对。推开同向对在语义上是错的,
                      因此这是**最锋利的证伪器**:若 L_dir 在它上面同样"有效",
                      说明 loss 纯粹是排斥项,方向主张直接不成立。

⚠️ `--source gt` 用 HDF5 真值位姿挖对(oracle)。它**不是** target-action-annotation-free,
只用于**机制验证**(先确认 loss 有没有作用),论文主线需换成自监督代理(光流/关键点 +
ego-motion 补偿);审计已测出两种口径逐对一致率 87.8%、Jaccard 0.662。

    python code/mine_dir_pairs.py --split train --out data/dir_pairs_train.npz
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_dir_pairs import _load, motion_vectors  # noqa: E402
from dataset import EgoDexDataset  # noqa: E402


def mine_episode(T, cam, skip, dt_lo, dt_hi, mag_pct, cos_thresh, fps, rng, max_pairs,
                 ortho_max=0.2):
    """Return (anti_pairs, ctrl_pairs) as arrays of [i, j, cos]."""
    mv = motion_vectors(T, cam, skip)
    if mv is None:
        return None, None, None
    v = mv.get("cam", mv["world"])          # ego-compensated when the camera track exists
    mag = np.linalg.norm(v, axis=1)
    if len(mag) < 4:
        return None, None, None
    gate = mag > max(np.percentile(mag, mag_pct), 1e-9)
    idx = np.flatnonzero(gate)
    if len(idx) < 2:
        return None, None, None
    u = v[idx] / mag[idx][:, None]
    C = u @ u.T
    iu = np.triu_indices(len(idx), k=1)
    dt = np.abs(idx[iu[1]] - idx[iu[0]]) / fps
    in_win = (dt >= dt_lo) & (dt <= dt_hi)
    if not in_win.any():
        return None, None, None
    cos = C[iu]
    anti = in_win & (cos < cos_thresh)
    n_anti = int(anti.sum())
    if n_anti == 0:
        return None, None, None

    def take(mask, n):
        sel = np.flatnonzero(mask)
        if len(sel) > n:
            sel = rng.choice(sel, n, replace=False)
        return np.stack([idx[iu[0]][sel], idx[iu[1]][sel], cos[sel]], axis=1)

    keep = min(n_anti, max_pairs)
    anti_pairs = take(anti, keep)
    # control A (kind=1): MATCHED NON-DIRECTIONAL repulsion — same Δt/magnitude
    #   gates, but anti-parallel pairs are EXCLUDED and near-orthogonal ones
    #   (|cos| < ortho_max) preferred. This is the primary control for H1: it
    #   applies the same repulsion budget to pairs carrying no direction relation.
    #   (v1 of this arm merely ignored cos and came out ~52% anti-parallel — far too
    #   weak to falsify anything; see notes/dir_pair_audit.md.)
    ortho = in_win & (np.abs(cos) < ortho_max)
    ctrl_pairs = take(ortho, keep) if ortho.sum() >= 1 else take(
        in_win & (cos >= cos_thresh), keep)
    # control B (kind=2): SAME-direction pairs (cos > +|cos_thresh|). Pushing these
    #   apart is semantically wrong, so this is the sharp falsifier: if L_dir works
    #   here too, the loss is pure repulsion and the direction claim is dead.
    same = in_win & (cos > abs(cos_thresh))
    same_pairs = take(same, keep) if same.any() else np.zeros((0, 3))
    return anti_pairs, ctrl_pairs, same_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/home/xuan/embodied-ai/data/egodex/test_240p")
    ap.add_argument("--split", default="train")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip", type=int, default=1)
    ap.add_argument("--dt_lo", type=float, default=0.5)
    ap.add_argument("--dt_hi", type=float, default=2.0)
    ap.add_argument("--mag_pct", type=float, default=80.0)
    ap.add_argument("--cos_thresh", type=float, default=-0.5)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--max_pairs_per_ep", type=int, default=200)
    ap.add_argument("--ortho_max", type=float, default=0.2,
                    help="kind=1 control keeps pairs with |cos| below this "
                         "(near-orthogonal = no direction relation)")
    ap.add_argument("--source", choices=["gt"], default="gt",
                    help="gt = HDF5 poses (oracle, mechanism check only). "
                         "A flow/keypoint proxy source is the label-free replacement.")
    ap.add_argument("--exclude_manifest", default="data/eval_manifest.json",
                    help="frozen eval episodes to EXCLUDE. Mining over them would leak "
                         "the held-out pool into L_dir training.")
    ap.add_argument("--out", default="data/dir_pairs_train.npz")
    args = ap.parse_args()

    ds = EgoDexDataset(
        data_root=args.data_root, img_h=240, img_w=320, downsample_factors=(1,),
        split=args.split, val_ratio=args.val_ratio, seed=args.seed,
        load_verbs=False, load_actions=False,
        exclude_manifest=args.exclude_manifest or None,
    )
    # one entry per video (the dataset index has one row per frame pair)
    vids = {}
    for rec in ds._index:
        vp, _, task_id, ep_id, hdf5, n = rec
        vids[str(vp)] = (vp, task_id, ep_id, hdf5, n)
    print(f"split={args.split}: {len(vids)} videos | rule: dt∈[{args.dt_lo},{args.dt_hi}]s "
          f"mag>p{args.mag_pct:.0f} cos<{args.cos_thresh} skip={args.skip}")

    rng = np.random.default_rng(args.seed)
    root = Path(args.data_root)
    cols = defaultdict(list)
    n_ep_ok = n_ep_seen = 0
    per_task = defaultdict(int)

    for k, (_key, (vp, task_id, ep_id, hdf5, n)) in enumerate(sorted(vids.items())):
        n_ep_seen += 1
        try:
            left, right, cam = _load(Path(hdf5))
        except Exception:
            continue
        dl = np.linalg.norm(np.diff(left[:, :3, 3], axis=0), axis=1).sum()
        dr = np.linalg.norm(np.diff(right[:, :3, 3], axis=0), axis=1).sum()
        T = left if dl >= dr else right
        if len(T) > n:                       # poses can outrun the decoded video
            T = T[:n]
            cam = cam[:n] if cam is not None else None
        a, c, sm = mine_episode(T, cam, args.skip, args.dt_lo, args.dt_hi, args.mag_pct,
                                args.cos_thresh, args.fps, rng, args.max_pairs_per_ep,
                                ortho_max=args.ortho_max)
        if a is None:
            continue
        n_ep_ok += 1
        per_task[task_id] += len(a)
        rel = str(Path(vp).relative_to(root))
        for arr, kind in ((a, 0), (c, 1), (sm, 2)):
            for ti, tj, cs in arr:
                cols["video"].append(rel)
                cols["task"].append(task_id)
                cols["ep"].append(ep_id)
                cols["t_i"].append(int(ti))
                cols["t_j"].append(int(tj))
                cols["skip"].append(args.skip)
                cols["cos"].append(float(cs))
                cols["kind"].append(kind)       # 0 = anti-parallel, 1 = matched control
                cols["n_frames"].append(int(n))
        if (k + 1) % 500 == 0:
            print(f"  ...{k+1}/{len(vids)}  episodes_with_pairs={n_ep_ok}")

    if not cols:
        raise SystemExit("no pairs mined — loosen the gates")
    kind = np.asarray(cols["kind"])
    n_anti = int((kind == 0).sum())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        video=np.asarray(cols["video"]), task=np.asarray(cols["task"]),
        ep=np.asarray(cols["ep"]), t_i=np.asarray(cols["t_i"], dtype=np.int32),
        t_j=np.asarray(cols["t_j"], dtype=np.int32),
        skip=np.asarray(cols["skip"], dtype=np.int16),
        cos=np.asarray(cols["cos"], dtype=np.float32), kind=kind.astype(np.int8),
        n_frames=np.asarray(cols["n_frames"], dtype=np.int32),
        config=np.asarray([repr(vars(args))]),
    )
    allcos = np.asarray(cols["cos"])
    print(f"\n=== 挖掘完成 ===")
    print(f"有配对的 episode : {n_ep_ok}/{n_ep_seen} ({n_ep_ok/max(1,n_ep_seen):.1%})")
    print(f"每 episode 中位   : {n_anti/max(1,n_ep_ok):.1f} 个反平行对")
    for kk, nm in ((0, "kind=0 反平行(方法)"), (1, "kind=1 近正交对照"), (2, "kind=2 同向对照")):
        cv = allcos[kind == kk]
        if len(cv) == 0:
            print(f"{nm:24s}: 空")
            continue
        print(f"{nm:24s}: n={len(cv):>8,}  cos 中位 {np.median(cv):+.3f}  "
              f"[{cv.min():+.3f},{cv.max():+.3f}]  反平行占比 {np.mean(cv < -0.5):.1%}")
    top = sorted(per_task.items(), key=lambda kv: -kv[1])
    print(f"task 覆盖        : {len(per_task)} 个;最多 {top[0][0]}={top[0][1]:,}  "
          f"最少 {top[-1][0]}={top[-1][1]:,}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
