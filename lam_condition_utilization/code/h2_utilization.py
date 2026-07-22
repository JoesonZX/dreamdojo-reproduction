"""H2 —— direction-specific decoder utilization(阶段 B 的主 Tier-1 指标)。

固定 (o_t, z_e),把 z_a 换成
  (a) same-task/same-verb/异 episode 捐赠者
  (b) same-task/OPPOSITE-verb 捐赠者
解码后对真值 o_{t+1} 打误差。运动能量归一化,阻止高运动 clip 主导分数:

    c_i = (MSE_opp,i − MSE_same,i) / (motion_energy_i + eps)
    C_m = mean_i c_i
    ΔC  = C_D − C_R          (主对比:方法 vs 匹配非方向对照)

c_i > 0 表示"相反 z_a 解出可测差异的未来" —— 方向是**可执行**的,不只是几何可分。
在**冻结评测 manifest** 上评估(encoder 从未训练过这些 episode)。

per-sample 的 c_i 与 (task, ep) 一并保留,供 eval_stage_b 做配对 hierarchical bootstrap。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

import _apath  # noqa: F401  # shared A (lam_cdlam_optimization) substrate on sys.path
from benchmark_lam import _decode_from_mu, _encode_mu_var, _norm_verb


@torch.no_grad()
def h2_swap_utilization(model, loader, device, n_pool=1500, decode_bs=16, seed=42):
    """Return dict with per-sample motion-normalized (opp − same) error + labels."""
    videos_all, tasks, eps, verbs = [], [], [], []
    seen = 0
    for batch in loader:
        videos_all.append(batch["videos"])
        tasks.extend(batch["task_id"])
        eps.extend(batch["ep_id"])
        verbs.extend(batch.get("verb_id", ["unknown"] * batch["videos"].shape[0]))
        seen += batch["videos"].shape[0]
        if seen >= n_pool:
            break
    if not videos_all:
        return None
    videos = torch.cat(videos_all, dim=0)[:n_pool]
    task_arr = np.asarray(tasks[:len(videos)])
    ep_arr = np.asarray(eps[:len(videos)])
    verb_arr = np.asarray([_norm_verb(v) for v in verbs[:len(videos)]])

    mus = []
    for i in range(0, len(videos), decode_bs):
        mu, _ = _encode_mu_var(model, videos[i:i + decode_bs].to(device))
        mus.append(mu.cpu())
    mu = torch.cat(mus, dim=0)

    rng = np.random.RandomState(seed)
    known = verb_arr != "unknown"
    triples: List[Tuple[int, int, int]] = []
    for i in range(len(videos)):
        if not known[i]:
            continue
        same_task = task_arr == task_arr[i]
        pos_pool = np.where(same_task & known & (verb_arr == verb_arr[i]) & (ep_arr != ep_arr[i]))[0]
        opp_pool = np.where(same_task & known & (verb_arr != verb_arr[i]))[0]
        if len(pos_pool) and len(opp_pool):
            triples.append((i, int(rng.choice(pos_pool)), int(rng.choice(opp_pool))))
    if not triples:
        return None

    ap = model.action_part
    c_vals, t_lab, e_lab = [], [], []
    for s in range(0, len(triples), decode_bs):
        chunk = triples[s:s + decode_bs]
        idx = [c[0] for c in chunk]
        vb = videos[idx].to(device)
        gt = vb[:, 1]
        motion = (vb[:, 1] - vb[:, 0]).abs().flatten(1).mean(1)          # per-sample energy
        z_own = mu[idx].to(device)
        z_same = z_own.clone()
        z_same[:, :ap] = mu[[c[1] for c in chunk], :ap].to(device)
        z_opp = z_own.clone()
        z_opp[:, :ap] = mu[[c[2] for c in chunk], :ap].to(device)
        e_same = ((_decode_from_mu(model, vb, z_same)[:, 0] - gt) ** 2).flatten(1).mean(1)
        e_opp = ((_decode_from_mu(model, vb, z_opp)[:, 0] - gt) ** 2).flatten(1).mean(1)
        c = ((e_opp - e_same) / (motion + 1e-6)).cpu().numpy()
        c_vals.extend(c.tolist())
        t_lab.extend([task_arr[i] for i in idx])
        e_lab.extend([ep_arr[i] for i in idx])
    return {"c": np.asarray(c_vals), "task": np.asarray(t_lab), "ep": np.asarray(e_lab)}


def paired_c_delta(res_a: dict, res_b: dict, n_boot: int = 2000, seed: int = 42) -> dict:
    """ΔC = C_a − C_b with a task->episode paired hierarchical bootstrap.

    Uses the intersection of (task, ep) anchors evaluated by both models, and both
    are resampled with the SAME draw each replicate (that is what makes it paired).
    """
    def by_te(r):
        d: Dict[Tuple[str, str], List[float]] = {}
        for c, t, e in zip(r["c"], r["task"], r["ep"]):
            d.setdefault((t, e), []).append(float(c))
        return d

    A, B = by_te(res_a), by_te(res_b)
    keys = sorted(set(A) & set(B))
    if not keys:
        return {"delta_C": float("nan"), "n_anchors": 0.0}
    tasks: Dict[str, List[Tuple[str, str]]] = {}
    for t, e in keys:
        tasks.setdefault(t, []).append((t, e))
    tlist = sorted(tasks)
    rng = np.random.RandomState(seed)

    def C_of(d, tsel, esel):
        vals = []
        for ti, ekeys in zip(tsel, esel):
            ev = [float(np.mean(d[k])) for k in ekeys]
            vals.append(np.mean(ev))
        return float(np.mean(vals))

    base_sel = [tasks[t] for t in tlist]
    obs = C_of(A, range(len(tlist)), [tasks[tlist[i]] for i in range(len(tlist))]) - \
        C_of(B, range(len(tlist)), [tasks[tlist[i]] for i in range(len(tlist))])
    boots = np.empty(n_boot)
    for i in range(n_boot):
        ts = rng.choice(len(tlist), len(tlist), replace=True)
        es = []
        for ti in ts:
            pool = tasks[tlist[ti]]
            es.append([pool[j] for j in rng.choice(len(pool), len(pool), replace=True)])
        boots[i] = C_of(A, ts, es) - C_of(B, ts, es)
    return {
        "delta_C": obs,
        "delta_ci_lo": float(np.percentile(boots, 2.5)),
        "delta_ci_hi": float(np.percentile(boots, 97.5)),
        "C_a": C_of(A, range(len(tlist)), base_sel),
        "C_b": C_of(B, range(len(tlist)), base_sel),
        "n_anchors": float(len(keys)), "n_tasks": float(len(tlist)),
    }
