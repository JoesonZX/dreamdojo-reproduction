"""条件方向 probe —— confirmatory 版本(阶段 B0)。

替代 `benchmark_lam._conditional_direction_probe` 的 exploratory 实现。
外部审阅指出后者有六处口径问题,全部在此修正:

| 旧问题 | 修正 |
|---|---|
| `n_perm=20` 且报 `p=0.00` | Monte Carlo p 的最小合法值是 `(0+1)/(B+1)`;此处用 `(b+1)/(B+1)`,B 可配(confirmatory ≥999) |
| CI 报的是 raw gain | 改报 **null-centered** gain 的 CI,并支持**模型间 paired 差值**的 CI |
| NLL 先按帧拼接再平均 | **先 episode 平均,再 task 平均**;bootstrap 先抽 task 再抽 episode(hierarchical) |
| `C=1.0` 固定 | 在 outer GroupKFold 的训练折内做 **nested group CV** 选 C |
| `auc_full>0.5` 当门 | AUC 降为辅助,只报 `AUC_full − AUC_base`;主统计是 null-calibrated held-out NLL |
| 各模型置换不共享 | 支持传入 `rng_seed` 固定 folds/permutations,实现 **common random numbers** |

核心量:
    G_m  = mean_task[ mean_episode[ NLL_base − NLL_full ] ]
    P_m  = G_m − E_perm[G_m]          # null-centered,这才是"无信息=0"的量
    ΔP   = P_method − P_control        # H1 的主对比(方法 vs 匹配非方向对照)

⚠️ 原始 G 天然偏负:full probe 比 base 多 32 个含噪特征、要付过拟合代价。
所以任何"G>0"式判读都是错的,必须减去置换 null。
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def _fit_fold(X_tr, y_tr, X_te, y_te, Cs, inner_groups, seed):
    """Logistic probe with nested group-CV over C. Returns per-sample held-out NLL."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import StandardScaler

    best_C, best_nll = Cs[0], np.inf
    n_inner = min(3, len(np.unique(inner_groups)))
    if n_inner >= 2 and len(Cs) > 1:
        for C in Cs:
            fold_nll = []
            for itr, ite in StratifiedGroupKFold(n_splits=n_inner).split(X_tr, y_tr, inner_groups):
                if len(np.unique(y_tr[itr])) < 2 or len(np.unique(y_tr[ite])) < 2:
                    continue
                sc = StandardScaler().fit(X_tr[itr])
                clf = LogisticRegression(max_iter=2000, C=C).fit(sc.transform(X_tr[itr]), y_tr[itr])
                p = np.clip(clf.predict_proba(sc.transform(X_tr[ite]))[:, 1], 1e-6, 1 - 1e-6)
                fold_nll.append(-(y_tr[ite] * np.log(p) + (1 - y_tr[ite]) * np.log(1 - p)).mean())
            if fold_nll and np.mean(fold_nll) < best_nll:
                best_nll, best_C = float(np.mean(fold_nll)), C
    sc = StandardScaler().fit(X_tr)
    clf = LogisticRegression(max_iter=2000, C=best_C).fit(sc.transform(X_tr), y_tr)
    p = np.clip(clf.predict_proba(sc.transform(X_te))[:, 1], 1e-6, 1 - 1e-6)
    return -(y_te * np.log(p) + (1 - y_te) * np.log(1 - p)), p


def _task_gain(z_static, z_trans, y, ep, Cs, seed, y_perm=None):
    """Per-episode (NLL_base − NLL_full) for one task. Returns {ep: gain} plus AUC bits."""
    from sklearn.model_selection import StratifiedGroupKFold

    yy = y if y_perm is None else y_perm
    if len(np.unique(yy)) < 2:
        return None
    # StratifiedGroupKFold needs each test fold to hold BOTH verbs, so n_splits is
    # capped by the SMALLER verb's episode count — not the total episode count.
    # (The old GroupKFold split by total episodes; with ~2 episodes/verb every fold's
    # test set ended up single-verb and was skipped, collapsing 57 tasks down to 5.)
    groups_per_class = min(len(np.unique(ep[yy == c])) for c in (0, 1))
    if groups_per_class < 2:
        return None
    n_splits = min(5, groups_per_class)
    delta = z_trans - z_static
    X_full = np.concatenate([z_static, delta], axis=1)
    X_base = np.concatenate([z_static, np.zeros_like(delta)], axis=1)
    per_ep: Dict[str, List[float]] = {}
    auc_rows = []
    for tr, te in StratifiedGroupKFold(n_splits=n_splits).split(X_full, yy, ep):
        if len(np.unique(yy[tr])) < 2 or len(np.unique(yy[te])) < 2:
            continue
        nb, pb = _fit_fold(X_base[tr], yy[tr], X_base[te], yy[te], Cs, ep[tr], seed)
        nf, pf = _fit_fold(X_full[tr], yy[tr], X_full[te], yy[te], Cs, ep[tr], seed)
        for k, e in enumerate(ep[te]):
            per_ep.setdefault(e, []).append(float(nb[k] - nf[k]))
        auc_rows.append((yy[te], pb, pf))
    if not per_ep:
        return None
    return {e: float(np.mean(v)) for e, v in per_ep.items()}, auc_rows


def conditional_direction_probe(z_trans, z_static, task, verb, ep, *,
                                n_perm: int = 999, n_boot: int = 2000,
                                Cs: Sequence[float] = (0.03, 0.3, 3.0),
                                seed: int = 42, norm_verb=None) -> dict:
    """Null-calibrated conditional direction probe with hierarchical bootstrap.

    Returns per-episode gains too, so several models can be compared with a PAIRED
    bootstrap afterwards (see `paired_delta`) without refitting anything.
    """
    norm_verb = norm_verb or (lambda v: str(v).strip().lower().replace("_", " "))
    verb_arr = np.asarray([norm_verb(v) for v in verb])
    task_arr, ep_arr = np.asarray(task), np.asarray(ep)
    rng = np.random.RandomState(seed)

    def collect(perm: bool):
        """{task: {episode: gain}} — perm=True permutes episode->verb inside each task."""
        out = {}
        for t in np.unique(task_arr):
            m = task_arr == t
            uv = [v for v in np.unique(verb_arr[m]) if v != "unknown"]
            if len(uv) != 2:
                continue
            keep = m & np.isin(verb_arr, uv)
            y = (verb_arr[keep] == uv[0]).astype(int)
            if y.sum() < 8 or (len(y) - y.sum()) < 8:
                continue
            e = ep_arr[keep]
            y_perm = None
            if perm:
                ue = np.unique(e)
                lab = dict(zip(ue, rng.permutation([y[e == x][0] for x in ue])))
                y_perm = np.asarray([lab[x] for x in e])
                if len(np.unique(y_perm)) < 2:
                    continue
            r = _task_gain(z_static[keep].astype(np.float64), z_trans[keep].astype(np.float64),
                           y, e, Cs, seed, y_perm=y_perm)
            if r is not None:
                out[t] = r[0] if not perm else r[0]
                if not perm:
                    out.setdefault("_auc", []).extend(r[1])
        return out

    obs = collect(perm=False)
    auc_rows = obs.pop("_auc", [])
    if not obs:
        return {"dirprobe_n_tasks": 0.0}

    def two_level_mean(d):
        return float(np.mean([np.mean(list(eps.values())) for eps in d.values()]))

    G = two_level_mean(obs)
    nulls = []
    for _ in range(n_perm):
        nd = collect(perm=True)
        if nd:
            nulls.append(two_level_mean(nd))
    nulls = np.asarray(nulls) if nulls else np.asarray([np.nan])
    null_mean = float(np.nanmean(nulls))
    P = G - null_mean
    # Monte Carlo p with the mandatory +1 correction (min attainable = 1/(B+1))
    b = int(np.sum(nulls >= G))
    p_val = (b + 1) / (len(nulls) + 1)

    # hierarchical bootstrap: resample TASKS, then EPISODES within task
    tasks = sorted(obs)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        tsel = rng.choice(len(tasks), len(tasks), replace=True)
        vals = []
        for ti in tsel:
            eps = list(obs[tasks[ti]].values())
            vals.append(float(np.mean(rng.choice(eps, len(eps), replace=True))))
        boots[i] = np.mean(vals)
    out = {
        "dirprobe_G": G, "dirprobe_null_mean": null_mean, "dirprobe_P": P,
        "dirprobe_P_ci_lo": float(np.percentile(boots, 2.5) - null_mean),
        "dirprobe_P_ci_hi": float(np.percentile(boots, 97.5) - null_mean),
        "dirprobe_perm_p": p_val, "dirprobe_n_perm": float(len(nulls)),
        "dirprobe_n_tasks": float(len(tasks)),
        "dirprobe_n_episodes": float(sum(len(v) for v in obs.values())),
        "_per_episode": obs,        # for paired model comparisons
    }
    if auc_rows:
        from sklearn.metrics import roc_auc_score
        ys = np.concatenate([a[0] for a in auc_rows])
        pb = np.concatenate([a[1] for a in auc_rows])
        pf = np.concatenate([a[2] for a in auc_rows])
        try:                        # secondary only — AUC_full is inflated by Z_static
            out["dirprobe_auc_base"] = float(roc_auc_score(ys, pb))
            out["dirprobe_auc_full"] = float(roc_auc_score(ys, pf))
            out["dirprobe_auc_delta"] = out["dirprobe_auc_full"] - out["dirprobe_auc_base"]
        except ValueError:
            pass
    return out


def paired_delta(res_a: dict, res_b: dict, n_boot: int = 2000, seed: int = 42) -> dict:
    """H1 main contrast: ΔP = P_a − P_b with a PAIRED hierarchical bootstrap.

    Only tasks/episodes present in both models are used, and each bootstrap replicate
    resamples the same tasks/episodes for both — that is what makes the CI paired.
    """
    A, B = res_a["_per_episode"], res_b["_per_episode"]
    tasks = sorted(set(A) & set(B))
    if not tasks:
        return {"delta_P": float("nan")}
    rng = np.random.RandomState(seed)
    shared = {t: sorted(set(A[t]) & set(B[t])) for t in tasks}
    tasks = [t for t in tasks if len(shared[t]) >= 1]

    def stat(tsel, esel):
        va, vb = [], []
        for ti, es in zip(tsel, esel):
            t = tasks[ti]
            va.append(np.mean([A[t][e] for e in es]))
            vb.append(np.mean([B[t][e] for e in es]))
        return float(np.mean(va) - np.mean(vb))

    obs = stat(range(len(tasks)), [shared[t] for t in tasks])
    boots = np.empty(n_boot)
    for i in range(n_boot):
        tsel = rng.choice(len(tasks), len(tasks), replace=True)
        esel = [list(rng.choice(shared[tasks[ti]], len(shared[tasks[ti]]), replace=True))
                for ti in tsel]
        boots[i] = stat(tsel, esel)
    # null offsets cancel only if both used the same permutation seed; report both
    d_null = res_a.get("dirprobe_null_mean", 0.0) - res_b.get("dirprobe_null_mean", 0.0)
    return {
        "delta_G": obs, "delta_null": d_null, "delta_P": obs - d_null,
        "delta_ci_lo": float(np.percentile(boots, 2.5) - d_null),
        "delta_ci_hi": float(np.percentile(boots, 97.5) - d_null),
        "n_tasks_shared": float(len(tasks)),
    }
