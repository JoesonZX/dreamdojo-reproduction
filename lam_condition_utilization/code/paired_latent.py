"""Phase 0.1 — paired-latent audit (顾问 §0.1).

逐 transition 比较两个 matched encoder(A0 α=0 vs A1 α=1)在**同一冻结 manifest pool**
上的 z_mu。分布汇总量(noise/SNR/eff-rank/类分离)相同,不代表逐样本映射
`μ(o_t,o_{t+1})` 相同;本审计测的是**动作坐标本身是否保持**。

输出(除注明外均在 action 子空间 [0:ap]):
  逐样本几何 : ‖μ_A0−μ_A1‖(raw + 归一化)、cosine、逐维 Pearson r
  表示相似性 : linear CKA、orthogonal-Procrustes 对齐后 R²、kNN 邻居重合率@k
  动作坐标   : 冻结 cross-encoder readout R² 矩阵(fit×eval ∈ {A0,A1})、
               latent 距离 vs GT 18D 动作距离的相关、frac(d_same<d_opp)

判读:cross-readout(A0→A1)≈ self 且 CKA/Procrustes 高 且逐维 r≈1 ⇒ 两 encoder 的
动作坐标基本一致 ⇒ 外部只看 μ 的 consumer 拿不到 A0/A1 差异,α 的收益在 decoder 侧。
若坐标显著不同(即便汇总统计相同)⇒ conditioning-data 变了,transfer 有可能。
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


def per_dim_corr(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Per-dimension Pearson r between matched columns of A and B ([N,D] → [D])."""
    Ac = A - A.mean(0, keepdims=True)
    Bc = B - B.mean(0, keepdims=True)
    num = (Ac * Bc).sum(0)
    den = np.sqrt((Ac ** 2).sum(0) * (Bc ** 2).sum(0)) + 1e-12
    return num / den


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two centred representations (rotation/scale-tolerant)."""
    Xc = X - X.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    yx = np.linalg.norm(Yc.T @ Xc, "fro") ** 2
    xx = np.linalg.norm(Xc.T @ Xc, "fro")
    yy = np.linalg.norm(Yc.T @ Yc, "fro")
    return float(yx / (xx * yy + 1e-12))


def procrustes_aligned_r2(X: np.ndarray, Y: np.ndarray) -> float:
    """Fraction of Var(Y) explained by the best orthogonal map of X onto Y.

    R = argmin_{R orthogonal} ‖X R − Y‖_F ; returns 1 − ‖XR−Y‖²/‖Y−mean(Y)‖².
    High value ⇒ the two latent clouds are the same up to a rigid rotation.
    """
    Xc = X - X.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    U, _, Vt = np.linalg.svd(Xc.T @ Yc, full_matrices=False)
    R = U @ Vt
    resid = float(((Xc @ R - Yc) ** 2).sum())
    denom = float((Yc ** 2).sum()) + 1e-12
    return 1.0 - resid / denom


def knn_overlap(X: np.ndarray, Y: np.ndarray, k: int = 10) -> float:
    """Mean fraction of shared k-nearest-neighbours between the two spaces (self excluded)."""
    def knn_idx(Z):
        Zc = Z.astype(np.float64)
        # squared euclidean distance matrix
        g = Zc @ Zc.T
        sq = np.diag(g)
        d = sq[:, None] + sq[None, :] - 2 * g
        np.fill_diagonal(d, np.inf)
        return np.argpartition(d, k, axis=1)[:, :k]

    ax, ay = knn_idx(X), knn_idx(Y)
    ov = [len(set(ax[i]) & set(ay[i])) / k for i in range(len(X))]
    return float(np.mean(ov))


def _readout_r2(mu_fit, mu_eval, act, groups, seed=42, test_frac=0.3, alpha=1.0):
    """Fit a ridge action-readout on mu_fit(train) and score on mu_eval(test).

    Same episode-grouped train/test split for both encoders (matched rows), and the
    scaler is fit on mu_fit-train, then applied to mu_eval-test — so mu_eval is scored
    in mu_fit's coordinate frame. Returns held-out R² (uniform average over 18 dims).
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    tr, te = next(gss.split(mu_fit, groups=groups))
    sc = StandardScaler().fit(mu_fit[tr])
    reg = Ridge(alpha=alpha).fit(sc.transform(mu_fit[tr]), act[tr])
    return float(reg.score(sc.transform(mu_eval[te]), act[te]))


def cross_readout_matrix(muA, muB, act, groups, ap, seed=42) -> Dict[str, float]:
    """Frozen-readout R² matrix: fit ∈ {A0,A1} × eval ∈ {A0,A1} on the action subspace."""
    A, B = muA[:, :ap], muB[:, :ap]
    return {
        "fitA0_evalA0": _readout_r2(A, A, act, groups, seed),   # self A0
        "fitA0_evalA1": _readout_r2(A, B, act, groups, seed),   # cross A0→A1
        "fitA1_evalA1": _readout_r2(B, B, act, groups, seed),   # self A1
        "fitA1_evalA0": _readout_r2(B, A, act, groups, seed),   # cross A1→A0
    }


def _dist_corr(mu, act, ap, n_pairs=20000, seed=0):
    """Pearson r between latent action-subspace pairwise distance and GT 18D distance."""
    rng = np.random.RandomState(seed)
    N = len(mu)
    i = rng.randint(0, N, n_pairs)
    j = rng.randint(0, N, n_pairs)
    m = i != j
    i, j = i[m], j[m]
    dl = np.linalg.norm(mu[i, :ap] - mu[j, :ap], axis=1)
    da = np.linalg.norm(act[i] - act[j], axis=1)
    return float(np.corrcoef(dl, da)[0, 1])


def _frac_same_lt_opp(mu, schemes, ap, scheme="dist"):
    """Fraction where the same-donor is closer than the opp-donor in latent action space."""
    sch = schemes[scheme]
    i = np.array([t[0] for t in sch]); s = np.array([t[1] for t in sch]); o = np.array([t[2] for t in sch])
    ds = np.linalg.norm(mu[i, :ap] - mu[s, :ap], axis=1)
    do = np.linalg.norm(mu[i, :ap] - mu[o, :ap], axis=1)
    return float(np.mean(ds < do))


def paired_audit(muA: np.ndarray, muB: np.ndarray, act: np.ndarray, groups: np.ndarray,
                 schemes: Dict, ap: int, k_knn: int = 10, seed: int = 42) -> dict:
    """Full A0-vs-A1 paired-latent audit. muA=A0(α=0), muB=A1(α=1), rows matched."""
    A, B = muA[:, :ap], muB[:, :ap]                     # action subspace
    diff = A - B
    norm_diff = np.linalg.norm(diff, axis=1)
    mu_norm = 0.5 * (np.linalg.norm(A, axis=1) + np.linalg.norm(B, axis=1)) + 1e-12
    cos = (A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12)
    pdc = per_dim_corr(A, B)
    pdc_full = per_dim_corr(muA, muB)

    return {
        "n": int(len(muA)), "ap": int(ap),
        # per-sample geometry (action subspace)
        "l2_diff_mean": float(norm_diff.mean()),
        "l2_diff_median": float(np.median(norm_diff)),
        "l2_diff_rel_mean": float((norm_diff / mu_norm).mean()),     # ‖Δ‖ / ‖μ‖
        "cosine_mean": float(cos.mean()),
        "cosine_median": float(np.median(cos)),
        "per_dim_corr_action_mean": float(pdc.mean()),
        "per_dim_corr_action_min": float(pdc.min()),
        "per_dim_corr_full_mean": float(pdc_full.mean()),
        # representational similarity
        "cka_action": linear_cka(A, B),
        "procrustes_r2_action": procrustes_aligned_r2(A, B),
        "knn_overlap_action": knn_overlap(A, B, k=k_knn),
        "k_knn": int(k_knn),
        # action-coordinate preservation
        "readout_r2": cross_readout_matrix(muA, muB, act, groups, ap, seed),
        "dist_corr_A0": _dist_corr(muA, act, ap, seed=seed),
        "dist_corr_A1": _dist_corr(muB, act, ap, seed=seed),
        "frac_same_lt_opp_A0_dist": _frac_same_lt_opp(muA, schemes, ap, "dist"),
        "frac_same_lt_opp_A1_dist": _frac_same_lt_opp(muB, schemes, ap, "dist"),
        "frac_same_lt_opp_A0_antipar": _frac_same_lt_opp(muA, schemes, ap, "antiparallel"),
        "frac_same_lt_opp_A1_antipar": _frac_same_lt_opp(muB, schemes, ap, "antiparallel"),
        # per-dim correlation array (action dims first) for the record
        "_per_dim_corr_full": pdc_full.tolist(),
    }
