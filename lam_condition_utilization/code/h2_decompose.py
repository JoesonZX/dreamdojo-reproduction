"""H2 分解 + posterior signal/noise 分解(零训练诊断,顾问第三轮回复 §1、§3)。

阶段 B 的主指标 C = E[(MSE_opp − MSE_same)/motion] 对**全臂** ≈ 0。在把它读成
"decoder 不用方向"之前,必须先排除 assay 本身的 construct-validity 风险:
  1. donor 语义不保证物理距离成立(同 verb donor 未必比反 verb donor 更近);
  2. 误差差值可掩盖响应(两 donor 输出不同图像,但相对 GT 同样错 ⇒ C=0);
  3. 归一化量纲不一致(旧实现分子是 squared error、分母是 mean-abs 帧差)。

本模块对每个 anchor 输出完整分解,供 §1 判读矩阵:
  E_own / E_same / E_opp / E_zero / E_shuffle   —— 各 latent 变体对 GT 的 MSE
  R_out = ‖D(o,z_opp) − D(o,z_same)‖            —— 直接输出响应量(与 GT 无关)
  同量纲归一化:以 persistence P = MSE(o_t, o_{t+1}) 作分母(修 #3 量纲)
  前景(运动 mask)/ 背景 分别统计
  GT-action donor 距离:验证 d(a_anchor,a_same) < d(a_anchor,a_opp)(修 #1 前提)

posterior_snr:逐维 signal=Var(mu)/noise=E[exp(logvar)]/SNR、逐维 KL、active dims、
same/opp 类中心距离相对 posterior noise scale —— 检验"采样噪声盖过方向"假设的前置量。
注意 z_var 是 **log-variance**(model.encode:std = exp(0.5·z_var),prior N(0,I))。

全部在**冻结评测 manifest** 上评估,latent 全走 z_mu(确定性,无采样)。
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch

import _apath  # noqa: F401  # shared A (lam_cdlam_optimization) substrate on sys.path
from benchmark_lam import _effective_rank, _encode_mu_var, _norm_verb


def _reduce(x: torch.Tensor, fg: torch.Tensor, bg: torch.Tensor,
            fg_n: torch.Tensor, bg_n: torch.Tensor):
    """Per-sample mean of a nonneg [B,H,W] map over all / foreground / background."""
    a = x.flatten(1).mean(1)
    f = (x * fg).flatten(1).sum(1) / fg_n
    b = (x * bg).flatten(1).sum(1) / bg_n
    return a.cpu().numpy(), f.cpu().numpy(), b.cpu().numpy()


@torch.no_grad()
def h2_decompose(model, loader, device, n_pool: int = 1500, decode_bs: int = 16,
                 fg_q: float = 0.90, seed: int = 42) -> dict | None:
    """Full same/opp/own/zero/shuffle error decomposition on the frozen manifest.

    Triples (anchor, same-verb donor, opp-verb donor) are built identically to
    ``h2_utilization.h2_swap_utilization`` (same pool, seed, order) so that
    ``c_h2_repro`` reproduces the reported H2 C as a continuity check.
    """
    videos_all, tasks, eps, verbs, actions = [], [], [], [], []
    seen = 0
    for batch in loader:
        videos_all.append(batch["videos"])
        tasks.extend(batch["task_id"])
        eps.extend(batch["ep_id"])
        verbs.extend(batch.get("verb_id", ["unknown"] * batch["videos"].shape[0]))
        actions.append(batch["action"])
        seen += batch["videos"].shape[0]
        if seen >= n_pool:
            break
    if not videos_all:
        return None
    videos = torch.cat(videos_all, dim=0)[:n_pool]
    N = len(videos)
    task_arr = np.asarray(tasks[:N])
    ep_arr = np.asarray(eps[:N])
    verb_arr = np.asarray([_norm_verb(v) for v in verbs[:N]])
    act = torch.cat(actions, dim=0)[:N]                        # [N,18] normalized

    # encode the pool once (z_mu, no sampling in eval mode)
    mus = []
    for i in range(0, N, decode_bs):
        mu, _ = _encode_mu_var(model, videos[i:i + decode_bs].to(device))
        mus.append(mu.cpu())
    mu = torch.cat(mus, dim=0)                                 # [N, D]
    ap, Dlat = model.action_part, model.latent_dim

    # anchor triples — identical construction to h2_swap_utilization
    rng = np.random.RandomState(seed)
    known = verb_arr != "unknown"
    triples: List[Tuple[int, int, int]] = []
    task_pool: Dict[str, List[int]] = defaultdict(list)
    for i in range(N):
        if known[i]:
            task_pool[task_arr[i]].append(i)
    for i in range(N):
        if not known[i]:
            continue
        same_task = task_arr == task_arr[i]
        pos_pool = np.where(same_task & known & (verb_arr == verb_arr[i]) & (ep_arr != ep_arr[i]))[0]
        opp_pool = np.where(same_task & known & (verb_arr != verb_arr[i]))[0]
        if len(pos_pool) and len(opp_pool):
            triples.append((i, int(rng.choice(pos_pool)), int(rng.choice(opp_pool))))
    if not triples:
        return None

    # within-task shuffle donor (generic-usage probe, verb-agnostic)
    srng = np.random.RandomState(seed + 1)
    shuf_of = {}
    for i, _, _ in triples:
        cand = [j for j in task_pool[task_arr[i]] if j != i]
        shuf_of[i] = int(srng.choice(cand)) if cand else i

    rec = defaultdict(list)
    for s in range(0, len(triples), decode_bs):
        chunk = triples[s:s + decode_bs]
        idx = [c[0] for c in chunk]
        vb = videos[idx].to(device)
        gt, prev = vb[:, 1], vb[:, 0]
        H, W = vb.shape[2:4]
        enc = model.encode(vb)
        patches = enc["patches"]

        z_own = mu[idx].to(device)

        def swap(donor_ids):
            z = z_own.clone()
            z[:, :ap] = mu[donor_ids, :ap].to(device)
            return z

        z_same = swap([c[1] for c in chunk])
        z_opp = swap([c[2] for c in chunk])
        z_shuf = swap([shuf_of[c[0]] for c in chunk])
        z_zero = z_own.clone()
        z_zero[:, :ap] = 0.0

        def dec(z):
            zr = z.reshape(z.shape[0], 1, 1, Dlat)
            return model.decode_with(patches, zr, H, W)[:, 0]     # [B,H,W,C]

        p_own, p_same, p_opp = dec(z_own), dec(z_same), dec(z_opp)
        p_zero, p_shuf = dec(z_zero), dec(z_shuf)

        # motion foreground mask (per-sample top-(1-fg_q) motion pixels, non-static)
        motion = (gt - prev).abs().mean(-1)                      # [B,H,W]
        thr = torch.quantile(motion.flatten(1), fg_q, dim=1)
        fg = (motion >= thr[:, None, None]) & (motion > 1e-3)
        bg = ~fg
        fg_n = fg.flatten(1).sum(1).clamp_min(1)
        bg_n = bg.flatten(1).sum(1).clamp_min(1)
        persist = ((gt - prev) ** 2).mean(-1)                    # [B,H,W] same units as SE

        def se(p):
            return ((p - gt) ** 2).mean(-1)                      # [B,H,W]

        for name, p in [("own", p_own), ("same", p_same), ("opp", p_opp),
                        ("zero", p_zero), ("shuf", p_shuf)]:
            a, f, b = _reduce(se(p), fg, bg, fg_n, bg_n)
            rec[f"E_{name}_all"].append(a)
            rec[f"E_{name}_fg"].append(f)
            rec[f"E_{name}_bg"].append(b)

        # direct output response ‖D(o,z_opp) − D(o,z_same)‖ (RMS over pixels)
        resp = ((p_opp - p_same) ** 2).mean(-1)                  # [B,H,W]
        ra, rf, rb = _reduce(resp, fg, bg, fg_n, bg_n)
        rec["Rout_all"].append(np.sqrt(ra))
        rec["Rout_fg"].append(np.sqrt(rf))
        rec["Rout_bg"].append(np.sqrt(rb))

        pa, _, _ = _reduce(persist, fg, bg, fg_n, bg_n)
        rec["persist"].append(pa)
        rec["fg_frac"].append((fg.flatten(1).float().mean(1)).cpu().numpy())

        # GT-action donor distances (18-D normalized): premise d_same < d_opp
        a_i = act[idx]
        d_same = (a_i - act[[c[1] for c in chunk]]).norm(dim=1).numpy()
        d_opp = (a_i - act[[c[2] for c in chunk]]).norm(dim=1).numpy()
        rec["d_same"].append(d_same)
        rec["d_opp"].append(d_opp)
        rec["task"].append(np.asarray([task_arr[i] for i in idx]))
        rec["ep"].append(np.asarray([ep_arr[i] for i in idx]))

    out = {k: np.concatenate(v) for k, v in rec.items()}
    return out


def summarize_decompose(r: dict) -> dict:
    """Per-arm scalar summary + the §1 judgment-matrix readouts."""
    eps = 1e-6
    P = r["persist"] + eps

    def mm(x):
        return {"mean": float(np.mean(x)), "median": float(np.median(x))}

    s = {
        "n_anchors": int(len(r["persist"])),
        "n_tasks": int(len(np.unique(r["task"]))),
        "fg_frac_mean": float(np.mean(r["fg_frac"])),
        # error levels
        "E_own": mm(r["E_own_all"]), "E_same": mm(r["E_same_all"]),
        "E_opp": mm(r["E_opp_all"]), "E_zero": mm(r["E_zero_all"]),
        "E_shuf": mm(r["E_shuf_all"]),
        # persistence-normalized contrasts (same units → dimensionless)
        "c_persist": mm((r["E_opp_all"] - r["E_same_all"]) / P),       # direction, corrected H2
        "c_persist_fg": mm((r["E_opp_fg"] - r["E_same_fg"]) / P),
        "usage_zero": mm((r["E_zero_all"] - r["E_own_all"]) / P),      # generic: zero vs own
        "usage_shuf": mm((r["E_shuf_all"] - r["E_own_all"]) / P),      # generic: shuffle vs own
        # legacy H2 C reproduction (mean-abs motion denom) — continuity check
        # (recomputed by caller if the mean-abs motion is retained; here persistence proxy)
        # direct output response
        "Rout": mm(r["Rout_all"]), "Rout_fg": mm(r["Rout_fg"]), "Rout_bg": mm(r["Rout_bg"]),
        # persistence level
        "persist": mm(r["persist"]),
        # donor-distance premise
        "donor_frac_same_lt_opp": float(np.mean(r["d_same"] < r["d_opp"])),
        "d_same": mm(r["d_same"]), "d_opp": mm(r["d_opp"]),
    }
    return s


@torch.no_grad()
def posterior_snr(model, loader, device, n_pool: int = 3000, encode_bs: int = 32,
                  kl_thresh: float = 0.01, seed: int = 42) -> dict:
    """Per-dim posterior signal/noise/KL + same-vs-opp separation relative to noise.

    signal_j = Var_data(mu_j);  noise_j = E_data[exp(logvar_j)];  SNR_j = signal/noise.
    KL_j = 0.5·E[mu^2 + var − logvar − 1]  (Gaussian posterior vs N(0,1) prior).
    active dims counted by KL_j > kl_thresh — the correct answer to "total KL ≠ active dims".
    """
    mus, lvars, tasks, verbs = [], [], [], []
    seen = 0
    for batch in loader:
        v = batch["videos"].to(device)
        mu, lv = _encode_mu_var(model, v)
        mus.append(mu.cpu()); lvars.append(lv.cpu())
        tasks.extend(batch["task_id"])
        verbs.extend(batch.get("verb_id", ["unknown"] * v.shape[0]))
        seen += v.shape[0]
        if seen >= n_pool:
            break
    mu = torch.cat(mus, dim=0)[:n_pool].numpy()                # [N,D]
    lv = torch.cat(lvars, dim=0)[:n_pool].numpy()
    var = np.exp(lv)
    ap = model.action_part
    N = len(mu)

    signal = mu.var(axis=0)                                    # [D]
    noise = var.mean(axis=0)                                   # [D]
    snr = signal / (noise + 1e-8)
    kl = 0.5 * (mu ** 2 + var - lv - 1.0).mean(axis=0)         # [D]
    active = kl > kl_thresh

    # same/opp class-center separation in the action subspace, vs posterior noise scale
    verb_arr = np.asarray([_norm_verb(x) for x in verbs[:N]])
    task_arr = np.asarray(tasks[:N])
    seps = []
    for t in np.unique(task_arr):
        m = task_arr == t
        vs = [x for x in np.unique(verb_arr[m]) if x != "unknown"]
        if len(vs) != 2:
            continue
        a_mask = m & (verb_arr == vs[0])
        b_mask = m & (verb_arr == vs[1])
        if a_mask.sum() < 2 or b_mask.sum() < 2:
            continue
        cA = mu[a_mask][:, :ap].mean(0)
        cB = mu[b_mask][:, :ap].mean(0)
        seps.append(float(np.linalg.norm(cA - cB)))
    noise_scale = float(np.sqrt(noise[:ap].mean()))            # posterior std scale (action)
    sep_mean = float(np.mean(seps)) if seps else float("nan")

    eff_rank_a, top5_a = _effective_rank(mu[:, :ap])
    eff_rank_all, _ = _effective_rank(mu)

    return {
        "n": N,
        "kl_total": float(kl.sum()),
        "kl_total_action": float(kl[:ap].sum()),
        "kl_total_env": float(kl[ap:].sum()),
        "active_dims": int(active.sum()),
        "active_dims_action": int(active[:ap].sum()),
        "active_dims_env": int(active[ap:].sum()),
        "eff_rank_action": float(eff_rank_a),
        "eff_rank_all": float(eff_rank_all),
        "snr_action_mean": float(snr[:ap].mean()),
        "snr_action_max": float(snr[:ap].max()),
        "noise_scale_action": noise_scale,
        "sep_same_opp_action": sep_mean,
        "sep_over_noise": float(sep_mean / (noise_scale + 1e-8)) if seps else float("nan"),
        "n_reversible_tasks": len(seps),
        # per-dim arrays (action subspace first) for optional .npz dump
        "_signal": signal, "_noise": noise, "_snr": snr, "_kl": kl,
    }
