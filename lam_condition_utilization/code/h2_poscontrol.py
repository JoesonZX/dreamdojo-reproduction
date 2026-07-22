"""H2 正对照(顾问第三轮 §1 硬门)。

问题:donor 分解已证 verb-based donor 未按动作距离排序(frac d_same<d_opp = 0.488)⇒
全臂 C≈0 既可能是"decoder 不用方向",也可能是"assay 用这批 donor 测不出方向"。
正对照用一个**已知强可控**的 decoder 走同一 anchor/scorer,看 C 能否变正来区分二者。

正对照 decoder = 训练无关的 **frame-delta 合成 oracle**:
    D_pos(o_t^anchor, donor) = o_t^anchor + (o1_donor − o0_donor)
按构造强可控;且 o_t^anchor 在 opp−same 差里精确抵消 ⇒
    C_pos ∝ mean((Δ_opp−Δ_true)²) − mean((Δ_same−Δ_true)²)   (纯 delta 空间,场景抵消)
这规避了顾问的警告:IDM 讲的是 encoding,此处是"会用注入动作的 decoder"。

三种 donor 方案(同一 anchor/scorer,施于 oracle 与真实 LAM decoder):
    verb   : 同 verb / 反 verb(旧构造;d_same<d_opp 仅 49% ⇒ 未排序)
    dist   : 按 18D GT 动作距离强排序(same=近四分位, opp=远四分位)⇒ 令前提为真
    random : same/opp 均随机同 task(负对照 ⇒ C 应≈0)

判读:
    oracle×dist ≫0 且 oracle×verb ≈0  ⇒ scorer 灵敏、旧 verb 构造是 bug;
    oracle×random ≈0                   ⇒ 负对照干净;
    LAM×dist ≈0(在 oracle×dist≫0 时)  ⇒ **真实 direction/action utilization deficit**。
    oracle×dist ≈0                     ⇒ assay/scorer 本身不灵敏,任何模型结论都不能下。

    CUDA_VISIBLE_DEVICES=<free> python code/eval_h2_poscontrol.py --device cuda:0
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch

import _apath  # noqa: F401  # shared A (lam_cdlam_optimization) substrate on sys.path
from benchmark_lam import _encode_mu_var, _norm_verb


def _build_schemes(task_arr, ep_arr, verb_arr, act, seed=42) -> Dict[str, List[Tuple[int, int, int]]]:
    """Same anchors for all schemes; only the donor selection differs.

    Anchors = the h2 set (known verb, same-task pos [same-verb diff-ep] and opp [diff-verb]
    pools both nonempty). Donor schemes:
      verb   : reproduce h2 (rng seed) — same-verb-diff-ep + diff-verb
      dist   : same = random from nearest 18D-action quartile (diff-ep); opp = random from
               farthest quartile — forces d_same < d_opp (tests action-MAGNITUDE use)
      random : same, opp = two random same-task known-verb candidates
      antiparallel : same = most PARALLEL (max cos), opp = most ANTI-PARALLEL (min cos),
               both drawn within a magnitude band [0.5,2]×‖a_anchor‖ so the two donors have
               matched magnitude → isolates signed DIRECTION use from magnitude use. A
               magnitude-only decoder gives C≈0 here; a direction-sensitive one gives C>0.
    """
    N = len(task_arr)
    known = verb_arr != "unknown"
    task_pool: Dict[str, List[int]] = defaultdict(list)
    for i in range(N):
        if known[i]:
            task_pool[task_arr[i]].append(i)

    rng_v = np.random.RandomState(seed)          # verb scheme (matches h2 order)
    rng_d = np.random.RandomState(seed + 7)      # dist scheme
    rng_r = np.random.RandomState(seed + 13)     # random scheme
    verb, dist, rand, antipar = [], [], [], []
    for i in range(N):
        if not known[i]:
            continue
        st = task_arr == task_arr[i]
        pos = np.where(st & known & (verb_arr == verb_arr[i]) & (ep_arr != ep_arr[i]))[0]
        opp = np.where(st & known & (verb_arr != verb_arr[i]))[0]
        if not (len(pos) and len(opp)):
            continue
        # verb scheme — identical draw order to h2_swap_utilization
        verb.append((i, int(rng_v.choice(pos)), int(rng_v.choice(opp))))

        cand = np.array([j for j in task_pool[task_arr[i]] if j != i])
        if len(cand) < 2:
            dist.append(verb[-1]); rand.append(verb[-1]); antipar.append(verb[-1]); continue
        d = np.linalg.norm(act[cand] - act[i], axis=1)
        order = cand[np.argsort(d)]
        q = max(1, len(order) // 4)
        near = [j for j in order[:q] if ep_arr[j] != ep_arr[i]] or list(order[:q])
        far = list(order[-q:])
        dist.append((i, int(rng_d.choice(near)), int(rng_d.choice(far))))
        two = rng_r.choice(cand, size=2, replace=len(cand) < 2)
        rand.append((i, int(two[0]), int(two[1])))

        # anti-parallel (signed DIRECTION test): same = nearest real donor to +a_i,
        # opp = nearest real donor to −a_i. NN-to-(−a_i) is the real donor best
        # approximating the anchor's NEGATED action → matched magnitude AND opposite
        # direction automatically. Global known-verb / diff-episode pool maximizes the
        # chance a genuine anti-parallel donor exists (same-task pools are too small).
        ai = act[i]
        gc = np.where(known & (ep_arr != ep_arr[i]))[0]
        if len(gc) >= 2:
            dpos = np.linalg.norm(act[gc] - ai, axis=1)
            dneg = np.linalg.norm(act[gc] + ai, axis=1)
            antipar.append((i, int(gc[int(np.argmin(dpos))]), int(gc[int(np.argmin(dneg))])))
        else:
            antipar.append(verb[-1])
    return {"verb": verb, "dist": dist, "random": rand, "antiparallel": antipar}


def _pool_stats(sch, act):
    """Median 18D donor distances + ordering fraction + cosine/magnitude diagnostics."""
    i = np.array([t[0] for t in sch]); s = np.array([t[1] for t in sch]); o = np.array([t[2] for t in sch])
    ds = np.linalg.norm(act[s] - act[i], axis=1)
    do = np.linalg.norm(act[o] - act[i], axis=1)
    magi = np.linalg.norm(act[i], axis=1) + 1e-8

    def cosr(a, b):
        return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)

    cs = cosr(act[i], act[s]); co = cosr(act[i], act[o])
    return {"d_same": float(np.median(ds)), "d_opp": float(np.median(do)),
            "frac_same_lt_opp": float(np.mean(ds < do)), "n": int(len(sch)),
            "cos_same": float(np.median(cs)), "cos_opp": float(np.median(co)),
            "magratio_same": float(np.median(np.linalg.norm(act[s], axis=1) / magi)),
            "magratio_opp": float(np.median(np.linalg.norm(act[o], axis=1) / magi)),
            "frac_opp_antiparallel": float(np.mean(co < -0.3))}


def _robust_C(diff: np.ndarray, P: np.ndarray) -> dict:
    """Robust motion-normalized contrast.

    Headline `C_pooled = Σ(E_opp−E_same)/ΣP` (ratio of means) avoids the per-sample
    division that blows up on near-static clips (consultant risk #3). `C_median` is the
    median of per-sample ratios; `C_mean` is the (unstable) mean, kept for continuity.
    """
    ratio = diff / (P + 1e-6)
    return {"C_pooled": float(diff.sum() / (P.sum() + 1e-12)),
            "C_median": float(np.median(ratio)),
            "C_mean": float(np.mean(ratio)),
            "raw_opp_minus_same": float(np.mean(diff)),
            "n": int(len(diff))}


def oracle_c(videos, sch, device, bs=32):
    """C for the frame-delta oracle (no model). Persistence-normalized, matches c_persist."""
    diffs, Ps = [], []
    for k in range(0, len(sch), bs):
        chunk = sch[k:k + bs]
        i = [t[0] for t in chunk]; s = [t[1] for t in chunk]; o = [t[2] for t in chunk]
        d_true = (videos[i, 1] - videos[i, 0]).to(device)
        d_same = (videos[s, 1] - videos[s, 0]).to(device)
        d_opp = (videos[o, 1] - videos[o, 0]).to(device)
        e_same = ((d_same - d_true) ** 2).mean(-1).flatten(1).mean(1)
        e_opp = ((d_opp - d_true) ** 2).mean(-1).flatten(1).mean(1)
        P = (d_true ** 2).mean(-1).flatten(1).mean(1)
        diffs.append((e_opp - e_same).cpu().numpy()); Ps.append(P.cpu().numpy())
    return _robust_C(np.concatenate(diffs), np.concatenate(Ps))


@torch.no_grad()
def lam_c(model, videos, mu, sch, device, bs=16):
    """C for the real LAM decoder under a given donor scheme (swap action subspace)."""
    ap, Dlat = model.action_part, model.latent_dim
    diffs, Ps = [], []
    for k in range(0, len(sch), bs):
        chunk = sch[k:k + bs]
        idx = [t[0] for t in chunk]
        vb = videos[idx].to(device)
        gt, prev = vb[:, 1], vb[:, 0]
        H, W = vb.shape[2:4]
        patches = model.encode(vb)["patches"]
        z_own = mu[idx].to(device)

        def swap(ids):
            z = z_own.clone(); z[:, :ap] = mu[ids, :ap].to(device); return z

        def dec(z):
            return model.decode_with(patches, z.reshape(z.shape[0], 1, 1, Dlat), H, W)[:, 0]

        e_same = ((dec(swap([t[1] for t in chunk])) - gt) ** 2).mean(-1).flatten(1).mean(1)
        e_opp = ((dec(swap([t[2] for t in chunk])) - gt) ** 2).mean(-1).flatten(1).mean(1)
        P = ((gt - prev) ** 2).mean(-1).flatten(1).mean(1)
        diffs.append((e_opp - e_same).cpu().numpy()); Ps.append(P.cpu().numpy())
    return _robust_C(np.concatenate(diffs), np.concatenate(Ps))
