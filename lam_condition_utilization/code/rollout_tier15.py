"""Phase 1 — Tier-1.5:原 LAM decoder 的 K 步自回归 rollout(within-LAM functional validation)。

顾问第四轮 §3 / prereg §3。**明确标 within-LAM,不作下游 WM 有效主张。** 在 Eval-A 上跑
(Eval-B 密封留给 Phase-2)。用原 LAM decoder `D_A0/D_A1`,不新训。

暴露顾问点名的三种失败:
  递归漂移   —— own-latent rollout 的 horizon PSNR 斜率;
  静止复制   —— 输出运动 / GT 运动 之比(≪1 = 只复制 o_t);
  符号失效   —— own vs neg(动作子空间取反)rollout 对 GT 的 PSNR 差(sign use)。

latent 全走 z_μ(确定性)。pairwise 编码(每相邻对 [f_i,f_{i+1}] 单独 encode,忠于 2-frame 训练)。
K_max 步 rollout 一次,horizon={4,8,16} 作前缀读出。rollout 在 clip 维度 batch、K 维度顺序。
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

_ADAWORLD = Path(__file__).resolve().parents[2] / "code" / "adaworld" / "lam"
if str(_ADAWORLD) not in sys.path:
    sys.path.insert(0, str(_ADAWORLD))
from lam.modules.blocks import patchify  # noqa: E402

try:
    import decord  # noqa: E402
    _HAVE_DECORD = True
except Exception:
    _HAVE_DECORD = False


def sample_clips(ds, n_clips: int, K: int, skip: int, seed: int = 42,
                 prefer_reversible: bool = True) -> List[dict]:
    """Deterministically pick episodes with ≥ K·skip+1 frames; centre the window.

    Reuses ds._index (one entry per (video,t)); we take one clip per episode, centred,
    reading frames [t, t+skip, …, t+K·skip]. Prefers reversible-task episodes so the
    same pool feeds target-transfer donors.
    """
    need = K * skip + 1
    by_ep = {}
    for rec in ds._index:
        vp, t, task, ep, hdf5, n = rec
        if n >= need and ep not in by_ep:
            v = ds._verb_map.get(ep)
            v = str(v).strip().lower().replace("_", " ") if v else "unknown"
            by_ep[ep] = dict(vp=vp, hdf5=hdf5, task=task, ep=ep, verb=v, n=n)
    # reversible-task set
    tv = defaultdict(set)
    for e in by_ep.values():
        if e["verb"] != "unknown":
            tv[e["task"]].add(e["verb"])
    rev_tasks = {t for t, vs in tv.items() if len(vs) >= 2}
    eps = sorted(by_ep)
    if prefer_reversible:
        eps.sort(key=lambda e: (by_ep[e]["task"] not in rev_tasks, e))
    rng = np.random.default_rng(seed)
    eps = eps[:max(n_clips * 2, n_clips)]
    pick = sorted(rng.choice(eps, size=min(n_clips, len(eps)), replace=False).tolist())
    clips = []
    for e in pick:
        info = by_ep[e]
        t0 = max(0, (info["n"] - K * skip) // 2)
        clips.append({**info, "t0": int(t0), "K": K, "skip": skip,
                      "reversible": info["task"] in rev_tasks})
    return clips


def load_clip(ds, spec: dict) -> Dict[str, torch.Tensor]:
    """Read K+1 frames at stride skip + K per-step 18D GT actions for one clip."""
    K, skip, t0 = spec["K"], spec["skip"], spec["t0"]
    idxs = [t0 + i * skip for i in range(K + 1)]
    raw = None
    if _HAVE_DECORD:
        try:
            vr = decord.VideoReader(str(spec["vp"]), ctx=decord.cpu(0))
            raw = vr.get_batch(idxs).asnumpy()                    # [K+1,H0,W0,C] one open
        except Exception:
            raw = None
    frames = []
    for j, fi in enumerate(idxs):
        f = raw[j] if raw is not None else ds._load_frame_opencv(spec["vp"], fi)
        frames.append(ds._center_crop_resize(f).astype(np.float32) / 255.0)
    frames = torch.from_numpy(np.stack(frames, axis=0))            # [K+1,H,W,C]
    acts = []
    try:
        for i in range(K):
            acts.append(ds._compute_action(spec["hdf5"], t0 + i * skip, skip))
        actions = torch.stack(acts, dim=0)                        # [K,18]
    except Exception:
        actions = torch.zeros(K, 18)
    return {"frames": frames, "actions": actions}


@torch.no_grad()
def encode_seq(model, frames: torch.Tensor, device, bs: int = 64) -> torch.Tensor:
    """Pairwise-encode a [K+1,H,W,C] clip → z_μ [K, latent] (faithful to 2-frame training)."""
    K = frames.shape[0] - 1
    pairs = torch.stack([frames[i:i + 2] for i in range(K)], dim=0)   # [K,2,H,W,C]
    mus = []
    for i in range(0, K, bs):
        out = model.encode(pairs[i:i + bs].to(device))
        mus.append(out["z_mu"].cpu())
    return torch.cat(mus, dim=0)                                      # [K, latent]


@torch.no_grad()
def rollout(model, f0: torch.Tensor, zseq: torch.Tensor, device, bs: int = 16) -> torch.Tensor:
    """Autoregressive K-step rollout, batched over clips, sequential over steps.

    f0: [N,H,W,C] initial true frames. zseq: [N,K,latent]. Returns [N,K,H,W,C].
    Step i: ô_{i+1} = D(ô_i, z_i); ô_0 = f0. Only the decoder runs (no re-encode).
    """
    N, K = zseq.shape[0], zseq.shape[1]
    Dlat = model.latent_dim
    out = torch.empty(N, K, *f0.shape[1:], dtype=torch.float32)
    for s in range(0, N, bs):
        cur = f0[s:s + bs].to(device)                                # [b,H,W,C]
        z = zseq[s:s + bs].to(device)                                # [b,K,latent]
        H, W = cur.shape[1:3]
        for i in range(K):
            vid = torch.stack([cur, cur], dim=1)                     # [b,2,H,W,C] (2nd unused)
            patches = patchify(vid, model.patch_size)
            zr = z[:, i].reshape(z.shape[0], 1, 1, Dlat)
            cur = model.decode_with(patches, zr, H, W)[:, 0]         # [b,H,W,C]
            out[s:s + bs, i] = cur.cpu()
    return out


def _psnr(mse: np.ndarray) -> np.ndarray:
    return -10.0 * np.log10(np.clip(mse, 1e-12, None))


def rollout_metrics(frames_all: torch.Tensor, roll_own, roll_zero, roll_neg) -> dict:
    """Horizon metrics from own/zero/neg rollouts vs GT.

    frames_all: [N,K+1,H,W,C] (GT, incl. f0). roll_*: [N,K,H,W,C].
    Returns per-horizon-step arrays (length K): PSNR_{own,zero,neg}, motion ratio, responses.
    """
    gt = frames_all[:, 1:]                                            # [N,K,H,W,C] targets
    prev = frames_all[:, :-1]                                         # [N,K,H,W,C] o_t per step
    N, K = gt.shape[0], gt.shape[1]

    def mse_k(pred):                                                  # [N,K]
        return ((pred - gt) ** 2).flatten(2).mean(2).numpy()

    e_own, e_zero, e_neg = mse_k(roll_own), mse_k(roll_zero), mse_k(roll_neg)
    # motion: output step-to-step vs GT step-to-step (static-copy diagnostic)
    def motion(seq):                                                 # [N,K]
        # step 0 uses f0 as previous; later steps use own previous output
        prev_own = torch.cat([frames_all[:, :1], seq[:, :-1]], dim=1)
        return (seq - prev_own).abs().flatten(2).mean(2).numpy()
    out_motion = motion(roll_own)
    gt_motion = (gt - prev).abs().flatten(2).mean(2).numpy()
    # decoder responses (mechanism): does latent / its sign move the output?
    resp_zero = ((roll_own - roll_zero) ** 2).flatten(2).mean(2).sqrt().numpy()
    resp_neg = ((roll_own - roll_neg) ** 2).flatten(2).mean(2).sqrt().numpy()

    # spatial total variation (artifact/drift guard): distinguishes controlled motion
    # from blur/drift. Compare to GT-frame TV as the natural reference.
    def tv(seq):                                                    # [N,K]
        dh = (seq[:, :, 1:] - seq[:, :, :-1]).abs().flatten(2).mean(2)
        dw = (seq[:, :, :, 1:] - seq[:, :, :, :-1]).abs().flatten(2).mean(2)
        return (0.5 * (dh + dw)).numpy()
    tv_own = tv(roll_own)
    tv_gt = tv(gt)

    def col(a):
        return a.mean(0).tolist()
    return {
        "n_clips": int(N), "K": int(K),
        "psnr_own": col(_psnr(e_own)), "psnr_zero": col(_psnr(e_zero)), "psnr_neg": col(_psnr(e_neg)),
        "mse_own": col(e_own), "mse_zero": col(e_zero), "mse_neg": col(e_neg),
        # generic use = own vs zero; sign use = own vs neg (both in PSNR, higher=better fidelity)
        "gen_use_psnr": col(_psnr(e_own) - _psnr(e_zero)),
        "sign_use_psnr": col(_psnr(e_own) - _psnr(e_neg)),
        "motion_ratio": col(out_motion / (gt_motion + 1e-8)),
        "resp_zero": col(resp_zero), "resp_neg": col(resp_neg),
        "gt_motion": col(gt_motion),
        "tv_own": col(tv_own), "tv_gt": col(tv_gt),
        "tv_ratio": col(tv_own / (tv_gt + 1e-8)),                  # >1 ⇒ artifact/drift vs real
    }
