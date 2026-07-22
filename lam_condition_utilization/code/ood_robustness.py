"""α=0 生成/OOD 鲁棒性诊断(零训练,decode-only)。

α=0 用确定性后验均值 z_μ 训 decoder ⇒ 可能对 off-μ-manifold 的 latent 脆弱
(顾问 §9 honesty-bound)。本模块在冻结 manifest 上比较 A0(α=0)vs A1(α=1):

  1. sampling robustness: 解码 z = μ + s·σ·ε,s∈{0,0.5,1,2}(σ=exp(0.5·logvar))。
     重建 MSE(对 GT o_{t+1})随 s 的曲线。A0 训练时从未见噪声 ⇒ 若 E(s) 随 s 陡升
     且比 A1 陡,说明它对采样噪声脆弱。也报输出对 μ-解码的偏移 dev(s)=RMS(D(z_s)−D(μ))。
  2. do(z_a) extrapolation: z_a → k·z_a,k∈{0,0.5,1,1.5,2}(仅缩放动作子空间,z_e 保留)。
     motion(k)=RMS(D(k·z_a)−o_t)(输出相对当前帧的位移;良好可控 decoder 应随 k 单调增);
     TV(k)=总变差(伪影代理;远超真实帧 TV ⇒ off-manifold 伪影)。
  参考:真实帧 TV(o_t、o_{t+1})作伪影基线。

全部 decode-only,复用 model.decode_with;不改任何权重。
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

import _apath  # noqa: F401  # shared A (lam_cdlam_optimization) substrate on sys.path
from benchmark_lam import _encode_mu_var  # noqa: F401  (kept for parity; encode done inline)


def _tv(x: torch.Tensor) -> torch.Tensor:
    """Per-sample total variation of [B,H,W,C] in [0,1]."""
    dh = (x[:, 1:, :, :] - x[:, :-1, :, :]).abs().mean(dim=[1, 2, 3])
    dw = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean(dim=[1, 2, 3])
    return dh + dw


def _rms(x: torch.Tensor) -> torch.Tensor:
    """Per-sample RMS over pixels of [B,H,W,C]."""
    return (x ** 2).flatten(1).mean(1).sqrt()


@torch.no_grad()
def ood_curves(model, loader, device, n_pool: int = 1000,
               scales=(0.0, 0.5, 1.0, 2.0), ks=(0.0, 0.5, 1.0, 1.5, 2.0),
               n_eps: int = 3, decode_bs: int = 16, seed: int = 42) -> dict:
    """Sampling-robustness + do(z_a)-extrapolation curves on the frozen manifest."""
    videos_all, seen = [], 0
    for batch in loader:
        videos_all.append(batch["videos"])
        seen += batch["videos"].shape[0]
        if seen >= n_pool:
            break
    if not videos_all:
        return {}
    videos = torch.cat(videos_all, dim=0)[:n_pool]
    ap, Dlat = model.action_part, model.latent_dim
    g = torch.Generator(device=device).manual_seed(seed)
    rec = defaultdict(list)

    for s0 in range(0, len(videos), decode_bs):
        vb = videos[s0:s0 + decode_bs].to(device)
        o_t, gt = vb[:, 0], vb[:, 1]
        H, W = vb.shape[2:4]
        enc = model.encode(vb)
        patches = enc["patches"]
        mu, logvar = enc["z_mu"], enc["z_var"]
        sigma = torch.exp(0.5 * logvar)

        def dec(z):
            return model.decode_with(patches, z.reshape(z.shape[0], 1, 1, Dlat), H, W)[:, 0]

        d_mu = dec(mu)                                    # μ-decode reference
        rec["tv_gt"].append(_tv(gt).cpu().numpy())
        rec["tv_ot"].append(_tv(o_t).cpu().numpy())
        rec["tv_mu"].append(_tv(d_mu).cpu().numpy())

        # 1) sampling robustness
        for s in scales:
            if s == 0.0:
                mse = ((d_mu - gt) ** 2).flatten(1).mean(1)
                dev = torch.zeros(len(vb), device=device)
            else:
                mses, devs = [], []
                for _ in range(n_eps):
                    eps = torch.randn(mu.shape, generator=g, device=device)
                    d = dec(mu + s * sigma * eps)
                    mses.append(((d - gt) ** 2).flatten(1).mean(1))
                    devs.append(_rms(d - d_mu))
                mse = torch.stack(mses).mean(0)
                dev = torch.stack(devs).mean(0)
            rec[f"mse_s{s}"].append(mse.cpu().numpy())
            rec[f"dev_s{s}"].append(dev.cpu().numpy())

        # 2) do(z_a) extrapolation (scale only the action subspace)
        for k in ks:
            z = mu.clone()
            z[:, :ap] = k * mu[:, :ap]
            d = dec(z)
            rec[f"motion_k{k}"].append(_rms(d - o_t).cpu().numpy())
            rec[f"tv_k{k}"].append(_tv(d).cpu().numpy())

    out = {k: float(np.mean(np.concatenate(v))) for k, v in rec.items()}
    out["n"] = int(len(videos))
    out["_scales"] = list(scales)
    out["_ks"] = list(ks)
    return out
