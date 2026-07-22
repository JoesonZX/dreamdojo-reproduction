"""Phase 0.2 — four-cell cross-decoding (顾问 §0.2).

对四个组合 D_{A0/A1} × E_{A0/A1} 逐格评测(latent 全用 z_mu):每格跑
own/dist/antiparallel/random 的 C_pooled、R_out(前景)与 own-latent fidelity。

判读矩阵(顾问 §0.2):
  D_A0 对两个 encoder 都强 & D_A1 对两个都弱      ⇒ 干净 decoder-side utilization 定位;
  只有 matched 对角强                              ⇒ encoder-decoder co-adaptation/坐标协商;
  A0-encoder 喂两个 decoder 都更好                 ⇒ α 改了 encoder 的条件表示;
  paired μ 近乎相同 且 只有 D_A0 强                ⇒ 外部只看 μ 的 ACWM 原则上拿不到 A0 收益。

一个 cell = (解码器 dec_model, 编码器给出的 latent mu_enc)。C 复用与 h2_poscontrol.lam_c
完全一致的 swap/归一化逻辑,所以对角格精确复现 poscontrol_alpha 的 C(correctness gate)。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from h2_poscontrol import _robust_C

# Read-only import of the (parameter-free) patchify — identical for both decoders
# (same patch_size), so we patchify directly instead of running a full encoder pass.
_ADAWORLD = Path(__file__).resolve().parents[2] / "code" / "adaworld" / "lam"
if str(_ADAWORLD) not in sys.path:
    sys.path.insert(0, str(_ADAWORLD))
from lam.modules.blocks import patchify  # noqa: E402

SWAP_SCHEMES = ("dist", "antiparallel", "random")


@torch.no_grad()
def cell_metrics(dec_model, mu_enc: torch.Tensor, videos: torch.Tensor,
                 schemes: Dict[str, List], device, bs: int = 16,
                 fg_q: float = 0.90) -> dict:
    """Decode with ``dec_model`` using latents ``mu_enc`` (from an arbitrary encoder).

    All schemes share the same anchor order (``_build_schemes`` builds them in lockstep),
    so own-latent recon is decoded once and reused. For every donor scheme:
      C_pooled = Σ(E_opp−E_same)/ΣP   (persistence-normalized, matches lam_c)
      R_out_fg = mean ‖D(o,z_opp) − D(o,z_same)‖ over motion-foreground pixels.
    own-latent: E_own (per-pixel MSE) and PSNR = −10·log10(E_own).
    """
    ap, Dlat = dec_model.action_part, dec_model.latent_dim
    anchors = [t[0] for t in schemes["dist"]]          # identical across schemes
    N = len(anchors)

    sc_diff = {s: [] for s in SWAP_SCHEMES}
    sc_P = {s: [] for s in SWAP_SCHEMES}
    sc_rout = {s: [] for s in SWAP_SCHEMES}
    e_own_all: List[np.ndarray] = []

    for k in range(0, N, bs):
        idx = anchors[k:k + bs]
        vb = videos[idx].to(device)
        gt, prev = vb[:, 1], vb[:, 0]
        H, W = vb.shape[2:4]
        patches = patchify(vb, dec_model.patch_size)
        z_own = mu_enc[idx].to(device)

        def dec(z):
            return dec_model.decode_with(patches, z.reshape(z.shape[0], 1, 1, Dlat), H, W)[:, 0]

        p_own = dec(z_own)
        e_own = ((p_own - gt) ** 2).mean(-1).flatten(1).mean(1)
        e_own_all.append(e_own.cpu().numpy())

        P = ((gt - prev) ** 2).mean(-1).flatten(1).mean(1)
        motion = (gt - prev).abs().mean(-1)                       # [B,H,W]
        thr = torch.quantile(motion.flatten(1), fg_q, dim=1)
        fg = (motion >= thr[:, None, None]) & (motion > 1e-3)
        fg_n = fg.flatten(1).sum(1).clamp_min(1)

        for s in SWAP_SCHEMES:
            sch = schemes[s][k:k + bs]                            # aligned with idx

            def swap(col):
                z = z_own.clone()
                z[:, :ap] = mu_enc[[t[col] for t in sch], :ap].to(device)
                return z

            p_same, p_opp = dec(swap(1)), dec(swap(2))
            e_same = ((p_same - gt) ** 2).mean(-1).flatten(1).mean(1)
            e_opp = ((p_opp - gt) ** 2).mean(-1).flatten(1).mean(1)
            sc_diff[s].append((e_opp - e_same).cpu().numpy())
            sc_P[s].append(P.cpu().numpy())
            resp = ((p_opp - p_same) ** 2).mean(-1)               # [B,H,W]
            rf = (resp * fg).flatten(1).sum(1) / fg_n
            sc_rout[s].append(torch.sqrt(rf).cpu().numpy())

    e_own = float(np.concatenate(e_own_all).mean())
    out = {"E_own": e_own, "psnr": float(-10.0 * np.log10(e_own + 1e-12)),
           "n_anchors": int(N)}
    for s in SWAP_SCHEMES:
        C = _robust_C(np.concatenate(sc_diff[s]), np.concatenate(sc_P[s]))
        out[s] = {"C_pooled": C["C_pooled"], "C_median": C["C_median"],
                  "R_out_fg": float(np.mean(np.concatenate(sc_rout[s])))}
    return out
