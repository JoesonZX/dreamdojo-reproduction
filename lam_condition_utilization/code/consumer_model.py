"""Phase 2 route-C:独立、从零初始化的 action-conditioned next-frame predictor(prereg §4)。

**禁止复用 D_A0/D_A1 权重** —— 这是独立 consumer,只通过 FiLM 消费一个条件向量 `c`
(= 冻结 encoder 的 z_μ action 子空间,或正对照的 GT-18D action)。one-step teacher-forced 训练
(o_t, c_t → ô_{t+1}),推理时自回归 K 步。预测**残差** ô = clip(o_t + Δ),使"静止复制"=零条件响应
的平凡 baseline,从而 target-vs-shuffle/antiparallel 的差异只能来自 consumer 真的用了 `c`。

小型 conv U-Net(从零),FiLM 在 bottleneck + 每个 up level 注入 `c`。与 LAM decoder 架构无关。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLM(nn.Module):
    """Per-channel affine (γ,β) predicted from the condition vector."""

    def __init__(self, cond_dim: int, ch: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cond_dim, ch * 2))

    def forward(self, x, c):
        gb = self.net(c)                                   # [B, 2*ch]
        g, b = gb.chunk(2, dim=1)
        return x * (1 + g[:, :, None, None]) + b[:, :, None, None]


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.n1 = nn.GroupNorm(8, cout)
        self.n2 = nn.GroupNorm(8, cout)

    def forward(self, x):
        x = F.silu(self.n1(self.c1(x)))
        return F.silu(self.n2(self.c2(x)))


class ConditionalUNet(nn.Module):
    """From-scratch FiLM-conditioned residual next-frame predictor.

    forward(o_t[B,3,H,W], c[B,cond_dim]) -> ô_{t+1}[B,3,H,W] in [0,1].
    """

    def __init__(self, cond_dim: int, base: int = 64, in_ch: int = 3):
        super().__init__()
        self.cond_dim = cond_dim
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        self.in_conv = ConvBlock(in_ch, c1)
        self.d1 = ConvBlock(c1, c2)
        self.d2 = ConvBlock(c2, c3)
        self.d3 = ConvBlock(c3, c4)
        self.pool = nn.AvgPool2d(2)
        self.film_b = FiLM(cond_dim, c4)
        self.up3 = nn.Conv2d(c4, c3, 3, padding=1)
        self.u3 = ConvBlock(c3 * 2, c3); self.film3 = FiLM(cond_dim, c3)
        self.up2 = nn.Conv2d(c3, c2, 3, padding=1)
        self.u2 = ConvBlock(c2 * 2, c2); self.film2 = FiLM(cond_dim, c2)
        self.up1 = nn.Conv2d(c2, c1, 3, padding=1)
        self.u1 = ConvBlock(c1 * 2, c1); self.film1 = FiLM(cond_dim, c1)
        self.out = nn.Conv2d(c1, in_ch, 3, padding=1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)   # start as identity (Δ=0)

    def _up(self, x, skip, conv, block, film, c):
        x = F.interpolate(x, size=skip.shape[2:], mode="nearest")
        x = conv(x)
        x = block(torch.cat([x, skip], dim=1))
        return film(x, c)

    def forward(self, o_t, c):
        s1 = self.in_conv(o_t)                              # [B,c1,H,W]
        s2 = self.d1(self.pool(s1))                         # H/2
        s3 = self.d2(self.pool(s2))                         # H/4
        b = self.d3(self.pool(s3))                          # H/8
        b = self.film_b(b, c)
        x = self._up(b, s3, self.up3, self.u3, self.film3, c)
        x = self._up(x, s2, self.up2, self.u2, self.film2, c)
        x = self._up(x, s1, self.up1, self.u1, self.film1, c)
        delta = self.out(x)
        return (o_t + delta).clamp(0.0, 1.0)               # residual → identity at init

    @torch.no_grad()
    def rollout(self, o0, cseq):
        """Autoregressive K-step rollout. o0[B,3,H,W], cseq[B,K,cond_dim] -> [B,K,3,H,W]."""
        B, K = cseq.shape[0], cseq.shape[1]
        outs = []
        cur = o0
        for i in range(K):
            cur = self.forward(cur, cseq[:, i])
            outs.append(cur)
        return torch.stack(outs, dim=1)
