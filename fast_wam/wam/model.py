"""
Multi-view Latent Action Model.

A latent action ``z`` is encoded from a frame pair of ONE camera view and is
required to reconstruct the next frame of ANOTHER (time-synchronised) view:

    z^v_t   = Enc(o^v_t, o^v_{t+1})
    L_self  = || Dec(o^v_t, z^v_t) - o^v_{t+1} ||^2
    L_cross = || Dec(o^w_t, z^v_t) - o^w_{t+1} ||^2        (w != v)

Anything in ``z`` that is specific to view ``v`` hurts L_cross, so view-specific
information is actively penalised and only the view-invariant cause of the
transition -- the action -- survives.

Note on where the view enters the network: the decoder is
``Dec(context_patches(o_t), z)``.  The view therefore enters ONLY through the
context patches, which is exactly why swapping the context to another view is a
sufficient implementation of cross-view reconstruction.

`num_views > 0` enables the X-VLA-style *view prompt* ablation: a learnable
per-view embedding is added to the decoder token stream, giving the decoder a
free channel to explain view-specific appearance.  That arm is trained WITHOUT
the cross-view term -- it tests "absorb the viewpoint difference" against this
model's "remove the viewpoint difference".

The transformer building blocks are imported read-only from the vendored
AdaWorld package (``code/adaworld/lam``); nothing there is modified.
"""

import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

# Read-only import of the unchanged AdaWorld building blocks.
# This file lives at <root>/fast_wam/wam/model.py; adaworld is at <root>/code/adaworld.
_ADAWORLD = Path(__file__).resolve().parents[2] / "code" / "adaworld" / "lam"
if str(_ADAWORLD) not in sys.path:
    sys.path.insert(0, str(_ADAWORLD))
from lam.modules.blocks import (  # noqa: E402
    patchify,
    unpatchify,
    SpatioTemporalTransformer,
    SpatioTransformer,
)


class MultiViewLAM(nn.Module):
    """Latent action VAE with optional cross-view decoding and view prompts."""

    def __init__(
        self,
        in_dim: int = 3,
        model_dim: int = 1024,
        latent_dim: int = 32,
        patch_size: int = 16,
        enc_blocks: int = 24,
        dec_blocks: int = 24,
        num_heads: int = 16,
        dropout: float = 0.0,
        num_views: int = 0,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        self.num_views = num_views
        patch_token_dim = in_dim * patch_size ** 2

        self.action_prompt = nn.Parameter(torch.empty(1, 1, 1, patch_token_dim))
        nn.init.uniform_(self.action_prompt, a=-1, b=1)

        self.encoder = SpatioTemporalTransformer(
            in_dim=patch_token_dim,
            model_dim=model_dim,
            out_dim=model_dim,
            num_blocks=enc_blocks,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.fc = nn.Linear(model_dim, latent_dim * 2)

        self.patch_up = nn.Linear(patch_token_dim, model_dim)
        self.action_up = nn.Linear(latent_dim, model_dim)
        self.decoder = SpatioTransformer(
            in_dim=model_dim,
            model_dim=model_dim,
            out_dim=patch_token_dim,
            num_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=dropout,
        )

        if num_views > 0:
            # View-prompt ablation only. Kept out of the default arms so the
            # latent has no side channel for viewpoint.
            self.view_embed = nn.Embedding(num_views, model_dim)
            nn.init.normal_(self.view_embed.weight, std=0.02)

    # ── encode ────────────────────────────────────────────────────────────────

    def encode(self, videos: Tensor) -> Dict[str, Tensor]:
        """videos: (B, T, H, W, C) in [0,1]. Returns latent moments and sample."""
        B, T = videos.shape[:2]
        patches = patchify(videos, self.patch_size)                 # (B,T,N,pd)
        action_pad = self.action_prompt.expand(B, T, -1, -1)
        padded = torch.cat([action_pad, patches], dim=2)            # (B,T,1+N,pd)

        h = self.encoder(padded)                                    # (B,T,1+N,E)
        h = h[:, 1:, 0]                                             # (B,T-1,E) action slot
        h = h.reshape(B * (T - 1), self.model_dim)

        z_mu, z_logvar = torch.chunk(self.fc(h), 2, dim=1)
        if self.training:
            z_rep = z_mu + torch.randn_like(z_logvar) * torch.exp(0.5 * z_logvar)
        else:
            z_rep = z_mu
        return {
            "z_rep": z_rep.reshape(B, T - 1, 1, self.latent_dim),
            "z_mu": z_mu,                                           # (B*(T-1), L)
            "z_logvar": z_logvar,
        }

    # ── decode ────────────────────────────────────────────────────────────────

    def decode(
        self,
        z_rep: Tensor,
        context_videos: Tensor,
        view_ids: Optional[Tensor] = None,
    ) -> Tensor:
        """Render o_{t+1} of ``context_videos``'s view, driven by ``z_rep``.

        Passing a *different* view's frames as ``context_videos`` is exactly the
        cross-view reconstruction term.
        """
        H, W = context_videos.shape[2:4]
        ctx = patchify(context_videos, self.patch_size)[:, :-1]     # (B,T-1,N,pd)
        tokens = self.patch_up(ctx) + self.action_up(z_rep)         # z broadcasts over N
        if self.num_views > 0 and view_ids is not None:
            tokens = tokens + self.view_embed(view_ids)[:, None, None, :]
        out = torch.sigmoid(self.decoder(tokens))
        return unpatchify(out, self.patch_size, H, W)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """batch: videos (B,T,H,W,C); optionally cross_videos / view_ids /
        cross_view_ids. Returns recon (+ recon_cross) and latent moments."""
        out = self.encode(batch["videos"])
        out["recon"] = self.decode(
            out["z_rep"], batch["videos"], batch.get("view_ids")
        )
        cross = batch.get("cross_videos")
        if cross is not None:
            out["recon_cross"] = self.decode(
                out["z_rep"], cross, batch.get("cross_view_ids")
            )
        return out


# ── losses ────────────────────────────────────────────────────────────────────

def kl_divergence(z_mu: Tensor, z_logvar: Tensor) -> Tensor:
    """Mean per-sample KL to N(0, I)."""
    return 0.5 * torch.sum(
        z_mu.pow(2) + z_logvar.exp() - 1.0 - z_logvar, dim=1
    ).mean()


def lam_loss(
    outputs: Dict[str, Tensor],
    gt_future: Tensor,
    gt_future_cross: Optional[Tensor],
    beta: float,
    lambda_cross: float,
) -> Dict[str, Tensor]:
    """L = L_self + lambda_cross * L_cross + beta * KL.

    gt_future / gt_future_cross: (B, T-1, H, W, C) target frames in [0,1].
    """
    mse_self = torch.mean((outputs["recon"] - gt_future) ** 2)
    kl = kl_divergence(outputs["z_mu"], outputs["z_logvar"])

    if "recon_cross" in outputs and gt_future_cross is not None:
        mse_cross = torch.mean((outputs["recon_cross"] - gt_future_cross) ** 2)
    else:
        mse_cross = torch.zeros((), device=mse_self.device, dtype=mse_self.dtype)

    total = mse_self + lambda_cross * mse_cross + beta * kl
    return {"loss": total, "mse_self": mse_self, "mse_cross": mse_cross, "kl": kl}
