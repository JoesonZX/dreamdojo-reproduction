"""
Latent Action Model for the disentanglement experiments.

Vendored from code/adaworld/lam/lam/modules/lam.py and EXTENDED with:
  - an optional environment subspace (latent split into action[:action_part]
    and env[action_part:], where action_part = latent_dim - env_dim), and
  - an optional action-prediction head that maps the action subspace to the
    real-action space (18-dim two-hand EE delta) for supervised purification.

The unchanged building blocks (patchify/unpatchify/transformers) are imported
read-only from the original AdaWorld package; nothing there is modified.
"""

import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Read-only import of unchanged building blocks from the original repo.
_ADAWORLD = Path(__file__).resolve().parents[2] / "code" / "adaworld" / "lam"
if str(_ADAWORLD) not in sys.path:
    sys.path.insert(0, str(_ADAWORLD))
from lam.modules.blocks import (  # noqa: E402
    patchify, unpatchify, SpatioTemporalTransformer, SpatioTransformer,
)


class LatentActionModel(nn.Module):
    """Latent action VAE with optional env subspace + action-prediction head."""

    def __init__(
        self,
        in_dim: int,
        model_dim: int,
        latent_dim: int,
        patch_size: int,
        enc_blocks: int,
        dec_blocks: int,
        num_heads: int,
        dropout: float = 0.0,
        env_dim: int = 0,            # NEW: dims reserved for environment info
        action_dim: int = 0,         # NEW: real-action dim (0 => no head)
        action_head: bool = False,   # NEW: enable action-prediction head
        contrastive: bool = False,   # NEW: add projection heads for contrastive losses
        proj_dim: int = 64,          # NEW: contrastive projection output dim
    ) -> None:
        super(LatentActionModel, self).__init__()
        self.model_dim = model_dim
        self.latent_dim = latent_dim            # TOTAL latent (action + env)
        self.patch_size = patch_size
        self.env_dim = env_dim
        self.action_part = latent_dim - env_dim  # action subspace = dims [0:action_part]
        if self.action_part <= 0:
            raise ValueError(f"action_part={self.action_part} must be > 0 "
                             f"(latent_dim={latent_dim}, env_dim={env_dim})")
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

        # Action-prediction head: action subspace -> real action (18-dim).
        self.action_head = action_head and action_dim > 0
        self.action_dim = action_dim
        if self.action_head:
            self.action_predictor = nn.Sequential(
                nn.Linear(self.action_part, 128),
                nn.GELU(),
                nn.Linear(128, action_dim),
            )

        # Contrastive projection heads (ConLA-style). Used ONLY by the contrastive
        # losses during training; never used at eval (probe/R² read raw z_mu). They
        # are new params, so a pretrained-LAM fine-tune leaves them random-init'd.
        self.contrastive = contrastive
        if contrastive:
            self.action_proj = nn.Sequential(
                nn.Linear(self.action_part, 128), nn.GELU(), nn.Linear(128, proj_dim),
            )
            if env_dim > 0:
                self.env_proj = nn.Sequential(
                    nn.Linear(env_dim, 128), nn.GELU(), nn.Linear(128, proj_dim),
                )

        self.mu_record = None

    def encode(self, videos: Tensor) -> Dict:
        B, T = videos.shape[:2]
        patches = patchify(videos, self.patch_size)
        action_pad = self.action_prompt.expand(B, T, -1, -1)
        padded_patches = torch.cat([action_pad, patches], dim=2)

        z = self.encoder(padded_patches)        # (B, T, 1+N, E)
        z = z[:, 1:, 0]                          # (B, T-1, E)

        z = z.reshape(B * (T - 1), self.model_dim)
        moments = self.fc(z)
        z_mu, z_var = torch.chunk(moments, 2, dim=1)
        if not self.training:
            z_rep = z_mu
        else:
            z_rep = z_mu + torch.randn_like(z_var) * torch.exp(0.5 * z_var)
        z_rep = z_rep.reshape(B, T - 1, 1, self.latent_dim)

        if not self.training:
            if self.mu_record is None:
                self.mu_record = z_mu
            else:
                self.mu_record = torch.cat([self.mu_record, z_mu], dim=0)

        out = {
            "patches": patches,
            "z_rep": z_rep,
            "z_mu": z_mu,        # [B*(T-1), latent_dim]
            "z_var": z_var,
        }
        if self.action_head:
            # Supervise ONLY the action subspace (dims [0:action_part]).
            out["pred_action"] = self.action_predictor(z_mu[:, :self.action_part])
        if self.contrastive:
            # L2-normalized projections of each subspace, for the contrastive losses.
            za = self.action_proj(z_mu[:, :self.action_part])
            out["z_a_proj"] = F.normalize(za, dim=1)
            if self.env_dim > 0:
                ze = self.env_proj(z_mu[:, self.action_part:])
                out["z_e_proj"] = F.normalize(ze, dim=1)
        return out

    def forward(self, batch: Dict) -> Dict:
        H, W = batch["videos"].shape[2:4]
        outputs = self.encode(batch["videos"])
        video_patches = self.patch_up(outputs["patches"][:, :-1])
        action_patches = self.action_up(outputs["z_rep"])
        video_action_patches = video_patches + action_patches

        del outputs["patches"]

        video_recon = self.decoder(video_action_patches)
        video_recon = F.sigmoid(video_recon)
        outputs.update({"recon": unpatchify(video_recon, self.patch_size, H, W)})
        return outputs
