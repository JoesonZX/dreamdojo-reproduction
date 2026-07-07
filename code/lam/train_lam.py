"""
LAM training script using HuggingFace Accelerate.

Usage:
  # Single node, 3 GPUs (skip GPU 0 which is occupied):
  CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch \\
      --num_processes 3 \\
      --mixed_precision bf16 \\
      /home/xuan/embodied-ai/code/lam/train_lam.py \\
      --config /home/xuan/embodied-ai/code/lam/config/lam_agibot.yaml

  # Dry-run (1 GPU, no wandb):
  CUDA_VISIBLE_DEVICES=1 python /home/xuan/embodied-ai/code/lam/train_lam.py \\
      --config /home/xuan/embodied-ai/code/lam/config/lam_agibot.yaml \\
      --dry_run
"""

import argparse
import math
import os
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, random_split

# ── AdaWorld path ──────────────────────────────────────────────────────────────
ADAWORLD_PATH = Path(__file__).resolve().parent.parent / "adaworld" / "lam"
sys.path.insert(0, str(ADAWORLD_PATH))

from lam.modules import LatentActionModel  # AdaWorld's core module

# ── Local dataset ──────────────────────────────────────────────────────────────
LAM_DATA_PATH = Path(__file__).resolve().parent
sys.path.insert(0, str(LAM_DATA_PATH))
from data.agibot_dataset import AgibotVideoDataset
from data.egodex_dataset import EgoDexDataset

# ── Accelerate ─────────────────────────────────────────────────────────────────
from accelerate import Accelerator
from accelerate.utils import set_seed


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

class SigLIPProjectionHead(torch.nn.Module):
    """Projects z_mu (32-dim) into SigLIP text embedding space (768-dim).

    A two-layer MLP with LayerNorm + L2 normalisation on the output so that
    cosine similarity between projected latent and text embedding is well-defined.
    """
    def __init__(self, latent_dim: int = 32, text_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, text_dim),
        )
        # Learnable temperature and bias (SigLIP paper, eq. 1)
        self.log_t = torch.nn.Parameter(torch.zeros(1))   # log temperature
        self.bias  = torch.nn.Parameter(torch.zeros(1))   # global bias

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.net(z)
        return torch.nn.functional.normalize(out, dim=-1)  # L2 norm


def siglip_loss(proj_z: torch.Tensor, text_emb: torch.Tensor,
                log_t: torch.Tensor, bias: torch.Tensor,
                verb_emb: "torch.Tensor | None" = None) -> torch.Tensor:
    """
    SigLIP contrastive loss with soft semantic labels.

    logits: similarity between projected latent z and full caption embeddings.
    labels: pairwise cosine similarity of verb embeddings (if provided) or
            full caption embeddings (fallback). Using verb embeddings avoids
            dilution from object/scene words — two "pick up X" and "pick up Y"
            captions share the same verb and get labels close to 1.0, whereas
            full-caption similarity would be pulled down by the differing objects.

    proj_z:   [B, D]  L2-normalised projected latent actions
    text_emb: [B, D]  L2-normalised full caption embeddings (precomputed)
    log_t:    scalar  learnable log temperature
    bias:     scalar  learnable global bias
    verb_emb: [B, D]  L2-normalised verb-only embeddings (optional)
    """
    text_emb = torch.nn.functional.normalize(text_emb.float(), dim=-1)

    t = log_t.exp().clamp(max=100.0)
    logits = t * (proj_z @ text_emb.T) + bias

    with torch.no_grad():
        if verb_emb is not None:
            # Verb-based soft labels: focus purely on action semantics
            verb_emb = torch.nn.functional.normalize(verb_emb.float(), dim=-1)
            labels = verb_emb @ verb_emb.T
        else:
            # Fallback: full caption similarity
            labels = text_emb @ text_emb.T

    loss = -torch.nn.functional.logsigmoid(labels * logits).mean()
    return loss


def lam_loss(
    outputs: dict,
    gt_future: torch.Tensor,
    beta: float,
    fg_mask: "torch.Tensor | None" = None,
    fg_weight: float = 5.0,
):
    """
    VAE loss: (optionally foreground-weighted) MSE + β * KL divergence.

    outputs["recon"]: [B, T-1, H, W, C] predicted future frames
    gt_future:        [B, T-1, H, W, C] ground-truth future frames
    fg_mask:          [B, H, W] bool — True = foreground pixel (optional)
    fg_weight:        loss multiplier for foreground pixels (default 5)
    """
    err = (gt_future - outputs["recon"]) ** 2  # [B, T-1, H, W, C]
    if fg_mask is not None:
        # Broadcast mask: [B, H, W] → [B, 1, H, W, 1]
        w = 1.0 + (fg_weight - 1.0) * fg_mask.float().unsqueeze(1).unsqueeze(-1)
        mse = (w * err).mean()
    else:
        mse = err.mean()
    # KL( N(mu, exp(var)) || N(0,1) )
    kl = -0.5 * torch.sum(
        1 + outputs["z_var"] - outputs["z_mu"] ** 2 - outputs["z_var"].exp(),
        dim=1
    ).mean()
    return mse + beta * kl, mse, kl


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(accelerator, model, optimizer, scheduler, step, output_dir, keep_last_n):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f"step_{step:07d}"
    accelerator.save_state(str(ckpt_path))

    # Clean up old checkpoints
    ckpts = sorted(output_dir.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    for old in ckpts[:-keep_last_n]:
        import shutil
        shutil.rmtree(old, ignore_errors=True)

    accelerator.print(f"[save] checkpoint → {ckpt_path}")


def _warm_start_from(accelerator, model, ckpt_path: str):
    """Load model weights from a checkpoint to warm-start a fine-tune run.
    Only loads model weights; optimizer/scheduler start fresh."""
    from safetensors.torch import load_file
    sf_path = Path(ckpt_path) / "model.safetensors"
    if not sf_path.exists():
        accelerator.print(f"[warm-start] WARNING: {sf_path} not found, skipping")
        return
    state_dict = load_file(str(sf_path), device="cpu")
    unwrapped = accelerator.unwrap_model(model)
    missing, unexpected = unwrapped.load_state_dict(state_dict, strict=False)
    if missing:
        accelerator.print(f"[warm-start] missing keys: {len(missing)} (proj_head params expected)")
    if unexpected:
        accelerator.print(f"[warm-start] unexpected keys: {len(unexpected)}")
    accelerator.print(f"[warm-start] model weights loaded from {sf_path}")


def resume_from_checkpoint(accelerator, output_dir):
    output_dir = Path(output_dir)
    ckpts = sorted(output_dir.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    if not ckpts:
        return 0
    latest = ckpts[-1]
    accelerator.load_state(str(latest))
    step = int(latest.name.split("_")[1])
    accelerator.print(f"[resume] loaded checkpoint from step {step}")
    return step


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint directory to resume from (overrides auto-resume)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Run 5 steps then exit (smoke test)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    mcfg  = cfg["model"]
    dcfg  = cfg["data"]
    tcfg  = cfg["training"]
    lcfg  = cfg["logging"]
    fgcfg = cfg.get("foreground", {})

    # ── Accelerator ───────────────────────────────────────────────────────────
    accelerator = Accelerator(
        mixed_precision=tcfg.get("mixed_precision", "bf16"),
        gradient_accumulation_steps=tcfg.get("gradient_accumulation_steps", 1),
        log_with="wandb" if lcfg.get("use_wandb") else None,
    )
    set_seed(42)

    if accelerator.is_main_process:
        if lcfg.get("use_wandb"):
            accelerator.init_trackers(
                project_name=lcfg.get("project", "dreamdojo-lam"),
                config=cfg,
                init_kwargs={"wandb": {"name": lcfg.get("run_name", "lam-run")}},
            )
        Path(tcfg["output_dir"]).mkdir(parents=True, exist_ok=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = LatentActionModel(
        in_dim=mcfg.get("image_channels", 3),
        model_dim=mcfg.get("lam_model_dim", 512),
        latent_dim=mcfg.get("lam_latent_dim", 32),
        patch_size=mcfg.get("lam_patch_size", 16),
        enc_blocks=mcfg.get("lam_enc_blocks", 8),
        dec_blocks=mcfg.get("lam_dec_blocks", 8),
        num_heads=mcfg.get("lam_num_heads", 8),
        dropout=mcfg.get("lam_dropout", 0.0),
    )
    beta       = mcfg.get("beta", 1e-6)
    use_fg     = fgcfg.get("use_fg_loss", False)
    fg_weight  = fgcfg.get("fg_weight", 5.0)
    fg_mask_type = fgcfg.get("fg_mask_type", "raft")   # "raft" or "sam3"
    siglip_cfg = cfg.get("siglip", {})
    use_siglip = siglip_cfg.get("use_siglip_loss", False)
    siglip_lambda = siglip_cfg.get("lambda_siglip", 0.1)
    use_verb_emb = siglip_cfg.get("use_verb_emb", False)
    accelerator.print(f"Foreground loss: {'ON (weight=%.1f)' % fg_weight if use_fg else 'OFF'}")
    accelerator.print(f"SigLIP loss:     {'ON (lambda=%.3f, verb_label=%s)' % (siglip_lambda, use_verb_emb) if use_siglip else 'OFF'}")

    proj_head = None
    if use_siglip:
        proj_head = SigLIPProjectionHead(
            latent_dim=mcfg.get("lam_latent_dim", 32),
            text_dim=siglip_cfg.get("text_dim", 768),
            hidden_dim=siglip_cfg.get("proj_hidden_dim", 256),
        )
    accelerator.print(
        f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M"
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset_type = dcfg.get("dataset_type", "agibot")
    accelerator.print(f"Dataset type: {dataset_type}")

    if dataset_type == "egodex":
        train_ds = EgoDexDataset(
            data_root=dcfg["data_root"],
            img_h=dcfg.get("img_h", 240),
            img_w=dcfg.get("img_w", 320),
            downsample_factors=tuple(dcfg.get("downsample_factors", [1, 2])),
            split="train",
            val_ratio=dcfg.get("val_ratio", 0.1),
            seed=dcfg.get("seed", 42),
            use_fg_mask=use_fg,
            fg_mask_type=fg_mask_type,
            use_caption=use_siglip,
            use_verb_emb=use_verb_emb,
        )
        val_ds = EgoDexDataset(
            data_root=dcfg["data_root"],
            img_h=dcfg.get("img_h", 240),
            img_w=dcfg.get("img_w", 320),
            downsample_factors=(1,),
            split="val",
            val_ratio=dcfg.get("val_ratio", 0.1),
            seed=dcfg.get("seed", 42),
            use_fg_mask=False,
            use_caption=False,
        )
    else:
        train_ds = AgibotVideoDataset(
            data_root=dcfg["data_root"],
            camera=dcfg.get("camera", "head_color"),
            img_h=dcfg.get("img_h", 240),
            img_w=dcfg.get("img_w", 320),
            downsample_factors=tuple(dcfg.get("downsample_factors", [1, 2, 3, 4])),
            split="train",
            val_ratio=dcfg.get("val_ratio", 0.05),
            seed=dcfg.get("seed", 42),
            use_fg_mask=use_fg,
        )
        val_ds = AgibotVideoDataset(
            data_root=dcfg["data_root"],
            camera=dcfg.get("camera", "head_color"),
            img_h=dcfg.get("img_h", 240),
            img_w=dcfg.get("img_w", 320),
            downsample_factors=(1,),
            split="val",
            val_ratio=dcfg.get("val_ratio", 0.05),
            seed=dcfg.get("seed", 42),
            use_fg_mask=False,
        )
    accelerator.print(f"Train pairs: {len(train_ds):,}  Val pairs: {len(val_ds):,}")

    def collate_fn(samples):
        batch = {}
        for key in samples[0]:
            if isinstance(samples[0][key], torch.Tensor):
                batch[key] = torch.stack([s[key] for s in samples])
            else:
                batch[key] = [s[key] for s in samples]  # str fields (task_id, ep_id)
        return batch

    train_loader = DataLoader(
        train_ds,
        batch_size=tcfg.get("per_gpu_batch_size", 32),
        shuffle=True,
        num_workers=tcfg.get("num_workers", 8),
        pin_memory=tcfg.get("pin_memory", True),
        drop_last=True,
        persistent_workers=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=tcfg.get("per_gpu_batch_size", 32),
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg.get("lr", 2.5e-5),
        weight_decay=tcfg.get("weight_decay", 0.01),
    )
    total_steps = tcfg.get("total_steps", 100000)
    warmup_steps = tcfg.get("warmup_steps", 1000)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Accelerate prepare ────────────────────────────────────────────────────
    if proj_head is not None:
        model, proj_head, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
            model, proj_head, optimizer, train_loader, val_loader, scheduler
        )
    else:
        model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, val_loader, scheduler
        )

    # ── Resume ────────────────────────────────────────────────────────────────
    # Priority: 1) latest checkpoint in output_dir (auto-resume same run)
    #           2) --resume path (cross-run warm-start, step resets to 0)
    global_step = resume_from_checkpoint(accelerator, tcfg["output_dir"])
    if global_step == 0 and args.resume:
        # No checkpoint in output_dir yet — warm-start weights from --resume
        # but do NOT inherit step number (fine-tune starts fresh from step 0)
        _warm_start_from(accelerator, model, args.resume)
    if global_step > 0:
        # Fast-forward the loader
        skip_batches = global_step % len(train_loader)
        train_loader_iter = iter(train_loader)
        for _ in range(skip_batches):
            try:
                next(train_loader_iter)
            except StopIteration:
                train_loader_iter = iter(train_loader)

    # ── Training loop ─────────────────────────────────────────────────────────
    log_every   = tcfg.get("log_every",   100)
    save_every  = tcfg.get("save_every",  5000)
    eval_every  = tcfg.get("eval_every",  2000)
    keep_last_n = tcfg.get("keep_last_n", 3)
    log_path    = mcfg.get("log_path", tcfg["output_dir"] + "/log_imgs")

    model.train()
    train_loader_iter = iter(train_loader)
    running_loss = running_mse = running_kl = running_siglip = 0.0

    accelerator.print(f"Starting training from step {global_step} → {total_steps}")

    while global_step < total_steps:
        try:
            batch = next(train_loader_iter)
        except StopIteration:
            train_loader_iter = iter(train_loader)
            batch = next(train_loader_iter)

        # batch["videos"]: [B, 2, H, W, C]
        with accelerator.accumulate(model):
            outputs = model(batch)
            gt_future = batch["videos"][:, 1:]  # [B, 1, H, W, C]
            fg_mask = batch.get("fg_mask", None)
            if fg_mask is not None:
                fg_mask = fg_mask.to(batch["videos"].device)
            loss, mse, kl = lam_loss(outputs, gt_future, beta, fg_mask, fg_weight)

            # SigLIP contrastive loss
            loss_siglip = torch.tensor(0.0, device=loss.device)
            if use_siglip and proj_head is not None:
                cap_emb = batch.get("caption_emb", None)
                if cap_emb is not None:
                    cap_emb = cap_emb.to(loss.device)
                    verb_emb = batch.get("verb_emb", None)
                    if verb_emb is not None:
                        verb_emb = verb_emb.to(loss.device)
                    proj_z = proj_head(outputs["z_mu"])
                    loss_siglip = siglip_loss(
                        proj_z, cap_emb,
                        proj_head.module.log_t if hasattr(proj_head, "module") else proj_head.log_t,
                        proj_head.module.bias  if hasattr(proj_head, "module") else proj_head.bias,
                        verb_emb=verb_emb,
                    )
                    loss = loss + siglip_lambda * loss_siglip

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            global_step += 1
            running_loss    += loss.item()
            running_mse     += mse.item()
            running_kl      += kl.item()
            running_siglip  += loss_siglip.item()

            # ── Logging ───────────────────────────────────────────────────────
            if global_step % log_every == 0 and accelerator.is_main_process:
                avg = 1.0 / log_every
                metrics = {
                    "train/loss":    running_loss   * avg,
                    "train/mse":     running_mse    * avg,
                    "train/kl":      running_kl     * avg,
                    "train/siglip":  running_siglip * avg,
                    "train/lr":      scheduler.get_last_lr()[0],
                    "global_step":   global_step,
                }
                siglip_str = f"| siglip {metrics['train/siglip']:.4f} " if use_siglip else ""
                accelerator.print(
                    f"step {global_step:6d} | loss {metrics['train/loss']:.4f} "
                    f"| mse {metrics['train/mse']:.4f} | kl {metrics['train/kl']:.6f} "
                    f"{siglip_str}| lr {metrics['train/lr']:.2e}"
                )
                if lcfg.get("use_wandb"):
                    accelerator.log(metrics, step=global_step)
                running_loss = running_mse = running_kl = running_siglip = 0.0

            # ── Validation ────────────────────────────────────────────────────
            if global_step % eval_every == 0:
                model.eval()
                val_loss_sum = val_mse_sum = 0.0
                val_n = 0
                with torch.no_grad():
                    for vbatch in val_loader:
                        vout = model(vbatch)
                        gt_v = vbatch["videos"][:, 1:]
                        vloss, vmse, _ = lam_loss(vout, gt_v, beta)
                        val_loss_sum += vloss.item() * vbatch["videos"].shape[0]
                        val_mse_sum  += vmse.item()  * vbatch["videos"].shape[0]
                        val_n        += vbatch["videos"].shape[0]
                        if val_n >= 512:  # Limit val for speed
                            break

                if accelerator.is_main_process and val_n > 0:
                    val_metrics = {
                        "val/loss": val_loss_sum / val_n,
                        "val/mse":  val_mse_sum  / val_n,
                    }
                    accelerator.print(
                        f"[val] step {global_step:6d} | loss {val_metrics['val/loss']:.4f} "
                        f"| mse {val_metrics['val/mse']:.4f}"
                    )
                    if lcfg.get("use_wandb"):
                        accelerator.log(val_metrics, step=global_step)

                    # Save a comparison image
                    _save_recon_image(vbatch, vout, global_step, log_path)

                model.train()

            # ── Save checkpoint ───────────────────────────────────────────────
            if global_step % save_every == 0 or global_step == total_steps:
                save_checkpoint(
                    accelerator, model, optimizer, scheduler,
                    global_step, tcfg["output_dir"], keep_last_n
                )

        if args.dry_run and global_step >= 5:
            accelerator.print("Dry run complete — exiting.")
            break

    if lcfg.get("use_wandb"):
        accelerator.end_training()
    accelerator.print("Training complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Image logging helper
# ─────────────────────────────────────────────────────────────────────────────

def _save_recon_image(batch, outputs, step, log_path):
    try:
        import numpy as np
        from PIL import Image

        Path(log_path).mkdir(parents=True, exist_ok=True)
        # Take first sample in batch
        gt_pair  = batch["videos"][0].cpu().float().clamp(0, 1)   # [2, H, W, C]
        recon    = outputs["recon"][0].cpu().float().clamp(0, 1)   # [1, H, W, C]
        # Top row: [input, gt_next]  Bottom row: [input, recon]
        blank    = gt_pair[0:1]
        top      = torch.cat([gt_pair[0:1], gt_pair[1:2]], dim=2)   # side by side
        bot      = torch.cat([blank,         recon[0:1]],  dim=2)
        compare  = torch.cat([top, bot],     dim=1)                  # stack vertically
        img_arr  = (compare[0] * 255).numpy().astype(np.uint8)
        Image.fromarray(img_arr).save(
            Path(log_path) / f"step_{step:07d}.png"
        )
    except Exception:
        pass  # Don't crash training over a logging failure


if __name__ == "__main__":
    main()
