#!/usr/bin/env python
"""
Train the multi-view LAM on preprocessed Ego-Exo4D.

    accelerate launch --num_processes 2 --mixed_precision bf16 \
        fast_wam/wam/train.py --config fast_wam/wam/config/mv_cross.yaml

All arms (sv / mv-data / mv-cross / view-prompt / mv-cross+hand) are the same
script driven by different YAML configs; they differ only in
``data.mode`` / ``loss.lambda_cross`` / ``model.num_views`` / ``loss.lambda_hand``.
"""

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from accelerate import Accelerator                      # noqa: E402
from accelerate.utils import set_seed                   # noqa: E402

from dataset import EgoExoMultiViewDataset, collate     # noqa: E402
from model import MultiViewLAM, lam_loss                # noqa: E402


# ── lr schedule ───────────────────────────────────────────────────────────────

def make_lr_lambda(warmup_steps: int, total_steps: int, num_processes: int = 1):
    """Cosine schedule with linear warmup, expressed in *global* steps.

    ``accelerator.prepare(scheduler)`` advances the wrapped scheduler once per
    process per optimizer step, so divide the raw counter by num_processes.
    """
    def lr_lambda(raw_step: int) -> float:
        step = raw_step / max(1, num_processes)
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
    return lr_lambda


# ── checkpointing ─────────────────────────────────────────────────────────────

def save_checkpoint(accelerator, step, output_dir, keep_last_n):
    out = Path(output_dir)
    ckpt = out / f"step_{step:07d}"
    accelerator.save_state(str(ckpt))
    if accelerator.is_main_process:
        existing = sorted(out.glob("step_*"))
        for old in existing[:-keep_last_n]:
            shutil.rmtree(old, ignore_errors=True)
        accelerator.print(f"[save] checkpoint -> {ckpt}")


def resume_if_any(accelerator, output_dir):
    ckpts = sorted(Path(output_dir).glob("step_*"))
    if not ckpts:
        return 0
    latest = ckpts[-1]
    accelerator.load_state(str(latest))
    step = int(latest.name.split("_")[1])
    accelerator.print(f"[resume] from step {step}")
    return step


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[],
                    help="dotted overrides, e.g. training.seed=1")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for ov in args.override:
        key, val = ov.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = yaml.safe_load(val)

    mcfg, dcfg, lcfg, tcfg = cfg["model"], cfg["data"], cfg["loss"], cfg["training"]

    accelerator = Accelerator(
        gradient_accumulation_steps=tcfg.get("gradient_accumulation_steps", 1),
        mixed_precision=tcfg.get("mixed_precision", "bf16"),
    )
    accelerator.even_batches = False
    seed = int(tcfg.get("seed", 42))
    set_seed(seed)

    out_dir = Path(tcfg["output_dir"])
    if accelerator.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "config.yaml", "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    # ── data ──────────────────────────────────────────────────────────────────
    load_actions = float(lcfg.get("lambda_hand", 0.0)) > 0.0
    common = dict(
        processed_root=dcfg["processed_root"],
        mode=dcfg["mode"],
        view_pairs=dcfg.get("view_pairs", "exo_exo"),
        val_ratio=dcfg.get("val_ratio", 0.05),
        seed=int(dcfg.get("split_seed", 42)),      # split is FIXED across arms/seeds
        img_h=dcfg.get("img_h", 240),
        img_w=dcfg.get("img_w", 320),
        downsample_factors=tuple(dcfg.get("downsample_factors", [1, 2])),
        pairs_per_take=dcfg.get("pairs_per_take", 64),
        load_actions=load_actions,
    )
    train_ds = EgoExoMultiViewDataset(split="train", **common)
    val_ds = EgoExoMultiViewDataset(split="val", **common)
    accelerator.print(f"takes: train={len(train_ds.entries)} val={len(val_ds.entries)}  "
                      f"virtual pairs: train={len(train_ds):,} val={len(val_ds):,}")

    train_loader = DataLoader(
        train_ds, batch_size=tcfg["per_gpu_batch_size"], shuffle=True,
        num_workers=tcfg.get("num_workers", 8), pin_memory=True,
        drop_last=True, collate_fn=collate, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=tcfg["per_gpu_batch_size"], shuffle=False,
        num_workers=max(2, tcfg.get("num_workers", 8) // 2), pin_memory=True,
        drop_last=True, collate_fn=collate,
    )

    # ── model ─────────────────────────────────────────────────────────────────
    model = MultiViewLAM(
        in_dim=mcfg.get("image_channels", 3),
        model_dim=mcfg["model_dim"],
        latent_dim=mcfg["latent_dim"],
        patch_size=mcfg["patch_size"],
        enc_blocks=mcfg["enc_blocks"],
        dec_blocks=mcfg["dec_blocks"],
        num_heads=mcfg["num_heads"],
        dropout=mcfg.get("dropout", 0.0),
        num_views=mcfg.get("num_views", 0),
    )
    n_params = sum(p.numel() for p in model.parameters())
    accelerator.print(f"Model params: {n_params/1e6:.1f}M  (num_views={mcfg.get('num_views', 0)})")

    hand_head = None
    if load_actions:
        hand_head = nn.Sequential(
            nn.Linear(mcfg["latent_dim"], 128), nn.GELU(), nn.Linear(128, 18)
        )

    params = list(model.parameters()) + (list(hand_head.parameters()) if hand_head else [])
    optimizer = torch.optim.AdamW(
        params, lr=float(tcfg["lr"]), weight_decay=float(tcfg.get("weight_decay", 0.01))
    )
    total_steps = int(tcfg["total_steps"])
    warmup_steps = int(tcfg.get("warmup_steps", 200))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, make_lr_lambda(warmup_steps, total_steps, accelerator.num_processes)
    )

    prepared = accelerator.prepare(model, optimizer, train_loader, val_loader, scheduler)
    model, optimizer, train_loader, val_loader, scheduler = prepared
    if hand_head is not None:
        hand_head = accelerator.prepare(hand_head)

    global_step = resume_if_any(accelerator, out_dir)

    beta = float(lcfg.get("beta", 1e-6))
    lambda_cross = float(lcfg.get("lambda_cross", 0.0))
    lambda_hand = float(lcfg.get("lambda_hand", 0.0))
    log_every = tcfg.get("log_every", 50)
    eval_every = tcfg.get("eval_every", 1000)
    save_every = tcfg.get("save_every", 1000)
    keep_last_n = tcfg.get("keep_last_n", 1)

    accelerator.print(
        f"Training -> {total_steps} steps | beta={beta} lambda_cross={lambda_cross} "
        f"lambda_hand={lambda_hand} | mode={dcfg['mode']} view_pairs={common['view_pairs']}"
    )

    # ── loop ──────────────────────────────────────────────────────────────────
    model.train()
    done = False
    while not done:
        for batch in train_loader:
            with accelerator.accumulate(model):
                gt_future = batch["videos"][:, 1:]
                gt_cross = batch["cross_videos"][:, 1:] if "cross_videos" in batch else None

                outputs = model(batch)
                losses = lam_loss(outputs, gt_future, gt_cross, beta, lambda_cross)
                loss = losses["loss"]

                hand_mse = torch.zeros((), device=loss.device)
                if hand_head is not None and "action" in batch:
                    pred = hand_head(outputs["z_mu"])
                    hand_mse = torch.mean((pred - batch["action"]) ** 2)
                    loss = loss + lambda_hand * hand_mse

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if not accelerator.sync_gradients:
                continue
            global_step += 1

            if global_step % log_every == 0:
                accelerator.print(
                    f"step {global_step:7d} | loss {loss.item():.4f} "
                    f"| self {losses['mse_self'].item():.5f} "
                    f"| cross {losses['mse_cross'].item():.5f} "
                    f"| kl {losses['kl'].item():.3f} "
                    f"| hand {hand_mse.item():.4f} "
                    f"| lr {scheduler.get_last_lr()[0]:.2e}"
                )

            if global_step % eval_every == 0:
                evaluate(accelerator, model, val_loader, beta, lambda_cross)
                model.train()

            if global_step % save_every == 0 or global_step >= total_steps:
                save_checkpoint(accelerator, global_step, out_dir, keep_last_n)

            if global_step >= total_steps:
                done = True
                break

    accelerator.print("Training complete.")


@torch.no_grad()
def evaluate(accelerator, model, val_loader, beta, lambda_cross, max_batches: int = 20):
    model.eval()
    agg = {"mse_self": 0.0, "mse_cross": 0.0, "kl": 0.0}
    n = 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        gt_future = batch["videos"][:, 1:]
        gt_cross = batch["cross_videos"][:, 1:] if "cross_videos" in batch else None
        outputs = model(batch)
        losses = lam_loss(outputs, gt_future, gt_cross, beta, lambda_cross)
        for k in agg:
            agg[k] += losses[k].item()
        n += 1
    if n == 0:
        return
    for k in agg:
        agg[k] /= n
    psnr = 10.0 * math.log10(1.0 / max(agg["mse_self"], 1e-12))
    accelerator.print(
        f"[val] self {agg['mse_self']:.5f} (psnr {psnr:.2f} dB) | "
        f"cross {agg['mse_cross']:.5f} | kl {agg['kl']:.3f}"
    )


if __name__ == "__main__":
    main()
