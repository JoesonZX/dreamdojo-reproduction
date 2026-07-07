"""
LAM disentanglement training (Accelerate).

Vendored from code/lam/train_lam.py, trimmed of the SigLIP path, and EXTENDED
with two optional losses driven by config flags:

  action loss        : MSE( action_predictor(z_mu[:, :action_part]), real_action )
  decorrelation loss : squared cross-covariance between the action subspace
                       z_mu[:, :action_part] and env subspace z_mu[:, action_part:]

Config flags (model:):  env_dim, action_dim, action_head, lambda_action, lambda_indep
Config flags (data:) :  load_actions

Usage:
  # smoke test (1 GPU, 5 steps):
  CUDA_VISIBLE_DEVICES=1 python lam_disentangle/code/train.py \
      --config lam_disentangle/code/config/exp2_split.yaml --dry_run

  # full run (3 GPUs):
  CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --num_processes 3 \
      --mixed_precision bf16 lam_disentangle/code/train.py \
      --config lam_disentangle/code/config/exp2_split.yaml
"""

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from model import LatentActionModel          # noqa: E402  (local, extended)
from dataset import EgoDexDataset, PKBatchSampler   # noqa: E402  (local, extended)

from accelerate import Accelerator           # noqa: E402
from accelerate.utils import set_seed        # noqa: E402


# ── Losses ────────────────────────────────────────────────────────────────────

def lam_loss(outputs, gt_future, beta, fg_mask=None, fg_weight=5.0,
             action_part=None, beta_env=None):
    """VAE loss: (optionally fg-weighted) MSE + KL.

    If `beta_env` and `action_part` are given, the KL is weighted PER SUBSPACE:
      beta      applied to the action subspace dims [0:action_part]   (= beta_action)
      beta_env  applied to the env subspace dims    [action_part:]
    A heavier beta on the action subspace turns it into an information bottleneck
    (it keeps only what the action loss defends), letting env info migrate to z_e.
    Otherwise a single `beta` is applied to all dims (original behaviour).
    """
    err = (gt_future - outputs["recon"]) ** 2
    if fg_mask is not None:
        w = 1.0 + (fg_weight - 1.0) * fg_mask.float().unsqueeze(1).unsqueeze(-1)
        mse = (w * err).mean()
    else:
        mse = err.mean()
    # per-dim KL( N(mu, e^var) || N(0,1) ):  [B, D]
    kl_per = -0.5 * (1 + outputs["z_var"] - outputs["z_mu"] ** 2 - outputs["z_var"].exp())
    if beta_env is not None and action_part is not None:
        kl_a = kl_per[:, :action_part].sum(dim=1).mean()
        kl_e = kl_per[:, action_part:].sum(dim=1).mean()
        kl_weighted = beta * kl_a + beta_env * kl_e
        kl = kl_a + kl_e                      # unweighted total, for logging
        return mse + kl_weighted, mse, kl
    kl = kl_per.sum(dim=1).mean()
    return mse + beta * kl, mse, kl


def action_loss(pred_action, gt_action):
    """MSE between predicted and (z-scored) real action. Both [B, action_dim]."""
    return F.mse_loss(pred_action, gt_action)


def decorrelation_loss(A, E):
    """Squared cross-covariance (VICReg/Barlow style) between subspaces.

    A: [B, da] action subspace,  E: [B, de] env subspace. Push all cross terms -> 0.
    """
    A = (A - A.mean(0, keepdim=True)) / (A.std(0, keepdim=True) + 1e-5)
    E = (E - E.mean(0, keepdim=True)) / (E.std(0, keepdim=True) + 1e-5)
    C = (A.T @ E) / (A.shape[0] - 1)         # [da, de]
    return (C ** 2).sum()


def supcon_loss(feats, labels, tau=0.1):
    """Supervised contrastive loss (Khosla 2020) — ConLA action-centric term.

    feats: [B, d] L2-normalized.  labels: LongTensor [B] (within-batch action/task id).
    Pulls together same-label embeddings, pushes apart different labels. Returns a
    zero (grad-carrying) tensor if no positive pair exists in the batch.
    """
    B = feats.shape[0]
    sim = (feats @ feats.T) / tau                                   # [B, B]
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()        # stability
    self_mask = torch.eye(B, dtype=torch.bool, device=feats.device)
    pos_mask = (labels[:, None] == labels[None, :]) & ~self_mask
    exp_sim = torch.exp(sim).masked_fill(self_mask, 0.0)
    log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-12)
    pos_counts = pos_mask.sum(1)
    valid = pos_counts > 0
    if valid.sum() == 0:
        return feats.sum() * 0.0
    mean_log_prob_pos = (pos_mask * log_prob).sum(1)[valid] / pos_counts[valid]
    return -mean_log_prob_pos.mean()


# ── Checkpoint helpers ──────────────────────────────────────────────────────────

def save_checkpoint(accelerator, model, optimizer, scheduler, step, output_dir, keep_last_n):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f"step_{step:07d}"
    accelerator.save_state(str(ckpt_path))
    ckpts = sorted(output_dir.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    for old in ckpts[:-keep_last_n]:
        import shutil
        shutil.rmtree(old, ignore_errors=True)
    accelerator.print(f"[save] checkpoint -> {ckpt_path}")


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


def load_pretrained_lam(model, ckpt_path, accelerator):
    """Initialize `model` from a pretrained LAM checkpoint for fine-tuning.

    Handles both safetensors and Lightning/torch .ckpt (which nests weights under
    a "state_dict" key and often prefixes them with "lam." / "model." / "module.").
    Loads with strict=False so:
      - new params (action_predictor, any extra env dims in fc) stay randomly init'd
      - shape-mismatched params (e.g. fc when latent 32->40) are skipped per-tensor
    Prints a summary of loaded / missing / skipped tensors.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"--init_from checkpoint not found: {ckpt_path}")

    # ── read raw state dict ──
    if ckpt_path.suffix == ".safetensors":
        from safetensors.torch import load_file
        sd = load_file(str(ckpt_path), device="cpu")
    else:
        obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        sd = obj.get("state_dict", obj) if isinstance(obj, dict) else obj

    # ── strip common prefixes so keys line up with our LatentActionModel ──
    model_keys = set(model.state_dict().keys())
    def strip(k):
        for pre in ("module.", "model.", "lam.", "net.", "ema_model."):
            if k.startswith(pre):
                k = k[len(pre):]
        return k
    remapped = {}
    for k, v in sd.items():
        nk = strip(k)
        remapped[nk] = v
    # if still no overlap, try stripping the longest leading dotted segment
    if not (set(remapped) & model_keys):
        alt = {}
        for k, v in sd.items():
            parts = k.split(".")
            for i in range(len(parts)):
                cand = ".".join(parts[i:])
                if cand in model_keys:
                    alt[cand] = v
                    break
        if alt:
            remapped = alt

    # ── keep only matching name AND shape ──
    cur = model.state_dict()
    keep, skip_shape = {}, []
    for k, v in remapped.items():
        if k in cur:
            if tuple(cur[k].shape) == tuple(v.shape):
                keep[k] = v
            else:
                skip_shape.append((k, tuple(v.shape), tuple(cur[k].shape)))
    missing = sorted(model_keys - set(keep))
    matched = len(keep)

    model.load_state_dict(keep, strict=False)
    accelerator.print(f"[init_from] {ckpt_path.name}: loaded {matched}/{len(model_keys)} tensors")
    if skip_shape:
        accelerator.print(f"[init_from] shape-mismatch skipped ({len(skip_shape)}): "
                          + ", ".join(f"{k}{s_ck}->{s_cur}" for k, s_ck, s_cur in skip_shape[:6])
                          + (" ..." if len(skip_shape) > 6 else ""))
    if missing:
        accelerator.print(f"[init_from] left random-init ({len(missing)}): "
                          + ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else ""))
    if matched == 0:
        raise RuntimeError(f"[init_from] 0 tensors matched — key naming likely differs; "
                           f"sample ckpt keys: {list(sd.keys())[:5]}")
    return matched


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry_run", action="store_true", help="Run 5 steps then exit")
    parser.add_argument("--init_from", default=None,
                        help="Pretrained LAM checkpoint (.ckpt/.safetensors) to fine-tune from. "
                             "Overrides config's model.init_from if given.")
    parser.add_argument("--freeze_encoder", action="store_true",
                        help="Freeze encoder + action_prompt; train only fc/action_predictor/decoder. "
                             "Overrides config's model.freeze_encoder if given.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    mcfg = cfg["model"]
    dcfg = cfg["data"]
    tcfg = cfg["training"]
    lcfg = cfg["logging"]
    fgcfg = cfg.get("foreground", {})

    accelerator = Accelerator(
        mixed_precision=tcfg.get("mixed_precision", "bf16"),
        gradient_accumulation_steps=tcfg.get("gradient_accumulation_steps", 1),
        log_with="wandb" if lcfg.get("use_wandb") else None,
    )
    set_seed(42)

    if accelerator.is_main_process:
        if lcfg.get("use_wandb"):
            accelerator.init_trackers(
                project_name=lcfg.get("project", "dreamdojo-lam-disentangle"),
                config=cfg,
                init_kwargs={"wandb": {"name": lcfg.get("run_name", "lam-run")}},
            )
        Path(tcfg["output_dir"]).mkdir(parents=True, exist_ok=True)

    # ── Disentanglement flags ───────────────────────────────────────────────────
    env_dim       = mcfg.get("env_dim", 0)
    action_dim    = mcfg.get("action_dim", 0)
    use_action    = mcfg.get("action_head", False)
    lambda_action = mcfg.get("lambda_action", 0.0)
    lambda_indep  = mcfg.get("lambda_indep", 0.0)
    indep_warmup  = mcfg.get("indep_warmup", 0)   # ramp lambda_indep 0->target over N steps
    use_indep     = lambda_indep > 0 and env_dim > 0
    # ── ConLA contrastive disentanglement flags ──
    contrastive     = mcfg.get("contrastive", False)
    lambda_supcon   = mcfg.get("lambda_supcon", 0.0)
    lambda_temporal = mcfg.get("lambda_temporal", 0.0)
    supcon_tau      = mcfg.get("supcon_tau", 0.1)
    contrastive_warmup = mcfg.get("contrastive_warmup", 0)   # ramp supcon+temporal 0->target
    use_contrastive = contrastive and (lambda_supcon > 0 or lambda_temporal > 0)
    load_actions  = dcfg.get("load_actions", False) or use_action

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
        env_dim=env_dim,
        action_dim=action_dim,
        action_head=use_action,
        contrastive=contrastive,
        proj_dim=mcfg.get("proj_dim", 64),
    )
    beta        = mcfg.get("beta", 1e-6)         # target KL weight for action subspace when beta_env set
    beta_env    = mcfg.get("beta_env", None)     # KL weight for env subspace (None => single global beta)
    beta_warmup = mcfg.get("beta_warmup", 0)     # anneal beta_a from beta_env -> beta over N steps (0=off)
    use_fg    = fgcfg.get("use_fg_loss", False)
    fg_weight = fgcfg.get("fg_weight", 5.0)
    action_part = model.action_part
    use_split_kl = beta_env is not None and env_dim > 0

    accelerator.print(
        f"latent_dim={model.latent_dim} action_part={action_part} env_dim={env_dim} | "
        f"action_head={use_action} (lambda_action={lambda_action}) | "
        f"indep={use_indep} (lambda_indep={lambda_indep}) | "
        f"split_kl={use_split_kl} (beta_a={beta}, beta_e={beta_env}) | "
        f"contrastive={use_contrastive} (supcon={lambda_supcon}, temporal={lambda_temporal}, tau={supcon_tau})"
    )
    accelerator.print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # ── Fine-tune init: load pretrained LAM weights, optionally freeze encoder ──
    # Only applies on a fresh run (no existing checkpoint to resume); see resume logic below.
    init_from = args.init_from or mcfg.get("init_from", None)
    freeze_encoder = args.freeze_encoder or mcfg.get("freeze_encoder", False)
    if init_from and not list(Path(tcfg["output_dir"]).glob("step_*")):
        load_pretrained_lam(model, init_from, accelerator)
    if freeze_encoder:
        frozen = 0
        for name, p in model.named_parameters():
            if name.startswith("encoder.") or name == "action_prompt":
                p.requires_grad_(False); frozen += p.numel()
        accelerator.print(f"[freeze] encoder+action_prompt frozen ({frozen/1e6:.1f}M params); "
                          f"trainable: {sum(q.numel() for q in model.parameters() if q.requires_grad)/1e6:.1f}M")

    # ── Dataset (EgoDex only) ───────────────────────────────────────────────────
    if dcfg.get("dataset_type", "egodex") != "egodex":
        raise ValueError("This experiment package supports dataset_type=egodex only.")

    # Verbs needed if we group/contrast by the scene-independent action verb.
    need_verbs = dcfg.get("load_verbs", False) or dcfg.get("pk_label", "task") == "verb"
    supcon_label_key = "verb_id" if dcfg.get("pk_label", "task") == "verb" else "task_id"

    train_ds = EgoDexDataset(
        data_root=dcfg["data_root"],
        img_h=dcfg.get("img_h", 240),
        img_w=dcfg.get("img_w", 320),
        downsample_factors=tuple(dcfg.get("downsample_factors", [1, 2])),
        split="train",
        val_ratio=dcfg.get("val_ratio", 0.1),
        seed=dcfg.get("seed", 42),
        use_fg_mask=use_fg,
        load_actions=load_actions,
        action_dim=action_dim if load_actions else 18,
        load_verbs=need_verbs,
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
        load_actions=load_actions,
        action_dim=action_dim if load_actions else 18,
    )
    accelerator.print(f"Train pairs: {len(train_ds):,}  Val pairs: {len(val_ds):,}")

    def collate_fn(samples):
        batch = {}
        for key in samples[0]:
            if isinstance(samples[0][key], torch.Tensor):
                batch[key] = torch.stack([s[key] for s in samples])
            else:
                batch[key] = [s[key] for s in samples]
        return batch

    if dcfg.get("pk_sampler", False):
        # Class-balanced PK batches so supervised-contrastive has same-CLASS positives.
        # pk_label="verb": scene-independent action verb (same verb spans many scenes);
        # pk_label="task": original task label (≈ scene in EgoDex — imports scene).
        accum = tcfg.get("gradient_accumulation_steps", 1)
        n_batches = tcfg.get("total_steps", 20000) * accum + 100
        pk_label = dcfg.get("pk_label", "task")
        if pk_label == "verb":
            index_labels = [train_ds._verb_map.get(rec[3]) for rec in train_ds._index]  # rec[3]=ep_id
        else:
            index_labels = [rec[2] for rec in train_ds._index]                          # rec[2]=task_id
        pk = PKBatchSampler(
            index_labels,
            classes_per_batch=dcfg.get("classes_per_batch", 2),
            samples_per_class=dcfg.get("samples_per_class", 4),
            num_batches=n_batches, seed=dcfg.get("seed", 42),
            exclude={None, "unknown"},
        )
        accelerator.print(f"PK sampler [{pk_label}]: P={pk.P} x K={pk.K} (batch {pk.P*pk.K}) over {len(pk.tasks)} classes")
        train_loader = DataLoader(
            train_ds, batch_sampler=pk,
            num_workers=tcfg.get("num_workers", 8), pin_memory=tcfg.get("pin_memory", True),
            persistent_workers=tcfg.get("num_workers", 8) > 0, collate_fn=collate_fn,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=tcfg.get("per_gpu_batch_size", 16), shuffle=True,
            num_workers=tcfg.get("num_workers", 8), pin_memory=tcfg.get("pin_memory", True),
            drop_last=True, persistent_workers=tcfg.get("num_workers", 8) > 0, collate_fn=collate_fn,
        )
    val_loader = DataLoader(
        val_ds, batch_size=tcfg.get("per_gpu_batch_size", 16), shuffle=False,
        num_workers=4, pin_memory=True, drop_last=False, collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=tcfg.get("lr", 2.5e-5),
        weight_decay=tcfg.get("weight_decay", 0.01),
    )
    total_steps = tcfg.get("total_steps", 20000)
    warmup_steps = tcfg.get("warmup_steps", 1000)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    global_step = resume_from_checkpoint(accelerator, tcfg["output_dir"])

    log_every   = tcfg.get("log_every", 100)
    save_every  = tcfg.get("save_every", 2000)
    eval_every  = tcfg.get("eval_every", 1000)
    keep_last_n = tcfg.get("keep_last_n", 3)
    log_path    = mcfg.get("log_path", tcfg["output_dir"] + "/log_imgs")

    model.train()
    train_iter = iter(train_loader)
    r_loss = r_mse = r_kl = r_act = r_ind = r_sup = r_tmp = 0.0

    accelerator.print(f"Starting training from step {global_step} -> {total_steps}")

    while global_step < total_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        with accelerator.accumulate(model):
            outputs = model(batch)
            gt_future = batch["videos"][:, 1:]
            fg_mask = batch.get("fg_mask", None)
            if fg_mask is not None:
                fg_mask = fg_mask.to(batch["videos"].device)
            # anneal the action-subspace KL weight beta_a (beta_env -> beta) if requested
            if use_split_kl and beta_warmup > 0:
                beta_a_eff = beta_env + (beta - beta_env) * min(1.0, global_step / beta_warmup)
            else:
                beta_a_eff = beta
            loss, mse, kl = lam_loss(outputs, gt_future, beta_a_eff, fg_mask, fg_weight,
                                     action_part=action_part if use_split_kl else None,
                                     beta_env=beta_env if use_split_kl else None)

            l_act = torch.tensor(0.0, device=loss.device)
            if use_action and "pred_action" in outputs:
                gt_action = batch["action"].to(loss.device).float()
                l_act = action_loss(outputs["pred_action"], gt_action)
                loss = loss + lambda_action * l_act

            l_ind = torch.tensor(0.0, device=loss.device)
            if use_indep:
                z_mu = outputs["z_mu"]
                l_ind = decorrelation_loss(z_mu[:, :action_part], z_mu[:, action_part:])
                # warmup: ramp the independence weight 0 -> lambda_indep over indep_warmup steps
                w = lambda_indep * (min(1.0, global_step / indep_warmup) if indep_warmup > 0 else 1.0)
                loss = loss + w * l_ind

            # ── ConLA contrastive disentanglement (replaces the KL separator) ──
            l_sup = torch.tensor(0.0, device=loss.device)
            l_tmp = torch.tensor(0.0, device=loss.device)
            if use_contrastive and "z_a_proj" in outputs:
                z_a_proj = outputs["z_a_proj"]
                # warmup: ramp contrastive weight 0 -> 1 so the action head settles first
                cw = min(1.0, global_step / contrastive_warmup) if contrastive_warmup > 0 else 1.0
                # Supervised contrastive on z_a using within-batch task labels (action
                # category prior): pulls same-action latents together ACROSS episodes.
                if lambda_supcon > 0:
                    uniq = {}
                    labels = torch.tensor(
                        [uniq.setdefault(s, len(uniq)) for s in batch[supcon_label_key]],
                        device=loss.device, dtype=torch.long,
                    )
                    l_sup = supcon_loss(z_a_proj, labels, supcon_tau)
                    loss = loss + cw * lambda_supcon * l_sup
                # Temporal reverse cue: re-encode the time-flipped pair. Motion flips
                # (z_a should change) but appearance is stable (z_e should not).
                # ENCODE-ONLY (skip the decoder) — we only need the projections, so this
                # avoids a second full forward and its memory blow-up on the 700M model.
                if lambda_temporal > 0:
                    rev_out = accelerator.unwrap_model(model).encode(batch["videos"].flip(dims=[1]))
                    l_act_sens = F.relu((z_a_proj * rev_out["z_a_proj"]).sum(1)).mean()
                    if "z_e_proj" in outputs and "z_e_proj" in rev_out:
                        l_env_inv = (1.0 - (outputs["z_e_proj"] * rev_out["z_e_proj"]).sum(1)).mean()
                    else:
                        l_env_inv = torch.tensor(0.0, device=loss.device)
                    l_tmp = l_act_sens + l_env_inv
                    loss = loss + cw * lambda_temporal * l_tmp

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            global_step += 1
            r_loss += loss.item(); r_mse += mse.item(); r_kl += kl.item()
            r_act += l_act.item();  r_ind += l_ind.item()
            r_sup += l_sup.item();  r_tmp += l_tmp.item()

            if global_step % log_every == 0 and accelerator.is_main_process:
                avg = 1.0 / log_every
                metrics = {
                    "train/loss": r_loss * avg, "train/mse": r_mse * avg,
                    "train/kl": r_kl * avg, "train/action": r_act * avg,
                    "train/indep": r_ind * avg, "train/supcon": r_sup * avg,
                    "train/temporal": r_tmp * avg, "train/lr": scheduler.get_last_lr()[0],
                    "global_step": global_step,
                }
                extra = ""
                if use_action:
                    extra += f"| act {metrics['train/action']:.4f} "
                if use_indep:
                    extra += f"| indep {metrics['train/indep']:.4f} "
                if use_contrastive:
                    extra += f"| supcon {metrics['train/supcon']:.4f} | temp {metrics['train/temporal']:.4f} "
                accelerator.print(
                    f"step {global_step:6d} | loss {metrics['train/loss']:.4f} "
                    f"| mse {metrics['train/mse']:.4f} | kl {metrics['train/kl']:.6f} "
                    f"{extra}| lr {metrics['train/lr']:.2e}"
                )
                if lcfg.get("use_wandb"):
                    accelerator.log(metrics, step=global_step)
                r_loss = r_mse = r_kl = r_act = r_ind = r_sup = r_tmp = 0.0

            if global_step % eval_every == 0:
                model.eval()
                val_loss_sum = val_mse_sum = 0.0
                val_n = 0
                with torch.no_grad():
                    for vbatch in val_loader:
                        vout = model(vbatch)
                        gt_v = vbatch["videos"][:, 1:]
                        vloss, vmse, _ = lam_loss(vout, gt_v, beta,
                                                  action_part=action_part if use_split_kl else None,
                                                  beta_env=beta_env if use_split_kl else None)
                        bs = vbatch["videos"].shape[0]
                        val_loss_sum += vloss.item() * bs
                        val_mse_sum  += vmse.item() * bs
                        val_n += bs
                        if val_n >= 512:
                            break
                if accelerator.is_main_process and val_n > 0:
                    psnr = 10 * math.log10(1.0 / (val_mse_sum / val_n + 1e-8))
                    accelerator.print(
                        f"[val] step {global_step:6d} | loss {val_loss_sum/val_n:.4f} "
                        f"| mse {val_mse_sum/val_n:.4f} | psnr {psnr:.2f} dB"
                    )
                    if lcfg.get("use_wandb"):
                        accelerator.log({"val/loss": val_loss_sum/val_n,
                                         "val/mse": val_mse_sum/val_n,
                                         "val/psnr": psnr}, step=global_step)
                    _save_recon_image(vbatch, vout, global_step, log_path)
                model.train()

            if global_step % save_every == 0 or global_step == total_steps:
                save_checkpoint(accelerator, model, optimizer, scheduler,
                                global_step, tcfg["output_dir"], keep_last_n)

        if args.dry_run and global_step >= 5:
            accelerator.print("Dry run complete — exiting.")
            break

    if lcfg.get("use_wandb"):
        accelerator.end_training()
    accelerator.print("Training complete.")


def _save_recon_image(batch, outputs, step, log_path):
    try:
        import numpy as np
        from PIL import Image
        Path(log_path).mkdir(parents=True, exist_ok=True)
        gt_pair = batch["videos"][0].cpu().float().clamp(0, 1)
        recon   = outputs["recon"][0].cpu().float().clamp(0, 1)
        blank   = gt_pair[0:1]
        top = torch.cat([gt_pair[0:1], gt_pair[1:2]], dim=2)
        bot = torch.cat([blank, recon[0:1]], dim=2)
        compare = torch.cat([top, bot], dim=1)
        img_arr = (compare[0] * 255).numpy().astype(np.uint8)
        Image.fromarray(img_arr).save(Path(log_path) / f"step_{step:07d}.png")
    except Exception:
        pass


if __name__ == "__main__":
    main()
