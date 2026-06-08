"""
LAM evaluation script.

Modes:
  --mode reconstruction   Compute MSE / PSNR on validation set (quantitative)
  --mode visualize        t-SNE of latent actions by motion direction (qualitative)

Usage:
  CUDA_VISIBLE_DEVICES=1 python eval_lam.py \\
      --checkpoint /home/xuan/embodied-ai/checkpoints/lam/step_0100000 \\
      --config /home/xuan/embodied-ai/code/lam/config/lam_agibot.yaml \\
      --mode reconstruction
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ADAWORLD_PATH = Path(__file__).resolve().parent.parent / "adaworld" / "lam"
sys.path.insert(0, str(ADAWORLD_PATH))
LAM_DATA_PATH = Path(__file__).resolve().parent
sys.path.insert(0, str(LAM_DATA_PATH))

from lam.modules import LatentActionModel
from data.agibot_dataset import AgibotVideoDataset


def load_model(checkpoint_dir: str, cfg: dict, device: torch.device) -> LatentActionModel:
    mcfg = cfg["model"]
    model = LatentActionModel(
        in_dim=mcfg.get("image_channels", 3),
        model_dim=mcfg.get("lam_model_dim", 512),
        latent_dim=mcfg.get("lam_latent_dim", 32),
        patch_size=mcfg.get("lam_patch_size", 16),
        enc_blocks=mcfg.get("lam_enc_blocks", 8),
        dec_blocks=mcfg.get("lam_dec_blocks", 8),
        num_heads=mcfg.get("lam_num_heads", 8),
        dropout=0.0,
    )
    # Accelerate saves full state including model weights in pytorch_model.bin
    ckpt_path = Path(checkpoint_dir)
    model_bin = ckpt_path / "pytorch_model.bin"
    if not model_bin.exists():
        # Try model.safetensors or model_0.safetensors
        candidates = list(ckpt_path.glob("*.safetensors")) + list(ckpt_path.glob("*.bin"))
        if not candidates:
            raise FileNotFoundError(f"No model weights found in {checkpoint_dir}")
        model_bin = candidates[0]
    if str(model_bin).endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(str(model_bin), device="cpu")
    else:
        state_dict = torch.load(model_bin, map_location="cpu")
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model


def eval_reconstruction(model, val_loader, device, n_samples=1000):
    """Compute per-pixel MSE and PSNR on validation frames."""
    total_mse = 0.0
    total_n = 0

    with torch.no_grad():
        for batch in val_loader:
            videos = batch["videos"].to(device)  # [B, 2, H, W, C]
            outputs = model({"videos": videos})
            gt = videos[:, 1:]          # [B, 1, H, W, C]
            recon = outputs["recon"]    # [B, 1, H, W, C]
            mse = ((gt - recon) ** 2).mean(dim=[1, 2, 3, 4])  # [B]
            total_mse += mse.sum().item()
            total_n   += videos.shape[0]
            if total_n >= n_samples:
                break

    avg_mse  = total_mse / total_n
    avg_psnr = 10 * np.log10(1.0 / (avg_mse + 1e-8))
    print(f"Eval on {total_n} samples:")
    print(f"  MSE  : {avg_mse:.6f}")
    print(f"  PSNR : {avg_psnr:.2f} dB")
    print(f"  (Reference: DreamDojo 'w/o pretrain' ≈ 20.3 dB)")
    return avg_mse, avg_psnr


def eval_visualize_latents(model, val_loader, device, n_samples=2000, out_path="latents.png", label=""):
    """
    Extract latent action vectors and plot t-SNE with two colorings:
      Left:  colored by task_id  (action semantic label)
      Right: colored by ep_id    (scene/appearance label)

    If the latent space is action-pure, left plot clusters well and
    right plot shows no structure (appearance not encoded).
    """
    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        print("sklearn and matplotlib required: pip install scikit-learn matplotlib")
        return

    all_mu      = []
    all_task_id = []
    all_ep_id   = []

    with torch.no_grad():
        for batch in val_loader:
            videos = batch["videos"].to(device)
            outputs = model({"videos": videos})
            all_mu.extend(outputs["z_mu"].cpu().numpy())
            # task_id / ep_id are lists of strings from the dataset
            all_task_id.extend(batch.get("task_id", ["unknown"] * videos.shape[0]))
            all_ep_id.extend(batch.get("ep_id",   ["unknown"] * videos.shape[0]))
            if len(all_mu) >= n_samples:
                break

    mu_arr      = np.array(all_mu[:n_samples])
    task_labels = all_task_id[:n_samples]
    ep_labels   = all_ep_id[:n_samples]

    print(f"Running t-SNE on {mu_arr.shape[0]} latent vectors…")
    emb = TSNE(n_components=2, random_state=42, perplexity=40).fit_transform(mu_arr)

    def _label_to_int(labels):
        uniq = sorted(set(labels))
        mapping = {v: i for i, v in enumerate(uniq)}
        return np.array([mapping[l] for l in labels]), uniq

    task_int, task_uniq = _label_to_int(task_labels)
    ep_int,   ep_uniq   = _label_to_int(ep_labels)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, color_int, uniq, title in [
        (axes[0], task_int, task_uniq, "Colored by task_id (action label)"),
        (axes[1], ep_int,   ep_uniq,   "Colored by ep_id (scene/appearance)"),
    ]:
        n_colors = len(uniq)
        cmap = cm.get_cmap("tab20" if n_colors <= 20 else "hsv", n_colors)
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=color_int, cmap=cmap,
                        s=3, alpha=0.6, vmin=0, vmax=n_colors - 1)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
        # Legend (cap at 20 entries to keep it readable)
        handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=cmap(i / max(n_colors - 1, 1)), markersize=6)
            for i in range(min(n_colors, 20))
        ]
        ax.legend(handles, uniq[:20], loc="best", fontsize=6,
                  markerscale=1.5, framealpha=0.6)

    title = f"t-SNE of LAM latent actions (z_mu){' — ' + label if label else ''}"
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved t-SNE plot → {out_path}")
    print(f"  Tasks: {task_uniq}")
    print(f"  Episodes: {len(ep_uniq)} unique")


def eval_linear_probe(model, val_loader, device, n_samples=5000, out_path="probe_results.txt", label=""):
    """
    Train linear probes on z_mu to measure action purity vs appearance contamination.

    Task probe:    z_mu → task_id    (high acc = action info is there)
    Episode probe: z_mu → ep_id      (high acc = scene/appearance is also encoded = contamination)

    Interpretation:
      Baseline:        task=high, ep=high  → latent encodes both action AND appearance
      Foreground-LAM:  task=high, ep=low   → latent encodes action, NOT appearance
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        from collections import Counter
    except ImportError:
        print("scikit-learn required: pip install scikit-learn")
        return

    all_mu      = []
    all_task_id = []
    all_ep_id   = []

    print(f"Extracting {n_samples} latent vectors…")
    with torch.no_grad():
        for batch in val_loader:
            videos = batch["videos"].to(device)
            outputs = model({"videos": videos})
            all_mu.extend(outputs["z_mu"].cpu().numpy())
            all_task_id.extend(batch.get("task_id", ["unknown"] * videos.shape[0]))
            all_ep_id.extend(batch.get("ep_id",   ["unknown"] * videos.shape[0]))
            if len(all_mu) >= n_samples:
                break

    mu_arr      = np.array(all_mu[:n_samples])
    task_labels = all_task_id[:n_samples]
    ep_labels   = all_ep_id[:n_samples]

    lines = [f"Linear Probe Results — {label}", "=" * 50]

    # ── Task probe ────────────────────────────────────────────────────────────
    le_task  = LabelEncoder()
    task_int = le_task.fit_transform(task_labels)
    X_tr, X_te, y_tr, y_te = train_test_split(
        mu_arr, task_int, test_size=0.2, random_state=42, stratify=task_int
    )
    clf_task = LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1)
    clf_task.fit(X_tr, y_tr)
    task_acc   = accuracy_score(y_te, clf_task.predict(X_te))
    task_chance = 1.0 / len(le_task.classes_)

    msg = (f"Task probe  (task_id):  {task_acc*100:5.1f}%  "
           f"[chance: {task_chance*100:.1f}%,  classes: {list(le_task.classes_)}]")
    print(msg); lines.append(msg)

    # ── Episode probe ─────────────────────────────────────────────────────────
    # Keep only episodes with ≥5 samples so the classifier has something to learn
    ep_counts = Counter(ep_labels)
    valid_eps = {ep for ep, cnt in ep_counts.items() if cnt >= 5}
    keep      = [i for i, ep in enumerate(ep_labels) if ep in valid_eps]
    mu_ep     = mu_arr[keep]
    ep_filt   = [ep_labels[i] for i in keep]

    le_ep  = LabelEncoder()
    ep_int = le_ep.fit_transform(ep_filt)
    X_tr, X_te, y_tr, y_te = train_test_split(
        mu_ep, ep_int, test_size=0.2, random_state=42, stratify=ep_int
    )
    clf_ep  = LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1)
    clf_ep.fit(X_tr, y_tr)
    ep_acc   = accuracy_score(y_te, clf_ep.predict(X_te))
    ep_chance = 1.0 / len(valid_eps)

    msg = (f"Episode probe (ep_id):  {ep_acc*100:5.1f}%  "
           f"[chance: {ep_chance*100:.2f}%,  {len(valid_eps)} episodes with ≥5 samples]")
    print(msg); lines.append(msg)

    # ── Interpretation ────────────────────────────────────────────────────────
    lines.append("")
    if ep_acc < 0.25:
        verdict = "CLEAN: latent is not contaminated by scene appearance."
    elif ep_acc < 0.5:
        verdict = "MILD contamination: some scene info encoded."
    else:
        verdict = "CONTAMINATED: latent encodes significant scene/appearance info."
    print(verdict); lines.append(verdict)

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text("\n".join(lines))
        print(f"Saved → {out_path}")

    return {"task_acc": task_acc, "ep_acc": ep_acc,
            "task_chance": task_chance, "ep_chance": ep_chance}


def eval_motion_cluster(model, cfg, device, n_samples=2000, out_path="motion_tsne.png", label=""):
    """
    t-SNE of latent actions colored by motion level derived from RAFT flow mask density.

    Motion level = fraction of foreground pixels in the precomputed RAFT mask.
      static  : < 5%   pixels moving  (robot repositioning / waiting)
      small   : 5-30%  pixels moving  (fine manipulation)
      large   : > 30%  pixels moving  (broad arm sweep)

    Uses the precomputed flow_masks saved by precompute_flow_masks.py.
    Falls back to 'no mask' category when a mask file is missing.
    """
    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("scikit-learn and matplotlib required.")
        return

    dcfg = cfg["data"]

    # Need fg masks to compute motion level — load val split with use_fg_mask=True
    val_ds_fg = AgibotVideoDataset(
        data_root=dcfg["data_root"],
        camera=dcfg.get("camera", "head_color"),
        img_h=dcfg.get("img_h", 240),
        img_w=dcfg.get("img_w", 320),
        downsample_factors=(1,),
        split="val",
        val_ratio=dcfg.get("val_ratio", 0.05),
        seed=dcfg.get("seed", 42),
        use_fg_mask=True,
    )

    def collate_fn(samples):
        batch = {}
        for key in samples[0]:
            if isinstance(samples[0][key], torch.Tensor):
                batch[key] = torch.stack([s[key] for s in samples])
            else:
                batch[key] = [s[key] for s in samples]
        return batch

    loader = torch.utils.data.DataLoader(
        val_ds_fg, batch_size=32, shuffle=True,
        num_workers=4, collate_fn=collate_fn,
    )

    all_mu    = []
    all_level = []   # 0=static, 1=small, 2=large, 3=no_mask
    all_task  = []

    THRESHOLDS = (0.01, 0.08)   # tuned to actual robot data density range (max ~18%)
    NAMES      = ["static (<1%)", "small (1-8%)", "large (>8%)", "no mask"]
    COLORS     = ["steelblue", "orange", "crimson", "lightgrey"]

    with torch.no_grad():
        for batch in loader:
            videos  = batch["videos"].to(device)
            outputs = model({"videos": videos})
            mu      = outputs["z_mu"].cpu()

            fg_mask = batch.get("fg_mask", None)   # [B, H, W] bool or None

            for i in range(mu.shape[0]):
                all_mu.append(mu[i].numpy())
                all_task.append(batch["task_id"][i] if "task_id" in batch else "?")

                if fg_mask is not None and i < fg_mask.shape[0]:
                    density = fg_mask[i].float().mean().item()
                    if density < THRESHOLDS[0]:
                        level = 0
                    elif density < THRESHOLDS[1]:
                        level = 1
                    else:
                        level = 2
                else:
                    level = 3
                all_level.append(level)

            if len(all_mu) >= n_samples:
                break

    mu_arr = np.array(all_mu[:n_samples])
    levels = all_level[:n_samples]
    tasks  = all_task[:n_samples]

    from collections import Counter
    cnt = Counter(levels)
    print("Motion level distribution:")
    for k, name in enumerate(NAMES):
        print(f"  {name}: {cnt[k]} samples ({cnt[k]/len(levels)*100:.1f}%)")

    print(f"Running t-SNE on {mu_arr.shape[0]} vectors…")
    emb = TSNE(n_components=2, random_state=42, perplexity=40).fit_transform(mu_arr)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: colored by motion level
    ax = axes[0]
    for lvl, name, color in zip(range(4), NAMES, COLORS):
        idx = [i for i, l in enumerate(levels) if l == lvl]
        if idx:
            ax.scatter(emb[idx, 0], emb[idx, 1], c=color, s=4, alpha=0.6,
                       label=f"{name} (n={len(idx)})")
    ax.set_title("Colored by motion level (RAFT density)", fontsize=11)
    ax.legend(fontsize=8, markerscale=2, framealpha=0.7)
    ax.axis("off")

    # Right: colored by task_id
    import matplotlib.cm as cm
    task_uniq = sorted(set(tasks))
    cmap      = cm.get_cmap("tab10", len(task_uniq))
    task_map  = {t: i for i, t in enumerate(task_uniq)}
    task_int  = [task_map[t] for t in tasks]
    ax2 = axes[1]
    sc  = ax2.scatter(emb[:, 0], emb[:, 1],
                      c=task_int, cmap=cmap, s=4, alpha=0.6,
                      vmin=0, vmax=len(task_uniq) - 1)
    handles = [mpatches.Patch(color=cmap(task_map[t]), label=f"task {t}") for t in task_uniq]
    ax2.legend(handles=handles, fontsize=8, framealpha=0.7)
    ax2.set_title("Colored by task_id", fontsize=11)
    ax2.axis("off")

    title = f"t-SNE — motion level clustering{' — ' + label if label else ''}"
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Accelerate checkpoint directory")
    parser.add_argument("--config",     required=True, help="Path to lam_agibot.yaml")
    parser.add_argument("--mode",       default="reconstruction",
                        choices=["reconstruction", "visualize", "probe", "motion"])
    parser.add_argument("--n_samples",  type=int, default=1000)
    parser.add_argument("--out_path",   default="latents_tsne.png",
                        help="Output path for t-SNE plot (visualize mode)")
    parser.add_argument("--label",      default="",
                        help="Optional label appended to the plot title, e.g. 'Baseline' or 'Foreground-LAM'")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    dcfg = cfg["data"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(args.checkpoint, cfg, device)
    print(f"Model loaded from {args.checkpoint}")

    val_ds = AgibotVideoDataset(
        data_root=dcfg["data_root"],
        camera=dcfg.get("camera", "top"),
        img_h=dcfg.get("img_h", 240),
        img_w=dcfg.get("img_w", 320),
        downsample_factors=(1,),
        split="val",
        val_ratio=dcfg.get("val_ratio", 0.05),
        seed=dcfg.get("seed", 42),
    )
    def collate_fn(samples):
        batch = {}
        for key in samples[0]:
            if isinstance(samples[0][key], torch.Tensor):
                batch[key] = torch.stack([s[key] for s in samples])
            else:
                batch[key] = [s[key] for s in samples]
        return batch

    val_loader = DataLoader(val_ds, batch_size=32, shuffle=True, num_workers=4,
                            collate_fn=collate_fn)
    print(f"Val dataset size: {len(val_ds):,}")

    if args.mode == "reconstruction":
        eval_reconstruction(model, val_loader, device, n_samples=args.n_samples)
    elif args.mode == "visualize":
        eval_visualize_latents(model, val_loader, device,
                               n_samples=args.n_samples, out_path=args.out_path,
                               label=args.label)
    elif args.mode == "probe":
        probe_out = Path(args.out_path).with_suffix(".txt")
        eval_linear_probe(model, val_loader, device,
                          n_samples=args.n_samples, out_path=str(probe_out),
                          label=args.label)
    elif args.mode == "motion":
        eval_motion_cluster(model, cfg, device,
                            n_samples=args.n_samples, out_path=args.out_path,
                            label=args.label)


if __name__ == "__main__":
    main()
