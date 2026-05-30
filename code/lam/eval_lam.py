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
    state_dict = torch.load(model_bin, map_location="cpu")
    # Accelerate may wrap keys with "module." prefix under DDP
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


def eval_visualize_latents(model, val_loader, device, n_samples=2000, out_path="latents.png"):
    """
    Extract latent action vectors and plot with t-SNE.
    This qualitative check verifies the latent space is structured.
    """
    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
    except ImportError:
        print("sklearn and matplotlib required for visualization: pip install scikit-learn matplotlib")
        return

    all_mu = []
    with torch.no_grad():
        for batch in val_loader:
            videos = batch["videos"].to(device)
            outputs = model({"videos": videos})
            # z_mu shape: [B*(T-1), latent_dim] = [B, 32] since T=2
            all_mu.append(outputs["z_mu"].cpu().numpy())
            if sum(x.shape[0] for x in all_mu) >= n_samples:
                break

    mu_arr = np.concatenate(all_mu, axis=0)[:n_samples]
    print(f"Running t-SNE on {mu_arr.shape[0]} latent vectors…")
    emb = TSNE(n_components=2, random_state=42, perplexity=40).fit_transform(mu_arr)

    plt.figure(figsize=(8, 8))
    plt.scatter(emb[:, 0], emb[:, 1], s=2, alpha=0.5)
    plt.title("t-SNE of LAM latent actions (z_mu)")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved t-SNE plot → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Accelerate checkpoint directory")
    parser.add_argument("--config",     required=True, help="Path to lam_agibot.yaml")
    parser.add_argument("--mode",       default="reconstruction",
                        choices=["reconstruction", "visualize"])
    parser.add_argument("--n_samples",  type=int, default=1000)
    parser.add_argument("--out_path",   default="latents_tsne.png",
                        help="Output path for t-SNE plot (visualize mode)")
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
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)
    print(f"Val dataset size: {len(val_ds):,}")

    if args.mode == "reconstruction":
        eval_reconstruction(model, val_loader, device, n_samples=args.n_samples)
    elif args.mode == "visualize":
        eval_visualize_latents(model, val_loader, device,
                               n_samples=args.n_samples, out_path=args.out_path)


if __name__ == "__main__":
    main()
