#!/usr/bin/env python
"""
End-to-end smoke test with synthetic data -- no Ego-Exo4D download required.

Builds a fake processed tree (a few takes x a few cameras of short random
videos), then exercises: dataset sampling in all three modes, model forward with
and without cross-view, the loss, a backward pass, and the view-prompt arm.

    python fast_wam/wam/scripts/smoke_test.py

Run this FIRST on any new machine. If it passes, the training script's failure
modes are limited to data paths and GPU memory.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from dataset import EgoExoMultiViewDataset, collate  # noqa: E402
from model import MultiViewLAM, lam_loss             # noqa: E402

N_TAKES, N_CAMS, N_FRAMES = 3, 4, 24


def make_fake_tree(root: Path) -> None:
    takes = []
    for ti in range(N_TAKES):
        uid = f"take{ti:03d}"
        d = root / uid
        d.mkdir(parents=True, exist_ok=True)
        for ci in range(N_CAMS):
            frames = (np.random.rand(N_FRAMES, 240, 320, 3) * 255).astype(np.uint8)
            raw = d / f"exo{ci}.raw"
            raw.write_bytes(frames.tobytes())
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "320x240",
                 "-r", "30", "-i", str(raw),
                 "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                 "-g", "1", "-pix_fmt", "yuv420p", str(d / f"exo{ci}.mp4")],
                check=True,
            )
            raw.unlink()
        # fake hand poses so the action path is exercised too
        left = np.tile(np.eye(4, dtype=np.float32), (N_FRAMES, 1, 1))
        right = left.copy()
        left[:, :3, 3] = np.cumsum(np.random.randn(N_FRAMES, 3).astype(np.float32) * 0.01, 0)
        right[:, :3, 3] = np.cumsum(np.random.randn(N_FRAMES, 3).astype(np.float32) * 0.01, 0)
        np.savez(d / "hand_poses.npz", left=left, right=right,
                 valid_left=np.ones(N_FRAMES, bool), valid_right=np.ones(N_FRAMES, bool))
        takes.append({
            "uid": uid, "take_name": uid, "scenario": "fake",
            "n_frames": N_FRAMES,
            "cams": [f"exo{c}" for c in range(N_CAMS)],
            "train_cams": [f"exo{c}" for c in range(N_CAMS - 1)],
            "heldout_cam": f"exo{N_CAMS-1}",
            "has_hand": True,
        })
    with open(root / "index.json", "w") as f:
        json.dump({"img_h": 240, "img_w": 320, "takes": takes}, f, indent=1)


def main() -> int:
    if subprocess.run(["which", "ffmpeg"], capture_output=True).returncode != 0:
        print("FAIL: ffmpeg not on PATH"); return 1

    # The vendored AdaWorld PositionalEncoding hardcodes `.cuda()`, so the model
    # only runs on GPU. This is inherited from upstream and left unmodified.
    if not torch.cuda.is_available():
        print("FAIL: CUDA required (AdaWorld blocks hardcode .cuda())"); return 1
    dev = "cuda"

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        print("[1/6] building fake processed tree ...")
        make_fake_tree(root)

        # small model so this runs in seconds
        common = dict(model_dim=64, latent_dim=8, patch_size=16,
                      enc_blocks=1, dec_blocks=1, num_heads=2)

        for mode, expect_cross in [("single", False), ("multi", False), ("cross", True)]:
            print(f"[2/6] dataset mode={mode} ...")
            ds = EgoExoMultiViewDataset(
                processed_root=str(root), mode=mode, view_pairs="exo_exo",
                split="train", val_ratio=0.34, seed=42, pairs_per_take=4,
                load_actions=True,
            )
            batch = collate([ds[i] for i in range(4)])
            assert batch["videos"].shape == (4, 2, 240, 320, 3), batch["videos"].shape
            assert ("cross_videos" in batch) == expect_cross, mode
            assert batch["action"].shape == (4, 18)
            assert batch["videos"].min() >= 0.0 and batch["videos"].max() <= 1.0
            print(f"      ok: videos {tuple(batch['videos'].shape)} "
                  f"cross={'yes' if expect_cross else 'no'} action {tuple(batch['action'].shape)}")

        batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}

        print("[3/6] model forward (cross) ...")
        model = MultiViewLAM(num_views=0, **common).to(dev)
        out = model(batch)
        assert out["recon"].shape == (4, 1, 240, 320, 3), out["recon"].shape
        assert out["recon_cross"].shape == (4, 1, 240, 320, 3)
        assert out["z_mu"].shape == (4, 8)

        print("[4/6] loss + backward ...")
        losses = lam_loss(out, batch["videos"][:, 1:], batch["cross_videos"][:, 1:],
                          beta=1e-6, lambda_cross=1.0)
        assert torch.isfinite(losses["loss"]), losses
        losses["loss"].backward()
        n_grad = sum(1 for p in model.parameters() if p.grad is not None)
        assert n_grad > 0
        print(f"      loss={losses['loss'].item():.4f} self={losses['mse_self'].item():.4f} "
              f"cross={losses['mse_cross'].item():.4f} kl={losses['kl'].item():.4f} "
              f"({n_grad} tensors got grad)")

        print("[5/6] view-prompt arm ...")
        vp = MultiViewLAM(num_views=8, **common).to(dev)
        out2 = vp(batch)
        assert out2["recon"].shape == (4, 1, 240, 320, 3)
        # the prompt must actually change the output
        b2 = dict(batch)
        b2["view_ids"] = (batch["view_ids"] + 1) % 8
        vp.eval()
        with torch.no_grad():
            a = vp(dict(batch, cross_videos=None))["recon"]
            b = vp(dict(b2, cross_videos=None))["recon"]
        assert not torch.allclose(a, b), "view prompt had no effect"
        print("      ok: view embedding changes the reconstruction")

        print("[6/6] eval-time determinism (z_mu, no sampling) ...")
        model.eval()
        with torch.no_grad():
            z1 = model.encode(batch["videos"])["z_mu"]
            z2 = model.encode(batch["videos"])["z_mu"]
        assert torch.allclose(z1, z2), "encode is not deterministic in eval mode"
        print("      ok")

    print("\nALL SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
