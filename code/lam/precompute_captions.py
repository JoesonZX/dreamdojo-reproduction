"""
Precompute SigLIP text embeddings for all EgoDex episodes.

Reads llm_description from each episode's HDF5 file, encodes with
SigLIP text encoder (google/siglip-base-patch16-224), and saves:
  - Per-episode: {task_name}/captions/{idx}.pt  — float32 [768]
  - Index file:  captions_index.pt              — dict {ep_id -> embedding}

Usage:
  CUDA_VISIBLE_DEVICES=0 python precompute_captions.py \\
      --data_root /home/xuan/embodied-ai/data/egodex/test

  # Dry run (first 10 episodes only):
  CUDA_VISIBLE_DEVICES=0 python precompute_captions.py \\
      --data_root /home/xuan/embodied-ai/data/egodex/test --dry_run
"""

import argparse
import sys
from pathlib import Path

import h5py
import torch
from tqdm import tqdm


def load_siglip_text_encoder(model_name: str, device: torch.device):
    from transformers import AutoTokenizer, AutoModel
    print(f"Loading SigLIP text encoder: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    print(f"  Parameters: {sum(p.numel() for p in model.text_model.parameters())/1e6:.1f}M (text encoder only)")
    return tokenizer, model


@torch.no_grad()
def encode_texts(texts: list, tokenizer, model, device: torch.device, max_length: int = 64) -> torch.Tensor:
    """Encode a batch of strings → float32 [B, 768]."""
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
        truncation=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    emb = model.text_model(**inputs).pooler_output   # [B, 768]
    return emb.cpu().float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",  default="/home/xuan/embodied-ai/data/egodex/test")
    parser.add_argument("--model_name", default="google/siglip-base-patch16-224")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--dry_run",    action="store_true", help="Process first 10 episodes only")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_root = Path(args.data_root)
    hdf5_files = sorted(data_root.glob("*/*.hdf5"))
    if not hdf5_files:
        print(f"No .hdf5 files found under {args.data_root}")
        sys.exit(1)

    if args.dry_run:
        hdf5_files = hdf5_files[:10]
        print(f"Dry run: processing first {len(hdf5_files)} episodes")
    else:
        print(f"Found {len(hdf5_files)} episodes")

    # ── Collect all captions ──────────────────────────────────────────────────
    print("Reading captions from HDF5 files...")
    ep_ids    = []   # "task_name/idx"
    captions  = []
    missing   = 0
    for p in tqdm(hdf5_files, desc="reading hdf5"):
        task_id = p.parent.name
        ep_id   = f"{task_id}/{p.stem}"
        try:
            with h5py.File(p, "r") as f:
                desc = f.attrs.get("llm_description", None)
                if desc is None:
                    desc = f.attrs.get("task", task_id)  # fallback to task name
                    missing += 1
        except Exception:
            desc = task_id
            missing += 1
        ep_ids.append(ep_id)
        captions.append(str(desc))

    print(f"  {len(captions)} captions loaded, {missing} used fallback (task name)")

    # ── Load model ────────────────────────────────────────────────────────────
    tokenizer, model = load_siglip_text_encoder(args.model_name, device)

    # ── Encode in batches ─────────────────────────────────────────────────────
    print(f"Encoding with batch_size={args.batch_size}...")
    all_embeddings = []
    for i in tqdm(range(0, len(captions), args.batch_size), desc="encoding"):
        batch_texts = captions[i : i + args.batch_size]
        emb = encode_texts(batch_texts, tokenizer, model, device, args.max_length)
        all_embeddings.append(emb)

    all_embeddings = torch.cat(all_embeddings, dim=0)  # [N, 768]
    print(f"Embeddings shape: {all_embeddings.shape}")

    # ── Save per-episode files ────────────────────────────────────────────────
    print("Saving per-episode .pt files...")
    saved = 0
    for ep_id, emb in tqdm(zip(ep_ids, all_embeddings), total=len(ep_ids), desc="saving"):
        task_name, idx = ep_id.split("/", 1)
        cap_dir = data_root / task_name / "captions"
        cap_dir.mkdir(exist_ok=True)
        torch.save(emb, cap_dir / f"{idx}.pt")
        saved += 1

    # ── Save index file ───────────────────────────────────────────────────────
    index = {ep_id: emb for ep_id, emb in zip(ep_ids, all_embeddings)}
    index_path = data_root.parent / "captions_index.pt"
    torch.save(index, index_path)

    print(f"\nDone.")
    print(f"  Per-episode files: {saved} saved under {{task}}/captions/{{idx}}.pt")
    print(f"  Index file: {index_path}  ({len(index)} entries)")
    print(f"  Embedding dim: {all_embeddings.shape[1]}")

    # ── Quick sanity check ────────────────────────────────────────────────────
    print("\nSanity check (5 samples):")
    for ep_id, caption in zip(ep_ids[:5], captions[:5]):
        print(f"  [{ep_id}]")
        print(f"    caption: {caption[:80]}")
        emb = index[ep_id]
        print(f"    emb norm: {emb.norm().item():.3f}")


if __name__ == "__main__":
    main()
