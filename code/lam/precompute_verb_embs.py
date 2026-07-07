"""
Precompute SigLIP text embeddings for verb phrases extracted from EgoDex episodes.

Reads llm_verbs from each episode's HDF5 file, encodes with the same SigLIP
text encoder used for captions, and saves:
  - Per-episode: {task_name}/verb_embs/{idx}.pt  — float32 [768]

The verb embedding is used as the soft label matrix in siglip_loss, so that
two episodes sharing the same action verb get labels close to 1.0 regardless
of differing object/scene descriptions in their full captions.

Usage:
  CUDA_VISIBLE_DEVICES=0 python precompute_verb_embs.py \
      --data_root /home/xuan/embodied-ai/data/egodex/test
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
    return tokenizer, model


@torch.no_grad()
def encode_texts(texts: list, tokenizer, model, device: torch.device, max_length: int = 32) -> torch.Tensor:
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
    parser.add_argument("--dry_run",    action="store_true")
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

    print(f"Found {len(hdf5_files)} episodes")

    # ── Collect verb phrases ──────────────────────────────────────────────────
    ep_ids = []
    verb_texts = []
    missing = 0
    for p in tqdm(hdf5_files, desc="reading hdf5"):
        task_id = p.parent.name
        ep_id   = f"{task_id}/{p.stem}"
        verb = None
        try:
            with h5py.File(p, "r") as f:
                verbs = list(f.attrs.get("llm_verbs", []))
                if verbs:
                    verb = verbs[0]
        except Exception:
            pass
        if verb is None:
            # Fall back to task name as a coarse verb proxy
            verb = task_id.replace("_", " ")
            missing += 1
        ep_ids.append(ep_id)
        verb_texts.append(str(verb))

    print(f"  {len(verb_texts)} verbs collected, {missing} used task-name fallback")

    # ── Encode ────────────────────────────────────────────────────────────────
    tokenizer, model = load_siglip_text_encoder(args.model_name, device)

    all_embeddings = []
    for i in tqdm(range(0, len(verb_texts), args.batch_size), desc="encoding"):
        batch = verb_texts[i : i + args.batch_size]
        emb = encode_texts(batch, tokenizer, model, device)
        all_embeddings.append(emb)

    all_embeddings = torch.cat(all_embeddings, dim=0)  # [N, 768]
    print(f"Embeddings shape: {all_embeddings.shape}")

    # ── Save per-episode files ────────────────────────────────────────────────
    saved = 0
    for ep_id, emb in tqdm(zip(ep_ids, all_embeddings), total=len(ep_ids), desc="saving"):
        task_name, idx = ep_id.split("/", 1)
        verb_dir = data_root / task_name / "verb_embs"
        verb_dir.mkdir(exist_ok=True)
        torch.save(emb, verb_dir / f"{idx}.pt")
        saved += 1

    print(f"\nDone. {saved} verb embeddings saved under {{task}}/verb_embs/{{idx}}.pt")

    print("\nSanity check (5 samples):")
    for ep_id, verb in zip(ep_ids[:5], verb_texts[:5]):
        print(f"  [{ep_id}]  verb: '{verb}'")


if __name__ == "__main__":
    main()
