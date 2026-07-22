# lam_cdlam_optimization (Project A)

CD-LAM–based optimization of the DreamDojo latent action: loss, latent structure,
and fine-grained (causal / reversible / opposite-action) action representation.

Studies whether the LAM latent `z_a` can be made **sufficient, cross-scene invariant,
controllable, and groundable** via CD-LAM debiasing, a 32-D action + 8-D environment
split, and auxiliary losses (split-KL, free-bits, L_zero, L_dir, foreground
reconstruction, reversible/opposite-action separation), benchmarked against raw
DreamDojo, CD-LAM, and ConLA-style methods.

- **What to read first:** [`notes/README.md`](notes/README.md) → `notes/concepts_and_pipeline.md`.
- **Full project record:** [`PROJECT_MANIFEST.md`](PROJECT_MANIFEST.md).
- **Lineage:** A → **B** ([`../lam_condition_utilization`](../lam_condition_utilization)) → **C** ([`../lam_world_model_control`](../lam_world_model_control)). B's "availability ≠ utilization" paper grew out of A's rejected direction line.

## Layout
```
code/            shared LAM substrate (train/model/dataset/eval/benchmark_lam/…) + A-specific analysis
code/config/     training/benchmark YAMLs (run resolves configs relative to project root)
notes/           current research notes (index in notes/README.md)
archive/notes/   pre-pivot history (rankings VOID — see legacy_lessons.md)
results/         benchmark & eval outputs; results/lam_benchmark_v2 = authoritative
logs/            run logs
scripts/         run_ours.sh, run_all.sh, run_finetune.sh, run_eval_all.sh, transcode/sam3
data/            dir_pairs_train.npz (L_dir mining output)
checkpoints/     (logical) — physical store is the shared /checkpoints/lam-dis/ (see manifest)
```

## Quickstart (run from this directory)
```bash
# train an arm
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --mixed_precision bf16 \
    code/train.py --config code/config/ft_l40_ours_r2_opp_l01_5k.yaml
# zero-train benchmark
python code/benchmark_lam.py --runs raw_lam kl_ft_l40_full_5k contrastive_v3_5k \
    --n_samples 2000 --out_dir results/lam_benchmark
```

Uses common data at `/home/xuan/embodied-ai/data/egodex/test_240p` and base weights at
`/home/xuan/embodied-ai/checkpoints/pretrained/`. Python env: see `venv-requirements.txt`
(`sam3-requirements.txt` for the SAM3 mask pipeline).
