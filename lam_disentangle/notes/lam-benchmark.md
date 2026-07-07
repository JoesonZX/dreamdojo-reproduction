# LAM-side Benchmark Plan

Purpose: evaluate whether a LAM latent is a clean action interface before doing
downstream world-model experiments. The benchmark compares existing checkpoints:

- `raw_lam`: original `LAM_400k.ckpt`, latent 32.
- `kl_ft_l40_full_5k`: KL-bottleneck fine-tune, latent 40 = 32 action + 8 env.
- `contrastive_v3_5k`: verb-SupCon/temporal contrastive fine-tune, latent 40.

## Run

Use the project Python environment:

```bash
cd /home/xuan/embodied-ai/lam_disentangle
/home/xuan/.venv/bin/python code/benchmark_lam.py \
  --n_samples 2000 \
  --perturb_samples 64 \
  --batch_size 16 \
  --num_workers 4 \
  --out_dir results/lam_benchmark
```

Outputs:

- `results/lam_benchmark/summary.md`: compact run-level table.
- `results/lam_benchmark/summary.csv`: same numbers in CSV.
- `results/lam_benchmark/per_dim_<run>.csv`: per-latent-dimension effect table.

## Run-level Metrics

`static_response_rel_median`:
Encodes duplicated-frame pairs `(o_t, o_t)` and reports the median latent norm,
normalized by the RMS norm of ordinary transition latents. Lower means
no-action inputs are closer to the latent origin.

`effective_rank`, `top5_eig_frac`, `active_units_std_gt_0.05`:
Latent health metrics. Effective rank measures how many covariance directions
are used; top-5 fraction reports whether a few directions dominate; active units
counts dimensions with standard deviation above `0.05`.

`kl_mean_total`:
Mean VAE KL across dimensions. This is a capacity/latent-information proxy, not
a content-routing metric.

`episode_shortcut_leakage`:
Compares same-episode/different-action distance against different-episode/
same-action distance in the action subspace. Positive values mean episode
context is pulling latents closer than shared action does.

## Per-dim Table

Columns:

- `std`, `kl_mean`, `active_score`: whether the coordinate is used.
- `action_r2_drop`: drop in Ridge action R2 when that coordinate is ablated at
  test time.
- `single_dim_action_r2`: action R2 from that coordinate alone.
- `decoder_total_effect`: mean absolute output change after perturbing that
  latent coordinate by `2 * std`.
- `task_eta2`, `ep_eta2`, `verb_eta2`: cheap per-dim label leakage scores,
  using variance explained by each label.
- `label`: heuristic triage label, not a claim that a latent dimension has a
  fixed physical meaning.

## Current Limitations

- Per-dim labels are coordinate-based and can change under latent rotations.
  Treat them as effect diagnostics, not semantic names.
- This is a LAM-side benchmark; downstream world-model action following remains
  the final encoder selection criterion.
