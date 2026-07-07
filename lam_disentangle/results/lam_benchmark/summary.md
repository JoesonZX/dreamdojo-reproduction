# LAM-Side Benchmark

| run | action_r2_full_latent | static_response_rel_median | effective_rank | top5_eig_frac | active_units_std_gt_0.05 | kl_mean_total | episode_shortcut_leakage | action_separation_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_lam | 0.1198 | 0.6727 | 26.09 | 0.3497 | 32 | 33.87 | nan | 1.062 |
| kl_ft_l40_full_5k | 0.3311 | 0.3576 | 13.49 | 0.6197 | 40 | 11.51 | nan | 1.038 |
| contrastive_v3_5k | 0.2704 | 0.4759 | 15.76 | 0.5767 | 40 | 52.08 | nan | 1.149 |

Notes:
- Static response is normalized by the RMS norm of ordinary transition latents.
- `episode_shortcut_leakage > 0` means same-episode/different-action pairs are closer than different-episode/same-action pairs.
- Per-dim labels are heuristic relative rankings for triage, not claims of fixed physical semantics.
