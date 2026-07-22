# LAM-Side Benchmark

| run | action_r2_full_latent | static_response_rel_median | camera_shift_h_median | camera_shift_v_median | episode_shortcut_leakage | task_shortcut_leakage | action_separation_ratio | rev_hard_margin | rev_nearest_pos_beats_opp | reverse_za_cos_median | reverse_ze_cos_median | effective_rank | top5_eig_frac | active_units_std_gt_0.05 | kl_mean_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_lam | 0.1198 | 0.6727 | 0.3825 | 0.3423 | nan | -0.01589 | 1.062 | -0.005463 | 0.7595 | -0.06771 | nan | 26.09 | 0.3497 | 32 | 33.87 |
| kl_ft_l40_full_5k | 0.3311 | 0.3576 | 0.3099 | 0.2859 | nan | -0.04464 | 1.038 | 0.01987 | 0.7606 | -0.2196 | 0.2161 | 13.49 | 0.6197 | 40 | 11.51 |
| contrastive_v3_5k | 0.2704 | 0.4759 | 0.5553 | 0.4282 | nan | 0.03751 | 1.149 | -0.04603 | 0.7304 | 0.2028 | 0.9918 | 15.76 | 0.5767 | 40 | 52.08 |
| ours_a_zero_5k | 0.3327 | 0.03524 | 0.2931 | 0.2858 | nan | -0.04419 | 1.034 | 0.02589 | 0.7606 | -0.2391 | -0.4179 | 13.07 | 0.6305 | 40 | 10.89 |
| ours_b_zero_reverse_5k | 0.3386 | 0.03104 | 0.3543 | 0.371 | nan | -0.001233 | 1.012 | -0.001707 | 0.7271 | -0.5084 | 0.9541 | 13.64 | 0.6213 | 40 | 11.41 |
| ours_c_zero_reverse_hardneg_5k | 0.3205 | 0.03954 | 0.4107 | 0.4137 | nan | 0.01173 | 1.008 | -0.004423 | 0.7573 | -0.4512 | 0.9506 | 15.46 | 0.5884 | 40 | 11.65 |

Notes:
- Static response is normalized by the RMS norm of ordinary transition latents.
- Camera-shift metrics are relative latent changes after applying the same zero-filled image translation to both frames.
- `episode_shortcut_leakage > 0` means same-episode/different-action pairs are closer than different-episode/same-action pairs.
- `task_shortcut_leakage > 0` means same-task/different-action pairs are closer than different-task/same-action pairs.
- `rev_hard_margin > 0` means same-task opposite verbs are farther than same-verb cross-episode positives.
- Lower `reverse_za_cos_median` means the action subspace is more direction-sensitive; higher `reverse_ze_cos_median` means the env subspace is more time-reversal invariant.
- Per-dim labels are heuristic relative rankings for triage, not claims of fixed physical semantics.
