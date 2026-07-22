# LAM-Side Benchmark

| run | action_r2_full_latent | static_response_rel_median | camera_shift_h_median | camera_shift_v_median | episode_shortcut_leakage | task_shortcut_leakage | action_separation_ratio | rev_hard_margin | rev_nearest_pos_beats_opp | reverse_za_cos_median | reverse_ze_cos_median | effective_rank | top5_eig_frac | active_units_std_gt_0.05 | kl_mean_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kl_ft_l40_full_5k | 0.3311 | 0.3576 | 0.3099 | 0.2859 | nan | -0.04464 | 1.038 | 0.01987 | 0.7606 | -0.2196 | 0.2161 | 13.49 | 0.6197 | 40 | 11.51 |
| ours_a_zero_5k | 0.3327 | 0.03524 | 0.2931 | 0.2858 | nan | -0.04419 | 1.034 | 0.02589 | 0.7606 | -0.2391 | -0.4179 | 13.07 | 0.6305 | 40 | 10.89 |
| ours_b2_zero_action_reverse_5k | 0.3361 | 0.02928 | 0.4073 | 0.4128 | nan | -0.008328 | 1.013 | 0.002597 | 0.7562 | -0.5 | -0.5008 | 13.42 | 0.6282 | 40 | 10.93 |
| ours_c_zero_reverse_hardneg_5k | 0.3205 | 0.03954 | 0.4107 | 0.4137 | nan | 0.01173 | 1.008 | -0.004423 | 0.7573 | -0.4512 | 0.9506 | 15.46 | 0.5884 | 40 | 11.65 |
| ours_r_zero_opposite_5k | 0.331 | 0.04727 | 0.2989 | 0.3018 | nan | -0.03064 | 1.063 | 0.01104 | 0.7327 | -0.04392 | -0.2438 | 14 | 0.6035 | 40 | 11.77 |

Notes:
- Static response is normalized by the RMS norm of ordinary transition latents.
- Camera-shift metrics are relative latent changes after applying the same zero-filled image translation to both frames.
- `episode_shortcut_leakage > 0` means same-episode/different-action pairs are closer than different-episode/same-action pairs.
- `task_shortcut_leakage > 0` means same-task/different-action pairs are closer than different-task/same-action pairs.
- `rev_hard_margin > 0` means same-task opposite verbs are farther than same-verb cross-episode positives.
- Lower `reverse_za_cos_median` means the action subspace is more direction-sensitive; higher `reverse_ze_cos_median` means the env subspace is more time-reversal invariant.
- Per-dim labels are heuristic relative rankings for triage, not claims of fixed physical semantics.
