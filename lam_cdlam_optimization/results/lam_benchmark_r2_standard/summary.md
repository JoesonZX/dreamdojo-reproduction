# LAM Benchmark

## Headline — standard, non-circular (frozen encoder, external target)
These adjudicate LAM quality. Every metric scores against the true 18-D action
or a decoded rollout, with the encoder frozen — none is an image of a training loss.

| run | mlp_action_r2 | mlp_action_l1 | mlp_action_max_l1 | mlp_action_r2_shuffled | action_r2_full_latent | labeleff_r2_10pct | labeleff_auc | action_knn_verb_top1 | action_knn_verb_maj5 | action_knn_chance | ctrl_delta_psnr | ctrl_psnr_inferred | ctrl_psnr_random_action | collapse_flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kl_ft_l40_full_5k | 0.308 | 0.277 | 0.3763 | -0.03759 | 0.3434 | -0.2668 | -0.1897 | 0.04188 | 0.004332 | 0.07292 | 1.22 | 34.7 | 33.48 | False |
| ours_a_zero_5k | 0.3257 | 0.2724 | 0.3799 | -0.03882 | 0.3629 | -0.135 | -0.09922 | 0.03755 | 0.002166 | 0.07292 | 1.22 | 34.71 | 33.49 | False |
| ours_b2_zero_action_reverse_5k | 0.3267 | 0.2784 | 0.3916 | -0.041 | 0.3665 | -0.2041 | -0.09914 | 0.03466 | 0.001444 | 0.07292 | 1.195 | 34.66 | 33.46 | False |
| ours_c_zero_reverse_hardneg_5k | 0.3321 | 0.2796 | 0.3763 | -0.05155 | 0.3658 | -0.08066 | -0.109 | 0.03321 | 0.000722 | 0.07292 | 1.397 | 35 | 33.61 | False |
| ours_r_zero_opposite_5k | 0.3654 | 0.2686 | 0.3666 | -0.0449 | 0.4025 | -0.1341 | -0.0508 | 0.04188 | 0.005776 | 0.07292 | 1.216 | 34.83 | 33.61 | False |
| ours_r2_opp_l01_5k | 0.3654 | 0.2692 | 0.3655 | -0.04768 | 0.402 | -0.1581 | -0.06839 | 0.04549 | 0.008664 | 0.07292 | 1.104 | 34.82 | 33.72 | False |
| ours_r2_opp_l03_5k | 0.3318 | 0.2725 | 0.3705 | -0.05815 | 0.3804 | -0.1875 | -0.1915 | 0.04765 | 0.006498 | 0.07292 | 1.25 | 34.86 | 33.61 | False |
| ours_r2_opp_l05_5k | 0.3281 | 0.2731 | 0.3669 | -0.04607 | 0.3812 | -0.04425 | -0.1187 | 0.04404 | 0.007942 | 0.07292 | 1.235 | 34.82 | 33.58 | False |
| ours_r2_opp_l08_5k | 0.3107 | 0.2763 | 0.3736 | -0.04375 | 0.3336 | -0.1569 | -0.1431 | 0.05343 | 0.009386 | 0.07292 | 1.254 | 34.74 | 33.49 | False |

Headline notes:
- `mlp_action_r2` / `mlp_action_l1` / `mlp_action_max_l1`: frozen z_mu -> true action via a
  small MLP (villa-X). `mlp_action_r2_shuffled` is a label-permutation control and MUST be ~0.
- `action_r2_full_latent`: the linear-probe counterpart (same input, linear readout).
- `labeleff_r2_10pct` / `labeleff_auc`: linear-probe R2 at 10% labels and area under the
  log-fraction curve — the label-efficiency discriminator (LAPO/Genie).
- `action_knn_verb_top1/maj5`: cross-episode kNN verb agreement (DINO/SigLIP, zero trained
  readout). `action_knn_chance` is the majority-verb baseline.
- `ctrl_delta_psnr = ctrl_psnr_inferred − ctrl_psnr_random_action`: Genie Δt PSNR via the LAM's
  own decoder (z_a shuffled across the batch). >0 means z_a genuinely controls the prediction.
- `collapse_flag=True` marks a degenerate latent (probe R2 < 0.1); its 'invariance' is untrustworthy.

## Diagnostics — loss-shaped (explain WHY; NOT for ranking)
These largely mirror training losses (static↔L_zero, rev↔L_reverse, shortcut↔hardneg), so a
model improving on them may just be optimizing its own objective. Read as failure-mode triage.

| run | static_response_rel_median | camera_shift_h_median | camera_shift_v_median | episode_shortcut_leakage | task_shortcut_leakage | rev_hard_margin | rev_nearest_pos_beats_opp | reverse_za_cos_median | reverse_ze_cos_median | effective_rank | top5_eig_frac | active_units_std_gt_0.05 | kl_mean_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kl_ft_l40_full_5k | 0.3564 | 0.3103 | 0.2859 | nan | -0.05347 | 0.05197 | 0.7778 | -0.2249 | 0.2158 | 13.4 | 0.6248 | 40 | 11.56 |
| ours_a_zero_5k | 0.03515 | 0.2952 | 0.2884 | nan | -0.05762 | 0.05598 | 0.7836 | -0.2429 | -0.4137 | 13 | 0.6344 | 40 | 11.02 |
| ours_b2_zero_action_reverse_5k | 0.02901 | 0.405 | 0.4127 | nan | -0.02439 | 0.02559 | 0.7924 | -0.5 | -0.4943 | 13.25 | 0.6395 | 40 | 11.06 |
| ours_c_zero_reverse_hardneg_5k | 0.03924 | 0.4169 | 0.4133 | nan | -0.00259 | 0.008063 | 0.7778 | -0.4529 | 0.9506 | 15.32 | 0.5916 | 40 | 11.81 |
| ours_r_zero_opposite_5k | 0.04732 | 0.3019 | 0.3038 | nan | -0.04124 | 0.03728 | 0.7924 | -0.05708 | -0.2438 | 14.02 | 0.6085 | 40 | 11.77 |
| ours_r2_opp_l01_5k | 0.04681 | 0.3072 | 0.3131 | nan | -0.03588 | 0.03307 | 0.7924 | 0.001243 | -0.2442 | 14.13 | 0.6089 | 40 | 11.98 |
| ours_r2_opp_l03_5k | 0.0554 | 0.3123 | 0.312 | nan | -0.01503 | 0.01141 | 0.769 | 0.2638 | -0.1793 | 14.36 | 0.6066 | 40 | 11.85 |
| ours_r2_opp_l05_5k | 0.06157 | 0.3161 | 0.3022 | nan | -0.006886 | 0.002177 | 0.7895 | 0.405 | -0.06523 | 13.54 | 0.6295 | 40 | 11.7 |
| ours_r2_opp_l08_5k | 0.06912 | 0.342 | 0.2813 | nan | 0.000278 | -0.003933 | 0.7953 | 0.52 | 0.08391 | 12.16 | 0.6622 | 40 | 11.47 |

Diagnostic notes:
- Static response is normalized by the RMS norm of ordinary transition latents.
- Camera-shift metrics are relative latent changes under a zero-filled image translation of both frames.
- `*_shortcut_leakage > 0` means same-scene/different-action pairs are closer than different-scene/same-action.
- `rev_hard_margin > 0` means same-task opposite verbs are farther than same-verb cross-episode positives.
- Lower `reverse_za_cos_median` = more direction-sensitive z_a; higher `reverse_ze_cos_median` = more time-invariant z_e.
