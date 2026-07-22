# LAM Benchmark

## Headline — standard, non-circular (frozen encoder, external target)
These adjudicate LAM quality. Every metric scores against the true 18-D action
or a decoded rollout, with the encoder frozen — none is an image of a training loss.
NOTE: a loss that consumes a signal loses headline rights on that signal's mirror
(e.g. adding an 18-D regression head demotes `mlp_action_r2` to a diagnostic).

### Probes — action decoding & subspace routing

| run | mlp_action_r2 | mlp_action_r2_za | mlp_action_r2_ze | mlp_action_r2_shuffled | mlp_action_l1 | mlp_action_max_l1 | action_r2_full_latent | action_r2_za_linear | action_r2_ze_linear | labeleff_r2_10pct | labeleff_auc | collapse_flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_lam | 0.09318 | 0.09318 | nan | -0.06192 | 0.3351 | 0.4241 | 0.1244 | 0.1244 | nan | 0.02948 | 0.02821 | True |
| kl_ft_l40_full_5k | 0.308 | 0.33 | 0.1582 | -0.03759 | 0.277 | 0.3763 | 0.3434 | 0.3542 | 0.1485 | 0.2089 | 0.1934 | False |
| contrastive_v3_5k | 0.2674 | 0.2604 | 0.03789 | -0.04906 | 0.3014 | 0.391 | 0.294 | 0.3028 | 0.02808 | 0.1219 | 0.1263 | False |
| ours_a_zero_5k | 0.3257 | 0.3023 | 0.148 | -0.03882 | 0.2724 | 0.3799 | 0.3629 | 0.3744 | 0.1599 | 0.235 | 0.2038 | False |
| ours_b_zero_reverse_5k | 0.3407 | 0.315 | 0.05654 | -0.04861 | 0.2754 | 0.3849 | 0.3702 | 0.3801 | 0.06841 | 0.2089 | 0.1919 | False |
| ours_b2_zero_action_reverse_5k | 0.3267 | 0.3048 | 0.1762 | -0.041 | 0.2784 | 0.3916 | 0.3665 | 0.3675 | 0.1576 | 0.2167 | 0.2016 | False |
| ours_c_zero_reverse_hardneg_5k | 0.3321 | 0.2998 | 0.0966 | -0.05155 | 0.2796 | 0.3763 | 0.3658 | 0.3742 | 0.08229 | 0.2213 | 0.1903 | False |
| ours_r_zero_opposite_5k | 0.3654 | 0.3659 | 0.1676 | -0.0449 | 0.2686 | 0.3666 | 0.4025 | 0.4041 | 0.1655 | 0.2598 | 0.2133 | False |
| ours_r2_opp_l01_5k | 0.3654 | 0.3599 | 0.1584 | -0.04768 | 0.2692 | 0.3655 | 0.402 | 0.4048 | 0.1525 | 0.2492 | 0.2101 | False |

- `mlp_action_r2`: frozen full z_mu -> true 18-D action via a small MLP (villa-X);
  `_za`/`_ze` are the SUBSPACE probes. The routing triple reads: `_ze` = action leaked
  into the env subspace; full − `_za` = action info z_a failed to absorb.
  `mlp_action_r2_shuffled` is a label-permutation control and MUST be ~0.
- `action_r2_*_linear` / `action_r2_full_latent`: linear-probe counterparts.
- `labeleff_*`: RidgeCV probe R2 at 10% labels + area under the log-fraction curve
  (LAPO/Genie). Values from the old fixed-alpha probe were probe variance; re-based.

### Retrieval — cross-episode verb kNN with a PROPER null

| run | action_knn_verb_top1 | action_knn_null_top1_mean | action_knn_null_top1_std | action_knn_top1_z | gtact_knn_verb_top1 | probeact_knn_verb_top1 | action_knn_verb_maj5 | action_knn_null_maj5_mean | action_knn_chance_majority |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_lam | 0.06065 | 0.03993 | 0.007307 | 2.836 | 0.02166 | 0.03682 | 0.007942 | 0.004585 | 0.07292 |
| kl_ft_l40_full_5k | 0.04188 | 0.03592 | 0.005803 | 1.027 | 0.02166 | 0.04043 | 0.004332 | 0.003682 | 0.07292 |
| contrastive_v3_5k | 0.07365 | 0.0352 | 0.009427 | 4.078 | 0.02166 | 0.03682 | 0.008664 | 0.00639 | 0.07292 |
| ours_a_zero_5k | 0.03755 | 0.03679 | 0.007554 | 0.1004 | 0.02166 | 0.02671 | 0.002166 | 0.003357 | 0.07292 |
| ours_b_zero_reverse_5k | 0.03249 | 0.03509 | 0.008514 | -0.3053 | 0.02166 | 0.02744 | 0.002166 | 0.002996 | 0.07292 |
| ours_b2_zero_action_reverse_5k | 0.03466 | 0.03578 | 0.007898 | -0.1417 | 0.02166 | 0.02816 | 0.001444 | 0.00343 | 0.07292 |
| ours_c_zero_reverse_hardneg_5k | 0.03321 | 0.03614 | 0.006251 | -0.4678 | 0.02166 | 0.02599 | 0.000722 | 0.003032 | 0.07292 |
| ours_r_zero_opposite_5k | 0.04188 | 0.03776 | 0.005142 | 0.8004 | 0.02166 | 0.03827 | 0.005776 | 0.004621 | 0.07292 |
| ours_r2_opp_l01_5k | 0.04549 | 0.037 | 0.006457 | 1.314 | 0.02166 | 0.03538 | 0.008664 | 0.005487 | 0.07292 |

- `action_knn_verb_top1`: cross-episode kNN verb agreement on raw z_a (zero trained
  readout). `action_knn_null_top1_mean/std` is an episode-level label-permutation null
  (the correct one for retrieval agreement); read `action_knn_top1_z` — |z| < 2 is
  chance-level. `action_knn_chance_majority` (the old 'chance') is a majority-class
  CLASSIFIER baseline, kept only for reference — do not compare retrieval against it.
- `gtact_knn_verb_top1`: the same kNN on the TRUE 18-D action (standardized) — the
  ceiling. If this is low, verb-kNN is a category error for a continuous action code.
- `probeact_knn_verb_top1`: kNN in the out-of-fold ridge-predicted action space. High
  here + low on raw z_a = info present but variance-misallocated (metric problem);
  low here too = action info genuinely missing at linear readout.

### Interventions & opposite actions — decoder-side, external

| run | opp_cls_bal_acc | opp_cls_bal_acc_static | opp_cls_direction_gain | opp_cls_bal_acc_shuffled | opp_cls_n_tasks | ctrl_delta_psnr | zeroact_motion_suppression | zeroact_still_margin | swap_opp_delta_psnr | swap_context_cost_psnr | swap_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_lam | 0.7485 | 0.9598 | -0.2113 | 0.469 | 28 | 6.551 | 1.367 | 4.702 | 0.261 | 5.598 | 101 |
| kl_ft_l40_full_5k | 0.7158 | 0.9643 | -0.2485 | 0.5041 | 28 | 1.166 | 0.6395 | 6.181 | 0.3842 | 1.008 | 101 |
| contrastive_v3_5k | 0.7753 | 0.9717 | -0.1964 | 0.4692 | 28 | 1.81 | -0.1707 | 5.615 | 0.3784 | 1.473 | 101 |
| ours_a_zero_5k | 0.7083 | 0.9747 | -0.2664 | 0.4991 | 28 | 1.159 | 0.94 | 6.571 | 0.4112 | 0.9414 | 101 |
| ours_b_zero_reverse_5k | 0.6845 | 0.9673 | -0.2827 | 0.495 | 28 | 1.207 | 1.31 | 6.604 | 0.3892 | 1.064 | 101 |
| ours_b2_zero_action_reverse_5k | 0.6845 | 0.9777 | -0.2932 | 0.4967 | 28 | 1.184 | 0.8891 | 6.628 | 0.369 | 0.9328 | 101 |
| ours_c_zero_reverse_hardneg_5k | 0.6756 | 0.9851 | -0.3095 | 0.5056 | 28 | 1.473 | 1.259 | 6.894 | 0.3045 | 1.17 | 101 |
| ours_r_zero_opposite_5k | 0.6979 | 0.9702 | -0.2723 | 0.5027 | 28 | 1.174 | 1.096 | 6.803 | 0.2786 | 0.9412 | 101 |
| ours_r2_opp_l01_5k | 0.7128 | 0.9628 | -0.25 | 0.4879 | 28 | 1.208 | 1.153 | 6.849 | 0.2624 | 0.9225 | 101 |

- `opp_cls_bal_acc`: within-task frozen logistic probe classifying the two reversible
  verbs (chance 0.5), on a TASK-STRATIFIED sample (up to 40 clips per verb per
  reversible task; the uniform sample yields n_tasks≈1 = statistically void).
  `_static` is the SAME probe on E(o_t,o_t) latents — the initial-state appearance
  confound; only `opp_cls_direction_gain` (= bal_acc − static) certifies direction
  encoding. `_shuffled` MUST sit at ~0.5.
- `ctrl_delta_psnr`: Genie Δt PSNR via the LAM's own decoder (z_a shuffled across batch).
- `zeroact_motion_suppression` / `zeroact_still_margin`: do(z_a=0) decode (CD-LAM do(u=0)
  analogue). >0 = zeroing z_a pulls the prediction toward staying still — the external
  version of the loss-shaped `static_response`.
- `swap_opp_delta_psnr`: same-task same-verb vs OPPOSITE-verb z_a swap decode, PSNR vs the
  true next frame (scene mismatch cancels in the delta). >0 = opposite actions decode to
  measurably different futures — the fair external adjudicator for reverse-family losses.

## Diagnostics — loss-shaped (explain WHY; NOT for ranking)
These largely mirror training losses (static↔L_zero, rev↔L_reverse, shortcut↔hardneg), so a
model improving on them may just be optimizing its own objective. Read as failure-mode triage.

| run | static_response_rel_median | camera_shift_h_median | camera_shift_v_median | episode_shortcut_leakage | task_shortcut_leakage | rev_hard_margin | rev_nearest_pos_beats_opp | reverse_za_cos_median | reverse_ze_cos_median | effective_rank | top5_eig_frac | active_units_std_gt_0.05 | kl_mean_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_lam | 0.6723 | 0.3797 | 0.3426 | nan | -0.01524 | 0.01006 | 0.7924 | -0.06105 | nan | 26.07 | 0.3498 | 32 | 34.02 |
| kl_ft_l40_full_5k | 0.3564 | 0.3103 | 0.2859 | nan | -0.05347 | 0.05197 | 0.7778 | -0.2249 | 0.2158 | 13.4 | 0.6248 | 40 | 11.56 |
| contrastive_v3_5k | 0.4726 | 0.5556 | 0.4281 | nan | 0.03901 | -0.03494 | 0.7193 | 0.2046 | 0.9917 | 15.55 | 0.5845 | 40 | 52.85 |
| ours_a_zero_5k | 0.03515 | 0.2952 | 0.2884 | nan | -0.05762 | 0.05598 | 0.7836 | -0.2429 | -0.4137 | 13 | 0.6344 | 40 | 11.02 |
| ours_b_zero_reverse_5k | 0.03085 | 0.3548 | 0.375 | nan | -0.01618 | 0.0181 | 0.7749 | -0.5092 | 0.9533 | 13.55 | 0.6255 | 40 | 11.58 |
| ours_b2_zero_action_reverse_5k | 0.02901 | 0.405 | 0.4127 | nan | -0.02439 | 0.02559 | 0.7924 | -0.5 | -0.4943 | 13.25 | 0.6395 | 40 | 11.06 |
| ours_c_zero_reverse_hardneg_5k | 0.03924 | 0.4169 | 0.4133 | nan | -0.00259 | 0.008063 | 0.7778 | -0.4529 | 0.9506 | 15.32 | 0.5916 | 40 | 11.81 |
| ours_r_zero_opposite_5k | 0.04732 | 0.3019 | 0.3038 | nan | -0.04124 | 0.03728 | 0.7924 | -0.05708 | -0.2438 | 14.02 | 0.6085 | 40 | 11.77 |
| ours_r2_opp_l01_5k | 0.04681 | 0.3072 | 0.3131 | nan | -0.03588 | 0.03307 | 0.7924 | 0.001243 | -0.2442 | 14.13 | 0.6089 | 40 | 11.98 |

Diagnostic notes:
- Static response is normalized by the RMS norm of ordinary transition latents.
- Camera-shift metrics are relative latent changes under a zero-filled image translation of both frames.
- `*_shortcut_leakage > 0` means same-scene/different-action pairs are closer than different-scene/same-action.
- `rev_hard_margin > 0` means same-task opposite verbs are farther than same-verb cross-episode positives.
- Lower `reverse_za_cos_median` = more direction-sensitive z_a; higher `reverse_ze_cos_median` = more time-invariant z_e.
