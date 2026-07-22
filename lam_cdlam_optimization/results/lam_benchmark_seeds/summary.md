# LAM Benchmark

## Headline — standard, non-circular (frozen encoder, external target)
These adjudicate LAM quality. Every metric scores against the true 18-D action
or a decoded rollout, with the encoder frozen — none is an image of a training loss.
NOTE: a loss that consumes a signal loses headline rights on that signal's mirror
(e.g. adding an 18-D regression head demotes `mlp_action_r2` to a diagnostic).

### Probes — action decoding & subspace routing

| run | mlp_action_r2 | mlp_action_r2_za | mlp_action_r2_ze | mlp_action_r2_shuffled | mlp_action_l1 | mlp_action_max_l1 | action_r2_full_latent | action_r2_za_linear | action_r2_ze_linear | action_r2_za_taskheld | action_r2_taskheld_gap | action_r2_trans | action_r2_rot | action_r2_left | action_r2_right | labeleff_r2_10pct | labeleff_auc | collapse_flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kl_ft_l40_full_5k_s1 | 0.3402 | 0.3219 | 0.1812 | -0.03615 | 0.2744 | 0.3803 | 0.3526 | 0.3482 | 0.1568 | 0.3042 | 0.04403 | 0.5317 | 0.2565 | 0.3273 | 0.3692 | 0.2054 | 0.2039 | False |
| kl_ft_l40_full_5k_s2 | 0.3625 | 0.3618 | 0.1437 | -0.05737 | 0.2714 | 0.3769 | 0.3765 | 0.3875 | 0.146 | 0.306 | 0.08145 | 0.5176 | 0.3224 | 0.3869 | 0.3881 | 0.2401 | 0.1982 | False |
| ours_a_zero_5k_s1 | 0.3555 | 0.3169 | 0.2182 | -0.03832 | 0.268 | 0.3727 | 0.3917 | 0.3916 | 0.1481 | 0.3246 | 0.06703 | 0.5701 | 0.3024 | 0.3831 | 0.4002 | 0.2716 | 0.2145 | False |
| ours_a_zero_5k_s2 | 0.3389 | 0.3549 | 0.1904 | -0.05573 | 0.2731 | 0.3808 | 0.365 | 0.3749 | 0.1652 | 0.3176 | 0.05721 | 0.5107 | 0.3069 | 0.3602 | 0.3896 | 0.2312 | 0.1866 | False |
| ours_r2_opp_l01_5k_s1 | 0.3291 | 0.3602 | 0.1741 | -0.03694 | 0.2752 | 0.3802 | 0.3677 | 0.3724 | 0.1249 | 0.301 | 0.0714 | 0.5296 | 0.2938 | 0.3499 | 0.3949 | 0.2794 | 0.1811 | False |
| ours_r2_opp_l01_5k_s2 | 0.3641 | 0.3814 | 0.1969 | -0.04709 | 0.2692 | 0.3559 | 0.3786 | 0.3891 | 0.1797 | 0.3264 | 0.06271 | 0.5543 | 0.3065 | 0.3651 | 0.4131 | 0.2777 | 0.2093 | False |

- `mlp_action_r2`: frozen full z_mu -> true 18-D action via a small MLP (villa-X);
  `_za`/`_ze` are the SUBSPACE probes. The routing triple reads: `_ze` = action leaked
  into the env subspace; full − `_za` = action info z_a failed to absorb.
  `mlp_action_r2_shuffled` is a label-permutation control and MUST be ~0.
- `action_r2_*_linear` / `action_r2_full_latent`: linear-probe counterparts.
- `action_r2_za_taskheld`: linear probe with HELD-OUT TASKS (GroupKFold) — external
  invariance measure; `_taskheld_gap` (shuffled − heldout) = context dependence of z_a.
- `action_r2_trans/rot/left/right`: per-target-group R2 — tracks the
  rotation-decodes-worse failure mode the fine-grained-control claim depends on.
- `labeleff_*`: RidgeCV probe R2 at 10% labels + area under the log-fraction curve
  (LAPO/Genie). Values from the old fixed-alpha probe were probe variance; re-based.

### Retrieval — cross-episode verb kNN with a PROPER null

| run | action_knn_verb_top1 | action_knn_null_top1_mean | action_knn_null_top1_std | action_knn_top1_z | gtact_knn_verb_top1 | probeact_knn_verb_top1 | action_knn_verb_maj5 | action_knn_null_maj5_mean | action_knn_chance_majority |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kl_ft_l40_full_5k_s1 | 0.03682 | 0.0383 | 0.007629 | -0.194 | 0.02166 | 0.03394 | 0.00361 | 0.004657 | 0.07292 |
| kl_ft_l40_full_5k_s2 | 0.04404 | 0.03588 | 0.008902 | 0.9165 | 0.02166 | 0.03538 | 0.001444 | 0.004549 | 0.07292 |
| ours_a_zero_5k_s1 | 0.03899 | 0.04014 | 0.008587 | -0.1345 | 0.02166 | 0.03177 | 0.006498 | 0.004477 | 0.07292 |
| ours_a_zero_5k_s2 | 0.04621 | 0.03711 | 0.005332 | 1.706 | 0.02166 | 0.04404 | 0.002166 | 0.003935 | 0.07292 |
| ours_r2_opp_l01_5k_s1 | 0.05776 | 0.03917 | 0.007493 | 2.481 | 0.02166 | 0.04332 | 0.00722 | 0.004116 | 0.07292 |
| ours_r2_opp_l01_5k_s2 | 0.04621 | 0.0357 | 0.007463 | 1.408 | 0.02166 | 0.05126 | 0.007942 | 0.003971 | 0.07292 |

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
| kl_ft_l40_full_5k_s1 | 0.7187 | 0.9807 | -0.2619 | 0.4863 | 28 | 0.8357 | 0.09162 | 4.883 | 0.4405 | 0.5153 | 101 |
| kl_ft_l40_full_5k_s2 | 0.7128 | 0.9851 | -0.2723 | 0.4874 | 28 | 0.7471 | 0.7714 | 6.832 | 0.3034 | 0.5071 | 101 |
| ours_a_zero_5k_s1 | 0.7083 | 0.9747 | -0.2664 | 0.5086 | 28 | 1.166 | 0.7836 | 6.577 | 0.349 | 0.9233 | 101 |
| ours_a_zero_5k_s2 | 0.6964 | 0.9792 | -0.2827 | 0.5022 | 28 | 0.9938 | 1.081 | 7.083 | 0.2885 | 0.7451 | 101 |
| ours_r2_opp_l01_5k_s1 | 0.7321 | 0.9732 | -0.2411 | 0.4993 | 28 | 0.7651 | 0.4144 | 6.945 | 0.34 | 0.6148 | 101 |
| ours_r2_opp_l01_5k_s2 | 0.7351 | 0.9762 | -0.2411 | 0.4822 | 28 | 1.316 | 1.03 | 6.768 | 0.4295 | 0.98 | 101 |

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

| run | static_response_rel_median | camera_shift_h_median | camera_shift_v_median | id_ratio_ep_top1 | id_ratio_ep_null | id_ratio_task_top1 | id_ratio_task_null | episode_shortcut_leakage | task_shortcut_leakage | rev_hard_margin | rev_nearest_pos_beats_opp | reverse_za_cos_median | reverse_ze_cos_median | effective_rank | top5_eig_frac | active_units_std_gt_0.05 | kl_mean_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kl_ft_l40_full_5k_s1 | 0.4466 | 0.2315 | 0.2396 | 0.194 | 0.00794 | 0.2233 | 0.01427 | nan | -0.07828 | 0.07581 | 0.7953 | 0.01872 | 0.07655 | 13.31 | 0.6518 | 40 | 11.23 |
| kl_ft_l40_full_5k_s2 | 0.4352 | 0.2544 | 0.2373 | 0.19 | 0.00794 | 0.22 | 0.01427 | nan | -0.07872 | 0.07347 | 0.8158 | 0.04393 | 0.06102 | 13.68 | 0.6299 | 40 | 10.39 |
| ours_a_zero_5k_s1 | 0.02634 | 0.2856 | 0.2935 | 0.14 | 0.00794 | 0.172 | 0.01427 | nan | -0.06295 | 0.06431 | 0.8012 | -0.2098 | -0.4037 | 13.26 | 0.6331 | 40 | 12.15 |
| ours_a_zero_5k_s2 | 0.1018 | 0.2945 | 0.2968 | 0.134 | 0.00794 | 0.164 | 0.01427 | nan | -0.06867 | 0.06948 | 0.8129 | -0.1623 | -0.1349 | 12.84 | 0.6459 | 40 | 11.1 |
| ours_r2_opp_l01_5k_s1 | 0.1249 | 0.313 | 0.316 | 0.1847 | 0.00794 | 0.218 | 0.01427 | nan | -0.0338 | 0.03167 | 0.7895 | 0.0937 | -0.262 | 14.33 | 0.6047 | 40 | 12.39 |
| ours_r2_opp_l01_5k_s2 | 0.1968 | 0.2441 | 0.2521 | 0.1807 | 0.00794 | 0.2133 | 0.01427 | nan | -0.03984 | 0.03291 | 0.7895 | 0.2851 | -0.0657 | 13.21 | 0.6425 | 40 | 12.98 |

Diagnostic notes:
- Static response is normalized by the RMS norm of ordinary transition latents.
- `id_ratio_*_top1`: CD-LAM-style identity-retrieval confound — fraction of cosine
  nearest neighbours in z_a sharing the query's episode/task (self excluded).
  Compare against `*_null` (random-retrieval expectation), not zero. High = z_a
  retrieves 'same scene' over 'same action' (CD-LAM reports 0.527->0.047).
- Camera-shift metrics are relative latent changes under a zero-filled image translation of both frames.
- `*_shortcut_leakage > 0` means same-scene/different-action pairs are closer than different-scene/same-action.
- `rev_hard_margin > 0` means same-task opposite verbs are farther than same-verb cross-episode positives.
- Lower `reverse_za_cos_median` = more direction-sensitive z_a; higher `reverse_ze_cos_median` = more time-invariant z_e.
