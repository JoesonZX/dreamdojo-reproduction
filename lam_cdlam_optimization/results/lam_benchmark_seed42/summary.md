# LAM Benchmark

## Headline — standard, non-circular (frozen encoder, external target)
These adjudicate LAM quality. Every metric scores against the true 18-D action
or a decoded rollout, with the encoder frozen — none is an image of a training loss.
NOTE: a loss that consumes a signal loses headline rights on that signal's mirror
(e.g. adding an 18-D regression head demotes `mlp_action_r2` to a diagnostic).

### Probes — action decoding & subspace routing

| run | mlp_action_r2 | mlp_action_r2_za | mlp_action_r2_ze | mlp_action_r2_shuffled | mlp_action_l1 | mlp_action_max_l1 | action_r2_full_latent | action_r2_za_linear | action_r2_ze_linear | action_r2_za_taskheld | action_r2_taskheld_gap | action_r2_trans | action_r2_rot | action_r2_left | action_r2_right | labeleff_r2_10pct | labeleff_auc | collapse_flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_lam | 0.09318 | 0.09318 | nan | -0.06192 | 0.3351 | 0.4241 | 0.1244 | 0.1244 | nan | 0.06679 | 0.05764 | 0.2066 | 0.08333 | 0.09911 | 0.1498 | 0.02948 | 0.02821 | True |
| kl_ft_l40_full_5k | 0.308 | 0.33 | 0.1582 | -0.03759 | 0.277 | 0.3763 | 0.3434 | 0.3542 | 0.1485 | 0.3 | 0.05414 | 0.5242 | 0.2692 | 0.345 | 0.3634 | 0.2089 | 0.1934 | False |
| ours_a_zero_5k | 0.3257 | 0.3023 | 0.148 | -0.03882 | 0.2724 | 0.3799 | 0.3629 | 0.3744 | 0.1599 | 0.3089 | 0.06547 | 0.5418 | 0.2907 | 0.3717 | 0.3771 | 0.235 | 0.2038 | False |
| ours_r2_opp_l01_5k | 0.3654 | 0.3599 | 0.1584 | -0.04768 | 0.2692 | 0.3655 | 0.402 | 0.4048 | 0.1525 | 0.3237 | 0.08116 | 0.5529 | 0.3308 | 0.3976 | 0.412 | 0.2492 | 0.2101 | False |

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
| raw_lam | 0.06065 | 0.03993 | 0.007307 | 2.836 | 0.02166 | 0.03682 | 0.007942 | 0.004585 | 0.07292 |
| kl_ft_l40_full_5k | 0.04188 | 0.03592 | 0.005803 | 1.027 | 0.02166 | 0.04043 | 0.004332 | 0.003682 | 0.07292 |
| ours_a_zero_5k | 0.03755 | 0.03679 | 0.007554 | 0.1004 | 0.02166 | 0.02671 | 0.002166 | 0.003357 | 0.07292 |
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
| raw_lam | 0.7485 | 0.9598 | -0.2113 | 0.469 | 28 | 6.599 | 1.367 | 4.702 | 0.261 | 5.598 | 101 |
| kl_ft_l40_full_5k | 0.7158 | 0.9643 | -0.2485 | 0.5041 | 28 | 1.232 | 0.6395 | 6.181 | 0.3842 | 1.008 | 101 |
| ours_a_zero_5k | 0.7083 | 0.9747 | -0.2664 | 0.4991 | 28 | 1.204 | 0.94 | 6.571 | 0.4112 | 0.9414 | 101 |
| ours_r2_opp_l01_5k | 0.7128 | 0.9628 | -0.25 | 0.4879 | 28 | 1.192 | 1.153 | 6.849 | 0.2624 | 0.9225 | 101 |

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

| run | static_response_rel_median | camera_shift_h_median | camera_shift_v_median | scene_knn_ep_top1 | scene_knn_ep_null | scene_knn_task_top1 | scene_knn_task_null | episode_shortcut_leakage | task_shortcut_leakage | rev_hard_margin | rev_nearest_pos_beats_opp | reverse_za_cos_median | reverse_ze_cos_median | effective_rank | top5_eig_frac | active_units_std_gt_0.05 | kl_mean_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_lam | 0.6723 | 0.3797 | 0.3426 | 0.1827 | 0.00794 | 0.2187 | 0.01427 | nan | -0.01524 | 0.01006 | 0.7924 | -0.06105 | nan | 26.07 | 0.3498 | 32 | 34.02 |
| kl_ft_l40_full_5k | 0.3564 | 0.3103 | 0.2859 | 0.1587 | 0.00794 | 0.1887 | 0.01427 | nan | -0.05347 | 0.05197 | 0.7778 | -0.2249 | 0.2158 | 13.4 | 0.6248 | 40 | 11.56 |
| ours_a_zero_5k | 0.03515 | 0.2952 | 0.2884 | 0.1173 | 0.00794 | 0.1447 | 0.01427 | nan | -0.05762 | 0.05598 | 0.7836 | -0.2429 | -0.4137 | 13 | 0.6344 | 40 | 11.02 |
| ours_r2_opp_l01_5k | 0.04681 | 0.3072 | 0.3131 | 0.176 | 0.00794 | 0.2147 | 0.01427 | nan | -0.03588 | 0.03307 | 0.7924 | 0.001243 | -0.2442 | 14.13 | 0.6089 | 40 | 11.98 |

Diagnostic notes:
- Static response is normalized by the RMS norm of ordinary transition latents.
- `scene_knn_*_top1`: scene/identity retrieval confound — fraction of cosine
  nearest neighbours in z_a sharing the query's episode/task (self excluded).
  Compare against `*_null` (random-retrieval expectation), not zero. High = z_a
  retrieves 'same scene' over 'same action'. NOT CD-LAM's id_ratio: that is
  their zero-transition response = our `static_response_rel_median`.
- Camera-shift metrics are relative latent changes under a zero-filled image translation of both frames.
- `*_shortcut_leakage > 0` means same-scene/different-action pairs are closer than different-scene/same-action.
- `rev_hard_margin > 0` means same-task opposite verbs are farther than same-verb cross-episode positives.
- Lower `reverse_za_cos_median` = more direction-sensitive z_a; higher `reverse_ze_cos_median` = more time-invariant z_e.
