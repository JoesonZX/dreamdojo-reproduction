# LAM Benchmark

## Headline — standard, non-circular (frozen encoder, external target)
These adjudicate LAM quality. Every metric scores against the true 18-D action
or a decoded rollout, with the encoder frozen — none is an image of a training loss.
NOTE: a loss that consumes a signal loses headline rights on that signal's mirror
(e.g. adding an 18-D regression head demotes `mlp_action_r2` to a diagnostic).

### Probes — action decoding & subspace routing

| run | mlp_action_r2 | mlp_action_r2_za | mlp_action_r2_ze | mlp_action_r2_shuffled | mlp_action_l1 | mlp_action_max_l1 | action_r2_full_latent | action_r2_za_linear | action_r2_ze_linear | action_r2_za_taskheld | action_r2_taskheld_gap | action_r2_trans | action_r2_rot | action_r2_left | action_r2_right | labeleff_r2_10pct | labeleff_auc | collapse_flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cdlam_official | 0.104 | 0.104 | nan | -0.06708 | 0.3332 | 0.4154 | 0.1266 | 0.1266 | nan | 0.06487 | 0.06168 | 0.225 | 0.07733 | 0.09531 | 0.1578 | 0.02703 | 0.02198 | False |

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
| cdlam_official | 0.05632 | 0.03968 | 0.006047 | 2.752 | 0.02166 | 0.04043 | 0.006498 | 0.005126 | 0.07292 |

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
| cdlam_official | 0.7366 | 0.9777 | -0.2411 | 0.4547 | 28 | 4.374 | 5.037 | 4.709 | 0.2999 | 3.374 | 101 |

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
| cdlam_official | 0.06671 | 0.4421 | 0.388 | 0.184 | 0.00794 | 0.218 | 0.01427 | nan | 0.01554 | -0.01413 | 0.7895 | -0.3339 | nan | 24.87 | 0.3729 | 32 | 8.868 |

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
