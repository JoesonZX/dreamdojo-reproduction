# Seed variance (stage0 T8, 2026-07-19)

Three headline families x {seed42, s1, s2}. `training.seed` varies; `data.seed=42`
is pinned so every seed sees the identical val split.

**All nine rows were benchmarked with the SAME code version (today).** seed42 was
re-run into `results/lam_benchmark_seed42` rather than reused from
`results/lam_benchmark_v2`: `_mlp_action_probe` postdates the last commit, so its
state at the v2 run (2026-07-15) cannot be verified from git, and mixing code
versions across a variance table would confound seed noise with code drift.

Regime caveats (conservative — they inflate the range, never shrink it):
`ours_a s1` trained in a 1-GPU regime; `ours_r2 s1/s2` on a different machine.
n=3 per family, so `range` is a coarse noise gate, not a significance test.

## mlp_action_r2

| family | seed42 | s1 | s2 | mean | range |
|---|---|---|---|---|---|
| kl_full | 0.3080 | 0.3402 | 0.3625 | 0.3369 | 0.0545 |
| ours_a | 0.3257 | 0.3555 [1gpu] | 0.3389 | 0.3400 | 0.0297 |
| ours_r2 | 0.3654 | 0.3291 [xmachine] | 0.3641 [xmachine] | 0.3529 | 0.0362 |

## opp_cls_direction_gain

| family | seed42 | s1 | s2 | mean | range |
|---|---|---|---|---|---|
| kl_full | -0.2485 | -0.2619 | -0.2723 | -0.2609 | 0.0238 |
| ours_a | -0.2664 | -0.2664 [1gpu] | -0.2827 | -0.2718 | 0.0164 |
| ours_r2 | -0.2500 | -0.2411 [xmachine] | -0.2411 [xmachine] | -0.2440 | 0.0089 |

## swap_opp_delta_psnr

| family | seed42 | s1 | s2 | mean | range |
|---|---|---|---|---|---|
| kl_full | 0.3842 | 0.4405 | 0.3034 | 0.3760 | 0.1371 |
| ours_a | 0.4112 | 0.3490 [1gpu] | 0.2885 | 0.3496 | 0.1227 |
| ours_r2 | 0.2624 | 0.3400 [xmachine] | 0.4295 [xmachine] | 0.3440 | 0.1671 |

## mlp_action_r2_za (exploratory)

| family | seed42 | s1 | s2 | mean | range |
|---|---|---|---|---|---|
| kl_full | 0.3300 | 0.3219 | 0.3618 | 0.3379 | 0.0400 |
| ours_a | 0.3023 | 0.3169 [1gpu] | 0.3549 | 0.3247 | 0.0526 |
| ours_r2 | 0.3599 | 0.3602 [xmachine] | 0.3814 [xmachine] | 0.3672 | 0.0215 |

## static_response_rel_median (exploratory)

| family | seed42 | s1 | s2 | mean | range |
|---|---|---|---|---|---|
| kl_full | 0.3564 | 0.4466 | 0.4352 | 0.4127 | 0.0902 |
| ours_a | 0.0351 | 0.0263 [1gpu] | 0.1018 | 0.0544 | 0.0754 |
| ours_r2 | 0.0468 | 0.1249 [xmachine] | 0.1968 [xmachine] | 0.1228 | 0.1499 |

## scene_knn_ep_top1 (exploratory)

| family | seed42 | s1 | s2 | mean | range |
|---|---|---|---|---|---|
| kl_full | 0.1587 | 0.1940 | 0.1900 | 0.1809 | 0.0353 |
| ours_a | 0.1173 | 0.1400 [1gpu] | 0.1340 | 0.1304 | 0.0227 |
| ours_r2 | 0.1760 | 0.1847 [xmachine] | 0.1807 [xmachine] | 0.1804 | 0.0087 |

## Verdicts

- **mlp_action_r2**: ours_r2 0.3529 vs kl_full 0.3369 -> advantage **+0.0160**; largest family range 0.0545 -> **WITHIN noise (NOT a defensible claim)**.
- **opp_cls_direction_gain**: ours_r2 -0.2440 vs kl_full -0.2609 -> advantage **+0.0169**; largest family range 0.0238 -> **WITHIN noise (NOT a defensible claim)**.
- **swap_opp_delta_psnr**: ours_r2 0.3440 vs kl_full 0.3760 -> advantage **-0.0321**; largest family range 0.1671 -> **WITHIN noise (NOT a defensible claim)**.

### Reproducibility check vs benchmark v2 (same checkpoints, older code)

| run | v2 mlp_action_r2 | today | delta |
|---|---|---|---|
| kl_ft_l40_full_5k | 0.3080 | 0.3080 | +0.0000 |
| ours_a_zero_5k | 0.3257 | 0.3257 | +0.0000 |
| ours_r2_opp_l01_5k | 0.3654 | 0.3654 | +0.0000 |

A non-trivial delta here means the probe changed between v2 and now — in that case
the v2 summary table must not be mixed with this one in any comparison.

### Discipline reminder

Every family carries `lambda_action=1.0`, so `mlp_action_r2` is quasi-circular:
the family-internal controlled comparison is fair, but the number is not an absolute
external claim and must not be compared across families that lack the action head
(raw_lam, cdlam_official, the lf twins). `opp_cls_direction_gain` and
`swap_opp_delta_psnr` are the external counterparts.
