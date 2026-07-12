# Config Index

The configs are grouped by experiment family. Paths are kept flat so existing
scripts do not need to change.

## From-Scratch Baselines

- `baseline.yaml`: 60M no-action baseline.
- `exp1_action.yaml`: add 18D action prediction.
- `exp2_split.yaml`: split latent into action + env subspaces.
- `exp3_indep.yaml`, `exp3_fix_a.yaml`, `exp3_fix_b.yaml`: decorrelation/independence variants.
- `exp4_kl_*.yaml`: KL bottleneck sweeps and annealing.

## 700M From-Scratch Runs

- `baseline_700m.yaml`
- `exp2_700m.yaml`, `exp2_700m_40k.yaml`
- `exp4_700m.yaml`, `exp4_700m_b.yaml`, `exp4_700m_40k.yaml`

## Official LAM Fine-Tuning

- `ft_l32_full.yaml`, `ft_l32_full_5k.yaml`
- `ft_l32_frozen.yaml`
- `ft_l40_full.yaml`, `ft_l40_full_5k.yaml`
- `ft_l40_frozen.yaml`

## Contrastive Fine-Tuning

- `ft_l40_contrastive.yaml`, `ft_l40_contrastive_5k.yaml`
- `ft_l40_contrastive_v2.yaml`, `ft_l40_contrastive_v2_5k.yaml`
- `ft_l40_contrastive_v3.yaml`, `ft_l40_contrastive_v3_5k.yaml`

`v3` is the current verb-SupCon branch. It is kept as the contrastive baseline
that tests whether direct semantic clustering helps or hurts action separation.

## Fine-Grained Causal LAM

- `ft_l40_ours_a_zero_5k.yaml`: KL trunk + zero-transition calibration.
- `ft_l40_ours_b_zero_reverse_5k.yaml`: Ours-A + stronger temporal reverse consistency.
- `ft_l40_ours_b2_zero_action_reverse_5k.yaml`: Ours-A + action-only temporal reverse. This tests whether removing the env reverse constraint avoids context leakage while preserving action direction sensitivity.
- `ft_l40_ours_c_zero_reverse_hardneg_5k.yaml`: Ours-B + phase-aware semantic reversible hard-negative contrast. Positive/negative labels come from resolved verbs; local motion and temporal position only soft-weight likely core-action frame pairs.
- `ft_l40_ours_c_zero_reverse_hardneg_fg_5k.yaml`: Ours-C plus SAM3 hand/object/contact reconstruction weights. Requires `sam3_hoc_masks`.
