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

`v3` is the current verb-SupCon branch; the next planned branch is the
reverse-calibrated hard-negative loss described in `../../notes/fine_grained_causal_lam_plan.md`.

