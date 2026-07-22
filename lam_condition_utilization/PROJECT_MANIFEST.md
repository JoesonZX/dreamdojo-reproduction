# PROJECT MANIFEST — `lam_condition_utilization` (Project B)

> Extracted from the former `lam_disentangle/` working directory on 2026-07-22.
> Descends from Project A (`../lam_cdlam_optimization`) at the L_dir failure; is the
> parent of Project C (`../lam_world_model_control`).

## 1. Research question & claim

**Question (paper_draft.md §1 / Abstract), verbatim:**
> “We ask whether a specific, interpretable degree of freedom — motion direction / opposite actions — is (i) *available* (linearly readable from `z`) and (ii) *utilized* (causally used by the decoder).”

**Claim — the diagnostic paper** (`notes/paper_draft.md`, working title):
> *“Availability is not Utilization: Diagnosing — and Removing — a Training-Noise Bottleneck in the Latent-Action-to-Decoder Channel.”*

Core finding: information can be **available** in `z` yet **not utilized** by the paired decoder; the gap is a training-noise (decoder-exposure) bottleneck. Closing the decoder's exposure to posterior sampling noise (**α=0**) restores the paired decoder's utilization while barely changing the encoder `μ`.

## 2. Experiments — done / failed / open

| Line | Status | Where |
|---|---|---|
| Stage-B 5-arm successive-halving (B/D/R/S/U) — H1 amplification & H2 utilization | done; **H1/H2 rejected for L_dir** (geometry moved, no external/decoder gain) | `notes/stage_b_plan.md`, `results/stage_b_eval.json` |
| H2 decompose + posterior-SNR; H2 positive control (hard gate) | done; **positive control PASSES** (scorer sensitivity proven), donor premise falsified | `results/stage_b_h2_*.json/.npz` |
| α = 0 / α = 1 decoder-exposure ablation (matched, seeds 42 & 1) | done; α=0 restores decoder utilization, μ ≈ unchanged | `code/config/ft_l40_sb_A{0,1}{,_s1}.yaml`, `results/alpha_*.log` |
| Phase 0.1 paired-latent; Phase 0.2 2×2 cross-decode | done; effect localized to the **decoder row** | `results/stage_b_phase0_{paired_latent,cross_decode}.json` |
| Tier-1.5 K-step / sign intervention rollout | done | `results/stage_b_tier15.json`, `code/{rollout_tier15,eval_tier15}.py` |
| Route-C independent consumer: GT18 vs zmu-A0/A1 (from-scratch 8.6 M consumer) | done; **A0 ≈ A1 for a μ-only consumer** (predicted-null) | `results/consumer_*.{json,pt,log}`, `code/{consumer_model,train_consumer,eval_consumer}.py` |
| OOD / generative robustness at α=0 | done | `results/stage_b_ood_robustness.json` |
| Tier-2 preregistration (route-C, Phase 0.3 freeze) | **prereg written; execution deferred → Project C** | `notes/tier2_prereg.md`, `notes/tier2_execution_handoff.md` |

## 3. Entrypoints
- **Stage-B / α / utilization LAM arms are trained through A's shared trainer:**
  `python ../lam_cdlam_optimization/code/train.py --config code/config/ft_l40_sb_<arm>.yaml` (run from this project root; arms differ only by loss knobs — see §4).
- **Diagnostic evaluators (this project):** `code/eval_stage_b.py`, `code/eval_phase0.py`, `code/eval_tier15.py`, `code/eval_h2_poscontrol.py`, `code/eval_h2_decompose.py`, `code/eval_ood_robustness.py`. Each imports A's substrate via **`code/_apath.py`** (documented read-only dependency) and writes to `results/`.
- **Route-C consumer:** `code/train_consumer.py --cond {gt_action,zmu} --arm <arm> …` (standalone CLI, no YAML; checkpoints to `results/consumer_*.pt`), evaluated by `code/eval_consumer.py`.
- **Freeze the held-out Eval-B split:** `code/make_eval_manifest_B.py`.

## 4. Config / checkpoint / result lineage
- **Configs — the `sb_*` Stage-B family** (`code/config/`), all init from `checkpoints/pretrained/LAM_400k.ckpt`, `lambda_zero=0.1`, output `checkpoints/lam-dis/ft_l40_sb_<arm>/`:

  | config | knob | arm |
  |---|---|---|
  | `ft_l40_sb_B.yaml` | zero-cal only | Stage-B baseline (shared control) |
  | `ft_l40_sb_D.yaml` | `lambda_dir 0.1, kind=0` | L_dir anti-parallel (tests A's method) |
  | `ft_l40_sb_R.yaml` | `lambda_dir 0.1, kind=1` | matched repulsion control |
  | `ft_l40_sb_S.yaml` | `lambda_dir 0.1, kind=2` (300 steps) | same-direction adverse sentinel |
  | `ft_l40_sb_U.yaml` | `lambda_use 0.1, use_margin 0.02` | utilization hinge |
  | `ft_l40_sb_A0.yaml` / `A1.yaml` (+`_s1`) | `decoder_noise_alpha 0.0 / 1.0` | α-exposure (seeds 42 & 1) |

- **Checkpoints (physically at shared gitignored `checkpoints/lam-dis/`; logically owned here):** `ft_l40_sb_{A0,A0_s1,A1,A1_s1,B,D,R,S,U}` — each `step_0001000/model.safetensors` ≈ 2.7 G (`sb_S` = `step_0000300`). Consumer checkpoints: `results/consumer_*.pt` (≈ 34 M each, owned here).
- **Results:** `results/stage_b_*.json/.npz` (authoritative diagnostic data, traceable to `paper_draft.md`), `results/consumer_*`, `results/alpha_*`, `results/{h2_*,phase0,tier15,ood_robustness}.log`. `results/stage_b_eval_5task_STALE.json` is a **known-bad** GroupKFold artifact — do not use.
- **Data / protocol (this project):** `data/eval_manifest.json` (Eval-A frozen split), `data/eval_manifest_B.json` (held-out Eval-B for route-C), `data/consumer_exclude.json` (consumer-train exclude). All **FROZEN** — do not regenerate.

## 5. Common data used (read-only)
`/home/xuan/embodied-ai/data/egodex/test_240p/`, `/home/xuan/embodied-ai/checkpoints/pretrained/`, `/home/xuan/embodied-ai/code/adaworld/lam/`.

## 6. Dependencies
- **On A (`../lam_cdlam_optimization`), read-only:**
  - **Code substrate** — `benchmark_lam.py` (arm registry `RUN_SPECS`), `eval.py`, `dataset.py`, `model.py`, `train.py`, `primitive_labels.py`, `dirprobe.py`. Made importable by `code/_apath.py`. See A's manifest for ownership.
  - **Checkpoints** — B benchmarks A arms (`raw_lam`, `kl_full`, `ours_a`, …) as controls; read-only, referenced by absolute path in `RUN_SPECS`.
  - **Data** — `../lam_cdlam_optimization/data/dir_pairs_train.npz` (used by `sb_D/R/S`).
- **Provides to C (`../lam_world_model_control`):** paired-latent & cross-decode tooling, the 8.6 M route-C consumer, the α negative-boundary result, and `data/eval_manifest_B.json` + `data/consumer_exclude.json` + `notes/tier2_prereg.md`.

## 7. Sealed status & next steps
- The diagnostic paper (`notes/paper_draft.md`) is the deliverable; its numbers are authoritative and traceable to `results/stage_b_*`.
- Tier-2 (matched 2B ACWM, downstream transfer) is **preregistered but deferred to Project C**.

## 8. Results that must NOT be reused out of context
- **α is a diagnostic fix of a low-rate/low-SNR training pathology, NOT a new method.** Do not present α=0 as a positive downstream result (see `notes/tier2_prereg.md`, C's brief §3.3).
- Route-C **A0 ≈ A1** is a *predicted-null* for a μ-only consumer; it does **not** show μ is unusable, nor does it establish any "better LAM → better control" claim — that is Project C's open question.
- `results/stage_b_eval_5task_STALE.json` — GroupKFold-bug artifact; excluded.
