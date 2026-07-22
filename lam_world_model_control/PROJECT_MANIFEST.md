# PROJECT MANIFEST — `lam_world_model_control` (Project C)

> New project (initialized 2026-07-22). Descends from Project B's α=0 fix.
> **Least mature line** — currently a preregistration + expert-consultation brief, no runs yet.

## 1. Research question & claim
**Question.** Does a *better exported latent-action interface* (more control-sufficient,
context-invariant, groundable, and **consumer-portable**) produce *better
action-conditioned world-model control*?

**Long-term claim to be earned (not yet demonstrated), verbatim from `notes/consult_new_project_prompt.md`:**
> “在相同 world-model 架构、数据、初始化分布和训练预算下，一个更具 control sufficiency、context invariance、groundability 和 consumer portability 的 LAM，能够带来更好的 action-conditioned rollout、更高效的 executable-action grounding，以及更强的规划或策略效果。”

Required evidence chain:
```
LAM-side independent property improvement
  → matched independent consumer / ACWM action-following improvement
  → executable-action adaptation / planning / policy utility improvement
```

## 2. Status
- **No experiments run.** Only the consultation brief `notes/consult_new_project_prompt.md`
  (structured-latent 32+8 question, held-out-consumer portability risk, phased go/no-go A/B/C).
- Phase-0 evidence inherited from Project B currently **predicts the headline transfer is ≈ null for a μ-only consumer** — C must *earn* the claim, not assume it.

## 3. Planned entrypoints (to be built in `code/`)
Held-out surrogate consumers (Conv-UNet / Transformer × FiLM/cross-attn/adapter),
K-step adaptation risk `R_port(E)`, matched independent ACWM (300M–1.3B → 2B), FDCE /
signed-displacement scorers. Target ACWM likely uses `/home/xuan/embodied-ai/code/cosmos-predict25/`.

## 4. Dependencies (read-only; do NOT copy)
- **On Project A (`../lam_cdlam_optimization`):** CD-LAM reproduction & fidelity
  (`notes/cdlam_repro_fidelity.md` §3 = FDCE/ACWM seed), the fine-grained benchmark, and
  base checkpoints. A's `benchmark_lam`/`eval` substrate may be reused for latent extraction.
- **On Project B (`../lam_condition_utilization`):** paired-latent & 2×2 cross-decode tooling,
  the 8.6 M route-C consumer (`code/consumer_model.py`, `train_consumer.py`), the α
  negative-boundary result (**motivation only**), and the frozen protocol
  `data/eval_manifest_B.json` + `data/consumer_exclude.json` + `notes/tier2_prereg.md`
  (C's Phase-A gate descends directly from B's Tier-2 prereg).
- **Common:** `data/egodex/test_240p`, `checkpoints/pretrained/`, `code/cosmos-predict25/`, `code/adaworld/`.

Base weights (shared anchors): `LAM_400k.ckpt` `sha256 d77bf1b3…`; `CDLAM_official_lam.pt` `sha256 084f9b1a…`.

## 5. Guardrails — what may NOT be claimed
- Project B's **α=0 result is a diagnostic boundary, not a positive downstream result** for C.
- No "better LAM → better control" claim until the matched-ACWM chain (§1) passes B's preregistered gates in `../lam_condition_utilization/notes/tier2_prereg.md`.
- Reconstruction/probe/paired-decoder gains ≠ exported-interface quality (the entire premise of this project).
