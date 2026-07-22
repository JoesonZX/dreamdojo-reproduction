# LAM World-Model Control

Research workspace for studying whether a better exported latent-action interface leads to
better action-conditioned world-model control.

The former `lam_disentangle/` working directory has been split into its two constituent
projects: **A** = [`../lam_cdlam_optimization`](../lam_cdlam_optimization) (CD-LAM loss /
latent structure / fine-grained action) and **B** = [`../lam_condition_utilization`](../lam_condition_utilization)
(the completed availability-vs-utilization diagnostic). **Lineage: A → B → C.** This project
descends from B's α=0 finding and depends read-only on both A and B — see
[`PROJECT_MANIFEST.md`](PROJECT_MANIFEST.md) §4. The initial expert-consultation brief is in
`notes/consult_new_project_prompt.md`; its Phase-A gate descends from
[`../lam_condition_utilization/notes/tier2_prereg.md`](../lam_condition_utilization/notes/tier2_prereg.md).

**Not yet run** — this is a preregistration/plan only, and B's evidence currently predicts the
headline transfer is ≈ null for a μ-only consumer, so the claim must be *earned* (manifest §5).

Directory ownership:

- `code/`: project-specific implementation;
- `notes/`: research questions, decisions, preregistrations, and paper notes;
- `results/`: project-specific experiment outputs;
- shared datasets and reusable dataset derivatives remain under `/home/xuan/embodied-ai/data`;
- genuinely cross-project preprocessing code remains under `/home/xuan/embodied-ai/code`.
