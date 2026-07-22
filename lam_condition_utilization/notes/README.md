# notes/ index — Project B (lam_condition_utilization)

Reading order (not alphabetical).

## 1. The paper & its decision log
| file | content |
|---|---|
| **`paper_draft.md`** | ⭐ the diagnostic paper — *"Availability is not Utilization…"*. 10 sections + 4 contributions + wording guardrails. Numbers authoritative, traceable to `../results/stage_b_*`. |
| **`stage_b_plan.md`** | ⭐ Stage-B decision single source of truth: 5-arm design, H1/H2 rejection, H2 decompose, positive-control gate (PASS), α ablation, Phase-0 cross-decode, Tier-1.5, Phase-0.3 freeze |
| `stage_b_mechanics.md` | Stage-B loss/eval implementation: 5 arms / 2 metrics / 3 controls, pair mining |

## 2. Consults & Tier-2 preregistration
| file | content |
|---|---|
| `consult_story_b_prompt.md` | A→B hinge: label-free-trunk collapse (A) → advisor reframe to availability×utilization (launches B) |
| `consult_related_work_contribution_roadmap.md` | contribution grading, λ-VAE novelty collision, routes A–E |
| `consult_tier2_prompt.md` | Tier-2 plan (localize → transfer → scale) + full advisor reply |
| **`tier2_prereg.md`** | ⭐ Tier-2 preregistration (Phase-0.3 freeze): route-C zmu A0/A1 consumer protocol + go/no-go gates. **Boundary artifact — its 2B-ACWM / "α→downstream" half is executed in Project C.** |
| `tier2_execution_handoff.md` | new-session execution prompt (Phase 0 → Tier-1.5 → route-C consumer) |

## 3. Upstream & downstream context
- **Upstream (Project A):** conceptual base and the direction line B tests —
  [`../../lam_cdlam_optimization/notes/concepts_and_pipeline.md`](../../lam_cdlam_optimization/notes/concepts_and_pipeline.md),
  [`../../lam_cdlam_optimization/notes/stage0_report.md`](../../lam_cdlam_optimization/notes/stage0_report.md).
- **Downstream (Project C):** whether the α fix / structured latent transfers to world-model control —
  [`../../lam_world_model_control/notes/consult_new_project_prompt.md`](../../lam_world_model_control/notes/consult_new_project_prompt.md).
  `tier2_prereg.md` here is the direct predecessor of C's Phase-A gate.
