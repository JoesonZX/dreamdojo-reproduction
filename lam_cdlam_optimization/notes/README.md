# notes/ index — Project A (lam_cdlam_optimization)

Reading order (not alphabetical). New session: start at §1.

## 1. Concepts & plan (read these first)
| file | content |
|---|---|
| **`concepts_and_pipeline.md`** | ⭐ conceptual single source of truth: `z_a` definition + 4 properties, 3-tier eval stack (by circularity), experiment↔metric alignment, opposite-action double-gate, loss↔acceptance table |
| `research_qa_synthesis.md` | 5 core research questions, final conclusions, phase 0–3 flow |
| `cdlam_repro_fidelity.md` | CD-LAM formula ↔ our implementation, 5 deviations, metric alignment, official ckpt usage. **Its §3 FDCE/ACWM is the seed for Project C.** |

## 2. Evaluation stack
| file | content |
|---|---|
| `benchmark_v2_changes.md` | metric dictionary (permutation-null kNN, routing triplet, do(z_a=0), swap) |
| `benchmark_v2_findings.md` | v2 results — 4 overturned conclusions + final failure list (numbers authoritative) |
| `dir_pair_audit.md` | L_dir anti-parallel pair mining audit (Δt-window enrichment 1.24×→2.59×) |
| `dirprobe_split_issue.md` | opp_cls episode-leak → `direction_gain` demoted to `static_shortcut_gap` |
| `sam3_foreground_investigation.md` | SAM3 hand+object foreground for L_emb weighted reconstruction |

## 3. Stage-0 results
| file | content |
|---|---|
| `stage0_execution_plan.md` | ⭐ execution discipline T1–T9 (read before running) |
| `stage0_report.md` | failure-list table (8 models), lf collapse, IDM 2×2, conditional-probe rewrite |
| `seed_variance.md` | T8 seed variance — "R2 has no headline advantage" verdict |

## 4. Long-lived reference
| file | content |
|---|---|
| `legacy_lessons.md` | salvaged facts: 18-D action physics, EgoDex pipeline, LAM_400k lineage, decorrelation lesson, figure index |
| `analysis-recon-perdim.md` | ⚠️ rankings VOID (v2); only 18-D action def + "PSNR≠disentanglement" survive |
| `throughput-debug.md` | data-pipeline perf (240p transcode, grad-accum trap) |
| `method_figure.drawio`, `Causal_Debiased_LAM*.pdf` | A method figure; CD-LAM preprint (external authority for formulas) |

## 5. Continuation into Project B (the direction line's outcome)
A's direction/opposite-action **separation claim was rejected** and reframed as
*availability vs utilization*. The follow-on lives in Project B:
- [`../../lam_condition_utilization/notes/paper_draft.md`](../../lam_condition_utilization/notes/paper_draft.md) — "Availability is not Utilization"
- [`../../lam_condition_utilization/notes/consult_story_b_prompt.md`](../../lam_condition_utilization/notes/consult_story_b_prompt.md) — the A→B hinge (label-free-trunk collapse → availability×utilization reframe)
- [`../../lam_condition_utilization/notes/stage_b_plan.md`](../../lam_condition_utilization/notes/stage_b_plan.md) — Stage-B rejection of L_dir

## 6. archive/notes/
Pre-pivot history (from-scratch exp1–4, ft_l32/l40, contrastive v1–v3, Ours-A/B/C).
**Rankings & recommendations are VOID** — most checkpoints deleted, conclusions overturned by v2.
Salvaged content is already in `legacy_lessons.md`. Consult for history only; do not cite conclusions.
Includes the two original indexes: `notes_README_orig.md`, `lam_disentangle_README_orig.md`.
