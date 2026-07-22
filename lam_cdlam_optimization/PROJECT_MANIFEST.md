# PROJECT MANIFEST — `lam_cdlam_optimization` (Project A)

> Extracted from the former `lam_disentangle/` working directory on 2026-07-22.
> This is the **oldest and largest** of the three latent-action research lines; B and C descend from it.

## 1. Research question & claim

**Question.** Can the latent action `z_a` of a DreamDojo-style continuous LAM be made *sufficient, cross-scene invariant, controllable, and groundable* — i.e. a clean causal action interface — by CD-LAM-style debiasing plus a 32-D action / 8-D environment split and a family of auxiliary losses?

**Working claim (concepts_and_pipeline.md §H, research_qa_synthesis.md §C3), verbatim:**
> “CD-LAM 修正一般 confounding 之后，运动方向仍是条件通道中未编码的自由度；我们用真实数据的运动学反平行对将其注入，并以非循环协议证明。”

**Foundational definition (concepts_and_pipeline.md §A):**
> “z_a = 引起两帧变化的、跨场景不变的原因，精确到视觉可分辨粒度。”

**Status: the direction claim was empirically overturned in place** — motion direction is weakly present even in the raw LAM, and the L_dir injection did not produce external/decoder gains (see Stage-B arms `sb_D/R/S`, now owned by Project B). This negative result **spawned Project B** (`../lam_condition_utilization`).

## 2. Experiments — done / failed / open

| Line | Status | Where |
|---|---|---|
| From-scratch 32+8 split, decorrelation → KL-bottleneck (β_a=3e-3) | done; decorrelation collapses, split-KL "directed purification" | `archive/notes/{design,preparation,results-v1,exp3-analysis}.md` |
| Finetune DreamDojo `LAM_400k` + ConLA-style contrastive v1/v2/v3 | done | `archive/notes/{results-v2-finetune,HANDOFF}.md`, `results/ft_l40_contrastive*` |
| Fine-grained causal LAM: L_zero / reverse / hard-neg / foreground / opposite | done | `notes/concepts_and_pipeline.md`, `results/lam_benchmark*` |
| CD-LAM reproduction & fidelity audit | done | `notes/cdlam_repro_fidelity.md`, arms `cdlam_repro`, `cdlam_official` |
| L_dir anti-parallel pair mining (Δt-window enrichment) | done | `notes/dir_pair_audit.md`, `data/dir_pairs_train.npz`, `code/mine_dir_pairs.py` |
| v2 benchmark overhaul (permutation-null kNN, routing triplet, do(z_a=0), swap) | done; 4 old conclusions overturned | `notes/benchmark_v2_{changes,findings}.md`, `results/lam_benchmark_v2/` |
| Stage-0 hardening (T1–T9), seed variance | done | `notes/stage0_{execution_plan,report}.md`, `notes/seed_variance.md` |
| Direction / opposite-action **separation via L_dir** | **FAILED** (double-metric rejection) → handed to Project B's Stage-B | `results/lam_benchmark_opposite`, `..._r*`; B's `stage_b_*` |
| `ours_e_dir*` arms | **incomplete** — checkpoints empty (`checkpoints/lam-dis/ft_l40_ours_e_dir*` = no `step_*`) | configs only |
| SAM3 hand-object-contact foreground masks | partial; see [`sam3-object-prompt-failure`] caveat in notes | `code/precompute_sam3_hoc_masks.py`, `results/sam3_hoc_preview/` |

## 3. Entrypoints

- **Train:** `python code/train.py --config code/config/<arm>.yaml` (Accelerate; config-driven losses). Run from this project root.
- **Benchmark (zero-train):** `python code/benchmark_lam.py --runs <arm...> --out_dir results/<dir>` — `RUN_SPECS` is the **shared arm registry** (lists A *and* B `sb_*` arms; checkpoint paths absolute). `_resolve()` finds configs in this project **and** in `../lam_condition_utilization` (for `sb_*`).
- **Eval (legacy 4-exp):** `python code/eval.py --checkpoint <ckpt> --config <cfg> --mode {action_probe,reconstruction,visualize,perturb}`.
- **L_dir mining:** `python code/mine_dir_pairs.py` → `data/dir_pairs_train.npz`; audits in `code/audit_dir_pairs{,2,3}.py`.
- **Batch drivers:** `scripts/run_ours.sh {a,b,b2,r,c,...}`, `scripts/run_all.sh`, `scripts/run_finetune.sh`, `scripts/run_eval_all.sh`; SAM3 masks `scripts/run_sam3_hoc_{full,rerun}.sh`; data prep `scripts/transcode_240p.sh`.

## 4. Config / checkpoint / result lineage

- **Configs:** `code/config/*.yaml` — legacy scratch (`exp*`, `kl_*`, `baseline*`, `ft_l32*`), CD-LAM anchors (`cdlam_official`, `ft_l40_cdlam_repro_5k`, `ft_l40_idm_5k`), and `ours_*` fine-tune arms. All train from `checkpoints/pretrained/LAM_400k.ckpt` (except scratch), output to `checkpoints/lam-dis/<name>/`.
- **Checkpoints (physically at the shared, gitignored `checkpoints/lam-dis/`; logically owned here):**

  | arm (dir under `checkpoints/lam-dis/`) | step | model | note |
  |---|---|---|---|
  | `ft_l40_cdlam_repro_5k` | 5000 | 2.7 G | CD-LAM reproduction |
  | `ft_l40_full_5k` (+`_lf`,`_s1`,`_s2`) | 5000 | 2.7 G | main split-KL "full" arm + twin + seeds |
  | `ft_l40_contrastive_v3_5k` | 5000 | 2.7 G | ConLA-style contrastive |
  | `ft_l40_idm_5k` | 5000 | 2.7 G | IDM oracle baseline |
  | `ft_l40_ours_a_zero_5k` (+`_s1`,`_s2`) | 5000 | 2.7 G | L_zero calibration + seeds |
  | `ft_l40_ours_b_zero_reverse_5k`, `ft_l40_ours_b2_zero_action_reverse_5k` | 5000 | 2.7 G | temporal reverse |
  | `ft_l40_ours_c_zero_reverse_hardneg_5k` | 5000 | 2.7 G | reverse + hard-negative |
  | `ft_l40_ours_r_zero_opposite_5k`, `ft_l40_ours_r2_opp_l01_5k` (+`_lf`,`_s1`,`_s2`) | 5000 | 2.7 G | opposite-action λ-sweep + seeds |
  | `ft_l40_ours_e_dir_5k`, `..._dirsame_5k` | — | **empty** | never completed |

  Base weights (shared): `checkpoints/pretrained/LAM_400k.ckpt` (`sha256 d77bf1b3…`), `CDLAM_official_lam.pt` (`sha256 084f9b1a…`).
- **Results:** every `results/*/` subdir is a benchmark/eval output for the arm named after it; the authoritative set is `results/lam_benchmark_v2/`. Top-level `results/dir_pair_audit*.json`, `dirprobe_rescore.csv`, `*_DONE.txt`, `eval.log`, `SUMMARY.txt`. Run logs in `logs/`.

## 5. Common data used (read-only, not owned here)
- `/home/xuan/embodied-ai/data/egodex/test_240p/` (EgoDex, 240p all-intra transcode; produced by `scripts/transcode_240p.sh`).
- `/home/xuan/embodied-ai/checkpoints/pretrained/` (base LAM & CD-LAM weights).
- `/home/xuan/embodied-ai/code/adaworld/lam/` (AdaWorld `lam.modules.blocks`, imported read-only by `model.py`/`benchmark_lam.py`).

## 6. Dependencies on / from other projects
- **Provides to B & C:** the shared LAM substrate (`code/{train,model,dataset,eval,benchmark_lam,primitive_labels,dirprobe}.py`), the fine-grained benchmark, `data/dir_pairs_train.npz`, and the CD-LAM reproduction. B imports this code via its `code/_apath.py` shim.
- **Reads from B:** `../lam_condition_utilization/data/eval_manifest.json` (Eval-A frozen split, created during Stage B; used by `dirprobe`/`rescore_dirprobe` and by `sb_D/R/S` configs).

## 7. Sealed status & next steps
- Benchmark v2 + Stage-0 are the frozen authoritative results. **Do not overwrite `results/lam_benchmark_v2/`.**
- The **direction/opposite-action separation claim is retired** (see §1). Continue direction work only under Project B's availability/utilization frame.
- `ours_e_dir*` and full SAM3 foreground pipeline are the only unfinished threads.

## 8. Results that must NOT be reused out of context
- Any **`archive/notes/`** ranking/recommendation — superseded by v2 (most old checkpoints deleted). Salvaged facts live in `notes/legacy_lessons.md`.
- `notes/analysis-recon-perdim.md` model ranking — **void**; only its 18-D action definition and "PSNR≠disentanglement" points survive.
- The direction-loss (`L_dir`) *success* narrative — **rejected**; cite the Stage-B rejection in `../lam_condition_utilization/notes/stage_b_plan.md`.
