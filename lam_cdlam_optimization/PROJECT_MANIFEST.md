# 项目清单 — `lam_cdlam_optimization`（项目 A）

> 2026-07-22 从原 `lam_disentangle/` 工作目录拆分而来。
> 这是三条潜在动作研究线中**最早、最大**的一条；B 与 C 都由它衍生。

## 1. 研究问题与 claim

**问题。** 能否让 DreamDojo 式连续 LAM 的潜在动作 `z_a` 变得*充分、跨场景不变、可控、可接地*——
即一个干净的因果动作接口——通过 CD-LAM 式 debiasing，加上 32 维动作 / 8 维环境的拆分，以及
一族辅助 loss？

**工作 claim（concepts_and_pipeline.md §H、research_qa_synthesis.md §C3，原文）：**
> “CD-LAM 修正一般 confounding 之后，运动方向仍是条件通道中未编码的自由度；我们用真实数据的运动学反平行对将其注入，并以非循环协议证明。”

**基础定义（concepts_and_pipeline.md §A）：**
> “z_a = 引起两帧变化的、跨场景不变的原因，精确到视觉可分辨粒度。”

**状态：该方向 claim 已被就地推翻**——运动方向在原始 LAM 里也弱可读，且 L_dir 注入没有带来
外部/decoder 增益（见 Stage-B 的 `sb_D/R/S` arm，现归项目 B）。这个负结果**催生了项目 B**
（`../lam_condition_utilization`）。

## 2. 实验 — 完成 / 失败 / 未完成

| 线 | 状态 | 位置 |
|---|---|---|
| 从零 32+8 拆分，decorrelation → KL 瓶颈（β_a=3e-3） | 完成；decorrelation 塌缩，split-KL "定向提纯" | `archive/notes/{design,preparation,results-v1,exp3-analysis}.md` |
| 微调 DreamDojo `LAM_400k` + ConLA 式 contrastive v1/v2/v3 | 完成 | `archive/notes/{results-v2-finetune,HANDOFF}.md`、`results/ft_l40_contrastive*` |
| 细粒度因果 LAM：L_zero / reverse / hard-neg / 前景 / opposite | 完成 | `notes/concepts_and_pipeline.md`、`results/lam_benchmark*` |
| CD-LAM 复现与保真审计 | 完成 | `notes/cdlam_repro_fidelity.md`、arm `cdlam_repro`、`cdlam_official` |
| L_dir 反平行配对挖掘（Δt 窗口富集） | 完成 | `notes/dir_pair_audit.md`、`data/dir_pairs_train.npz`、`code/mine_dir_pairs.py` |
| v2 benchmark 重构（permutation-null kNN、路由三元组、do(z_a=0)、swap） | 完成；4 个旧结论被推翻 | `notes/benchmark_v2_{changes,findings}.md`、`results/lam_benchmark_v2/` |
| Stage-0 加固（T1–T9）、seed 方差 | 完成 | `notes/stage0_{execution_plan,report}.md`、`notes/seed_variance.md` |
| **用 L_dir 做方向/相反动作分离** | **失败**（双指标否决）→ 移交项目 B 的 Stage-B | `results/lam_benchmark_opposite`、`..._r*`；B 的 `stage_b_*` |
| `ours_e_dir*` arm | **未完成** — checkpoint 为空（`checkpoints/lam-dis/ft_l40_ours_e_dir*` 无 `step_*`） | 仅有 config |
| SAM3 手-物-接触前景 mask | 部分完成；注意 notes 里的 object-prompt 失效告警 | `code/precompute_sam3_hoc_masks.py`、`results/sam3_hoc_preview/` |

## 3. 入口

- **训练：** `python code/train.py --config code/config/<arm>.yaml`（Accelerate；loss 全由 config 驱动）。在项目根目录运行。
- **Benchmark（零训练）：** `python code/benchmark_lam.py --runs <arm...> --out_dir results/<dir>` — `RUN_SPECS` 是**共享 arm 注册表**（同时列 A 与 B 的 `sb_*` arm；checkpoint 路径为绝对路径）。`_resolve()` 会在本项目**以及** `../lam_condition_utilization`（找 `sb_*`）里解析 config。
- **Eval（旧 4-exp）：** `python code/eval.py --checkpoint <ckpt> --config <cfg> --mode {action_probe,reconstruction,visualize,perturb}`。
- **L_dir 挖掘：** `python code/mine_dir_pairs.py` → `data/dir_pairs_train.npz`；审计见 `code/audit_dir_pairs{,2,3}.py`。
- **批处理驱动：** `scripts/run_ours.sh {a,b,b2,r,c,...}`、`scripts/run_all.sh`、`scripts/run_finetune.sh`、`scripts/run_eval_all.sh`；SAM3 mask `scripts/run_sam3_hoc_{full,rerun}.sh`；数据预处理 `scripts/transcode_240p.sh`。

## 4. config / checkpoint / result 谱系

- **config：** `code/config/*.yaml` — 旧的从零探索（`exp*`、`kl_*`、`baseline*`、`ft_l32*`）、CD-LAM 锚点（`cdlam_official`、`ft_l40_cdlam_repro_5k`、`ft_l40_idm_5k`）、以及 `ours_*` 微调 arm。除从零外均从 `checkpoints/pretrained/LAM_400k.ckpt` 训练，输出到 `checkpoints/lam-dis/<name>/`。
- **checkpoint（物理位于共享、已 gitignore 的 `checkpoints/lam-dis/`；逻辑归属在此）：**

  | arm（`checkpoints/lam-dis/` 下目录） | step | model | 备注 |
  |---|---|---|---|
  | `ft_l40_cdlam_repro_5k` | 5000 | 2.7 G | CD-LAM 复现 |
  | `ft_l40_full_5k`（+`_lf`,`_s1`,`_s2`） | 5000 | 2.7 G | 主 split-KL "full" arm + twin + seed |
  | `ft_l40_contrastive_v3_5k` | 5000 | 2.7 G | ConLA 式 contrastive |
  | `ft_l40_idm_5k` | 5000 | 2.7 G | IDM oracle baseline |
  | `ft_l40_ours_a_zero_5k`（+`_s1`,`_s2`） | 5000 | 2.7 G | L_zero 标定 + seed |
  | `ft_l40_ours_b_zero_reverse_5k`、`ft_l40_ours_b2_zero_action_reverse_5k` | 5000 | 2.7 G | 时间反转 |
  | `ft_l40_ours_c_zero_reverse_hardneg_5k` | 5000 | 2.7 G | reverse + 硬负 |
  | `ft_l40_ours_r_zero_opposite_5k`、`ft_l40_ours_r2_opp_l01_5k`（+`_lf`,`_s1`,`_s2`） | 5000 | 2.7 G | 相反动作 λ-sweep + seed |
  | `ft_l40_ours_e_dir_5k`、`..._dirsame_5k` | — | **空** | 从未训完 |

  基础权重（共享）：`checkpoints/pretrained/LAM_400k.ckpt`（`sha256 d77bf1b3…`）、`CDLAM_official_lam.pt`（`sha256 084f9b1a…`）。
- **结果：** 每个 `results/*/` 子目录是同名 arm 的 benchmark/eval 输出；权威集是 `results/lam_benchmark_v2/`。顶层有 `results/dir_pair_audit*.json`、`dirprobe_rescore.csv`、`*_DONE.txt`、`eval.log`、`SUMMARY.txt`。运行日志在 `logs/`。

## 5. 使用的公共数据（只读，不归本项目所有）
- `/home/xuan/embodied-ai/data/egodex/test_240p/`（EgoDex，240p 全关键帧转码；由 `scripts/transcode_240p.sh` 产出）。
- `/home/xuan/embodied-ai/checkpoints/pretrained/`（基础 LAM 与 CD-LAM 权重）。
- `/home/xuan/embodied-ai/code/adaworld/lam/`（AdaWorld `lam.modules.blocks`，被 `model.py`/`benchmark_lam.py` 只读 import）。

## 6. 与其它项目的依赖
- **提供给 B & C：** 共享 LAM 底座（`code/{train,model,dataset,eval,benchmark_lam,primitive_labels,dirprobe}.py`）、细粒度 benchmark、`data/dir_pairs_train.npz`、CD-LAM 复现。B 经其 `code/_apath.py` 垫片 import 本项目代码。
- **读取自 B：** `../lam_condition_utilization/data/eval_manifest.json`（Eval-A frozen split，Stage B 期间创建；被 `dirprobe`/`rescore_dirprobe` 及 `sb_D/R/S` config 使用）。

## 7. 封存状态与下一步
- Benchmark v2 + Stage-0 是冻结的权威结果。**不要覆盖 `results/lam_benchmark_v2/`。**
- **方向/相反动作分离 claim 已退役**（见 §1）。方向工作只在项目 B 的 availability/utilization 框架下继续。
- `ours_e_dir*` 与完整 SAM3 前景管线是仅有的未竟线索。

## 8. 不得脱离上下文复用的结果
- `archive/notes/` 里任何**排名/推荐** — 已被 v2 取代（多数旧 checkpoint 已删）。可用事实已打捞进 `notes/legacy_lessons.md`。
- `notes/analysis-recon-perdim.md` 的模型排名 — **作废**；只有其 18 维动作定义与 "PSNR≠解耦" 论点仍有效。
- 方向 loss（`L_dir`）的*成功*叙事 — **已否决**；引用时用 `../lam_condition_utilization/notes/stage_b_plan.md` 里的 Stage-B 否决。
