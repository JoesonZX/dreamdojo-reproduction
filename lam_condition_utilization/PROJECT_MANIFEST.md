# 项目清单 — `lam_condition_utilization`（项目 B）

> 2026-07-22 从原 `lam_disentangle/` 工作目录拆分而来。
> 在 L_dir 失败处从项目 A（`../lam_cdlam_optimization`）衍生；是项目 C（`../lam_world_model_control`）的母项目。

## 1. 研究问题与 claim

**问题（paper_draft.md §1 / Abstract，原文）：**
> “We ask whether a specific, interpretable degree of freedom — motion direction / opposite actions — is (i) *available* (linearly readable from `z`) and (ii) *utilized* (causally used by the decoder).”

**claim — 诊断论文**（`notes/paper_draft.md`，工作标题）：
> *“Availability is not Utilization: Diagnosing — and Removing — a Training-Noise Bottleneck in the Latent-Action-to-Decoder Channel.”*

核心发现：信息可以在 `z` 中**可得**却不被配对 decoder **利用**；这个 gap 是一个训练噪声
（decoder-exposure）瓶颈。关闭 decoder 对 posterior 采样噪声的暴露（**α=0**）能恢复配对
decoder 的利用，同时几乎不改变 encoder `μ`。

## 2. 实验 — 完成 / 失败 / 未完成

| 线 | 状态 | 位置 |
|---|---|---|
| Stage-B 五臂 successive-halving（B/D/R/S/U）— H1 放大 & H2 利用 | 完成；**L_dir 的 H1/H2 被否决**（几何变了，无外部/decoder 增益） | `notes/stage_b_plan.md`、`results/stage_b_eval.json` |
| H2 分解 + posterior-SNR；H2 正对照（硬门） | 完成；**正对照通过**（scorer 灵敏度已证），donor 前提被证伪 | `results/stage_b_h2_*.json/.npz` |
| α = 0 / α = 1 decoder-exposure 消融（matched，seed 42 & 1） | 完成；α=0 恢复 decoder 利用，μ ≈ 不变 | `code/config/ft_l40_sb_A{0,1}{,_s1}.yaml`、`results/alpha_*.log` |
| Phase 0.1 paired-latent；Phase 0.2 2×2 cross-decode | 完成；效应定位到 **decoder 行** | `results/stage_b_phase0_{paired_latent,cross_decode}.json` |
| Tier-1.5 K 步 / 符号干预 rollout | 完成 | `results/stage_b_tier15.json`、`code/{rollout_tier15,eval_tier15}.py` |
| Route-C 独立 consumer：GT18 vs zmu-A0/A1（从零 8.6M consumer） | 完成；**μ-only consumer 下 A0 ≈ A1**（预测为 null） | `results/consumer_*.{json,pt,log}`、`code/{consumer_model,train_consumer,eval_consumer}.py` |
| α=0 下的 OOD / 生成鲁棒性 | 完成 | `results/stage_b_ood_robustness.json` |
| Tier-2 预注册（route-C，Phase 0.3 freeze） | **prereg 已写；执行延后 → 项目 C** | `notes/tier2_prereg.md`、`notes/tier2_execution_handoff.md` |

## 3. 入口
- **Stage-B / α / utilization 的 LAM arm 通过 A 的共享 trainer 训练：**
  `python ../lam_cdlam_optimization/code/train.py --config code/config/ft_l40_sb_<arm>.yaml`（在本项目根目录运行；各 arm 仅 loss 旋钮不同 — 见 §4）。
- **诊断评测器（本项目）：** `code/eval_stage_b.py`、`code/eval_phase0.py`、`code/eval_tier15.py`、`code/eval_h2_poscontrol.py`、`code/eval_h2_decompose.py`、`code/eval_ood_robustness.py`。每个都经 **`code/_apath.py`** import A 的底座（文档化的只读依赖），并写入 `results/`。
- **Route-C consumer：** `code/train_consumer.py --cond {gt_action,zmu} --arm <arm> …`（独立 CLI，无 YAML；checkpoint 写到 `results/consumer_*.pt`），由 `code/eval_consumer.py` 评测。
- **冻结 held-out Eval-B split：** `code/make_eval_manifest_B.py`。

## 4. config / checkpoint / result 谱系
- **config — `sb_*` Stage-B 族**（`code/config/`），全部从 `checkpoints/pretrained/LAM_400k.ckpt` 初始化，`lambda_zero=0.1`，输出到 `checkpoints/lam-dis/ft_l40_sb_<arm>/`：

  | config | 旋钮 | arm |
  |---|---|---|
  | `ft_l40_sb_B.yaml` | 仅 zero 标定 | Stage-B baseline（共享对照） |
  | `ft_l40_sb_D.yaml` | `lambda_dir 0.1, kind=0` | L_dir 反平行（检验 A 的方法） |
  | `ft_l40_sb_R.yaml` | `lambda_dir 0.1, kind=1` | matched 排斥对照 |
  | `ft_l40_sb_S.yaml` | `lambda_dir 0.1, kind=2`（300 步） | 同向不利哨兵 |
  | `ft_l40_sb_U.yaml` | `lambda_use 0.1, use_margin 0.02` | utilization hinge |
  | `ft_l40_sb_A0.yaml` / `A1.yaml`（+`_s1`） | `decoder_noise_alpha 0.0 / 1.0` | α-exposure（seed 42 & 1） |

- **checkpoint（物理位于共享、已 gitignore 的 `checkpoints/lam-dis/`；逻辑归属在此）：** `ft_l40_sb_{A0,A0_s1,A1,A1_s1,B,D,R,S,U}` — 每个 `step_0001000/model.safetensors` ≈ 2.7 G（`sb_S` 为 `step_0000300`）。consumer checkpoint：`results/consumer_*.pt`（每个 ≈ 34 M，归本项目）。
- **结果：** `results/stage_b_*.json/.npz`（权威诊断数据，可追溯到 `paper_draft.md`）、`results/consumer_*`、`results/alpha_*`、`results/{h2_*,phase0,tier15,ood_robustness}.log`。`results/stage_b_eval_5task_STALE.json` 是**已知有 bug**的 GroupKFold 产物 — 不要用。
- **数据 / 协议（本项目）：** `data/eval_manifest.json`（Eval-A frozen split）、`data/eval_manifest_B.json`（route-C 的 held-out Eval-B）、`data/consumer_exclude.json`（consumer-train 排除表）。全部 **FROZEN** — 不要重新生成。

## 5. 使用的公共数据（只读）
`/home/xuan/embodied-ai/data/egodex/test_240p/`、`/home/xuan/embodied-ai/checkpoints/pretrained/`、`/home/xuan/embodied-ai/code/adaworld/lam/`。

## 6. 依赖
- **对 A（`../lam_cdlam_optimization`），只读：**
  - **代码底座** — `benchmark_lam.py`（arm 注册表 `RUN_SPECS`）、`eval.py`、`dataset.py`、`model.py`、`train.py`、`primitive_labels.py`、`dirprobe.py`。由 `code/_apath.py` 使其可 import。归属见 A 的 manifest。
  - **checkpoint** — B 把 A 的 arm（`raw_lam`、`kl_full`、`ours_a`…）当对照 benchmark；只读，`RUN_SPECS` 中以绝对路径引用。
  - **数据** — `../lam_cdlam_optimization/data/dir_pairs_train.npz`（被 `sb_D/R/S` 使用）。
- **提供给 C（`../lam_world_model_control`）：** paired-latent 与 cross-decode 工具、8.6M route-C consumer、α 负边界结果，以及 `data/eval_manifest_B.json` + `data/consumer_exclude.json` + `notes/tier2_prereg.md`。

## 7. 封存状态与下一步
- 诊断论文（`notes/paper_draft.md`）是交付物；其数值权威，可追溯到 `results/stage_b_*`。
- Tier-2（matched 2B ACWM、下游迁移）**已预注册但延后到项目 C**。

## 8. 不得脱离上下文复用的结果
- **α 是对低-rate/低-SNR 训练病态的诊断性修复，不是新方法。** 不要把 α=0 当作正向的下游结果（见 `notes/tier2_prereg.md`、C 的 brief §3.3）。
- Route-C 的 **A0 ≈ A1** 是 μ-only consumer 的*预测 null*；它**不**表明 μ 不可用，也不建立任何"更好 LAM → 更好控制"的 claim —— 那是项目 C 的开放问题。
- `results/stage_b_eval_5task_STALE.json` — GroupKFold bug 产物；已排除。
