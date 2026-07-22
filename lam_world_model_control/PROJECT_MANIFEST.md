# 项目清单 — `lam_world_model_control`（项目 C）

> 新项目（2026-07-22 初始化）。由项目 B 的 α=0 修复衍生。
> **最不成熟的一条线** —— 目前只是一份预注册 + 专家咨询 brief，尚无实验。

## 1. 研究问题与 claim
**问题。** 一个*更好的 exported latent-action 接口*（更具 control sufficiency、跨场景不变、
可接地、以及**consumer 可迁移性**）是否带来*更好的动作条件世界模型控制*？

**待挣得（尚未证明）的长期 claim，原文引自 `notes/consult_new_project_prompt.md`：**
> “在相同 world-model 架构、数据、初始化分布和训练预算下，一个更具 control sufficiency、context invariance、groundability 和 consumer portability 的 LAM，能够带来更好的 action-conditioned rollout、更高效的 executable-action grounding，以及更强的规划或策略效果。”

所需证据链：
```
LAM 侧独立性质改善
  → matched 独立 consumer / ACWM action-following 改善
  → executable-action 适配 / 规划 / 策略效用改善
```

## 2. 状态
- **无实验运行。** 只有咨询 brief `notes/consult_new_project_prompt.md`
  （结构化 latent 32+8 问题、held-out-consumer 可迁移性风险、分阶段 A/B/C go/no-go）。
- 从项目 B 继承的 Phase-0 证据当前**预测 μ-only consumer 的 headline 迁移 ≈ null** —— C 必须*挣得*该 claim，不能预设成立。

## 3. 规划中的入口（将在 `code/` 中构建）
held-out 代理 consumer（Conv-UNet / Transformer × FiLM/cross-attn/adapter）、
K 步适配风险 `R_port(E)`、matched 独立 ACWM（300M–1.3B → 2B）、FDCE / signed-displacement
scorer。目标 ACWM 大概率使用 `/home/xuan/embodied-ai/code/cosmos-predict25/`。

## 4. 依赖（只读；**不复制**）
- **对项目 A（`../lam_cdlam_optimization`）：** CD-LAM 复现与保真度
  （`notes/cdlam_repro_fidelity.md` §3 = FDCE/ACWM 种子）、细粒度 benchmark、基础 checkpoint。
  可复用 A 的 `benchmark_lam`/`eval` 底座做 latent 提取。
- **对项目 B（`../lam_condition_utilization`）：** paired-latent 与 2×2 cross-decode 工具、
  8.6M route-C consumer（`code/consumer_model.py`、`train_consumer.py`）、α 负边界结果
  （**仅作 motivation**），以及冻结协议 `data/eval_manifest_B.json` + `data/consumer_exclude.json`
  + `notes/tier2_prereg.md`（C 的 Phase-A 门槛直接承接自 B 的 Tier-2 prereg）。
- **公共：** `data/egodex/test_240p`、`checkpoints/pretrained/`、`code/cosmos-predict25/`、`code/adaworld/`。

基础权重（共享锚点）：`LAM_400k.ckpt` `sha256 d77bf1b3…`；`CDLAM_official_lam.pt` `sha256 084f9b1a…`。

## 5. 守门 —— 不得声称的内容
- 项目 B 的 **α=0 结果是诊断性边界，不是 C 的正向下游结果**。
- 在 matched-ACWM 链（§1）通过 B 在 `../lam_condition_utilization/notes/tier2_prereg.md` 里预注册的门槛之前，不得声称"更好 LAM → 更好控制"。
- 重建/probe/配对 decoder 的增益 ≠ exported 接口质量（这正是本项目的前提）。
