# LAM 世界模型控制（项目 C）

研究**更好的 exported latent-action 接口是否带来更好的动作条件世界模型控制**的工作区。

原 `lam_disentangle/` 工作目录已拆分为它包含的两个项目：**A** =
[`../lam_cdlam_optimization`](../lam_cdlam_optimization)（CD-LAM loss / latent 结构 /
细粒度动作）与 **B** = [`../lam_condition_utilization`](../lam_condition_utilization)
（已完成的 availability-vs-utilization 诊断）。**谱系：A → B → C。** 本项目由 B 的 α=0 发现
衍生，且只读依赖 A 与 B —— 见 [`PROJECT_MANIFEST.md`](PROJECT_MANIFEST.md) §4。初始专家咨询
brief 在 `notes/consult_new_project_prompt.md`；其 Phase-A 门槛承接自
[`../lam_condition_utilization/notes/tier2_prereg.md`](../lam_condition_utilization/notes/tier2_prereg.md)。

**尚未开跑** —— 目前只有预注册/计划，且 B 的证据当前预测 μ-only consumer 的 headline 迁移
≈ null，所以这个 claim 必须被*挣得*（见 manifest §5）。

## 目录归属
- `code/`：项目专属实现；
- `notes/`：研究问题、决策、预注册、论文笔记；
- `results/`：项目专属实验输出；
- 共享数据集与可复用数据衍生物仍在 `/home/xuan/embodied-ai/data`；
- 真正跨项目的预处理代码仍在 `/home/xuan/embodied-ai/code`。
