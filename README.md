# embodied-ai

围绕潜在动作模型（LAM）与动作条件世界模型的多项目研究工作区，起源于在 AgiBot-World 与
EgoDex 上对 DreamDojo LAM 训练管线（arXiv:2602.06949，NVIDIA/UCB）的复现。

**完整地图见 [`WORKSPACE_MANIFEST.md`](WORKSPACE_MANIFEST.md)：** 项目、依赖图，以及公共
data/code/checkpoint 的归属。

## 项目
- **[`lam_cdlam_optimization/`](lam_cdlam_optimization/)**（A）— CD-LAM loss / latent 结构 / 细粒度动作。
- **[`lam_condition_utilization/`](lam_condition_utilization/)**（B）— action availability 与 decoder utilization 之分；"Availability is not Utilization" 诊断。*依赖 A。*
- **[`lam_world_model_control/`](lam_world_model_control/)**（C）— 可迁移 exported LAM → 世界模型控制（预注册阶段）。*依赖 A + B。*
- `dreamdojo/`、`lam-agibot/`、`lam-egodex/` — 最初的 DreamDojo 复现 / LAM 训练项目。
- `fast_wam/` — 独立的 Fast-WAM / 多视角 WAM 线。

## 公共资源（各自单一来源）
- `data/` — 数据集与可复用衍生物（EgoDex、AgiBotWorld）。*已 gitignore。*
- `checkpoints/pretrained/` — 公共基础权重；`checkpoints/lam-dis/` — A+B 共享的微调 checkpoint 存储。*已 gitignore。*
- `code/` — `lam/`（基础 LAM 训练）、第三方 `adaworld/`、`sam3/`、`cosmos-predict25/`。

各项目的中文研究笔记在各自的 `notes/` 下。
A/B 于 2026-07-22 从原 `lam_disentangle/` 工作目录拆分而来。
