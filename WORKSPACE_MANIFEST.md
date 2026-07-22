# 工作区清单 — `embodied-ai`

多项目研究工作区。单一 Git 仓库（`dreamdojo-reproduction`）。
2026-07-22 重整：把混合的 `lam_disentangle/` 工作目录，按研究问题与实验谱系，拆分为三个
边界清晰的项目（A、B、C）。

## 项目一览

| 目录 | 职责 | 状态 | 核心 claim |
|---|---|---|---|
| **`lam_cdlam_optimization/`**（A） | CD-LAM loss / latent 结构 / 细粒度动作 | 成熟；方向线 claim 已否决 | 用 CD-LAM + 32+8 拆分实现 `z_a` 的充分/不变/可控/可接地 |
| **`lam_condition_utilization/`**（B） | availability vs decoder utilization；α、cross-decode、Tier-2 | 成熟；论文草稿就绪 | "Availability is not Utilization"（α=0 修复 decoder，μ 基本不变） |
| **`lam_world_model_control/`**（C） | 可迁移 exported LAM → 世界模型控制 | 仅预注册，无实验 | "更好的 LAM → 更好的世界模型控制"（有待验证） |
| `dreamdojo/` | DreamDojo 复现笔记/结果 | 既有 | — |
| `lam-agibot/` | 在 AgiBot-World 上训练 LAM（700M、baseline/fg） | 既有 | — |
| `lam-egodex/` | 在 EgoDex 上训练 LAM（baseline/siglip） | 既有 | — |
| `fast_wam/` | Fast-WAM / 多视角 WAM（EgoExo4D → LIBERO） | 独立线 | 与 A/B/C 无关 |

## 依赖图（只读、已文档化——不复制任何东西）
```
        公共:  data/ , checkpoints/pretrained/ , code/{lam,adaworld,sam3,cosmos-predict25}
                              │
   A  lam_cdlam_optimization ─┤  （拥有共享的 CD-LAM 训练/benchmark 底座）
        │  代码底座、benchmark、dir_pairs、CD-LAM 复现、A 的 checkpoint
        ▼
   B  lam_condition_utilization   （经 code/_apath.py import A 的代码；benchmark A 的 ckpt；
        │  consumer/cross-decode/paired 工具、α 边界、Eval-B + prereg    读 A/data 的 dir_pairs；
        ▼                                                                 A 反向读 B 的 eval_manifest.json）
   C  lam_world_model_control      （读 A 的 benchmark/FDCE 种子 + B 的工具/prereg/α；ACWM 用 cosmos-predict25）
```
- A 自包含。**B 不能独立运行**——必须有 A 在场（已文档化的垫片）。C 需要 A + B。

## 公共资源归属（各自单一来源）

**`data/`（164 G，已 gitignore）** — 原始数据集与可复用衍生物，与具体模型 arm 无关：
`egodex/`（含 `test_240p/` 240p 转码）、`agibotworld/`、`hf_cache/`。
实验专属的 split/manifest **不在这里**——它们在各项目自己的 `data/`
（A：`dir_pairs_train.npz`；B：`eval_manifest*.json`、`consumer_exclude.json`）。

**`checkpoints/`（241 G，已 gitignore）**
- `pretrained/` — 公共基础权重：`LAM_400k.ckpt`（`sha256 d77bf1b3…`）、`CDLAM_official_lam.pt`（`sha256 084f9b1a…`）。
- `lam-dis/` — A 与 B 微调的**共享物理存储**（保持原位；归属为逻辑归属，见各项目 `PROJECT_MANIFEST.md`；config/结果以绝对路径引用）。A 的 arm：`ft_l40_{cdlam_repro,contrastive,full,idm,ours_*}`；B 的 arm：`ft_l40_sb_{A0,A1,B,D,R,S,U}`。
- `lam-egodex-siglip/` — 归 `lam-egodex/`。

**`code/`（13 G）** — 跨项目、单一来源的公共代码：
- `lam/` — 基础 DreamDojo LAM 训练（`lam-agibot`、`lam-egodex` 使用）。
- `adaworld/` ⑂、`sam3/` ⑂ — 第三方 clone（各自有 `.git`；`adaworld` 已 gitignore）。`model.py`/`benchmark_lam.py` 只读 import `adaworld/lam/lam.modules.blocks`。
- `cosmos-predict25/` — 第三方世界模型（供项目 C 的 ACWM）。
> 注意：CD-LAM 的训练/benchmark 底座（`train/model/dataset/eval/benchmark_lam/…`）**不在这里**——它归项目 A，B/C 经文档化的垫片消费；原因是其 `RUN_SPECS` arm 注册表与具体实验强绑定。

## 标准项目结构
```
<project>/
  README.md            快速上手 + 研究问题
  PROJECT_MANIFEST.md  claim、实验、入口、谱系、依赖、封存状态、误用警告
  code/  code/config/  项目代码 + YAML（config 放 code/config 下，以保证 cwd 相对解析可用）
  notes/               研究笔记（索引在 notes/README.md）
  results/  logs/  scripts/  data/  archive/
  checkpoints/         仅逻辑归属——大权重留在共享的 /checkpoints 存储
```
约定：入口一律从项目根目录运行。大 checkpoint/数据在项目间**从不复制**——跨项目使用是
一条只读引用，记录在消费方的 manifest 里。

## 关于本次重整
- 原 `lam_disentangle/` 已删除（原为空壳，仅剩失效的空 `.git`/`.agents`/`.codex` 与一个死 `.gitignore`）。
- Git：本次拆分为**同仓库内**移动，被移动的已跟踪文件在提交时经内容相似度自动识别为 rename
  以保留历史（`git add -A` 会把 delete/add 配成 rename）。
