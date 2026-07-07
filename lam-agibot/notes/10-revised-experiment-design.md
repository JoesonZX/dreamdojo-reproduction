# 修订实验设计：Foreground-Aware LAM with SAM3 + SigLIP

## 核心研究问题

DreamDojo LAM 的 latent action 是否被场景外观污染？能否通过前景感知学习得到更纯净的动作表示？

---

## 实验总览

```
阶段 A：诊断（已完成 AgiBot，进行中 EgoDex）
  └─ 证明问题存在：baseline latent 同时编码动作和外观

阶段 B：改进方案一 —— SAM3 前景感知（等待 SAM3 权限）
  └─ 用 SAM3 生成语义前景 mask，替代 RAFT 光流 mask

阶段 C：改进方案二 —— SigLIP 对比 loss（核心贡献）
  └─ 用 caption 作为语义锚点，训练动作纯净的 latent
```

---

## 数据集选择

**主数据集：EgoDex test split**

| 属性 | 内容 |
|---|---|
| 规模 | 111 个 task，3243 个 episode，约 7 小时 |
| 动作标签 | task 文件夹名（111 种，`basic_pick_place`、`fold`、`boil_serve_egg` 等）|
| Caption | 每个 episode 的 HDF5 有 `llm_description`（GPT-4 生成）|
| 背景多样性 | 高（不同桌面、不同背景颜色、不同物体）|
| 为什么合适 | task = 动作类别，背景多样 → task probe 高才真正说明动作被编码 |

**对照数据集：AgiBot（已有结果，用于对比）**

导师说 AgiBot 也可以做动作聚类，前提是拿到 caption。需要从 HuggingFace 下载 metadata：

```bash
huggingface-cli download agibot-world/AgiBotWorld-Alpha \
    --include "metadata/*" \
    --local-dir /home/xuan/embodied-ai/data/agibotworld/metadata
```

---

## 阶段 A：诊断实验（Baseline）

### A1. EgoDex Baseline LAM 训练

**状态**：进行中（step 4000/10000）

配置：`code/lam/config/lam_egodex.yaml`
- 700M，10k steps，lr=2.5e-5，EgoDex test split

### A2. t-SNE 可视化

```bash
CUDA_VISIBLE_DEVICES=0 python code/lam/eval_lam.py \
    --checkpoint checkpoints/lam-egodex-baseline/step_0010000 \
    --config code/lam/config/lam_egodex.yaml \
    --mode visualize \
    --n_samples 2000 \
    --out_path results/tsne_egodex_baseline.png \
    --label "EgoDex Baseline (10k steps)"
```

**预期**：
- 左图（by task_id）：111 种颜色有一定聚类 → 说明动作信息存在
- 右图（by ep_id）：同 episode 内点聚集 → 说明外观污染存在

### A3. Linear Probe

```bash
CUDA_VISIBLE_DEVICES=0 python code/lam/eval_lam.py \
    --checkpoint checkpoints/lam-egodex-baseline/step_0010000 \
    --config code/lam/config/lam_egodex.yaml \
    --mode probe \
    --n_samples 5000 \
    --out_path results/probe_egodex_baseline.txt \
    --label "EgoDex Baseline (10k steps)"
```

**关键指标**：
- Task probe（111 分类，chance=0.9%）：预期 >50%
- Episode probe（chance 极低）：如果也很高 → 外观污染确认

---

## 阶段 B：SAM3 前景感知（等待 HF 权限）

### 为什么换 SAM3 而不是 RAFT

| | RAFT | SAM3 |
|---|---|---|
| 原理 | 光流（基于像素运动） | 语义分割（基于物体理解）|
| 提示方式 | 无，纯计算 | 文本提示（"hand", "robot arm"）|
| 静止时 | mask 为空（手臂停止时失效）| 持续追踪，不依赖运动 |
| 语义质量 | 低（运动噪声、背景抖动也会触发）| 高（真正的前景物体）|

EgoDex 是第一人称手部操作，SAM3 可以用 `"hands"` 作为文本提示直接追踪。

### B1. 安装 SAM3（权限通过后）

```bash
conda create -n sam3 python=3.12
conda activate sam3
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/facebookresearch/sam3.git code/sam3
cd code/sam3 && pip install -e .
huggingface-cli login
```

### B2. 生成 mask 图看效果

对 EgoDex 几个 task 各取 1 个 episode，用文本提示 `"hands"` 生成前景 mask，保存对比图：
- 原始帧 | SAM3 mask | mask 叠加可视化

脚本：`code/lam/visualize_sam3_masks.py`（待实现）

### B3. 批量预计算 SAM3 mask

对所有 EgoDex episodes 预计算，存储格式与 RAFT mask 兼容：
```
{task_name}/flow_masks/{idx}/{t:06d}_skip{skip}.pt
```

### B4. Foreground-LAM fine-tune（SAM3 mask 版本）

从 EgoDex baseline checkpoint fine-tune，`fg_weight=20`（SAM3 mask 质量高，可以用更大权重）。

---

## 阶段 C：SigLIP 对比 Loss（核心贡献）

### 核心思路

EgoDex 每个 episode 有 `llm_description`（例如 "Pick up a black stapler from the metal table and place it in the box lid."）。

用 SigLIP 的文本编码器将 description 编码为文本向量，作为 latent action 的语义锚点：

```
正样本对：同一 description 的两个 episode 的 z_mu 应该相近
负样本对：不同 description 的 z_mu 应该分开
```

Loss 设计：

```
L = L_recon + β·L_KL + λ·L_SigLIP
```

其中：

```python
# z_mu: [B, 32]  text_emb: [B, D_text]
# 用一个小 projection head 把 z_mu 投影到文本空间
proj_z = projection_head(z_mu)          # [B, D_text]
L_SigLIP = siglip_loss(proj_z, text_emb)
```

### SigLIP vs CLIP 的区别

SigLIP 用 sigmoid loss 替代 softmax，在小 batch 下更稳定，适合你的实验规模（batch=288）。

### C1. 准备 caption 编码

预计算所有 EgoDex episode 的 text embedding，存为 `.pt` 文件：

```python
from transformers import AutoTokenizer, AutoModel
# google/siglip-base-patch16-224 的文本编码器
```

### C2. 修改 dataset

`EgoDexDataset.__getitem__` 加载对应 episode 的 text embedding。

### C3. 修改 loss

`train_lam.py` 加 SigLIP contrastive loss，在 LAM loss 基础上叠加。

### C4. 训练 SigLIP-LAM

从 EgoDex baseline checkpoint fine-tune，观察 task probe 和 episode probe 的变化。

---

## 对比实验矩阵

| 模型 | 数据 | 前景方案 | 对比 Loss | Task probe | Ep probe |
|---|---|---|---|---|---|
| Baseline | EgoDex | 无 | 无 | ? | ? |
| RAFT-LAM | EgoDex | RAFT fg_weight=5 | 无 | ? | ? |
| SAM3-LAM | EgoDex | SAM3 fg_weight=20 | 无 | ? | ? |
| SigLIP-LAM | EgoDex | 无 | SigLIP caption | ? | ? |
| SAM3+SigLIP-LAM | EgoDex | SAM3 | SigLIP caption | ? | ? |

理想结果：右边两列，Task↑，Episode↓

---

## 当前任务优先级

| 优先级 | 任务 | 状态 | 备注 |
|---|---|---|---|
| 🔴 进行中 | EgoDex baseline 训练 | step 4000/10000 | 等待完成 |
| 🟡 等待 | SAM3 HF 权限 | 已申请 | 通常数小时到数天 |
| 🟢 可以现在做 | 下载 AgiBot metadata | 未开始 | 验证 caption 可用性 |
| 🟢 可以现在做 | 预计算 EgoDex caption embedding | 未开始 | 不依赖 SAM3 |
| 🟢 可以现在做 | 实现 SigLIP loss 代码 | 未开始 | 不依赖 SAM3 |

---

## 文件索引

| 文件 | 状态 |
|---|---|
| `code/lam/config/lam_egodex.yaml` | 已有 |
| `code/lam/data/egodex_dataset.py` | 已有 |
| `code/lam/eval_lam.py` | 已有（支持 egodex）|
| `code/lam/precompute_flow_masks.py` | 已有（支持 egodex + CPU 模式）|
| `code/lam/visualize_sam3_masks.py` | 待实现（等 SAM3 权限）|
| `code/lam/precompute_sam3_masks.py` | 待实现（等 SAM3 权限）|
| `code/lam/precompute_captions.py` | 待实现（可以现在做）|
| `code/lam/config/lam_egodex_siglip.yaml` | 待实现（可以现在做）|
