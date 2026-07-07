# Foreground-Aware LAM 实验报告

## 1. 背景与动机

DreamDojo 的核心组件之一是 Latent Action Model（LAM），一个 700M 参数的 Spatiotemporal Transformer VAE。LAM 的训练目标是：给定相邻两帧 $f_t$ 和 $f_{t+1}$，Encoder 输出 32 维 latent action $\hat{a}_t$；Decoder 仅凭 $f_t$ 和 $\hat{a}_t$ 重建 $f_{t+1}$。Loss 是全图 MSE + β·KL（β=1e-6）。

**问题假设**：LAM 的训练目标没有约束 latent 只编码运动信息。由于全图 MSE 对所有像素一视同仁，encoder 可能走"外观捷径"——把背景纹理、场景信息也编码进 $\hat{a}_t$，而不是专注于运动本身。这对 cross-embodiment transfer 是有害的：迁移到机器人环境时，latent 里的背景信息变成噪声。

**研究问题**：DreamDojo LAM 的 latent action 是否被场景外观信息污染？能否通过前景感知的 loss 设计来改善？

---

## 2. 数据集

**AgiBot-World Alpha**，使用其中两个任务：

| Task ID | 任务描述 | Episodes | 视频时长/ep |
|---|---|---|---|
| 359 | 双臂机器人将食物从托盘转移到蓝色容器（白色桌面） | ~220 | ~40s |
| 362 | 双臂机器人折叠衣物（裤子，灰色布料桌面） | ~220 | ~45s |

- 训练视角：`head_color`（俯视 RGB，640×480，30fps），训练时缩放至 320×240
- 每个 episode 包含 8 路摄像头视频 + depth，约 1200-1600 帧
- 两个 task 外观差异极大（白桌食物 vs 灰布衣物），是外观污染问题的天然对照

---

## 3. 实验设计

### 实验 1：700M Baseline LAM 训练

复现 DreamDojo LAM，使用原始 MSE + KL loss，不加任何前景约束。

**模型配置**（`lam_agibot.yaml`）：

```yaml
model:
  lam_model_dim: 1024
  lam_latent_dim: 32
  lam_enc_blocks: 24
  lam_dec_blocks: 24
  lam_num_heads: 16
  beta: 1.0e-6

training:
  per_gpu_batch_size: 16
  gradient_accumulation_steps: 6   # effective batch ≈ 288
  total_steps: 10000
  lr: 2.5e-5
  warmup_steps: 1000
  mixed_precision: bf16
```

**硬件**：3× NVIDIA RTX 6000 Ada (48GB)，CUDA_VISIBLE_DEVICES=1,2,3，Accelerate 多卡训练

**Checkpoint**：`checkpoints/lam-700m-baseline/step_0010000`

---

### 实验 2：t-SNE 可视化 Baseline Latent

从 validation set 采样 2000 个 latent $z_\mu$（32维），用 t-SNE（perplexity=40）降至 2D，分别用两套 label 染色：

- **左图**：按 task_id 染色 → 测试 latent 是否按动作语义组织
- **右图**：按 episode_id 染色 → 测试 latent 是否编码了场景外观信息

---

### 实验 3：RAFT 光流 Mask 预计算

使用 `torchvision` 内置 RAFT-small 对所有 episode 的相邻帧对预计算光流，生成前景运动 mask。

**设计思路**：离线预计算，训练时直接从磁盘读取，避免在线推理的计算开销。

```
threshold = 1.5 px/frame
mask = (flow_magnitude > threshold)，再做 3×3 max pool 膨胀
保存路径：{ep_dir}/flow_masks/{t:06d}_skip{skip}.pt
```

**实际数据统计**（500 个 mask 抽样）：
- 前景像素占比 mean = 3.4%，max = 18.1%
- p50 = 0%（超过一半的帧机器人完全静止）
- p90 = 12.2%，p99 = 15.8%
- 机械臂在俯视全景中面积占比本身就很小，前景密度天花板约 18%

**总计**：242 万个 mask 文件，约 850MB

---

### 实验 4：Foreground-Aware LAM Fine-tune

在 Baseline checkpoint 基础上继续训练，加入前景加权 loss：

```python
weight = 1.0 + (fg_weight - 1.0) * fg_mask   # 背景=1.0，前景=fg_weight
mse = (weight * (recon - gt)²).mean()
loss = mse + β * KL
```

**配置**：

```yaml
foreground:
  use_fg_loss: true
  fg_weight: 5.0

training:
  total_steps: 20000   # 从 baseline step_0010000 resume，再训 10k 步
  lr: 1.0e-5           # fine-tune 用更小 lr
  warmup_steps: 200
```

**Checkpoint**：`checkpoints/lam-700m-fg/step_0010000`（即总共 20k 步）

---

### 实验 5：Linear Probe 诊断

冻结 encoder，在 $z_\mu$ 上训练两个 Logistic Regression：

- **Task probe**：$z_\mu \rightarrow$ task\_id（2分类，chance=50%）
- **Episode probe**：$z_\mu \rightarrow$ episode\_id（22分类，chance=4.55%）

采样 5000 个 latent，80/20 train/test split。

**解读逻辑**：
- Task probe 高 → latent 编码了动作语义（好）
- Episode probe 高 → latent 同时编码了场景外观（外观污染）
- 理想结果：Task↑，Episode↓

---

### 实验 6：Motion Level 聚类 t-SNE

用 RAFT mask 的前景像素密度（density）作为运动强度指标，把帧分为三级：

| 等级 | Density 范围 | 物理含义 |
|---|---|---|
| Static | < 1% | 机器人停止 / 定位 |
| Small | 1–8% | 精细操作 |
| Large | > 8% | 大幅运动 |

在 t-SNE 上用三种颜色标注，观察 latent 是否按运动强度分离。

---

## 4. 实验结果

### 4.1 Linear Probe 对比

| 指标 | Baseline LAM | Foreground-LAM | 变化 |
|---|---|---|---|
| Task probe (task_id) | **94.4%** | **96.8%** | +2.4% |
| Episode probe (ep_id) | **65.0%** | **65.9%** | +0.9%（反升） |
| Task chance | 50.0% | 50.0% | — |
| Episode chance | 4.55% | 4.55% | — |

### 4.2 t-SNE 可视化（Baseline）

- **左图（by task_id）**：两个 task（蓝/青）严重混杂，无明显边界，latent 对任务类别不敏感
- **右图（by episode_id）**：同一 episode 的点倾向聚集在局部小团簇，说明 latent 编码了 episode 级别的外观信息

### 4.3 t-SNE 可视化（Foreground-LAM）

与 Baseline 相比，结构无明显变化，两图视觉上差异不显著。

### 4.4 Motion Level 聚类（Baseline）

- Static（1184个）、Small（438个）、Large（398个）三级分布较均匀
- t-SNE 上三种运动级别存在一定的空间分离趋势，但混杂明显
- 说明 latent 对运动强度有一定响应，但不是主导的组织维度

---

## 5. 结论与分析

### 5.1 Baseline 诊断结论

Baseline LAM 的 latent action **同时编码了动作信息和场景外观信息**：
- Task probe 94.4%（远高于 50% chance）：动作语义信息确实存在
- Episode probe 65.0%（远高于 4.55% chance，高出 14 倍）：外观污染明确存在

这支持了最初的假设：全图无权重 MSE 让 encoder 走了"外观捷径"。

### 5.2 Foreground-LAM 为什么没效果

**fg_weight=5 的实际影响过小**。

以典型帧为例（前景占 5%，即 3840/76800 像素）：

| | 像素数 | 权重 | 加权像素数 | 占总 loss |
|---|---|---|---|---|
| 背景 | 72960 | 1.0 | 72960 | **80%** |
| 前景 | 3840 | 5.0 | 19200 | **20%** |

背景仍然贡献了 80% 的梯度信号，模型没有足够动力放弃背景的重建。

此外，两个 task 的背景本质上完全不同（白桌 vs 灰布），这个差异通过前景 mask 无法消除——mask 只覆盖机械臂运动区域，不覆盖整个背景。因此即使加权 loss 生效，task 级别的外观差异仍然会污染 latent。

---

## 6. 后续方向

### 方向 A：提高 fg_weight（验证方法本身）

将 fg_weight 提高到 20–50，使前景 loss 占主导，从头 fine-tune 30k 步，真正测试"抑制背景重建"的效果上限。

```
fg_weight=20，前景占 5%：前景贡献 ≈ 50%，背景 ≈ 50%
fg_weight=50，前景占 5%：前景贡献 ≈ 76%，背景 ≈ 24%
```

### 方向 B：单 task 内做对比（排除 task 级外观干扰）

只用 task 359 的 220 个 episode，按 episode 分组做 probe，测试前景 loss 能否降低 episode 级别的外观污染。这样排除了两个 task 背景本来就不同的干扰，更干净地验证方法。

### 方向 C：对比 loss（导师建议方向之一）

用 task 语言描述作为弱监督，构造跨 episode 的正负样本对，做对比学习。同一 task 内不同 episode 的 latent 应该相近，不同 task 之间应该分开。这直接给 latent 提供了动作语义的锚点，是比加权 loss 更强的监督信号。

### 方向 D：Soft mask 替代二值 mask

当前 mask 是二值的（运动/不运动），可以改用光流幅度的连续值作为权重（soft mask），更平滑地区分运动强度，避免阈值敏感性问题。

---

## 7. 文件索引

| 文件 | 说明 |
|---|---|
| `code/lam/train_lam.py` | 训练脚本，含前景加权 loss |
| `code/lam/eval_lam.py` | 评估脚本（reconstruction / visualize / probe / motion 四种模式） |
| `code/lam/precompute_flow_masks.py` | RAFT 离线预计算 mask |
| `code/lam/config/lam_agibot.yaml` | 当前训练配置（fg fine-tune 版本） |
| `checkpoints/lam-700m-baseline/step_0010000` | Baseline 700M checkpoint |
| `checkpoints/lam-700m-fg/step_0010000` | Foreground-LAM checkpoint |
| `results/tsne_baseline.png` | Baseline t-SNE（by task / by episode） |
| `results/tsne_fg.png` | Foreground-LAM t-SNE |
| `results/tsne_motion_baseline.png` | Baseline motion level 聚类 |
| `results/tsne_motion_fg.png` | Foreground-LAM motion level 聚类 |
| `results/probe_baseline.txt` | Baseline linear probe 数字 |
| `results/probe_fg.txt` | Foreground-LAM linear probe 数字 |
