# DreamDojo 阅读笔记

**论文**：DreamDojo: A Generalist Robot World Model from Large-Scale Human Videos  
**来源**：arXiv:2602.06949v1，2026年2月  
**机构**：NVIDIA、UC Berkeley、HKUST 等  

---

## 1. 问题与创新方案概述

### 解决的问题

现有机器人世界模型存在三个核心瓶颈：

1. **数据分布窄**：机器人数据采集成本高，硬件差异大，覆盖的场景/技能远不足以支撑泛化。现有模型只能在分布内（in-distribution）场景下工作，遇到新物体、新环境就失效。
2. **动作标签稀缺**：大规模视频没有精细的动作标注，无法直接用于训练 action-conditioned 世界模型。
3. **无法实时运行**：现有视频扩散模型推理慢（需要 50 步去噪），且双向注意力机制导致生成长度固定，无法支持 live teleoperation 等在线应用。

### 创新方案

DreamDojo 提出三项核心创新：

| 创新点 | 内容 |
|--------|------|
| **大规模人类视频数据集 DreamDojo-HV** | 44k 小时第一人称视频，规模和多样性超越以往数据集 15×（时长）、96×（技能数）、2000×（场景数） |
| **连续潜在动作（Continuous Latent Actions）作为统一代理标签** | 用自监督方式从无标注视频中提取跨具身（cross-embodiment）的动作表示，解决动作标签稀缺问题 |
| **Self Forcing 蒸馏流水线** | 将双向注意力模型蒸馏为因果自回归模型，推理速度提升 4×，达到 10.81 FPS 实时生成 |

---

## 2. 方案具体细节与原理

### 2.1 整体训练流程（三阶段）

```
阶段1: 人类视频预训练（Human Video Pretraining）
  └─ 数据: In-lab + EgoDex + DreamDojo-HV（共 44,711 小时）
  └─ 条件: Continuous Latent Actions 作为代理动作

阶段2: 目标机器人后训练（Robot Post-Training）
  └─ 数据: 目标机器人少量数据（G1 / GR-1 / AgiBot / YAM）
  └─ 重新初始化 action MLP 第一层，全量微调

阶段3: 自回归蒸馏（Autoregressive Distillation）
  └─ Teacher → Student（双向注意力 → 因果注意力）
  └─ 两阶段: Warmup + Distribution Matching
```

### 2.2 基础模型：Cosmos-Predict2.5

- 基于潜在视频扩散模型（Latent Video Diffusion Model）
- 使用 WAN2.2 tokenizer，时序压缩比为 4（每个 latent frame 对应 4 个像素帧）
- 架构为 DiT（Diffusion Transformer），用 flow matching 损失训练：

$$\mathcal{L}_{\text{flow}}(\theta) = \mathbb{E}_{x,\epsilon,c,t} \| u(x_t, t, c; \theta) - v_t \|^2$$

其中 $v_t = \epsilon - x$ 是目标速度，$c$ 包含文本、条件帧和动作条件。

### 2.3 潜在动作模型（Latent Action Model）

**核心思想**：用一个信息瓶颈 VAE 从相邻帧对中提取"动作"，不需要任何人工标注。

#### 2.3.1 整体架构

700M 参数的 Spatiotemporal Transformer VAE，分为编码器和解码器两部分：

```
输入: [f_t, f_{t+1}]（连续两帧，各 320×240）
           │
    ┌──────┴──────────────────────────────────┐
    │           编码器（24层 ST-Transformer）   │
    │                                          │
    │  1. 把两帧都切成 16×16 的 patch           │
    │     每帧: 15×20 = 300 个 patch            │
    │                                          │
    │  2. 在每帧的 patch 序列前加一个            │
    │     可学习的 action_prompt token          │
    │     → [action | p1 | p2 | ... | p300]    │
    │                                          │
    │  3. Spatiotemporal Attention：            │
    │     - 空间 attention：同一帧内的 patch 互看  │
    │     - 时间 attention：同位置 patch 跨帧互看  │
    │                                          │
    │  4. 只取第2帧（f_{t+1}）的 action token    │
    │     → Linear → [μ, log σ²]（各32维）      │
    └──────────────────────────────────────────┘
                    │
                    │ 重参数化：â_t = μ + σ·ε（ε~N(0,I)）
                    ↓
              â_t（32维 latent action）
                    │
    ┌───────────────┴──────────────────────────┐
    │           解码器（24层 SpatioTransformer）  │
    │                                          │
    │  1. 只输入 f_t 的 patch（不含 f_{t+1}）   │
    │     → Linear 投影到 model_dim            │
    │                                          │
    │  2. â_t → Linear 投影到 model_dim         │
    │     与 f_t patches 相加（加法融合）        │
    │                                          │
    │  3. Spatial Attention（只做空间，不跨帧）  │
    │                                          │
    │  4. 反 patchify → 重建 f_{t+1}            │
    │     → Sigmoid 归一化到 [0,1]             │
    └──────────────────────────────────────────┘
                    │
              重建的 f_{t+1}
```

**关键设计：为什么 action token 能捕获"动作"？**

编码器同时看到 $f_t$ 和 $f_{t+1}$，两帧之间的差异通过 temporal attention 传递到 action token。解码器只拿到 $f_t$，它需要依赖 $\hat{a}_t$ 才能重建出 $f_{t+1}$——因此 $\hat{a}_t$ 被迫编码所有"从 $f_t$ 变化到 $f_{t+1}$ 所需的信息"，即动作。

**信息瓶颈的作用**：32 维 + KL 惩罚（β=1e-6）强迫模型丢弃无关信息（背景、光照），只压缩运动相关的核心信息。β 极小意味着约束很宽松，允许模型保留较多信息，但"变化到下一帧"这个任务自然引导模型关注运动。

#### 2.3.2 与 AdaWorld 实现的对应关系

DreamDojo 论文的 LAM 架构与 AdaWorld（ICML 2025）一致，AdaWorld 代码已开源。实际实现中（见 `adaworld/lam/lam/modules/lam.py`）：

- 编码器用 `SpatioTemporalTransformer`（同时做空间和时间 attention）
- 解码器用 `SpatioTransformer`（只做空间 attention，因为解码时只有单帧）
- action token 通过 `nn.Parameter` 定义为可学习的 action_prompt
- 解码时用**加法融合**：`video_patches + action_patches`（而非 cross-attention）

论文用 700M（24层×2），AdaWorld 默认配置是 60M（8层×2）。我们使用 AdaWorld 的小模型进行复现。

#### 2.3.3 损失函数详解

$$\mathcal{L} = \underbrace{\|f_{t+1} - \hat{f}_{t+1}\|^2}_{\text{重建损失（MSE）}} + \beta \underbrace{D_{KL}(q(\hat{a}|f^{t:t+1}) \| \mathcal{N}(0,I))}_{\text{KL 散度}}$$

KL 散度展开为：
$$D_{KL} = -\frac{1}{2}\sum_{j=1}^{32}(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2)$$

- $\beta = 10^{-6}$：极小，KL 约束很弱，主要靠重建损失驱动
- 训练时采样（随机性），推理时只用 μ（确定性）

**LAM 训练数据**：人类视频（In-lab 55h + EgoDex 829h + DreamDojo-HV 43,827h）**以及**机器人视频（G1、GR-1、AgiBot、YAM）。LAM 不只用人类视频，机器人数据也参与。

**关键发现**：不同具身执行相似动作时 latent action 相似（见论文 Figure 3），说明信息瓶颈成功剥离了具身外观信息。

### 2.4 世界模型的动作注入与初始化

#### 2.4.1 动作如何注入 Cosmos-Predict2.5

Cosmos-Predict2.5 是 DiT（Diffusion Transformer）架构，每个 DiT block 通过 AdaLN（Adaptive Layer Normalization）接受条件信号。

动作注入的路径：

```
â_t 或真实关节角度
        │
        │ Action MLP（轻量3层 MLP）
        ↓
action_embed（与 timestep_embed 同维度）
        │
        + （加法，不是 concat）
        │
timestep_embed + action_embed
        │
        ↓
      AdaLN
  ┌────┴────┐
scale    shift     → 调制每个 DiT block 的 LayerNorm
```

动作和时间步共享同一个 AdaLN 入口，而文本通过独立的 cross-attention 注入。这意味着动作对模型的影响方式和"当前去噪步骤"是类似的——都是全局调制整个特征图的尺度和偏移。

#### 2.4.2 Action MLP 的两阶段初始化策略

这是 DreamDojo 的一个关键工程细节：

**预训练阶段（用 latent action）**：
- MLP 最后一层权重和偏置全部初始化为 **0**
- 效果：训练开始时 action_embed = 0，对模型的扰动为零
- 原因：Cosmos-Predict2.5 已经预训练好了，如果一开始 action 注入随机噪声，会破坏预训练的物理知识，导致训练不稳定
- 随着训练进行，MLP 逐渐学会产生有意义的 action_embed

**Post-Training 阶段（换用真实关节角度）**：
- MLP **第一层**重新初始化（随机初始化）
- MLP **最后一层**保留零初始化
- 原因：从 32 维 latent action 切换到 56 维真实关节角度（4 chunk × 14 DoF），输入维度完全不同，第一层必须重建映射；但最后一层仍保持零初始化以稳定切换初期的训练

```python
# 预训练时
nn.init.zeros_(action_mlp[-1].weight)
nn.init.zeros_(action_mlp[-1].bias)

# Post-Training 时
nn.init.kaiming_normal_(action_mlp[0].weight)  # 第一层重新初始化
nn.init.zeros_(action_mlp[-1].weight)           # 最后一层仍为零
nn.init.zeros_(action_mlp[-1].bias)
```

#### 2.4.3 Chunked 动作注入

WAN2.1 tokenizer（Cosmos-Predict2.5 使用，注：原笔记写 WAN2.2，实为 WAN2.1）时序压缩比为 4：每个 latent frame $x^i$ 对应像素空间的 4 帧 $f_{4i:4i+4}$。

动作注入需要与 latent frame 对齐：
- 将 4 个连续原始动作 $a_{4i:4i+4}$ 拼接成一个 chunk
- 每个 latent frame 注入对应的 action chunk
- 这样 latent frame 只看"自己对应时间段"的动作，不跨帧

**为什么不能全局广播一个动作？** 如果把序列平均动作注入所有 latent frame，模型无法区分"哪个时间段发生了什么"，导致因果混淆（causality confusion）——模型分不清是哪个动作导致了哪个帧的变化。

#### 2.4.4 相对动作（Relative Actions）

原始数据是绝对关节角度（从零点量起的绝对值），训练时转换为相对动作：

```python
# 每 4 个时间步为一个 chunk，以 chunk 起始姿态为基准
for i in range(0, T, 4):
    baseline = abs_actions[i]          # chunk 起点的绝对姿态
    chunk = abs_actions[i:i+4] - baseline  # 相对位移
    rel_actions.append(chunk.flatten())    # [4 * 14DoF = 56维]
```

相对动作的分布比绝对动作窄得多（绝对值可能跨越整个关节范围，相对值集中在小范围内），更易于 MLP 学习。

### 2.5 训练目标：时序一致性损失

除标准 flow matching 损失外，增加时序一致性损失：

$$\mathcal{L}_{\text{temporal}}(\theta) = \mathbb{E}\left[\sum_{i=1}^{K-1} \|(z^{i+1} - z^i) - (v^{i+1} - v^i)\|^2\right]$$

最终损失：$\mathcal{L}_{\text{final}} = \mathcal{L}_{\text{flow}} + \lambda \mathcal{L}_{\text{temporal}}$，其中 $\lambda = 0.1$。

作用：约束相邻帧的速度场变化与真实值匹配，显著提升 action following 和物体完整性。

### 2.6 蒸馏流水线（Self Forcing）

**目标**：将 Teacher（双向注意力，35步去噪）蒸馏为 Student（因果注意力，4步去噪）。

**Warmup 阶段**：用 Teacher 的 ODE 轨迹监督 Student 回归：
$$\mathcal{L}_{\text{warmup}} = \mathbb{E}_{x,t} \|G_{\text{student}}(x_t, t) - x_0\|^2$$
Student 用 teacher forcing（上下文来自 Teacher 生成的帧）。

**Distillation 阶段**：Student 上下文改为自己生成的历史帧，通过 KL 散度分布匹配损失对齐 Teacher 分布：
$$\mathcal{L}_{\text{distill}} = D_{KL}(p_{\text{teacher}} \| p_{\text{student}})$$

**长程稳定性技巧**：训练时让 Student 生成 $N' > N$ 帧（13~49帧），但只在最后 N 帧上计算损失，模拟更长的自回归 rollout，减少训练-推理分布偏差（compounding error）。

---

## 3. 实验设计

### 3.1 评测基准（6个OOD benchmark）

所有 benchmark 均为机器人执行与人类视频中相似物理交互的场景，但相对机器人训练数据是 OOD：

| Benchmark | 说明 |
|-----------|------|
| In-lab Eval | 实验室场景，新物体/动作 |
| EgoDex Eval | Apple Vision Pro 采集的灵巧操作 |
| DreamDojo-HV Eval | 众包采集的多样日常场景 |
| Counterfactual Eval | 不在机器人数据集中的反事实动作（如轻拍玩具、伸手够但没拿到） |
| EgoDex-novel Eval | EgoDex + Gemini 2.5 生成的新背景（无 GT，人类偏好评测） |
| DreamDojo-HV-novel Eval | DreamDojo-HV + 新背景（无 GT，人类偏好评测） |

主体机器人：Fourier GR-1 人形机器人。

### 3.2 评测指标

- **自动指标**：PSNR（越高越好）、SSIM（越高越好）、LPIPS（越低越好）
- **人类偏好评测**：12名评测者通过 Web UI 对比两个模型生成的视频，从"物理正确性"和"动作跟随"两个维度选择更好的一方

### 3.3 消融实验设计

| 实验 | 目的 |
|------|------|
| 不同动作条件对比（无预训练 / action-free / latent action / GT action） | 验证 latent action 的有效性（Sec 4.2） |
| 不同数据组合（逐步添加 In-lab、EgoDex、DreamDojo-HV） | 验证数据多样性的价值（Sec 4.3） |
| 不同模型大小（2B vs 14B）× 不同数据配置 | 验证泛化到未见场景（Sec 4.4） |
| 逐步添加 relative / chunked / temporal loss | 验证每个架构设计贡献（Sec 4.5） |
| Teacher vs Student 长时 rollout | 验证蒸馏效果（Sec 4.6） |

### 3.4 下游应用验证

- **Policy Evaluation**：AgiBot 水果打包任务，20 个场景，与真实世界 success rate 做 Pearson 相关性
- **Model-based Planning**：10 个场景，5 个策略 checkpoint 集成，DINOv2-based value model 选最优动作
- **Live Teleoperation**：PICO VR 控制器 + RTX 5090，实时遥操作虚拟 G1 机器人

---

## 4. 实验结果

### 4.1 Latent Action vs. 其他条件方式（Table 2）

| 方法 | In-lab PSNR | EgoDex PSNR |
|------|-------------|-------------|
| 无预训练 | 20.576 | 19.952 |
| Action-free 预训练 | 20.797 | 19.924 |
| **Latent Action 预训练** | **20.913** | **20.344** |
| GT Action（需要额外设备） | 20.960 | 20.474 |

**结论**：Latent action 预训练显著优于 action-free，性能接近使用专业设备采集的 GT action，且无需任何标注设备。

### 4.2 数据混合消融（Table 3）

逐步加入 In-lab → EgoDex → DreamDojo-HV，所有 4 个 benchmark 指标持续提升。最终 DreamDojo-14B 在所有设置中表现最优。

### 4.3 OOD 泛化（Table 4，人类偏好）

- DreamDojo-2B > Cosmos-Predict2.5：物理正确性 62.5%，动作跟随 63.5%
- DreamDojo-14B > Cosmos-Predict2.5：物理正确性 73.5%，动作跟随 72.6%
- DreamDojo-14B > DreamDojo-2B：物理正确性 72.5%，动作跟随 65.5%

### 4.4 架构设计消融（Table 5）

| 配置 | GR-1 Val PSNR | Counterfactual PSNR |
|------|--------------|---------------------|
| baseline | 16.199 | 19.448 |
| +relative | 16.522 | 19.482 |
| +relative +chunked | 17.626 | 20.783 |
| +relative +chunked +temporal | **17.630** | **20.980** |

每个改进都有贡献，chunked 注入贡献最大。

### 4.5 蒸馏结果（Table 6）

| 模型 | PSNR | FPS | 预测帧数 | 上下文帧数 |
|------|------|-----|---------|-----------|
| Teacher | 14.086 | 2.72 | 12 | 1 |
| Student | 13.146 | 10.81 | 4 | 12 |

速度提升 **4×**（2.72 → 10.81 FPS），质量轻微下降，且 Student 由于有 12 帧上下文，在遮挡和相机移动时表现更稳定。

### 4.6 下游应用结果

- **Policy Evaluation**：Pearson r = 0.995，MMRV = 0.003，与真实世界 success rate 高度相关。
- **Model-based Planning**：相比均匀采样，使用 DreamDojo 进行 test-time planning 成功率提升约 2×（对高方差策略组最高提升 17%）。

---

## 5. 遗留问题（论文明确指出的 Limitations）

1. **非常规动作生成质量差**：对 slapping（击打）、fast waving（快速挥手）等不常见动作效果不佳，训练数据分布覆盖不足。
2. **Policy Evaluation 的绝对成功率偏高**：DreamDojo 生成的 rollout 中失败场景的绝对成功率通常高于真实值，说明模型在生成精细失败场景上有局限，不能精确区分"刚好失败"和"明显成功"。
3. **推理速度还有提升空间**：10.81 FPS 已满足基本实时，但通过进一步工程优化（量化、编译等）理论上可继续提升。
4. **不支持多视角（Multi-view）生成**：当前模型只能生成单视角视频，而 state-of-the-art 机器人策略（如 GR00T N1.5）通常需要多摄像头输入。
5. **预训练知识遗忘问题未深入研究**：后训练（post-training）阶段如何在学习新机器人动作空间的同时保留预训练的通用物理知识，没有系统研究（作者建议探索 LoRA 等参数高效微调策略）。
6. **动作分布覆盖仍不完整**：建议未来引入策略 rollout 数据来覆盖更广泛的动作分布。

---

## 6. 复现关键点

### 6.1 数据

| 数据集 | 类型 | 规模 | 是否公开 |
|--------|------|------|---------|
| EgoDex | 人类（Apple Vision Pro） | 829h | 公开 |
| In-lab | 人类（Manus 手套 + Vive Tracker） | 55h | 未公开 |
| DreamDojo-HV | 人类（众包） | 43,827h | 未公开 |
| AgiBot-World | 机器人（LAM训练 + 后训练） | 2.9k h / 1M 轨迹 | 公开（alpha版）|
| DROID | 机器人 | 350h | 公开 |

> ⚠️ **重要说明**：论文中的 AgiBot 数据是作者内部采集的 in-house 数据，与公开的 AgiBot-World Alpha 数据集不完全相同。公开 Alpha 版在论文 Table 1 中仅作为对比 baseline 引用。

**复现可行数据组合**：EgoDex（公开，用于 LAM）+ AgiBot-World alpha（公开，用于 LAM 和后训练）。
- LAM 训练：AgiBot 视频即可（论文也用了机器人视频），没有 DreamDojo-HV 会影响跨具身泛化，但不影响 Post-Training 效果。

### 6.2 模型组件

| 组件 | 规模 | 是否公开 |
|------|------|---------|
| Cosmos-Predict2.5（基础模型） | 2B / 14B DiT | 公开（NVIDIA） |
| WAN2.2 tokenizer | - | 公开 |
| 潜在动作模型 | 700M Spatiotemporal Transformer | 未公开（需自实现）|
| DreamDojo 世界模型 | 2B / 14B | 未公开 |

### 6.3 超参数

| 参数 | 值 |
|------|----|
| 潜在动作维度 | 32 |
| 潜在动作模型架构 | 700M Transformer，24层编码器 + 24层解码器 |
| LAM 训练步数 | 400k steps，batch=256 |
| LAM 学习率 | 2.5e-5（AdamW，weight_decay=0.01） |
| LAM 的 KL 系数 β | 1e-6 |
| 视频分辨率（LAM） | 320×240 |
| 视频分辨率（世界模型） | 640×480 |
| 序列长度 | 13 帧（1帧条件 + 12帧预测） |
| 时序降采样因子 | {1, 2, 3, 4} 随机 |
| 世界模型预训练步数 | 140k steps，batch=1024，256× H100 |
| 世界模型学习率 | 1.6e-4（AdamW，weight_decay=0.1） |
| 时序一致性损失权重 λ | 0.1 |
| 后训练步数 | 50k steps，batch=512，128× H100 |
| 采样频率（后训练） | ~10 Hz |
| 蒸馏 Warmup：ODE 轨迹数 | 10k |
| 蒸馏 Warmup：迭代步数 | 10k |
| 蒸馏 Distillation 步数 | 3k |
| 推理去噪步数（Teacher） | 35 |
| 推理去噪步数（Student） | 4 |
| Student 上下文帧数 | 12 |

### 6.4 代码与权重开源情况

- **Cosmos-Predict2.5**：已开源（NVIDIA GitHub）
- **WAN2.2 tokenizer**：已开源
- **DreamDojo 完整模型/权重**：**未开源**（截至论文发布时）
- **潜在动作模型代码**：未开源（参考 Genie 2024 的 Spatiotemporal Transformer 实现）
- **项目主页**：dreamdojo-world.github.io（含演示视频）

---

## 7. 可扩展方向

基于遗留问题和论文分析，以下方向值得继续探索：

### 7.1 数据层面
- **引入机器人策略 rollout 数据**：论文明确指出未来应使用 policy rollout 扩展动作分布，可以从 AgiBot-World 或自采数据中生成
- **数据配比优化**：论文用 In-lab:EgoDex:DreamDojo-HV = 1:2:10 的采样比，这个比例对不同规模的公开数据集效果如何值得实验
- **AgiBot-World 作为机器人后训练数据**：alpha 版本有 1M 条轨迹、87 种技能，比 DROID 更大，适合后训练

### 7.2 模型层面
- **参数高效微调（PEFT）**：论文作者建议用 LoRA 等方法在后训练阶段减少预训练知识遗忘，这是一个明确的改进方向
- **多视角世界模型**：扩展到多摄像头输入/输出，支持当前 SOTA 策略（如 GR00T N1.5、π0）的多视角需求
- **长时一致性改进**：当前 Student 1分钟后开始退化，长时序建模仍有提升空间
- **Failure-aware 生成**：改进模型对失败场景的精细建模，提升 policy evaluation 绝对成功率准确性

### 7.3 应用层面
- **强化学习环境**：将 DreamDojo 作为 RL 的虚拟环境，结合 WMPO / World4RL 等方法训练策略
- **数据增强**：利用世界模型生成反事实轨迹（counterfactual trajectories）作为数据增强，提升策略的鲁棒性
- **Test-time Scaling**：进一步研究 model-based planning 中 value model 设计，论文中的 DINOv2-based value model 是一个简单基线

---

*笔记整理时间：2026-05-27*
