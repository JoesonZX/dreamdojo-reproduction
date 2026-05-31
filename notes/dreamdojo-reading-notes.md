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

**架构**：
- 700M 参数的时空 Transformer（Spatiotemporal Transformer）
- 编码器：输入连续两帧 $f_t, f_{t+1}$，输出 32 维连续潜在动作 $\hat{a}_t$
- 解码器：输入 $\hat{a}_t$ + $f_t$，重建 $f_{t+1}$

**损失函数**：

$$\mathcal{L}^{\text{pred}}_{\theta,\varphi}(f^{t+1}) = \mathbb{E}_{q_\varphi(\hat{a}|f^{t:t+1})} \log p_\theta(f^{t+1}|\hat{a}, f^t) - \beta D_{KL}(q_\varphi(\hat{a}|f^{t:t+1}) \| p(\hat{a}))$$

$\beta = 10^{-6}$，信息瓶颈强迫模型压缩出最关键的动作信息，自然实现跨具身迁移。

**LAM 训练数据**：人类视频（In-lab 55h + EgoDex 829h + DreamDojo-HV 43,827h）**以及**机器人视频（G1、GR-1、AgiBot、YAM）。LAM 不是只用人类视频训练，机器人数据也参与。

**关键发现**：在第一人称人类视频中，这个 embedding 特别能捕捉手部/肢体动作，且不同具身执行相似动作时 latent action 相似（见论文 Figure 3）。

### 2.4 架构改进：动作注入方式

**改进1：相对动作（Relative Actions）**

原始绝对关节姿态 → 以每个 latent frame 开始的姿态为基准做差，转为相对动作。

- 相对动作分布更集中（窄），模型更易学习
- 增强对连续、组合动作的泛化能力

**改进2：Chunked 动作注入**

由于 tokenizer 时序压缩比为 4，将 4 个连续动作 $a_{t:t+4}$ 拼接成一个 chunk，注入对应的 latent frame。

- 满足因果性：当前帧只看当前动作，不看未来动作
- 显著减少因果混淆（causality confusion），提升学习效率

**动作条件注入方式**：将动作通过轻量 MLP 投影到与 timestep embedding 相同的维度，叠加后输入 DiT block 的自适应层归一化（AdaLN）。MLP 最后一层初始化为全零（zero initialization）以稳定早期训练。

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
