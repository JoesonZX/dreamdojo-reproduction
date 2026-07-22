# Motus 精读笔记

> **论文**：*Motus: A Unified Latent Action World Model*
> **作者/机构**：Hongzhe Bi, Hengkai Tan, Shenghao Xie 等（清华 THBI/BNRist + 北大 + 地平线 Horizon Robotics），2025-12（arXiv 2512.13030v2）
> **链接**：arXiv 2512.13030 ｜ 项目页 https://motus-robotics.github.io/motus
> **配套笔记**：见同目录 `fast_wam_notes.md`（Fast-WAM 把 Motus 当基线，二者对照阅读很有价值）
>
> 一句话总结：Motus 用一个 **MoT 三专家（视频生成 + 动作 + VLM 理解）+ UniDiffuser 式调度器** 的统一架构，把具身智能的 **5 种建模分布（VLA / WM / IDM / VGM / 视频-动作联合）统一进一个模型**；并用 **光流驱动的"像素级 delta action"latent action** 在海量无标签视频上做**大规模动作预训练**，三阶段训练后在 RoboTwin 上比 π0.5 提升 +45%、真实场景 +11~48%。**这就是"在世界模型上用 LAM 预训练"的直接范本。**

---

## 一、研究背景与动机

### 1. 问题：能力被割裂在孤立模型里
一个通用具身智能体需要统一多种认知能力——理解场景/指令、想象未来、预测后果、生成动作。但现有方法把这些能力**孤立开**：
- 有的用 **VLA**（从视觉语言学静态策略）；
- 有的用 **世界模型 / 视频生成模型（VGM）**（基于预测未来）；
- 像 F1[32] 用 IDM 把 VLA 与逆动力学结合，但**排除了世界模型/VGM**，统一不完整。

作者指出这应统一为一个系统里的 **5 种建模任务**：
- **VLA**：`p(a_{t+1:t+k} | o_t, ℓ)`
- **WM（世界模型）**：`p(o_{t+1:t+k} | o_t, a_{t+1:t+k})`
- **IDM（逆动力学）**：`p(a_{t+1:t+k} | o_{t:t+k})`
- **VGM（视频生成）**：`p(o_{t+1:t+k} | o_t, ℓ)`
- **视频-动作联合预测**：`p(o_{t+1:t+k}, a_{t+1:t+k} | o_t, ℓ)`

### 2. 两大挑战
- **挑战 1：统一多模态生成能力。** 现有统一世界模型（UWM[64]）多为从头训练或小模型，缺少 VLM 的视觉语言先验 **或** VGM 的物理交互先验,拿不到完整世界知识。
- **挑战 2：利用异构数据。** 不同机器人的动作空间在维度/范围/语义上差异巨大,策略难学通用先验;而互联网视频、第一人称人类视频**缺动作标签**,无法直接用来预训练动作专家。→ 这正是 latent action 要解决的核心痛点。

---

## 二、核心思路

1. **用一个通用生成模型统一 5 种分布**，靠**复用预训练基础模型的先验**（VGM + VLM）避免"需要大量对齐多模态数据"的不现实要求。
2. **引入 latent action** 来吃下海量**无动作标签**数据（互联网视频、人类视频、多机器人轨迹），把动作预训练规模化。
3. 用 **UniDiffuser 式调度器**给不同模态（视频/动作）分配不同的 rectified-flow 时间步与噪声尺度,从而能在推理时**自适应切换**到 VLA / WM / IDM / VGM / 联合预测等模式。

---

## 三、创新点

1. **统一具身基础模型**：一个模型集成 5 大主流范式（VLA、WM、IDM、VGM、视频-动作联合预测），且**不牺牲通用多模态先验**。
2. **可扩展的机器人训练配方**：三阶段训练管线 + 六层数据金字塔,用**基于光流的 latent action** 学跨本体（cross-embodiment）可迁移的运动知识。
3. **大幅超越 SOTA**：仿真中比 X-VLA +15%、比 π0.5 +45%;真实场景 +11~48%。证明大规模通用先验 + 领域特定先验的融合能显著增强策略学习。

---

## 四、方法与架构细节

### 1. Tri-model Joint Attention（MoT 三专家，Figure 1）
三个专家各自保留独立 Transformer 模块,但把各自的**多头自注意力层拼接在一起**做联合注意(即 Tri-model Joint Attention)——既保留各专家专长、避免任务干扰,又实现跨模态特征融合:
- **Video Generation Model（生成专家）**：用 **Wan2.2 5B** 作视频基础模型;每个 block 含 AdaLN(注入 rectified-flow 时间步)+ FFN + Cross-Attn。
- **Action Expert（动作专家）**：与 Wan 同深度的 Transformer block(AdaLN + FFN + 联合注意)。
- **Understanding Expert（理解专家）**：用 **Qwen3-VL-2B**(强 3D grounding / 空间理解 / 物体定位),输入取其最后一层对应 token。

### 2. 联合流匹配目标(rectified flow)
训练时联合预测视频 chunk 与动作 chunk:
- 动作损失 `l_action = E‖v_a^θ − (ε_a − a_{t+1:t+k})‖²`
- 观测损失 `l_obs = E‖v_o^θ − (ε_o − o_{t+1:t+k})‖²`
- 总:`l = l_action + l_obs`
- 给视频和动作**分配不同时间步 τ_v / τ_a 与噪声**,构成 UniDiffuser 式调度器,推理时可切换建模模式。

### 3. Action-Dense Video-Sparse Prediction(Figure 2)
问题:视频 token 数量远超动作 token → 训练/推理效率低、视频帧预测冗余、注意力失衡导致**过拟合视频 token、削弱动作预测**。
解法:训练与推理时**下采样视频帧**,使视频 token 与动作 token 数量平衡(如把视频帧率设为动作帧率的 1/6)。动作密集、视频稀疏。

### 4. Latent Action:像素级"delta action"(Figure 3,重点)
这是"在 WAM 上用 LAM 预训练"的核心机制:
- **用光流(optical flow)作为运动的自然表示**:相邻帧的像素级位移,由 **DPFlow[33]** 计算并转成 RGB 图像。
- **压缩到控制级空间**:用**深度卷积变分自编码器 DC-AE[13]** 重构光流,同时把它编码成 **4×512 维 token**;再用一个轻量 encoder 投影成 **14 维向量**——**大致匹配典型机器人动作空间的尺度**。→ 这个"维度对应"让 latent action 天然能与真实机器人控制对齐,充当感知与动作的桥梁。
- **分布对齐**:引入 task-agnostic 数据(用 Curobo 随机采样目标机器人动作空间,任务无关地采集 image-action 对)提供**真实动作监督**,让 VAE 学到反映可行运动的 embedding,把 latent action 锚定到真实控制分布。
- **训练混合**:90% 无标签数据做自监督重构 + 10% 有标签轨迹做弱动作监督(含 task-agnostic 与标准机器人演示)。
- **VAE 损失**:`L = L_recon + λ_a·‖a_real − a_pred‖² + β·L_KL`(重构 + 潜动作与真实动作对齐 + KL 正则)。

### 5. 三阶段训练 + 六层数据金字塔(Table 1 / Figure 4)
渐进式地把物理交互先验从多样数据迁移到目标机器人:
- **Stage 0(现成)**：预训练基础模型——VGM(Wan2.2) + VLM(Qwen3-VL),用 Level-1 Web 数据。
- **Stage 1:学习视觉动力学**——只训 VGM,用多机器人轨迹 + 人类视频(Level 2/3/5),让 VGM 能从语言+初始图生成合理未来视频。
- **Stage 2:学习 latent action 表征(联合)**——冻结 VLM,训练整个 Motus 三专家(**带 latent action**),用 Level 2/3/4/5;把运动/交互知识注入 latent action 空间,**初始化动作专家**。← 这一步就是 LAM 预训练。
- **Stage 3:目标机器人 SFT**——用 Level-6 目标机器人轨迹微调,三专家(**带真实动作**),让先验完全适配目标本体动力学/运动学。

**六层数据金字塔**(底→顶,数据量递减、质量递增):Level 1 Web 数据 → 2 第一人称人类视频 → 3 合成数据 → 4 task-agnostic 数据 → 5 多机器人轨迹 → 6 目标机器人轨迹。排除纯语言等缺视觉模态的数据。

---

## 五、实验与验证

### 1. Baselines
π0.5、X-VLA;并额外对比 **from-scratch(w/o pretrain)** 与 **只做 Stage-1** 的消融版本。

### 2. 仿真(RoboTwin 2.0,Table 2)
- 50 个代表性任务,多任务训练:2500 clean(50/任务)+ 25000 randomized(500/任务);全部从预训练 checkpoint 微调 **40k 步**;每任务 100 trials。
- 平均成功率:**Motus Clean 88.66% / Rand. 87.02%** vs. π0.5(42.98 / 43.84)、X-VLA(72.80 / 72.84)、w/o Pretrain(72.8 / 77.00)、Stage1(82.86 / 81.86)。
- 结论:统一 MoT 解决挑战 1;latent action 让其能利用有标签+大规模无标签数据(挑战 2),显著提升泛化与运动先验。

### 3. 真实世界(Table 3,两个双臂平台 AC-One / Agilex-Aloha-2)
- 覆盖空间理解、可形变物体、精细流体控制、视觉理解、长程规划(叠毛巾、滴滤咖啡、磨咖啡豆等)。
- 每任务 100 条轨迹,多任务联合训练;用**部分成功率(partial success rate,按子目标给分)**评估长程任务。
- Motus 在两个平台上**全面超过 π0.5**:AC-One 平均 63.22 vs. π0.5 14.79 vs. w/o pretrain 25.86;Agilex 平均 59.30 vs. 48.60 vs. 26.60。

### 4. 消融(Figure 6,RoboTwin randomized 多任务成功率)
- **w/o Pretrain 77.00% → Stage1 pretrain 81.86%(+~5%)→ Stage2 pretrain(完整 Motus)87.02%(+11.10% 相对 w/o)。**
- Clean 设定同样:77.56% → 82.26% → 88.66%。
- 结论:**每个预训练阶段都带来正增益,Stage 2(即 latent action 预训练)贡献最大**——直接验证了"在世界模型上做 latent action 预训练"的价值。

---

## 六、与 Fast-WAM 的对照解读(对"LAM 预训练 WAM"方向最关键)

> 详见同目录 `fast_wam_notes.md` 第六节。这里给出两篇论文放在一起看的结论。

| 维度 | **Motus** | **Fast-WAM** |
|---|---|---|
| 定位 | LAM + WAM 的统一预训练范本 | 质疑"测试时想象未来"的必要性 |
| 是否 LAM 预训练 | **是**(核心卖点,Stage 2 光流 latent action 预训练) | 否(直接用真实动作标签监督,无具身预训练) |
| 骨干 | Wan2.2-5B(VGM)+ Qwen3-VL-2B(VLM),MoT 三专家 | Wan2.2-5B 视频 DiT + 动作专家,MoT 两分支 |
| 训练期表征来源 | 光流 latent action + 视频生成 | 视频协同训练(真实动作) |
| 推理 | UniDiffuser 可切换多模式 | 单次前向,跳过未来生成(190ms) |
| RoboTwin(randomized/平均) | Motus 87.02%(**带预训练**) | Fast-WAM 91.8%(**无预训练**) |

**两个关键 takeaway:**
1. **Motus 正面证明了"LAM 预训练有用"**:其消融显示 Stage 2 latent action 预训练带来最大增益(+11% 相对无预训练)。这是支持"在 WAM 上用 LAM 预训练"的正向证据。
2. **但 Fast-WAM 是清醒的反例**:Fast-WAM **无任何预训练**(RoboTwin 91.8%)反而**超过带完整预训练的 Motus**(87.8%,Fast-WAM 论文表1口径)。
   - 二者口径/训练步数/数据不完全可比,但至少说明:**latent action 预训练的收益,并不必然超过"简单的视频协同训练 + 真实动作监督"。**
   - 对做方案的启示:若要投入 LAM 预训练,应设一个 Fast-WAM 式的"视频协同训练 + 真实动作"强基线做对照,确认 latent action 预训练的净增益;并注意 Motus 的增益可能高度依赖其**六层数据金字塔 + 大规模异构无标签数据**——LAM 预训练的价值大概率要在**数据规模上去后**才充分体现(呼应 Fast-WAM 的 future work)。

---

## 附:一句话记忆点

- **Motus**：用光流 latent action 在无标签视频上大规模预训练一个 MoT 统一世界模型,分三阶段迁移到目标机器人 → "LAM 预训练 WAM"的完整工程范本。
- **对我的方向**:Motus = 怎么做(three-stage recipe + 光流 delta action + 数据金字塔);Fast-WAM = 要不要做的反问(先跑赢视频协同训练基线再说)。
