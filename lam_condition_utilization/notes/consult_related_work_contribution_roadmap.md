# 咨询提问稿：Related work、当前贡献与下一步研究路线（2026-07-21）

这是第五轮、可直接发给外部研究顾问的自包含咨询稿。前四轮讨论和实验记录见
consult_story_b_prompt.md、consult_tier2_prompt.md 与 stage_b_plan.md。

这次不再只问“下一个实验怎么做”，而是希望顾问站在论文作者和审稿人的双重视角，
重新判断三个相互关联的问题：

1. 我们和已有工作的真实重合边界在哪里；
2. 目前已经完成的工作，究竟能构成什么层级的研究贡献；
3. 接下来应该沿哪条路线继续，以及什么结果应当让我们停止某条路线。

请不要默认我们现在的故事一定成立，也不要为了形成完整论文而强行推荐把所有模块拼起来。
我们更希望得到一个会主动质疑前提、允许否定已有方向、并能在有限预算下作出取舍的判断。

---

## 一、项目设定

我们研究的是连续 Latent Action Model（LAM）。模型从相邻观测中推断 latent action：

~~~text
encoder: (o_t, o_t+1) -> μ, σ
sample:  z = μ + σ · ε
decoder: (o_t, z) -> o_t+1
~~~

当前模型基于 DreamDojo 的约 700M LAM，在 EgoDex 上微调。我们把 latent 分成：

- z_a：32 维，目标是承载动作相关变化；
- z_e：8 维，目标是承载环境或其他变化；
- split KL：β_a = 3e-3，β_e = 1e-6；
- 主线 trunk 还带一个 18D 双手腕部位姿辅助头。

因此，完整训练流程是 pose-regularized / supervised，而不是完全 label-free。需要特别说明，
18D 只覆盖腕部位姿，并不等于现实中所有可控变化：它没有完整覆盖手指、接触、力、身体动作
以及其他可能影响画面的控制变量。它可以作为“已知的部分控制通道”和正对照，却不能被称为
完整 latent action 的 ground-truth 上限。

下游 DreamDojo ACWM 使用的是 LAM encoder 输出的确定性 μ，而不是 LAM 自己的重建 decoder。
这条接口事实对当前工作的价值判断至关重要。

---

## 二、我们是怎样发现问题的

### 2.1 起点：方向 loss 改变了 latent 几何，却没有改善功能

我们原本设计了方向约束 L_dir，希望同一动作的相反运动在 z_a 中形成相反方向。实验中，
L_dir 确实明显改变了局部几何，例如 action cosine 从约 +0.16 变为 -0.30，active fraction
从约 0.90 降到 0.34；但它没有通过 held-out representation 和 decoder-use 的预注册门。
也就是说，“latent 看起来变了”没有转化成“decoder 更服从该 latent”。

进一步审计发现，旧的 verb donor 构造本身无效：同 verb donor 比 opposite donor 更接近
anchor GT18 动作的比例只有 0.488，基本等于随机。随后我们用按 18D 动作距离严格排序的 donor，
并构造训练无关的 frame-delta oracle 作为正对照：

- oracle × distance donor 的 utilization 分数约为 1.149，说明 scorer 能检测强条件使用；
- 原 LAM baseline 只有约 0.039 / 0.078，即 oracle 的约 3%–7%；
- 因此原来的近零结果一部分来自 donor 设计错误，但修正 donor 后仍然存在真实的低利用率。

这使问题从“方向表示是否学到”转为：

> 动作信息是否已经存在于 μ 中，但 decoder 没有把它当成可信条件使用？

### 2.2 α 消融：只改变 decoder 训练时看到的噪声

我们保持 KL、action head 和 encoder 参数化不变，只改变重建 decoder 的训练输入：

~~~text
z_dec = μ + α · σ · ε

A1: α = 1，标准随机重参数输入
A0: α = 0，decoder 只看到确定性 μ
~~~

两 seed、各 1k 步得到：

| 指标 | A1：α=1 | A0：α=0 |
|---|---:|---:|
| C_dist / oracle ceiling | 0.039 / 0.078，即 3%–7% | 0.830 / 0.825，即约 72% |
| antiparallel signed-direction use | 0.064 / 0.073 | 0.381 / 0.386 |
| val PSNR | 29.92 / 29.92 dB | 32.96 / 33.27 dB |
| posterior noise / per-dim SNR / effective rank | 约 1.00 / 0.54 / 10 | 基本相同 |

A0 还在 K=4/8/16 的原 LAM decoder 自回归 rollout 中保持优势。与 A1 相比，
own-rollout PSNR 高约 4–5 dB，对 magnitude-matched sign flip 的符号使用高约 5–7 dB。
这说明 one-step utilization 指标对应真实的、多步的 decoder 行为，而不只是 probe 数字。

代价是：A0 没见过 posterior sampling noise，所以对 μ+sσε 的采样更脆弱；但对直接选择和缩放
z_a 的 do-intervention，它的响应单调且没有观察到额外伪影。

### 2.3 因果定位：改善发生在 decoder，不在 encoder

最初我们担心“虽然 posterior 汇总统计相同，α=0 也许仍通过反向传播改善了逐样本 encoder”。
随后做了 paired-latent audit 和四格 cross-decoding。

同一 transition 上，A0 与 A1 的 encoder μ 几乎相同：

| 指标 | seed 42 | seed 1 |
|---|---:|---:|
| cosine(μ_A0, μ_A1) | 0.9999 | 0.999 |
| linear CKA | 0.9999 | 0.999 |
| Procrustes aligned R² | 0.9998 | 0.998 |
| kNN overlap@10 | 0.967 | 0.889 |
| norm(Δμ) / norm(μ) | 0.016 | 0.048 |

四格 cross-decoding 的 C_dist 为：

| decoder ↓ / encoder → | E_A0 s42 | E_A1 s42 | E_A0 s1 | E_A1 s1 |
|---|---:|---:|---:|---:|
| D_A0 | 0.830 | 0.824 | 0.825 | 0.817 |
| D_A1 | 0.039 | 0.039 | 0.078 | 0.078 |

行差 11–21 倍，而更换 encoder 的列差不到 1%。这把结果很干净地定位为：

> μ 中原本有可读的动作信息；低 SNR 的随机 decoder exposure 让带 observation shortcut 的
> decoder 学会忽略 latent；α=0 恢复的是本 LAM decoder 的 conditional utilization，
> 并没有改善导出给下游的 encoder μ。

因此，我们现在不再把这个现象简单描述成 classic posterior collapse。更准确的区分是：

- **capacity / availability failure**：latent 本身没有足够信息；
- **semantic routing failure**：信息存在，但没有进入我们希望的 z_a，可能进入低 KL 的 z_e、
  observation shortcut 或无关 nuisance；
- **conditional-use / utilization failure**：信息已经在 μ 中，但 decoder 不依赖它。

当前 pose-regularized A0/A1 的主要问题属于第三种。之前的 label-free twins 更接近第一和
第二种：去掉 18D anchor 后，split KL 下 reconstruction 没有理由把动作放进代价更高的 z_a。
所以不能把两个现象混成“KL 让 32 维完全装不进信息”。GT18 解释了 semantic routing 和
anti-collapse；α 消融解释的是已有 μ 为何没有被 decoder 使用。

---

## 三、Related work：我们认为的重合边界

下面是截至 2026-07-21 的初步定位。希望顾问重点指出遗漏的直接先例，而不只是补一份宽泛书目。

### 3.1 与 decoder 噪声和 VAE collapse 最接近的工作

| 工作 | 与本项目的关系 | 对新颖性主张的限制 |
|---|---|---|
| [Fixing a Broken ELBO](https://proceedings.mlr.press/v80/alemi18a.html) | 用 rate–distortion 视角说明强 decoder 可以在良好 likelihood 下忽略 latent | “强 decoder 会忽略 latent”不是新发现 |
| [From Variational to Deterministic Autoencoders](https://arxiv.org/abs/1903.12436) | 把 Gaussian VAE sampling 解释成向确定性 decoder 输入注入噪声，并研究 deterministic regularized autoencoder | “去掉采样噪声有帮助”不是全新操作 |
| [λ-VAE: Variance Equalization for Posterior Collapse](https://arxiv.org/abs/2607.05531) | 使用 z=μ+σ^λ ε，同时 KL 仍使用原 σ²；把 gradient imbalance 与 information gap 作为 collapse 原因 | 这是最直接的技术碰撞。我们不能声称首次把 reconstruction exposure noise 与 KL posterior 解耦 |

我们与 λ-VAE 的可能差异是：

1. λ-VAE 面向一般 VAE 的 posterior collapse 与信息容量；我们面对的是 conditional LAM 中
   “μ 可读但 decoder 不用”的 utilization failure。
2. 我们的 α 直接缩放噪声。当前 σ≈1 时，σ^λ 在某一时刻仍接近 1，而 α=0 会直接消除 exposure
   noise；但 λ 从训练早期改变动态，不能仅凭这个代数观察断言它在本设定无效。
3. 我们用正对照 donor assay、paired-latent audit 和 cross-decoding 把 availability 与 use 分开，
   并测动作幅度、符号方向及 K-step control；λ-VAE 主要报告标准图像 VAE 的容量和生成指标。
4. λ-VAE 试图改善 encoder posterior；我们的现有 α=0 结果恰恰没有改变 encoder。

因此，我们目前更愿意把关系写成：

> λ-VAE 覆盖了“修改重参数噪声以缓解 VAE collapse”的方法新颖性；本项目可能仍有价值的部分，
> 是 LAM 中 available-but-ignored failure 的测量、因果定位、动作语义功能验证，以及对
> downstream interface 不继承 decoder 修复这一边界的证明。

我们不确定这个差异是否足以支撑 main paper，还是最多支撑一篇诊断型 workshop paper。

### 3.2 与 LAM 的动作可用性、去混淆和监督锚相关的工作

| 工作 | 解决的主要问题 | 与我们的区别或重合 |
|---|---|---|
| [What Do Latent Action Models Actually Learn?](https://arxiv.org/abs/2506.15691) | 分析可控变化与外生噪声何时进入 latent，并讨论数据、augmentation 和 action prediction | 更接近 availability / semantic correctness，不直接测 decoder 是否使用已有 μ |
| [Latent Action Learning Requires Supervision in the Presence of Distractors](https://proceedings.mlr.press/v267/nikulin25a.html) | 证明 distractor 下少量动作监督可显著改善 latent 与 downstream | 支持 GT18 作为训练锚的合理性，也限制我们把 pose supervision 说成无关细节 |
| [LAPO](https://proceedings.iclr.cc/paper_files/paper/2024/hash/27985d21f0b751b933d675930aa25022-Abstract-Conference.html) 与 [LAPA](https://proceedings.iclr.cc/paper_files/paper/2025/hash/45d74e190008c7bff2845ffc8e3facd3-Abstract-Conference.html) | 从 observation-only 视频恢复离散 latent action，并用于 policy/VLA 预训练 | VQ latent 没有 σε 这一连续采样通道，可作为 α 机制不应出现的边界或负对照 |
| [CD-LAM](https://arxiv.org/abs/2607.09185) | 用 embodiment reconstruction、action contrastive 和 calibration 减少 action-irrelevant bias，并在 2B/14B ACWM 上评估 | 更完整地覆盖 encoder 表征与真实 downstream；我们的诊断指标更细，但目前没有改善 μ-only 下游接口 |
| [ConLA](https://arxiv.org/abs/2602.00557) | 用动作类别先验和时间线索做 motion/content contrastive disentanglement | 与我们原 L_dir 的“精细方向语义”路线相邻，也提示简单方向约束可能已经处于拥挤赛道 |
| [Olaf-World](https://arxiv.org/abs/2602.10104) | 以可观察 control effect 跨场景对齐 latent action 坐标 | 直接关联“全局可迁移动作坐标”，比只在本地配对上改变 cosine 更接近下游控制需求 |
| [AC-LAM](https://arxiv.org/abs/2604.03340) | 对 latent action 加 identity、inverse、cycle 和短时 additive composition 结构，强调 motion magnitude calibration | 与重新启动 L_dir / 精细控制路线有较强碰撞；我们需要说明单步方向 loss 能提供什么额外价值 |
| [Co-Evolving Latent Action World Models](https://arxiv.org/abs/2510.26433) | LAM 与 world model 联合训练，使 world-model gradient 反过来塑造控制接口，并处理联合 collapse | 它比“先修一个之后被丢弃的 LAM decoder”更直接回答 decoder 梯度能否改善 encoder |

从这张表看，相关工作不是一个单一问题，而是至少四层：

1. VAE information capacity / posterior collapse；
2. latent 是否承载真正的可控变化；
3. consumer 是否使用 latent；
4. 导出的 latent 是否能让独立世界模型或 policy 获益。

我们现有最强证据集中在第 3 层，并对第 2 与第 3 层的混淆做了因果拆解；但领域论文通常最终
以第 4 层作为价值证明。

---

## 四、我们目前认为的主要贡献

请顾问不要把下面每一项都默认算作 contribution。我们希望你判断哪些是论文主贡献，
哪些只是必要的实验卫生，哪些不够新。

### 4.1 可能成立的贡献

1. **availability、generic utilization、direction-specific causal use 的分解。**
   我们展示了 probe 能读出动作、latent 几何被 loss 改变，并不意味着 decoder 会使用它。

2. **带正对照校准的 utilization assay。**
   旧 verb donor 对 oracle 也接近零；distance-ordered donor 让 oracle 达到 1.149，从而把
   metric bug 与真实低 utilization 分开。这比直接把 swap 结果当因果证据更严格。

3. **一个有信息量的负结果：L_dir 改几何但不改功能。**
   这否定的是当前 loss 配方和“几何更漂亮就更可控”的推理，不是否定精细控制本身。

4. **对 available-but-ignored failure 的因果定位。**
   α 消融、两 seed、paired-latent audit 与四格 cross-decoding 联合说明：encoder μ 几乎不变，
   利用率由 decoder 决定。这比仅报告 KL、rank 或 reconstruction 更接近机制证据。

5. **decoder 内部的有效修复。**
   α=0 把动作幅度利用从 oracle 的 3%–7% 提到约 72%，恢复 signed direction，并改善 one-step
   reconstruction 和 K-step rollout。它不是只让指标变好。

6. **一个重要的负边界。**
   因为 downstream ACWM 只消费 μ，而 μ_A0≈μ_A1，所以修好被丢弃的 LAM decoder 不会自动改善
   下游接口。这个结果阻止我们做一个看似合理、实则错误的 downstream claim。

### 4.2 我们认为不能这样声称

- 不能声称首次发现 VAE decoder 会忽略 latent；
- 不能声称首次解耦重参数噪声与 KL，λ-VAE 已经直接覆盖；
- 不能把 α=0 称为仍在优化标准 ELBO：它更接近确定性 mean reconstruction 加原 KL regularizer；
- 不能声称 α=0 改善了 LAM encoder 或下游 ACWM；
- 不能声称 18D GT 是完整动作真值或完整性能上限；
- 不能因为 label-free trunk 塌缩就说“KL 数学上不允许 32D 装任何动作信息”；
- 不能因为 L_dir 失败就说“LAM 不可能精细控制动作”；
- 不能仅凭当前单架构、单数据域、两 seed、1k-step 结果声称普遍机制。

### 4.3 对论文层级的暂时判断

我们的自我判断是：

- **workshop / analysis paper**：目前已有较完整的核心，主题可定位为
  “Available but Ignored: Conditional-Use Collapse in Continuous Latent Action Models”；
- **一般表征学习 main paper**：还缺跨架构、跨数据和与 λ-VAE 等直接 baseline 的系统比较；
- **CoRL / RSS 式 embodied paper**：还缺真实独立 consumer、ACWM action-following 或 policy 价值；
- **“比 CD-LAM 更好的 LAM 方法”**：当前证据不支持，因为导出的 μ 没有改善。

我们想请顾问判断：这是否低估或高估了现有贡献？“诊断 + 因果定位 + 简单修复 + 明确不迁移边界”
本身能否是一篇有价值的主论文，还是必须有一个能改变 encoder/downstream 的新方法才够？

---

## 五、未来路线：请以开放、可否证的方式帮助决策

下面不是我们已经决定的 roadmap，而是互相竞争的研究假设。每条路线都列出“为什么可能值得”
和“为什么也可能是错的”。请顾问帮助选择，而不是把它们都建议做完。

### 路线 A：先验证诊断是否跨架构成立

**问题：** conditional-use failure 是连续 LAM 的普遍现象，还是 DreamDojo 这一套 40D、
pose-regularized、observation-shortcut decoder 的实现特例？

可能值得做的理由：

- 如果第二个连续 Gaussian LAM 也出现“μ 可读、采样低 SNR、decoder 不用、clean exposure 恢复”，
  当前工作可从项目 debugging 升格为一类机制；
- 原生 32D DreamDojo/CD-LAM trunk、第二数据域或不同 decoder shortcut 强度可以提供外部有效性；
- VQ-LAM 没有 σε，可作为机制边界，而不是要求所有 LAM 都复现。

让我们怀疑它的理由：

- 当前 μ 的动作可用性可能主要由 18D pose anchor 提供；
- 其他连续 LAM 的 posterior variance、decoder 条件注入或训练目标可能完全不同；
- α=0 的巨大效果可能只是某个初始化、loss scale 或特定 shortcut 的工程现象。

最小判定实验：

- 一个第二连续架构或原生 32D trunk；
- 先做 availability + 正对照 utilization assay，再做 α∈{0,1}；
- 若不能复现“row effect 大、encoder column effect 小”的 cross-decoding 结构，就收窄为当前模型案例，
  不继续宣称一般机制。

### 路线 B：探索 decoder 训练是否能真正塑造 label-free encoder

**问题：** 如果从训练早期就让 decoder 看到干净、稳定的 μ，reconstruction gradient 是否可能
长期流向 encoder，最终改善导出的 μ？

这个问题值得探索，因为 downstream 使用 encoder；如果 decoder-side intervention 能反过来塑造
encoder，它才可能产生实际 LAM 接口价值。

但我们不再把这个假设当作理所当然，原因是：

- 现有 A0/A1 已经给出强反例：反向传播存在，但 μ 几乎没有变化；
- 在 split KL 下，z_a 的 β 比 z_e 高 3000 倍。没有 18D anchor 时，reconstruction 即使需要 latent，
  也没有理由把动作放进昂贵 z_a，可能选择便宜 z_e、背景 nuisance 或 observation shortcut；
- clean exposure 解决的是 utilization，不自动解决 capacity 和 semantic routing；
- 更高 KL、更高 rank 或更强 reconstruction 都不等于更好的 action representation。

因此我们考虑的不是单独押注 α=0，而是一个小型因果矩阵：

| | 标准/高 β_a | 低 β_a、free-bits 或 target-rate |
|---|---|---|
| α=1 | KL cost + noise exposure | capacity 放开但仍有 noisy exposure |
| α=0 | clean exposure 但 routing 仍无锚 | capacity 与 utilization 同时放开 |

同时记录 reconstruction-to-encoder gradient norm、raw per-dim KL、active dims、z_a/z_e 中的
motion/action proxy、cross-decoding 和独立 consumer 性能。

可能结果与解释：

- 只有降低 β/free-bits 有效：主因是 capacity/rate，而不是 exposure；
- 只有 α=0 有效：clean decoder gradient 确实能塑造 encoder；
- 两者同时才有效：capacity 与 utilization 存在交互；
- rank 上升但 action 不进 z_a：真正缺的是 semantic routing，需要 flow/effect/cross-context proxy；
- 全部失败：18D anchor 可能是当前架构唯一有效路由信号，应停止把“纯 label-free”当短期主线。

请顾问判断：这个 2×2 是否是一个真正能回答问题的最小实验，还是依然混入太多因素？

### 路线 C：沿 α / λ-VAE 做一个更一般的 exposure 方法

候选包括：

- α sweep：0、0.25、0.5、1；
- α schedule：先干净 exposure，再逐步增加采样噪声；
- λ-VAE baseline；
- dual-path reconstruction：同时训练 D(o_t, μ) 与 D(o_t, μ+σε)，在 utilization 与采样鲁棒性间取 Pareto。

可能值得做的理由：

- A0 的利用率提升很大，但 posterior sampling robustness 明显下降，存在真实的
  reconstruction–generation trade-off；
- α、λ 或 dual-path 的差异可能在 σ≈1 的 LAM regime 下形成一个不同于标准图像 VAE 的结论；
- 若 across architecture/data 都成立，可形成 general conditional-VAE mechanism paper。

让我们怀疑它的理由：

- 这条路线最容易只把被丢弃的 LAM decoder 做得更好；
- λ-VAE 已经提出更完整的理论和自适应 exponent，简单 α sweep 的方法新颖性可能很弱；
- “整体指标超过 λ-VAE”是一个模糊目标。即使 reconstruction 更好，也不代表导出的 μ 或独立 consumer 更好；
- 如果目标是 embodied LAM，下游不使用随机 posterior，这个 generation trade-off 可能不是核心问题。

我们希望顾问判断：应该把 λ-VAE 当作必须超过的主要 baseline，还是把它当作界定新颖性边界，
然后把重心放在 LAM utilization 的测量和 downstream interface 上？

### 路线 D：重新启动 L_dir，追求比 CD-LAM 更精细的动作控制

我们最初的直觉是：CD-LAM 解决 action awareness 和 bias，但 latent 仍可能缺乏动作幅度、符号方向
或组合结构；因此 L_dir 也许能提供更精细控制。

现在我们对此有明显怀疑：

- 当前 L_dir 已经改变几何，却没有改善 held-out availability 或 decoder use；
- A0 baseline 不加 L_dir 已经恢复明显的 signed-direction response，说明原来的“方向缺失”
  一部分其实是 utilization failure；
- AC-LAM、Olaf-World、ConLA 等已经从 inverse/cycle、control effect、temporal contrastive 等角度
  处理动作结构，单一 cosine 方向 loss 未必足够；
- 局部 antiparallel 配对不保证形成跨场景、全局一致、物理可组合的动作坐标。

如果仍要给 L_dir 最后一次机会，我们倾向只做一个决定性比较：

~~~text
R + best exposure
versus
D + best exposure
~~~

并预注册四个必须同时过的门：held-out direction availability、antiparallel signed use、
fidelity、独立 consumer action-following。若只改变 latent cosine 或不超过 A0 baseline，就关闭
这条路线，不再继续调 loss。

请顾问判断：在已有负结果与拥挤 related work 下，这个实验仍值得做吗？如果值得，L_dir 需要怎样
重构才不是简单重跑失败配方？例如应否从 pairwise direction 改成 multi-step composition、
effect alignment 或跨场景坐标一致性？

### 路线 E：直接面向真实下游 consumer

这是应用价值最直接的路线，但也最可能推翻 α=0 的论文故事。

Phase-0 已表明 μ_A0≈μ_A1，因此只吃 μ 的独立 consumer 理论上应看到 A0≈A1。我们不应预设
A0 会胜出。一个轻量 matched consumer 仍可能有诊断价值：

- 若同一 μ 上 probe 和轻量 consumer 都能 action-follow：encoder interface 已经可用，问题在
  大 ACWM 是否学习条件；
- 若 probe 可读但独立 consumer 不 action-follow：可读性不足以构成易学习的生成条件，需要新的
  interface/encoder objective；
- 若 GT18 条件的正对照 consumer 都失败：consumer 或 scorer 无效，不能解释 LAM 结果；
- 若 A0≈A1：确认 decoder 修复不迁移，应收窄论文；
- 若 A0 意外优于 A1：必须寻找 paired audit 没捕捉到的差异，不能直接用结果反推原故事正确。

GT18 在这里应只作为 partial positive control，不是完整 latent 的上限。评估还应包含 hand/object
track、signed displacement、背景 fidelity 与 motion lower bound，并区分“未被 18D 覆盖的可控动作”
和“只是导致像素变化的外生因素”。

更激进的版本是：

- 让真实 ACWM 在训练时明确暴露于 deterministic μ 与扰动 μ，研究 consumer 自身的 condition use；
- 或像 CoLA-World 一样联合训练 LAM encoder 与 world model，让下游 gradient 真正塑造接口。

但这可能已经是一个新项目，不是当前 α=0 结果的自然补实验。请顾问判断它是否应该成为主线，
还是应该把当前工作先收束为诊断论文，再单独开启该项目。

---

## 六、我们最需要顾问作出的决策

请直接回答以下问题，即使答案是“当前工作不够发 main”或“应放弃原 loss 主线”。

1. **Related-work collision：** λ-VAE 是否已经覆盖了我们最可能声称的方法新颖性？除了本文列出的
   工作，还有没有更早、更直接的“deterministic mean exposure + original KL”先例？

2. **贡献定级：** 当前最有价值的是 α=0 方法、available-but-ignored 现象、正对照测量框架、
   cross-decoding 因果定位，还是“不迁移到 μ-only downstream”的负边界？请按强弱排序。

3. **论文形态：** 现有证据最适合 workshop、analysis paper，还是已经可能形成 main paper？
   若要 main，缺口究竟是跨架构广度、理论、独立 consumer，还是必须提出改善 encoder 的新方法？

4. **路线选择：** A–E 中只能优先选择一条时，你会选择哪条？请说明它消除的最大不确定性、
   最小实验、成功门和明确 stop condition。

5. **decoder→encoder 假设：** 在 split KL 和无 18D routing signal 下，clean decoder exposure
   是否有足够理论理由改善 label-free encoder？上面的 2×2 是否值得做，还是应直接加入
   motion/effect routing objective？

6. **L_dir 去留：** 方向 loss 的失败应被当作论文中的负结果，还是仍值得在 utilization 修好后
   重启？什么结果才足以证明它提供了“比 CD-LAM 更精细的控制”，而不只是几何变化？

7. **GT18 的正确角色：** 如何同时利用它作为 partial positive control 和训练锚，又不把它误写成
   完整 latent-action oracle？对于 GT18 未覆盖的真实可控因素，应怎样建立可评价的 proxy？

8. **最小投稿策略：** 在共享 GPU、只能承担少量关键 runs 的情况下，请给出：
   - 一条当前即可收束的保守投稿路线；
   - 一条有机会升级 main 的高价值路线；
   - 一个超过成本或风险阈值就应停止的条件。

---

## 七、希望顾问采用的回复格式

为了避免得到“每条都值得探索”的泛泛答案，希望按以下格式回复：

1. **五句话 verdict**：现有工作是什么、不是什么、最可能的 venue 层级、最大 novelty collision、
   唯一优先下一步；
2. **related-work collision 表**：直接先例、部分重合、只是背景，分别列出；
3. **claim 守门表**：现在能说、补一个实验后能说、目前不能说；
4. **A–E 路线排序**：每条给预期信息增益、成本、成功概率、失败后是否仍有论文价值；
5. **唯一推荐的最小实验包**：明确模型数、seed、指标、正负控制和 stop/go 门；
6. **反方审稿意见**：请用最强反对者视角写出三条可能拒稿理由，并说明哪一条最致命。

---

## 顾问第五轮回复记录

以下判断基于本文档所列证据、已完成的 Phase 0/1/2 产物，以及截至 2026-07-21
能够核查到的原始论文。需要先区分两个状态：Phase 0、Tier-1.5 和原 LAM decoder 的
Route-C 结论已经相当稳定；独立 consumer 的 `GT-18D / zmu A0 / zmu A1` 当前只有一个
LAM seed 和一个 downstream seed，并且只在 dev gate 上运行，因此属于**很有信息量的
预实验**，还不是预注册意义上的最终 Tier-2 证据。

## 1. 五句话 verdict

1. 这项工作目前最准确的定位是：**一个关于连续条件 LAM 中“action information
   available but not used”故障的机制诊断与因果定位案例**，而不是一个更好的可导出 LAM。
2. `alpha=0` 应写成定位故障的干预工具，而不是核心算法贡献；2026 年 7 月出现的
   [lambda-VAE](https://arxiv.org/abs/2607.05531) 已经直接覆盖了“重参数采样噪声与原 KL
   方差解耦”这一更一般的原理，虽然它的 `sigma^lambda` 与本文的全局 `alpha` 不是同一
   参数化。
3. 新的独立 consumer dev 结果进一步收窄了机制：新 consumer 能强烈使用 A0、A1
   两套近乎相同的 `mu`，而二者效应接近，因此 alpha 修复的是**原 decoder 的条件使用
   路径**，不是 A1 `mu` 固有地不可用于下游，也没有形成可迁移的 encoder 改进。
4. 以目前单一 DreamDojo Gaussian LAM、EgoDex、wrist-pose 锚和未最终封存的 Tier-2
   证据，足够形成一篇较强的 workshop / analysis paper；离 CoRL/NeurIPS main 的主要差距
   是跨实现复现、严格的 action-transfer construct validity，以及实际导出收益，而不是再加
   一条普通 loss。
5. 在关闭当前 Route-C 的评估债务后，唯一优先的新研究路线应是 **A：在第二个连续
   Gaussian LAM / 原生 32D trunk 上复现 paired-latent + cross-decoding 的完整“行效应大、
   列效应小”签名**；不建议启动 2B ACWM、alpha sweep、`L_dir` 重启或含混的 label-free
   2x2。

## 2. 事实核查与 related-work collision 表

### 2.1 最重要的事实核查

- [lambda-VAE](https://arxiv.org/abs/2607.05531) 是 2026-07-06 提交的 arXiv v1，不是已
  同行评审的既有经典工作。论文明确提出两种 collapse 原因：gradient imbalance 和
  information gap；其方法在重参数化时按维度使用 `sigma^lambda` 缩放噪声，而 KL 仍使用
  原 posterior variance。这与本文“decoder exposure 与 KL 解耦”的方法原则是**直接碰撞**。
- `alpha=0` 不是 lambda-VAE 的严格同义实现。lambda-VAE 是依赖每维 `sigma` 的指数变换，
  本文是全局乘子；尤其在 `sigma` 接近 1 时，二者的即时作用不同。不过这种差异只能支持
  “不同的简化实例/端点”，不足以继续声称解耦原则本身新颖。
- 在本轮有针对性的检索中，没有找到一篇把“训练 reconstruction 固定使用 posterior mean，
  同时完整保留标准 Gaussian KL”作为标题方法的公认早期先例。但这不是穷尽性证明；
  [Regularized Autoencoders](https://arxiv.org/abs/1903.12436) 虽然强调去除采样噪声，却用
  确定性 encoder 和其他正则替代 VAE prior/KL，因而不是同一算法。这个实现过于简单且与
  常见 mean decoding 相邻，不宜把“未找到更早 exact match”当成 method novelty 的支柱。
- 标准 posterior collapse 通常指 `q(z|x)` 退化到 prior、latent 不含信息；本文 A1 的
  `mu` 仍有 GT18 可解码信息，因此更准确的术语是 **conditional-use collapse / decoder-side
  latent ignoring**，不能无修饰地声称发现了新的 posterior collapse。
- [Generative Skip Models](https://proceedings.mlr.press/v89/dieng19a.html) 已通过增强
  latent-to-likelihood 链接来减少 latent ignoring；[inverse-Lipschitz decoder](https://proceedings.mlr.press/v202/kinoshita23a.html)
  也已经把“强制 decoder 对 latent 保持可区分/敏感”作为有理论保证的方法。因此“让 decoder
  使用 z”这个大方向不新；本文可争取的是 LAM 场景中的**诊断组合与因果定位证据**。

### 2.2 碰撞分级

| 分级 | 工作 | 与本文的关系 | 对 claim 的后果 |
|---|---|---|---|
| 直接先例 | [lambda-VAE](https://arxiv.org/abs/2607.05531) | 同样把 reconstruction 所见的采样噪声与原 KL variance 解耦，并以此改变 collapse attractor | 放弃“新 VAE 原理/新 exposure family”；可把 `alpha=0` 保留为 LAM 中的最小因果干预和强 baseline |
| 直接机制邻居 | [Mitigating Posterior Collapse in Strongly Conditioned VAEs](https://openreview.net/forum?id=rJlHea4Kvr) | 强条件或强 decoder 可能绕过 latent；与本文“信息存在但 conditional edge 不使用”高度相关 | 不能把 decoder ignoring 当成首次发现；强调 LAM 的 paired/cross-decode 定位 |
| 直接机制邻居 | [Generative Skip Models](https://proceedings.mlr.press/v89/dieng19a.html)、[inverse-Lipschitz decoder](https://proceedings.mlr.press/v202/kinoshita23a.html) | 从架构或约束上增强 decoder 对 latent 的依赖/可区分性 | “decoder sensitivity”不是新目标；本文贡献必须是测量、定位和 LAM 特有证据 |
| 部分重合 | [Regularized Autoencoders](https://arxiv.org/abs/1903.12436) | 把 VAE sampling 看作 decoder 输入噪声并研究确定性替代，但不保留原 Gaussian KL | 可作为 alpha 思想的早期背景，不能说它就是本文 exact method |
| 部分重合 | [What Do LAMs Actually Learn?](https://arxiv.org/abs/2506.15691) | 分析 controllable 与 exogenous variation，并建议 augmentation、cleaning、action prediction | availability/semantic-content 问题已有理论；本文聚焦 information 已存在却未被原 consumer 使用 |
| 部分重合 | [Latent Action Learning Requires Supervision in the Presence of Distractors](https://proceedings.mlr.press/v267/nikulin25a.html) | 证明 distractor 下少量 action supervision 对 latent 与 downstream 很重要 | GT18 锚有明确先例；必须把本文写成 partial supervision / positive control，而非纯无监督 LAM |
| 部分重合 | [CD-LAM](https://arxiv.org/abs/2607.09185) | 直接优化 embodiment/action bias、calibration，并在 2B/14B ACWM 上报告 action following | 本文不能声称更强的 debiasing 或 downstream controllability；可对比“表示内容偏差”与“consumer-use failure” |
| 部分重合 | AC-LAM、Olaf、ConLA | 分别关注 compositional action geometry、跨上下文 control-effect alignment、类别/时序先验 | `L_dir` 或 geometry 不是空白；若无下游功能提升，单纯几何变化价值低 |
| 部分重合 | CoLA-World 等 joint IDM/world-model 方法 | 通过联合训练避免学习后丢弃 FDM，直接优化 world-model conditional use | 强化 reviewer 对“只修复被丢弃 decoder 有何用”的质疑；也支持本文把 modular failure 当诊断边界 |
| 背景 | beta-VAE、free bits、target-rate、delta-VAE、lagging inference、broken-ELBO 系列 | 调节 rate/capacity 或优化平衡，主要回答 latent 是否编码信息 | 必须引用，但它们不能替代本文 availability/utilization 的因果区分 |
| 边界而非 baseline | LAPO/LAPA/VQ 离散 latent-action 方法 | 离散瓶颈、目标与训练动态不同 | 可说明适用边界，不宜用来证明 alpha 在 Gaussian LAM 中的普遍优越性 |

### 2.3 本工作的贡献强度排序

1. **paired-latent + 四格 cross-decoding 的因果定位。** A0/A1 encoder 的 `mu` 几乎相同，
   结果却随 decoder 行变化而不随 encoder 列变化，这是当前最难被 related work 直接吞没的
   实证签名。
2. **“availability 与 utilization 必须分测”的 LAM 操作化。** 抽象概念并非全新，但 GT18
   可解码、donor 干预、正对照和原 consumer rollout 组合成了清楚的 falsifiable protocol。
3. **多步与独立 consumer 的功能边界。** Tier-1.5 证明原 decoder 修复延伸到 K-step；当前
   Route-C dev 又表明 A1 `mu` 能被新 consumer 使用，因而把问题精确限定在原 decoder 的
   co-adaptation/training dynamics。
4. **alpha 作为最小因果干预。** 它能产生很强的 row effect，作为诊断工具重要；作为独立
   方法贡献弱，并受到 lambda-VAE 的直接 novelty collision。
5. **`L_dir` 的负结果。** 它说明几何 probe 改善不能替代 functional use，但更适合作为
   supporting negative result，不是主贡献。
6. **严格 split、manifest、fidelity guardrail 和审计。** 这些是可信论文的必要条件，不应
   单列成 scientific contribution。

## 3. claim 守门表

| 状态 | 可以写的 claim | 必须附带的限定 |
|---|---|---|
| 现在能说 | 在所测 DreamDojo continuous Gaussian LAM、EgoDex 和 wrist-pose 辅助锚设置中，A1 `mu` 保留可测 action information，但原 decoder 基本不使用它 | 只能说该实现/数据/监督设置，使用 conditional-use collapse，而非泛化的 posterior collapse |
| 现在能说 | `alpha=0` 在两个 LAM seeds 上修复原 decoder 的 donor sensitivity 和 K-step controllability，且 paired/cross-decode 把变化定位到 decoder | 把 alpha 称为 intervention/tool，不称新 ELBO、新 VAE 原理或更好的 action representation |
| 现在能说 | A0/A1 `mu` 在配对表征指标上近乎相同，alpha 的主要效果不是 encoder representation 改写 | 只对已测统计和样本成立，不声称数学等价 |
| 现在可作 preliminary 说 | 一个独立轻量 consumer 在 dev 上能使用 A0 和 A1 `mu`，且两者效应相近，未见 alpha-specific transfer | 明示仅一个 LAM seed、一个 consumer seed、dev-only；在 protocol 修复和 Eval-B 前不能写 confirmatory null |
| 补一个关键实验包后能说 | 同一种 available-but-ignored / decoder-side failure signature 在第二个连续 Gaussian LAM 实现上复现，是一个跨架构 recurring failure mode | 要求第二实现独立、至少 2 seeds、相同正对照与 cross-decode；仍不外推到所有 LAM/VAE |
| 补当前 Route-C confirmatory 后能说 | 新 consumer 可学习 A1 `mu`，且 A0 相对 A1 的可迁移收益小于预先冻结的实用阈值 | 必须复用 train normalization、真实 donor、每 clip 记录、层级 CI、2 LAM seeds x 2 consumer seeds 和一次性 Eval-B |
| 目前不能说 | alpha 产生更好的 exported LAM、改善 ACWM policy/action following，或优于 CD-LAM | 当前没有 alpha-specific downstream gain，现有 dev 结果反而不支持这一方向 |
| 目前不能说 | alpha 是新 VAE 方法、统一的 posterior-collapse 解法，或适用于离散/VQ LAM | lambda-VAE 与既有 collapse 文献已覆盖宽泛方法/机制，且本文证据只来自一个 Gaussian family |
| 目前不能说 | GT18 是完整 ground-truth action oracle，或 `zmu > GT18` 说明 latent action 更真实 | GT18 只覆盖 wrist pose；`mu` 还可能包含手指、物体、接触、外观和 future leakage |
| 目前不能说 | label-free disentanglement 已解决，或 `z_a` 被可识别地路由为 action | 无标签 factorization 本身不可识别，gradient norm 与 2x2 interaction 也不能证明语义归属 |

## 4. A-E 路线排序

| 排名 | 路线 | 预期信息增益 | 成本 | 成功概率 | 失败后论文价值 | 判断 |
|---:|---|---|---|---|---|---|
| 1 | A：跨架构诊断泛化 | 高 | 中 | 中 | 高；失败会清楚限定为单实现 case study | **唯一新研究优先项**；直接修复当前最大 external-validity 缺口 |
| 2 | E：真实下游 consumer | 高 | 当前 closeout 低、2B 很高 | closeout 高、2B 低 | 高；A0=A1 也是机制边界 | 当前轻量 Route-C 应严谨收尾；**不启动 2B**，因为 dev 已显示两套 `mu` 都可学且 alpha 不改变接口 |
| 3 | B：label-free 2x2 | 中 | 中高 | 低 | 中；负结果可说明不可识别性 | 暂缓。`low beta/free bits/target-rate` 是三种不同干预，不能混成一个 rate 轴；没有 routing 假设时 2x2 很难解释 |
| 4 | C：alpha/lambda exposure method | 低中 | 中 | 中高 | 低；容易被判定为调参 | 若论文保留 exposure 方法 claim，只做小型 lambda-VAE 对照；不做大 sweep/schedule |
| 5 | D：重启 `L_dir` | 低 | 中 | 低 | 低；既有负结果已经足够 | 关闭。除非提出全新的跨上下文、多步 control-effect 假设并以 exported consumer 为主，否则不再消耗 runs |

这里将 E 排第二并不意味着要“再开一条路线”。已经完成的 `GT-18D / A0 / A1` 是正确的
轻量 consumer 方向，只是需要完成 confirmatory hygiene；研究上新增的一块只选 A。

### 对 B 的更具体判断

不建议按原案直接跑 `alpha x rate` 的 label-free 2x2，原因有三点：

1. `beta`、free bits、target rate 的作用不同，事后任选一个会使 interaction 无法解释；
2. 双 latent 中若 `z_e` 是更便宜的重建通道，单靠 alpha 不会给 `z_a` 任何语义可识别性；
3. 即使观察到 `||grad_{z_a}||` 增大，也只能说明 decoder 使用该通道，不能说明它编码 controllable
   action 而不是 future appearance。

如果未来必须检验该方向，应先冻结一种明确的 rate-control 算法与 target rate，并用 effect proxy
建立 routing criterion；更干净的第一步反而是单 latent 或对称 rate 的模型，而不是一次混入三种
rate tricks。它不是当前唯一最小实验。

## 5. 唯一推荐的最小实验包

### 5.1 先关闭 Route-C，但不把它计为新路线

在任何 Eval-B unseal 前，先完成 Tier-2 文档末尾的 protocol 修订。当前两个 zmu consumer
都显著优于 shuffle/anti 且 A0/A1 接近，这已经足以做 go/no-go：**不做 2B，补齐轻量
confirmatory matrix 后收束 alpha-transfer 结论。**不过 GT18 正对照只在 signed direction 上清楚
通过；其 endpoint error 并未优于 zero/static（h=1 时约为 2.695 vs 2.591，长 horizon 也如此），
所以当前 scorer 不能笼统记为“正对照全通过”，endpoint/persistence construct 仍需校准。

### 5.2 Route A 的最小配置

**模型与 runs**

- 选一个独立的 continuous Gaussian LAM 实现，优先原生 32D trunk，而不是在当前 40D
  split 上改名字；数据可先保持相同，以隔离 architecture/training-pipeline generality。
- 保持同等级的 GT18 partial anchor，保证首先能验证 availability；label-free 是另一问题，
  不要和跨架构复现同时改变。
- `alpha in {0, 1} x 2 training seeds`，共 **4 个 LAM runs**。不要先做 sweep。
- 若单个附加 baseline 成本可承受，在同一第二架构加一个 lambda-VAE 最佳/论文默认设置，
  但它是 novelty control，不属于成功门槛；资源不足时先保证 4 个主 runs。

**固定测量**

- GT18 availability：held-out linear/nonlinear probe，并报告 task-stratified CI；
- utilization：与当前完全相同的 target / within-task shuffle / distance-matched / real
  antiparallel donor assay，先由 GT18 positive-control decoder 验证 donor construct；
- paired `mu`：cosine、CKA、Procrustes `R^2`、kNN overlap、norm difference；
- 2x2 cross-decode：`D_A0(E_A0(x))`、`D_A0(E_A1(x))`、`D_A1(E_A0(x))`、
  `D_A1(E_A1(x))`；
- Tier-1.5：至少 K={4,8} 的 rollout action following 与 fidelity/motion guardrail；
- 统计：paired per-clip 记录、task -> episode/clip hierarchical bootstrap，而非只存 aggregate mean。

**正负控制**

- 正控制：GT18-conditioned matched consumer 对 wrist-motion 指标必须通过 target vs shuffle/
  antiparallel；
- 负控制：zero、within-task shuffle、真实 magnitude-matched antiparallel donor；
- 不把 `-z_standardized` 当物理 antiparallel，只可保留为 OOD sign-flip diagnostic；
- 影像 fidelity、motion magnitude 和 first-frame consistency 必须同时过门，避免“动作更大”冒充
  controllability。

**go 门**

- 两个 seeds 均满足：A1 encoder 有明确 GT18 availability，但 A1 decoder utilization 低；
- A0 相对 A1 的 primary functional metric 改善至少达到预先冻结的实用阈值，且 paired CI 排除 0；
- cross-decode 的 decoder row effect 大，encoder column effect 小且 CI 包含零附近实用区间；
- K-step 改善不以 fidelity/motion guardrail 失败为代价。

**stop 门**

- 第二架构的 A1 本来就强烈使用 `mu`，或两个 seeds 中不能复现显著 row effect；
- alpha 效应主要变成 encoder-column/co-adaptation 效应，无法复现当前 causal signature；
- 四个主 runs 或两周共享 GPU 预算内无法完成端到端评估。

命中任一 stop 门，就停止“普遍机制/main”路线，把论文严格收窄为 DreamDojo case study 并投
workshop；不要用更多 sweep 挽救故事。

## 6. 三条最强反方审稿意见

### 拒稿理由 1：这是一次 implementation debugging，不是新研究问题

`alpha=0` 是极简单的训练选择，lambda-VAE 又给出更一般的近期形式；只在一个 LAM 实现上
展示原 decoder 没学会使用已有 `mu`，可能只是该 pipeline 的 optimization accident。没有第二
架构复现，就不足以把它提升为 LAM 的 recurring failure mode。

### 拒稿理由 2：所谓 action utilization 可能由 future leakage 和无效 donor 构造产生

`mu=E(o_t,o_{t+1})` 天然包含 future appearance。当前轻量 consumer 的 cross-task roll shuffle
和 `-standardized_z` 不是预注册的 within-task、in-distribution magnitude-matched donor；如果 eval
还用测试 manifest 自己重新估计 normalization，就有 test-distribution leakage。target 比这些 OOD
条件好，只能证明 consumer 用了 condition，不能证明它使用了可迁移 action coordinate。

### 拒稿理由 3：修复了一个会被丢弃的 decoder，却没有改善导出的 latent action

paired `mu` 几乎不变，而新 consumer 对 A0/A1 的 dev 表现相近。对 embodied/world-model
读者而言，这意味着 alpha 修复没有产生实际 downstream value；与 CD-LAM 已有 2B/14B action
following 结果相比，贡献可能停留在诊断层面。

**当前最致命的是理由 2**，因为它会动摇现有 Route-C 对“action”的构念解释，且属于必须在
Eval-B 前修复的 protocol 问题。修复后，限制 main-paper 上限的将是理由 1；对 CoRL 式 embodied
venue，理由 3 也会非常严重。因此最合理的叙事不是夸大 downstream，而是把“原 consumer
co-adaptation 可失效、probe 不能替代 functional intervention”作为诊断结论。

## 7. `L_dir` 与 GT18 的专门回答

### `L_dir`

把现有失败写成负结果并关闭该线：改变 latent cosine/方向几何而不提高 donor sensitivity、
K-step 或独立 consumer action following，正好支持“geometry/probe 不等于 utilization”的主论点。
只有同时满足以下三项才值得重启：跨上下文同一 control effect 的 alignment 提高；未见上下文
的 donor intervention 提高；独立 exported consumer 获得超过 CD-LAM/无 `L_dir` matched baseline
的功能增益。仅有线性 probe、cosine 或 cluster 更漂亮远远不够，而且这将是一个新项目。

### GT18

GT18 的正确名字是 **wrist-pose partial supervision / partial positive control**。它既可在训练中
锚住一部分可测动作，也可验证 scorer 与 donor assay 是否能检测 wrist control；不能把它称为
完整 latent-action ground truth。独立 consumer 中 `zmu` 若优于 GT18 也不代表它更“真实”，因为
`zmu` 可能携带 fingers、object/contact、camera、task identity 和 future appearance。

对 GT18 未覆盖因素，使用多种独立 effect proxies，而不是再造一个伪 oracle：hand/finger
keypoints、object tracks/SE(3)、contact onset、gripper aperture、segmented optical flow、跨视角
几何和可重复的 object displacement。每个 proxy 都要把 controllable embodiment/object effect
与 camera/background motion 分离，并使用跨上下文 donor transfer 或 held-out task 验证；报告
“哪一部分 control effect 被覆盖”，不要把多 proxy 合成未经校准的单一 action score。

## 8. 最小投稿策略

**当前可收束路线：**先按 Tier-2 修订完成轻量 consumer 的 confirmatory closeout；以
“When Available Is Not Used”式 diagnostic paper 组织，主证据是 availability/utilization 分离、
cross-decode row effect、K-step 和独立 consumer 的 non-transfer boundary。alpha 只作因果干预，
目标是高质量 workshop 或 analysis track，不承诺 superior LAM/downstream。

**升级 main 的高价值路线：**只执行 Route A 的 4-run 跨架构复现；若完整签名跨实现、跨 seed
成立，再补一个紧凑的 lambda-VAE comparison。对通用 ML venue，这可形成新的 diagnostic
failure mode；若目标是 embodied main venue，最终仍需要一个真正改善 exported interface 的方法，
而当前 alpha 不是它。

**停止条件：**第二架构两 seed 不能复现 decoder-row-dominant signature，或最小 4 runs 超过两周
共享 GPU/既定预算，就停止 main 升级；不转向 2B、alpha sweep 或 `L_dir` 来继续寻找阳性结果。
