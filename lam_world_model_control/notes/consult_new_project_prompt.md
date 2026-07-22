# 新项目专家咨询：从结构化 Latent Action 到可迁移的 World-Model Control Interface

## 一、顾问角色与本轮目标

请把自己视为 latent action learning、action-conditioned world model、机器人学习和因果表示学习
方向的资深研究者。我们正在决定是否启动一个独立于既有 `lam_disentangle` 诊断项目的新项目，
目标不是为旧结果寻找包装，而是判断下面两个研究方向是否包含足以支撑 embodied-AI main
conference 的新问题、方法和证据链。

请基于截至当前日期的论文和公开实现做事实核查，尤其关注 CD-LAM、AC-LAM、Olaf-World、
CoLA-World、SCAR、AdaWorld、LAPO/LAPA、DreamDojo 以及其他近期 latent-action/world-model
工作。请主动寻找与我们设想直接冲突的先例，不要因为项目已经投入了工作就默认它值得继续。

我们希望顾问回答三个层面的问题：

1. 研究命题是否成立、是否重要、是否已有工作基本解决；
2. 哪一种方法假设最可能产生真正改变 exported latent interface 的结果；
3. 用什么最小实验可以在投入 1B/2B world-model 训练前否决错误方向。

---

## 二、目标 claim 与证据要求

长期目标是支持下面这条有严格限定的结论：

> 在相同 world-model 架构、数据、初始化分布和训练预算下，一个更具 control sufficiency、
> context invariance、groundability 和 consumer portability 的 LAM，能够带来更好的
> action-conditioned rollout、更高效的 executable-action grounding，以及更强的规划或策略效果。

这里的“更好”不能由 LAM 自己的 reconstruction、probe 或配对 decoder 单独定义；“更好的世界
模型控制”也不能只由视频观感定义。至少需要形成：

```text
LAM-side 独立性质改善
-> matched independent consumer / ACWM action following 改善
-> executable-action adaptation、planning 或 policy utility 改善
```

我们不希望声称所有 LAM、所有 world model 或所有 embodiment 上都普遍成立。请帮助确定一个
可验证、可投稿而不过度外推的 claim 范围。

---

## 三、已有项目及其给新项目的教训

### 3.1 基础设置

既有项目建立在 DreamDojo 的约 710M continuous Gaussian LAM 上，原生 latent 为 32D；数据为
EgoDex 第一视角人类操作视频，包含 18D 双手 wrist-pose 相对变化。我们复现并参考了
[CD-LAM](https://arxiv.org/abs/2607.09185) 的 LAM-side debiasing 与 ACWM 评估逻辑。

在我们的细粒度 benchmark 中，CD-LAM 对背景、zero transition 和一般 foreground bias 有改进，
但 push/pull、open/close、insert/remove 等相反或近似反平行动作的分离与因果使用仍不理想。
这个判断来自当前数据和 scorer，不应写成 CD-LAM 在所有设置中“不能区分相反动作”。

### 3.2 已探索的 32D action + 8D environment 结构

我们曾把原生 32D latent 扩展为：

```text
z = [z_a, z_e]
z_a in R^32: 候选 action / controllable-change interface
z_e in R^8: 对 source transition 重建有帮助、但不应作为动作干预原因的 residual/environment code
```

设计动机是：两帧 reconstruction 需要解释的变化不全是 agent action。背景、相机、外观、未控制
物体或其他不可执行因素可能有利于重建，却不应混入 world model 的 action condition。与单一 KL
瓶颈强迫所有信息争夺同一 latent 相比，`z_e` 给这些 residual factors 一个显式去处，而 world
model 的控制接口只读取 `z_a`。

我们对“好的 `z_a`”给出过如下功能定义：

> `z_a` 是引起两帧变化的、跨场景不变的原因，精确到视觉可分辨粒度；它是可干预的运动学句柄，
> 不是语义标签，也不等同于 18D wrist pose。

四项可测性质为：

| 性质 | 操作性含义 | 可能的外部检验 |
|---|---|---|
| 充分 | 给定 `o_t,z_a`，包含预测所有 action-caused change 所需的信息 | held-out effect prediction、相对 full latent 的 sufficiency gap |
| 跨场景不变 | 相同动作/可控效果在不同 scene、episode、viewpoint 下使用相容坐标 | cross-context probe、donor transport、context leakage |
| 可控 | 对 `z_a` 做 target/zero/scale/swap/antiparallel 干预会以相应方式改变预测效果 | signed displacement、FDCE、object/hand tracks、multi-step rollout |
| 可接地 | 少量 executable action 可以稳定映射到/读出该空间 | low-label action bridge、held-out action readout、adaptation curve |

GT18 只覆盖 wrist pose，不包含完整的 finger/contact/force/object interaction，因此只是可接地性和
部分充分性的下界，不是完整 latent-action oracle。

### 3.3 已知问题与负结果

这条 32+8 路线目前没有被证明成功：

- split-KL 只提供了 information supply path，没有保证 nuisance 一定进入 `z_e`、action 一定进入
  `z_a`，实测存在双向 leakage；
- 无 intervention 或跨环境配对时，`z_a/z_e` 分解一般不可识别，固定 32/8 容量也可能只是人为
  partition；
- 全监督版本的 action head 直接读取 `z_a`，因此 GT18 probe 带有循环性，不能单独证明 latent
  是好的控制接口；
- label-free twins 在较强 action KL 压力下发生低-rate collapse；
- 进一步的 alpha 实验发现：关闭 LAM reconstruction decoder 所见的 posterior sampling noise
  可以显著修复该 decoder 的 action utilization，但几乎不改变 encoder `mu`；
- paired-latent 与 cross-decoding 把 alpha 效应定位在 decoder row，而一个从零训练的独立轻量
  consumer 能使用 alpha=0 和 alpha=1 的两套近似相同 `mu`，二者没有稳定性能差异。

因此 alpha 主要是旧项目中低-rate/低-SNR训练动力学的诊断性修复，不是新项目的方法起点。
这些结果也提出一个更一般的警告：paired decoder 的可控性和 latent probe 都不等价于 exported
interface quality。

### 3.4 请顾问判断的结构化 latent 问题

请不要预设 32+8 设计有价值。重点判断：

1. `z_e` 是否是合理的 exogenous/residual variable，还是会成为 future-information leakage 的
   廉价通道并破坏 action identification？
2. `z_e` 应只供 LAM reconstruction 使用、作为 world model 的 stochastic exogenous condition，
   还是最终完全丢弃？这三种语义对应不同模型和 claim。
3. “充分、跨场景不变、可控、可接地”四项是否足以定义 exported action interface？是否还需要
   composition、minimality、uncertainty 或 embodiment equivariance？
4. 在 observation-only video 中，哪些性质原则上不可识别？需要何种最少 supervision、paired
   environments、multi-view、robot action 或 intervention 才能识别？
5. 应继续研究 32+8 factorization，改成 object/effect/residual 等其他结构，还是放弃显式 split？
6. 如果保留，该结构相对 CD-LAM、DiLA、AC-LAM、SCAR 等工作的真正 novelty 是什么？

---

## 四、候选新角度：面向未见 Consumer 的 Portable Latent Action

### 4.1 核心观察与假设

标准 LAM 优化一个 encoder-decoder pair：

```text
min_{E,D} L_recon(D(o_t, E(o_t,o_{t+1})), o_{t+1})
```

它只保证 `E` 与共同训练的 `D` 可以形成有效 private code。一个 latent 可以 reconstruction 好、
probe 好、甚至被自己的 decoder 因果使用，却仍可能难以被新的 world model 学习。相反，旧项目的
独立 consumer 结果也说明，配对 decoder 的失败不代表 `mu` 固有地不可用。

候选新定义是：

> 好的 LAM 应让一个未与它共同训练的 consumer，在固定数据量和固定 adaptation budget 下，
> 快速、稳定地学会正确使用 latent condition，并在新 context 中执行相应的 controllable effect。

可以把目标风险写为：

```text
R_port(E) = E_{C ~ P(C)} [ L_ctrl(C_{phi_K(E)}, E; V) ]
phi_K(E) = K-step-Adapt(C, latent_dataset(E); T)
```

`C` 来自 consumer/conditioning architecture 的分布，`T/V` 是不相交的 adaptation/evaluation
episodes。优化对象是 encoder/exported latent，而不是让某一个大 world model 与其联合共适应。

请顾问重点判断：这个 held-out-consumer risk 是否是一个有意义且相对新的 LAM 定义，还是只是
meta-learning/linear-accessibility 的昂贵改写。

### 4.2 候选方法设计

一个尚未验证的实现方案如下：

1. 从健康的原生 32D continuous LAM 或 CD-LAM/free-bits trunk 出发，不使用旧项目的强瓶颈；
2. 建立多个便宜且异构的 surrogate consumers，例如 Conv-UNet、Transformer predictor，以及
   FiLM、cross-attention、additive adapter 等不同 conditioning paths；
3. episodically 重置 conditioning adapter，或对其只做 K 步 inner-loop adaptation，减少 encoder
   与单个 consumer 长期形成 private code；
4. outer loop 在 held-out episode/context 上更新 LAM encoder，使新 consumer 的 condition-use
   和 action-effect prediction 在固定 K 步后最好；
5. outer control loss 使用 target donor 相对 within-task shuffle、zero 和真实 magnitude-matched
   antiparallel donor 的差，而不是只优化 source-pair pixel reconstruction；
6. 使用 wrist、hand、object tracks、contact、foreground displacement 等多个 effect proxies，
   并抑制 background、camera、appearance、episode identity 等 nuisance；
7. 保留 base reconstruction、capacity calibration 和 fidelity guardrails；最终 target ACWM 架构、
   checkpoint 和 evaluation tasks 在 LAM 训练期间完全 held out。

候选总目标可写成：

```text
L = L_base
  + lambda_port   L_heldout_consumer
  + lambda_effect L_donor_effect_transport
  + lambda_nuis   L_context_invariance
```

这里 `L_heldout_consumer` 才是候选核心贡献。`L_effect` 与 `L_nuis` 都有大量先例，不能单独作为
novelty。还需要顾问判断是否有必要做真正的 bi-level gradient，还是 consumer adapter churn、
first-order meta-learning 或 representation-conditioned training 已足够。

### 4.3 与现有工作的已知边界

| 工作 | 已覆盖内容 | 本方向若成立必须与其区分之处 |
|---|---|---|
| [CD-LAM](https://arxiv.org/abs/2607.09185) | embodiment reconstruction、coarse action contrast、zero/capacity calibration，并验证 2B/14B ACWM | 不能再泛称 causal debiasing；应验证未见 consumer 的固定预算适配风险 |
| [AC-LAM](https://arxiv.org/abs/2604.03340) | identity/inverse/cycle 与 additive composition，报告下游 policy improvement | 不能把 latent algebra/方向结构本身作为唯一 novelty |
| [Olaf-World](https://arxiv.org/abs/2602.10104) | 跨 context control-effect alignment、zero-shot action transfer、低标签适配 | 是最接近的 collision；必须证明 held-out consumer/architecture portability 不是其已有 protocol 的换名 |
| [CoLA-World](https://arxiv.org/abs/2510.26433) | LAM 与一个 world model 的 joint co-evolution | 本方向强调 target consumer 未参与 LAM 训练；若最终仍依赖指定 ACWM 共适应，则 novelty 消失 |
| [SCAR](https://arxiv.org/abs/2605.16412) | adversarial invariance 与跨 embodiment continuous action representation | 不能只声称 embodiment invariance 或 unified action representation |
| [AdaWorld](https://openreview.net/forum?id=QQegZj99sk) | latent-action world-model adaptation 与跨 context action reuse | 必须比较相同 adaptation budget 下的 learning curve，而不只是展示 transfer 视频 |

请继续检索是否已经存在直接最小化 new-decoder/new-world-model adaptation risk 的 LAM 工作。
如果已有直接先例，请据此否决或重构该方向。

### 4.4 证明“更好 LAM -> 更好世界模型”的实验链

**第一层：LAM/exported-interface evidence**

- consumer architecture 和 evaluation task 均 held out；
- 固定 K-step 后的 action-following、adaptation-curve AUC 和 seed variance；
- cross-context donor effect transport；
- GT18/robot action 的 low-label grounding curve；
- context/episode/camera leakage 与 latent rate；
- 证明改变发生在逐样本 exported `mu`，而非只发生在配对 decoder。

**第二层：matched independent ACWM**

- 只改变 LAM 产生的 latent dataset；
- 固定 ACWM architecture、initialization distribution、clips、steps、optimizer 和 adapter；
- 至少两个 downstream training seeds；
- CD-LAM/strong baseline、候选方法、GT-action upper bound、shuffle/zero controls；
- primary 使用 FDCE、signed wrist/object displacement 和 target-action transfer；
- PSNR/SSIM/LPIPS、motion magnitude、first-frame consistency 仅作 fidelity/degeneracy guardrails。

**第三层：embodied utility**

至少需要一项不只是 rollout image metric 的结果：

- executable robot-action bridge 在不同 label/step budgets 下的 sample efficiency；
- world-model planning success 或 policy-ranking agreement；
- 使用 world-model rollout 进行数据增强后的 policy improvement；
- held-out task/embodiment 的 closed-loop success。

如果只有 FDCE 而没有 planning/policy/grounding utility，请顾问判断是否足以进入目标 venue。

### 4.5 分阶段 go/no-go 草案

**Phase A：便宜 falsification**

- 使用健康的 32D trunk，不复用旧 40D low-rate failure；
- 比较 CD-LAM/strong baseline 与 portable-objective candidate；
- 使用至少两种 surrogate consumers，并保留一个训练时完全未见的 consumer architecture；
- 每个关键臂至少两个 LAM/consumer seeds；
- 预先冻结 primary action-effect metric、fidelity guardrail 和 practical margin。

建议 go 门：candidate 的 exported `mu` 确有变化；在 held-out consumer、held-out task/context 上，
固定 adaptation budget 后的 action-following 至少改善 10%--15%，层级 CI 排除 0，且 context
leakage/fidelity 不恶化。否则停止，不进入大模型。

**Phase B：中型 ACWM**

- 300M--1.3B 级独立 world model；
- baseline/candidate matched training，至少两个 downstream seeds；
- action following、adaptation AUC 和跨 task transfer 必须共同改善。

若只有 reconstruction/fidelity 改善，或控制收益依赖 target ACWM 参与 LAM 训练，则停止
“portable LAM”claim。

**Phase C：main-paper evidence**

- 仅在 Phase B 通过后运行 2B 级 matched ACWM；
- 最小核心矩阵可为 CD-LAM vs candidate，各两个 ACWM seeds；
- 加 robot-action label/step scaling 和至少一个 planning/policy utility；
- 若资源不允许 matched 大模型与 embodied utility，不应预先承诺 main claim。

---

## 五、两条方向的关系仍是开放问题

可能存在四种结论，请顾问明确排序而非默认组合：

1. **只做结构化 split：**32+8 本身提供合理 inductive bias，consumer portability 只是评估；
2. **只做 portable objective：**显式 `z_a/z_e` 不可识别且没有必要，让 held-out-consumer risk
   直接塑造单一 latent；
3. **组合：**`z_a` 接受 portable/control objective，`z_e` 只承担不可控 residual，并通过
   intervention/independence 约束防 leakage；
4. **两者都不做：**近期工作已经覆盖核心思想，或者资源条件无法形成 main 级证据链。

组合两条路线会增加解释和消融负担。请判断是否应先用单一 latent 验证 portability，再决定是否
加入 split，而不是一次改变 latent structure、loss、consumer training 和监督信号。

---

## 六、资源和已有资产

- 已有 DreamDojo 约 710M LAM、CD-LAM 论文/公开配置和 EgoDex pipeline；
- 已有 18D wrist-pose、donor manifests、paired-latent/cross-decode 工具和轻量 8.6M conditional
  consumer；
- 已有 action availability、condition utilization、K-step rollout 与部分 effect scorer；
- GT18 之外的 object/contact proxy 和严格 target-action transfer protocol 仍需建设；
- 共享 GPU，2B ACWM matched runs 成本高，因此必须有便宜、严格的 Phase-A gate；
- 旧项目的 alpha 结果可作 motivation/negative boundary，不应成为新方法的正结果。

---

## 七、请顾问作出的具体判断

1. “consumer portability / held-out-consumer risk”是否是重要且相对未被占据的问题？最直接的
   novelty collision 是哪篇工作？
2. 32+8 `z_a/z_e` 分解是否有可识别、可测试的科学价值？`z_e` 的正确概率/因果语义是什么？
3. 四项 `z_a` 定义是否充分？请修改定义并指出哪些性质彼此冲突。
4. 在“结构化 split、portable objective、二者组合、放弃”四种选择中，请只推荐一个首要方向。
5. 如果选择 portable objective，最小且技术上可行的 loss/训练算法是什么？需要真正 bi-level
   optimization 吗？
6. 哪些 supervision/proxy 可以用于训练，哪些必须留作独立评估，才能避免循环性？
7. 最便宜的 falsification experiment 是什么？请给模型数、seed、数据 split、指标和 stop/go 门。
8. 要支持 embodied main，最小 matched ACWM 与 planning/policy evidence 是什么？FDCE 是否足够？
9. 必须纳入哪些强 baseline：CD-LAM、Olaf-World、AC-LAM、SCAR、AdaWorld 中哪些不可缺？
10. 请写出三条最强拒稿理由，并判断在当前资源下该项目的成功概率和建议 venue。

---

## 八、希望采用的回复格式

1. **六句话 verdict**：问题价值、主要 collision、推荐路线、不推荐路线、最低证据、总体 go/no-go；
2. **related-work collision 表**：直接先例、部分重合、仅背景，并附原始来源；
3. **两条路线裁决表**：结构化 split 与 portable objective 的假设、可识别性、成本、失败价值；
4. **唯一推荐方法**：给出最小 objective、训练数据流和为什么可能改变 exported `mu`；
5. **分阶段实验包**：Phase A/B/C 的 runs、seeds、controls、metrics 和 stop/go；
6. **claim 守门表**：什么结果后能说什么，哪些话即使成功也不能说；
7. **反方审稿**：三条最强拒稿理由和最致命一条；
8. **资源裁决**：在共享 GPU 约束下，明确建议启动、缩小还是放弃。

请保持自我怀疑：区分事实、基于现有证据的推断和未经验证的设计直觉；对新的 loss 不要只讨论
为什么可能成功，也要说明它最可能通过什么 shortcut 失败。若两个方向都不足以支撑 main，请直接
建议停止，而不是为了延续项目给出更多消融。
