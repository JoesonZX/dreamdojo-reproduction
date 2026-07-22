# 咨询提问稿:label-free 主线遭遇塌缩后的故事定位(2026-07-20)

以下为发给外部顾问(ChatGPT 等)的自包含提问稿。仓库内读者:结论回来后记录在本文件末尾。

---

我在做一篇机器人学习方向的论文,遇到一个"故事定位 × 实验成本"的决策问题。请你作为
熟悉表征学习、世界模型和顶会审稿的研究者,帮我分析。背景较长但都是必要信息。

## 一、研究背景

**课题**:Latent Action Model(LAM)的精细化。LAM 从视频帧对 (o_t, o_{t+1}) 无监督
编码一个 latent action z,作为世界模型(action-conditioned world model)的条件通道,
z 的质量决定世界模型"听不听指挥"。

**我们的设定**:在 NVIDIA DreamDojo 的预训练 LAM(~700M 参数,latent 32 维)上
微调 5000 步(小 lr),数据是 EgoDex(第一视角人类操作视频,含 18 维双手腕部位姿
真值,可作 probe 目标)。我们把 latent 扩到 40 维 = 32 动作维(z_a)+ 8 环境维(z_e),
用非对称 KL(split-KL:β_a=3e-3 压动作子空间做信息瓶颈,β_e=1e-6 放松环境子空间),
并加一个 18 维动作回归头(λ_action=1.0)。基线家族在此 trunk 上加不同的自监督 loss。

**论文一句话贡献(已定)**:"最强的已发表 LAM 去偏方法(CD-LAM,NVIDIA 的后续工作)
修正了一般性混淆之后,**运动方向仍是条件通道中未编码的自由度**;我们用真实数据的
运动学反平行帧对将其注入,并以非循环评估协议证明。"

**方向失败的证据(非常扎实,三臂闭合)**:我们的 direction_gain 指标 = 相反动作分类
准确率 − 静止帧对照(只有超过"仅凭初始帧外观"的部分才算方向证据)。结果:
CD-LAM 论文配方在我们 trunk 上的复现 = −0.287;CD-LAM 官方 checkpoint(与我们同一
预训练底座,可做干净 before/after)= −0.241;我们全家族 ×3 seeds = −0.24~−0.28
(range ≤0.024)。即:没有任何模型、任何 seed、任何标签空间编码了方向,
表面上 70%+ 的相反动作可分性全部可被初始帧外观解释。

**计划中的方法(stage-1,尚未实施)**:L_dir——在 episode 内部挖"运动反平行"的帧对
(伸手 vs 收手、推进 vs 回撤),用自监督运动代理(光流方向/手部关键点轨迹)判定
反平行,配 margin loss 把方向注入 z_a。注意:**L_dir 本身不需要任何标签**
(不需要动词标签,更不需要 18D 真值)。

## 二、故事定位问题的由来

我们原本选定的定位是**"自监督 + 弱标签"**(下称故事 b):主线方法不依赖 18D 强标签
(caption 动词等弱标签允许)。理由:(1) CD-LAM 的 Stage-1 明确定位是
"action-unlabeled video",我们要与它公平可比;(2) 我们家族全员带 18D 回归头
⇒ probe R² 是"准循环"指标(表示被训练成可线性解码),外部主张必须落在没被监督
直接优化的指标上;若主线 label-free,R² 就变成完全外部的指标,主张更硬。

**计划中的验证**:训练两个"label-free 孪生"(把 18D 回归头去掉、其余不变),
看 z_a 的动作信息掉多少——"掉得少 ⇒ 主线可以 label-free;掉得多 ⇒ 说明路由靠监督
锚定,故事要重新讨论"。

## 三、关键实验结果(决策的直接输入)

**lf 孪生双双彻底塌缩**:
- kl_full_lf(只去监督头):有效秩 **2.8/40**(监督版 13.4),probe R² = **−0.002**
  (监督版 0.308,连原始预训练模型的 0.093 都不如);对解码器做干预测试
  (z_a 置零 / 换别人的 z_a / 换相反动作的 z_a),**解码输出纹丝不动**——解码器
  完全无视 z_a,仅靠 o_t 重建。训练本身正常收敛(重建 mse 正常)。
- r2_lf(去监督头,但保留两个自监督 loss:零动作校准 L_zero + 相反动词 margin):
  塌缩**减缓但未阻止**(秩 6.7,R² 0.024,干预响应 ≈0)。

**机制诊断**(与我们早期一个 decorrelation 塌缩事故的教训同构):β_a=3e-3 的信息
瓶颈"只删动作 loss 不保护的信息"——18D 监督一直是 z_a 唯一的防塌锚。拔掉它,
没有任何 loss 替 z_a 付 KL 的钱,瓶颈把它压死;解码器学会只靠 o_t 复制+糊化运动区。
旁证 1:原始 DreamDojo LAM β=1e-6(无瓶颈),不塌。
旁证 2:**CD-LAM 的 Stage-1 也是 action-free 的,却安全**——他们用 free-bits KL
(每维 KL 低于地板免罚)+ 训练停机门槛(有效秩跌幅 >20% 即停、每维 KL 均值 <0.05
即停)。他们的公开配方证明:action-free 的去偏微调,防塌机制是**必需组件**。

**另外两个相关事实**:
- 我们带监督的 trunk 本身也有轻度压秩问题(有效秩 13–14/32,去均值后残余成对
  余弦 0.10 = 低维子空间聚集)——free-bits 恰好也是这个问题的教科书解法;
- 我们刚发现全家族的 lr 调度有 bug(2 卡下调度推快 2 倍,lr 呈 V 形:中途降到 0
  再回升到满 lr)。家族内比较不受损(人人相同),但**如果重建 trunk,顺带修调度
  并重跑基线的边际成本为零**(反正要重跑)。

**算力约束**:共享服务器,2×48G GPU 可用时每个 5k 步训练 ≈5.5h。重跑基线家族
(2–3 个模型 × 可能 2–3 个 seed)+ free-bits 调参探索(地板值 τ 至少试 2 档)
≈ 1–2 周的机器时间。论文目标周期 ~2 个月。

## 四、摆在桌上的选项

**选项 (a) 重建 label-free trunk,保住故事 b**:
split-KL + 每维 free-bits 地板(或 β_a 降回 ~1e-6 + 靠自监督 loss 群塑形),
重训 lf 基线家族(顺带修调度 bug)。
- 收益:主线纯 label-free,与 CD-LAM 定位完全对齐;R² 变成完全外部指标,所有主张
  更硬;低秩问题一并解决。
- 风险:free-bits 地板值要调(塌缩 vs 瓶颈失效之间的窗口未知,可能不止 1–2 个 run);
  时间线膨胀;L_dir 等后续 loss 在新 trunk 上的行为要重新验证;
  且**不能保证** free-bits lf trunk 的动作信息量够(CD-LAM 官方 checkpoint 的
  外部 R² 只有 0.104,和原始模型 0.093 差不多——action-free 训练本来就不产生
  高线性可解码性,这可能是本质而非工程问题)。

**选项 (b) 主线保留监督,"supervision-as-scaffold"定位**:
把"18D 监督是分区瓶颈下唯一的防塌锚"本身升格为一个发现(lf 塌缩实验 = 量化证据:
监督贡献的不是一部分路由,是全部);主线故事改为"用廉价的 18D 腕部位姿监督做
表征脚手架 + 用 label-free 的 L_dir 注入方向";外部主张完全落在非循环指标
(direction_gain、干预/swap 测试、held-out task 泛化)上,R² 仅作家族内受控比较。
- 收益:零额外训练成本;诚实(实验结果本来就是这样);L_dir 的 label-free 性质
  不受影响(它不依赖 trunk 有没有监督头)。
- 风险:审稿人问"你号称方向 loss 是 label-free 的,但整个表征靠 18D 监督撑着,
  这算什么 label-free?";与 CD-LAM(action-unlabeled)的可比性论证变弱——虽然
  我们保留了"CD-LAM 配方复现时也加了同样的监督头"这个受控设计,家族内公平;
  18D 标签在很多数据集上没有(EgoDex 有是特例),方法的可移植性叙事受损。

**选项 (c) 折中**:主线按 (b) 走,另做**一个**(不是一族)free-bits lf trunk 的
验证 run:证明"L_dir 在无监督 trunk 上同样把 direction_gain 推正"——作为泛化性
证据放在消融/分析章,不承担主线。成本 ~2–3 个 run。

## 五、请你回答的具体问题

1. 从顶会(CoRL/ICLR/RSS 级别)审稿人视角,(a)(b)(c) 哪个故事最站得住?
   "supervision-as-scaffold"这种框法是会被读成"诚实的发现"还是"给捷径找借口"?
   有没有已发表工作用过类似框法(监督仅作表征锚、贡献点在别处)可以引用?
2. 如果走 (b),"label-free 的 L_dir + 带监督的 trunk"这个组合的正确表述方式是
   什么?怎么划清"监督参与了什么/没参与什么"的边界,才能挡住"你的方法其实依赖
   强标签"的攻击?(注:L_dir 的配对、损失、验收指标确实都不碰 18D 标签;
   但 z_a 的存活依赖监督。)
3. 如果走 (a),free-bits 地板 + split-KL 的组合有没有已知的坑?调参的最小实验
   设计怎么排(先 1k 步烟测扫地板值再挑一档跑 5k?)?判断"lf trunk 健康"的
   最小指标集是什么(我们现有:有效秩、每维 KL、干预响应、外部 probe R²)?
4. 有没有第四条路:在"弱标签允许"的边界内,用别的自监督/弱监督信号替代 18D 头
   做防塌锚?候选:(i) 12 类粗动词的对比 loss(CD-LAM 的 L_ctr,我们已实现)——
   但它是 clip 级弱信号,防塌强度存疑;(ii) 光流/运动代理回归头(预测帧间光流
   统计量而非 18D 位姿)——自监督、且直接锚运动信息;(iii) latent 空间预测
  (V-JEPA 式)。哪个最值得先试?
5. CD-LAM 官方 checkpoint 的外部 probe R² 只有 0.104(≈原始模型)。这是否说明
   "action-free 去偏本来就不提升线性可解码性",从而故事 b 的"lf 版 R² 不塌"
   这个预期本身设错了?如果是,lf 线的正确成功标准应该是什么(干预响应?下游
   世界模型的 action-following?)?
6. 综合算力约束(1–2 周机器时间的余量)和 2 个月论文周期,你会怎么排这个决策的
   实验顺序,使得"最少的 run 消除最大的不确定性"?

---

## 顾问回复记录

核查日期:2026-07-20。以下判断基于本地论文、官方源码当前 `main`
(`2e730ad`)和本仓库现有 checkpoint/benchmark 产物。CD-LAM 在本文中应称为
**2026-07 的 arXiv 预印本**，不是已经同行评审的顶会论文。

### 0. 结论先行

1. **选 (c)，但必须改成配对版 `c+`**:主线先按带 18D pose anchor 的
   semi-supervised / pose-regularized 方法推进；另选定一个健康的 LF trunk，在同一
   checkpoint 上分叉成 `LF-base` 和 `LF+L_dir`。只跑一个 `LF+L_dir`、没有匹配
   LF baseline，不能证明增益来自 `L_dir`。
2. 不要把主线叫 `label-free LAM`，也不要把 18D 监督包装成无关紧要的
   scaffold。准确说法是:**完整训练方法依赖 18D wrist-pose 辅助监督；`L_dir`
   本身不使用目标数据集的 18D/action annotation。**
3. 在花 GPU 重建 trunk 之前，先修正 headline metric。当前
   `direction_gain=A_transition-A_static` **不能证明方向没有编码**。它可靠地说明
   transition latent 没有胜过静态外观捷径，但不是条件信息增益。
4. free-bits 值得试，但不是“高秩的教科书保证”。它只撤销 KL 地板以下的梯度，
   不会主动把塌掉的维度拉起来，也不能保证 decoder 使用 `z_a`。
5. 第四条路线优先级是:**前景/手部中心的运动代理回归 + 显式 variance/decoder-use
   防塌约束** > 细分方向后的粗动词弱监督 > JEPA 式大改。
6. CD-LAM 的 `R2=0.104` 是 MLP probe，只支持“这个公开 checkpoint 在这个 probe
   下没有明显提升”，不支持“action-unlabeled 方法本质上不能提高可解码性”。LF
   的首要成功标准应是 decoder 真实使用 latent 和下游 action-following，而不是
   R2 必须上升。

### 1. 事实核查

| 原判断 | 核查结论 | 证据与应改表述 |
|---|---|---|
| CD-LAM 是 NVIDIA 的后续工作/已发表工作 | **错误** | 论文作者单位是 Aether AI 和 UC San Diego；它建立在 NVIDIA DreamDojo 上，但不是 NVIDIA 工作。当前是 2026-07-10 提交的 [arXiv 预印本](https://arxiv.org/abs/2607.09185)。写“CD-LAM 预印本”而非“已发表 CD-LAM”。 |
| Stage-1 是 label-free | **过宽** | 它不使用 executable robot action labels，但使用 caption 派生的 12 类 primitive 弱标签和 SAM3 masks。官方也采用 `action-unlabeled` 这个窄定义，见 [pipeline 文档](https://github.com/yufanwei/CD-LAM/blob/main/docs/PIPELINE.md)及 [Stage-1 配方](https://github.com/yufanwei/CD-LAM/blob/main/configs/stage1_recipe.yaml)。应写“action-unlabeled / 无可执行动作标签”，不能写“无标签”。 |
| CD-LAM 用每维 free-bits，`tau=0.5` | **正确** | 官方实现是 `sum_j max(E_batch KL_j, 0.5)`，见 [KL 实现](https://github.com/yufanwei/CD-LAM/blob/main/third_party/acwm_overlay/cdlam_integration/lam/optimizer_helpers.py)。其 KL beta 从 `1e-4` ramp 到 `1e-3`，不能把 `tau=0.5` 原样视为本项目 `beta_a=3e-3` 下的最优值。 |
| 有效秩跌幅 >20% 会停 | **正确** | `eff_rank_drop_frac_max: 0.20` 在配置中，当前 trainer 也实际读取并执行。 |
| 每维 KL 均值 <0.05 会停 | **当前源码下错误** | `kl_per_dim_mean_min: 0.05` 虽写在 YAML，但当前 `train.py` 没有读取它。不能把一个 inert config key 写成已生效机制。warmup gate 实际检查的是 latent norm、总 KL 和 usage gap；详见 [trainer](https://github.com/yufanwei/CD-LAM/blob/main/third_party/acwm_overlay/cdlam_integration/lam/train.py)。 |
| CD-LAM 公开配方证明 free-bits 是安全的原因/必要组件 | **证据不足** | 配方表明 CD-LAM 使用了 free-bits 和健康门，但没有把“未塌缩”因果归因到 free-bits 的公开消融。可写“CD-LAM 提供了可行先例”，不能写“证明它是必要且充分的原因”。 |
| 官方 checkpoint 等价于论文完整 Stage-1 | **错误** | 本地下载权重 metadata 是 step 300，而论文/官方 pipeline 的 Stage-1 protocol 是 1000 steps。官方 [model card](https://huggingface.co/yufanwei/CD-LAM)也明确公开项是 research artifacts，不单独构成完整 headline-table reproduction。因此这个 checkpoint 是很好的公开锚点，但不是完整论文结论的替身。 |
| `raw_lam -> cdlam_official` 是同 base 的 before/after | **本地证据强，成立** | 本地已做 839/839 同名同形张量核对、相对 L2 距离 0.0015，且 metadata 指向 DreamDojo `iter_000140000`。该判断可用于当前 benchmark，但应同时披露公开权重只有 300 steps。 |
| 三臂 `direction_gain<0` 证明方向没有编码 | **不成立** | 本地值可核对且跨 seed 稳定，但指标解释过强，见下一节。可写“所有模型均未通过当前静态基线差值测试”，不能写“证明方向这个自由度不存在”。 |
| LF twins 真塌缩 | **成立，高置信** | `rank=2.8/6.7`、外部 R2 近零、shuffle/zero/swap decoder 响应同时近零，多个互补指标一致。这里不是单一 probe 假阴性。 |
| `R2=0.104` 衡量线性可解码性 | **错误** | `0.104` 是两层 `256x256` ReLU `MLPRegressor` 的 `mlp_action_r2`。真正的 linear counterpart 是 `action_r2_full_latent`:官方 CD-LAM `0.1266`、raw LAM `0.1244`，差值只有 `0.0022`。后文应分别称 MLP probe 和 linear probe。 |
| `collapse_flag` 可作 LF 健康门 | **当前不成立** | 当前代码只用 `mlp_action_r2 < 0.1` 置 flag，因而把 rank 26 的 raw LAM 也标成 collapse。LF 健康判断必须改成与 18D 标签无关的结构/使用指标；R2 另列。 |
| 相同 V 形 LR 下家族比较完全不受损 | **表述过强** | 相同错误调度保留了 matched-protocol 的一部分可比性，但不同 objective 可能对末段升 LR 有不同响应，不能保证排名不变；1-GPU arm 更不是同 regime。修复后至少要重跑关键 baseline，而非宣称旧比较无偏。 |
| 光流/关键点版 `L_dir` 是 label-free | **需要限定** | 它可以不使用目标数据的 18D/action labels；但外部 flow/keypoint 模型可能在别处用过监督。最稳妥写法是 `target-action-annotation-free`，并在 signal accounting 中披露 proxy 的预训练来源。 |

还有一个版本问题:本地 PDF 写“约 30x / 3%”，当前 arXiv 摘要写“more than
12x / 6k steps”。引用性能数字时要绑定 PDF/arXiv 版本，不能混写。

### 2. 必须先修正 `direction_gain`

当前实现分别训练两个 logistic probe:

```text
A_transition = probe(z(o_t, o_t+1) -> opposite verb)
A_static     = probe(z(o_t, o_t)   -> opposite verb)
direction_gain = A_transition - A_static
```

这不是条件信息量 `I(Y; Z_transition | Z_static)`。本地 `A_static=0.960-0.994`，
所以正增益的理论余量只有 `0.006-0.040`；transition latent 即使同时去掉部分外观并
加入方向信息，也可能从 0.98 降到 0.90，得到负 gain。两个独立 probe 的准确率之差
既有天花板效应，也没有显式条件化。相关 probe 文献明确区分“与 baseline 比性能”
和“在 baseline 已知后增加可用信息”，见 [Conditional Probing, EMNLP 2021](https://aclanthology.org/2021.emnlp-main.122/)。

建议把旧指标重命名为 `static_shortcut_gap`，保留作泄漏诊断；headline 改为:

1. 固定同一 episode-grouped split，baseline probe 输入 `[Z_static; 0]`，完整 probe
   输入 `[Z_static; Z_transition]` 或 `[Z_static; Z_transition-Z_static]`。
2. 报告 held-out **log-loss/NLL 的下降**，不要只报接近饱和的 balanced accuracy；
   两个 probe 用同一模型族、嵌套 CV 选正则。
3. 对样本做 paired bootstrap，并在 episode/task 内 permutation，报告 95% CI。
4. 若可能，增加初始状态匹配的 opposite outcomes；仅“同 task”并不等于同 scene。
5. 最终仍以 opposite-latent swap、zero/shuffle intervention 和 Tier-2 action-following
   裁决。probe 只说明可读出，不说明 decoder 因果使用。

因此现阶段最硬的结论是:“CD-LAM 与本项目模型都保留强初始状态捷径，且没有在旧
差值测试和 decoder-side opposite intervention 上展示出可信的方向增益。”这已经
足够 motivate `L_dir`，但不要写“方向必然未编码”。

### 3. 问题 1:选哪个故事，scaffold 是否站得住

**推荐 `c+`，主线实质上是 (b)，LF 做配对泛化验证。**理由如下:

- (a) 在语义上最干净，但它把两个月项目的最大风险从“`L_dir` 是否有效”换成
  “能否先造出健康的无动作标签 trunk”。这不是当前贡献的最短路径。
- (b) 科学上可以成立，但只能作为 supervised/semi-supervised 方法。把强标签依赖
  隐藏在 `scaffold` 这个词后面，会被读成规避边界；主动展示 `0% label` 时塌缩，
  并把它解释为 bottleneck 下的 rate starvation，反而是诚实且有价值的分析。
- 一个未配对的 LF `+L_dir` run 不能支撑 (c)。最低限度是同一个健康 LF checkpoint
  分叉 `base/+L_dir`，否则 trunk 选择、free-bits 和 `L_dir` 三个因素混在一起。

最接近的已有结果不是泛泛的 privileged learning，而是 ICML 2025 的
[Latent Action Learning Requires Supervision in the Presence of Distractors](https://proceedings.mlr.press/v267/nikulin25a.html):
作者发现只在 LAM 训练中加入 2.5% 动作标签，平均下游性能可提高 4.2x。这直接支持
“distractor 下少量监督可作表征锚”，也说明把监督加入 LAM 不是不可发表的污点。
[Learning by Cheating](https://proceedings.mlr.press/v100/chen20a.html) 和
[APRiL](https://proceedings.mlr.press/v155/salter21a.html)也提供 training-only
privileged information 的先例，但要注明区别:那些工作明确让 deployed policy
脱离 privileged input；本项目虽然推理时不输入 18D，**训练方法仍依赖它**。

若 `L_dir` 主结果成立后还有余量，最有价值的补充不是重跑整个 LF 家族，而是做
`{0%, 2.5%, 10%, 100%}` pose-label fraction 小消融。它能把“scaffold”从修辞变成
可量化的 supervision-efficiency 结果；在主结果未成立前不要做。

### 4. 问题 2:正确表述与监督边界

推荐方法描述:

> We introduce a target-action-annotation-free directional regularizer, `L_dir`,
> and evaluate it on a pose-regularized latent-action trunk. The full training
> objective uses 18-D wrist-pose labels through an auxiliary readout; `L_dir`'s
> pair mining and optimization never access these labels. Removing the pose
> anchor under our split-KL bottleneck causes posterior and conditional-use
> collapse. Therefore, the full training pipeline is supervised, while the
> proposed directional objective is target-action-annotation-free.

中文一句话可写:

> 我们在腕部位姿监督稳定的 LAM trunk 上，引入一个不访问目标 18D 动作标注的
> 运动方向正则；完整训练方法是半监督/pose-regularized，而非 label-free。

建议在论文放一张 signal accounting 表，而不是靠段落辩护:

| 信号 | 训练 | 模型选择 | 最终评估 | 边界 |
|---|---:|---:|---:|---|
| 18D wrist pose | 主线 `L_pose` 使用 | 主线可使用 | R2 使用 | 完整主线依赖强标签 |
| episode 内 flow/keypoint | `L_dir` 使用 | 可使用 proxy 质量门 | 不应作为唯一 headline | 无目标动作标注；披露 proxy 预训练监督 |
| caption/12-way primitive | `L_ctr` 使用 | 可使用 | verb probe 使用 | 弱监督，不是无标签 |
| SAM3 mask | `L_emb`/pair mining 使用 | 可使用 | FDCE 也使用时须说明共享依赖 | 伪标签/外部模型依赖 |
| 18D R2 | LF 训练禁用 | **LF 的 tau/ckpt 选择禁用** | 最终揭盲一次 | 否则 label-free model selection 被污染 |
| swap/intervention/Tier-2 | 不进训练 | 可预注册作门 | headline | 更接近因果使用，但需独立数据和 CI |

不要写“18D 很廉价”，除非有真实 annotation-time/cost 数据。可写“EgoDex 已提供，
因此在本数据集没有新增标注成本；跨数据集可用性有限”。“同一个监督头保证公平”只
能说明 pose-supervised 家族内控制一致；它不能把对 CD-LAM 的改造版变成 faithful
method-level comparison。应分别报告:

- 官方 CD-LAM checkpoint:外部 action-unlabeled anchor；
- `CD-LAM objectives + common pose head`:同 trunk 的受控 objective ablation。

### 5. 问题 3:free-bits + split-KL 的坑和最小实验

本项目建议先只对 `z_a` 加地板:

```text
L_KL = beta_a * sum_j max(E_batch KL_a,j, tau_a)
     + beta_e * sum_k E_batch KL_e,k
```

第一轮令 `tau_e=0`。否则主动给 `z_e` 免费容量，可能保住的恰好是相机/背景 nuisance，
削弱 split 的目的。

主要风险:

1. **free-bits 是 dead zone，不是 minimum-rate reward。**当原始 KL 小于 `tau`，
   clamp 后该维 KL 梯度为零；它不会产生把 KL 从 0 拉到 `tau` 的梯度。因此已经塌掉
   的 encoder 和会绕过 latent 的 decoder 仍可能维持塌缩。free-bits 最早作为
   “minimum information budget”启发式使用，见 [IAF 附录](https://arxiv.org/abs/1606.04934)，
   但不能把 floor 误读成强制 rate。
2. **高秩不等于动作。**过高 `tau_a` 可能允许 `z_a` 免费携带场景、身份和相机信息；
   VICReg 式 variance/covariance 也只防常数解，不保证语义正确。
3. **decoder shortcut 仍在。**当前 decoder 可直接看 `o_t`；即便 encoder 有非零 KL，
   decoder 仍可不用 `z_a`。强 decoder 导致 posterior collapse 是已知问题，
   [Generative Skip Models](https://proceedings.mlr.press/v89/dieng19a.html)也强调应显式
   加强 latent 到 likelihood 的依赖。
4. **scale 不可直接移植。**CD-LAM 是 `beta <=1e-3, tau=0.5`；本项目
   `beta_a=3e-3`，且 batch、维数、loss scale 不同。
5. **日志容易误导。**clamped KL 含常数地板，不能当实际 rate。必须同时记录 raw
   per-dim KL、active dims、clamped loss 和 `z_mu` covariance spectrum。
6. **batch composition 敏感。**官方实现先 batch-mean 再逐维 clamp；不同 episode/
   primitive 混合会改变哪些维度跨过地板。

最小 successive-halving 设计:

1. 先修 LR scheduler，并写测试断言 2-GPU 下 LR 单调、终点接近 0。修复后必须有
   `tau=0` 的同调度 baseline，旧 LF run 不能承担这项控制。
2. 固定其他超参，`tau_a in {0, 0.05, 0.2, 0.5}` 各跑 300 steps。
3. 用纯 annotation-free 健康门淘汰明显失败者，最好的两档继续到 1k。
4. 在 1k 选定一档，把该 1k checkpoint 作为共同 warmup 起点，分叉
   `LF-base` 与 `LF+L_dir`，两臂各继续到 5k。这样 step 数、初始化和调度完全匹配。
5. 若所有档在 1k 前都失败，再比较 `beta_a` 降低或 decoder-use/variance 约束，
   不继续盲扫更多 tau。若论文最终要求 `L_dir` 从 step 0 开始，可在机制通过后只给
   赢家补这一组确认，而不是在筛选阶段支付成本。

最小健康指标集:

- **结构:**`effective_rank(z_a)` 相对初始化/原始 LAM 的保持率、top-5 eigenvalue
  fraction、per-dim std；不使用当前 R2-based `collapse_flag`。
- **实际 rate:**raw KL 分布、active KL dims，`z_a/z_e` 分开报告。
- **decoder usage:**zero/shuffle/swap 后 foreground prediction 的 `DeltaMSE/DeltaPSNR`，
  用 bootstrap CI；不能只看 encoder latent 在变。
- **保真:**reconstruction/foreground MSE 不出现预注册的不可接受退化。
- **选择纪律:**tau 和 checkpoint 只用上述无 18D 指标选择；18D R2 在选定后揭盲，
  作为外部 non-inferiority 诊断。

可预注册一个简单 gate:300/1k 时 rank 相对 M0 不下降超过 20%，至少一种 decoder-use
指标的 CI 排除 0，且 foreground reconstruction 不恶化超过预先规定容差。具体容差
应依据 raw/base 的重复评估噪声定，不要事后按赢家调整。

### 6. 问题 4:第四条路先试什么

**先试 (ii) 运动代理，但把它设计成“motion anchor”，而不只是多一个回归头。**

推荐 target 是经过手/前景 mask 和全局相机运动补偿的局部 flow/track statistics，
至少包含归一化方向、幅度和置信度；proxy network 冻结并 stop-gradient。让 `z_a`
预测这些量，同时加小权重的 per-dim variance hinge 和 covariance penalty，必要时对
`o_t` 条件做轻度 dropout 或用 decoder residual usage loss。VICReg 的明确贡献正是
用逐维 variance floor 配合 covariance 防止常数表示，见
[VICReg](https://arxiv.org/abs/2105.04906)。机器人领域也已有非常接近的路线:
[Generalist Robot Manipulation beyond Action-Labeled Data](https://proceedings.mlr.press/v305/spiridonov25a.html)
先用手/夹爪附近的动态 3D 点做自监督，再用少量标签作 action alignment。

注意两点:

- 光流同时含相机和背景运动，必须做 foreground/ego-motion 处理；否则只是换一种
  nuisance anchor。
- 若训练和 headline 都用同一个 flow estimator，存在循环验收。最终仍用 held-out
  18D、opposite intervention 和 ACWM action-following 验收。

候选优先级:

1. **motion proxy + variance/usage safeguard:**连续、与 frame pair 对齐、直接约束
   方向，最符合当前失败机制。
2. **更细的 verb weak supervision:**可作辅助，但现有 12 类把
   `pick/place, insert/remove, stack/unstack, scoop/dump` 合并，天然丢掉多组符号方向；
   当前 `r2_lf` 也已说明粗语义 margin 单独不够。若拆成方向类，应诚实称弱监督。
3. **V-JEPA 式预测:**[V-JEPA](https://ai.meta.com/research/publications/revisiting-feature-prediction-for-learning-visual-representations-from-video/)
   证明纯 feature prediction 可学视频表征，但移植到 conditional LAM 是架构级改动；
   EMA/stop-grad 能避免两支共同塌缩，却不能自动阻止 decoder 绕过 `z_a`，也不保证
   target feature 排除了外观。放到后续，不作为两个月周期的第一修复。

### 7. 问题 5:如何解释 CD-LAM 的 R2=0.104

问题的前提先要纠正:`0.104` 是两层 ReLU MLP probe，不是线性 probe。真正的 linear
R2 是 CD-LAM `0.1266` 对 raw `0.1244`，同样几乎没变。两组结果都只能说明“在当前
公开 checkpoint 和当前协议下没有实质可解码性增益”，不能据此推出
“action-unlabeled 去偏本质上不能提高可解码性”，原因有四个:

1. 只有一个公开 checkpoint 和一次 protocol，`0.104-0.093=0.011` 没有跨 seed/CI。
2. 公开权重 metadata 是 300 steps，论文 Stage-1 protocol 是 1000 steps。
3. CD-LAM 优化的是 foreground bias、zero calibration、action following 等目标，未直接
   优化本项目的 18D wrist-pose geometry。
4. probe 可读出与 decoder 因果使用不是同一件事。已有干预式 probing 工作也指出
   仅靠 probe performance 不能证明模型实际使用该信息，见
   [Amnesic Probing](https://aclanthology.org/2021.tacl-1.10/)。

LF 线应按以下层级判定:

1. **Tier 0, 必要条件:**无结构塌缩，raw KL/variance 健康，decoder 对
   zero/shuffle/swap 有稳定响应，重建不过度退化。
2. **Tier 1, `L_dir` 机制:**相对匹配 LF-base，条件方向 probe 的 held-out NLL 显著
   改善，opposite swap 响应增强；旧 `direction_gain>0` 不再作为硬门。
3. **Tier 2, 主结论:**ACWM action-following/FDCE、held-out task 或第二数据集迁移有
   提升，同时视觉保真不退化。
4. **外部 R2:**作为 non-inferiority/解释性诊断。低 R2 但 Tier 1/2 成功，可能是
   非线性或非 18D 同构的可用动作码；R2 近 0 且 interventions 也近 0，则是真塌缩。

### 8. 问题 6:最少 run 的实验顺序

**阶段 A, 0 个训练 run:**

1. 实现条件方向 probe，把旧指标降级为 `static_shortcut_gap`；对所有现有 checkpoints
   离线重评，并给 conditional NLL、swap 指标加 episode-level bootstrap CI。
2. 修 scheduler，并用 1/2-GPU 单元测试验证全局 step 语义。
3. 离线审计 `L_dir` pair mining:每 task/episode 的通过率、角度分布、flow confidence、
   相机运动残差及人工抽样。若反平行 pair 本身噪声大，不启动训练。

**阶段 B, 约 2 个完整 run:**

4. 在修正调度下跑一个 pose-supervised baseline 和一个 `+L_dir`，先各到 1k 看机制；
   `L_dir` 没有改善 conditional probe 或 decoder intervention 就停并改 pair mining。
5. 通过 1k gate 后两者到 5k。只有匹配 baseline 才能隔离 scheduler 修复和 `L_dir`。

**阶段 C, 约 0.5 个完整 run 等价:**

6. 同时做 `tau={0,.05,.2,.5}` 的 300-step LF screen，best two 继续到 1k，
   再选一档用于分叉。筛选训练量共 `4*300+2*700=2.6k` steps。

**阶段 D, 约 1.6 个完整 run:**

7. 从选定的同一个 1k LF checkpoint 分叉，`LF-base` 和 `LF+L_dir` 各继续 4k steps。
   若共同 checkpoint 不健康，不要让 `L_dir` 同时承担“救活 latent”和“注入方向”
   两项归因。

**阶段 E, 只给赢家:**

8. 先做 Tier-2 ACWM action-following，再给关键 `base/+L_dir` 两臂补第二 seed；不重跑
   2-3 个旧家族。最后做 held-out task/第二数据集验证。

这样首轮约 `4.1` 个 5k-run 等价；补两个关键 seed 后约 `6.1` 个，按当前 5.5h/run
约 34 GPU-wall-hours，留有数据和 Tier-2 调试余量。决策门是:

- 新条件 probe 已显示 CD-LAM 有增量方向信息:先修改“未编码”假设，再设计 `L_dir`；
- `L_dir` 在监督 trunk 失败:停止 LF 扩展，修机制；
- `L_dir` 在监督 trunk 成功、LF 不健康:按诚实的 pose-regularized 主线投稿；
- LF 配对也成功:再把 action-unlabeled 泛化升为强结果，而不是预先承诺主故事。

### 最终建议的论文主张

在完成新 probe 和 `L_dir` 实验前，建议把原主张从:

> CD-LAM 修正一般 confounding 之后，运动方向仍是未编码自由度。

改为:

> CD-LAM 显著改善若干去偏与校准指标，但在我们的数据上，latent 仍表现出强初始
> 状态捷径，并未在条件方向探针和 decoder-side opposite-action intervention 中
> 展示可信的增量方向控制。我们据此引入一个不访问目标动作标注的运动学方向正则。

这个版本与现有证据严格对齐，也为后续可能出现的“latent 可读出方向但 decoder 不用”
留出空间。真正能支撑顶会 headline 的不是 `direction_gain` 从负变正，而是匹配对照下
的条件信息增益、干预响应和 Tier-2 action-following 三者同时改善。

---

# 阶段 A 同步更新(2026-07-20,发给顾问)

你的建议已全部采纳并执行完阶段 A(三项零训练成本工作)。**其中三项结果实质性地改变了
你推理时所依据的前提**,尤其是你自己设的决策门 #1 已经触发。以下按对结论影响排序。

## 更新 1:条件方向 probe 已实现并重评 15 个 checkpoint —— 你的决策门 #1 触发了,
## 但转折方向与预期相反

按你的方案实现:baseline `[Z_static; 0]` vs full `[Z_static; Z_trans − Z_static]`,
同特征布局,按 **episode** 分组 CV,报 held-out NLL,episode 级聚类 bootstrap +
episode→verb 置换 null。合成数据双向验证通过(有信号 vs_null=+0.198/p=0.00/AUC 0.845;
无信号 vs_null=−0.031/p=0.80)。

**实测(15 个 checkpoint,train episode 池)**:

| 家族 | vs_null 均值 | seed range |
|---|---|---|
| kl_full | +0.0592 | 0.0270 |
| ours_a | +0.0416 | 0.0243 |
| ours_r2 | +0.0697 | 0.0457 |
| **raw_lam(从未在 EgoDex 上训练)** | **+0.0774** | — |
| **cdlam_official** | **+0.0811** | — |

- 12/15 的置换 p ≤ 0.05 ⇒ **"方向未编码"被推翻,该说法已废弃**。
- 你的决策门 #1 原文是"新条件 probe 已显示 CD-LAM 有增量方向信息:先修改'未编码'
  假设,再设计 L_dir" —— cdlam_official 的 +0.0811/p=0.00 确实触发了它。
- **但转折是:`raw_lam` 排 4/15,高于我们全部三个家族的均值。** 它是现成的 DreamDojo
  LAM,从未在本数据上训练。⇒ 这点信息**不是任何训练带来的**,而是任意非退化
  transition encoder 天然携带的(运动本身与 verb 相关);CD-LAM 的去偏也没提高它。
- **量级微不足道**:ΔAUC(full−base)全表在 **−0.008 ~ +0.017**,过半为负;
  家族间差异(r2−kl = +0.011)远小于 seed range(0.024–0.046)。

**⚠️ 实现上的一个坑,供你判断指标是否可信**:原始 `gain = NLL_base − NLL_full`
**系统性偏负** —— full probe 比 base 多 32 个真实(含噪)特征、要付过拟合代价,
而 base 的零块被 L2 正则化掉了。所以**无信息基线不是 0,而是置换 null 的均值**。
我们据此改报 `gain_vs_null`,并加两道判读门:`auc_full > 0.5` ∧ `perm p ≤ 0.05`。
(若不做这一步,raw_lam 的 `gain = −0.31` 会被误读成"无方向信息"。)

**因此主张改为(第三版)**:
> 条件方向信息在所有 LAM 中都只有**痕量**存在——包括一个从未在本数据上训练过的
> 现成 LAM,其量与全部训练变体没有区别;CD-LAM 的去偏也没有提高它。与此同时,
> 没有任何模型在 decoder 侧展示出可靠的相反动作可控性(swap_opp_delta 的 seed
> range 0.12–0.17 覆盖了全部家族间差异)。**缺口不在"方向不在 latent 里",
> 而在"没有任何训练方法增加它,且解码器不使用它"。**

⇒ **L_dir 的目标随之改变**:从"注入方向信息"变为"**放大到可用量级 + 打通到
decoder 的因果通路**";成功标准从"vs_null > 0"改为"**显著高于匹配 baseline**"。

## 更新 2:旧 `opp_cls` 的泄漏比"天花板效应"更严重 —— 是 episode 记忆

你建议把旧指标降级为 `static_shortcut_gap` 泄漏诊断。实现时发现它泄漏得更根本:

| split | 可逆 task | **少数 verb 的 episode 数中位** | 少数 verb ≥3 ep 的 task |
|---|---|---|---|
| val | 28 | **1** | 5 |
| train | 68 | **8** | 57 |

旧实现用 **task 内随机切分**,同一条 episode 的帧同时进训练和测试。当一个 verb 只有
1–3 条 episode 时,"分类 verb"退化为"认出这帧来自哪条 episode"。
⇒ **`opp_cls_bal_acc_static = 0.96–0.99` 主要衡量 episode 身份记忆,而非
"初始外观→verb"规则**。这与天花板效应叠加,共同解释了 gap 为何恒负。
⇒ 失败模式②("外观在 z_a")的证据强度已下调;条件 probe 改用 train 池
(少数 verb 中位 8 ep)。失败③(R²_ze)、④(R²_rot)不受影响——它们回归 18D
连续动作、按样本切分,与 verb 标签和 episode 分组无关。

## 更新 3:L_dir 配对审计 —— 朴素规则确实噪声大(如你所料),但找到了可用窗口

你说"若反平行 pair 本身噪声大,不启动训练"。审计结果:

- **朴素规则(全 episode 范围内"反平行+幅度")不成立**:真实反平行率 30.99% vs
  各向同性 null **24.90%(lift 仅 1.24×)**;且反平行对的 Δt 分布与全体对
  **完全一致**(中位 5.47s vs 5.60s,≤2s 占比 25.2% vs 25.0%)⇒ 挑出的绝大多数是巧合。
  按此训练,L_dir 会退化成弥散排斥项(≈ uniformity 正则)。
- **但富集窗口存在**:扫 Δt × 幅度后,
  **p90 幅度门 + Δt∈[0.5,1]s ⇒ 反平行率 66.3% vs null 25.6%(lift 2.59×)**;
  Δt∈[1,2]s 为 2.19×。而 **Δt<0.5s 的 lift 只有 0.24**(相邻帧沿同方向 continue,
  物理上必然)——这是"测量本身有效"的 sanity check。朴素规则失败的原因是**池化**:
  >8s 的巧合对(54 万)淹没了 0.5–1s 窗口(8.9 万)。
- **定稿挖掘规则** = 同 episode ∧ Δt∈[0.5,2.0]s ∧ 幅度>p80 ∧ cos<−0.5(ego-compensated)。
  估算 ~370 对/episode × ~3200 episode = 百万量级候选,充足。
- **相机运动残差中位 18.3%(p75 46%)** ⇒ 你的警告成立,代理必须做 ego-motion 补偿
  (EgoDex 的 HDF5 自带 `transforms/camera`,补偿是现成的)。
- **代理可行性**:`own`(18D 标签口径)与 `cam`(代理口径)逐对一致率 87.8%、
  Jaccard 0.662。
- **覆盖偏置(你提醒过)**:按 task 的反平行占比从 `dry_hands` 45% 到
  `assemble_disassemble_furniture_bench_lamp` **9%** —— 精密装配类任务的反转极少。
  验收将用全任务指标 + task-held-out。
- 已采纳你的对照要求:**必配"同等对数的随机对排斥"对照臂**,否则无法排除
  "L_dir 只是 uniformity 正则"。

## 更新 4:LR 调度 bug 已修(你指出"不能宣称旧比较无偏",正确)

`accelerate.prepare(scheduler)` 在 2 卡下每全局 step 推进调度器 2 次,且余弦对
progress>1 未截断 ⇒ 实际 lr 呈 **V 形**(2500 步降到 0,5000 步回到 9.96e-6 近满值)。
已抽成 `make_lr_lambda(warmup, total, num_processes)` 并写单测(含旧 bug 回归见证)。
阶段 B 的 baseline 将在修正调度下重跑,旧 run 不作对照。

---

## 请你据此回答

1. **主张第三版是否站得住?** 尤其"raw_lam 也有同等痕量信息"这个对照——
   它是把动机变强了(没有任何方法增加方向信息)还是变弱了(方向信息一直都在,
   我们只是没放大)?哪种写法对审稿人更有说服力?
2. **L_dir 的成功标准怎么预注册?** 既然 vs_null > 0 对所有模型(含 raw)都成立,
   那门槛应该是"显著高于匹配 baseline"。用什么统计量?两个模型的 vs_null 之差
   + 配对 bootstrap?还是直接比 decoder 侧指标、把 probe 降为辅助?
3. **既然痕量信息已经可读出但 decoder 不用,是否应该把重心从 encoder 侧 loss
   (L_dir)转向 decoder 侧的 usage 约束?** 我们注意到 CD-LAM 的公开实现里有一个
   硬性 `L_use`("decoder must use z:real_z 重建优于 zero_z/shuffled_z 达 margin")
   ——这正是我们 lf 塌缩的死因,也可能是"可读出但不可控"的直接解药。
   L_dir 与 usage-gap 铰链,哪个该先做?还是必须同时?
4. **阶段 B 的排期是否需要因上述改动调整?**(你原方案:先 2 个 run 验 L_dir 机制,
   再进 LF 分叉。)现在多了一个随机对对照臂,以及可能的 usage 约束臂。

---

# 顾问第二轮回复(基于阶段 A 更新)

## 0. 修订后的总判断

阶段 A 成功排除了旧 `direction_gain`，也把真正问题定位得更准确。不过目前应区分
**exploratory 结果**和可写进论文的 confirmatory 结论:

1. 第三版从“方向是否存在”转向“弱可读信号是否被放大并被 decoder 方向特异地使用”，
   科学问题更强，也更接近 controllable world model 的核心。但当前措辞仍有三处过强:
   不能说“任意非退化 encoder 天然携带”、不能说“所有 LAM”、也不能说“decoder
   不使用 z”。
2. 当前 probe 测到的严格说是 **conditional opposite-verb signal**，不是纯运动方向。
   `Z_trans-Z_static` 还可能编码接触、速度、阶段和结果状态；它们也能区分 opposite
   verbs。真正的方向结论仍要靠运动向量外部读出和 decoder-side intervention。
3. `raw_lam` 是一个在大规模视频上预训练过的 DreamDojo LAM，不是随机/任意 encoder。
   它的高分证明“EgoDex fine-tuning 不是该信号存在的必要条件”，但不能证明“任何
   transition encoder 都自然有它”。要支持后者，至少补 random-weight encoder、
   pooled frame-difference/random projection 或 frozen flow baseline。
4. 新 `opp_cls` 审计成立:旧随机帧切分存在 episode 泄漏，应从 headline 和“外观在
   `z_a`”的主要证据中撤下。它最多是 episode identity leakage 的反例。
5. LR 修复已在本机复跑单测:1/2/4 GPU 的 matched-global-step 曲线完全一致，midpoint
   为 0.533、终点为 0，旧 2-GPU V 形 bug 也被回归测试复现。这项可以关闭。
6. 配对窗口是有价值的 **GT oracle 机制发现**，但还不是 proxy 可行性证明。当前
   `cam` 仍由 HDF5 的 GT hand trajectory 加 GT camera transform 算出，不是光流或
   keypoint 输出；`own/cam=87.8%` 只验证了两个 GT 坐标口径的一致性。

此外，关于 CD-LAM `L_use` 必须更正:官方源码确实实现了 usage-gap hinge，但
[Stage-1 配方](https://github.com/yufanwei/CD-LAM/blob/main/configs/stage1_recipe.yaml)
设置 `lambda_use: 0.0`，而 trainer 在 masked reconstruction 开启时也显式跳过
`L_use`。论文公式 `L_cal=L_KL-fb+L_zero` 不含 `L_use`。所以它是公开源码中的
**未启用实验机制**，不是 CD-LAM 论文已验证的组成部分；而且它是 soft hinge loss，
不是硬约束。

## 1. 先修正阶段 A probe 的统计口径

当前实现比旧指标正确得多，但还不足以用“12/15 显著”作论文结论。

| 当前问题 | 影响 | confirmatory 版本 |
|---|---|---|
| `n_perm=20` 且 p 可报 `0.00` | 20 次置换的最小合法 Monte Carlo p 应为 `(0+1)/(20+1)=0.0476`；`0.00` 不成立，卡在 0.05 的多行极不稳定 | 至少 999 次置换，用 `(b+1)/(B+1)`；只预注册少数主对比，或对 15 个探索性模型做 FDR |
| CI 是 raw `gain` 的 CI，不是 `gain_vs_null` | 表中 CI 全为负，不能用来证明 null-centered effect 大于 0 | 保存 episode-level loss；对 null-centered gain 和模型间差值分别做 paired hierarchical bootstrap |
| train episode pool 被 LAM fine-tuning 看过 | GroupKFold 只阻止 probe 的 episode 泄漏，不能撤销 encoder 已看过这些 episode | 对 Stage B 预先冻结一个不进入 LAM 训练的 verb-balanced episode/task manifest；旧 checkpoints 的 train-pool结果只作 exploratory |
| NLL 先按帧拼接再平均 | 长 episode 和样本多的 task 权重更大 | 先 episode 平均，再 task 平均；bootstrap 先抽 task、再抽 episode |
| `C=1.0` 固定 | 不同 latent scale/noise 下 full probe 的过拟合惩罚不一致 | 在 outer GroupKFold 的 train folds 内做 nested group CV 选 `C` |
| `auc_full>0.5` 作为门 | full 输入本来含 `Z_static`，即使 delta 完全无用也可能 >0.5 | AUC 只辅助报告 `AUC_full-AUC_base`；主统计继续用 null-calibrated held-out NLL |
| 同一套 20 个 permutation 未明确跨模型共享 | 模型差值混入 Monte Carlo null 方差 | 所有模型使用同 folds、同 donor/sample 和同 label permutations(common random numbers) |

按现有 CSV，6 行是 `p=0.00`、另 6 行是 `p=0.05`。仅加标准的 `+1` 修正后，前者变成
`1/21=0.0476`，后者变成 `2/21=0.0952`；所以“12/15 在 0.05 显著”会先降为**至多
6/15**，还没有计入多重比较。这不推翻信号存在的可能性，但足以说明现阶段不能写
“所有模型均有显著痕量信息”。

建议把当前指标名从 `dirprobe` 改成 `conditional_verb_probe`。若要直接声称运动方向，
应另加一个 proxy-independent evaluation-only readout:从 frozen latent 预测 held-out
18D 中的归一化平移方向/反平行关系，报告 cosine、angular error 或 pair ranking。
对最终 proxy-trained LF 模型，18D 只用于揭盲评估是合法的；对当前 GT-oracle
`L_dir` arm，这个指标是训练信号镜像，只能作机制诊断。

## 2. 问题 1:第三版主张是否站得住

**方向站得住，但要改成 availability-utilization gap，而不是“痕量方向普遍存在”。**

`raw_lam` 对照同时带来一强一弱:

- **变强的部分:**它否定了“EgoDex 上加了某个 objective 才产生 conditional verb
  signal”。Stage-0 objectives 没有显示超出 pretrained base 和 seed variation 的
  可复现放大，这使“现有 debiasing 不等于增强可控动作条件”更有说服力。
- **变弱的部分:**“把缺失方向注入 latent”不再成立，单纯 encoder separability 的
  novelty 降低。如果 `L_dir` 最终只提高 probe、没有提高 decoder intervention 和
  Tier-2 action-following，贡献会明显弱于原计划。

论文暂定措辞建议写成:

> Preliminary conditional probes show that the pretrained DreamDojo LAM and most
> evaluated fine-tuned checkpoints expose weak opposite-verb information beyond a
> static-latent baseline. None of the Stage-0 objectives yields a reproducible
> amplification over matched baselines, nor a reproducible improvement in
> direction-specific decoder control. We therefore target the gap between weak
> probe-accessible information and causally usable action conditioning.

中文版本:

> 初步条件探针表明，预训练 DreamDojo LAM 及多数微调 checkpoint 已含少量超出静态
> latent 的相反动词信息；但 Stage-0 目标没有相对匹配基线产生可复现的放大，也没有
> 带来可复现的方向特异 decoder 控制增益。本文因此研究弱可读信息到因果可用动作
> 条件之间的缺口。

在 paired bootstrap、独立 evaluation manifest 和多置换完成前，保留“初步”。完成后
若 `raw -> CD-LAM` 的 paired difference CI 包含 0，可写“no detectable improvement”，
不要写“完全相同”。也不要根据一个 raw checkpoint 推出“任意非退化 encoder”。

“decoder 不用它”也要降级。`ctrl_delta_psnr` 和 CD-LAM 的 zero-action response 已说明
decoder **总体上会使用 z**；现有证据只说明没有观察到稳定的**方向特异增益**。
`swap_opp_delta` 的绝对值约 0.3 并非零，目前缺的是相对 matched baseline 的 paired CI
和跨训练 seed 重复，而不是已经证明绝对不可控。

## 3. 问题 2:`L_dir` 成功标准如何预注册

把假设拆成三层，probe 不作唯一主指标。

### H1:方向特异的 representation amplification

对模型 `m` 定义:

```text
G_m = mean_task,episode[NLL_base - NLL_full]
P_m = G_m - E_perm[G_m]
Delta_P = P_Ldir - P_random-control
```

主对比必须是 **anti-parallel `L_dir` vs matched non-directional repulsion**，不是只和
no-loss baseline 比。后者回答方法有没有用，前者才回答是否来自方向配对。

- 相同 folds、相同 episode、相同 permutation，做 task -> episode 的 paired
  hierarchical bootstrap。
- smoke test 单 seed 只作 go/no-go，不使用“显著”措辞；最终结论至少两个训练 seed，
  training seed 放在 bootstrap/混合模型的最高层。
- 预注册 `95% CI(Delta_P)>0`，并同时要求超过 practical margin `delta_P`。
  `delta_P` 应在看 Stage-B 结果前，用修正调度 baseline 的重复方差确定。现有最大 seed
  range 约 0.046 nats，因此 `0.05 nats` 可作为保守候选，但应在新 evaluation protocol
  上重新冻结，而非直接沿用旧 V-schedule 数字。

当前 `kind=1` control 约 52% 仍是反平行，证伪力太弱。建议改为同 episode、同
`Delta t`/双端幅度分箱，但显式排除 `cos<-0.5`，优先采 `|cos|<0.2` 的近正交对。
`kind=2` 同向对是很好的 adverse control，但“把同向对推开”本身会伤害表征，不能
替代 matched random control；它适合 300-step sentinel，而非唯一完整对照。

### H2:方向特异的 decoder utilization

这是 Stage B 的**主要 Tier-1 指标**。推荐继续使用固定 context 的 same-vs-opposite
swap，但改成 foreground/motion-energy normalized error，并保留 PSNR 版本便于对照:

```text
C_m = E_task,episode[(MSE_opp - MSE_same) / (motion_energy + eps)]
Delta_C = C_Ldir - C_random-control
```

所有模型复用同 anchor 和 donor triples。预注册:

- `95% CI(Delta_C)>0`，且超过由 corrected-schedule baseline seeds 冻结的 practical
  margin；
- `L_dir` 相对 no-loss baseline 也改善，排除“control 变坏造成相对胜出”；
- `swap_context_cost` 和 reconstruction/foreground fidelity 不超过预注册退化容差；
- global zero/shuffle usage gap 只作机制指标，不能替代 direction-specific `C_m`。

### H3:下游可控性

最终 headline 仍是相同 ACWM 协议下的 action-following/FDCE 和 held-out task。若 H1
改善而 H2/H3 不动，结论是“encoder geometry 被塑形但没有成为可控条件”，不能称
方法成功。若 H2/H3 改善而 H1 只小幅变化，方法仍可成功，因为目标是 causal use，
不是让线性 probe 好看。

因此预注册的总成功条件是:

1. `L_dir > random-control` 至少在 H2 成立；
2. H3 相对 matched baseline 成立且 fidelity 守门通过；
3. H1 是支持机制的 secondary endpoint；
4. 同向 adverse control 不应复制 H1/H2 的改善。

## 4. 问题 3:先 `L_dir` 还是先 usage-gap

**两者不是替代关系，也不应第一批就无消融地同时打开。**

- `L_dir` 解决 **availability/specificity**:哪些 transition 应在 latent 中沿方向区分。
  它可以只改 encoder geometry，decoder 完全忽略。
- `L_use` 解决 **generic utilization**:real z 至少要比 zero/shuffled z 更有助于重建。
  它不知道哪部分是方向，可能奖励场景、身份、接触或任意高频 code。

usage hinge 还有两个已知风险:

1. `relu(margin-(MSE_corrupt-MSE_real))` 同时允许降低 real error 和主动抬高 corrupted
   error，模型可能学到对任意 latent 扰动敏感，而不是正确方向控制。
2. full-frame gap 容易由背景/外观满足；shuffle 跨 context 时更严重。应使用
   foreground mask、motion-energy normalization 和 within-task donor，并把视觉保真、
   高频伪影和 context leakage 设为 guardrail。

具体优先级应按 trunk 分开:

- **当前 pose-supervised Stage B trunk:**已有明显 global `ctrl_delta`，没有证据表明
  decoder 全局 bypass。因此先隔离 `L_dir` vs matched repulsion，同时加一个短的
  `L_use-only` 诊断臂。不要默认 `L_use` 是主解药。
- **LF trunk:**现有 zero/shuffle/swap 近零，确有 decoder bypass。free-bits 只撤掉
  KL 压力，不能强制使用 z；这里 `free-bits + foreground L_use` 是优先组合，但必须
  有 `free-bits-only` 配对臂，才能判断谁救活了 trunk。

若 Stage-B smoke 出现“`L_dir` 提升 H1、不提升 H2；`L_use-only` 提升 global usage”，
再进入组合实验，并比较 `L_dir+L_use` 对 `random-control+L_use`。缺少后者时，仍无法
排除“任意 repulsion 在强制 decoder 使用 z 后都有效”。

## 5. 问题 4:调整后的 Stage B 排期

### B0:训练前再做三个零成本修正

1. 将 permutation 提到至少 999，修正 p 值，输出 null-centered CI 和模型间 paired
   CI；按 episode/task 平衡，固定 nested-CV 和 common permutations。
2. 为新 run 冻结一个 LAM 从未见过的 evaluation manifest。旧 train-pool probe 保留
   为 exploratory，不再混进 confirmatory 表。
3. 把 `kind=1` 改成排除反平行的 matched near-orthogonal control；保留 `kind=2` 作为
   短 adverse control。并明确当前 `.npz` 是 GT-oracle pairs。

### B1:1k successive-halving，不立刻跑全因子到 5k

从同一 pretrained checkpoint、修正调度启动四个主臂:

| arm | 目的 |
|---|---|
| `B` = corrected baseline | 新调度基线 |
| `R` = matched non-directional repulsion | 排除 generic uniformity |
| `D` = GT-oracle anti-parallel `L_dir` | 先验证方向配对机制上限 |
| `U` = foreground normalized `L_use` only | 判断 generic decoder usage 是否是主瓶颈 |

另给 same-direction adverse arm `S` 只跑 300 steps；若它和 `D` 同样改善 probe，立即
停止并判定 repulsion/metric 仍有漏洞。四个主臂到 1k 合计 `4k steps=0.8` 个完整 run，
`S` 再加 0.06 个，不构成预算压力。

1k 决策树:

- `D` 不胜 `R`:停止方向故事，先修 pairing/loss；不进入 LF。
- `D` 胜 `R` 且 H2 也改善:继续 `B/R/D` 到 5k，`U` 停在分析章。
- `U` 单独改善 H2/下游代理而 `D` 不改善:主问题是 utilization，优先 `B/U`，
  `L_dir` 降级。
- `D` 只改善 H1、`U` 改善 usage:启动 `R+U` 与 `D+U`；最终方向对比是二者之差。
- 四臂均无外部改善:不要靠调 lambda 延长训练，先重做 decoder bridge 或指标。

组合情形不需要第一天就支付完整 2x3 factorial。只有互补性出现后，才把
`B/U/D/R+U/D+U` 中形成必要因果链的 arms 继续到 5k。虽然最坏情况仍需 4-5 个完整
arms，但 successive-halving 会在 1k 淘汰大部分错误机制，比一开始同时打开
`L_dir+L_use` 更节省，也保留可归因性。

### B2:Tier-2 前置，LF 后移

任何 5k winner 先进入小规模 ACWM action-following smoke。只有 H2/Tier-2 同时通过，
才开始 LF free-bits 分叉。当前研究问题已经扩展到 availability × utilization；若在
主 trunk 上尚未知道需要 `D`、`U` 还是 `D+U`，提前做 LF 会把 tau、collapse、pairing
和 decoder usage 四个因素重新缠在一起。

### 最终主张模板

若 `D+U > R+U > U` 且 Tier-2 成立，最强故事是:

> Pretrained LAMs expose weak direction-correlated information, but existing
> debiasing neither amplifies it nor makes it reliably usable by the decoder.
> Direction-aware latent structuring and an explicit utilization constraint are
> complementary: the former supplies direction specificity, while the latter
> closes the latent-to-decoder path.

若只有 `U` 成立，应诚实把工作改成 latent utilization，而不是保留 `L_dir` 标题。若
只有 `D` 成立且 decoder/Tier-2 不动，则最多是 representation analysis，尚不足以支撑
controllable ACWM 的主贡献。

---

# 阶段 B 结果同步(2026-07-21):方向线双指标否决,请判断转向

四臂 successive-halving 已按你的方案跑完(B 基线 / R 匹配非方向排斥 / D GT-oracle
反平行 L_dir / U 前景运动能量归一化 L_use;各 1k 步 + S 同向哨兵 300 步)。
修正调度、冻结评测 manifest(390 ep,encoder 从未训练;57 可逆 task 过 probe)、
StratifiedGroupKFold probe(见末尾 bug 说明)、999 置换、task→episode 配对
hierarchical bootstrap —— confirmatory 口径。

## 结果

**H1(representation amplification)** ΔP:
- **D − R = +0.008 CI[−0.007, +0.034] 含 0**(主对比不显著)
- D − B = −0.005 CI[−0.045, +0.040](方法未胜基线)
- U − B = −0.007 CI[−0.030, +0.002];S − R = −0.001(哨兵正常)
- 单臂 P:B **+0.022** > D +0.018 > U +0.015 > R +0.010 > S +0.009(基线最高)

**H2(direction-specific decoder utilization,你指定的主 Tier-1 指标)**
C = E[(MSE_opp−MSE_same)/(motion_energy+ε)],840 anchor / 55 task:
- 单臂 C 全部 ≈ 0:B +0.0001 / R −0.0002 / **D −0.0000** / U −0.0008 / S +0.0010
- **ΔC(D − R) = +0.0005 CI[−0.0001, +0.0012] 含 0**;五个 ΔC 配对全部 n.s.
- ΔC(U − B) = −0.0038 CI[−0.031, +0.020](极宽,含 0)

⇒ 按你的决策树第一分支"D 不胜 R":**停方向故事,不投 proxy 管线**。
这是 GT-oracle 上界配置;proxy 只会更弱。Tier-0 机制确实生效
(D 的 za_cos +0.16→−0.30,active_frac 0.90→0.34)——几何被改变但外部零收益。

## 一个可能比"缺哪个 loss"更根本的观察 —— 请你判断

H2 的 `C≈0` 对**所有**臂成立,包括基线。也就是说:无论用什么 loss,
把 z_a 换成相反 vs 同向动作的捐赠者,解码器输出几乎不变 —— **decoder 对 z_a
的方向零响应,这不是 D/R/U 之间的差异,是全体的共性。**

我们怀疑根因在 trunk 而非缺某个 loss:此 checkpoint 的 KL 总量仅 ~19/40 维,
后验方差偏大,训练时解码器收到的是采样 latent(z_mu + noise),噪声可能盖过 z_a
的方向分量。这与更早的两个观察同源:
① lf 塌缩(去监督头后 rank 2.8,解码器完全无视 z_a);
② L_use 烟测时 gap_zero = −0.47(零 latent 解码优于采样 latent)。
三者都指向 **decoder utilization / posterior collapse-lite,而非 latent 不可分**。

## 请你回答

1. **转向判断**:H2 全臂 C≈0 是否足以把研究问题从"方向 availability"正式转为
   "decoder utilization"?还是 C≈0 本身可能是 **swap 协议或 motion 归一化的伪影**
   (例如 opposite 与 same 捐赠者在这个数据上解码差异本就极小,指标缺乏灵敏度),
   需要先验证 H2 指标对一个"已知强可控"的正对照(如 IDM oracle 或 GT-action 条件
   世界模型)有非零 C,才能采信它的零?
2. **utilization 线证据强度**:sb_U 的 H1 单臂最高但所有 U 配对(H1 与 H2)都不显著。
   现在把重心转向 L_use / usage 约束,证据是否够?还是应先做一个**最小可控性正对照**
   (证明"在这个 trunk 上 decoder 原则上能被 z_a 方向驱动")再决定?
3. **trunk 假设**:若根因是采样 latent 噪声盖过方向,正确的第一步是
   (a) 降后验方差 / free-bits + usage-gap 重建 trunk,还是
   (b) 评测时改用 z_mu(而非采样)重跑 H2 排除"噪声掩盖"这一解释?
   —— (b) 是零训练成本的诊断,能立刻区分"信息在但被采样噪声淹没"vs"信息真的没进 decoder"。
4. **论文定位**:方向线否定后,现有可辩护的贡献是否退回到
   **"诊断 + 评估方法学"**(CD-LAM 官方 checkpoint 级失败清单 + availability/utilization
   分解 + confirmatory 协议),而把"如何修复"留作 open problem?这样的论文在你看来
   够 CoRL/ICLR workshop 还是 main?

## bug 说明(供你评估结果可信度)
初版 probe 用 GroupKFold 按 episode 留出,但 manifest 每 verb 仅 ~2 ep,每折 test
单 verb 被跳,57 task 塌成 5,曾误报 ΔP(D−R)=+0.103"显著"。已改 StratifiedGroupKFold
(n_splits 由较小 verb 的 ep 数决定),57 task 全进,合成数据四档信号单调性验证通过。
5-task 假象存档 `results/stage_b_eval_5task_STALE.json`。

---

# 顾问第三轮回复(基于 Stage B confirmatory 结果)

## 先给结论

1. **停止当前方向-loss/proxy 路线是正确的。**H1 已经足以否决这版机制投资:
   `D-R` 的点估计只有 `+0.008`,CI 包含 0,而且上界 `+0.034` 低于先前冻结的
   practical margin `0.05`。这比“没有显著”更强,是对**当前 checkpoint、1k 配方和
   指标口径下的实际有效性**作 futility 判断。
2. **但不能由全臂 `C≈0` 直接宣布根因已经是 posterior noise / decoder
   utilization。**它目前只证明“经这套 donor-swap assay,没有检测到 direction-specific
   causal use”。在正对照通过前,`decoder 不用方向`和`assay 对方向不敏感`仍未区分。
3. **当前 U 臂不是 utilization 方法的正结果,而是应当淘汰的实现。**它没有赢 paired
   H1/H2,并且明显损伤 validation fidelity。研究问题可以转向“如何测量并建立
   utilization”,但不能直接转向“继续调当前 `L_use`”。
4. **零训练的 `z_mu` 复评其实已经完成。**当前
   `code/h2_utilization.py` 的 same/opp 两路都调用 `_decode_from_mu`;所以观测到的
   `C≈0` 不是评测阶段 posterior sampling 的产物。训练阶段的采样噪声仍可能让 decoder
   学会忽略 latent,但这是下一个待检验机制,不是当前结果已经证明的原因。

最准确的转向表述是:

> The direction-specific intervention failed despite changing latent geometry.
> We therefore pivot from designing another directional regularizer to validating
> and diagnosing the latent-to-decoder causal channel.

不要提前写成:

> Posterior noise causes the decoder to ignore direction.

后一句还缺实验链条。

## 1. `C≈0` 是否足以转向

### 对停止 `L_dir`:足够

`za_cos: +0.16 -> -0.30` 说明优化器确实塑形了它负责的训练统计;H1 和 H2 却都没有
外部收益。因此这不是“loss 没跑起来”,而是“loss 实现了局部几何目标,但该目标没有
转化为所需功能”。继续做 proxy pairs 只是在一个尚未显示 functional value 的目标上
提高标签可获得性,投资顺序不成立。

不过,“GT oracle 是严格上界,proxy 必然更弱”应稍微降格。GT pairing 是目前最干净、
最强的机制检验,足以作项目停止门;但它不是数学上界,因为有噪声的 proxy 偶尔可能产生
不同的覆盖或正则化效应。论文里应写“the cleanest oracle-pairing test failed, so we do
not invest in proxy construction”,不要写成已经证明任何 proxy 都不可能有效。

### 对确认 utilization 根因:不够,必须做正对照

当前 `C` 有至少三个 construct-validity 风险:

1. **donor 语义不保证物理距离成立。**“同 verb” donor 可能在幅度、相位、左右手、
   接触状态上比“反 verb” donor 更远。若两类 donor 对当前 anchor 都是同样错误的动作,
   `MSE_opp-MSE_same≈0` 不表示 decoder 不响应。
2. **误差差值可掩盖响应。**decoder 对两个 donor 可能输出不同图像,但二者相对 GT
   同样错误,此时 `C=0`;这应判为“有响应但响应不对齐”,而不是“完全不使用 z”。
3. **归一化量纲不一致。**当前分子是 squared error,分母实现为平均绝对帧差。所得比值
   不是无量纲量,且低运动样本会放大方差。它可以作为固定口径的 score,但不能自然解释
   成“相对运动能量百分比”。

因此在把全臂零值解释成系统性 bypass 前,需要一个已知强可控的正对照。IDM oracle
**本身不自动是正对照**:IDM 能正确编码动作,不等于被测 decoder 会使用它。优先级如下:

1. 在同一数据、同一 anchor/donor 和同一 scorer 上评测一个 GT-action-conditioned world
   model;或给当前 trunk 加一个小型 GT-action adapter/bridge,以明确的 action-swap
   reconstruction 训练到可控,再作为 assay positive control。
2. 同时加 random-donor negative control。正对照应得到稳定正 `C`;随机 donor 不应复制
   opposite 的有序效应。
3. 对每个 anchor 记录真实 18-D action distance,验证
   `d(a_anchor,a_same) < d(a_anchor,a_opp)`。不满足者单独报告或剔除,不能只靠 verb 名称。

如果正对照也得到 `C≈0`,先修 pairing/scorer,不能据此诊断任何模型。如果正对照为正、
而 B/R/D/U 均为零,才可以把“direction-specific utilization deficit”作为正式发现。

### 对现有样本可立即做的 H2 分解

无需训练,为每个 arm 和 anchor 输出:

- `E_own`:自身 `z_mu` 的重建误差；
- `E_same`, `E_opp`:现有两类 donor 的误差；
- `R_out = ||D(o,z_opp)-D(o,z_same)||`:直接输出响应量；
- persistence-relative score,例如以 `MSE(o_t,o_{t+1})` 作同量纲分母；
- foreground/motion-mask 内与背景区分别统计的上述量。

判读矩阵是:

| 观察 | 判断 |
|---|---|
| `R_out≈0`,正对照 `C>0` | 被测 decoder 对该方向干预近似不变,支持 utilization deficit |
| `R_out>0`,但 `C≈0` | decoder 有响应,但方向错误或 donor/GT 误差指标不对齐 |
| `E_own < E_same≈E_opp` | decoder 使用 episode-specific latent,但 swap 没隔离出方向 |
| `E_own≈E_same≈E_opp`,且 zero/shuffle 也相同 | 更强的 latent bypass 证据 |
| 正对照 `C≈0` | assay 未通过灵敏度验证,不能作模型结论 |

这一步比先训练新的 `L_use` 更有信息量。

## 2. 当前证据是否支持 `L_use`

**不支持。**问题中的“sb_U 的 H1 单臂最高”与上方结果不一致:单臂 `P` 是
`B 0.022 > D 0.018 > U 0.015`;U 只是某些附属统计(例如 `DeltaAUC`)较高。单臂相对
自身 permutation null 的显著性也不是 U 相对 B 的 treatment effect;主因果对比仍应看
`U-B`,而它在 H1/H2 都没有通过。

训练日志还给出了更明确的否决信号:

| step 1000 | B | U |
|---|---:|---:|
| validation MSE | 0.0010 | 0.0018 |
| validation PSNR | 29.92 dB | 27.46 dB |

U 损失把训练 batch 上的 `gap_zero` 从早期 `-0.5707` 推到 `+0.3130`,并把
`gap_shuffle` 推到约 `+0.0191`;说明 hinge 确实改变了它直接优化的 statistic。但
2.46 dB 的验证 PSNR 退化表明它至少没有通过预注册 fidelity guardrail。现有 hinge
可以通过抬高 corrupted reconstruction error、制造任意 latent 敏感性来满足 margin,
不必学到正确动作控制。当前 U 的最合理结论是:

> The usage hinge successfully opens a generic real-vs-corrupted gap, but does
> not improve direction-specific use and degrades reconstruction fidelity.

所以现在应做“最小可控性正对照 + 使用路径分解”,而不是把 U 延长到 5k,也不应直接把
`free-bits + L_use` 作为新主方法。

## 3. sampling-noise / trunk 假设如何检验

### 先纠正两个概念

1. 日志中的 `KL≈19--21` 是 40 维 latent 的**总 KL(nats)**,不是“19/40 个 active
   dimensions”。active dimensions 必须从逐维 KL、跨数据 `Var(mu_j)` 等统计另算。
2. 高 posterior variance、posterior collapse 和 decoder bypass 相关但不等价。当前有
   action probe 信号和非零总 KL,最多先称“direction-specific conditional-use failure”或
   “collapse-lite hypothesis”,不能把完整 posterior collapse 当成已证事实。

### 选项 (b) 已经是当前 H2 口径

`h2_utilization.py` 中 same/opp 均由 `_decode_from_mu(...)` 构造 `z_rep=z_mu` 后解码。
因此重跑同一个 `z_mu` H2 不会区分任何东西。这里能排除的是:

> evaluation-time random epsilon happened to wash out C.

但还不能排除:

> exposure to noisy posterior samples during training taught the decoder to
> down-weight the latent branch.

后一个假设需要下面的分解。

### 零训练诊断

对 B 与 U checkpoint 至少计算:

1. 每维 `signal_j = Var_data(mu_j)`、`noise_j = E_data[exp(logvar_j)]` 和
   `SNR_j = signal_j/(noise_j+eps)`;
2. same/opposite 类中心距离相对 posterior noise 的比率,而不只看 `za_cos`;
3. 同一输入分别用 `mu`、K 个 posterior samples、zero、within-task shuffle 解码,
   报告误差均值/方差及直接输出距离;
4. 沿 `z_opp-z_same` 做小步有限差分,或计算 decoder/action bridge 的局部 Jacobian norm,
   判断 decoder 在该方向是否有增益。

只有当 `E_sample` 显著差于 `E_mu`,并且 direction separation 小于 posterior noise scale,
才有证据支持“训练采样噪声盖过方向”。若 `mu/zero/shuffle` 的输出本来就几乎相同,
问题更像 bridge/decoder bypass;若直接输出距离大但 C 为零,则是 semantic alignment 或
H2 scorer 问题。

### 若噪声假设成立,首个训练消融也不应是 free-bits

free-bits 的作用是让低于阈值的每维 KL 不再收到继续压低 rate 的梯度;它**不直接降低
posterior variance**,也不保证 decoder 使用 latent。最有针对性的最小因果实验是保持
encoder、KL 和 action head 口径不变,只改变训练时喂给 decoder 的 latent:

```text
z_dec = mu + alpha * sigma * epsilon
alpha in {0, 0.25, 1}
```

至少比较 `alpha=0` 与现有 `alpha=1` 的 300/1k matched arms。若 `alpha=0` 恢复
`R_out/C` 且保真不退,才说明 decoder-exposure noise 是瓶颈。随后再分别消融 free-bits、
usage 约束和二者组合。否则直接上 free-bits 会同时改变 rate、方差、active dims 和
reconstruction trade-off,仍然无法归因。

## 4. 论文定位

### 当前最可辩护的论文不是“新修复方法”,而是诊断与评估方法学

可以形成的贡献链是:

1. 指出普通 action probe 会把初始状态/episode 泄漏误当动作信息,并给出修正的
   group-stratified confirmatory protocol;
2. 把 action **availability**、generic latent **utilization** 和 direction-specific
   **causal use** 分开,证明三者不能相互替代;
3. 展示一个正交的 negative result:oracle `L_dir` 强烈改变预期 latent geometry,
   却没有改善 held-out representation amplification 或 decoder causal use;
4. 给出经正对照验证的 intervention assay 和 fidelity guardrails,说明 generic usage
   gap 可以被优化但可能以图像质量退化为代价。

措辞必须限制在实际证据范围:

- 写“public CD-LAM checkpoint behavior under our protocol”,不要写“CD-LAM 方法失败”;
  本地公开权重 metadata 是 step 300,并不等价于论文完整 1000-step Stage-1。
- 写“this `L_dir` recipe at the 1k stopping point failed”,不要推广成“direction losses
  cannot work”。
- 在 H2 正对照通过前,写“no detected direction-specific use under our assay”,不要写
  “the decoder is direction-invariant”。

### workshop 与 main 的判断

**CoRL/ICLR workshop:可以,而且诚实的 negative-result + methodology 故事是合适的。**
最低条件是补上 H2 正对照、修正量纲/配对审计、公开 manifest 与 stale-result 处理流程。
是否找到修复不是 workshop 成立的必要条件。

**CoRL/ICLR main:以当前证据不够。**主要短板不是“没有新 loss”,而是诊断的外部效度
仍局限于一个 DreamDojo/CD-LAM 生态、一个数据分布,且核心 H2 尚未通过已知正对照。
纯诊断论文可以进 main,但通常需要把结论做成跨模型、跨数据且能预测真实下游失败的
一般规律。至少应补到:

1. 2 个以上 LAM 架构/训练目标和 2 个以上数据域或 embodiment;
2. positive/negative controls 证明每个核心指标的灵敏度与特异性;
3. availability/utilization 指标能预测 Tier-2 action following,而不只是重建 swap;
4. 所有关键比较采用冻结 manifest、task/episode 层级统计和 fidelity guardrails;
5. 最好有一个简单、可归因的干预恢复 causal use;没有修复也可以,但此时广度和下游
   预测力必须更强。

## 建议的立即排期

1. **正式停止 D/proxy 管线**,保留 checkpoint 与否定结果,不跑 D/U 到 5k。
2. **用现有 checkpoint 重算 H2 分解**:`own/same/opp/zero/shuffle`、直接输出距离、
   同量纲归一化、前景/背景、GT action donor distance。
3. **建立 H2 正对照**。这是转成 utilization 论文前的硬门,不是可选附录。
4. **做 posterior signal/noise 分解**;不要把总 KL 当 active dims。
5. 若且仅若证据支持 training-time sampling noise,跑 `alpha=0 vs 1` 的短 matched
   ablation。若不支持,优先检查 decoder bridge/conditioning architecture。
6. 只有在某个干预同时改善 direction-specific use 且通过 fidelity 后,才恢复 Tier-2
   和方法论文路线;否则按 diagnosis + methodology workshop 稿收束。

最终判断不是“方向信息不存在”,也不是“已经证明 posterior collapse”。现有结果真正
证明的是:**当前方向监督能改表示几何,却不能产生可测的外部增益;当前 generic usage
hinge 能改训练 gap,却不能产生方向控制且损伤保真。**下一步的最高价值不是再加 loss,
而是先用正对照把 causal-use 测量闭环做实。
