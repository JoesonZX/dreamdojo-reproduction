# 概念、评估栈与 pipeline 定稿(2026-07-19)

本文档是**概念层的单一事实源**:好 LAM 的定义、三层评估栈、实验↔指标对齐、
相反动作分离的设计、以及从定义到失败清单的完整闭环。
数据支撑见 `benchmark_v2_findings.md`;指标语义见 `benchmark_v2_changes.md`;
执行任务见 `stage0_execution_plan.md`;CD-LAM 对照见 `cdlam_repro_fidelity.md`。

标注 ⚠️ 的段落是**对常见误述的纠正**,写论文时特别注意。

---

## A. 好的 z_a 是什么

**z_a = 引起两帧变化的、跨场景不变的原因,精确到视觉可分辨粒度。**

四条可测性质:
| 性质 | 含义 | 外部指标 |
|---|---|---|
| 充分 | 固定 o_t,z_a 覆盖 o_{t+1} 里所有由动作引起的变化 | R²_za / R²_full / labeleff |
| 跨场景不变 | 同一动作在不同场景映射到相近 z_a | task-held-out R²、swap_context_cost、camera_shift |
| 可控 | 改 z_a ⇒ 改解码/预测的运动 | ctrl ΔPSNR、do(z_a=0)、swap_delta |
| 可接地 | 能被真实动作线性读出 | R²(线性 probe)——**充分性的可验证下界** |

**不等于 18D 标注动作**。若把目标定成"编码 18D",就砍掉了对可控性有用但没标注的
运动自由度(手指、物体接触、被操作物的运动)。18D 读出只是充分性的**下界检验**。

**视觉等价类**:在**所有场景下**视觉后果都不可分辨的两个动作,会坍缩成同一个 z_a。
⚠️ 量词很重要——两个动作只在**当前**场景看不出差别(按硬桌子 vs 按海绵),在别的
场景可分,它们就不该同类。这正是"精确到视觉可分辨粒度"的含义:跨场景一致性只被
要求到视觉可恢复的程度。这是妥协,不是缺陷。

**z_a 是干预句柄,不是标签**(详见 §G1)。"纯粹"是手段不是目标:定位为**运动学码**
(Δt 内运动),场景相关的解释由世界模型结合 o_t 完成。

---

## B. 三层评估栈 ⚠️ 按"循环性"分层,不是按"用途"分层

这是最容易搞错的一点。三层的划分依据是**指标是不是训练 loss 的镜像**:

| 层 | 是什么 | 能做什么 | 绝不能做什么 |
|---|---|---|---|
| **Tier-0 诊断** | loss 的镜像(static_response、rev_cos、shortcut、rev_margin) | 机制检查:loss 动了自己的量吗?没动 = 实现 bug。否决 + 解释 | **永远不能进 headline**;不能用来"证明"失败模式存在 |
| **Tier-1 裁决** | 从定义反推的**外部**指标(冻结 encoder + 真值目标) | **双重职责**:跑在 baseline 上 = 定位失败模式;跑在新模型上 = 裁决 loss | 被自己 loss 消费过的信号,失去 headline 资格 |
| **Tier-2 终审** | 下游世界模型(ACWM + FDCE + 干预) | 判断"对世界模型有没有用" | — |

⚠️ **常见误述**:"Tier-0 用来诊断 baseline 的失败模式"。错。Tier-0 是 loss 镜像,
拿它当失败模式的证据就是**层级提升**错误(Ours-B2 的教训:rev_cos 达标 ⇒ 宣称
"方向敏感",而所有外部指标毫无改善)。**诊断 baseline 和裁决新 loss 用的是同一层
(Tier-1),区别只是跑在谁身上。**

**核心纪律**:循环性是 **(loss, 指标) 二元组**属性,不是指标固有属性。
谁的 loss 消费了什么信号,谁就丧失对应镜像的 headline 资格。
(例:全家族带 `lambda_action=1.0` ⇒ `mlp_action_r2` 对全家族都是准循环指标——
家族内受控比较仍公平,但**不能作为绝对外部主张**,也不能与 raw_lam / CD-LAM 等
无监督模型跨家族比。)

每个 claim 过两问:① 指标测的性质 = 主张的性质吗?② 指标是不是该 loss 的镜像?

---

## C. 实验目的 ↔ 指标对齐

| 模型 | 设计意图(机制假设) | 正确的裁决指标 |
|---|---|---|
| **KL baseline**(split-KL) | 容量瓶颈把场景挤进 z_e(供给侧) | 路由三元组 R²_za/R²_ze/R²_full;camera_shift 作参考 |
| **Ours-A**(L_zero) | 用重复帧把零动作校准到原点 | **do(z_a=0)** 解码(zeroact_motion_suppression / still_margin) |
| **Ours-R/R2**(真实相反 verb margin)<br>*已取代 Ours-B/B2/C* | 分离相反动作 | direction_gain、swap_opp_delta_psnr |
| **Ours-D**(L_emb 前景加权) | 把外观挤出 z_a(需求侧压力) | opp_cls_static↓、swap_context_cost↓、R²_za 持平 |
| **cdlam_repro**(L_pctr) | CD-LAM 配方对照 | 无 Tier-1 预期,不判死刑(见 `cdlam_repro_fidelity.md` §2 规模偏离) |
| **IDM oracle**(λ_recon=0) | 可解码 ≠ 可控的反例 | 预期 R² 最高 + ctrl/swap/zeroact 最差 |

⚠️ **纠正:do(z_a=0) 测的不是"动作维有没有环境信息"**,测的是**零动作被校准到原点**
(置零 z_a 后解码拉回当前帧 = 干预句柄的原点标定),这是 L_zero 的外部验收。
"z_a 有没有环境信息"由 **opp_cls_static、camera_shift、swap_context_cost** 回答。
两件事都重要,但别混:**L_zero 校准了原点 ≠ z_a 干净**。

⚠️ **Ours-R/R2 的定位(T8 后定稿,2026-07-19)**:v2 的"R² 最高 + 路由最干净"胜点
**已死于 seed 方差**(对 kl_full:r² +0.016/噪声 0.055、gain +0.017/0.024、
swap −0.032/0.167,全在 range 内;见 `seed_variance.md`)。R/R2 **改判为
"纯语义 margin"消融臂**:同 task 相反 verb 的 clip 级配对、无运动学门——它的
三项零改善 + scene_knn 回升(0.176 vs ours_a 0.117)正是"语义门单独不够"的实证,
直接 motivate L_dir 的运动学门。**不再给 R2 家族训任何新东西;stage-1 trunk 用
Ours-A(KL+L_zero),不用 R2**(其 margin 与 L_dir 目标重叠且回吸场景几何)。
相反分离仍是未解问题 = 论文靶子。

⚠️ **lf 孪生的教训(T7,2026-07-20)**:kl_lf 与 r2_lf **双塌缩**(rank 2.8/6.7,
解码器无视 z_a)。β_a=3e-3 瓶颈下 **18D 监督是唯一防塌锚**;L_zero+margin 只减缓。
⇒ "λ_action=0 消融"不是免费的,label-free 线需要重建 trunk(见 §G7)。
lf 行的一切指标是塌缩伪影,不作任何主张的证据。

---

## D. 相反动作分离的设计

**顺序不可颠倒:先把外观挤出 z_a,再做方向分离。**
理由:z_a 里只要还有外观,任何相反 loss 都会走外观捷径(v2 静态对照 0.96 实证)。
所以 L_emb(SAM3 前景加权)是 L_dir 的**前置条件**,不是并列项。

⚠️ 机制表述要精确:z_a 从来没有"必须编码背景"的压力(背景从 o_t 免费可得),
问题是 z_a **顺手携带**了外观相关性(训练分布里状态与动作相关)。L_emb 的机制是:
把重建收益压到 embodiment 前景 ⇒ 背景像素的梯度收益消失 ⇒ z_a 携带外观无利可图。

**因果链(学名 fidelity–following gap)**:外观进 z_a **不伤**视频预测(z_a 与 o_{t+1}
外观一致,teacher forcing 照样低 loss),**伤在干预**:训练时拟合了"状态→顺势动作"
的相关性;干预时条件与 o_t 自相矛盾 ⇒ 模型要么忽略条件、要么出鬼影 ⇒ 不听指挥。

### 配对规则:语义 ∧ 运动学 **双门**

单用运动方向不行(一个 clip 里不只有核心动作:伸手、推/拉、收手;仅对动作向量
取负会把"伸手"和"收手"也配成相反对)。单用语义也不行,帧级有两个漏洞:
1. **外观混淆**:初始状态泄漏 verb,margin 偷学外观;
2. **相位错配**:insert 和 remove 的大量帧对是"伸手接近"阶段,两边瞬时运动几乎相同,
   clip 级标签下发到帧级产生大量错误配对。

**双门 AND**:配对必须同时通过
- **语义门**:同 task 相反 verb(或同 episode 内的反向片段);
- **运动学门**:帧对瞬时运动**反平行**且幅度超阈。

⚠️ **运动学门必须加时间窗(A3 审计实测,2026-07-20,详见 `dir_pair_audit.md`)**:
不加时间窗的"反平行 + 幅度"规则**不成立**——全 episode 范围内反平行率仅比
各向同性 null 高 **1.24×**,且反平行对的 Δt 分布与随机配对**完全一致**
(中位 5.47s vs 5.60s)⇒ 挑出的绝大多数是巧合,照此训练 L_dir 会退化成
弥散排斥项(≈ uniformity 正则)。**富集只存在于 Δt≈0.5–2s**:
p90 幅度门 + Δt 0.5–1s ⇒ 反平行率 66.3% vs null 25.6%(**lift 2.59×**);
而 Δt<0.5s 的 lift 只有 0.24(相邻帧同向continue,物理必然——这是测量有效的 sanity check)。
**定稿规则 = 同 episode ∧ Δt∈[0.5,2.0]s ∧ 幅度>p80 ∧ cos<−0.5(ego-compensated)**。
必配对照:**同等对数的随机对排斥项**——若效果相同,方向主张不成立。

被门正确排除的例子:insert/remove 的"伸手"帧运动同向 ⇒ 不反平行 ⇒ 不配对
(相位错配解决);"手放回桌子"幅度/方向不匹配 ⇒ 不配对。
**关键**:取负发生在**帧对级**,clip 级标签只做语义门。

⚠️ 故事 (b) 约束:运动学门必须用**自监督运动代理**(光流方向 / 手部关键点轨迹),
**不能用 GT 18D twist**(强标签进主线 loss 与"自监督+弱标签"定位冲突)。
GT-twist 版降级为 oracle 消融,用于展示 headroom。

补充:twist 取负 ≠ 视频倒放。倒放(Ours-B)是**合成假输入**;twist 取负是
**真实输入 + 标签侧配对规则**。标签本是手系相对位姿 `inv(T_t)·T_{t+skip}`,
log 空间取负对旋转+平移同时良定义(拧入↔拧出 = ω 反号),
Ours-C 式 6D 直接取负的问题不存在。

### 怎么"让相反对共享初始帧"(§G3b 的操作化)

这是**配对时的过滤条件,不是新加的 loss**:
- **首选:episode 内配对**。伸手/收手、推进/回撤在真实数据里大量存在。同场景同外观
  是天然共享 ⇒ z_a 想用外观区分它们**没有信号可用**,只能编方向。钳死外观通路的最强形式。
- **跨 episode 相反对**:用 static latent `E(o_t,o_t)`(或其 z_e 部分)做**初始帧匹配门**
  ——余弦近邻检索,只保留初始外观相近的对。⚠️ 不是加 loss 逼 static latent 相近,
  是**筛选配对分布**,把外观差异从配对里剔除,margin 就没外观可偷。
- **负样本**:对方向 margin 而言,"负"就是反平行对本身(推远),"正"是跨 episode
  同向对(拉近,运动学门判定)。不需要 InfoNCE 式大负集。

### 数据量与过拟合(双门通过率问题)

**通过率目前无实测数字**——stage-1 动手前的第一件事是离线配对审计
(flow/关键点挖一遍,按 episode 报告通过率;CPU+轻 GPU)。推理层面的三个锚点:
1. **配对源不稀缺**:~74 万训练帧对,几乎每条 episode 都有 reach/retract 相位
   (episode 内配对**不依赖任务可逆、不依赖 verb 标签**——这正是 L_dir 能 label-free
   的原因)。对照:R2 的语义配对在 task-PK 下 ~54% micro-batch 有 anchor、照常收敛;
   CD-LAM 的 L_ctr 只有 **36.6%** 的对有标签(论文原文)。辅助配对 loss 部分覆盖是
   常态,因为没配上的样本仍在主 loss(recon/KL/action)里。
2. **经典过拟合不存在**:L_dir 是 latent 几何上的 margin 约束,无 per-pair 可学参数。
   真实风险是**覆盖偏置**(门只放行大幅运动帧 → 慢速相位不被塑形)与**代理噪声**
   (光流/关键点判错反平行)。
3. **对策有先例**:软权重代替硬门(`_core_action_weights` 模板)、λ 小 + warmup、
   每步打印 pair 数(pctr 同款)、per-step 最少 pair 阈值不足则 zero-loss fallback、
   验收用**全任务** direction_gain + taskheld R²(外部指标天然惩罚覆盖偏置)。
   初始帧匹配门只作用于跨 episode 次要源;主源(episode 内)天然 100% 共享场景,零损耗。

---

## E. 从定义到失败清单的闭环

**路径**:功能定义 → 每条性质一组外部指标 → 跑 baseline 家族 → **幸存的失败 = 论文靶子**。

| 性质 | 测量 |
|---|---|
| 充分 | R²_za / R²_ze / R²_full 三元组、labeleff |
| 不变 | task-held-out R²(+gap)、swap_context_cost、camera_shift |
| 可控 | ctrl ΔPSNR、do(z_a=0)、swap_opp_delta |
| (方向失败模式)| **opp_cls_direction_gain** |

⚠️ "方向"**不是定义的第四条性质**,是**充分性在方向自由度上的具体失败模式**。
按测量组织时可以并排,但概念上别把失败模式升格成定义性质——否则审稿人会问
"定义里为什么有 direction 没有 magnitude"。

### v2 定稿的失败清单(**四**项,不是三项)
1. **方向未编码**:direction_gain 全负(−0.20 ~ −0.31),无任何模型例外;
2. **外观在 z_a**:opp_cls_static 0.96–0.98、camera_shift 0.3;
3. ⚠️ **双向路由泄漏**:R²_ze = 0.15–0.18,**所有模型都漏**(最容易被漏掉的一项);
4. **旋转维弱**:待分组 R²(action_r2_rot)确认。

### 执行顺序
a) 组 cdlam_repro 配置训 5k(✅ 已完成)
b) 三 baseline(raw / KL / cdlam_repro)+ lf 孪生 + ours_d + idm 过完整 benchmark(T7)
c) 幸存失败排序 → 定稿失败清单与论文动机图表(T9)

---

## F. Loss ↔ 验收对照(每个 loss 绑定一个失败模式)

**规则:验收指标必须不是该 loss 的镜像。**

| Loss | 打哪个失败 | Tier-0 机制检查 | Tier-1 验收裁决 |
|---|---|---|---|
| recon + split-KL + L_action + L_zero | (trunk) | static_response | R² 不回退 + do(z_a=0) |
| **L_emb** 前景加权 | 外观在 z_a | FG-PSNR、camera_shift | opp_cls_static↓、swap_context_cost↓、R²_za 持平 |
| **L_dir** 反平行对 | 方向未编码 | 自己的 margin | direction_gain>0、swap_delta↑ |
| **decorrelation** | 动作漏进 z_e | 交叉协方差 | R²_ze↓ 且 R²_za 持平 |
| **L_pctr**(对照) | — | pctr 下降、pair 数 | 无预期,不判死刑 |

流程:失败模式 → loss(=机制假设)→ Tier-0 查机制(**没动 = 实现 bug**)→
Tier-1 查目的(**没改善 = 机制假设错**)→ Tier-2 抽查入围者。不过线不进下一轮。

---

## G. 五个关键问答

### G1. 什么是干预句柄?为什么 z_a 不是标签?

区别**不在训练时怎么用,在推理时的语义**。

训练 ACWM 时 z_a 确实像伪标签(teacher forcing:编码真实转移、拟合
`P(o_{t+1}|o_t, z_a)`)。但世界模型的用途是**反事实生成**:推理时插入的 z_a
**不是从当前真实转移编码来的**(那个转移还没发生),而是你**选择**的——策略输出的、
检索的、手工指定的。此时语义是 `do(z_a = v)`:固定其余、把条件通道设为 v,
期望输出是对应动作的后果。

- **"标签"描述已发生的数据;"句柄"控制未发生的生成。**
- **操作后果**:z_a 若混了外观,teacher-forcing 照样低 loss(z_a 与 o_{t+1} 外观一致),
  **干预时才穿帮**——把厨房的 z_a 插进卧室,条件与 o_t 矛盾。
- ⇒ 句柄的质量标准是**干预一致性(swap/do)**,不是拟合精度。

**充分性的代理 vs 可控性本身**:
- R² = "信息在、可线性读出"(**代理/下界**);
- 可控 = "解码端确实跟着 z_a 变"(**行为本身**)。
- 信息在 ≠ 因果通路在用。**IDM oracle(λ_recon=0)就是设计来演示这个分裂的**:
  预期 R² 全场最高、可控性全场最差 ⇒ 论文的 2×2 论据。

### G2'. ⚠️ direction_gain 的解释要收窄(2026-07-20,外部审阅后定稿)

**旧读法(过强,已废)**:"gain 全负 ⇒ 方向未被编码"。
**问题(天花板效应)**:`gain = A_transition − A_static` 是**两个独立 probe 的准确率之差**,
不是条件信息量 `I(Y; Z_trans | Z_static)`。我们的 `A_static = 0.960–0.994`
⇒ **正增益的理论余量只剩 0.006–0.040**。一个同时"挤掉部分外观 + 加入方向信息"的
transition latent,完全可能从 0.98 掉到 0.90 而得到**负** gain——负值既可能来自
"没有方向",也可能来自"外观被挤掉"。两者当前无法区分。

**处置**:
- 旧指标**改名 `static_shortcut_gap`**,降级为**泄漏诊断**(它可靠说明的是:
  transition latent 没能胜过静态外观捷径);
- **新 headline 用条件 probe**:同一 episode-grouped split,baseline 输入
  `[Z_static; 0]` vs 完整输入 `[Z_static; Z_trans]`(或 `[Z_static; Z_trans−Z_static]`),
  报告 **held-out NLL/log-loss 的下降**(不用接近饱和的 balanced accuracy),
  同模型族 + 嵌套 CV 选正则,paired bootstrap + episode 内 permutation 给 95% CI;
- 最终裁决仍归 **decoder 侧**(opposite swap、zero/shuffle 干预)与 Tier-2
  ——probe 只说明"可读出",不说明"decoder 因果使用"。

### 📊 条件 probe 的实测结果(15 个 checkpoint,train episode 池,2026-07-20)

`results/dirprobe_rescore.csv`。判读需两道门:`auc_full>0.5`(probe 真有预测力)
∧ `p≤0.05`(胜过置换 null)。⚠️ 原始 `gain` **系统性偏负**(full probe 多 32 个
含噪特征、要付过拟合代价,base 的零块被正则化掉)⇒ 无信息基线是 null 均值,不是 0。

| 家族 | vs_null 均值 | seed range |
|---|---|---|
| kl_full | +0.0592 | 0.0270 |
| ours_a | +0.0416 | 0.0243 |
| ours_r2 | +0.0697 | 0.0457 |
| **raw_lam(未在 EgoDex 训练)** | **+0.0774** | — |
| **cdlam_official** | **+0.0811** | — |

⚠️⚠️ **统计口径更正(2026-07-20,外部审阅)——本表全部降级为 exploratory**:
`n_perm=20` 时 Monte Carlo p 的**最小合法值是 (0+1)/(20+1)=0.0476**,报 `p=0.00` 无效。
按标准 `(b+1)/(B+1)` 修正后,"12/15 显著"实际只剩 **6/15**(且未计多重比较)。
另有四处待修:CI 报的是 raw gain 而非 null-centered、NLL 先按帧拼接(长 episode 权重过大)、
`C=1.0` 未做 nested CV、train 池被 encoder 训练时见过。
⇒ **confirmatory 版本需 ≥999 次置换 + episode→task 两级平均 + 模型间 paired
hierarchical bootstrap + 冻结一份 encoder 从未见过的评测 manifest**(阶段 B0)。

**三条读数(措辞已按审阅收窄)**:
1. **预训练 LAM 与多数微调 checkpoint 含少量超出静态 latent 的相反动词信息**
   ⇒ "方向未编码"被推翻。但**不能写"所有 LAM"**,也**不能写"任意非退化 encoder 天然携带"**
   ——raw_lam 是在大规模视频上预训练过的 DreamDojo LAM,不是随机 encoder;
   要支持后者需补 random-weight encoder / frame-difference / frozen-flow 基线。
   另注:probe 测的严格说是**条件相反动词信号**,不是纯运动方向
   (`Z_trans−Z_static` 也可能编码接触、速度、阶段、结果状态)。
2. **但 `raw_lam`(现成的 DreamDojo LAM,从未在 EgoDex 上训练)排第 4/15,
   高于我们全部三个家族的均值** ⇒ 这点信息**不是任何训练带来的**,
   而是任意非退化 transition encoder 天然携带的(运动本身与 verb 相关)。
3. **量级微不足道**:ΔAUC(full−base)全表在 **−0.008 ~ +0.017** 之间,
   过半模型为负;家族间差异(r2−kl=+0.011)远小于 seed range(0.024–0.046)。
   ⇒ **没有任何训练方法在这个指标上做出可辨别的改变**。

**主张相应收窄(论文口径)**:
> ~~CD-LAM 修正一般 confounding 后,方向仍是未编码的自由度。~~(实测推翻)
> ~~第三版:痕量方向信息在所有 LAM 中普遍存在、decoder 不使用它。~~(措辞过强,见下)
>
> **第四版(availability–utilization gap,2026-07-20 定稿)**:
> 初步条件探针表明,**预训练 DreamDojo LAM 及多数微调 checkpoint** 已含少量超出
> 静态 latent 的相反动词信息;但 **Stage-0 的各个目标没有相对匹配基线产生可复现的
> 放大**,也没有带来可复现的**方向特异** decoder 控制增益。本文因此研究
> **弱可读信息 → 因果可用动作条件**之间的缺口。

⚠️ 三处不可越界的措辞(外部审阅指出,均已收窄):
- ❌"任意非退化 encoder 天然携带" → raw_lam 是大规模预训练模型,不是随机 encoder;
- ❌"所有 LAM" → 只能说"多数被评测的 checkpoint";
- ❌"decoder 不使用 z" → `ctrl_delta_psnr` 与 zeroact 已证明 decoder **总体上会用 z**,
  `swap_opp_delta≈0.3` 也非零;现有证据只支持"**未观察到稳定的方向特异增益**"。
  在 paired CI + 多 seed 完成前一律加"初步"。

⇒ **L_dir 的目标**:把弱可读信息**放大**并**打通到 decoder 的因果通路**。
成功标准见 §F2(三层 H1/H2/H3 预注册),主对比是 **L_dir vs 匹配的非方向排斥对照**,
不是 vs no-loss baseline。

### G2. "96–98% 静态对照"这个结论怎么来的?

benchmark 的 `opp_cls` 协议(v2 新增静态对照):
1. 在 **28 个可逆 task** 上分层采样(每 task 两个相反 verb 各至多 40 clip);
2. 冻结 encoder,logistic probe 用 **z_a** 分两个 verb → `bal_acc` = **0.68–0.78**;
3. **对照组**:同一批样本、同一 probe 协议,输入换成**静止对 `E(o_t, o_t)`**
   ——两帧相同,构造上零运动信息,latent 里只可能有初始帧外观 → **0.96–0.98**;
4. `shuffled` 对照 ≈ 0.5,排除 probe 过拟合。

**推理链**:可逆 task 里初始状态与 verb 完全相关(待 remove 的插头必然插着、
待 open 的抽屉必然关着——**数据的构造性混淆**)。只含外观的表示反而分得**更好**
⇒ transition latent 的那 70% 可被外观**完全解释**,运动编码甚至冲淡了外观线索
(`direction_gain = bal_acc − static` 全负)。

⚠️ 结论的准确表述:**"没有任何模型给出了按方向区分的证据"**,
而不是"方向一定没编码"。现有证据全部可被外观解释——这正是 direction_gain
作为**唯一合格证据**的原因。

### G3. 方向配对怎么设计?
见 §D(双门配对 + 初始帧匹配门)。

### G4. cdlam_repro 复现的是什么?
**用 CD-LAM 的 Stage-1 loss 配方在我们自己的 trunk 上复现,证明其配方修正一般
confounding 之后,方向仍未编码。** 不原样复现是因为要**受控归因**。
完整的公式对照、五处偏离(尤其 **batch 3072 vs 64** 和 free-bits KL vs split-KL)、
以及"能与不能声称什么"见 **`cdlam_repro_fidelity.md`**。

⚠️ **标签空间三层事实(2026-07-20 定稿,详见 `cdlam_repro_fidelity.md` §3b)**:
① 论文 Appendix B = 12 类、相反动词合并;② **发布配方 `stage1_recipe.yaml` 与论文
逐字一致**(官方 checkpoint 按此训练)⇒ "合并 ⇒ 构造性失败"对官方产物**成立**,
我们的 repro 不是稻草人;③ 代码库另有未启用的 13 类 + `OPPOSITE_PAIRS` 扩展,
在发布配方下四对里只有 (open,close) 能解析,外加同 episode 硬负采样(p=0.5)。

**动机论证两头都硬**:合并是他们的正式选择(①②);且代码证明他们**知道**
相反对问题(③),启用的那部分(open/close 相反对 + 同 episode 硬负)仍未换来
direction_gain(官方 −0.241)——"不是没意识到,是 clip 级语义配对治不了",
相位错配漏洞适用 ⇒ **L_dir 的运动学反平行门仍是真正的差异点**。

**关键实证(动机段落最锋利的一句)**:CD-LAM 的 headline debias 指标
(项目页 `id_ratio`= 论文的 median zero-transition response)0.527→0.043,
我们的 `static_response_rel_median` raw_lam 0.672 → ours_a **0.035** / ours_r2 0.047
——**我们把这个指标打到了同一水平,而 direction_gain 仍然全负**。
即:*聚合 confounding 指标宣告"已修复"时,方向自由度依然缺失*。

官方已放出代码(github.com/yufanwei/CD-LAM)和 **32D LAM checkpoint**
(huggingface.co/yufanwei/CD-LAM)⇒ 建议接成 benchmark 的**外部锚点行**,
直接测官方模型的 direction_gain(证据强度高于我们的 repro)。

### G5. split(动作维/环境维)是好设计吗?KL baseline 会不会被推翻?

**诚实结论:split 的"分离功能"已经部分被推翻了。** v2 证明双向都在漏
(R²_ze 0.15–0.18);CD-LAM 无 split 全靠压力,反而压得更干净。
定稿判断:**有用但不完整**——split 是**供给侧**(给外观一个去处),
缺**需求侧压力**(把外观赶出 z_a)。

**为什么现有实验不会白做**(这是关键):
1. **压力栈(L_zero / L_emb / L_dir)在有无 split 下都定义良好**。KL trunk 是全家族
   共享的受控底座——**即使 split 的分离价值为零,家族内的所有比较照样成立**;
2. split 有独立于分离的**接口价值**:下游 ACWM 需要明确的条件通道,
   no-split 得事后 probe 找子空间;
3. 裁决消融已在 backlog:**no-split 对照**(单 40 维 + 同样压力栈,评测用 probe
   找动作子空间)。打平 ⇒ split 降级为"接口便利"如实声明;胜 ⇒ 拿到直接证据。
   **两种结果都写得进论文**,最坏情况是叙事从"split+压力"改成"压力为主、split 为接口",
   压力栈的全部结论原样保留。

**z_e 结构性 typing 是条件性必要,不是无条件必要**:
- 判据:**L_emb 之后 R²_ze 仍 ≈0.15 就上**;若 L_emb + decorrelation 已压下去就省掉。
- 它是**构造性根治**:z_e 改为只编码 o_t(`E(o_t)`,从不见第二帧)⇒
  "动作漏进 z_e"与"z_e wormhole 偷载下一帧"在构造上不可能。压力方法对此只是软约束。
- **次序**:先 no-split 消融(便宜、裁决 split 本身)→ 再决定 typing → z_e 扩容(8→16)排最后。

### G6. L_zero"校准原点"到底在干什么?

类比:**给秤去皮归零**。`L_zero = [‖E(o_t,o_t)‖ / sg(s_Δ) − m]₊²`——把"同一帧复制
两遍"(语义上确定无疑的"什么都没发生")的 latent 范数压到 ~0。它**不挤出任何信息**,
只是**给坐标系钉一个有语义的参考点**:训练后 z=0 代表"无动作"。没有它,
`do(z_a=0)` 毫无意义;有了它,置零应解码出"世界静止"(`zeroact_motion_suppression`
测的就是这个)。所以"L_zero 校准了原点 ≠ z_a 干净"——原点对了,离原点远的地方
装了什么(外观?方向?)它管不着。⛔ 推论:`static_response` 类指标**绝不可中心化**
(原点有语义,不是任意坐标;见 `cdlam_repro_fidelity.md` §3c)。

### G7. free-bits vs split-KL:正交,且 free-bits 已升级为 lf 线的必需品

两者回答不同问题:split-KL 决定压力**打在哪**(β_a 压动作/β_e 放环境——路由供给侧);
free-bits 决定**最深压到哪**(每维 KL 低于地板免罚——防维度压死)。可组合:
split-KL + 每维地板。两条实证让它从"卫生消融"升级:
1. **低秩已实测**:eff_rank 13–14/32、中心化后残余余弦 0.10——β_a=3e-3 无地板压秩
   的典型症状,free-bits 是教科书解法;
2. **lf 塌缩(T7)**:去掉 18D 监督,瓶颈直接压死 z_a(rank 2.8);CD-LAM action-free
   却安全,靠的正是 free-bits + stop 门槛(`eff_rank_drop_frac_max 0.20`、
   `kl_per_dim_mean_min 0.05`)。⇒ **label-free trunk 没有 free-bits(或 β_a≈0)
   活不下来**。注意:free-bits 改变 KL 语义,其 run 的数字单独成列,不与现表混比。

### G8. scene_knn 的发现(及一次数字修正)

`scene_knn_ep_top1` = z_a 余弦近邻中同 episode 占比(排除自身,对照解析 null)。
同码九行 + seed42:**ours_a 0.117/0.140/0.134(均值 0.130)vs kl 0.159/0.194/0.190
(均值 0.181)**——差 −0.051 vs 最大 range 0.035,超噪声 **~1.5×**
(早期口头说的"10×"基于只有 s1/s2 时的 range,已修正)。机制:ours_a 与 kl 只差
L_zero,它压掉的正是 episode 内所有帧共享的静态外观分量 → 同场景检索能力下降;
而 **r2 反弹回 0.176**——纯语义 margin(同 task 配对)把场景几何拉了回来。
一正一反 = §D 双门设计的又一证据。另:官方 CD-LAM 削锥后 scene_knn 纹丝不动
(0.183→0.184,23× null)——**公共偏移与场景聚类是两种独立混淆**(分析章素材)。

---

## H. 论文的一句话贡献

> "CD-LAM 修正一般 confounding 之后,**运动方向仍是条件通道中未编码的自由度**;
> 我们用真实数据的运动学反平行对将其注入,并以非循环协议证明。"

叠加故事 (b):主线 loss 不依赖 18D 强标签(caption 动词弱标签允许)。
所有实验都应服务这句话。

**headline 预注册五指标**:受控 R²、direction_gain、swap_delta、task-held-out gap、
Tier-2 FDCE。其余全部标记 exploratory;循环指标进 diagnostics 区并明说其地位;
seed 方差成表。
