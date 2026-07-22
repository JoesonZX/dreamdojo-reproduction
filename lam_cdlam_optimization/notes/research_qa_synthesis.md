# 研究问题综合梳理(2026-07-15,与 benchmark v2 结果对齐)

五个核心问题的定稿结论 + 讨论中涌现的问题 + 论文前还需回答的问题 + 当前流程。
数据支撑见 `results/lam_benchmark_v2/summary.md` 与 `notes/benchmark_v2_findings.md`。

---

## A. 五个核心问题(定稿)

### A1. 好的 z_a 怎么定义
**z_a = 引起两帧变化的、跨场景不变的原因,精确到视觉可分辨粒度。**
- 不是"运动本身"(物理全量不可恢复);不是"运动造成的变化"(绑定场景,swap 不迁移)。
- "原因"的检验是行为学的:干预 z_a 的效果 = 执行对应动作的效果(swap/do-test)。
- z_a 是**条件变量/干预句柄**,不是监督标签;定位为**运动学码**(Δt 内运动),
  场景相关的解释由世界模型结合 o_t 完成。"纯粹"是手段不是目标。
- 四条可测性质:充分 / 跨场景不变 / 可控 / 可接地。

### A2. 从定义到 benchmark 到失败模式
- 路径:功能定义 → 每条性质一组**外部**指标 → 跑 baseline 家族 → 幸存失败 = 研究靶子。
- 三层评估栈:Tier-0 诊断(loss 镜像,只否决/解释)→ Tier-1 冻结 encoder+外部目标
  (裁决模型)→ Tier-2 下游世界模型(终审)。循环性是 (loss, 指标) 二元组属性。
- baseline 双参照:**DreamDojo LAM**(证明问题存在)+ **CD-LAM 同尺度配方复现**
  (证明修正后问题幸存;三块料 L_zero/L_emb/SigLIP 对比代码库里都有,只差组配置 +
  12 类粗 primitive 标签)。repro 忠实保留其"相反动词合并同 primitive"的标签设计,
  使其在方向指标上的失败是构造性的 → 动机段落直接成立。
- v2 已确认的失败清单:①方向未编码(direction_gain 全负)②外观在 z_a
  (opp_cls_static 0.96、camera 0.3)③双向路由泄漏(R²_ze 0.15)④旋转维弱(待分组 R² 确认)。

### A3. 怎么判断 loss 设计得怎么样
- 流程:失败模式 → loss(=机制假设)→ **Tier-0 查机制**(loss 动了自己的量吗?
  没动=实现 bug)→ **Tier-1 查目的**(外部指标改善且无回退?没改善=机制假设错,
  例:B2 的 rev_cos 达标但外部全无改善)→ Tier-2 抽查入围者。
- 两类历史错误:**层级提升**(拿镜像当证据:B 用 rev_cos 宣称方向敏感)、
  **性质错配**(拿 A 性质指标证 B 性质主张:R 用连续保真 R² 证"相反可分")。
- 每个 claim 过两问:指标测的性质 = 主张的性质吗?指标是不是该 loss 的镜像?
- 当前 loss↔验收对照:L_emb → opp_cls_static↓ + swap_context_cost↓(R²_za 持平);
  L_dir → direction_gain>0 + swap_delta↑;decorrelation → R²_ze↓(R²_za 持平);
  SupCon-lite(缓行)→ 只认 swap/do/R²,永不 kNN。

### A4. 相反动作怎么分离
- 纯语义 verb 对在帧级有两个漏洞:**外观混淆**(初始状态泄漏 verb,margin 偷学外观,
  v2 静态对照 0.96 实证)与**相位错配**(insert/remove 的接近段瞬时运动相同,
  clip 级标签在帧级产生大量错误配对)。
- 倒放(Ours-B)= 合成假输入;**twist 取负 = 真实输入 + 标签侧配对规则**,本质不同。
  标签本就是手系相对位姿 `inv(T_t)·T_{t+skip}`,log 空间取负对旋转+平移同时良定义
  (拧入↔拧出 = ω 反号),Ours-C 式 6D 直接取负的问题不存在。
- 理想监督 = **语义 ∧ 运动学双门** + **episode 内反平行 twist 对**(同场景钳死外观
  通路,z_a 只能靠方向区分;伸手/收手、推进/回撤在真实数据里大量存在)。
- 前置条件:先把外观挤出 z_a(A5),否则任何相反 loss 都走外观捷径。
- 现状:相反分离在本数据上尚未被任何模型实现(含 reverse 系)。这是论文靶子,不是包袱。

### A5. 动作/环境维分离设计有没有用
- **有用但不完整**:split-KL 是供给侧(给外观一个去处),缺**需求侧压力**(把外观赶
  出 z_a)。v2 证明双向都在漏;CD-LAM 无 split 全靠压力,反而压得更干净。
- 概念修正:目标不是"模型不用初始外观"(它从 o_t 免费获得),是"**条件通道不携带
  外观**"。外观进 z_a 伤干预/可控性,不伤视频预测(fidelity–following gap),
  伤在:训练时拟合"状态→顺势动作"相关;干预时条件与 o_t 自相矛盾。
- 压力选项排序:L_emb 前景加权(现在,Ours-D)> 增广一致性(廉价搭档)>
  温和 decorrelation > 架构相机通道(camera 顽固再做)> V-JEPA latent 预测(下代主干)。
  对抗慎用(动作-场景在数据中相关 → 不可辨识 + 不稳)。
- "手动 vs 涌现"的正解:SAM3 mask 也是学出来的模型输出;这是组合先验,不是手写规则。
  embodiment 归纳偏置正是本任务该有的;纯涌现的代价是数量级更多数据。

---

## B. 讨论中涌现的其他有价值问题

1. **⚠️ 全家族带 18D 动作监督 ⇒ R² 是"准循环"指标(重要自我纠正)**:
   所有 ft 模型 `lambda_action=1.0`,action head 直接读 z_a——表示被训练成线性可解码,
   probe R² 部分是在重读训练头学过的映射。含义:
   - **家族内受控比较仍然公平**(所有模型同等监督,差异归因于附加 loss)——v2 的
     排序结论保留;
   - 但 **"我们的 LAM 动作可解码"不能作为绝对主张**(它被训练成这样);
   - 与 raw_lam / CD-LAM / DreamDojo(均无动作监督)的跨家族 R² 比较**不公平**;
   - 论文的外部主张必须落在:可控性(swap/do)、不变性(task-held-out、camera)、
     direction_gain、Tier-2 下游——这些没有被 18D 监督直接优化。
2. **循环性三分级**:同量同数据(真循环)/ 同量 held-out(弱证据)/ 异量外部(硬证据)。
3. **loss 与其镜像指标可判决相反**:CD-LAM L_ctr——kNN 类指标无效,但下游 FDCE
   +1.84px 证明有用。指标死 ≠ loss 死。
4. **verb-kNN 是范畴错误**:真值 18D 动作自己 top1=0.022 < null 0.037;为它优化的代价
   在 v3 实证(kNN z=4.1,R² 掉至 0.267)。已退役。
5. **指标资格规则**:loss 消费了什么信号,就失去对应镜像的 headline 资格(第 1 条是
   此规则对 L_action 的迟到执行)。
6. **相位/粒度错配**:LAM 编码在帧对级,动作语义在 clip 级;所有 clip 级标签下发到
   帧级都要运动学门控。
7. **seed 方差纪律**:0.02 量级差异先当噪声;跨机器/并行度差异会混入 σ(保守方向),
   报告时标注 regime。
8. 工程教训:长任务增量落盘;缺 checkpoint 跳过不崩;null 与指标同口径(permutation
   null,不是 majority 频率)。

---

## C. 完成一篇好论文还需要回答的问题

1. **监督定位(最紧急,影响故事根基)**:你的 LAM 是"supervised fine-grained LAM"
   还是"自监督+弱标签"?三选一:(a) 拥抱监督故事(EgoDex 标签是资源,但与
   CD-LAM/DreamDojo 的可比性要重新论证);(b) 最优配置补 λ_action=0 消融,证明
   附加 loss 不依赖标签;(c) 把"如何用连续动作标签塑形 latent action"本身当贡献。
   建议 (b)+(c) 组合:主线保留监督,消融证明鲁棒。
2. **Tier-2 缺位**:最小方案是轻量 ACWM(冻结 z_a 做条件的 next-frame 预测器)+
   FDCE 式前景位移评测 + do/transfer 干预;完整方案接 Cosmos。没有 (b) 级别的
   Tier-2,"对世界模型有用"始终是推测。
3. **贡献一句话**:"CD-LAM 修正一般 confounding 之后,**运动方向仍是条件通道中未
   编码的自由度**;我们用真实数据的运动学反平行对将其注入,并以非循环协议证明。"
   所有实验都应服务这句话。
4. **数据集外推**:单 EgoDex 必被审稿人问。最低配 task-held-out(已加);
   进阶配一个迁移集(HOI4D / Ego4D 子集 / AgiBot)。
5. **失败案例分析**:按 task 分层的 direction_gain——预测状态锁定型任务(抽屉/插座)
   最差、自由空间运动较好。验证 = 支持外观混淆机制;证伪 = 机制假设要改。论文分析章素材。
6. **z_e 的故事**:8 维 z_e 到底装了什么(per-dim η² 已有数据)?若塌缩或无信息,
   split 叙事需要修改。
7. **统计纪律**:headline 指标预注册(建议:受控 R²、direction_gain、swap_delta、
   task-held-out gap、Tier-2 FDCE 五个),其余全部标记为 exploratory;seed 方差成表。
8. **规模外推**:5k 步微调的结论如何外推?借 CD-LAM 的 less-is-more 叙事:LAM 侧小
   干预撬动下游大收益 + 数据分层实验(1h/10h/100h 式)。

---

## D. 当前流程(定稿版)

**阶段 0 — 地基(本周)**
1. 回收远端 3 个 seed(r2-s1/s2、a-s1)→ 跑 `lam_benchmark_seeds` → 方差表 →
   确认 R2 优势(0.365 vs 0.308)是否超噪声。
2. 组 `cdlam_repro_5k` 配置(KL trunk + L_zero + L_emb + 12 类粗 primitive SigLIP,
   保留 action head 以受控)并训练;SAM3 mask precompute 续跑补覆盖。
3. {raw, KL, cdlam_repro} 过完整 benchmark(含新加的分组 R²、task-held-out R²)
   → 定稿失败清单与论文动机图表。

**阶段 1 — loss 迭代循环(1-2 周)**
- 每轮:选一个失败 → 设计 loss → 1k 步烟测(Tier-0 机制动了吗)→ 5k 全跑 →
  Tier-1 验收(指定的外部指标改善 + R²/可控性无回退)→ 不过线不进下一轮。
- 排期:① Ours-D(L_emb)→ 验收 opp_cls_static/swap_context_cost;
  ② L_dir(episode 内反平行 twist 对 + 语义∧运动学双门)→ 验收 direction_gain/swap_delta;
  ③ 视 ①② 结果决定 decorrelation / 增广一致性;④ 最优配置补 λ_action=0 消融。

**阶段 2 — Tier-2 终审**
- 轻量 ACWM(冻结 z_a 条件)+ FDCE 式评测 + do(0)/target-transfer 干预;
  最优 2-3 个配置进入;(可选)Cosmos Stage-2。

**阶段 3 — 论文**
- 叙事 = 认识论结构(CD-LAM 同构):诊断(audit + 新失败清单)→ 方法(loss,每个
  绑定一个失败)→ Tier-1 → Tier-2 → 消融与失败分析。
- headline 预注册五指标;循环指标全部进 diagnostics 区并明说其地位。
