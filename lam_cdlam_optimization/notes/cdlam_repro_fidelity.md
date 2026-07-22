# CD-LAM 复现忠实度与指标对照(2026-07-19)

来源:`notes/Causal_Debiased_LAM (18).pdf`(论文正文,权威)、
项目页 https://yufanwei.github.io/CD-LAM-project-page/、
代码 https://github.com/yufanwei/CD-LAM、权重 https://huggingface.co/yufanwei/CD-LAM。

本文档回答三件事:cdlam_repro **复现了什么**、**哪里偏离**、**指标怎么和他们对齐**。
论文的 repro 章节直接引用本文档。

---

## 1. cdlam_repro 的目的(一句话)

**用 CD-LAM 的 Stage-1 loss 配方,在我们自己的 trunk / 数据 / 协议上训一遍,
证明它修正了一般 confounding 之后,"方向"这个自由度仍然没有被编码。**

不是验证他们的论文(那是他们的活),是**受控比较**:要把"方向仍失败"归因到
配方本身而不是 trunk/数据/规模差异,分母必须与我们家族全同。这也是为什么
不能"原样复现"——原样跑他们的 2B/14B + 他们的数据混合,任何结论都无法归因。

**为什么这个失败是构造性的(动机段落的核心)**:他们的 12 类 primitive 标签
**故意把若干相反动词合并**(pick–place、insert–remove、stack–unstack、
scoop–dump 各自合成一类;只有 open/close、turn on/off 分开)。L_ctr 把同
primitive 的转移拉近 ⇒ **相反动作被 loss 主动拉到一起**。这不是他们的疏忽,
是他们的粒度选择;我们的工作正是补上这一层粒度。

---

## 2. 逐项对照:他们的公式 vs 我们的实现

### L_ctr(Eq. 10)—— 完全一致 ✅
```
L_ctr = (1/|P|) Σ_{(i,j)∈P} softplus( −y_ij (τ v_i^T v_j + b) )
v_i = norm(r_ω(z_i)),  τ、b = learned temperature and bias
y_ij = +1 同 primitive,−1 异 primitive
```
我们:`train.py::primitive_ctr_loss`,`v = z_a_proj`(model 的投影头 = 他们的
`r_ω`),τ=exp(logit_scale) 初 10、b 初 −10 均为可学习标量,y 同上,排除自身对,
`primitive_of` 返回 None 的样本不进任何对。**逐符号对齐。**

### L_emb(Eq. 8–9)—— 一致,我们用了更细的 mask ⚠️
```
W_t = α_fg M_t + α_bg (1−M_t),  α_fg > α_bg
L_emb = (1/|Ω|) ‖ W_t^{1/2} ⊙ (ô_{t+1} − o_{t+1}) ‖²
```
我们:`lam_loss` 的加权 MSE,数学形式相同(W ⊙ 平方误差)。
**偏离**:他们是单通道 SAM3 前景 mask M_t ∈[0,1] + 两个标量;我们是 3 通道
`[hand, object, contact]` + 权重 `[3,5,10]`、背景 1×。我们的是更细的变体
(信息更多,方向一致)。缺 mask 的样本回退均匀权重。

### L_cal(Eq. 11–12)—— **一半一致,一半偏离** ⚠️⚠️
```
L_cal = L_KL-fb + L_zero
L_zero = E_{o_t} [ ‖z_t^0‖ / (sg(s_Δ)+ε) − m_zero ]_+²        (Eq. 12)
```
- **L_zero 完全一致** ✅:`zero_transition_loss` 就是这个式子(stop-grad 的
  running RMS 归一化 + margin + 平方 hinge)。
- **L_KL-fb 偏离** ⚠️:他们用 **free-bits KL**(逐维 free-bits floor 之上才罚),
  **单一 d 维 latent,无 split**。我们用**分区 KL**(β_a=3e-3 压动作子空间、
  β_e=1e-6 放松环境子空间)。
  **理由**:split-KL 是我们全家族共享的受控 trunk;换成 free-bits 就和其余
  8 个 run 不可比,归因失效。**代价**:cdlam_repro 不是 CD-LAM 的容量控制方案,
  论文里必须写明"我们移植了他们的三个目标项,容量控制沿用我们的 trunk"。
  (若要补,free-bits KL 是一个便宜的额外消融。)

### 保留 action head —— 已知偏离 ⚠️
他们 Stage-1 完全 action-free。我们保留 `lambda_action=1.0`,与家族其余模型一致。
后果:cdlam_repro 的 R² 是准循环指标(和我们家族其他模型一样);
**lf 孪生(`ft_l40_full_5k_lf`、`ft_l40_ours_r2_opp_l01_5k_lf`)才是与他们
Stage-1 设定对齐的 action-free 对照**——这也是故事 (b) 的落点。

### 规模 —— **最大的偏离,必须诚实声明** ⚠️⚠️⚠️
| | CD-LAM Stage-1 | 我们 cdlam_repro |
|---|---|---|
| GPU | 96×H100 | 2×L40/A6000 |
| per-GPU batch | 32 | P×K=8 |
| 有效 batch | **3072** | 8×accum4×2 = **64** |
| 步数 | 1k | **5k**(给了 5× 预算) |
| debias 数据 | 100h tier | EgoDex test_240p |

**为什么这条对 L_ctr 特别致命**:pairwise loss 的对数 |P| ∝ B²。他们每步约
9.4M 对,我们每 micro-batch 只有 56 个有序对(实测 pos≈15–21、**neg≈5–8**)。
负对稀缺是 task-PK(每批 2 个 task)+ 12 类粗 primitive 大类聚集的直接后果。

**因此本 repro 能与不能声称什么**:
- ✅ 能说:"在受控 trunk 下移植其配方,**相反动作方向仍未编码**"——因为这个
  失败是**标签设计的构造性后果**(相反动词合并),与 batch 规模无关。
- ❌ 不能说:"L_ctr 无效 / 我们复现不出他们的收益"——我们的 batch 小 48×,
  对比学习本身没有被公平检验。L_ctr 的 Tier-1 验收因此**不设预期、不判死刑**。
- ✅ 反驳"训练不够":我们给了 5× 步数预算。

---

## 3. 指标对照 —— **重要纠正**

### CD-LAM 的 `id_ratio` = 零转移响应,不是身份检索
项目页图里标的 "id_ratio 0.527 → 0.047" 与论文正文的
> "the median **zero-transition response** drops from 0.527 to 0.043"
是同一个量。**id = identical frames(重复帧输入),不是 identity/scene**。
它就是 `‖μ(o_t,o_t)‖ / s_Δ` —— **我们早就有的 `static_response_rel_median`**。

**直接可比的数字(v2 已有,无需重跑)**:
| | 基线 | 加了零转移校准后 |
|---|---|---|
| CD-LAM 论文 | DreamDojo **0.527** | CD-LAM **0.043** |
| 我们 | raw_lam **0.672** | ours_a_zero **0.035** / ours_r2 **0.047** |

**这是论文动机最锋利的一句话**:我们的 L_zero 把他们的 headline debias 指标
打到同一水平(0.035–0.047 vs 0.043),而 **`opp_cls_direction_gain` 仍然是
−0.25 ~ −0.27(全负)**。即:*聚合 confounding 指标宣告"已修复"时,方向自由度
依然缺失* —— 混淆被修正 ≠ 条件通道完整。

### 他们 Table I 的三个诊断 ↔ 我们的三个
| CD-LAM Table I | 数值 | 我们的对应 |
|---|---|---|
| zero-transition response(=id_ratio) | 0.527 → 0.043 | `static_response_rel_median` ✅ 直接可比 |
| camera-shift response | 缩小 3.6–8.3× | `camera_shift_h/v_median` ✅ 同族(口径未必同) |
| shortcut leakage | 0.151 → 0.014 | `task_shortcut_leakage` ⚠️ 我们自己的操作化,定义未公开,**不可直接比数值** |

### `scene_knn_*`(2026-07-19 新加)是我们自己的指标
top-k 余弦近邻中同 episode/同 task 的比例 + 解析随机 null。它填补 v2 里全 nan 的
`episode_shortcut_leakage` 槽位。**与 CD-LAM 无任何数值可比性**——最初命名为
`id_ratio` 是基于项目页的误读,已于本日改名。
⚠️ `results/lam_benchmark_seeds`(T8)的 CSV 列名仍是旧的 `id_ratio_*`,
计算完全相同,仅表头是改名前的产物;T7 及之后为 `scene_knn_*`。

### FDCE(Eq. A.3–A.5)—— Tier-2 直接照抄他们的口径
相对首帧的位移向量 `a_j^s = p_j^s − p_j^0`;轨迹间距离
`c_ij = (1/H) Σ_s ‖â_i^s − a_j^s‖₂`;对称 Chamfer
`FDCE = (1/2N_g) Σ_i min_j c_ij + (1/2N_r) Σ_j min_i c_ij`。
SAM3 前景 mask + CoTracker 点轨迹,每对 rollout 至多 16 个锚点,报 mean 和 median。
**stage-2 的 ACWM 评测直接按此实现,数字与他们同量纲可比。**

### 数据集选择被验证
他们 Stage-1/2 的 latent 条件 rollout 评测用 **300 条 held-out EgoDex**;
Stage-3 用 AgiBot。⇒ 我们选 EgoDex 与他们主设定一致;
**迁移集选 AgiBot** 就是在他们的地盘上直接可比,优于 HOI4D/Ego4D。

---

## 3b. 论文 vs 已发布代码的标签空间——**三层事实,逐一核实(2026-07-20 定稿)**

初版本节曾断言"已发布实现用 13 类、pick/place 分开"——**过度修正,已订正**。
逐字核对后的三层事实:

1. **论文 Appendix B**:12-way canonical verb set,原文逐字:
   *"pick–place, insert–remove, stack–unstack, scoop–dump, open, close, turn on,
   turn off, wash–rinse, cut, stir, and pour"* —— 相反动词**合并**;
   clean-ego 索引 25,192 个有标签对(36.6%),长尾,batch 内有效 8–10 类。
2. **发布的训练配方 `configs/stage1_recipe.yaml`**:
   `canonical_primitives: [pick_place, insert_remove, stack_unstack, scoop_dump,
   open, close, turn_on, turn_off, wash_rinse, cut, stir, pour]`
   —— **与论文逐字一致(12 类合并版)**。⇒ 官方 checkpoint 按此配方训练,
   **"相反动词合并"对官方产物成立,原"构造性失败"论证有效**。
3. **代码库 `lam/contrastive.py` 里存在一个未被配方启用的扩展**:docstring 说
   "canonical 13"(pick/place 分开),定义了 `OPPOSITE_PAIRS`
   {(pick,place),(open,close),(push,pull),(fold,unfold)} 作加权硬负样本,
   并注明 canonical 里缺的对**静默跳过**。在发布配方的 12 类合并空间下,
   **四对里只有 (open, close) 能解析**(pick_place 是一个类;push/pull/fold/unfold
   不在 canonical 里)。另:`p_negative_same_episode: 0.5` —— 负样本一半取自
   **同 episode**(外观受控的负样本,与我们 §D 的 episode 内配对同理,但只用在负边)。

**对动机论证的净影响(最终版,两头都硬)**:
- "合并 ⇒ 构造性失败"**对官方 checkpoint 成立**(第 2 层),原论证保留;
- **且**代码库证明他们**知道**方向/相反对问题(第 3 层的 OPPOSITE_PAIRS、
  low-motion/camera-dominant ignore 图),唯一启用的相反对 (open,close) 加上
  同 episode 硬负采样**仍未换来 direction_gain**(官方 −0.241)——
  "不是没意识到,是 clip 级语义配对治不了"——相位错配漏洞适用,
  **L_dir 的运动学反平行门仍是真正的差异点**;
- 我们的 `cdlam_repro`(12 类合并)复现的正是**论文与发布配方共同采用**的空间,
  不是稻草人。

**附:发布配方里的其他可比信息**(2026-07-20 逐项核实):
- `identity_ratio_max: 0.20`(id_ratio = identity-pair response = 零转移响应,再证
  §3 的指标对应)、`pairwise_cos_raw_max: 0.65`(公共锥 0.62 的出处,是训练停机线)、
  `eff_rank_drop_frac_max: 0.20` —— ✅ **已核实 `train.py` 确实读取并执行**。
- ⚠️ **更正**:`kl_per_dim_mean_min: 0.05` 虽写在 YAML,但**全仓搜索无任何代码读取**
  (train.py / eval_guardrails.py / optimizer_helpers.py / stage1.py 均 0 次命中)
  —— 它是 **inert config key**,不是生效机制。此前本文档与 memory 把它写成"停机门槛"
  是错误的,已订正。
- `beta_kl_init 1e-4 → target 1e-3(500 步 ramp)`:比我们 β_a=3e-3 低一档,
  且他们对全 32 维统一施压 + free-bits 地板,我们对 32/8 分区非对称施压。

### ⚠️⚠️ 更正(2026-07-20):`L_use` 在官方配方里是**关闭的**,不是 CD-LAM 的组成部分

本节初版称"他们的防塌机制还有一个硬性 decoder-use 约束"——**错误,已核实并订正**:
- `configs/stage1_recipe.yaml` 第 71 行:**`lambda_use: 0.0`**;
- `lam/train.py` 第 284 行注释:`masked_reconstruction_enabled=True` 时
  **trainer 显式跳过 `L_id, L_use_full, L_use_inter, L_aux_triplet, L_rec_inter`**;
- 论文公式 `L_cal = L_KL-fb + L_zero` 里**不含 `L_use`**。

⇒ 它是**公开源码中未启用的实验机制**,不是论文验证过的成分;而且是 soft hinge,
不是"硬约束"。我此前据其 docstring("L_use is HARD requirement")下的结论无效。
**它仍可作为我们自己的候选方案**(见 stage-1 待办),但**不能援引 CD-LAM 作为背书**。

### 源码里那段 docstring 的原文(供参考,注意它描述的是未启用路径)
核实 free-bits 实现时挖到(`optimizer_helpers.py` 模块 docstring + `latent_losses.py`):
```
- L_use is HARD requirement: decoder must use z
  (real_z reconstructs better than zero_z and shuffled_z by margin_use).
- KL free-bits opt-in (KL_dim_eff = max(KL_dim, free_bit)),  free_bit = 0.5
- masked_usage_gap: 铰链 MSE(zero_z) > MSE(real_z) + margin,shuffle 同理;
  full-image 与 interaction-region 各一份
```
**这直接命中我们 lf 塌缩的死因**:我们的 lf 双子正是"decoder 完全无视 z_a"
(ctrl/zeroact/swap ≈ 0),而 **L_use 就是把"解码器必须用 z"写成显式 loss**。
free-bits 只是"不再惩罚低 rate"(dead zone,**不产生把 KL 拉起来的梯度**),
它无法阻止 decoder 绕过 latent —— 所以只加 free-bits 很可能救不活 lf。
⇒ **lf trunk 重建的最小组合是 free-bits(供给 rate)+ usage-gap 铰链(强制使用)**,
而不是只加 free-bits。这是本轮最有操作价值的发现。

### 论文引用注意(2026-07-20 核实)
- **作者单位是 Aether AI + UC San Diego,不是 NVIDIA**(本地 PDF 首页;它建立在
  NVIDIA DreamDojo 之上但非 NVIDIA 工作)。此前文档中"NVIDIA 的后续工作"是错误,已订正。
- 当前是 **2026-07-10 的 arXiv 预印本**(2607.09185),引用时写"预印本"而非"已发表"。
- 官方公开权重 metadata 是 **step 300**,而论文/官方 pipeline 的 Stage-1 是 **1000 步**
  ⇒ 锚点行是**公开产物锚点,不是论文完整结论的替身**,写论文时必须披露这一点。
- 他们 Stage-1 的准确定位是 **action-unlabeled(无可执行动作标签)**,
  **不是 label-free** —— 它仍用 caption 派生的 12 类 primitive 弱标签 + SAM3 mask。

**已验证一致的部分**(源码 `SigLIPHead`):2 层 MLP + L2 归一化、可学习 scale/bias、
`init_log_scale = log(10) ≈ 2.302585`、`init_bias = −10.0` —— **与我们 T2 的实现逐项相同**。
且注释明确 "`z_mu` itself remains the latent the WM consumes; we only use `P` during training"
⇒ **世界模型吃的是 raw μ,不经投影头、不经 centerer** ⇒ 我们锚点行读 raw μ 是正确的。

### checkpoint 里的额外发现
- `step: 300`(不是论文说的 1k);`model_scale: 2B`;
  metadata 里 base = `nvidia/DreamDojo` 的 `2B_pretrain/iter_000140000`。
- ✅ **已实测证实:他们的 base 就是我们的 base**。逐张量比对官方 CD-LAM 权重与
  `LAM_400k.ckpt`:839/839 同名同形,**全局相对 L2 距离 ‖A−B‖/‖A‖ = 0.0015(0.15%)**
  ——正是 300 步小 lr 微调的量级。⇒ DreamDojo 的 2B ACWM 预训练期间 LAM 基本冻结,
  iter_140k 里打包的 LAM 与 LAM_400k 数值上不可区分。
  **重大含义:`raw_lam → cdlam_official` 是同 base 的干净 before/after**,
  其差值**可以**直接归因于 CD-LAM 已发布的 Stage-1 debiasing 训练。
  (改动集中在 `encoder.out.*` 与 `fc.*`(latent 头,相对变化 3.5–5%),trunk 几乎没动
  ——300 步主要在重塑 latent 投影,与"debias 而非重学表征"的定位一致。)
  ⇒ **无需更换我们的 baseline**;`raw_lam` 行本身就是 CD-LAM 的起点。
- `siglip_head` 学到的值:**τ = exp(2.267) = 9.65、b = −3.02**。
  对照我们 cdlam_repro:τ 10.0→10.2、b −10.00→**−9.97**(几乎没动)。
  ⇒ 大 batch 下 b 会大幅漂移以校准正负比例;我们的小 batch régime 里它基本不动。
  这是"batch 规模偏离"在参数上的直接证据。
- `centerer`(alpha=0.95 的 EMA 均值,`center` 范数 1.019):**只用于对比 loss 内部**
  (源码 `EMACenterer` + `centered_supcon_loss`,注释:"raw pairwise cosine ~0.62 …
  we never reward the bias direction"),**不是推理时的 latent 变换**。
  💡 他们实测 raw z 的**平均成对余弦 ≈0.62**(存在"共同锥")。**我们已实测对照,见 §3c。**

---

## 3c. 共同锥实测(2026-07-19,n=600 val 样本,z_a 子空间)

`cone_ratio = ‖mean(z)‖ / RMS(‖z‖)`;`cos_raw/cos_cen` = 中心化前/后的平均成对余弦。

| run | cone_ratio | cos_raw | ±sd | cos_cen | ±sd | ‖z‖中位数 |
|---|---|---|---|---|---|---|
| raw_lam | 0.398 | 0.209 | 0.245 | **0.009** | 0.236 | 4.07 |
| **cdlam_official** | **0.237** | **0.058** | 0.227 | 0.006 | 0.242 | 2.89 |
| kl_ft_l40_full_5k | 0.267 | 0.246 | 0.327 | **0.102** | 0.366 | 2.13 |
| ours_a_zero_5k | 0.260 | 0.242 | 0.332 | **0.110** | 0.380 | 2.14 |
| ours_r2_opp_l01_5k | 0.345 | 0.337 | 0.254 | **0.096** | 0.342 | 2.33 |

(latent-32 模型的 z_a = 全部 32 维;latent-40 模型取前 32 维,与 benchmark 全局口径一致。)

**四条读数**:
1. **我们的锥比他们温和得多**:cos_raw 0.21–0.34,**远低于 CD-LAM 报告的 0.62**
   (那是他们数据/训练下的值)。⇒ 余弦几何压缩问题对我们真实存在但**中等**,
   不至于让余弦类诊断失效。
2. **官方 CD-LAM 的锥最小**(cone_ratio 0.237、cos_raw **0.058**,近乎平均正交)。
   由于 §3b 已证实**同 base**,这是一个干净的 before/after:
   **raw_lam 0.209 → cdlam_official 0.058**,即他们的 Stage-1 确实**把公共锥显著削掉了**。
3. **我们的训练反而略微增大了锥**(kl 0.246 / ours_a 0.242 / ours_r2 0.337 vs raw 0.209);
   ours_r2 最高,与它的 margin loss 直接在余弦空间操作一致(值得 stage-1 复查)。
4. ⚠️ **中心化之后仍有残留结构**:raw_lam / cdlam_official 中心化后 cos ≈ 0.006–0.009
   (锥就是全部),而**我们的三个模型仍有 0.096–0.110**。
   ⇒ 我们的 latent 不只是"偏移",而是**挤在一个低维子空间里**。这与 v2 的
   `effective_rank`(raw 26.07 → kl 13.4 → ours 13–14)完全吻合:**KL 瓶颈把有效维数
   砍了一半,残留的正相关是低秩结构而非公共偏移**。中心化治不了这个。

**结论(操作层)**:
- **不要**把余弦类指标改成中心化版本(会破坏与 v2 的可比性,且我们的锥不足以使其失效);
  **新增** `cone_ratio` / `cos_raw` / `cos_cen` 三个诊断,余弦类指标另出中心化变体做稳健性检查。
- ⛔ **`static_response_rel_median` 绝不可中心化**:L_zero 把 **z=0 校准为"无动作"**,
  `do(z_a=0)` 依赖这个原点 ⇒ 原点在我们的设计里**有语义**,不是任意坐标。
- ✅ **headline 指标本来就免疫**:`opp_cls`(LogisticRegression)、`mlp_action_r2`、
  全部 ridge/labeleff 都先过 `StandardScaler`(自带减均值),常数偏移被 bias 项吸收。
  ⇒ **`direction_gain` 这个动机核心数字不受共同锥影响**。
- 📌 真正值得追的是**低秩**(effective_rank 13–14/32),不是锥。列入 stage-1 待办。

## 3d. cdlam_official 全量结果(n=1500,`results/lam_benchmark_anchor`,2026-07-19)

| 指标 | cdlam_official | 参照 | 读法 |
|---|---|---|---|
| static_response_rel_median | **0.067** | raw 0.672;论文自报 0.043 | ✅ debias 确认,忠实度校准通过 |
| opp_cls_direction_gain | **−0.241**(0.737/0.978,shuf 0.455) | 我们家族 −0.24~−0.28 | 💥 **动机核心:官方产物同样失败** |
| zeroact_motion_suppression | **5.04**(still_margin 4.71) | raw 1.37;我们 0.6–1.4 | 他们的零校准外部效果**极强**(300 步全花在这) |
| mlp_action_r2(**MLP** probe) | 0.104 | raw 0.093 | 它无 18D 监督 ⇒ 完全外部;与 raw 同水平 |
| action_r2_full_latent(**线性** probe) | **0.1266** | raw **0.1244** | ⚠️ 差值仅 **0.0022**;谈"线性可解码性"必须引这一行,不是上一行 |
| action_r2_trans / rot | 0.225 / **0.077** | — | **旋转弱(失败④)在官方模型上同样成立** |
| swap_opp_delta | 0.30 | 我们家族 0.26–0.44(range 0.12–0.17) | 在噪声带内,方向无证据,与 gain 一致 |
| swap_context_cost | **3.37** | raw 5.60;我们 ~0.9–1.5 | 动作码跨场景迁移代价仍高(介于两者) |
| scene_knn_ep_top1 | **0.184**(null 0.0079,**23×**) | kl 0.192 / r2 0.183 / ours_a 0.137 | **锥削掉了,但同场景检索混淆原封不动** |
| camera_shift_h/v | 0.442/0.388 | raw 0.380/0.343 | ⚠️ 我们的口径下**没有**降(论文自报 3.6–8.3× 降);协议不同,不可直接比,写论文时如实注明 |
| effective_rank / collapse | 24.9 / False | raw 26.1 | 未塌缩 |

三条分析素材:
1. **"锥 ≠ 场景混淆"**:cos_raw 0.209→0.058(锥删干净)而 scene_knn 仍 23× null
   ——公共偏移与"同场景检索"是两种独立的混淆,削锥治不了后者。
2. **他们 300 步的钱主要花在 do(u=0) 上**(zeroact 5.04 一骑绝尘),
   与论文以 do(u=0) 为 headline 完全自洽;但方向、旋转、跨场景迁移都没动。
3. camera_shift 的分歧展示了**协议敏感性**:同一主张(相机鲁棒)在不同操作化下
   结论相反,支撑我们"指标语义要逐个审"的方法论章。

### 💥 动机段落的最终形态(同 base、官方产物、逐项可验证)
> 在 CD-LAM 自己针对的每一项聚合混淆度量上,官方 checkpoint 都**确实被去偏了**:
> 零转移响应 0.672 → **0.067**(论文自报 0.527→0.043)、公共锥 0.209 → **0.058**。
> 然而它的 `opp_cls_direction_gain` = **−0.241**,与我们家族里每一个模型
> (−0.24 ~ −0.28,三 seed range 仅 0.009–0.024)**没有区别**。
> 即:**混淆被修正 ≠ 条件通道完整;方向仍是未编码的自由度。**

---

## 4. 已发布资源与建议用法

| 资源 | 地址 | 建议 |
|---|---|---|
| 论文 | 本地 `notes/Causal_Debiased_LAM (18).pdf` / arXiv 2607.09185 | 权威源,公式以它为准 |
| 代码 | github.com/yufanwei/CD-LAM | Stage-1 训练码 + `run.sh score-fdce` 评测码 |
| 权重 | huggingface.co/yufanwei/CD-LAM,`models/lam/model.pt`,**32D latent** | 见下 |

**官方 LAM checkpoint 的正确用法 = benchmark 的外部锚点行**(不是替代 repro):
- 像 `raw_lam` 一样注册进 `RUN_SPECS`,在我们 EgoDex val 上过全套指标;
- 价值 ①:校准 repro 忠实度——它的 `static_response_rel_median` 应落在 0.043 附近;
- 价值 ②:**直接测官方模型的 `opp_cls_direction_gain`**。若也为负,动机就有了
  **官方 checkpoint 级证据**,比我们的 repro 更硬,审稿人无从辩驳;
- 价值 ③:它没消费 18D 监督 ⇒ R² 对它是完全外部的公平指标;
- 工程注意:**latent 32 维、无 action/env split** ⇒ 所有 `_za`/`_ze` 子空间指标
  不适用(按 raw_lam 的处理方式走 full-latent 口径);需确认其 encoder 接口能否
  套进 `load_model`。
- `run.sh score-fdce` 可直接复用于 stage-2,省掉自己实现 FDCE 的验证成本。

---

## 5. 待办
- [ ] 接入官方 LAM checkpoint 作为 benchmark 外部锚点行(优先级高,证据强度最高)
- [ ] (可选)free-bits KL 消融,补齐 L_cal 的另一半
- [ ] stage-2 FDCE 按 Eq. A.3–A.5 实现,或直接复用其 `score-fdce`
- [ ] 迁移集用 AgiBot
