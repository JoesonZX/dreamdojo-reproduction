# 咨询提问稿:α=0 修复已成立,Tier-2 该怎么做 / 要不要做(2026-07-21)

发给外部顾问(ChatGPT 等)的自包含提问稿。仓库内读者:结论回来后记录在本文件末尾。
本轮是接着"阶段 B 方向线否决 → availability/utilization 分解 → 正对照硬门 → α 消融"的
第四轮咨询;前三轮见 `consult_story_b_prompt.md`。

---

## 一、一句话现状

方向线(L_dir)被双指标否决后,我们把问题重定位为 **availability × utilization**,并用一个
**正对照验证过的干预 assay** 证明了瓶颈是 **decoder 不使用 z_a**(不是信息不在 latent 里);
随后一个**极简、可归因、两 seed 复现**的干预——**训练时关掉喂给 decoder 的采样噪声(α=0)**
——把 decoder 对动作条件的利用率从 oracle 上限的 ~3% 抬到 ~72%,并恢复了**符号方向**使用。
现在唯一挡在"方法主张"前的缺口是 **Tier-2 下游 action-following**。本轮想请你判断:
**Tier-2 该怎么做、要不要做、以及一个我们自己拿不准的架构层面的桥接问题。**

## 二、关键结果(供你判断证据强度)

### 1. 正对照验证过的 utilization assay(硬门已过)
- 我们的 H2 指标 `C = E[(MSE_opp − MSE_same)/motion]`:把 anchor 的 z_a 换成"同/反"捐赠者的
  z_a,解码后对真值打误差。此前全臂 C≈0,曾被怀疑是"decoder 不用 z_a"还是"assay 不敏感"。
- 建了一个**训练无关的 frame-delta oracle**(按构造强可控的 decoder)+ 三种 donor 方案:
  `verb`(旧构造,d_same<d_opp 仅 0.488=未排序)、`dist`(按 18D 动作距离强排序,frac=1.0)、
  `random`(负对照)。结果:**oracle×dist=+1.15**(scorer 灵敏)、oracle×verb=+0.03
  (旧 verb 构造对完美 decoder 也近零 ⇒ 旧 C≈0 是 donor 伪影)、oracle×random=+0.06(干净)。
  ⇒ **assay 灵敏度已证**;旧 verb donor 构造是 bug。

### 2. α 消融:decoder-exposure 噪声是瓶颈(两 seed 复现)
只改训练时喂给 **decoder** 的重参数噪声 `z_dec = μ + α·σ·ε`(KL / action-head / L_zero 仍读 z_μ),
同 trunk / 同 split / 各 1k 步,两 seed(42, 1)。值 = seed42 / seed1:

| 指标 | A0(α=0,decoder 见 z_μ) | A1(α=1,标准 reparam) |
|---|---|---|
| C_dist / oracle 上限(动作**幅度**利用) | +0.830 / +0.825 = **72%** | +0.039 / +0.078 = 3–7% |
| C_antiparallel(幅度匹配、方向翻转 ⇒ **符号方向**利用) | **+0.381 / +0.386** | +0.064 / +0.073 |
| R_out 前景(swap 下解码输出位移) | 0.039 | 0.018 |
| E_own 重建 / val PSNR | 0.00025 / 32.96–33.27 dB | 0.00053 / 29.92 dB |
| 编码器后验(noise/per-dim SNR/eff-rank/类分离) | 1.00 / 0.54 / ~10 / 1.9σ | **1.00 / 0.54 / ~10 / 1.9σ(完全相同)** |

**读法**:α=0 恢复利用率(幅度到 72% oracle、方向 ~6× baseline)、**且保真 +3 dB**、
**且编码器后验完全不变**。机制:α=1 时 σ≈先验 ⇒ 逐维 SNR<1 ⇒ decoder 收到的是噪声主导的 latent,
学会忽略它、只从 o_t 复制;关掉噪声(α=0)让它重新信任并使用 latent。

### 3. α=0 的生成/OOD 鲁棒性(零训练,两 seed)
- **posterior 采样 regime(z=μ+s·σ·ε)**:A0 **脆弱**(recon MSE s=0→2:0.0003→0.005–0.007,
  ~5–10×;A1 平)——它训练时没见过噪声。**仅当下游需要采样 z 才是代价。**
- **do(z_a) 干预 regime(z_a→k·z_a,可控世界模型实际用的)**:A0 **良好**——motion(k) 单调增
  (0.012→0.023)、无伪影(TV 低于真实帧);A1 **惰性**(motion 平)。
⇒ 对 chosen-z_a 干预,α=0 不仅安全且更优。

## 三、我们自己拿不准的桥接问题(请重点判断)

α=0 修的是 **LAM 自己的重建 decoder**(VAE:`encode(o_t,o_{t+1})→z`,`decode(o_t,z)→ô_{t+1}`)
对 z_a 的使用。但我们设想的"下游可控世界模型"是 **DreamDojo 的 2B ACWM**,它是**另一个** decoder,
消费 LAM 的 **raw z_μ** 作为动作条件(已核实:ACWM 吃 raw μ,不经投影头/centerer)。

**矛盾点**:我们已经证明 **α=0 的编码器后验统计与 α=1 无异**。若 ACWM 消费的是(冻结)LAM 的 z_μ,
那么把 α=0-LAM 的 z_μ vs α=1-LAM 的 z_μ 喂给同一个 ACWM,**下游可能看不到差异**——因为 α 只改了
LAM 的**重建 decoder**,没改编码器给出的 z_μ 分布。(注:LAM encoder 未冻结,A0/A1 的 μ 映射不必
逐点相同,但其分布统计相同。)

这意味着 Tier-2 的"是什么"取决于架构选择,而这正是我们想请你判断的:
- **选项 A(Genie 式)**:把 **LAM 自己的 decoder** 当世界模型,做多步 action-conditioned rollout
  / do(z_a) 干预。α=0 的修复**直接适用**,成本低(不需要外部 2B 模型)。但它不是"真正的大世界模型"。
- **选项 B(独立 ACWM + FDCE)**:用 DreamDojo 2B ACWM 跑 rollout,按 CD-LAM 的 FDCE(前景位移
  Chamfer)评 action-following。这是"真 WM",但 α=0 的相关性存疑(见上),且可能需要把 α=0 原则
  **搬到 ACWM 训练**里才有效——那是大得多的工程。

## 四、可行性 / 范围(诚实交代)

- **Tier-2 / FDCE / ACWM / rollout / CoTracker 在我们代码库里完全未建**(grep 零命中);
  是否有 2B ACWM 权重 + rollout harness 本地可用,我们**不确定**。
- FDCE 还会重新用到 SAM3 前景 mask,而我们的 SAM3 object 通道 ~40% 任务相关缺失(已知问题)。
- 共享服务器,可用余量约几天机器时间。论文目标周期还剩约 1.5 个月。

## 五、请你回答的具体问题

1. **(最重要)LAM-decoder ↔ ACWM 桥接**:
   (a) LAM 自己 decoder 的 utilization(α=0 修的量)是下游 ACWM action-following 的**有效
   predictor/proxy** 吗?还是必须把 α=0 原则搬进 ACWM 训练才有意义?
   (b) 你更推荐**选项 A(LAM-decoder 多步 rollout)**还是**选项 B(独立 ACWM + FDCE)**作为
   我们的 Tier-2?给出理由与各自能/不能声称什么。
2. **Tier-2 是 workshop 必需还是 main 必需**:我们现在有一个正的、可归因、两 seed 复现、带 OOD
   分析的 decoder 侧修复。"诊断 + 修复"**不带 Tier-2** 够 workshop 吗?Tier-2 是 main 的硬需求,
   还是 main 更该先补 **cross-model / cross-data 广度**?给一个 (venue × 缺口) 的优先级排序,
   目标是"最少 run 拿到最高 venue"。
3. **Tier-2 协议 + 正负控制预注册**(沿用我们 H2 正对照的纪律):若做 Tier-2,成功标准、
   **已知强可控的正对照**、负对照、fidelity 守门各怎么设?action-following 用什么量
   (FDCE?do(z_a) rollout 的轨迹跟随误差?held-out task?),以及怎么避免"重建 swap 换个
   马甲"这一循环性陷阱。
4. **测量→预测桥**(你之前列的 main 门第 3 项):怎么设计实验证明"我们的 utilization 指标
   **能预测** Tier-2 action-following",而不只是与它同源?A0-vs-A1(utilization 差 20×)是不是
   现成的因果对照——若 A0 的下游 action-following 也显著优于 A1,就同时拿到"指标预测下游"
   + "α=0 有下游价值"两个结论?
5. **intermediate-α / α-schedule**:OOD 显示 α=0 只在采样 regime 脆弱。若 Tier-2 的 rollout
   需要采样(多步生成),是否该先跑一个 α∈{0.25} 或 schedule 的 run 再进 Tier-2?还是 do(z_a)
   干预不采样、无需?
6. **最小可行 Tier-2**:若本地没有 2B ACWM 权重/harness,避免多天工程的**最小**下游验证是什么
   (例如:LAM-decoder 的 K 步自回归 rollout + do(z_a) 轨迹跟随,用 GT 18D 或光流读出运动方向)?

---

## 顾问回复记录

# 顾问第四轮回复(α=0 修复后的 Tier-2 决策)

## 结论摘要

1. **α=0 已经是一个很强的 LAM 内部结果,但还不是外部 ACWM 结果。**它证明改变
   LAM 训练时的 decoder-conditioning channel 能恢复本 LAM decoder 的 action use;
   不能仅凭这一点推断任意新 decoder 都会更好地使用同一个 `z_mu`。
2. **不要把 α“搬进 ACWM”作为默认下一步。**如果 ACWM 本来就读取确定性的 raw
   `z_mu`,它没有 LAM posterior sampling 这项噪声;视频 diffusion 自身的采样也不是
   `q(z|o_t,o_{t+1})` 的采样。ACWM 是否忽略条件是另一个训练问题。
3. **固定一个已训练 ACWM,直接替换 A0/A1 encoder 不是有效主对比。**正确的 CD-LAM
   Stage-2 口径是分别用各自 LAM 提取的 latent 继续训练 matched ACWM,而不是对同一
   frozen ACWM 做 encoder plug-and-play。
4. **选项 A 适合作为立即做的 `Tier-1.5`,不能单独承担“下游世界模型有效”主张。**
   若 A/B 必须二选一作为真正 Tier-2,选 B;但在当前预算下,我更推荐第三条中间路线:
   **冻结 A0/A1 encoder,分别训练一个独立、轻量、从零初始化的 action-conditioned
   predictor**,先证明修复能跨 decoder 转移。
5. **workshop 不以 Tier-2 为硬门**,前提是标题、摘要和结论明确限于 LAM decoder。
   main 若声称“改善 downstream ACWM action following”,至少需要独立 decoder Tier-2;
   不一定非得是 2B,但只做原 LAM decoder 的重建/rollout 不够。
6. **当前不要先跑 α=0.25。**chosen-`z_mu` rollout 不采样 posterior;先完成 A0/A1
   的独立 decoder 对比。只有应用确实要从 posterior/latent policy 采样,或需要第三个点
   检验 utilization-to-Tier-2 单调关系时,α=0.25 才进入下一轮。

## 0. 在 Tier-2 前必须补的零训练定位实验

当前“编码器后验完全不变”只由 noise、平均 SNR、effective rank、类分离等**分布汇总量**
支持。两个 encoder 的这些汇总量相同,不代表逐样本映射
`mu_A0(o_t,o_{t+1}) == mu_A1(o_t,o_{t+1})`。α 位于重建路径上,其梯度会同时改变 encoder
和 decoder 的训练轨迹;因此现在更准确的因果表述是:

> Removing noise on the encoder-to-decoder training edge causes a matched
> encoder-decoder pair to learn a much more usable action channel.

而不是尚未完全证明的:

> Only the decoder changed; the encoder is unchanged.

先在同一冻结 manifest 上做两个零训练 audit。

### 0.1 paired-latent audit

逐 transition 比较 A0 与 A1:

- `||mu_A0-mu_A1||`,cosine 和逐维相关;
- linear CKA / orthogonal Procrustes 后的残差;
- kNN 邻居重合率、same/opposite action distance ordering;
- 用**同一个冻结 action readout**做 cross-encoder 预测,而非各自重新拟合 readout。

“平均 SNR 相同”只说明 rate/noise budget 相似;上述结果才说明 action coordinate 是否真的
保持不变。

### 0.2 四格 cross-decoding

计算四个组合,所有 latent 都用 `mu`:

| | `E_A0` | `E_A1` |
|---|---|---|
| `D_A0` | `D_A0(E_A0)` | `D_A0(E_A1)` |
| `D_A1` | `D_A1(E_A0)` | `D_A1(E_A1)` |

每格都跑 own/dist/antiparallel/random 的 `C`、`R_out` 和 fidelity。判读为:

- 若 `D_A0` 对两个 encoder 都强、`D_A1` 对两个都弱,才是很干净的 **decoder-side
  utilization** 定位;
- 若只有 matched diagonal 强,说明收益包含 encoder-decoder **co-adaptation/坐标协商**;
- 若 A0 encoder 喂给两个 decoder 都更好,α 已经实质改变了 encoder 的条件表示;
- 若 paired `mu` 几乎一致且只有 `D_A0` 强,那么外部 ACWM 仅看到 `mu` 时原则上不应
  自动获得 A0 的收益。

这是决定 Tier-2 架构的最高价值实验,比先搭 2B harness 更优先。

## 1. LAM decoder 与 ACWM 的桥接判断

### 1(a). LAM-decoder utilization 是 ACWM action following 的 predictor 吗

**目前是机制相关的候选 proxy,不是已验证 predictor。**两个 decoder 的条件路径、容量、
训练目标和 observation shortcut 都不同。LAM decoder 学会服从 `z` 不会作为参数或状态
“传给”另一个 ACWM;外部 ACWM 真正继承的是 encoder 产生的 latent dataset 及其条件结构。

这里应把两种“迁移”分开:

1. **parameter transfer:**不存在。`D_A0` 的利用能力不会进入独立 ACWM。
2. **conditioning-data transfer:**可能存在。α=0 虽未改变汇总 posterior statistics,
   仍可能改变逐样本 `mu` 的几何、局部平滑性或动作条件充分性,使新 ACWM 更容易学习。

因此正确的外部实验是:

```text
W_A0 = Train(W_init, o, mu_A0)
W_A1 = Train(W_init, o, mu_A1)
```

两边必须使用相同 `W_init`、数据、batch 顺序、步数、condition adapter、生成 seed 和评测
population。不能做:

```text
W_fixed(o, mu_A0) versus W_fixed(o, mu_A1)
```

除非 `W_fixed` 明确在两个 latent space 上共同训练过;否则差值主要反映 interface OOD/
coordinate mismatch。

也**不必把 α=0 公式机械搬进 ACWM**。按 prompt 已核实的接口,ACWM 输入已经是 raw
deterministic `mu`;那里没有 `sigma*epsilon` 可关。如果 matched ACWM 仍忽略条件,需要诊断
的是 ACWM 自己的 conditioning path,例如 observation-only shortcut、condition dropout/
guidance 或 counterfactual training,而不是 LAM posterior α。

### 1(b). A、B 与推荐的中间路线

| 路线 | 实际证据层级 | 能声称 | 不能声称 |
|---|---|---|---|
| A:原 LAM decoder K-step rollout | Tier-1.5 / within-model functional test | α=0 改善同一模型的短期/多步 latent controllability | 修复可迁移到独立 ACWM;改善 2B 世界模型 |
| C:独立轻量 ACWM,冻结 LAM encoder | **最小真正 Tier-2** | latent channel 的改进能否跨 decoder 学习并产生 action following | 与 2B diffusion WM 等价;大规模收益 |
| B:matched 2B ACWM Stage-2 | 完整系统 Tier-2 | 对 DreamDojo 级独立 ACWM 和 FDCE 有下游价值 | 自动推广到其他架构/数据 |

**推荐顺序是 `cross-decode -> A(快速) -> C(主 Tier-2) -> B(预算允许才做)`。**如果只能
在 A/B 选一个写成 Tier-2,概念上选 B;但以几天共享机器和 1.5 个月周期,直接选择 B 的
工程风险不合理。

## 2. 关于 CD-LAM/DreamDojo 可行性的事实核查

新公开信息比 prompt 中“完全不确定”稍好,但还不是 turnkey reproduction:

1. CD-LAM 论文的 Stage 2 是用 Stage-1 LAM 提取的 latent **继续训练 ACWM 2,000
   optimizer steps**,2B 的 per-GPU batch size 是 12。见官方
   [pipeline](https://github.com/yufanwei/CD-LAM/blob/main/docs/PIPELINE.md)。因此论文自身
   也采用 matched ACWM adaptation,支持上面的判断。
2. 官方仓库现在提供三个 2B research entries,其中 `models/pretrain/model.pt` 与公开
   **32D** LAM 是兼容对。见[仓库 README](https://github.com/yufanwei/CD-LAM)和
   [artifact contract](https://github.com/yufanwei/CD-LAM/blob/main/docs/ARTIFACTS.md)。
   这可以作为环境/rollout 正对照,但你们 A0/A1 是 40D,不能直接替换进去。
3. 官方明确写明:移植其他架构需要 matching adapter、LAM 和 preprocessing contract;
   完整 runtime 还依赖 NVIDIA base、tokenizer/text encoder、数据和 CUDA 环境。
4. 官方公开了确定性的 FDCE reducer,但**没有可移植的历史 rollout scripts、精确 300-clip
   manifest 或完整 rollout-to-SAM3-to-CoWTracker runner**。见官方
   [evaluation boundary](https://github.com/yufanwei/CD-LAM/blob/main/docs/EVALUATION.md)。
5. 更重要的是,官方当前的
   [evaluation protocol](https://github.com/yufanwei/CD-LAM/blob/main/docs/EVAL_PROTOCOL.md)
   披露了 manuscript formula 与历史 headline table 的 reduction order 不一致:当前
   canonical code 是“先对固定 track pair 跨时间平均,再 Chamfer”,历史表是“每帧
   Chamfer 后再跨时间平均”。两者不可交换。因此新结果应标注 canonical FDCE,**不能和
   论文 headline 数字作精确复现式比较**。

结论:官方资产值得做一个最多 1 天的 `runtime-doctor/dry-run` 可行性审计,但不要在审计
通过前把 2B Stage-2 放进主排期。

## 3. workshop/main 的缺口优先级

### Workshop

**现有 diagnosis + positive-control assay + α=0 两 seed + OOD 分析,在不做 Tier-2 时可以
构成强 workshop 稿。**条件是:

- 补上 paired-latent/cross-decoding,收紧“decoder-only”因果措辞;
- 把 claim 限定为 LAM reconstruction decoder / latent-conditioning edge;
- 不在摘要中声称“improves downstream ACWM action following”;
- 最好增加 A 的 K-step rollout,但它不是硬门。

### Main

不同 venue 的优先级略不同:

| 目标 | 第一缺口 | 第二缺口 | 第三缺口 |
|---|---|---|---|
| CoRL main | 独立 action-conditioned rollout(C 或 B) | 第二数据域/embodiment | 第二 LAM/decoder 架构 |
| ICLR main | 跨架构/跨数据的机制一般性 | 独立 decoder Tier-2 | 大规模 2B 系统结果 |
| Workshop | cross-decoding 因果定位 | A 或 C 的轻量扩展 | 2B 非必要 |

若论文仍把“world model consumes z”作为核心动机,**main 至少需要 C 级别的独立 decoder
证据**。完整 2B 不是形式上的硬要求;一个预注册严谨、正负控制齐全、跨 held-out task 的
轻量 ACWM 比一个单 seed、接口勉强接通的 2B demo 更可信。

“最少 run 拿到最高 venue”的建议顺序:

1. 零训练 paired-latent + cross-decoding;
2. A0/A1 两 LAM seeds 上做独立轻量 ACWM(C),下游至少两个 matched seeds;
3. 在未用于 H2 的 held-out split 或第二小数据域复现;
4. 在原生 32D trunk 或第二 LAM architecture 上复现 α 机制;
5. 只有 2B runtime 审计通过且算力明确时,再做 A0/A1 matched Stage-2 smoke/full run;
6. α=0.25/schedule 是后续机制曲线,不是当前投稿硬门。

## 4. 推荐的 Tier-2 协议与预注册

### 4.1 模型和数据

最低成本的独立 Tier-2(C):

1. 冻结四个 encoder:`A0-seed42/A1-seed42/A0-seed1/A1-seed1`;
2. 离线缓存 train transitions 的 raw `z_mu`,不回传 encoder;
3. 对每个 encoder 分别训练同架构的小型 next-frame/video predictor,从同一初始化开始,
   用 FiLM/cross-attention 等明确条件通道消费 `z_mu`;
4. 模型只看 `(o_<=t,z_t:t+K-1)`,不看真实未来帧;推理时自回归 K 步;
5. task/episode-held-out,且 Tier-2 test clips 与 H2 scorer-validation clips 分离。

可以复用 LAM decoder 的**架构定义**,但权重必须从零初始化且训练时 encoder 冻结;这样它
是独立 consumer,而不是原 reconstruction pair。若实现成本允许,用一个更小的 U-Net/
ConvNet predictor 更能证明跨 decoder transfer。

### 4.2 两个评测 regime

1. **own-latent rollout:**从 reference clip 提取真实 latent sequence,从同一 clip 初始帧
   rollout。测条件充分性、长期误差和 fidelity。
2. **target-action transfer:**固定 source context,插入另一个 target clip 的 latent
   sequence。donor 必须用 GT 18D distance/antiparallel 门控,不能回到已证伪的 verb donor。
   测 source 场景中是否复现 target 的运动方向/幅度。

K 建议先做 `4/8/16` 的 horizon curve,以 `K=8` 或 `16` 预注册主点。官方 FDCE 默认 49
frames;若只做短 horizon,应写 `FDCE-style displacement error` 或
`canonical-FDCE@K`,不能与 CD-LAM 49-frame paper numbers混称。

### 4.3 指标

**Primary action-following:**

- canonical FDCE/track-displacement error,使用 reference-defined foreground points;
- 对 magnitude-matched antiparallel donors,增加 signed displacement cosine 或 endpoint
  direction error,避免 Chamfer 均值掩盖符号;
- `Delta_do = Error(shuffled/antiparallel z) - Error(target z)`,直接测模型是否服从选定
  condition。

**Secondary:**

- 冻结 IDM 从生成 frame pairs 读出 18D action 后的 normalized L1/cosine;IDM 必须先在
  real/reference、persistence 和强可控正对照上校准生成域偏差;
- horizon-wise error slope,而不只报最后一帧;
- condition-response `R_out` 只作机制指标,不能替代 action following。

**Fidelity guardrails:**

- own-latent rollout 的 PSNR/FG-PSNR、SSIM、LPIPS;
- source-background LPIPS/PSNR,防止 target donor 携带外观;
- temporal TV/flow smoothness、有效 track 数和 catastrophic-failure rate;
- motion lower bound,防止静止复制通过像素 fidelity。

### 4.4 正负控制

**已知强可控正对照:**同一轻量 WM 架构直接以 GT 18D action 为条件,或一个在相同数据上
验证过的 GT-action-conditioned predictor。它必须在 target-vs-shuffle 和
target-vs-antiparallel 上显著分开。真实 reference/GT tracks 只验证 metric lower bound,
不能替代“模型会服从条件”的正对照。

**负对照:**

- observation-only/zero condition;
- within-task time-shuffled latent sequence;
- magnitude-matched antiparallel latent;
- persistence/static rollout。

所有模型共用 rollout seed 和 clip/donor triples。positive control 不过门,先修 Tier-2
scorer;不能用 A0/A1 的零结果反推模型。

### 4.5 成功门

在看到 A0/A1 Tier-2 结果前冻结 practical margin。margin 应由 A1 重复 inference 和
positive/negative control 的自然间距校准,不要从最终 effect 倒推。可采用如下形式:

1. A0 相对 A1 的 paired canonical-FDCE 改善达到预注册相对阈值(例如至少 10%),且
   task->episode/clip hierarchical 95% CI 排除 0;
2. A0 内部 `target z` 显著优于 shuffled 和 antiparallel,而 A1 的差距显著更小;
3. signed-direction metric 同向成立,不能只有像素/Chamfer 改善;
4. PSNR 非劣容差可先设为 `-0.5 dB`,LPIPS/背景 fidelity 容差由 A1 repeated-inference
   方差冻结;
5. positive control 和 negative control ordering 通过;
6. 以 downstream training seed 为独立复现单位;clip 级大 N 不能替代至少两个 matched
   training seeds。

### 4.6 SAM3 缺失的处理

40% object-channel 缺失不是随机噪声,很可能按任务类型选择性缺失。不得只在成功 mask 的
clips 上报一个不披露覆盖率的 FDCE。应:

- 在看模型输出前冻结 mask-eligible subset,各 arm 完全共用;
- 报总覆盖率、每 task 覆盖率和 included/excluded task 分布;
- mask 只从 reference/source 定义,不因某模型生成质量改变纳入概率;
- 主 FDCE 之外增加覆盖全 clips 的光流/手部关键点 displacement 指标;
- 不把新 canonical FDCE 与论文历史 FDCE headline 数字直接横比。

## 5. 如何避免 Tier-2 只是“重建 swap 换皮”

至少建立四个独立性:

1. **consumer 独立:**Tier-2 decoder 不复用 `D_A0/D_A1` 权重;
2. **数据独立:**H2 用 Eval-A 做模型选择,Tier-2 最终只在未看的 Eval-B/held-out task
   解封一次;
3. **horizon 独立:**H2 是 one-step,Tier-2 是 autoregressive K-step;
4. **metric 独立:**H2 是 donor-swap pixel MSE,Tier-2 primary 是 foreground track
   displacement + signed action metric,像素重建只作 fidelity guardrail。

满足这些条件后,A0 的 Tier-2 改善不是循环论证;它证明 H2 识别出的 conditioning failure
会跨 consumer 和 horizon 产生功能后果。

## 6. “测量能预测 Tier-2”的主张强度

### A0-vs-A1 能证明什么

若 matched independent ACWM 上 `A0 > A1`,可以同时写:

1. α=0 的收益从原 LAM decoder **迁移到独立 downstream consumer**;
2. H2 预先选出的高-utilization checkpoint 在未见 Tier-2 上也有更强 action following。

这是很强的 prospective causal validation。但只有 A0/A1 两个 x 值时,不要写成一般统计意义
的“utilization score predicts downstream performance”;两个点总能形成一条线。

### 要写 predictor 还需什么

- 在 H2 结果冻结后预注册 Tier-2,避免用 Tier-2 反向调 metric;
- 至少用 5--6 个 checkpoint-level points 覆盖 utilization range,例如现有
  `A0/A1` 两 seed、B/R/D/U,必要时再加 α=0.25;
- 每个 checkpoint 用 matched independent decoder,相关分析以 checkpoint/training seed
  为单位,不是把 clip 当独立样本;
- 预注册 Spearman rank/单调趋势,并做 leave-one-intervention-family-out 检查;
- 最好在第二数据域计算 H2,在第一数据域训练/评 Tier-2,或反过来,排除同数据 scorer 共性。

若只有 A0/A1,推荐措辞是“prospectively transfers”或“is aligned with downstream
action following”,不是“universally predicts”。

## 7. intermediate α / schedule

这里必须区分两种 sampling:

- **LAM posterior sampling:**`z=mu+s*sigma*epsilon`,A0 确实未见过;
- **ACWM/video sampling:**diffusion/生成器对视频噪声的采样,即使 condition 是固定 `mu`
  也会发生。

后者不要求 LAM 在 `z` 上用 α>0。只要 Tier-2 输入是检索/策略选择的 raw `mu`,A0 的
posterior-sampling OOD 不是阻塞项。因此:

1. A、C、B 的 deterministic-`mu` Tier-2 都直接用 A0,无需先跑 0.25;
2. 若实际系统要从 `q(z|o_t,o_{t+1})` 抽样、从 stochastic latent policy 采样,或做
   prior exploration,才需要 α>0 robustness;
3. 需要折中时先跑固定 α=0.25,不要先上 schedule。schedule 同时引入噪声大小与训练时序,
   归因更差;
4. α=0.25 也可作为 utilization->Tier-2 曲线的第三点,但应在 A0/A1 外部 transfer 已成立
   后做,不是进入 Tier-2 的前置门。

## 8. 最小可行排期

### 第 0 天:零训练硬门

- paired-latent audit;
- 四格 cross-decoding;
- 冻结 Tier-2 manifest、donors、metrics 和 practical margins。

### 第 1--2 天:Tier-1.5

用原 LAM decoder 做 `K={4,8,16}` own-latent 与 target-transfer rollout,比较 A0/A1 两
seed。它能快速暴露递归漂移、静止复制和符号失效,也能复用后续 tracking scorer。但论文
中明确标为 within-LAM functional validation。

### 第 2--5 天:最小真正 Tier-2

训练独立轻量 conditional predictor:

- A0 latent、A1 latent、GT18-action positive control;
- zero/shuffle/antiparallel 作为 inference controls,通常无需各训一模型;
- 两个 matched downstream seeds;
- K-step rollout + canonical-FDCE@K + signed displacement + fidelity。

若小模型也出现 `A0 >> A1` 且过 guardrails,已经拿到 workshop 很强、main 有意义的
downstream bridge。如果 `A0 == A1`,结合 cross-decoding 判断:

- paired `mu` 相同且 D_A0-only:α=0 是本 decoder 的修复,应收窄论文;
- paired `mu` 不同但外部相同:新 decoder 可重新学习两种 latent,原 H2 不是跨 decoder
  predictor;
- 所有条件包括 GT18 都无 action following:Tier-2 architecture/scorer 未过正对照。

### 2B go/no-go

只在以下条件同时满足时进入 B:

1. 官方 runtime doctor/dry-run 在目标机器 1 天内通过;
2. 40D conditioning adapter 的形状与 checkpoint lineage 明确;
3. rollout generation、SAM3/CoWTracker 和 canonical scorer 小样本端到端跑通;
4. 有预算为 A0/A1 各训 matched Stage-2,而不是只替换 frozen ACWM 输入;
5. 轻量 Tier-2 已显示非零效应,足以证明高成本放大实验值得做。

若任一不满足,停止 2B 工程,按 lightweight Tier-2 或 workshop 范围收束。

## 最终回答

α=0 的当前价值非常明确:**它修复了 LAM 自身 action-conditioning edge,且该修复在两个
seed、幅度/符号干预和 fidelity 上一致。**但它是否改善 ACWM,取决于 α 是否改变了可被
新 consumer 学习的逐样本 latent structure,以及新 ACWM 是否会使用该 condition;汇总 posterior
统计相同不能回答这两件事。

所以最合理的研究路径不是“立刻把 α 搬进 2B ACWM”,也不是把原 LAM decoder rollout
包装成完整 Tier-2,而是:

> 先用 cross-decoding 定位修复发生在哪里,再用一个独立轻量 decoder 做跨-consumer
> causal transfer;只有 transfer 成立且 2B runtime 可行,才升级到 matched DreamDojo
> Stage-2。

这条路径用最少算力同时保护了因果解释和投稿价值。

---

# 第五轮后的 Tier-2 状态修订（2026-07-21，覆盖上文未执行计划）

本节根据已经完成的 Phase 0（0.1 + 0.2）、Phase 0.3 freeze、Phase 1 Tier-1.5、
Phase 2 Route-C GT-18D 正对照，以及 Route-C `zmu A0/A1` 结果修订任务。上文仍保留为
决策记录；若与本节冲突，以本节为准。

## 1. 已完成结果的正确解释

### Phase 0 / 1 已形成稳定证据

- 两个 LAM seeds 中，A0/A1 paired `mu` 极其接近：cosine/CKA/Procrustes `R^2` 约为
  `0.999 / 0.999 / 0.998` 量级；
- 四格 cross-decode 的变化由 decoder 行主导，encoder 列内差异约小于 1%，而 A0/A1
  decoder 行差达到约 11--21 倍；
- Tier-1.5 在 K={4,8,16} 和 sign intervention 上一致支持 A0 原 decoder 的可控性优势。

因此可确认：在当前 LAM 中，alpha 修复主要发生在原 decoder 的 conditional-use path，
不是 encoder `mu` 的大幅改写。

### 当前 Route-C 是通过 dev gate 的 smoke study，不是最终 confirmatory test

GT-18D consumer 的 target 相对 shuffle/anti 在 direction 指标上更好，说明轻量 consumer
与 signed-direction scorer 至少具有部分正对照灵敏度。但 GT18 target 的 endpoint error 没有
优于 zero/static（h=1 约为 `2.695 vs 2.591`，h=4/8 也更差），因此不能把 GT18 写成
“canonical scorer 全面过门”；endpoint/persistence 部分仍需校准。A0、A1 `zmu` consumer 都
强烈依赖自己的 condition：例如 horizon 1 的 target direction alignment 分别约为 `0.640`、
`0.605`，明显高于 shuffle 和 anti；在 horizon 4/8，两者仍有明显 condition effect。

A0 与 A1 没有稳定的性能排序：A0 在 h=1 略好，A1 在 h=4/8 的 endpoint/方向部分指标
略好。最合理的当前结论是：

> 一个从头训练的独立 consumer 能学习并使用 A1 `mu`；alpha=0 对原 LAM decoder 的修复
> 没有通过近乎相同的 `mu` 自动迁移给新 consumer。

这否定的是“alpha 已经产生更好的 exported encoder interface”，不是“A1 latent 对任何下游
都不可用”。在补齐 seeds 与 Eval-B 前，这一结论必须标为 dev/preliminary。

## 2. Eval-B 前必须修复的 protocol 问题

1. **冻结并复用 train normalization。** `train_consumer.py` 训练时使用 train `mu/std`，但
   当前 `eval_consumer.py` 会从 evaluation manifest 重新估计 zmu normalization。这违反训练
   conditioning contract，并泄漏测试分布统计。checkpoint/artifact 必须保存 train `mu/std`，
   Eval-A/B 只能读取它，绝不能在测试集重算。
2. **保存完整 lineage。** 每个 checkpoint 保存 conditioning mode、alpha/LAM seed、consumer
   seed、train clip IDs 或 manifest hash、normalization、模型/optimizer config 和 git commit。
3. **恢复预注册 donor 语义。** shuffle 必须是 within-task donor；antiparallel 必须使用预计算的
   in-distribution、magnitude-matched GT18 donor，再取对应 clip 的真实 zmu。`-standardized_z`
   只保留为 OOD sign-flip diagnostic，不能叫 physical antiparallel control。
4. **逐 clip 输出。** JSON 必须保存 clip/task/episode ID、每 horizon endpoint/direction/fidelity/
   motion 指标和 donor ID，才能做 paired task -> episode/clip hierarchical bootstrap；aggregate
   mean 不能支持 confirmatory CI。
5. **修复 eval summary。** `dev_gate` 返回的是 `condition -> horizon -> metrics` 的嵌套结构，
   当前打印逻辑却直接从 condition 层读取 `endpoint_err`，应按 horizon 遍历。
6. **重新校准 positive-control gate。** GT18 的 signed direction 可以作为已通过的 primary
   construct；endpoint error 必须加入 zero/persistence comparator，并查明静态预测为何更优。
   在解决前，不用 endpoint aggregate 宣称完整 action following。
7. **保持 Eval-B 封存。** 上述修复必须先在 dev/Eval-A 重跑并通过 artifact audit；不能为了
   调代码预览 Eval-B。

## 3. 修订后的最小完成矩阵

预注册要求 A0/A1 两个 LAM seeds 且每个至少两个 matched downstream seeds。因此正式矩阵为：

| LAM arm | LAM seed | consumer seeds | 当前状态 |
|---|---:|---|---|
| A0 | 42 | 0, 1 | seed 0 已有旧协议 dev；修复后重跑 0，并补 1 |
| A1 | 42 | 0, 1 | seed 0 已有旧协议 dev；修复后重跑 0，并补 1 |
| A0 | 1 | 0, 1 | 待训练 |
| A1 | 1 | 0, 1 | 待训练 |

即总计 8 个轻量 zmu consumers；现有两个 checkpoint 可用于 debugging，但在 normalization/
artifact contract 修复后必须用冻结 train stats 重新评估。GT-18D 至少保留当前 seed 0 正对照；
若预算允许补 seed 1，用于证明 positive-control 稳定性，但它不替代 8 个 A0/A1 matched runs。

## 4. 冻结的统计判断

- primary comparison 必须是同一 clip/donor 上 A0 vs A1 的 paired difference；
- 先报告 target 相对 within-task shuffle、real antiparallel 的 condition-use effect，再比较
  A0/A1；未通过 condition-use gate 的 consumer 不进入 alpha transfer 判断；
- 使用预注册的 `10%` practical improvement 作为 superiority/futility threshold；
- 若 A0 的 paired hierarchical CI 下界超过 `+10%`，才称 alpha-specific transfer；
- 若 CI 上界低于 `+10%`，称“排除了达到预注册实用阈值的 A0 benefit”，不要事后发明一个
  equivalence margin；
- 若 CI 跨过 `10%`，结论是 inconclusive，不把 point estimate 接近写成 equivalence；
- fidelity、motion magnitude、first-frame consistency 任一 guardrail 失败，都不能把更大的
  direction/endpoint movement 写成 controllability improvement。

## 5. 新的 go/no-go

### 必须完成

1. 修复 normalization、donor、per-clip artifact 和 summary bug；
2. 在 dev/Eval-A 对现有两个 arms 重跑 protocol audit；
3. 补齐 2 alpha x 2 LAM seeds x 2 consumer seeds；
4. 冻结 checkpoint hashes、donor manifest、scorer 和统计代码；
5. 只进行一次 Eval-B unseal，并对完整 matched matrix 使用同一 clip/donor 集合评估。

### 明确停止

- **停止 2B Phase 2B。** 当前 dev 已显示新 consumer 能从 A0/A1 `mu` 学到 condition use，
  且没有 alpha-specific encoder signal；高成本 2B matched training 的预期信息增益不足。
- **不把当前 dev 结果写成 ACWM downstream improvement 或正式 null。** 它是强方向性证据，
  但尚缺第二 LAM seed、第二 consumer seed、正确 donor 和 Eval-B。
- **不新增 alpha sweep、schedule 或 `L_dir` 组合。** 这些不能解决 exported-interface null，且
  会扩大多重比较。
- Route-C confirmatory closeout 后，若继续研究，只执行 roadmap 中的 Route A 跨架构最小包；
  label-free 2x2 和完整 2B ACWM 均不进入当前资源计划。

## 6. 最终 Tier-2 定位

当前最可能的 Tier-2 结论不是“alpha improves downstream”，而是更精确也更有机制价值的
negative boundary：

> 原 LAM decoder 可以在 encoder 已含 action information 时学会忽略该 condition；alpha=0
> 能修复这个特定 decoder 的使用路径，但由于 A0/A1 `mu` 近乎不变，一个新 consumer 可以
> 同样重新学会使用二者，因此该修复不会自动成为可迁移的 latent-action improvement。

完成上述 confirmatory matrix 后，这个结论可以作为 cross-decoding 因果定位的独立 consumer
验证；在此之前只按 preliminary dev evidence 表述。
