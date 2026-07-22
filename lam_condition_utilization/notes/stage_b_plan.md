# 阶段 B 执行计划(2026-07-20 定稿,采纳顾问第二轮回复)

研究问题已从"方向是否存在"扩展为 **availability × utilization**:
弱可读的方向信息**能否被放大**(availability),**以及能否被 decoder 因果使用**(utilization)。

---

## B0 训练前的零成本修正 —— ✅ 全部完成

| 项 | 状态 | 产物 |
|---|---|---|
| ① 对照臂改为**匹配非方向排斥** | ✅ | kind=1 中位 cos **+0.028**,82.9% 在 \|cos\|<0.2,**0% 反平行残留**(v1 有 52%,是伪对照) |
| ② confirmatory probe 统计重做 | ✅ | `code/dirprobe.py`,六处口径全修;合成数据四档信号单调性验证通过 |
| ③ 冻结评测 manifest | ✅ | `data/eval_manifest.json`:390/2942 ep(13.3%),107 task 全覆盖,**68 个可逆 task 全覆盖** |

**probe 验证结果**(sig=0/0.5/1.5/3.0,20 task,199 次置换):
P = −0.006 → +0.053 → +0.096 → +0.122,ΔAUC 同步单调,null 正确不显著(p=0.715)。
**配对对比更灵敏**:sig=0.5 时单模型 CI 含 0,而 ΔP CI 排除 0
⇒ 印证 H1 的主对比**必须是配对的方法 vs 对照**,不能是 vs 零。

**顺带修掉的污染点**:配对索引原从完整 train split 挖,含 390 条 held-out episode。
已重挖,交集验证为 **0**(236,784 反平行 / 106,434 近正交 / 138,100 同向)。

---

## B1 四臂 successive-halving(各 1k 步;S 只 300 步)

同一预训练起点(`LAM_400k.ckpt`)、**修正后的 LR 调度**、**排除评测 manifest**、
trunk 统一为 Ours-A(KL + L_zero + action head)。

| arm | config | 目的 |
|---|---|---|
| `B` | `ft_l40_sb_B.yaml` | 修正调度下的基线(旧 run 因 V 形调度不可作对照) |
| `R` | `ft_l40_sb_R.yaml` | 匹配非方向排斥(近正交对)—— **H1 的主对照** |
| `D` | `ft_l40_sb_D.yaml` | GT-oracle 反平行 L_dir —— 方法臂 |
| `U` | `ft_l40_sb_U.yaml` | 仅 L_use(运动能量归一化 hinge)—— utilization 是否是主瓶颈 |
| `S` | `ft_l40_sb_S.yaml` | 同向 adverse 哨兵(300 步);若它也改善 probe ⇒ loss 只是排斥项,立即停 |

预算:四主臂 4×1k = 0.8 个完整 run,加 S 约 0.86。

### 1k 决策树(顾问原案)
- `D` 不胜 `R` ⇒ 停方向故事,先修 pairing/loss,不进 LF;
- `D` 胜 `R` 且 H2 改善 ⇒ `B/R/D` 继续到 5k,`U` 留在分析章;
- 仅 `U` 改善 H2/下游 ⇒ 主问题是 utilization,优先 `B/U`,`L_dir` 降级;
- `D` 只改 H1、`U` 改 usage ⇒ 启动 `R+U` 与 `D+U`,方向结论取二者之差;
- 四臂均无外部改善 ⇒ 不调 lambda 硬撑,先重做 decoder bridge 或指标。

## 预注册的三层成功标准

- **H1 representation amplification**:`ΔP = P_D − P_R`,配对 hierarchical bootstrap
  (先抽 task 再抽 episode),要求 95% CI > 0 **且**超过实用边界 `δ_P`
  (候选 0.05 nats,须在看结果前用修正调度 baseline 的重复方差冻结)。
- **H2 direction-specific decoder utilization(Stage B 的主 Tier-1 指标)**:
  `C_m = E[(MSE_opp − MSE_same)/(motion_energy+ε)]`,`ΔC = C_D − C_R`;
  同时要求 `D` 相对 `B` 也改善(排除"对照变坏导致相对胜出"),
  且 `swap_context_cost` 与重建保真不超过预注册退化容差。
- **H3 下游可控性**:同协议 ACWM action-following / FDCE + held-out task。
  H1 改善而 H2/H3 不动 ⇒ "几何被塑形但没成为可控条件",**不算成功**。

总成功条件:H2 上 `D > R` 成立 + H3 相对匹配 baseline 成立且保真守门通过;
H1 为支持机制的 secondary;**同向 adverse 臂不得复制 H1/H2 的改善**。

---

## 实现记录(本轮新增)

- `model.decode_with(patches, z_rep, H, W)` —— 从 `forward` 抽出的独立解码入口,
  使干预可替换 z 而无需重编码(L_use 与后续 decoder 侧评测共用)。
  `forward` 默认仍 `del patches` 保持峰值内存;仅 `keep_patches=True` 时保留。
- `train.latent_usage_loss` —— 运动能量归一化的 usage hinge。
  **不用 SAM3 mask 归一化**:mask 覆盖率仅 ~20%,会让 loss 只在 1/5 数据上生效。
  `err_real` 取训练时的**采样** latent(解码器实际收到的就是它),
  这样 hinge 会同时压后验方差 + 逼解码器使用 z,正对 lf 塌缩的机制。
- 烟测观察(step 5,预训练权重):`gap_zero = −0.47`、`gap_shuf = +0.013`
  ⇒ **此时零 latent 解码优于采样 latent,打乱 latent 几乎无影响**
  (KL 仅 4.75/40 维,后验≈先验,采样噪声大)。这正是 U 臂要打的靶子。

## ⚠️ 诚实性边界(写论文时必须声明)

1. `sb_D` 用 **GT 位姿挖的配对(oracle)**,**不是** target-action-annotation-free。
   它只回答"方向配对机制的上限"。主线需换成光流/关键点代理。
2. 审计里 `own/cam` 逐对一致率 87.8% **不构成 proxy 可行性证明** ——
   `cam` 仍由 GT 手部轨迹 + GT 相机变换算出,两者都是 GT 口径。真正的 proxy 验证未做。
3. `L_use` **不能援引 CD-LAM 背书**:其 Stage-1 配方 `lambda_use: 0.0`,
   且 masked reconstruction 开启时 trainer 显式跳过;论文公式里也没有它。
   它是我们自己的候选机制,已知失败模式是"抬高 corrupted error"而非"改善 real"。
4. 阶段 A 在 train 池上的 probe 数字全部是 **exploratory**
   (encoder 见过那些 episode);confirmatory 结论必须用冻结 manifest 重跑。

---

## B1 结果与判定(2026-07-21,confirmatory)

在冻结 manifest(390 ep,encoder 从未训练过;57 个可逆 task 过 probe)上评估。

### H1 —— representation amplification(条件方向 probe)
单臂 P(null-centered gain,999 置换):B **+0.022** / D **+0.018** / R +0.010 / U +0.015 / S +0.009。
**基线 B 反而最高**;方法 D 夹在中间。配对(hierarchical bootstrap,先 task 后 episode):

| 配对 | ΔP | 95% CI | 判定 |
|---|---|---|---|
| **D − R(主对比)** | +0.008 | [−0.007, +0.034] | **含 0,不显著** |
| D − B(守门) | −0.005 | [−0.045, +0.040] | 含 0(D 未胜基线) |
| U − B | −0.007 | [−0.030, +0.002] | 含 0 |
| S − R(保险丝) | −0.001 | [−0.022, +0.024] | 正常(哨兵未复制) |

⇒ **L_dir 相对匹配非方向对照没有可检测的方向信息放大。**
⚠️ 5-task 版曾误报 ΔP(D−R)=+0.103"显著"——那是 GroupKFold 按 episode 留出、
每 verb 仅 ~2 ep 导致每折 test 单 verb 被跳、57 task 塌成 5 的假象。
已改 StratifiedGroupKFold(按较小 verb 的 ep 数定 n_splits)修复,57 task 全进。
旧假象存档 `results/stage_b_eval_5task_STALE.json`。

### H2 —— direction-specific decoder utilization(主 Tier-1 指标)
`C = E[(MSE_opp − MSE_same)/(motion_energy+ε)]`,同 manifest 池,840 anchor / 55 task。
单臂 C:B +0.0001 / R −0.0002 / **D −0.0000** / U(待) / S(待)。
**全部 ≈ 0** —— 把 z_a 换成相反 vs 同向动作,解码出的未来几乎无差别;
解码器根本不响应 z_a 的方向。D 不优于 R。
**ΔC(D−R)=+0.0005 CI=[−0.0001,+0.0012] 含 0**;五个 ΔC 配对全 n.s.;C 全在 ±0.001 噪声带。

### 决策树判定:**停止方向故事,不进 proxy 管线**
顾问决策树第一条:"D 不胜 R ⇒ 停方向故事,先修 pairing/loss,不进 LF"。
- H1:ΔP(D−R) CI 含 0,且 D 未胜基线;
- H2:C 全场 ≈ 0,decoder 对 z_a 方向零响应,D 不优于 R;
- 两个主指标一致指向负面。

**这不是"方向 loss 永远没用"的终判**,而是:
**当前配方(余弦 margin + GT-oracle 配对 + 1k 步)在 availability 和 utilization 上
都没有相对匹配对照的可检测收益。** 边界:
1. Tier-0 机制确实生效(D 的 za_cos +0.16→−0.30),几何被改变,但未转化为外部收益;
2. 1k 步;5k 是否不同未测(但机制在 1k 已饱和,active_frac 0.34,继续训收益存疑);
3. GT-oracle 是上界配置,proxy 只会更弱 ⇒ **不投入光流/关键点代理管线**(止损)。

### 转向:utilization 线索(⚠️ 本节的推断已被顾问第三轮回复修正,以下节为准)
`sb_U`(L_use)单臂 ΔAUC=+0.0115 全场最高、置换 p=0.023,但 **U−B 配对 H1/H2 均不显著**;
H2 的 C≈0 对全臂成立。**当日初读**为"问题在解码器不用 z_a,不在 latent 不可分",
遂考虑把重心转向 decoder 侧 utilization(L_use 单独臂 + 更强 usage 约束)。
**这一步推断被顾问驳回**——C≈0 在正对照通过前**不能**区分"decoder 不用方向" vs
"assay 对方向不敏感";且 U 臂损伤保真、并未赢 paired 对比,不是 utilization 的正结果。
详见下节。

---

## 阶段 B 顾问第三轮判定与转向(2026-07-21 定稿)

第三轮回复全文见 `consult_story_b_prompt.md` 末尾(顾问第三轮回复)。核心是把
"是否转向 utilization 线"这个二选一**替换成"先把 causal-use 测量闭环做实"的排期**。
以下为消化后的决策状态。

### 1. 已确证的结论
- **停 L_dir / proxy 管线 = 正确**。这是 **futility 判定**,不只是"不显著":
  ΔP(D−R) 点估 +0.008、CI 含 0,**上界 +0.034 低于冻结的 practical margin 0.05**。
  Tier-0 机制确实生效(D 的 za_cos +0.16→−0.30)⇒ "loss 实现了局部几何目标,
  但该目标没转化为所需功能"。**不跑 D/U 到 5k**,保留 checkpoint + 否定结果。
  (措辞收窄:GT-oracle 是最干净的机制检验、足以作停止门,但**不是数学上界**——
  有噪声的 proxy 偶尔会有不同覆盖/正则效应。论文写"最干净的 oracle-pairing 测试失败,
  故不投资 proxy 构造",不写"已证明任何 proxy 都不可能有效"。)

### 2. 不能过度推断的结论
- **不能**由全臂 C≈0 直接宣布根因是 posterior noise / decoder utilization。C≈0 目前
  只证明"经这套 donor-swap assay,没检测到方向特异因果使用";**在正对照通过前,
  "decoder 不用方向" 与 "assay 对方向不敏感" 无法区分**。
- **准确的转向表述**(顾问原话,写论文用):
  > The direction-specific intervention failed despite changing latent geometry.
  > We therefore pivot from designing another directional regularizer to validating
  > and diagnosing the latent-to-decoder causal channel.
  **不要**提前写成 "Posterior noise causes the decoder to ignore direction."(缺实验链条。)

### 3. U 臂的正确定性:应淘汰的实现,不是 utilization 正结果
- 修正一处记录错误:**单臂 P 是 B +0.022 > D +0.018 > U +0.015**,U 并非"H1 单臂最高";
  它只是 ΔAUC 等附属统计较高。主因果对比 U−B 在 H1/H2 都不显著。
- 训练日志给出更硬的否决:step 1000 时 **U 的 val MSE 0.0018 / PSNR 27.46 dB
  vs B 的 0.0010 / 29.92 dB(−2.46 dB)**。usage hinge 只是抬高了它直接优化的
  gap_zero(−0.57→+0.31),可通过"抬高 corrupted error 制造任意 latent 敏感性"满足
  margin,**不必学到方向控制,且损伤保真**。⇒ 研究问题可转向"如何测量并建立
  utilization",但**不能直接转成"继续调当前 L_use",也不能把 free-bits+L_use 定为新主方法**。

### 4. ✅ 代码核实:z_mu 诊断(#3b)已内建于 H2,重跑区分不出任何东西
逐行核对 `code/h2_utilization.py`:same/opp 两路的 latent 都由 `_encode_mu_var`
取 **z_mu**(不采样)构造,再经 `_decode_from_mu` **确定性解码**(benchmark_lam.py
L246–259,全程无 ε)。⇒ **C≈0 不是评测期采样噪声的产物**;"零成本 z_mu 版 H2"
本就是当前口径,重跑无信息。
- 它能排除的:evaluation-time ε 恰好洗掉了 C。
- 它**不能**排除:训练期喂给 decoder 的采样 latent 教会 decoder 下调 latent 分支
  (这是**下一个待检验假设**,不是已证事实)。真正待做的零成本诊断见 §5 的 2、4。

### 5. 立即排期(取代"二选一",顾问定序)
1. **正式停 D/proxy**,保留 checkpoint 与否定结果,不跑 D/U 到 5k。
2. **用现有 checkpoint 重算 H2 分解**(零训练):每 anchor 输出 `E_own / E_same /
   E_opp / E_zero / E_shuffle`、直接输出响应量 `R_out=‖D(o,z_opp)−D(o,z_same)‖`、
   同量纲归一化(以 `MSE(o_t,o_{t+1})` 作分母)、前景/背景分开、GT-action donor 距离
   `d(a_anchor,a_same) < d(a_anchor,a_opp)` 校验。判读矩阵见顾问回复 §1 表。
3. **建 H2 正对照(硬门,非可选附录)**:在同数据/同 anchor·donor/同 scorer 上评一个
   GT-action-conditioned WM,或给当前 trunk 加小 action adapter 训到可控,作 assay
   positive control;并加 random-donor negative control。正对照 C≈0 ⇒ 先修 assay,
   不能据此下任何模型结论。**IDM oracle 不自动是正对照**(它能编码动作 ≠ 被测 decoder 会用它)。
4. **posterior signal/noise 分解**:逐维 `signal_j=Var(mu_j)`、`noise_j=E[exp(logvar_j)]`、
   `SNR_j`;same/opp 类中心距离 相对 posterior noise 的比率(不只看 za_cos)。
   ⚠️ **总 KL ≈19–21 nats ≠ "19/40 active dims"**——active dims 须另算。
5. **当且仅当**证据支持训练期采样噪声,跑最小因果消融 `z_dec = mu + α·σ·ε`,
   `α∈{0,0.25,1}`,先比 α=0 vs 现有 α=1 的 300/1k matched arms;**先于 free-bits**
   (free-bits 只撤 KL 梯度、不直接降后验方差、不保证 decoder 用 latent)。若不支持,
   优先查 decoder bridge / conditioning 架构。
6. 只有当某个干预**同时**改善方向特异 use **且**过 fidelity 守门,才恢复 Tier-2 与
   方法论文路线;否则按 diagnosis + methodology 的 workshop 稿收束。

### 6. 论文定位(顾问判)
- **CoRL/ICLR workshop:可成立**——诚实的 negative-result + methodology 故事合适。
  最低条件:补 H2 正对照、修量纲/配对审计、公开 manifest 与 stale-result 处理流程;
  是否找到修复**不是**成立的必要条件。
- **CoRL/ICLR main:当前证据不够**。短板不是"没有新 loss",而是外部效度仍局限于
  单一 DreamDojo/CD-LAM 生态 + 单一数据分布,且核心 H2 未过正对照。补强方向:
  ≥2 LAM 架构/目标 × ≥2 数据域/embodiment、正负对照证明每个指标的灵敏度/特异性、
  availability/utilization 指标能预测 Tier-2 action-following、全比较用冻结 manifest +
  层级统计 + fidelity 守门,最好有一个可归因的干预恢复 causal use。

### 7. 论文措辞守门(顾问)
- 写 "public CD-LAM checkpoint behavior under our protocol",不写 "CD-LAM 方法失败"
  (公开权重 metadata 是 step 300,≠ 论文完整 1000 步 Stage-1)。
- 写 "this L_dir recipe at the 1k stopping point failed",不推广成 "direction losses
  cannot work"。
- H2 正对照通过前,写 "no detected direction-specific use under our assay",不写
  "the decoder is direction-invariant"。

---

## 阶段 B 诊断实测:H2 分解 + posterior SNR(2026-07-21)

顾问排期第 2、4 步(零训练)。代码 `code/h2_decompose.py` + `code/eval_h2_decompose.py`;
产物 `results/stage_b_h2_decompose.{json,log}`、`_perdim.npz`、`_persample.npz`、
`stage_b_h2_donor_gate.json`。冻结 manifest、z_mu 确定性解码、840 anchor/55 task
(h2_pool 1500)、snr_pool 3000。前景=运动 mask(top-10% 运动像素,规避 SAM3 覆盖问题)。

### 各臂关键量(E_* = 对 GT 的 per-pixel MSE 均值)
| arm | E_own | E_same | E_opp | E_zero | usage_zero | R_out | c_persist |
|---|---|---|---|---|---|---|---|
| B | 0.00053 | 0.00063 | 0.00063 | 0.00073 | +0.53 | 0.0102 | −0.001 |
| R | 0.00054 | 0.00063 | 0.00063 | 0.00070 | +0.37 | 0.0104 | −0.022 |
| D | 0.00053 | 0.00063 | 0.00063 | 0.00070 | +0.39 | 0.0108 | −0.013 |
| U | **0.00165** | 0.00174 | 0.00174 | **0.00300** | **+4.56** | 0.0216 | −0.092 |
| S | 0.00067 | 0.00071 | 0.00072 | 0.00084 | +0.47 | 0.0117 | +0.015 |

### 落位:顾问 §1 判读矩阵的 **"R_out>0 但 C≈0 → decoder 有响应,但方向错/donor-GT 不对齐"**,
### **不是** "latent bypass" 行。三条支撑:

1. **💥 donor 语义前提被证伪 —— C≈0 的可解读性被消解(本轮最重要发现)**:
   - `donor_frac(d_same<d_opp)=`**`0.488`**(全臂同一 donor 池;18D 归一化动作距离),
     中位 d_same 2.898 ≈ d_opp 2.828。即"同 verb donor 比反 verb donor 更接近 anchor
     动作"只有 ~49%,基本抛硬币。⇒ C 的分子 (E_opp−E_same) **本就没有理由为正**,
     即使 decoder 完全方向敏感。**旧 H2 的 C≈0 是 donor 构造伪影,不是 utilization
     deficit 的证据**(顾问风险 #1 实测坐实)。
   - **修复后仍无信号**:仅保留 d_opp>d_same 的 410 anchor(+0.5 margin 的 247 个),
     方向对比 c 依旧碎:B +0.012 / **D −0.007~−0.011(方法臂仍负)** / R −0.043。
     **gate 没救出任何 D>R 或 D>B ⇒ L_dir futility 在修复 donor 混淆后依然成立。**
   - decoder 误差**弱**跟随注入动作距离:`corr(E_opp−E_same, d_opp−d_same)=`
     **`+0.15~0.17`**(B/R/D)。⇒ decoder 用 latent 的动作**幅度**(generic),
     **但不做 verb-方向区分**。(U/S 为负 = 误差尺度失真,见第 3 点。)

2. **decoder 不 bypass latent**:E_own 恒为最低(B 0.00053 < E_same/opp 0.00063 <
   E_zero 0.00073);usage_zero 强正;R_out>0(前景更大 0.018>背景)。⇒ "decoder 忽略
   z_a" 的强 bypass 行**不成立**;准确说法="decoder 泛化地用 z_a,但 swap assay 因
   donor 未按方向排序而测不出方向对比"。E_same≈E_opp 逐位相等(raw 差 ~3e-6,比
   E_own→E_same 的 ~1e-4 gap 小两个数量级)。

3. **sb_U 确认应淘汰**:E_own 暴涨到 0.00165(≈B 3×,对应 −2.46 dB);usage_zero=+4.56
   (hinge 造了巨大 generic zero-gap)但 direction 仍 null(raw +1.35e-6);usage_shuf=
   −1.90、gate 后 c=−2.9(误差尺度失真)。⇒ hinge 只放大 corrupted error,没造出方向
   控制、且毁保真。与顾问一致。

### posterior SNR:collapse-lite 已量化(前提成立),但非因果证明
- **noise=exp(logvar)≈1.0 逐维**(全 32 动作维 min 0.98/max 1.02):后验方差被 β_a=3e-3
  钉在先验 N(0,1)。
- **snr>1 的动作维 = 0**:无任何一维 signal(Var(mu))超过其 noise(~1.0);
  snr_act_mean ~0.54。⇒ 训练时 z=mu+ε(σ≈1.0),注入噪声 std(1.0)>mu 展布 std(√0.54≈0.73)
  逐维成立 ⇒ **"训练采样噪声可能教 decoder 下调 latent 分支"的 SNR 前提被满足**。
- **"total KL ≠ active dims" 实证**:KL_act_total ~9.3 nats 摊在 32 维,per-dim 中位
  0.276、>0.3 仅 14 维、>0.5 仅 1 维,**eff_rank ≈ 10/32**。结构上活跃 ~10–14 维,
  不是松阈值(kl>0.01)数出的 40。
- ⚠️ **不可过度读噪声故事**:same/opp 类中心分离 `sep/noise ≈ 1.9σ`(超噪声)。
  方向信息在**类均值**层面**高于**噪声地板,decoder 却不响应它 ⇒ 更接近
  **"弱可用但不被使用"**,而非"被噪声完全淹没"。噪声假设仍需 α=0 vs 1 消融证因果。

### 对下一步的影响(更新排期)
1. **H2 正对照(硬门)更必要**:C≈0 已证含 donor 伪影;正对照标定 assay 灵敏度前,
   C 绝对值不可作模型结论。**donor gate(只留 d_opp>d_same+margin)应内置进未来 H2**。
2. **corr≈0.16 是内部 mini-sensitivity 迹象**(assay 能测动作幅度、不能测方向),
   但不替代正对照。
3. **α=0 vs 1 消融**是区分"训练噪声掩盖" vs "decoder 不用方向"的关键最小实验:
   SNR 前提成立(逐维<1)但被 class-sep>noise 部分反驳,只有消融能定因果。
   顺序仍在正对照之后(先证 assay 能测出已知强可控,再谈根因)。

---

## 阶段 B H2 正对照(硬门 ✅ 通过,2026-07-21)

顾问排期第 3 步(转 utilization 论文前的**硬门**)。代码 `code/h2_poscontrol.py` +
`code/eval_h2_poscontrol.py`;产物 `results/stage_b_h2_poscontrol.{json,log}`。
正对照 decoder = **训练无关的 frame-delta 合成 oracle**
`D_pos(o_t^a, donor)=o_t^a+(o1_donor−o0_donor)`(按构造强可控;o_t^a 在 opp−same 差里
精确抵消 ⇒ C 纯 delta 空间,场景抵消)。三种 donor 方案(同 840 anchor/同 scorer):
`verb`(旧构造)/ `dist`(按 18D 动作距离强排序:same=近四分位、opp=远四分位)/
`random`(负对照)。robust 统计 **C_pooled = Σ(E_opp−E_same)/ΣP**(ratio-of-means,
避开 per-sample 除法在低运动样本上爆炸——修顾问风险 #3;mean-of-ratios 不可信)。

### donor 排序(证明三方案构造有效)
| 方案 | d_same | d_opp | frac(d_same<d_opp) |
|---|---|---|---|
| verb | 2.371 | 2.350 | **0.488**(未排序 = 旧 bug) |
| dist | 1.178 | 4.201 | **1.000**(强排序) |
| random | 2.309 | 2.468 | 0.550(近对称) |

### ✅ 硬门通过:scorer 灵敏度已证
| decoder × 方案 | C_pooled | median | 读法 |
|---|---|---|---|
| **oracle × dist** | **+1.149** | +0.985 | 强可控 decoder + 排序 donor ⇒ C 强正 |
| oracle × verb | +0.035 | +0.023 | verb donor 打不出信号(**即使完美 decoder**) |
| oracle × random | +0.065 | +0.044 | 负对照干净 |

⇒ 三条同时成立:scorer 对已知强可控 decoder **灵敏**(dist +1.15)、**不被随机 donor
愚弄**(+0.06)、且**旧 verb 构造对完美 decoder 也只给 +0.03** ⇒ 旧 H2 的 C≈0
**被独立证明是 donor 构造伪影**,不是 utilization 的证据。

### LAM 全臂:真实 utilization deficit(D 无优势)
| arm × dist | C_pooled | median | 相对 oracle ceiling |
|---|---|---|---|
| sb_B | +0.039 | −0.068 | 3.4% |
| sb_R | +0.032 | −0.088 | 2.8% |
| sb_D | +0.046 | −0.093 | 4.0% |

- 在 donor 已修(dist,frac=1.0)、scorer 灵敏度已证的前提下,**全臂 C 仅为 oracle 上限
  的 ~3–4%,median 甚至微负** ⇒ **decoder 对注入的动作(距离)几乎不响应 = 真实
  utilization deficit**,不是 assay 不敏感。
- **D(+0.046)不胜 B(+0.039)/R(+0.032)**(pooled 在噪声内、median 全 ~−0.08)
  ⇒ L_dir 在灵敏度已证、donor 已修的 assay 上**仍无 utilization 优势**。futility 定论。

### 结论:顾问硬门的确定性答复
把顾问的开放问题 **"C≈0 是 deficit 还是 assay 不敏感"确定性关闭**:
**assay 灵敏(oracle 证)+ LAM 真实 deficit(~3–4% ceiling)+ L_dir 无优势**。
⇒ 论文口径可从 "no detected direction-specific use under our assay" **加强**为
"under a positive-control-validated assay, LAM decoders utilize the injected action
(distance) condition at only ~3–4% of an oracle ceiling, and no training objective
improves this."(仍不写 "direction-invariant"——见诚实性边界 1。)

### 诚实性边界
1. **dist 按动作距离幅度排序(非方向/反平行符号)** ⇒ 严格测的是**动作-幅度 utilization**
   (比 direction-specific **更容易**);连它都近零 ⇒ 更难的方向特异 utilization
   a fortiori 也缺失。**反平行 donor 方案**(用已挖的 236,784 反平行对)可作后续加固,
   非必需。
2. oracle 是**像素空间 decoder(旁路 latent)**,验证的是 **scorer + donor 构造**的灵敏度
   (正对照本职),**不是 LAM 架构本身**。若要 LAM 架构内正对照,需训练 GT-action bridge
   (顾问 option b,成本更高、当前非必需——scorer 已证灵敏)。

### 对排期的影响
- **H2 正对照硬门已过** ⇒ 可正式把 C≈0 读成 utilization deficit,"assay 不敏感"这一
  竞争解释已排除。**donor-dist 排序应成为 H2 标准构造**(verb 构造已证无效);未来任何
  H2 先过 oracle 灵敏度校准。
- 顾问 §3 剩余的 **α=0 vs 1 消融**(区分"训练采样噪声掩盖" vs "架构/bridge 限制"):
  ✅ **已跑,见下节**。

---

## 阶段 B α 消融:decoder-exposure 噪声 = utilization 瓶颈(✅ 2026-07-21)

`model.decoder_noise_alpha` 旋钮(仅缩放 decoder 训练时收到的 reparam 噪声,
KL/action-head/L_zero 仍读 z_mu);matched pair `sb_A0`(α=0,decoder 见 z_mu)vs
`sb_A1`(α=1,标准 reparam,≡ sb_B),各 1k 步、2 卡。产物
`results/stage_b_h2_{poscontrol,decompose}_alpha.json`。

| 指标 | A1(α=1) | A0(α=0) |
|---|---|---|
| **C_dist / oracle 上限** | +0.039 = **3.4%** | **+0.830 = 72.3%** |
| C_dist median | −0.068 | **+0.700** |
| R_out 前景(swap 下输出位移) | 0.0181 | **0.0394**(2.2×) |
| E_own 重建 | 0.00053 | **0.00025** |
| val PSNR | 29.92 dB | **32.96 dB**(+3.04) |
| posterior noise/SNR/eff-rank/类分离 | 1.00/0.54/9.6/1.95σ | **1.00/0.54/9.6/1.95σ(完全相同)** |

**结论(顾问 α 消融的确定性答复)**:α=0 把 decoder utilization 从 ~3% 抬到
**~72% oracle 上限**、输出响应翻倍、**且保真 +3 dB**,而**编码器后验完全不变**
(availability 一模一样)⇒ **瓶颈既非 availability 缺失、也非 conditioning 架构,而是
"训练期 decoder 收到的采样噪声(σ≈先验 ⇒ 逐维 SNR<1)教会 decoder 下调 latent 分支"**。
A0 出现真正的方向-使用签名:**错 donor 比 zero 更伤**(E_same/opp 0.00067 > E_zero
0.00051 > E_own 0.00025);A1 则 zero 最差(generic 使用)。这是**简单、可归因、恢复
causal use 的干预**(顾问 main 门第 5 项)。

**诚实性边界**:① 单 trunk/单 seed/1k 步,需 seed 复现 + Tier-2 action-following 才成
方法主张;② α=0 用确定性后验均值训 decoder,可能削弱 VAE 生成/OOD 平滑性,下游
do(z_a) 生成质量未测。

### 反平行 donor 测:α=0 恢复的是 signed-direction,不只是 magnitude(✅ 2026-07-21)
`code/h2_poscontrol.py` 新增 `antiparallel` 方案(same=最近 +a_i 的真 donor、opp=最近
−a_i 的真 donor ⇒ **幅度匹配** mag_o/i=0.97、方向翻转 cos_same+0.92/cos_opp−0.55),
产物 `results/stage_b_h2_antiparallel.json`。幅度受控下 E_opp>E_same 只能来自 decoder
跟随动作**符号**。结果:

| 方案 | A0(α=0) | A1(α=1) | A0/A1 |
|---|---|---|---|
| dist(幅度,mag_o/i 2.3) | +0.830 | +0.039 | 21× |
| **antiparallel(方向,mag_o/i 0.97)** | **+0.381** | **+0.064** | **~6×** |
| random(floor) | +0.069 | +0.014 | — |

A1 方向响应 +0.064 ≈ random/场景受限 oracle 噪声地板(~+0.06);A0 +0.381 = ~6×A1、
~5.5× 自身 random floor ⇒ **α=0 恢复 signed-direction utilization,非仅幅度**。
⚠️ frame-delta oracle 对跨场景反平行 donor 不适用(+0.058,像素 delta 不跨场景迁移)
⇒ 方向结论靠 A0-vs-A1 对比 + 内部地板,非 oracle ceiling;灵敏度由 dist oracle 背书。
donor 非完美反平行(cos_opp −0.55)⇒ +0.381 是 A0 方向使用的**下界**。已折入
`paper_draft.md` §9.1。

### 第二 seed 复现(✅ 2026-07-21):α=0 效应 seed-stable
matched pair `sb_A{0,1}_s1`(training.seed=1,data.seed=42 pin 同 split),
产物 `results/stage_b_h2_{poscontrol,decompose}_s1.json`。值 = seed42 / seed1:

| 指标 | A0(α=0) | A1(α=1) |
|---|---|---|
| C_dist / oracle | +0.830 / **+0.825**(72%) | +0.039 / +0.078(3–7%) |
| C_antiparallel | +0.381 / **+0.386** | +0.064 / +0.073 |
| R_out 前景 | 0.0394 / 0.0393 | 0.0181 / 0.0183 |
| E_own | 0.00025 / 0.00024 | 0.00053 / 0.00052 |
| val PSNR | 32.96 / 33.27 dB | 29.92 / 29.92 dB |
| posterior noise/SNR/eff-rank | 1.00/0.54/9.6–10 | 1.00/0.54/9.6–10 |

⇒ **两 seed 数值几乎逐项一致**,A0 恢复的 utilization(dist 0.83/0.825、antiparallel
0.38/0.39)与 posterior 统计都 seed-stable ⇒ 非幸运初始化。"单 seed" 诚实性边界已退役。

### α=0 生成/OOD 鲁棒性(✅ 2026-07-21,`code/ood_robustness.py`,`results/stage_b_ood_robustness.json`)
零训练 decode-only,A0/A1 两 seed,测两种 regime:
- **posterior 采样(z=μ+s·σ·ε)**:A0 **脆弱** —— recon MSE 从 0.00028(s=0)升到 0.0024–0.0029
  (s=1)到 0.0053–0.0073(s=2),~5–10× 恶化(训练没见过噪声);A1 平(ΔMSE ~+0.0002)。
  A0 在 μ 处胜、A1 在采样处胜 ⇒ **仅当下游 model 采样 z 才是代价**。
- **do(z_a) 干预(z_a→k·z_a,可控 WM 实际用的 regime)**:A0 **良好** —— motion(k) **单调增**
  (0.012→0.023,k=0→2)、**无伪影**(TV 0.010–0.011,低于真实帧 TV 0.0127);A1 **惰性**
  (motion 平 ~0.014–0.016)。两 seed 都复现。
⇒ 脆弱只限采样 regime;**对 chosen-z_a 干预 α=0 不仅安全且更优**(响应+无伪影 vs 惰性)。
若需可采样生成 latent,自选 fix = 中间 α(如 0.25)或 α schedule(未测,一 run 可补)。
已折入 `paper_draft.md` §9.2。

**剩余方法主张缺口** = **Tier-2 action-following**(顾问已选:OOD 先做→再咨询顾问)。
下一步:带两 seed α 结果 + 本 OOD 结果咨询顾问,预注册 Tier-2 协议/控制/目标 venue。

**对论文的影响**:已折入 `paper_draft.md`(标题改为 "Diagnosing — and Removing —"、
摘要加第 5 贡献、§9 新节、§7 前引更新)。论文从"纯诊断+开放问题"升级为
"诊断 + 可归因的 decoder 侧修复",workshop 更稳、向 main 靠(仍缺跨生态/数据 + Tier-2)。

## Tier-2 Phase 0 零训练定位硬门(✅ 2026-07-21,顾问第四轮 §0)

代码 `code/{paired_latent,cross_decode,eval_phase0}.py`;产物
`results/stage_b_phase0_{paired_latent,cross_decode}.json`、`results/phase0.log`。
在冻结 Eval-A pool(N=1500,840 anchors,latent 全 z_μ)上跑,两 seed(42/1)。donor 方案
与 H2 正对照完全一致(frac verb=0.488、dist mag 2.29、antipar cos_opp −0.55、oracle×dist
1.149——逐项复现)。**对角格 dist C 复现 poscontrol(0.830/0.039 seed42、0.825/0.078 seed1)
= correctness gate 通过。**

### 0.1 paired-latent audit:两 encoder 的 μ 逐样本近乎相同
| 量(action 子空间) | seed42 | seed1 | 读法 |
|---|---|---|---|
| cosine(μ_A0,μ_A1) mean | 0.9999 | 0.999 | 逐样本方向几乎一致 |
| per-dim Pearson r | ~1.000 | ~1.000 | 逐维一一对应 |
| linear CKA | 0.9999 | 0.999 | 表示等价 |
| orthogonal-Procrustes 对齐 R² | 0.9998 | 0.998 | 连旋转都几乎不需要 |
| kNN 邻居重合率@10 | 0.967 | 0.889 | 局部几何保持 |
| ‖Δμ‖/‖μ‖ | 0.016 | 0.048 | 幅度位移极小 |
| **冻结 readout cross vs self R²** | A0→A1 .243 / self .242 | A0→A1 .234 / self .232 | **动作坐标共享:同一 readout 套两 encoder 无损** |
| dist_corr(latent↔GT18) A0/A1 | 0.729/0.729 | 0.724/0.724 | 动作几何保持一致 |
| frac(d_same<d_opp) dist A0/A1 | 0.877/0.877 | 0.864/0.867 | 排序一致 |

⇒ α=0 **没有改变 encoder 的逐样本 μ 坐标**(不只是汇总统计相同)。§9 的"matched summaries
≠ per-sample identical"⚠️ 现按 audit **收紧为"encoder μ 图基本不变"**(已折入 §9.3)。

### 0.2 四格 cross-decoding:利用率完全由 decoder 决定(干净 decoder-side)
dist C_pooled(行=decoder,列=喂给它的 encoder μ):

| | E_A0(s42) | E_A1(s42) | E_A0(s1) | E_A1(s1) |
|---|---|---|---|---|
| **D_A0** | +0.830 | +0.824 | +0.825 | +0.817 |
| **D_A1** | +0.039 | +0.039 | +0.078 | +0.078 |

- **行差 11–21×,列差 <1%**:D_A0 对两个 encoder 都强、D_A1 对两个都弱 ⇒ 顾问 §0.2 的
  **"很干净的 decoder-side utilization 定位"**(不是 co-adaptation、不是 encoder 条件表示变了)。
- antiparallel(符号方向)同型:D_A0 ≈0.38/0.39、D_A1 ≈0.064/0.073;fidelity 也由 decoder 定
  (D_A0 PSNR ≈36dB、D_A1 ≈32.7dB,与 encoder 无关)。

### 判读落位(顾问 §8 决策树)= **"paired μ 相同 且 只有 D_A0 强"**
这条正是顾问列的:*"α=0 是本 decoder 的修复,应收窄论文"*;且 §0.2 第四诊断:
**"paired μ 近乎相同 且只有 D_A0 强 ⇒ 外部只看 μ 的 ACWM 原则上拿不到 A0 收益。"**
→ **route-C 独立 consumer(冻结 encoder、只吃 μ)预计 A0≈A1(无 transfer)**,因为两臂交给
下游的 conditioning 数据本身几乎一样。α=0 的价值目前 = **本 LAM decoder 的 conditioning-edge
修复**(workshop 强);要写"下游"必须证明修复能跨 consumer 迁移,而本定位使"仅靠 μ 迁移"的
先验期望很低。

**决策点(交用户,顾问排期要求每阶段末交决策)**:
(a) 收窄为 LAM-decoder 结论 + 可选 Phase-1 Tier-1.5(原 decoder K 步 rollout,within-model);
(b) 仍做 Phase-2 route-C 独立轻量 predictor——但要接受"很可能 A0≈A1",其价值变成
**证伪"μ-only 迁移"** 并把 α=0 原则的迁移问题清楚地转到"consumer 训练时喂确定性 latent"这一侧;
(c) 若目标是"下游有价值",真正能动的旋钮是**把 α=0 训练原则搬进独立 consumer 的训练**
(给 consumer 训练时喂确定性 z_μ,而非 μ-only plug-in)——顾问明确说这不是默认下一步,需用户拍板。

**✅ 用户决定(2026-07-21)**:先做 **Phase 1 Tier-1.5**,且**先冻结 Phase 0.3 预注册**。
- **Phase 0.3 已冻结**:Eval-B = `data/eval_manifest_B.json`(14 held-out **task** / 290 ep,10 可逆 +
  4 非可逆,与 Eval-A 不相交,`code/make_eval_manifest_B.py`);完整预注册 = `notes/tier2_prereg.md`
  (数据划分/四独立性/donor 门控/指标/正负控制/成功门 + margin 校准程序/SAM3 处理/2B go-no-go/措辞守门)。
  **Eval-B 密封,只在 Phase-2 route-C 最终解封一次;Phase 1 在 Eval-A 上跑。**
- **✅ Phase 1 Tier-1.5 完成**(`code/{rollout_tier15,eval_tier15}.py`,`results/stage_b_tier15.json`,
  200 clip / 149 可逆,K_max=16 skip=2,own-latent rollout,latent 变体 own/zero/neg,两 seed):

  | A0−A1 horizon 对比(dB) | @4 | @8 | @16 |
  |---|---|---|---|
  | ΔPSNR_own(seed42/seed1) | +4.1/+4.3 | +4.4/+4.5 | +4.6/+4.8 |
  | **Δsign_use = own−neg(seed42/seed1)** | +4.9/+5.4 | +5.2/+5.9 | **+5.2/+6.7** |

  绝对值:A0 sign_use ≈6.0–7.5 dB(喂**取反**动作子空间 ⇒ 对 GT 保真掉 6–7 dB),A1 ≈1.0–1.2 dB。
  A0 PSNR_own 30→25、A1 26→20(A0 领先 4–5 dB **随 horizon 扩大**)。⇒ **α=0 的 decoder
  利用率(保真 + signed-direction following)不仅在 K 步自回归 rollout 中存活,还随 horizon 增强**,
  两 seed 一致(within-LAM functional validation)。
  **诚实框定**:① 主判别量是 `sign_use`(own vs neg,幅度匹配);`gen_use`(own vs zero)被
  baseline 保真混淆(A1 的 zero-rollout 更差 ⇒ 其 gap 反而更大,K=16 时 Δgen_use 变负),不作判别。
  ② 两臂都非纯静止复制(motion_ratio 0.4–0.7);A1 是**运动更大但方向无关的漂移**(tv_ratio 0.79–0.90、
  PSNR 低、sign_use≈1),A0 是**更小、更平滑(tv_ratio 0.53–0.55)、符号锁定**的运动。
  ③ Eval-A、within-LAM,**不作下游 WM 主张**;Eval-B 仍密封。
