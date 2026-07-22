# 新会话启动 prompt:Tier-2 执行(顾问第四轮已定,localize→transfer→scale)

把下面「---」之间的内容整段贴进新会话作为开场。它是自包含的执行 handoff。

---

我在 `/home/xuan/embodied-ai/lam_disentangle` 做 Latent Action Model(LAM)精细化研究。
上一阶段:方向线(L_dir)被否决 → 把问题重定位为 **availability × utilization** → 用
**正对照验证过的干预 assay** 证明瓶颈是 **decoder 不使用 z_a** → 一个极简干预
**训练时关掉喂给 decoder 的采样噪声(α=0)** 把 decoder 对动作条件的利用率从 oracle 上限
~3% 抬到 ~72%(幅度)、~6×(符号方向),**两 seed 复现**、保真 +3dB。外部顾问第四轮
已就"Tier-2 该怎么做"给出完整方案。**这个会话执行顾问的方案。**

## 先按顺序读这几份(建立上下文,先读完再动手)
1. `notes/paper_draft.md` —— 合并后的诊断论文草稿(availability≠utilization + α=0 修复);
   §9 是 α 消融、§9.1 反平行方向测、§9.2 OOD 鲁棒性。数值权威、可追溯到 `results/`。
2. `notes/consult_tier2_prompt.md` —— **末尾「顾问第四轮回复」是本会话要执行的方案**
   (§0 零训练硬门、§4 Tier-2 协议与预注册、§5 四独立性、§8 最小可行排期)。
3. `notes/stage_b_plan.md` —— 阶段 B 决策状态单一事实源(H2 分解/正对照/α 消融/OOD 各节)。
4. 代码骨架:`code/h2_poscontrol.py`(donor 方案 dist/antiparallel/random/verb + oracle +
   `lam_c`)、`code/h2_decompose.py`(own/same/opp/zero/shuffle 分解 + posterior SNR)、
   `code/ood_robustness.py`、`code/model.py`(`decode_with` 干预入口、`decoder_noise_alpha`
   旋钮)、`code/benchmark_lam.py`(`_encode_mu_var`/`_decode_from_mu`/`_make_loader`/RUN_SPECS)。

## 关键既有资产(直接复用,别重造)
- **四个 encoder/decoder checkpoint**:`checkpoints/lam-dis/ft_l40_sb_A{0,1}{,_s1}/step_0001000`
  (A0=α0、A1=α1;`_s1`=seed1;无 `_s1`=seed42)。RUN_SPECS 里名字 `sb_A0/sb_A1/sb_A0_s1/sb_A1_s1`。
- **冻结评测 manifest**:`data/eval_manifest.json`(390 ep,encoder 从未训练)。⚠️ 这是 H2 用的
  **Eval-A**;Tier-2 最终解封必须用**另一个 held-out split(Eval-B)/held-out task**(见纪律)。
- **donor 方案**:`h2_poscontrol._build_schemes` 已实现 dist(幅度,按 18D 距离排序 frac=1.0)、
  **antiparallel**(符号方向,same=最近+a_i、opp=最近−a_i,幅度匹配 mag_o/i≈0.97、cos_opp≈−0.55)、
  random(负对照)、verb(已证伪,勿用作方向)。oracle×dist=+1.15 已证 scorer 灵敏。
- Python:`/home/xuan/.venv/bin/python`。

## 顾问定的路径(一句话)
**先用 cross-decoding 定位修复发生在哪里(零训练),再用一个独立轻量 decoder 做跨-consumer
causal transfer;只有 transfer 成立且 2B runtime 可行,才升级到 matched DreamDojo Stage-2。**
⚠️ 顾问纠正了一个措辞:现有"编码器后验完全不变"只由**分布汇总量**支持,不等于逐样本
`μ(o_t,o_{t+1})` 相同;准确表述是"**去掉 encoder→decoder 训练边上的噪声,让一个 matched
encoder-decoder pair 学到更可用的动作通道**",不是"只有 decoder 变了"。paper_draft §9 已加 ⚠️,
Phase 0 的 audit 结果出来后收紧措辞。

## 执行排期(顾问 §8;每阶段结束把决策点交给用户,不要自己拍板叙事)

### Phase 0 — 零训练硬门(最高优先,先做,决定后面一切)
0.1 **paired-latent audit**:在 Eval-A 上逐 transition 比 A0 vs A1(两 seed):
    `‖μ_A0−μ_A1‖`、cosine、逐维相关;linear CKA + orthogonal Procrustes 残差;
    kNN 邻居重合率;same/opp action-distance ordering;**用同一个冻结 action readout** 做
    cross-encoder 预测(训一个 readout,套两个 encoder 的 μ,不各自重拟合)。
0.2 **四格 cross-decoding**:`D_{A0/A1} × E_{A0/A1}`,latent 全用 μ。每格跑
    own/dist/antiparallel/random 的 `C`、`R_out`、fidelity。判读(顾问 §0.2):
    D_A0 对两 encoder 都强 & D_A1 都弱 ⇒ 干净 decoder-side;只有对角强 ⇒ co-adaptation;
    A0-encoder 喂两个 decoder 都更好 ⇒ α 改了 encoder 条件表示;paired μ 近乎相同且只有
    D_A0 强 ⇒ 外部只看 μ 的 ACWM 原则上拿不到 A0 收益。**这是决定 Tier-2 架构的最高价值实验。**
0.3 **冻结** Tier-2 的 manifest(Eval-B / held-out task,与 H2 clips 分离)、donors(GT-18D
    dist/antiparallel 门控)、metrics、practical margins —— 都在看任何 Tier-2 结果之前。
0.4 按 audit 结果收紧 `paper_draft.md` 的 decoder-only 因果措辞。
→ **决策点**:cross-decoding 定位结果 + 是否预期 Tier-2 transfer,交用户。

### Phase 1 — Tier-1.5(1–2 天):原 LAM decoder 的 K 步 rollout
用**原 LAM decoder**做 `K={4,8,16}` 自回归 rollout,两 regime:own-latent + target-transfer
(donor 必须 GT-18D 门控,禁用 verb donor)。比 A0/A1 两 seed。**论文中明确标为 within-LAM
functional validation**,不作"下游世界模型有效"主张。快、暴露递归漂移/静止复制/符号失效,
且能复用后续 tracking scorer。

### Phase 2 — 最小真正 Tier-2(2–5 天):独立轻量 conditional predictor(路线 C)
- 冻结四个 encoder;离线缓存 train transitions 的 raw `z_μ`(不回传 encoder)。
- 每个 encoder 分别训一个**同架构或更小的 U-Net/ConvNet** next-frame/video predictor,
  **从零初始化**、encoder 冻结,用 FiLM/cross-attention 明确消费 `z_μ`;只看 `(o_≤t, z_{t:t+K-1})`,
  推理自回归 K 步。⚠️ **禁止复用 `D_A0/D_A1` 权重**(consumer 独立)。
- **正对照**:同架构 WM 直接以 **GT 18D action** 为条件(必须能把 target 与 shuffle/antiparallel
  分开;不过门就先修 scorer,不能用 A0/A1 零结果反推)。**负对照**:obs-only/zero、
  within-task time-shuffle、幅度匹配 antiparallel、persistence。
- **指标**(metric 独立):primary = canonical-FDCE@K / 前景 track displacement + signed
  displacement cosine + `Δ_do=Error(shuffle/antipar z)−Error(target z)`;secondary = 冻结 IDM
  读 18D(需先校准生成域)、horizon 误差斜率;像素 MSE 只作 fidelity guardrail。
  fidelity 守门:PSNR/FG-PSNR/SSIM/LPIPS、source-bg fidelity、motion 下界(防静止复制)。
- **≥2 个 matched downstream training seed**(clip 大 N 不能替代 seed)。
- **成功门(看结果前冻结,§4.5)**:A0 vs A1 paired canonical-FDCE 相对改善 ≥10% 且
  task→episode/clip hierarchical 95% CI 排除 0;A0 的 target-z 显著优于 shuffle/antiparallel
  而 A1 gap 更小;signed-direction 指标同向;PSNR 非劣容差 −0.5dB;正负对照 ordering 通过。
- **SAM3 40% object 缺失**(已知、任务相关):看输出前冻结 mask-eligible 子集(各 arm 共用)、
  报覆盖率、mask 只从 reference/source 定义;另加覆盖全 clip 的光流/手部关键点 displacement;
  **不与 CD-LAM 论文 49-frame FDCE headline 横比**。
→ **决策点**:A0≫A1 且过守门 ⇒ 强 workshop + 有意义 main bridge;A0==A1 ⇒ 结合 cross-decoding
  判读(paired μ 相同且仅 D_A0 → 收窄为本 decoder 修复;paired μ 不同但外部相同 → 新 decoder
  能重学两种 latent、H2 非跨 decoder predictor;连 GT18 都无 following → Tier-2 scorer 未过正对照)。

### 2B(路线 B)go/no-go —— 五个条件同时满足才进:
① 官方 runtime doctor/dry-run 在本机 ≤1 天跑通;② 40D conditioning adapter 形状与
checkpoint lineage 明确;③ rollout+SAM3/CoTracker+canonical scorer 小样本端到端通;
④ 有预算为 A0/A1 各训 **matched Stage-2**(不是替换 frozen ACWM 输入);⑤ 轻量 Tier-2 已显非零效应。
任一不满足 ⇒ 停 2B,按轻量 Tier-2 或 workshop 收束。

## 不可越界的纪律
- **数据独立**:H2 用 Eval-A 做模型选择;Tier-2 最终只在未看的 Eval-B/held-out task **解封一次**。
- **四独立性**(顾问 §5):consumer(不复用 D_A0/D_A1 权重)、data(Eval-A≠Eval-B)、
  horizon(H2 一步 vs Tier-2 K 步)、metric(H2 是 donor-swap 像素 MSE,Tier-2 primary 是 track
  displacement + signed 指标)。
- **正对照先行**:GT-18D 正对照不过门就先修 Tier-2 scorer,不能用被测模型的零结果下结论。
- **margin 冻结**:practical margin 由 A1 repeated-inference + 正负对照的自然间距校准,
  **不从最终 effect 倒推**。
- **intermediate α**:deterministic-μ Tier-2 直接用 A0,**不需要先跑 α=0.25**;只有系统要
  从 `q(z|o,o')`/stochastic latent policy 采样才需要;α=0.25 是 utilization→Tier-2 曲线的
  第三点,在 transfer 成立后才做。
- **"指标预测下游"措辞**:只有 A0/A1 两点时写 "prospectively transfers"/"aligned with",
  不写 "predicts";真 predictor 主张需 5–6 checkpoint(A0/A1 两 seed + B/R/D/U [+α0.25])、
  预注册 Spearman + leave-one-family-out、最好跨数据域。
- **GPU**(共享服务器):每次启动前 `nvidia-smi` 确认、选真正空的卡(GPU 2/3 常空,0/1 常有他人)、
  **不动他人进程**、训练一律 2 卡、跑长任务用 `python -u`(无缓冲日志便于看进度/发现争用)。
- **不编造**:canonical-FDCE 的 reduction order 与 CD-LAM 论文 headline 不一致(顾问 §2.5),
  新数字标 `canonical-FDCE@K`,不做复现式横比。

## 顾问核实过的可行性事实(供 2B 决策)
CD-LAM Stage-2 = 用 Stage-1 latent 续训 ACWM **2000 步**,2B per-GPU batch 12;官方仓库有 3 个
2B research entry,`models/pretrain/model.pt` 与公开 **32D** LAM 是兼容对——但**我们 A0/A1 是 40D,
不能直接替换**;官方**没有**可移植的 rollout scripts / 精确 300-clip manifest / 完整
rollout→SAM3→CoTracker runner。⇒ 2B 前先做 ≤1 天 runtime-doctor 审计。

先只做 **Phase 0(0.1 + 0.2)**,把结果和判读交我,再决定 Phase 1/2 怎么走。GPU 任务前先 nvidia-smi。

---
