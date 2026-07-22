# Tier-2 预注册(Phase 0.3 冻结,2026-07-21)

**在看任何 Tier-2 结果前冻结。** 本文件是 Tier-2 的单一预注册事实源:数据划分、donor 门控、
指标、正负控制、成功门、margin 校准程序、SAM3 处理。改动须记明日期与理由;冻结项一旦解封不可回改。

依据:顾问第四轮 `notes/consult_tier2_prompt.md` §4–§5;Phase-0 定位结果 `notes/stage_b_plan.md`
末节 + `paper_draft.md` §9.3。前置事实:**α=0 修复已定位为 decoder-side,且 A0/A1 的 encoder μ
逐样本近乎相同**(CKA≈1、cross-readout=self)⇒ **route-C 只吃 μ 的 consumer 预计 A0≈A1**。
本预注册据此把 route-C 的成功门与"零结果如何判读"都提前写死,避免事后叙事。

---

## 0. 范围与密封状态

- **已解封(model selection 用完)**:Eval-A = `data/eval_manifest.json`(390 ep,episode 级 holdout)。
  H2 分解/正对照/α 消融/Phase-0 定位全在此。**Tier-1.5(Phase 1)也在 Eval-A 上跑**(within-LAM,
  探索性,不算 Tier-2 解封)。
- **密封(只解封一次)**:Eval-B = `data/eval_manifest_B.json`(290 ep,14 held-out **task**,10 可逆
  + 4 非可逆)。**仅用于 Phase-2 route-C 最终 A0-vs-A1 解封。** 在此之前任何脚本、可视化、margin
  校准都**不得**读取 Eval-B 的 A0-vs-A1 对比量(margin 校准只允许读 A1 单臂 + 控制,见 §7)。

## 1. 四独立性(顾问 §5,防"重建 swap 换皮")

1. **consumer 独立**:Phase-2 predictor **从零初始化**,禁止复用 `D_A0/D_A1` 权重;可复用架构定义。
2. **data 独立**:consumer 训练池 = 全部 − Eval-A − Eval-B 的 14 个 task;Eval-B 只在最终解封。
3. **horizon 独立**:H2 是 one-step donor swap;Tier-2 是 autoregressive **K 步** rollout。
4. **metric 独立**:H2 primary = donor-swap 像素 MSE;Tier-2 primary = 前景 track displacement +
   signed 方向指标;像素重建只作 fidelity guardrail。

## 2. 数据与模型

- **冻结四 encoder**:`sb_A0 / sb_A1 / sb_A0_s1 / sb_A1_s1`(step_0001000)。encoder 不回传梯度。
- **latent**:全用确定性 `z_μ`(Phase-0 已证 A0/A1 的 μ 近乎相同;OOD §9.2 证 chosen-z 不需采样)。
  ⇒ **不需要先跑 α=0.25**;α=0.25 只在 route-C transfer 成立后作 utilization→Tier-2 曲线第三点。
- **consumer 训练池 μ 缓存**:离线缓存 train transitions 的 raw `z_μ`,不回传 encoder。

## 3. Phase 1 — Tier-1.5(Eval-A,within-LAM functional validation)

- **decoder**:原 LAM decoder(`D_A0 / D_A1`),不新训。
- **rollout**:K ∈ {4, 8, 16} 自回归;每步 `ô_{t+1} = D(ô_t, z_t)`,`ô_t` 回喂下一步。
- **两 regime**:
  1. **own-latent**:从 reference clip 提取真实 `z_μ` 序列,同 clip 初始帧起 rollout。
  2. **target-transfer**:固定 source context,插入另一 clip 的 `z_μ` 序列。**donor 必须 GT-18D 门控
     (dist / antiparallel),禁用已证伪的 verb donor。**
- **比较**:A0 vs A1,两 seed。**论文明确标 within-LAM functional validation,不作下游 WM 主张。**
- **产出**:暴露递归漂移 / 静止复制 / 符号失效;并**复用后续 tracking scorer**(与 Phase-2 共用)。

## 4. Phase 2 — route-C 最小真正 Tier-2(Eval-B 解封)

- 每个 encoder 分别训一个**同架构或更小的 U-Net/ConvNet** next-frame/video predictor,从零初始化、
  encoder 冻结,用 **FiLM / cross-attention** 明确消费 `z_μ`;只看 `(o_≤t, z_{t:t+K-1})`,推理 AR K 步。
- **禁止复用 `D_A0/D_A1` 权重**(consumer 独立)。
- **≥2 个 matched downstream training seed**(clip 大 N 不能替代 seed)。
- 两 regime 同 §3(own-latent / target-transfer),K 预注册主点 = **K=8**(并报 4/16 curve)。

## 5. 指标(metric 独立)

**Primary(action-following):**
- **canonical-FDCE@K** / 前景 track displacement error(reference-defined foreground points);
- **signed displacement cosine** 或 endpoint direction error(对 magnitude-matched antiparallel donor,
  防 Chamfer 均值掩盖符号);
- **Δ_do = Error(shuffle/antiparallel z) − Error(target z)**(直接测是否服从选定 condition)。

**Secondary:**
- 冻结 IDM 从生成 frame pair 读 18D 后的 normalized L1/cosine(IDM 须先在 real/persistence/正对照上
  校准生成域偏差);horizon-wise error slope;`R_out` 仅作机制量,不替代 action-following。

**Fidelity guardrail:** PSNR / FG-PSNR / SSIM / LPIPS;source-bg fidelity(防 target donor 携外观);
temporal TV / flow smoothness;有效 track 数 + catastrophic-failure rate;**motion 下界**(防静止复制)。

⚠️ canonical-FDCE reduction order 与 CD-LAM 论文 headline 不一致(顾问 §2.5)⇒ 标 `canonical-FDCE@K`,
**不与论文 49-frame headline 横比**。

## 6. 正负控制(顾问 §4.4)

- **正对照(已知强可控)**:同一轻量 WM 架构**直接以 GT 18D action 为条件**。必须能把 target 与
  shuffle / antiparallel 显著分开。**不过门 ⇒ 先修 Tier-2 scorer,不能用 A0/A1 零结果反推。**
- **负对照**:obs-only / zero-condition;within-task time-shuffled latent;magnitude-matched
  antiparallel latent;persistence / static rollout。
- 所有 arm 共用 rollout seed 和 clip / donor triples。

## 7. 成功门(看 A0-vs-A1 前冻结)

**Margin 校准程序(不看 effect)**:practical margin 由 **A1 重复 inference 方差** + **正/负控制自然间距**
校准。这两项只读 A1 单臂与控制,**不读 A0-vs-A1 对比**,故可在解封前算。**禁止从最终 effect 倒推 margin。**

**判定(全部满足才算 route-C transfer 成立)**:
1. A0 相对 A1 的 paired **canonical-FDCE@K 相对改善 ≥ 10%**,且 task→episode/clip hierarchical
   **95% CI 排除 0**;
2. A0 内部 `target z` 显著优于 shuffle 和 antiparallel,而 **A1 的 gap 显著更小**;
3. **signed-direction 指标同向**成立(不能只有像素/Chamfer 改善);
4. **PSNR 非劣容差 −0.5 dB**;LPIPS/bg-fidelity 容差由 A1 repeated-inference 方差冻结;
5. **正对照(GT-18D)与负对照 ordering 通过**;
6. 以 **downstream training seed** 为独立复现单位(≥2),clip 级大 N 不替代 seed。

## 8. route-C 结果判读(结合 Phase-0,顾问 §8 决策树)

- **A0 ≫ A1 且过门** ⇒ 强 workshop + 有意义 main bridge:α=0 收益跨 consumer 迁移 +
  H2 预选的高-utilization checkpoint 在未见 Tier-2 上 action-following 更强(prospective causal validation)。
- **A0 ≈ A1(Phase-0 的先验预期)** ⇒ **不是失败,是预注册的确认**:结合 Phase-0
  (paired μ 相同 + 仅 D_A0 强)判为 **"α=0 是本 decoder 的修复,μ-only consumer 不继承"**,
  论文收窄为 LAM-decoder 结论;若要下游价值,唯一能动的旋钮是**把 α=0 训练原则搬进 consumer 训练**
  (给 consumer 训练时喂确定性 z),非 μ-only plug-in——那是单独的下一轮决策。
- **连 GT-18D 正对照都无 following** ⇒ Tier-2 scorer/架构未过正对照,先修 scorer,不下模型结论。

## 9. SAM3 40% object 缺失处理(顾问 §4.6)

- 看输出前**冻结 mask-eligible 子集**(各 arm 完全共用);报总覆盖率 + per-task 覆盖率 +
  included/excluded task 分布;
- mask **只从 reference/source 定义**,不因某模型生成质量改变纳入概率;
- 主 FDCE 之外**另加覆盖全 clip 的光流 / 手部关键点 displacement**;
- **不与 CD-LAM 49-frame FDCE headline 横比**。

## 10. 2B(route B)go/no-go —— 五条同时满足才进

① 官方 runtime doctor/dry-run 本机 ≤1 天跑通;② 40D conditioning adapter 形状与 checkpoint lineage
明确;③ rollout+SAM3/CoTracker+canonical scorer 小样本端到端通;④ 有预算为 A0/A1 各训 matched
Stage-2;⑤ 轻量 Tier-2 已显非零效应。任一不满足 ⇒ 停 2B,按轻量 Tier-2 或 workshop 收束。

## 11. 措辞守门

- 只有 A0/A1 两点:写 "prospectively transfers" / "aligned with",**不写 "predicts"**;
  真 predictor 主张需 5–6 checkpoint(A0/A1 两 seed + B/R/D/U [+α0.25])+ 预注册 Spearman +
  leave-one-family-out,最好跨数据域。
- "canonical-FDCE@K",不做 CD-LAM headline 复现式横比。
- Tier-1.5 一律标 "within-LAM functional validation",不作下游 WM 主张。

---

**冻结清单**:Eval-B(`data/eval_manifest_B.json`,14 task/290 ep)、donor 门控(GT-18D dist/antiparallel)、
指标(§5)、控制(§6)、成功门 + margin 校准程序(§7)、SAM3 子集规则(§9)。**以上在 Phase-2 解封前不改。**
