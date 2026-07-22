# Benchmark v2 结果简报(2026-07-15,9 个 checkpoint)

结果在 `results/lam_benchmark_v2/summary.md`。三个新指标直接改写了几个旧结论。

## 发现 1:旧的 "action-kNN 低于随机" 是**基线用错的假象**,已推翻
- 旧对照 `chance=0.073` 是"永远猜多数类"分类器的基线,不是检索一致率的零假设。
- 正确的 permutation null(episode 层打乱 verb)≈ 0.035–0.040。逐模型 z-score:
  - ours_a: top1=0.038,**z=0.10(正好随机)**
  - kl: 0.042,z=1.03;ours_r2: 0.045,z=1.31 —— 都在随机水平(|z|<2)
  - **contrastive_v3(verb-SupCon): 0.074,z=4.08 —— 唯一显著高于随机**
  - raw_lam: 0.061,z=2.84
- 结论:大多数模型 kNN **在随机水平,不是低于**。SupCon 确实能把这个指标推上去(v3 做到了)。

## 发现 2(关键):真值 18D 动作自己也过不了 verb-kNN —— 这个指标是**范畴错误**
- `gtact_knn_verb_top1 = 0.0217`(所有模型相同,因为跑在同一份真值上),
  **比 permutation null(0.037)还低**。
- 含义:即使拥有完美的连续动作信息,verb-kNN top1 也只有 ~0.02。因为真值动作按
  **方向/幅度**聚类,这个结构**横切** verb 类别(慢速 open 和慢速 close 的动作向量,
  可能比快慢两个 open 更近)。
- 直接推论:**verb-kNN 不该作为 LAM 质量指标**;为它优化(SupCon 拉近同 verb)=
  逼近一个连续动作码本就不满足的量。v3 用 R² 下降(0.267 vs R 的 0.365)换来了这个
  指标上升 —— 迎合了坏指标,牺牲了真正想要的东西。

## 发现 3(关键):opp_cls 的"相反动作可分"几乎全是**初始帧外观混淆**
- `opp_cls_bal_acc`(z_a probe 分 pull/push)≈ 0.68–0.78,看着不错;
- 但 `opp_cls_bal_acc_static`(同 probe 跑在静止帧 E(o_t,o_t) 上)= **0.96–0.98**;
- `opp_cls_direction_gain = bal_acc − static` **全为负**(−0.20 ~ −0.31)。
- 含义:可逆 task 里初始场景状态(抽屉开/关)已经能 96%+ 分出 verb,transition latent
  的动作信息不但没加分,反而更差(运动编码把外观线索冲淡了)。**没有任何模型能证明
  "按方向"分开了相反动作** —— 表面 70% 的成功全是外观泄漏。

## 发现 4:swap 解码测试(公平外部裁决)不支持 reverse 系 loss
- `swap_opp_delta_psnr`(相反 z_a vs 同 verb z_a 解码,对真值 PSNR 差):
  ours_a=0.411(最高)、kl=0.384、v3=0.378、ours_r=0.279、**ours_r2=0.262(最低)**。
- reverse 系 loss(R/R2)不但没提升解码器的方向敏感度,反而略降。ours_a(纯 zero-cal)最好。
- 但差值都很小(0.26–0.41 dB)且无方差,只能说"reverse loss 没有正面证据"。

## 路由三元组(R²_full / R²_za / R²_ze)
- ours_r / ours_r2:R²_za=0.366/0.360(最高),R²_full≈R²_za(动作全在 z_a,路由干净)。
- 但 R²_ze(env 子空间的动作泄漏)= 0.15–0.18,**所有模型都漏**,ours 不例外。
- contrastive_v3 是唯一 R²_ze 低的(0.038)—— SupCon 把动作挤出 z_e 最彻底,
  但代价是 R²_za 也低(0.260)。**"挤干净 z_e" 和 "z_a 高保真" 目前是此消彼长。**

## 对 headline 的净结论
- **Ours-R2(λ=0.1)的真实、可辩护的胜点是:动作 R²(0.365)最高 + 路由最干净**
  (动作全进 z_a)。它**不是** "相反动作分离" 的赢家 —— 那个宣称被外观混淆推翻了。
- 把 Ours-R2 的故事从 "分开相反动作" 改写成 "**在不损失连续保真的前提下把动作干净地
  收进 z_a**" 才站得住。相反动作分离在这份数据上目前是**未解**问题。
