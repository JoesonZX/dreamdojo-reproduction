# Benchmark v2 — 指标修正与新增(2026-07-14)

本次改动动机:三个实现层问题(kNN 随机基线用错、labeleff 全负、headline probe 读全 latent
而非 z_a),以及缺少"零训练成本就能裁决关键争论"的诊断指标。全部改动在
`code/benchmark_lam.py`;旧 结果目录(`results/lam_benchmark*`)的数值不受影响,但
**旧 kNN 的 "低于随机" 结论作废**(基线用错,见下)。

## 1. kNN:permutation null 取代 majority 基线
- 旧 `action_knn_chance` = 多数 verb 频率(0.073)。这是"永远猜多数类"分类器的基线,
  不是 kNN 检索一致率的零假设;检索一致率的解析 null ≈ Σ_c p_c²,长尾 85 verb 下低得多。
- 新:`action_knn_null_top1_mean/std` — verb 标签在 **episode 层**打乱 ×20(保 verb 直方图、
  episode 结构、近邻图),用同一近邻图打分。读 `action_knn_top1_z`,|z|<2 即随机水平。
- 旧值保留为 `action_knn_chance_majority`,仅供参考,不得用于比较。
- 合成数据验证:null(0.136) 确实 ≠ majority(0.175)。

## 2. kNN 两个对照(裁决 "env 泄漏 vs 连续运动组织 vs 指标范畴错误")
- `gtact_knn_verb_top1`:同一 kNN 跑在 **18 维真值动作**(列标准化)上 = verb-kNN 的天花板。
  若真值本身也低 → verb-kNN 对连续动作码是范畴错误,该降级的是指标不是模型。
- `probeact_knn_verb_top1`:kNN 跑在 **out-of-fold ridge 预测动作空间**(z_a→18D,5-fold)。
  此处高 + 原始 z_a 低 → 信息在但方差分配错位(度量问题);此处也低 → 线性可读的动作信息不足。
- 判定链:gt 高 & probe 高 & raw 低 → 方差错位,才轮到讨论 SupCon vs 环境排除。

## 3. 拆分 probe:R²(z_a) / R²(z_e) / R²(full) 路由三元组
- 旧 headline `mlp_action_r2` 读的是全 40 维 mu(z_e 也在里面),并不测 "z_a 干净度"。
- 新增 `mlp_action_r2_za` / `mlp_action_r2_ze`(MLP)与 `action_r2_za_linear` / `action_r2_ze_linear`。
- 读法:`_ze` = 动作漏进 env 子空间;`full − _za` = z_a 没吃下的动作信息。
  这是用外部目标直接量化路由质量的非循环指标。

## 4. labeleff 修复
- 旧:固定 alpha=1.0 的 Ridge,在 1% 档(~12 样本 × 40 维)严重过拟合,全线负 R²,
  数值全是 probe 方差 → **旧 labeleff 列不可比较,已 re-base**。
- 新:RidgeCV(LOO-CV,alpha ∈ logspace(-2,5,15))。

## 5. opp_cls 静态对照(初始状态外观混淆)
- 可逆 task 内初始场景状态与 verb 完全相关(待 remove 的插座是插着的),只编码外观状态
  的 z_a 也能分对。
- 新:同一 probe 跑在静止对 latent E(o_t,o_t) 上(`opp_cls_bal_acc_static`);
  只有 `opp_cls_direction_gain = bal_acc − static` 才能记作"编码了方向"。

## 6. do(z_a=0) 解码测试(static_response 的非循环版,对齐 CD-LAM do(u=0))
- `zeroact_motion_suppression` = PSNR(D(o_t, 0⊕z_e), o_t) − PSNR(D(o_t, z), o_t)。
  >0:置零 z_a 把预测拉回当前帧(运动被抑制)。
- `zeroact_still_margin`:零动作解码更像 o_t 还是 o_{t+1}。

## 7. 相反对 swap 测试(reverse 系 loss 的公平外部裁决,mini 版 target-action transfer)
- 固定 (o_t, z_e),z_a 换成 (a) 同 task 同 verb 异 episode 捐赠者、(b) 同 task 相反 verb
  捐赠者,解码后对真值 o_{t+1} 打 PSNR。
- `swap_opp_delta_psnr = PSNR(a) − PSNR(b)`,>0 = 相反 z_a 解码出可测差异的未来
  (方向"可执行",不只是几何可分)。两个捐赠者都跨 episode,场景失配在差值中相消。
- `swap_context_cost_psnr` = own − same-verb swap = 动作码的跨场景迁移代价(越低越不依赖上下文)。

## 8. 评估协议纪律(写死在 summary.md 头部)
- 谁的 loss 消费了什么信号,谁就丧失对应镜像指标的 headline 资格
  (例:加 18D 回归头 ⇒ `mlp_action_r2` 降级为诊断)。
- 循环性是 (loss, 指标) 二元组属性,不是指标固有属性。

## 训练侧配套
- `train.py`:`training.seed` 可配置(set_seed + PK sampler);`data.seed`(split)恒为 42,
  保证各 seed 的 val 集完全一致。
- `dataset.py`:修复 3 通道 sam3_hoc mask 的 shape 检查 bug(旧代码对 [3,H,W] 走进非法
  5D interpolate);缺 mask 样本回退为全零 mask(= 均匀权重),使 collate 一致。
- 新配置:`ft_l40_ours_d_r2_fg_5k.yaml`(Ours-D = R2 λ=0.1 + SAM3 HOC 前景加权重建),
  `ft_l40_{full,ours_a_zero,ours_r2_opp_l01}_5k_s{1,2}.yaml`(seed 方差)。
- `run_ours.sh` 新入口:`d`、`seeds`。
- 注意:SAM3 HOC mask 覆盖率当时约 20%(16.6 万/约 80 万 framepair;部分 task 全覆盖、
  部分为 0)。长跑 Ours-D 前建议先扩覆盖:
  `conda run -n sam3 python code/precompute_sam3_hoc_masks.py --data_root /home/xuan/embodied-ai/data/egodex/test_240p --downsample_factors 1 2`
  (脚本自动跳过已有文件,可断点续跑)。覆盖不全时 Ours-D 仍可跑(缺 mask 样本均匀加权),
  但 L_emb 压力被稀释且与 task 相关——解读时要记得这个混淆。
