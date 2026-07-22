# 历史实验打捞的长效事实与教训(2026-07-19 整理)

`notes/archive/` 里的 12 份文档记录的**模型排名与推荐已全部作废**
(对应 checkpoint 多数已删,结论被 benchmark v2 推翻)。
但其中若干**与模型排名无关的事实**至今有效,集中收录于此,免得被连带丢弃。
每条注明出处,细节回查 `archive/<文件>`。

---

## 1. 数据与标签(永久事实)

**18D 动作的物理含义**(`archive/analysis-recon-perdim.md` Q2;布局已在
`code/benchmark_lam.py::_action_r2_groups` 固化):
```
[0:3]   左手平移 dp        [3:9]   左手 6D 旋转
[9:12]  右手平移 dp        [12:18] 右手 6D 旋转
```
- 是**手腕/末端位姿**,不是关节角;手指被**故意排除**;
- 运动表达在手自身坐标系:`inv(T_t) · T_{t+skip}`;
- 四条经验规律(v2 的 `action_r2_trans/rot/left/right` 就是在测它们):
  **平移 ≫ 旋转、右手 ≫ 左手、深度轴最低、最后一个 6D 列最低(~0.074)**。
- ⇒ "R² 只有 0.14 很低"往往是**线性 probe 读 6D 旋转本来就难**,必须看分组 R²。

**EgoDex 数据管线**(`archive/preparation.md`,已用 h5py 核实):
- HDF5 `transforms/<joint>` 为 `[T,4,4]`,与视频帧 **1:1 对齐**,69 个关节;
- **AgiBot 盘上没有 proprioception** ⇒ 当前只用 EgoDex(迁移集要另外找数据);
- `action_stats_18d.npz` 是确定性归一化缓存,train/eval **共用同一份**;
- 正确性不变量:动作必须在 `__getitem__` 里 **skip/t 裁剪之后**再计算;
- 索引规模:**train 739,559 / val ~80,509** 帧对(与当前训练日志一致)。

**可逆动词对**(`archive/stage1_lam_side_audit.md`;现固化在
`code/train.py::_opposite_verb_map`,16 对):insert/remove、screw/unscrew、
open/close、zip/unzip、fold/unfold、push/pull、stack/unstack…… `opp_cls` 仍在用。

**SAM3 mask 的已知失效 task**(`archive/CURRENT_HANDOFF_2026_07_11.md`):
`add_remove_lid`、`arrange_topple_dominoes` 等 task 的 **object/contact 通道经常为空**。
⇒ L_emb 的压力在这些 task 上被稀释,且**稀释与 task 相关**——解读 Ours-D 时必须
记住这个混淆。前景通道权重设计:背景 1× / hand 3× / object 5× / contact 10×。

---

## 2. 预训练 checkpoint 溯源(载入逻辑的依据)

`archive/results-v2-finetune.md` §2,已核实:
- 来源 **HuggingFace `nvidia/DreamDojo` → `LAM_400k.ckpt`**,8.52 GB,400k 步,Lightning ckpt;
- ckpt 自带 hparams(权威):`model_dim=1024, latent_dim=32, enc/dec_blocks=24,
  num_heads=16, patch_size=16, β=1e-6`;
- key 前缀 `lam.`,剥离后与 `model.py` 命名对齐;
- **分辨率无关**(patchify + RoPE,无固定尺寸位置嵌入)⇒ 直接吃 240×320,无需重转 256;
- 载入行为:latent=32 灌入 839/843 张量;**latent=40 会跳过 3 个扩维张量**
  (`fc` 64→80、`action_up` 32→40)并随机初始化——每次训练日志都能看到,是**正常现象**。

---

## 3. 仍然有效的方法论决定

| 决定 | 证据 | 出处 |
|---|---|---|
| **全微调,绝不冻结 encoder** | 冻结 1k 步 R²=0.128 ≈ 0 步锚点 0.127 ⇒ 动作监督梯度进不去 encoder | `archive/results-v2-finetune.md` |
| **5k 步,不是 1k** | 1k 步未追平从零 40k;5k 步 R² 0.346 反超 | 同上 |
| **小 lr(1e-5)** | 防灾难性遗忘 | 同上 |
| **高重建 ≠ 好动作** | `ft_l32_frozen` PSNR 38.6(最高)而 R² 0.128(最低)——正因没动 encoder。**PSNR 不是解耦指标** | `archive/analysis-recon-perdim.md` Q1 |
| **不要用 z_a→ep% vs z_e→ep% 论证纯度** | 32 维 vs 8 维不是公平比较;纯度指**z_a 内部**动作 ≫ 场景 ⇒ 这是路由三元组读法的前身 | `archive/ppt-v2-finetune.md` §5 |
| **训练/benchmark 不同机** | CPU 视频解码互拖 10× | `throughput-debug.md`(仍在 live) |

---

## 4. decorrelation 失败的完整教训(若 stage-1 要走这条线,先读)

`archive/exp3-analysis.md`。exp3 = KL trunk + 去相关(λ=0.02,无 warmup)⇒ **整体塌缩**:
动作子空间 R² 0.133 → **−0.002**,动作 task 分类 51.1% → 10.8%(≈随机)。

**根因**:去相关的**廉价解**是"压低某子空间的方差",而不是"各自保留有用且互不相关的信息"。
动作监督还没站稳时,优化器直接走了这条捷径。训练日志证据:early step indep=50,
×λ0.02 ≈ 1.0,与动作 loss(~1.15)同量级 ⇒ 表征刚成形就被去相关梯度主导;
动作 loss 全程停在 1.15(= 预测均值)。

**修复与残留代价**:加 `indep_warmup`(λ 线性升)后,λ=0.002+warmup2k 使动作 task
回到 29.5%、环境 task 降到 7.5%(exp2 为 19%)——**不塌了,但动作保真仍不及对照**。
因为去相关移除的是两子空间的**共享/相关**成分,不是"搬运"。

**若要重启这条线,三件事缺一不可**:
1. **VICReg 方差保持项** `mean(relu(1 − std(z)))`,杜绝"压方差作弊降相关";
2. **归一化**:当前是 32×8=256 项平方**和**,量级随维度膨胀 ⇒ 改 `.mean()`;
3. **warmup**,让动作子空间先成形。

**一句话结论**:**主动机制(KL 瓶颈)优于被动惩罚(去相关)**——瓶颈"只删动作 loss
不保护的信息",天然带方向且先验方差为 1 不会塌成 0;去相关**没有指派信息归属**。
⇒ 这是 `concepts_and_pipeline.md` §F 里 decorrelation 排在 L_emb 之后的原因。

---

## 5. 相关工作映射(论文 related-work 直接可用)

`archive/fine_grained_causal_lam_plan.md` §3:

| 方法 | 提供什么 | 我们怎么用 |
|---|---|---|
| **CD-LAM** | "现象-诊断-loss-结果"主线:zero response、shortcut leakage、action-centric contrast、latent calibration | 复现其 Stage-1 诊断与关键 loss,**不**完整复现 ACWM(见 `cdlam_repro_fidelity.md`) |
| **ConLA** | action-centric SupCon、temporal reverse,用动作结构替代 KL 分离 | 部分复现;改造成 hard-negative contrast |
| **V-JEPA** | latent-space prediction 取代像素重建 | 暂不复现;下一代主干候选 |
| **DINO/DINOv2** | teacher-student、view-invariant、防塌缩 | 表示学习参考 |
| **LAPA/LAPO** | 先学 latent action 再对接 robot action | 下游 VLA 桥接参考 |
| **villa-X** | latent action 接入 VLA pretraining | 后续系统实验参考 |
| **SigLIP** | pairwise sigmoid contrast,不依赖 batch softmax,**适合小 batch** | 已用于 `L_pctr`(CD-LAM Eq.10) |

**⚠️ 一个早期的预见性警告(值得在论文里提)**:相反动作的标签必须**从 HDF5 重新计算**
`inv(T_t)·T_{t+skip}` 再取负,**不能直接对 6D 旋转表示取负**——这是 Ours-C 的坑,
也是 `concepts_and_pipeline.md` §D "twist 取负 ≠ 6D 取负"论证的源头。

---

## 6. 可视化资产与教训

图在 **`results/figs/`**(仍在盘上):`fig1_r2_progression.png`、`figA_loss.png`、
`figB_perdim_r2_mse.png`、`figC_perturb_diff.png`、`figD_tsne_motion.png`、
`figE_pred_vs_true.png`、`fig3_purity.png`。索引出处 `archive/ppt-v2-finetune.md`。

两条仍然成立的教训:
- **静态扰动网格不可读** ⇒ 用 ±3σ + |diff| 热图;
- **t-SNE 看不出结构是正常的**,因为 latent action 是**连续量**不是类别 ⇒
  用 pred-vs-true 散点代替。

---

## 7. 归档清单与作废原因

| 归档文件 | 作废原因 | 打捞进本文档的内容 |
|---|---|---|
| `results-v1.md` | 从零训练 exp1–4 + 700M 全程;checkpoint 已删,非对称 KL 推荐已否 | 规模/欠训练记录(未收录,回查原文) |
| `design.md` | 从零四实验设计,已改为 LAM_400k 微调;目录结构已不符 | — |
| `ppt-summary.md` | 已退役故事的讲稿;数字是 results-v1 的子集 | — |
| `preparation.md` | 实验计划作废 | §1 数据事实 |
| `results-v2-finetune.md` | 数值多数来自已删 checkpoint | §2 溯源、§3 决定 |
| `HANDOFF.md` | ~80% 与 analysis-recon-perdim 重复;v3 结论已推翻 | 代码改动表(回查原文) |
| `lam-benchmark.md` | v1 spec,已被 benchmark_v2_changes 取代;其命令违反"不覆盖 v2"纪律 | per-dim CSV 列表(回查原文) |
| `exp3-analysis.md` | exp3 checkpoint 已删 | **§4 全部** |
| `fine_grained_causal_lam_plan.md` | Stage 2/3/4 计划已执行且被自身结果推翻 | **§5 全部** |
| `stage1_lam_side_audit.md` | 解释层全部建立在 Tier-0 镜像指标上(已失格) | §1 动词对、相位混合审计(→ `concepts_and_pipeline.md` §D) |
| `CURRENT_HANDOFF_2026_07_11.md` | 表格是 pre-v2;Ours-A/B/B2 判定已作废 | §1 SAM3 失效 task |
| `ppt-v2-finetune.md` | headline 故事已退役 | §3 纯度规则、§6 图索引 |
