# Stage-0 报告(T9,2026-07-20 草稿 — idm 训练中、ours_d 等 mask 覆盖)

数据源:`results/lam_benchmark_stage0`(6 行,当日代码)、`results/lam_benchmark_anchor`
(官方锚点)、`results/lam_benchmark_seed42` + `results/lam_benchmark_seeds`(九行同码
seed 方差)。指标语义见 `benchmark_v2_changes.md`;概念见 `concepts_and_pipeline.md`;
CD-LAM 对照见 `cdlam_repro_fidelity.md`。**全部数字出自同一版 benchmark 代码**
(与 v2 的复现性 delta = +0.0000,见 `seed_variance.md`)。

---

## 1. 失败清单表(核心交付)

| 失败模式 | 指标 | raw_lam | **cdlam官方** | **cdlam复现** | kl_full | ours_a | ours_r2 | kl_**lf** | r2_**lf** |
|---|---|---|---|---|---|---|---|---|---|
| ①方向未编码 | direction_gain | −0.211 | **−0.241** | **−0.287** | −0.249 | −0.266 | −0.250 | (−0.134)* | (−0.153)* |
| ②外观在 z_a | opp_cls_static | 0.960 | 0.978 | **0.994** | 0.964 | 0.975 | 0.963 | 0.981 | 0.987 |
| ②外观在 z_a | camera_shift_h | 0.380 | 0.442 | **0.237** | 0.310 | 0.295 | 0.307 | — | — |
| ③路由泄漏 | R²_ze | n/a | n/a | 0.140 | 0.158 | 0.148 | 0.158 | (0.032)* | (0.018)* |
| ④旋转弱 | R²_rot / R²_trans | .08/.21 | .08/.23 | .30/.54 | .27/.52 | .29/.54 | .33/.55 | * | * |
| (参照)校准 | static_response | 0.672 | **0.067** | 0.134 | 0.356 | **0.035** | 0.047 | 0.770 | 0.214 |
| (参照)保真 | mlp_action_r2 | 0.093ᵉ | 0.104ᵉ | 0.359 | 0.308 | 0.326 | 0.365 | −0.002ᵉ | 0.024ᵉ |
| (参照)塌缩 | eff_rank / flag | 26.1/T† | 24.9/F | 12.7/F | 13.4/F | 13.0/F | 14.1/F | **2.8/T** | **6.7/T** |

ᵉ = 该模型无 18D 监督,R² 为完全外部指标(可跨比);其余带监督,R² 只做家族内受控比较。
\* = lf 双子已塌缩,其行内数字是塌缩伪影(外观仍可分而运动近零),不作证据。
† raw_lam 的 collapse_flag=True 是因 R²<0.1 的门槛(它无监督),非真塌缩(rank 26)。

> ⚠️ **失败①的表述已于 2026-07-20 被条件 probe 推翻并重写**(见 §4c);
> **失败②的证据强度已下调**(旧 opp_cls 协议含 episode 泄漏,见 `dirprobe_split_issue.md`)。
> 本表的 `direction_gain` 行现按 `static_shortcut_gap` 读(泄漏诊断,非方向证据)。
> **③④ 不受影响**——它们回归 18D 连续动作、按样本切分,与 verb 标签/episode 分组无关。

**四大失败全部幸存(①②表述已修正),且①④有官方 checkpoint 级证据:**
- **①方向**:三臂闭合——论文 12 类配方复现 **−0.287**(全场最负,static 0.994:合并标签
  + pctr 把相反动作主动拉近,构造性预测命中)/ 官方 checkpoint **−0.241**(同 base
  before/after)/ 我们全家族 −0.24~−0.28(九 seed range ≤0.024)。
- **②外观**:静态对照 0.96–0.99 全场;camera_shift 上 **cdlam_repro 的 0.237 是全场最低**
  (L_emb 前景加权的可见效果,Tier-0 机制在动)——但仍远未解决。
- **③路由**:R²_ze 0.14–0.16,带监督家族全漏,无一例外。
- **④旋转**:rot/trans ≈ 0.3/0.55,全场一致;官方 CD-LAM 也是 0.08/0.23。

## 2. 阶段-0 两大新发现(超出原失败清单)

### 2a. lf 孪生双塌缩:18D 监督是 β_a=3e-3 瓶颈下唯一的防塌锚
kl_lf:rank 2.8/40、R²=−0.002、zeroact/swap/ctrl≈0(解码器完全无视 z_a)。
r2_lf:L_zero+margin 减缓(rank 6.7、static 0.21——L_zero 仍在工作)但未阻止
(ctrl 0.004)。机制 = "KL 瓶颈只删动作 loss 不保护的信息"的极端形式:去掉监督,
无人为 z_a 付 KL 的钱。raw_lam 不塌因为 DreamDojo 原始 β=1e-6 无瓶颈;CD-LAM
action-free 却安全因为 **free-bits 地板 + stop 门槛**(其配方里 `eff_rank_drop_frac_max
0.20`、`kl_per_dim_mean_min 0.05` 正是防此)。
⇒ **故事 (b) 的 label-free 主线需要重建 trunk**(free-bits 或 β_a≈0),
"λ_action=0 消融"不再是免费的。

### 2b. T8:ours_r2 对 kl_full 无任何可辩护的 headline 优势
r² +0.016/噪声 0.055、gain +0.017/0.024、swap −0.032/0.167 —— 全在 seed range 内。
**R2 改判为"纯语义 margin"消融臂**:它的三项零改善 + scene_knn 回升(0.176 vs
ours_a 0.117)正是"语义门单独不够"的实证,直接 motivate L_dir 的运动学门。

## 3. 各 run 的 Tier-0 / Tier-1 判定(一句话)

| run | Tier-0(机制) | Tier-1(裁决) |
|---|---|---|
| cdlam_repro | ✅ pctr 2.58→0.5、pair 非零、fg 生效(camera 最低) | 方向失败如构造性预测(−0.287);L_pctr 价值按纪律**不判死刑**(batch 64 vs 3072) |
| kl_lf / r2_lf | ✅ 训练收敛、无 act 项 | ❌ 双塌缩 → 故事(b) trunk 需重建(本身是量化发现) |
| cdlam_official | ✅ 839/839 加载、同 base 证实(L2 0.0015) | debias 真(0.067)、方向假(−0.241)、zeroact 5.04(其 300 步的去向) |
| idm(训练中) | mse 无下降压力 ✓;act 下降**边缘**(1k 步 ~1.0 vs 监督 ~0.7) | 待 benchmark;若 R² 非最高,2×2 缺角如实写 |
| ours_d | 未训(等 SAM3 覆盖) | — |

## 4. 故事对账(定稿视角)

一句话贡献不变:"CD-LAM 修正一般 confounding 之后,运动方向仍是条件通道中未编码的
自由度;我们用真实数据的运动学反平行对将其注入,并以非循环协议证明。"

- **第一幕(诊断)**:超预期完成。三臂方向证据 + 官方 before/after(static 0.672→0.067、
  锥 0.209→0.058)与 direction_gain 不动的对照 = "混淆修正 ≠ 条件通道完整"。
- **R2 胜点**:死于 T8;转岗为语义-only 消融,反而为 L_dir 铺路。
- **第二幕(L_dir)**:未开始,全部筹码所在。前置(L_emb/ours_d)等 mask 覆盖。
- **故事 (b)**:受 2a 冲击,需决策(见 §5)。
- **分析章素材已备**:削锥≠去场景(0.058 锥下 scene_knn 23× null)、L_zero 去场景/
  语义 margin 回吸场景(ours_a 0.117 vs kl 0.181 vs r2 0.176,超噪声 ~1.5×)、
  低秩(rank 13–14 + 中心化后余弦 0.10)、camera_shift 协议敏感性、IDM 2×2(待)。

## 4b. IDM oracle 结果(T6 完成,`results/lam_benchmark_stage0_idm`)

| 指标 | idm | kl_full | ours_r2 | cdlam_repro | 2×2 预期 |
|---|---|---|---|---|---|
| mlp_action_r2 | 0.332 | 0.308 | **0.365** | 0.359 | ❌ 预期最高,实际第 3 |
| **ctrl_delta_psnr** | **0.397** | 1.115 | 1.220 | 1.104 | ✅ **全场最差(可控性最低)** |
| swap_context_cost | **0.397** | 1.008 | 0.923 | 0.760 | 迁移代价最低(动作码最不依赖上下文) |
| swap_opp_delta | **0.608** | 0.384 | 0.262 | 0.409 | ❌ 预期最差,实际最高 |
| zeroact_suppression | **9.411** | 0.639 | 1.153 | 1.050 | 置零响应极端 |
| **R²_ze** | **0.012** | 0.158 | 0.158 | 0.140 | 💥 **路由泄漏几乎消失(全场最低 10×)** |
| direction_gain | −0.228 | −0.249 | −0.250 | −0.287 | 与全家族无异 |

**判定:2×2 只成立一半,但换来一个更有价值的发现。**
- ✅ **"可控性最差"成立**:`ctrl_delta 0.397` 全场最低(把 z_a 跨 batch 打乱,解码
  几乎不变)——去掉重建压力后,latent 与解码器的因果耦合确实断了;
- ❌ **"R² 最高"不成立**(0.332,低于 ours_r2/cdlam_repro):说明**重建压力并不与
  动作可解码性竞争**——恰恰相反,没有重建做正则,表征漂移反而略伤 R²
  (训练日志佐证:act loss 停在 ~0.7,高于同期监督 run 的 ~0.6);
- 💥 **意外发现:R²_ze = 0.012,比全家族低一个数量级**。失败③(双向路由泄漏)
  在 IDM 下几乎消失 ⇒ **z_e 的动作泄漏主要由重建压力驱动**(z_e 为了帮重建而
  偷载运动信息),不是 KL 分区不力。这是失败③的**机制定位**,比原 2×2 更有价值。
- ⚠️ `zeroact 9.41` / `swap_opp 0.608` 极端值需谨慎解读:解码器几乎不依赖 z_a 时,
  这些 PSNR 差值的分母结构改变,可能是**退化伪影**而非真实可控性。列为诊断,不进主表。

## 4c. 💥 条件方向 probe 重写了失败①(阶段 A1,2026-07-20)

旧指标 `direction_gain` 有两个致命缺陷(天花板效应 + episode 泄漏,见
`concepts_and_pipeline.md` §G2' 与 `dirprobe_split_issue.md`)。换成条件 probe
(baseline `[Z_static;0]` vs full `[Z_static;Z_trans−Z_static]`,按 episode 分组 CV,
held-out NLL,相对置换 null 判读)后,15 个 checkpoint 的结果:

| | vs_null 均值 | seed range | ΔAUC |
|---|---|---|---|
| kl_full | +0.0592 | 0.0270 | +0.002~+0.006 |
| ours_a | +0.0416 | 0.0243 | −0.003~+0.004 |
| ours_r2 | +0.0697 | 0.0457 | −0.008~+0.007 |
| **raw_lam(未在 EgoDex 训练)** | **+0.0774** | — | +0.0095 |
| **cdlam_official** | **+0.0811** | — | +0.0014 |

**结论(三点)**:
1. **"方向未编码"被推翻**:12/15 的 p≤0.05,痕量条件方向信息普遍存在;
2. **但它与训练无关**:`raw_lam` 排 4/15,**高于我们三个家族的均值**
   ⇒ 这是任意非退化 transition encoder 天然携带的(运动本身与 verb 相关),
   不是任何 loss 的功劳;CD-LAM 的去偏同样没提高它;
3. **量级微不足道**:ΔAUC 全表 −0.008~+0.017,过半为负;家族间差异
   (r2−kl=+0.011)远小于 seed range(0.024–0.046)。

⇒ **主张改写**:缺口不在"方向不在 latent 里",而在
**"没有任何训练方法增加它,且 decoder 不使用它"**(decoder 侧证据不变:
swap_opp_delta 的 seed range 0.12–0.17 覆盖全部家族差异)。
⇒ **L_dir 的目标随之改变**:从"注入方向信息"变为"**放大到可用量级 + 打通因果通路**"。

## 5. 待决策(用户)

1. **lf trunk 重建方案**:(a) split-KL + 每维 free-bits 地板(推荐:同治已实测的低秩,
   CD-LAM 验证过的 action-free 生存机制,保留 split 接口);(b) β_a 降回 ~1e-6 +
   靠自监督 loss 群塑形。任选都需 1–2 个 5k run。
2. **故事 (b) 的定位**:lf 主线是否值得重建成本?备选:主线保留监督但把"18D 监督
   = 防塌锚"本身写成发现(supervision-as-scaffold),lf 塌缩实验做量化证据,
   L_dir 仍按 label-free 设计(它不依赖 trunk 的监督)。
3. **T5 时机**:SAM3 覆盖率(另一台机)现在多少?>80% 即可启动 ours_d。
4. **L_dir 配对离线审计**(回答双门通过率):flow/关键点挖 episode 内反平行对,
   报告 per-episode 通过率。CPU+轻 GPU,可先行。

## 5b. ⚠️ 训练卫生发现:全家族的 LR 调度是 V 形(2026-07-20 确认)

`accelerator.prepare(scheduler)` 在 2 卡下每个全局 step 推进底层 scheduler **2 步**,
而 `lr_lambda` 的余弦对 progress>1 不截断 ⇒ 实际调度:**lr 在 step 2500 降到 0,
然后余弦回升,step 5000 回到 ~满 lr(9.96e-6)**。日志证据:idm/cdlam 在 2500 处
lr=0.00e+00、cdlam 在 5000 处 lr=9.96e-06(2 倍速预测值完全吻合)。

影响评估:
- **家族内受控比较不受损**:所有 2 卡 5k run 共享同一(怪)调度,差异归因仍成立;
- 例外 **a_s1(1 卡 regime)**:调度是正确的单调衰减——它在 T8 里是 ours_a 家族最高值
  (0.3555),可能部分是调度差异而非 seed;regime 标注已覆盖此点;
- 终点 checkpoint 保存在**满 lr 时刻**,末段训练噪声偏大(cdlam 末段 act 波动与此一致)。

**决定**:stage-0 内(含 T5 ours_d)**保留现状**以维持家族可比;
**stage-1 trunk 重建时一并修复**(总步数乘 num_processes 或调度按全局步推进),
并趁重建把 kl/a 基线在正确调度下重跑——反正 trunk 换 free-bits 也要重跑,零边际成本。

## 6. 复现命令与数据位置

- 全表:`results/lam_benchmark_stage0/summary.{csv,md}`(6 行合并,含 part2)
- 锚点:`results/lam_benchmark_anchor`;seed 方差:`notes/seed_variance.md`
- idm 训完后补行:`benchmark_lam.py --runs idm_5k --out_dir results/lam_benchmark_stage0_idm`
  然后并入本表(勿覆盖既有目录)。
