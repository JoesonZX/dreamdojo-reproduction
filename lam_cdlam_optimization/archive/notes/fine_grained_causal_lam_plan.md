# Fine-Grained Causal LAM 实验规划书

## 1. 研究目标

当前 LAM 能从两帧视频中提取 latent action，但这个 latent action 容易混入场景、外观和 episode shortcut；同时在 push/pull、open/close、insert/remove 这类视觉相似但方向相反的动作上分离不够。我的目标不是简单复现某一篇方法，而是先把已有 LAM 在当前数据和任务中的失败模式审计清楚，再针对这些失败模式设计 loss 和实验。

最终希望得到一个更适合双手精细控制的 latent action model，使动作子空间 `z_a` 同时满足：

- 连续动作保真：能线性预测 EgoDex 的 18 维双手 SE(3) 相对动作。
- 反向动作可分：push/pull、open/close、insert/remove 等 hard negatives 在 latent 中明显分开。
- 跨场景动作一致：同一动作跨 episode 更近，而不是同一 episode 的不同动作更近。
- 零动作校准：`E(o_t, o_t)` 接近 zero-action latent。
- 可控性有效：固定场景替换或缩放动作 latent 后，生成/预测运动跟随动作变化。

## 2. 核心研究逻辑

本项目遵循如下路线：

```text
复现 CD-LAM 的 LAM-side 诊断逻辑
→ 审计 raw LAM / KL LAM / ConLA-style LAM 的失败模式
→ 总结当前任务中特有的问题：反向动作混淆 + 精细控制不足
→ 针对问题设计 loss
→ 做 ablation 和 hard-negative 实验
→ 再从 ConLA / V-JEPA / DINO / LAPA / villa-X / SigLIP 等方法吸收组件
```

这里不优先完整复现 CD-LAM 的 ACWM Stage 2/3，而是优先复现它的 Stage 1 LAM-side audit，因为当前贡献点在 latent action 本身。

## 3. 相关方法的作用

| 方法 | 对本项目的启发 | 本阶段处理方式 |
|---|---|---|
| CD-LAM | 提供“现象-诊断-loss-结果”的主线：zero response、shortcut leakage、action-centric contrast、latent calibration | 复现诊断和关键 loss，不完整复现 ACWM |
| ConLA | 启发 action-centric SupCon、temporal reverse，用动作结构替代 KL 分离 | 部分复现，重点改造成 hard-negative contrast |
| V-JEPA | 启发 latent-space prediction，而不是 pixel reconstruction | 暂不复现，作为后续 world-model/control 方向 |
| DINO / DINOv2 | 启发 teacher-student、view-invariant representation、collapse prevention | 作为表示学习参考，不作为主线 |
| LAPA / LAPO | 启发 latent action pretraining：先学 latent action，再对接 robot action | 作为下游 VLA/动作桥接参考 |
| villa-X | 启发如何把 latent action 接入 VLA pretraining | 后续系统实验参考 |
| SigLIP / SigCLR-style loss | 启发 pairwise sigmoid contrast，不依赖 batch softmax，适合小 batch hard positive/negative | 可替代 SupCon 做实验 |

## 4. Stage 1：LAM-Side Audit

目标是建立本项目自己的 “Table I”，用同一套指标审计已有模型和新模型。

评估对象：

```text
raw LAM_400k
KL ft_l40_full_5k
contrastive v3
new loss variants
```

核心指标：

| 指标 | 发现的问题 | 对应的后续 loss |
|---|---|---|
| static response: `E(o_t, o_t)` norm | 静止帧仍被编码成动作 | `L_zero` |
| camera/crop shift response | 视觉扰动被误当动作 | calibration / robustness |
| shortcut leakage | 同 episode 不同动作过近 | hard negative contrast |
| action R² / per-dim R² | 精细动作信息不足 | `L_action_reg` |
| reverse consistency | pull/push 等方向不分 | `L_reverse` |
| verb/action cluster | 动作语义结构弱 | contrastive loss |
| PSNR / FG-PSNR | 重建是否被破坏 | recon loss |

Stage 1 的判定标准不是“谁所有指标最好”，而是定位不同方法的失败模式。例如当前已有结果显示：KL baseline 的连续动作 R² 更强，而 verb-SupCon v3 的动作聚类更强，但 episode 子簇和 static response 仍有问题。

## 5. Stage 2：主 Loss 设计

建议主方法先暂定为 Reverse-Calibrated Fine-Grained LAM。

总 loss：

```text
L_total =
  L_recon_fg
+ lambda_act  L_action_reg
+ lambda_zero L_zero
+ lambda_rev  L_reverse
+ lambda_ctr  L_hard_contrast
+ lambda_kl   L_KL_freebits
```

各项职责：

- `L_recon_fg`：保持下一帧重建，但让手和物体等前景区域权重大于背景。
- `L_action_reg`：保留 18 维动作监督，这是连续精细控制的锚。
- `L_zero`：编码重复帧 `(o_t, o_t)`，要求 zero-transition latent 接近零参考，但不压塌普通 transition。
- `L_reverse`：编码反向帧对 `(o_{t+k}, o_t)`，要求动作方向相反、环境子空间稳定。
- `L_hard_contrast`：用 hard positive / hard negative 替代普通 verb-SupCon，重点区分视觉相似但动作相反的 transition。
- `L_KL_freebits`：KL 只作为 capacity control，不再作为内容分离器。

`L_zero` 形式：

```text
z0 = E(o_t, o_t)
L_zero = relu(||z0|| / stopgrad(rms(||z_delta||)) - margin)^2
```

`L_reverse` 形式：

```text
L_reverse =
  ||P(z_a_forward) + P(z_a_reverse)||^2
+ 1 - cos(z_e_forward, z_e_reverse)
+ MSE(pred_action_reverse, real_reverse_action)
```

注意：`real_reverse_action` 应该从 HDF5 按反向帧对重新计算，不建议简单对 18 维动作取负，因为 6D rotation 不能直接取负。

`L_hard_contrast` 的 pair 构造：

- positive：同 canonical primitive、动作方向相近、不同 episode。
- hard negative：同场景或同 task，但动作相反，例如 push/pull、open/close、insert/remove。
- same-episode positive 降权，避免 episode 子簇被进一步压紧。

## 6. Stage 3：实验矩阵

主实验：

| Run | 目的 |
|---|---|
| raw LAM | 原始基线 |
| KL ft_l40_full_5k | 当前连续 R² 最强基线 |
| contrastive v3 | 当前动作聚类最强基线 |
| Ours-A: KL + `L_zero` | 验证 zero calibration |
| Ours-B: A + `L_reverse` | 验证反向动作分离 |
| Ours-C: B + hard contrast | 验证 hard negative 动作结构 |
| Ours-D: C + foreground recon | 验证前景控制和重建 |

执行策略：

- 先用 1k 或 2k 步筛选 loss 是否方向正确。
- 对最好的 2 个配置跑 5k。
- 每个新 loss 必须有对应指标改善，否则不进入下一轮。

## 7. Stage 4：Reversible Action Benchmark

专门构造 reversible action benchmark：

```text
push vs pull
open vs close
insert vs remove
pick up vs put down
turn on vs turn off
```

报告指标：

- pairwise classification accuracy
- confusion matrix
- nearest-neighbor retrieval
- same-action diff-episode distance
- same-episode opposite-action distance
- reverse latent cosine
- action R² by translation / rotation / left hand / right hand

这组实验是区别于 CD-LAM 的关键：CD-LAM 更强调一般 action-irrelevant confounding，本项目强调 fine-grained reversible action disentanglement。

## 8. Stage 5：可控性实验

如果暂时不接完整 ACWM，可以先做轻量替代：

1. 固定首帧 `o_t`。
2. 取两个 latent，例如 `z_push` 和 `z_pull`。
3. 用 LAM decoder 解码 `D(o_t, z)`。
4. 比较前景光流、手部位移方向、物体位移方向。

最终最好做下游 world model 或动作桥接评估，但可以作为最后阶段。

## 9. 预期论文故事

```text
已有 CD-LAM 发现 LAM 中存在 action-irrelevant confounding。
但在双手精细控制场景中，我们进一步发现：
当前 latent action 的主要瓶颈不是一般场景偏置，
而是 reversible primitives 和 continuous control 的耦合混淆。

为此，我们提出 reverse-calibrated fine-grained latent action learning：
用 zero calibration 消除静止动作偏置，
用 reverse consistency 显式建模方向性，
用 hard-negative contrast 分离视觉相似但动作相反的 transitions，
同时保留 continuous action regression 来维持精细控制。

实验显示，该方法在保持或提升 18D action R² 的同时，
显著改善 push/pull、open/close 等 hard negative 分离，
并降低 zero/static response 与 episode shortcut leakage。
```

## 10. 最小执行计划

当前优先级：

1. 补齐 CD-LAM-style audit：static response、camera shift、shortcut leakage、hard-negative retrieval。
2. 先加 `L_zero`，跑 1k/5k，对比 KL baseline。
3. 再加 stronger temporal reverse，重点看 push/pull、open/close。
4. 最后把 v3 SupCon 改成 hard-negative sigmoid contrast。
5. 只有当前三步有效后，再考虑 V-JEPA/DINO/LAPA/villa-X 的更大系统启发。

