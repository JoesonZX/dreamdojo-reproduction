# Stage 1: LAM-Side Audit

## 目标

本阶段不训练新模型，先用 CD-LAM 风格的诊断指标审计已有 LAM。核心问题是：当前 latent action 的失败模式到底是一般场景偏置、零动作未校准，还是 push/pull、open/close 等反向动作混淆。

审计对象：

```text
raw_lam
kl_ft_l40_full_5k
contrastive_v3_5k
```

审计脚本：

```bash
/home/xuan/.venv/bin/python code/benchmark_lam.py \
  --runs raw_lam kl_ft_l40_full_5k contrastive_v3_5k \
  --n_samples 2000 \
  --perturb_samples 64 \
  --batch_size 16 \
  --num_workers 4 \
  --device cuda:1 \
  --out_dir results/lam_benchmark
```

`benchmark_lam.py` 是单进程单 GPU 审计脚本。默认 `--device auto` 会使用默认 CUDA device，通常是 `cuda:0`；如果 GPU 0 被占用，应显式传 `--device cuda:1`，或者用 `CUDA_VISIBLE_DEVICES=1` 把物理 GPU 1 映射成进程内的 `cuda:0`。

更稳的写法：

```bash
CUDA_VISIBLE_DEVICES=1 /home/xuan/.venv/bin/python code/benchmark_lam.py \
  --runs raw_lam kl_ft_l40_full_5k contrastive_v3_5k \
  --n_samples 2000 \
  --perturb_samples 64 \
  --batch_size 16 \
  --num_workers 4 \
  --device cuda:0 \
  --out_dir results/lam_benchmark
```

如果仍然 OOM，先降级不影响主审计结论的参数：

```bash
CUDA_VISIBLE_DEVICES=1 /home/xuan/.venv/bin/python code/benchmark_lam.py \
  --runs raw_lam kl_ft_l40_full_5k contrastive_v3_5k \
  --n_samples 1000 \
  --perturb_samples 16 \
  --batch_size 4 \
  --num_workers 4 \
  --device cuda:0 \
  --out_dir results/lam_benchmark
```

快速 smoke test：

```bash
/home/xuan/.venv/bin/python code/benchmark_lam.py \
  --runs raw_lam \
  --n_samples 8 \
  --perturb_samples 2 \
  --batch_size 2 \
  --num_workers 0 \
  --out_dir /tmp/lam_stage1_smoke
```

Smoke test 已完成，输出：

```text
/tmp/lam_stage1_smoke/summary.md
/tmp/lam_stage1_smoke/summary.csv
/tmp/lam_stage1_smoke/per_dim_raw_lam.csv
```

注意：`n_samples=8` 只用于验证代码路径，action R² 和 hard-negative 指标没有统计意义。

## 已有指标

`code/benchmark_lam.py` 原本已有：

- static-pair response：`E(o_t, o_t)` 是否接近 zero-action reference。
- latent health：effective rank、top eigen fraction、active units、KL。
- shortcut leakage：同 episode 不同动作 vs 跨 episode 同动作的距离关系。
- action R²：冻结 latent 后用 Ridge 预测 18 维真实动作。
- per-dim triage：每个 latent 维度的 action informativeness、decoder effect、label leakage。

## 本阶段新增指标

### 1. Camera Shift Response

对两帧同时做相同的 zero-filled 平移，再比较 shifted latent 和 normal latent 的相对距离：

```text
camera_shift_h_median
camera_shift_v_median
```

期望：越低越好。它测试视觉扰动是否被误当成 action。

### 2. Temporal Reverse Response

编码正向帧对 `(o_t, o_{t+k})` 和反向帧对 `(o_{t+k}, o_t)`：

```text
reverse_za_cos_median
reverse_ze_cos_median
reverse_za_delta_rel_median
reverse_ze_delta_rel_median
```

期望：

- `reverse_za_cos_median` 低：动作子空间对方向敏感。
- `reverse_ze_cos_median` 高：环境子空间对时间反转稳定。

raw LAM 没有 env 子空间，所以 `reverse_ze_*` 为 NaN。

### 3. Reversible Hard-Negative Geometry

用 verb 标签构造反向动作对：

```text
insert/remove
assemble/disassemble
stack/unstack
charge/uncharge
open/close
screw/unscrew
tie/untie
zip/unzip
fold/unfold
stock/unstock
pick/put
scoop/dump
lock/unlock
add/remove
push/pull
```

报告：

```text
rev_d_same_verb_diff_ep
rev_d_same_task_opposite
rev_d_any_opposite
rev_hard_margin
rev_nearest_pos_beats_opp
```

其中：

```text
rev_hard_margin = d(same-task opposite-verb) - d(same-verb diff-episode)
```

期望：`rev_hard_margin > 0`，即视觉/任务上下文相似但动作相反的 transition 应该比跨场景同动作 transition 更远。

## 当前解读方式

Stage 1 的核心不是追求单一总分，而是定位失败模式：

- 如果 `static_response_rel_median` 高，说明需要 `L_zero`。
- 如果 `camera_shift_*` 高，说明需要更强视觉扰动鲁棒性或 calibration。
- 如果 `reverse_za_cos_median` 高，说明动作子空间没有显式编码方向性，需要 stronger temporal reverse。
- 如果 `rev_hard_margin <= 0`，说明反向动作 hard negatives 没有分开，需要 hard-negative contrast。
- 如果 action R² 掉，说明 loss 损害了连续精细控制，需要保留或增强 `L_action_reg`。

## 2026-07-07 正式结果

输出位置：

```text
results/lam_benchmark/summary.md
results/lam_benchmark/summary.csv
results/lam_benchmark/per_dim_*.csv
```

核心表：

| run | action R2 | static rel ↓ | camera h/v ↓ | rev margin ↑ | rev z_a cos ↓ | rev z_e cos ↑ | rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw_lam | 0.120 | 0.673 | 0.382 / 0.342 | -0.005 | -0.068 | n/a | 26.09 |
| KL ft_l40_full_5k | **0.331** | **0.358** | **0.310 / 0.286** | **+0.020** | **-0.220** | 0.216 | 13.49 |
| contrastive_v3_5k | 0.270 | 0.476 | 0.555 / 0.428 | -0.046 | 0.203 | **0.992** | 15.76 |

结论：

1. **KL baseline 是当前最稳的主干。** 它的 action R2 最高、static response 最低、camera shift response 最低、reverse hard-negative margin 为正，并且正反帧对的 `z_a` cosine 最低。这说明 KL 版本虽然不该被叙述成“内容分离器”，但作为 action-reg + capacity control 的 fine-tune 主干非常强。
2. **contrastive v3 的问题被 Stage 1 明确抓出来了。** 它的 action separation ratio 最高，说明 verb-SupCon 确实让动作类别更聚；但 camera shift response 最高、`rev_hard_margin` 为负、`reverse_za_cos_median` 为正，说明它把同 verb 拉近了，却没有真正解决视觉扰动和反向动作分离。
3. **v3 的 env 子空间过于时间反转不变。** `reverse_ze_cos_median=0.992` 很好，但结合 `kl_mean_total=52.08` 和较大的 latent norm，看起来更像 contrastive 几何把 env 变成稳定标签/上下文缓存，而不是自然的低容量环境子空间。
4. **episode_shortcut_leakage 这次不能判定。** 该列为 NaN，因为采样池里 `same-episode different-action` 对不足。后续如果要严肃报告 shortcut leakage，需要改 sampler，专门抽同 episode 多 verb 或 reversible episodes。
5. **下一步不应继续单纯加 SupCon。** 现在的证据支持从 KL baseline 出发，先加 `L_zero` 和 stronger temporal reverse，再做 hard-negative contrast。普通 verb-SupCon 已经暴露出“动作类别聚类变好，但反向动作混淆更差”的副作用。

下一轮优先级：

1. **Ours-A = KL + `L_zero`**：目标是把 static rel 从 0.358 继续压低，同时保持 action R2 不掉。
2. **Ours-B = Ours-A + stronger reverse**：目标是保持/降低 `reverse_za_cos_median`，并提高 `rev_hard_margin`。
3. **Ours-C = Ours-B + phase-aware semantic hard-negative sigmoid contrast**：只针对 reversible pairs，不再使用普通 same-verb SupCon 作为主力。
4. **补一个 shortcut 专项评估**：构造同 episode 不同 verb 的采样池，修复 `episode_shortcut_leakage=NaN` 的指标空洞。

## 2026-07-07 Ours-C 设计修正

### 发现

人工查看 `audit_reversible_episodes.py` 挑出的高/中/低 `opposite_local_frac` episode 后，有三个关键现象：

1. 高 `opposite_local_frac` 的样本常常不是语义上的反向动作，而是安装/组装类任务中的阶段混合。完整视频包含从桌面拿物体、移动到装置、旋转安装、再离开桌面等阶段。
2. 中等样本里经常有复合行为，例如从抽屉里拿东西后又关上抽屉。一个 episode 里并不只有一个原子动作。
3. 仅用 translation 审计会低估旋转主导动作。assemble/install/screw/open/close 等任务的核心变化往往在 rotation，而不是平移。

因此，`opposite_local_frac` 更适合解释为“局部运动阶段混合/非单调”的数据诊断，而不能直接当作语义反向标签。

### 分析

普通 verb-SupCon 和旧版 action-aware hard-negative 都有弱点：

- verb-SupCon 把同 resolved verb 的 batch 样本整体拉近，但同一个 verb 里可能混有 approach、core manipulation、retreat，不一定都是同一种核心动作。
- 旧版 hard-negative 用 local action cosine 决定 positive/negative pair，假设相反语义动作的局部轨迹应当相反。但 drawer take/put、push/pull 等动作可能有相似手部轨迹，语义却相反。

这说明 action geometry 不应该负责定义语义正负样本；语义关系应由 resolved verb 和 reversible verb map 决定。local action 只适合估计当前 frame-pair 是否更像核心操作阶段。

### 决策

Ours-C 改为 phase-aware semantic hard-negative sigmoid contrast：

```text
positive pair:
same resolved reversible verb, different episode

negative pair:
opposite resolved verbs, preferably same task/object context

pair weight:
core_weight_i * core_weight_j
```

其中一个训练 sample 不是完整 episode，而是一个局部 frame-pair：

```text
sample_i = (frame_t, frame_t+skip)
```

一个 episode 会产生多个 sample。`core_weight_i` 是这个局部 transition 像不像核心动作的软权重，而不是数据筛选开关：

```text
core_weight = f(translation magnitude, rotation magnitude, temporal position)
```

所有 sample 仍然参与 reconstruction、KL、`L_action_reg`、`L_zero` 和 temporal reverse。只有 semantic hard-negative contrast 被软加权。这样保留了 approach/retreat 的泛化价值，同时降低它们对动作语义聚类的污染。

训练时需要关注 `train/hardneg_pairs` 和 `train/hardneg_neg_pairs`。如果它们长期接近 0，说明当前 batch 采样没有提供足够的 reversible hard negatives，需要进一步改 sampler，而不是直接判断 loss 无效。

## 2026-07-08 Ours-B2 决策

Ours-B 的 benchmark 显示 temporal reverse 能显著降低 `reverse_za_cos_median`，说明 action subspace 学到了更强的方向性；但同时 `camera_shift_h/v`、`task_shortcut_leakage` 和 `rev_hard_margin` 变差。一个合理解释是：原始 reverse loss 同时强迫 env subspace 在时间反转下保持稳定，鼓励 `z_e` 缓存 scene/task/context，从而把 context shortcut 重新带入 latent geometry。

因此新增：

```text
Ours-B2 = Ours-A + action-only temporal reverse
```

实现上保留 action reverse margin，设置：

```yaml
reverse_env_weight: 0.0
lambda_reverse: 0.05
```

期望：

```text
reverse_za_cos_median < Ours-A
camera_shift_h/v 接近 Ours-A
task_shortcut_leakage 接近 Ours-A
rev_hard_margin 不低于 Ours-A 太多
action_r2_full_latent 不下降
```

### 论文价值

这部分值得写进论文，因为它提供了完整的“现象 -> loss 设计 -> 实验验证”链条：

```text
现象：EgoDex video-level verb 是弱标签，episode 内存在阶段混合和旋转主导动作。
问题：直接 SupCon 会把同 verb 的非核心阶段也拉近；action-cosine mining 会误判轨迹相似但语义相反的动作。
方法：用语义定义正负样本，用 motion/phase 只做软权重。
验证：比较 Ours-B、旧式 C 或 SupCon baseline、新式 Ours-C 的 reverse margin、shortcut leakage、action R2 和 camera shift。
```

## Ours-A/B/C 运行命令

Configs:

```text
code/config/ft_l40_ours_a_zero_5k.yaml
code/config/ft_l40_ours_b_zero_reverse_5k.yaml
code/config/ft_l40_ours_c_zero_reverse_hardneg_5k.yaml
```

训练：

```bash
cd /home/xuan/embodied-ai/lam_disentangle
CUDA_VISIBLE_DEVICES=0,1 bash run_ours.sh a
CUDA_VISIBLE_DEVICES=0,1 bash run_ours.sh b
CUDA_VISIBLE_DEVICES=0,1 bash run_ours.sh c
```

如果只想用 1 张 GPU：

```bash
CUDA_VISIBLE_DEVICES=1 GPU_COUNT=1 bash run_ours.sh a
```

训练后跑完整 Stage-1 benchmark：

```bash
CUDA_VISIBLE_DEVICES=1 /home/xuan/.venv/bin/python code/benchmark_lam.py \
  --runs raw_lam kl_ft_l40_full_5k contrastive_v3_5k \
         ours_a_zero_5k ours_b_zero_reverse_5k ours_c_zero_reverse_hardneg_5k \
  --n_samples 2000 \
  --perturb_samples 64 \
  --batch_size 16 \
  --num_workers 4 \
  --device cuda:0 \
  --out_dir results/lam_benchmark
```

新增 shortcut 指标：

```text
task_shortcut_leakage = d(diff-task same-action) - d(same-task diff-action)
```

`task_shortcut_leakage > 0` 表示同 task/上下文的不同动作比跨 task 的同动作更近，即 task/context shortcut 仍然强。

## 2026-07-07 前景权重设计

根据 CD-LAM 的因果叙事，前景不应只理解为“所有会动的像素”。对 embodied manipulation 来说，真正和动作因果相关的区域至少有三层：

```text
hand: 执行动作的 agent 区域
object: 被操作、发生状态变化的 patient 区域
contact: hand-object 交互边界，通常是动作效果最集中的区域
```

因此，本阶段不再只使用单 prompt 的 `hands` mask，而是新增 SAM3 H/O/C mask：

```text
channel 0: hand mask, from SAM3 prompt "hands"
channel 1: object mask, from SAM3 prompt "objects being manipulated"
channel 2: contact mask, derived by dilated(hand) & dilated(object)
```

contact 不直接用文本 prompt 提取，因为“接触区域”不是稳定物体类别，SAM3 文本分割可能不可靠。用 hand/object 的膨胀重叠推导 contact 更可控，也更符合动作因果：只有手和物体空间上接近时才提高权重。

训练时 H/O/C mask 不参与语义正负样本定义，只用于 reconstruction loss 的像素权重：

```text
background: 1x
hand:       3x
object:     5x
contact:   10x
```

设计意图：

1. object 权重大于 hand，避免模型只重建手而忽略物体状态变化。
2. contact 权重最高，但区域很小，强调动作产生效果的位置。
3. background 仍然保留 1x，避免模型完全丢失场景上下文。

对应脚本和配置：

```bash
conda run -n sam3 python code/precompute_sam3_hoc_masks.py \
  --data_root /home/xuan/embodied-ai/data/egodex/test_240p \
  --downsample_factors 1 2

CUDA_VISIBLE_DEVICES=1 GPU_COUNT=1 bash run_ours.sh c-fg
```

这应该作为 Ours-C 的后续 ablation，而不是直接替代 Ours-C：

```text
Ours-C:    semantic hard-negative + soft core weighting
Ours-C+FG: Ours-C + hand/object/contact reconstruction weighting
```

如果 Ours-C+FG 提升 camera shift、static response 或 object-state reconstruction，但 action R2 明显下降，说明前景权重过强，需要降低 `contact` 或 `object` 权重。
