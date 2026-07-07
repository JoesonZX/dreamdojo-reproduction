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
