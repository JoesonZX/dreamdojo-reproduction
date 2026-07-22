# lam_condition_utilization（项目 B）

对动作信息**可得性（availability）**（某因素能否从 latent `z` 线性读出？）与
decoder **利用（utilization）**（它是否被 decoder 因果使用？）之别的诊断研究，
针对 DreamDojo/CD-LAM 的潜在动作模型。

核心结果——论文草稿 *"Availability is not Utilization: Diagnosing — and Removing —
a Training-Noise Bottleneck in the Latent-Action-to-Decoder Channel"*：方向信息大体上
是*可得的*却未被*利用*；一个 decoder-exposure 噪声设置（**α=0**）能恢复利用，而几乎不改变
encoder `μ`。覆盖：α=0/1 exposure、paired-latent、2×2 cross-decoding、Phase 0 / 0.3、
Tier-1.5 K 步/符号干预、Route-C GT18 vs zmu-A0/A1 独立 consumer，以及 Tier-2 预注册。

- **先读：** [`notes/README.md`](notes/README.md) → `notes/paper_draft.md`、`notes/stage_b_plan.md`。
- **完整记录：** [`PROJECT_MANIFEST.md`](PROJECT_MANIFEST.md)。
- **谱系：** A（[`../lam_cdlam_optimization`](../lam_cdlam_optimization)）→ **B** → C（[`../lam_world_model_control`](../lam_world_model_control)）。

## 对项目 A 的依赖
本项目**import A 的共享 LAM 底座**（`train/model/dataset/eval/benchmark_lam/
primitive_labels/dirprobe`）作为只读依赖。每个需要它的评测脚本都先 `import _apath` ——
见 [`code/_apath.py`](code/_apath.py)。它的 α/utilization LAM arm 通过 *A 的* `code/train.py`
训练。因此 B **不能独立运行**，必须有 `../lam_cdlam_optimization` 在场。见 `PROJECT_MANIFEST.md` §6。

## 目录结构
```
code/            B 专属诊断（consumer、cross-decode、h2、tier15、phase0、α、ood）+ _apath.py 垫片
code/config/     ft_l40_sb_*.yaml 的 Stage-B arm（喂给 A 的 train.py）
notes/           paper_draft.md、stage_b_*、tier2_*（索引见 notes/README.md）
results/         stage_b_*.json/.npz、consumer_*、alpha_*（权威诊断数据）
logs/            sb_*.log、stage_b_*.log
data/            eval_manifest.json（Eval-A）、eval_manifest_B.json（Eval-B）、consumer_exclude.json — 全部 FROZEN
checkpoints/     （逻辑归属）——物理存储是共享的 /checkpoints/lam-dis/ft_l40_sb_*（见 manifest）
```

## 快速上手（在本目录下运行）
```bash
# 训练一个 Stage-B / alpha arm（通过 A 的共享 trainer）
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --mixed_precision bf16 \
    ../lam_cdlam_optimization/code/train.py --config code/config/ft_l40_sb_A0.yaml
# Stage-B 确认性 eval  /  Route-C consumer
python code/eval_stage_b.py
python code/train_consumer.py --cond zmu --arm sb_A0 --seed_tag s0
```
