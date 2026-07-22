# lam_cdlam_optimization（项目 A）

基于 CD-LAM 对 DreamDojo 潜在动作的优化：loss、latent 结构，以及细粒度（因果 /
可逆 / 相反动作）动作表示。

研究能否通过 CD-LAM debiasing、32 维动作 + 8 维环境的拆分，以及一组辅助 loss（split-KL、
free-bits、L_zero、L_dir、前景重建、可逆/相反动作分离），把 LAM 的 latent `z_a` 变得
**充分、跨场景不变、可控、可接地**，并与原始 DreamDojo、CD-LAM、ConLA 类方法对比。

- **先读什么：** [`notes/README.md`](notes/README.md) → `notes/concepts_and_pipeline.md`。
- **完整项目记录：** [`PROJECT_MANIFEST.md`](PROJECT_MANIFEST.md)。
- **谱系：** A → **B**（[`../lam_condition_utilization`](../lam_condition_utilization)）→ **C**（[`../lam_world_model_control`](../lam_world_model_control)）。B 的"availability ≠ utilization"论文，源自 A 被否决的方向线。

## 目录结构
```
code/            共享 LAM 底座（train/model/dataset/eval/benchmark_lam/…）+ A 专属分析
code/config/     训练/benchmark 的 YAML（运行时 config 按项目根目录相对解析）
notes/           当前研究笔记（索引见 notes/README.md）
archive/notes/   转向前的历史（排名结论已作废——见 legacy_lessons.md）
results/         benchmark 与 eval 输出；results/lam_benchmark_v2 = 权威结果
logs/            运行日志
scripts/         run_ours.sh、run_all.sh、run_finetune.sh、run_eval_all.sh、transcode/sam3
data/            dir_pairs_train.npz（L_dir 配对挖掘输出）
checkpoints/     （逻辑归属）——物理存储是共享的 /checkpoints/lam-dis/（见 manifest）
```

## 快速上手（在本目录下运行）
```bash
# 训练一个 arm
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --mixed_precision bf16 \
    code/train.py --config code/config/ft_l40_ours_r2_opp_l01_5k.yaml
# 零训练 benchmark
python code/benchmark_lam.py --runs raw_lam kl_ft_l40_full_5k contrastive_v3_5k \
    --n_samples 2000 --out_dir results/lam_benchmark
```

公共数据在 `/home/xuan/embodied-ai/data/egodex/test_240p`，基础权重在
`/home/xuan/embodied-ai/checkpoints/pretrained/`。Python 环境见 `venv-requirements.txt`
（SAM3 mask 管线见 `sam3-requirements.txt`）。
