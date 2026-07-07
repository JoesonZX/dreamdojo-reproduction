# LAM 动作解耦实验 (lam_disentangle)

研究 DreamDojo LAM 的 latent action 是否混入环境信息，并用真实动作监督、环境子空间、
KL/对比/时间反转等目标来提纯动作表示。阅读顺序见 [`notes/README.md`](notes/README.md)，
当前交接见 [`notes/HANDOFF.md`](notes/HANDOFF.md)，下一阶段规划见
[`notes/fine_grained_causal_lam_plan.md`](notes/fine_grained_causal_lam_plan.md)。

**不修改任何原有代码**：`code/` 下是从 `code/lam/` 与 `code/adaworld/` vendoring 的独立
版本；不变的底层模块（blocks/embeddings）只读 import 自原仓库。

## 四个 run（每步只改一个变量）

| Run | config | latent | env | action_head | λ_action | λ_indep |
|-----|--------|--------|-----|-------------|----------|---------|
| Baseline | `baseline.yaml` | 32 | 0 | 关 | 0 | 0 |
| Exp1 动作监督 | `exp1_action.yaml` | 32 | 0 | 开(全32维) | 1.0 | 0 |
| Exp2 32+8 拆分 | `exp2_split.yaml` | 40 | 8 | 开(0:32) | 1.0 | 0 |
| Exp3 +独立性 | `exp3_indep.yaml` | 40 | 8 | 开(0:32) | 1.0 | 0.02 |

真实动作 = 18 维（双手 Δ平移3 + 6D旋转6），来自 EgoDex HDF5 `transforms/{left,right}Hand`。

## 前置：数据预转码（一次性，必做）

EgoDex 原视频是 1080p mpeg4，随机解码极慢（4 路并行时 GPU 会挨饿）。先转码成
240p 全关键帧版本到 `data/egodex/test_240p/`（约 2.7GB，~1.5 分钟，HDF5 自动软链接）：

```bash
cd /home/xuan/embodied-ai/lam_disentangle
bash transcode_240p.sh          # 转码
bash transcode_240p.sh --check  # 校验帧数对齐
```

4 个 config 的 `data_root` 已指向 `test_240p`。详见 [`notes/preparation.md`](notes/preparation.md) §4b。

## 训练

```bash
cd /home/xuan/embodied-ai/lam_disentangle
# 冒烟测试（1 GPU，5 步）
CUDA_VISIBLE_DEVICES=1 python code/train.py \
    --config code/config/exp3_indep.yaml --dry_run

# 正式（按可用 GPU 数调整 --num_processes）；四个 config 依次跑
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --mixed_precision bf16 \
    code/train.py \
    --config code/config/baseline.yaml
```

## 评估

```bash
CKPT=/home/xuan/embodied-ai/checkpoints/lam-dis/exp3_indep/step_0020000
CFG=code/config/exp3_indep.yaml
python code/eval.py --checkpoint $CKPT --config $CFG \
    --mode action_probe --n_samples 4000 --out_path results/exp3_probe.txt   # Eval 1+2
python code/eval.py --checkpoint $CKPT --config $CFG \
    --mode reconstruction --n_samples 1000                                   # Eval 3 代理
python code/eval.py --checkpoint $CKPT --config $CFG \
    --mode visualize --n_samples 2000 --out_path results/exp3_tsne.png       # 聚类
python code/eval.py --checkpoint $CKPT --config $CFG \
    --mode perturb --out_path results/exp3_perturb.png                       # Eval 4
```

## Stage 1 LAM-side Benchmark

```bash
CUDA_VISIBLE_DEVICES=1 python code/benchmark_lam.py \
    --runs raw_lam kl_ft_l40_full_5k contrastive_v3_5k \
    --n_samples 2000 --perturb_samples 64 --batch_size 16 \
    --device cuda:0 --out_dir results/lam_benchmark
```

If the selected GPU is busy, lower `--batch_size` first; the benchmark itself uses
one GPU.

预期：动作子空间预测真实动作的 R² ≫ 环境子空间，且差距 Baseline→Exp1→Exp2→Exp3 拉大；
重建 PSNR 基本持平；扰动动作维改变运动、扰动环境维改变外观。
