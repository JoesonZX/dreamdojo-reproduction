# A6000 机器:seed 训练说明(3×A6000,~10 小时窗口)

## 跑什么(与本机分工)
| 卡 | 任务 | 配置 | 预计 |
|---|---|---|---|
| GPU 0+1(双卡,与原始 run 同批量) | r2-s1 → r2-s2 串行 | `ft_l40_ours_r2_opp_l01_5k_s1.yaml` → `_s2.yaml` | 每个 ~5.5h,共 ~11h |
| GPU 2(单卡,accum 已翻倍) | a-s1 | `ft_l40_ours_a_zero_5k_s1_1gpu.yaml` | ~10-11h |

本机(GPU 2/3 空出后):`bash run_ours.sh seeds-local`(kl-s1 → kl-s2 → a-s2)。

注意:r2 两个 run 合计 ~11h,若 10h 必须硬停,r2-s2 会在 ~step 4500 截断
(save_every=1000,只留 step_0004000,不可与 5k 比较)——能多给 1 小时就完整。

## 传过去什么
解压 `a6000_seed_bundle.tar.gz` 到远端 `lam_disentangle/` 目录(覆盖 `code/` 与 `run_ours.sh`):
- `code/train.py` **必须更新**——`training.seed` 支持是新加的,旧版 train.py 会忽略
  seed 字段,跑出来仍然是 42(等于白跑)。
- `code/dataset.py`、`code/model.py` 等一并同步保持一致。
- `code/config/*_s1.yaml`、`*_s2.yaml`、`*_s1_1gpu.yaml`:6+1 个新配置。

远端已有、无需传:EgoDex 数据、`checkpoints/pretrained/LAM_400k.ckpt`、venv/accelerate 环境
(之前 l03/l05/l08 就在那台机器训的)。

## 路径检查(重要)
配置里全是绝对路径 `/home/xuan/embodied-ai/...`(init_from / output_dir / log_path / data_root)。
若远端根目录不同,一行替换:
```bash
sed -i 's|/home/xuan/embodied-ai|<REMOTE_ROOT>|g' code/config/ft_l40_*_s[12]*.yaml
```
确认存在:`<ROOT>/checkpoints/pretrained/LAM_400k.ckpt` 和 `<ROOT>/data/egodex/test_240p`。

## 启动命令(在远端 lam_disentangle 目录下)
```bash
PY=<远端python>   # 之前训 l03 用的那个解释器

# GPUs 0+1:r2 两个 seed 串行(注意 main_process_port,避免与另一路冲突)
nohup bash -c "
  CUDA_VISIBLE_DEVICES=0,1 $PY -m accelerate.commands.launch --num_processes 2 \
    --mixed_precision bf16 --main_process_port 29511 \
    code/train.py --config code/config/ft_l40_ours_r2_opp_l01_5k_s1.yaml && \
  CUDA_VISIBLE_DEVICES=0,1 $PY -m accelerate.commands.launch --num_processes 2 \
    --mixed_precision bf16 --main_process_port 29511 \
    code/train.py --config code/config/ft_l40_ours_r2_opp_l01_5k_s2.yaml
" > r2_seeds.log 2>&1 &

# GPU 2:a-s1 单卡(accum 4→8,有效 batch 不变)
nohup bash -c "
  CUDA_VISIBLE_DEVICES=2 $PY -m accelerate.commands.launch --num_processes 1 \
    --mixed_precision bf16 --main_process_port 29512 \
    code/train.py --config code/config/ft_l40_ours_a_zero_5k_s1_1gpu.yaml
" > a_s1.log 2>&1 &

tail -f r2_seeds.log   # step 50 应在几分钟内出现
```

## 跑完传回来什么
只需每个 run 的 `model.safetensors`(2.8GB;`optimizer.bin` 5.7GB 不要传):
```bash
for r in ft_l40_ours_r2_opp_l01_5k_s1 ft_l40_ours_r2_opp_l01_5k_s2 ft_l40_ours_a_zero_5k_s1; do
  rsync -av --include 'model.safetensors' --exclude '*' \
    <remote>:<ROOT>/checkpoints/lam-dis/$r/step_0005000/ \
    /home/xuan/embodied-ai/checkpoints/lam-dis/$r/step_0005000/
done
```
传回后本机直接评测(RUN_SPECS 已注册这些名字):
```bash
python code/benchmark_lam.py --runs kl_ft_l40_full_5k_s1 kl_ft_l40_full_5k_s2 \
  ours_a_zero_5k_s1 ours_a_zero_5k_s2 ours_r2_opp_l01_5k_s1 ours_r2_opp_l01_5k_s2 \
  --n_samples 1500 --perturb_samples 48 --batch_size 32 --num_workers 8 \
  --device cuda:0 --out_dir results/lam_benchmark_seeds
```

## 解读时的两个标记
- a-s1 是单卡 accum×2 训的(有效 batch 相同):A 模型的 seed 波动估计里混入了
  机器/并行度差异,是保守方向(高估波动),报告时注明。
- r2-s1/s2 与本机 seed42 跨机器:同理,σ 略保守。r2-s1 vs r2-s2 之间同机同配置,干净。
