# DreamDojo 复现操作日志

记录每一步做了什么、用了什么命令、结果如何。

---

## 阶段 0：项目初始化

### 创建目录结构

```bash
mkdir -p /home/xuan/embodied-ai/{notes,code,data,checkpoints}
mkdir -p /home/xuan/embodied-ai/data/agibotworld
mkdir -p /home/xuan/embodied-ai/checkpoints/lam
```

最终目录布局：
```
/home/xuan/embodied-ai/
├── notes/          ← 笔记和文档（本文件所在位置）
├── code/
│   ├── adaworld/   ← 克隆的 AdaWorld 仓库（LAM 参考实现）
│   └── lam/        ← 我们自己的训练代码
├── data/
│   └── agibotworld/
│       ├── observations/   ← 从 HuggingFace 下载的原始 tar 文件
│       └── extracted/      ← 解压 + 转码后的 episode 数据
└── checkpoints/
    └── lam/        ← 训练产出的 checkpoint
```

---

## 阶段 1：环境配置

### Python 环境说明

服务器上存在两套 Python 环境，**实际使用 `.venv`**：

| 环境 | 路径 | 说明 |
|------|------|------|
| `.venv` | `/home/xuan/.venv/` | **实际使用的环境**，VSCode Server 自动激活，所有依赖装在这里 |
| conda dreamdojo | `/home/xuan/miniconda3/envs/dreamdojo/` | 创建了但依赖未安装到此，不使用 |

> **注意**：VSCode Server 启动时会将 `.venv/bin` 插入 PATH 最前，导致 `conda activate` 后 `python` 命令仍指向 `.venv`。解决方法是始终用绝对路径 `/home/xuan/.venv/bin/python`。

### 安装依赖

```bash
# 所有依赖均通过 .venv 的 pip 安装
/home/xuan/.venv/bin/pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

/home/xuan/.venv/bin/pip install \
    accelerate==1.13.0 \
    einops \
    decord \
    h5py \
    opencv-python-headless \
    wandb \
    tqdm \
    piq \
    lightning \
    pyyaml
```

安装后主要版本：
- Python 3.10.12
- PyTorch 2.3.0+cu121
- Accelerate 1.13.0
- einops 0.8.2
- decord 0.6.0

### 克隆 AdaWorld 仓库

AdaWorld（Gao et al., ICML 2025）是 DreamDojo LAM 的直接前驱，Apache-2.0 开源，复用其 `LatentActionModel` 实现。

```bash
cd /home/xuan/embodied-ai/code
git clone https://github.com/Little-Podi/AdaWorld.git adaworld
```

验证：
```bash
CUDA_VISIBLE_DEVICES=1 /home/xuan/.venv/bin/python -c "
import sys
sys.path.insert(0, 'adaworld/lam')
from lam.modules import LatentActionModel
import torch
m = LatentActionModel(in_dim=3, model_dim=512, latent_dim=32,
                      patch_size=16, enc_blocks=8, dec_blocks=8, num_heads=8).cuda()
batch = {'videos': torch.rand(2, 2, 240, 320, 3).cuda()}
out = m(batch)
print('recon:', out['recon'].shape)  # torch.Size([2, 1, 240, 320, 3])
print('OK')
"
```

---

## 阶段 2：LAM 训练代码

在 `/home/xuan/embodied-ai/code/lam/` 下创建了以下文件：

| 文件 | 说明 |
|------|------|
| `data/agibot_dataset.py` | AgiBot 数据加载，输出 `{"videos": [B,2,H,W,C]}` |
| `config/lam_agibot.yaml` | 训练超参数配置 |
| `train_lam.py` | Accelerate 多卡训练主脚本 |
| `eval_lam.py` | 评估脚本（MSE/PSNR + t-SNE 可视化） |
| `preprocess_videos.py` | tar 解压 + AV1→h264 转码脚本 |

关键超参数（参考 DreamDojo 论文）：

```yaml
lam_latent_dim: 32      # 32维连续潜在动作
beta: 1.0e-6            # KL 散度系数（论文值，AdaWorld 默认 0.01 太大）
lr: 2.5e-5              # AdamW 学习率
per_gpu_batch_size: 32  # 3 GPU × 32 × 3 grad_accum = 288 有效 batch
```

---

## 阶段 3：数据准备

### GPU 情况

```
GPU 0: nextgdt 进程占用 ~44GB → 不可用于训练
GPU 1/2/3: 各 49GB 空闲 → 用于训练
```

训练命令统一使用 `CUDA_VISIBLE_DEVICES=1,2,3`。

### 下载数据

使用服务器上已有的 `hf` CLI（v1.16.4）下载 AgiBot-World Alpha。

> **注意**：下载前必须先用 `--dry-run` 确认大小，否则可能下载几百 GB 超出预期。

```bash
# 先确认大小
hf download --type dataset --dry-run \
    --include "observations/362/**" \
    agibot-world/AgiBotWorld-Alpha
# 输出: Will download 17 files totalling 771.0G ← 这就需要三思了

# 只下载每个任务的第 1 个 tar
hf download --type dataset \
    --local-dir /home/xuan/embodied-ai/data/agibotworld \
    --include "observations/410/686871-686871.tar" \
    agibot-world/AgiBotWorld-Alpha

hf download --type dataset \
    --local-dir /home/xuan/embodied-ai/data/agibotworld \
    --include "observations/362/649552-654138.tar" \
    --include "observations/359/648638-681118.tar" \
    agibot-world/AgiBotWorld-Alpha
```

已下载数据：

| Task ID | tar 文件 | 大小 | Episodes |
|---------|----------|------|---------|
| 410 | `686871-686871.tar` | 198MB | 1（测试用）|
| 362 | `649552-654138.tar` | 46GB | 220 |
| 359 | `648638-681118.tar` | 46GB | 220 |

### 磁盘清理（下载超量后的处理）

下载时误下载了 task 362 的全部 17 个 tar（771GB），导致磁盘满。清理步骤：

```bash
# 1. 清 pip 缓存（释放 36GB，已安装包不受影响）
/home/xuan/.venv/bin/pip cache purge

# 2. task 362：只保留第 1 个 tar，删除其余 16 个
cd /home/xuan/embodied-ai/data/agibotworld/observations/362/
ls | grep -v "649552-654138.tar" | xargs rm -f

# 3. task 359：只保留第 1 个 tar，删除其余 3 个
cd /home/xuan/embodied-ai/data/agibotworld/observations/359/
rm -f 681122-688735.tar 688742-694016.tar 694024-694109.tar
```

清理后可用空间：4.4GB → **843GB**。

### 解压 + 转码

AgiBot 数据有两个特点需要预处理：
1. 以 `.tar` 打包分发，需要解压
2. 视频为 **AV1 编码**，服务器的 OpenCV/decord 不支持解码，需转码为 h264

```bash
# 在 tmux 里运行（约 1-2 小时）
tmux new-session -s preprocess

/home/xuan/.venv/bin/python /home/xuan/embodied-ai/code/lam/preprocess_videos.py \
    --tar_root /home/xuan/embodied-ai/data/agibotworld/observations \
    --out_root /home/xuan/embodied-ai/data/agibotworld/extracted \
    --camera head_color \
    --n_workers 4
```

解压后目录结构：
```
extracted/
├── 410/
│   └── 686871/
│       └── videos/
│           └── head_color.mp4   ← h264，OpenCV 可读
├── 359/
│   ├── 648638/videos/head_color.mp4
│   ├── 648675/videos/head_color.mp4
│   └── ...（约 220 个 episode）
└── 362/
    ├── 649552/videos/head_color.mp4
    └── ...（约 220 个 episode）
```

---

## 阶段 4：LAM 训练

### Pipeline 验证（dry run + 小数据）

```bash
cd /home/xuan/embodied-ai/code/lam

# Dry run（5 steps，验证代码无误）
CUDA_VISIBLE_DEVICES=1 /home/xuan/.venv/bin/python train_lam.py \
    --config config/lam_agibot.yaml --dry_run
# 输出: Model params: 60.6M / Train pairs: 1,298 / Dry run complete

# Pipeline 验证训练（task 410，1 个视频，验证全流程通畅）
CUDA_VISIBLE_DEVICES=1,2,3 /home/xuan/.venv/bin/accelerate launch \
    --num_processes 3 --mixed_precision bf16 \
    train_lam.py --config config/lam_agibot.yaml
# 约 2-4 小时完成（数据量太少，仅验证用）
```

### 正式训练（440 episodes，100k steps）

完成 preprocess_videos.py 后执行：

```bash
# 新建 tmux session（关闭 VSCode 后训练仍继续）
tmux new-session -s lam_train

CUDA_VISIBLE_DEVICES=1,2,3 /home/xuan/.venv/bin/accelerate launch \
    --num_processes 3 --mixed_precision bf16 \
    /home/xuan/embodied-ai/code/lam/train_lam.py \
    --config /home/xuan/embodied-ai/code/lam/config/lam_agibot.yaml

# 分离 session（可以关 VSCode）
# Ctrl+B, 然后 D

# 重新连接查看进度
tmux attach -t lam_train
```

训练进度查看：
```bash
# 查看日志输出（在 tmux 内）
# 日志格式：step  6 | loss 0.0812 | mse 0.0812 | kl 0.000123 | lr 1.50e-05

# 查看 checkpoint 保存情况
ls /home/xuan/embodied-ai/checkpoints/lam/

# 查看 GPU 使用情况
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

预期训练时间（3× RTX 6000 Ada，440 episodes，100k steps）：**8-12 小时**

### 训练完成后评估

```bash
# 定量评估（MSE / PSNR）
CUDA_VISIBLE_DEVICES=1 /home/xuan/.venv/bin/python eval_lam.py \
    --checkpoint /home/xuan/embodied-ai/checkpoints/lam/step_0100000 \
    --config config/lam_agibot.yaml \
    --mode reconstruction
# 期望: PSNR ≥ 20 dB（论文无预训练基线 ~20.3 dB）

# 可视化 latent action 分布（t-SNE）
CUDA_VISIBLE_DEVICES=1 /home/xuan/.venv/bin/python eval_lam.py \
    --checkpoint /home/xuan/embodied-ai/checkpoints/lam/step_0100000 \
    --config config/lam_agibot.yaml \
    --mode visualize \
    --out_path /home/xuan/embodied-ai/checkpoints/lam/latents_tsne.png
```

---

## 阶段 5：LAM Rollout 评估（2026-05-27）

```bash
cd /home/xuan/embodied-ai/code/lam

# 定量评估（PSNR）
CUDA_VISIBLE_DEVICES=1 /home/xuan/.venv/bin/python eval_lam.py \
    --checkpoint /home/xuan/embodied-ai/checkpoints/lam/step_0100000 \
    --config config/lam_agibot.yaml \
    --mode reconstruction \
    --n_samples 500
# 结果: MSE=0.000206, PSNR=36.86 dB（过拟合，原因是数据循环约 86 遍）
```

可视化对比图（手动脚本，取 12 个不同场景样本）：
```
输出: checkpoints/lam/rollout_imgs/lam_rollout_grid.png
      checkpoints/lam/rollout_imgs/sample_XX_psnrXX.X.png（共 12 张）
格式: f_t（输入）| f_{t+1} GT | recon（LAM 重建）
```

**结果摘要**：
- 整体 PSNR 36.86 dB，过拟合（训练数据循环约 86 遍）
- 运动幅度最大的样本 PSNR 24.7 dB，接近正常泛化范围
- 重建图视觉清晰，LAM 功能正常，pipeline 验证成功
- 详细分析见 `06-lam-rollout.md`

---

## 阶段 6：整理结果目录 + Cosmos 环境配置（2026-05-31）

### 结果目录整理

```bash
# 新建 results 目录，将 LAM rollout 图片移入
mkdir -p /home/xuan/embodied-ai/results/lam-rollout
mkdir -p /home/xuan/embodied-ai/results/cosmos-rollout
mv /home/xuan/embodied-ai/checkpoints/lam/rollout_imgs/* \
   /home/xuan/embodied-ai/results/lam-rollout/
```

### 安装 uv 包管理器

Cosmos-Predict2.5 使用 `uv` 管理依赖，不支持标准 pip 安装：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

### 克隆 Cosmos-Predict2.5 仓库

```bash
cd /home/xuan/embodied-ai/code
git clone https://github.com/nvidia-cosmos/cosmos-predict2.5.git cosmos-predict25
cd cosmos-predict25
```

### 安装依赖（指定 Python 3.10）

```bash
export PATH="$HOME/.local/bin:$PATH"
# 必须指定 --python 3.10，否则 uv 默认用 Python 3.13，flash-attn 不兼容
uv sync --extra=cu128 --python 3.10
```

验证：
```bash
.venv/bin/python -c "import torch; print(torch.__version__)"
# 输出: 2.7.0+cu128
```

### 准备推理输入

```bash
# 提取 AgiBot 视频第一帧作为 conditioning frame
ffmpeg -i /home/xuan/embodied-ai/data/agibotworld/extracted/362/649552/videos/head_color.mp4 \
    -vframes 1 /home/xuan/embodied-ai/results/cosmos-rollout/cond_frame.jpg -y

# 创建推理 JSON
cat > /home/xuan/embodied-ai/results/cosmos-rollout/agibot_input.json << 'EOF'
{
    "inference_type": "image2world",
    "name": "agibot_zero_shot",
    "prompt": "A robot arm performing a manipulation task on a table",
    "input_path": "/home/xuan/embodied-ai/results/cosmos-rollout/cond_frame.jpg"
}
EOF
```

### 启动 Zero-Shot 推理

```bash
cd /home/xuan/embodied-ai/code/cosmos-predict25
export PATH="$HOME/.local/bin:$PATH"

# 在 tmux 中运行（预计 30-90 分钟）
tmux new-session -s cosmos

CUDA_VISIBLE_DEVICES=1 .venv/bin/python examples/inference.py \
    -i /home/xuan/embodied-ai/results/cosmos-rollout/agibot_input.json \
    -o /home/xuan/embodied-ai/results/cosmos-rollout/ \
    --inference-type=image2world \
    --model=2B/pre-trained \
    --disable-guardrails
```

模型权重（tokenizer.pth、ema_bf16.pt、Cosmos-Reason1-7B）在首次运行时自动下载到 `~/.cache/huggingface/`。

---

## 当前状态

| 步骤 | 状态 |
|------|------|
| 目录结构创建 | ✅ |
| Python 环境（.venv）配置 | ✅ |
| AdaWorld 克隆 | ✅ |
| LAM 训练代码编写 | ✅ |
| task 410/359/362 数据下载 + 解压 + 转码 | ✅ 共 441 episodes |
| Pipeline 验证（dry run）| ✅ |
| LAM 100k 步正式训练 | ✅ |
| LAM Rollout 评估 | ✅ PSNR 36.86 dB（过拟合，pipeline 正常）|
| 结果目录整理（results/）| ✅ |
| Cosmos-Predict2.5 环境配置 | ✅ PyTorch 2.7+cu128 |
| Cosmos Zero-Shot 推理 | 🔄 进行中（tmux: cosmos）|
| Post-Training（Cosmos + AgiBot 关节数据）| ⏳ |
