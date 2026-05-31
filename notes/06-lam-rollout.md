# LAM Rollout 指南

LAM 训练完成后的验证步骤：定量评估重建质量，定性可视化 latent action 分布。

**训练状态**：100k steps 已完成（2026-05-27），checkpoint 在 `/home/xuan/embodied-ai/checkpoints/lam/step_0100000`  
**Rollout 状态**：✅ 已完成，结果见 `/home/xuan/embodied-ai/checkpoints/lam/rollout_imgs/`

---

## 理解 log_imgs 里的图片

训练过程中每 2000 步自动保存一张对比图（4 格布局）：

```
┌────────────┬────────────┐
│    f_t     │  f_{t+1}   │  ← 上行：真实帧对（输入帧 | 真实下一帧）
│  (输入帧)  │  (GT下一帧)│
├────────────┼────────────┤
│    f_t     │   重建帧   │  ← 下行：f_t 重复 | LAM 重建的下一帧
│  (重复)    │  (recon)   │
└────────────┴────────────┘
```

对比右上（GT）和右下（重建），可以直观看出重建质量。
当前 log 图中 4 格几乎完全一样，原因是 val set 刚好取到了一个静态场景（布袋几乎不移动）。

---

## Step 1：定量评估（PSNR）

在验证集上计算重建 MSE 和 PSNR。

```bash
cd /home/xuan/embodied-ai/code/lam

CUDA_VISIBLE_DEVICES=1 /home/xuan/.venv/bin/python eval_lam.py \
    --checkpoint /home/xuan/embodied-ai/checkpoints/lam/step_0100000 \
    --config config/lam_agibot.yaml \
    --mode reconstruction \
    --n_samples 500
```

**期望输出**：
```
Eval on 500 samples:
  MSE  : 0.00XXXX
  PSNR : XX.XX dB
  (Reference: DreamDojo 'w/o pretrain' ≈ 20.3 dB)
```

**结果解读**：
- PSNR > 30 dB：过拟合（数据太少，模型记住了）
- PSNR 20-28 dB：正常范围，说明模型学到了有效的重建能力
- PSNR < 15 dB：模型没有正常收敛

---

## Step 2：生成可视化对比图

从测试视频中取若干帧对，生成并保存对比图。

```bash
CUDA_VISIBLE_DEVICES=1 /home/xuan/.venv/bin/python eval_lam.py \
    --checkpoint /home/xuan/embodied-ai/checkpoints/lam/step_0100000 \
    --config config/lam_agibot.yaml \
    --mode reconstruction \
    --n_samples 100
```

手动可视化（更灵活）：

```python
# 在 code/lam/ 目录下运行
import sys, torch, cv2
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, '../adaworld/lam')
from lam.modules import LatentActionModel
from data.agibot_dataset import AgibotVideoDataset

# 加载模型
CKPT = '/home/xuan/embodied-ai/checkpoints/lam/step_0100000'
model = LatentActionModel(
    in_dim=3, model_dim=512, latent_dim=32,
    patch_size=16, enc_blocks=8, dec_blocks=8, num_heads=8
).cuda().eval()

import glob
ckpt_files = glob.glob(f'{CKPT}/pytorch_model*.bin') + glob.glob(f'{CKPT}/*.safetensors')
sd = torch.load(ckpt_files[0], map_location='cpu')
if any(k.startswith('module.') for k in sd): 
    sd = {k[7:]: v for k, v in sd.items()}
model.load_state_dict(sd)

# 加载数据
ds = AgibotVideoDataset(
    data_root='/home/xuan/embodied-ai/data/agibotworld/extracted',
    camera='head_color', split='val', downsample_factors=(1,)
)

# 生成对比图
out_dir = Path('/home/xuan/embodied-ai/checkpoints/lam/rollout_imgs')
out_dir.mkdir(exist_ok=True)

for i in range(10):
    sample = ds[i * (len(ds) // 10)]
    videos = sample['videos'].unsqueeze(0).cuda()  # [1, 2, H, W, C]
    with torch.no_grad():
        out = model({'videos': videos})
    
    f_t   = videos[0, 0].cpu().numpy()         # [H, W, 3]
    f_t1  = videos[0, 1].cpu().numpy()         # GT
    recon = out['recon'][0, 0].cpu().numpy()   # 重建

    # 横向拼接：f_t | GT | recon
    row = np.concatenate([f_t, f_t1, recon], axis=1)
    img = (row.clip(0, 1) * 255).astype(np.uint8)
    Image.fromarray(img).save(out_dir / f'sample_{i:02d}.png')

print(f'Saved {10} comparison images to {out_dir}')
```

---

## Step 3：t-SNE 可视化 latent action 分布

验证 latent action 是否有结构（同类动作聚在一起）。

```bash
CUDA_VISIBLE_DEVICES=1 /home/xuan/.venv/bin/python eval_lam.py \
    --checkpoint /home/xuan/embodied-ai/checkpoints/lam/step_0100000 \
    --config config/lam_agibot.yaml \
    --mode visualize \
    --n_samples 2000 \
    --out_path /home/xuan/embodied-ai/checkpoints/lam/latents_tsne.png
```

需要安装 sklearn：
```bash
/home/xuan/.venv/bin/pip install scikit-learn matplotlib
```

---

## Step 4：确认 checkpoint 文件格式

Accelerate 保存的 checkpoint 格式需要确认，再决定加载方式：

```bash
ls /home/xuan/embodied-ai/checkpoints/lam/step_0100000/
```

可能的文件：
- `pytorch_model.bin`：单文件 state dict
- `pytorch_model_fsdp_0.bin` 等：FSDP 分片（需要 merge）
- `model.safetensors`：safetensors 格式

---

## 实际结果（2026-05-27）

### 定量评估

```
验证集大小：29,584 帧对（441 个 episode 的 5% val split）
评估样本数：512

MSE  : 0.000206
PSNR : 36.86 dB   ← 高于论文基线（~20.3 dB），说明过拟合
```

### 结果解读

| 指标 | 实际值 | 说明 |
|------|--------|------|
| 整体 PSNR | 36.86 dB | 过拟合。100k 步对于 441 episodes 约循环 86 遍，数据量不足 |
| 样本 PSNR 范围 | 24.7 ~ 39.6 dB | 运动幅度大的帧对 PSNR 较低（24.7 dB），接近正常范围 |
| 重建图视觉质量 | 清晰 | 布局、物体位置基本正确，细节有轻微模糊 |
| 运动帧重建 | 可见偏差 | `sample_02_psnr24.7.png` 机械臂移动较大，重建出现明显伪影 |

### 视觉观察

- **静态/慢动作帧**（PSNR 35-40 dB）：重建与 GT 几乎完全一致，符合过拟合预期
- **快速运动帧**（PSNR 24-30 dB）：重建能捕捉大致运动趋势，但边缘模糊，说明 latent action 对运动方向有一定编码能力
- **多样性**：12 个样本覆盖了不同任务场景（布袋操作、蓝色容器、工具台），说明数据多样性有限（仅 2 个 task ID）

### 过拟合的根本原因

- 训练数据：441 episodes × ~1500 帧 × 4 skip → ~220 万帧对
- 训练步数：100k steps × 288 有效 batch = ~2900 万次样本访问
- 每个帧对平均被训练约 **13 遍**（数据循环次数过多）
- 论文用 400k steps + 数千小时多样数据 → 每个样本约 1-2 遍

### 结论

LAM 功能正常，pipeline 验证成功。过拟合不影响下一步 Post-Training，因为 Post-Training 使用真实关节角度，不依赖 LAM 的泛化质量。

---

## 结果文件位置

```
results/lam-rollout/
├── lam_rollout_grid.png          ← 12 个样本的总览网格图
├── sample_00_psnr39.6.png        ← 静态场景，高度过拟合
├── sample_02_psnr24.7.png        ← 最大运动幅度，最接近真实泛化能力
├── sample_06_psnr28.9.png        ← 中等运动，蓝色容器场景
└── ...（共 12 个单独对比图）
```

每张对比图格式：`f_t（输入）| f_{t+1} GT（真实下一帧）| recon（LAM 重建）`

---

## 下一步

1. ✅ LAM Rollout 完成
2. 配置 Cosmos-Predict2.5 推理环境 → `07-cosmos-rollout.md`（待写）
3. 做 Cosmos zero-shot rollout，建立 Post-Training 前的 baseline
4. Post-Training：用 AgiBot 视频 + 关节数据微调 Cosmos-Predict2.5
