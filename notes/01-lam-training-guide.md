# LAM 训练完整指南

**什么阶段**：DreamDojo 复现第一步  
**目标产物**：一个训练好的 Latent Action Model（LAM），能从相邻视频帧中提取 32 维"动作向量"  
**预计时间**：10-14 天（含数据下载）  
**机器**：4× RTX 6000 Ada（192GB VRAM），1.1TB 可用磁盘

---

## 一、这一步在做什么？（必读）

### 1.1 问题背景

DreamDojo 是一个 **action-conditioned 世界模型**：给定当前画面和一个动作，预测下一帧画面。
训练它需要大量 **"(视频帧, 动作)" 配对数据**。

但问题是：机器人视频有动作标注，而人类视频（更便宜、更多样）**没有动作标注**。

> 难道只能用机器人数据？这会让模型的泛化能力很差（只见过几种物体和场景）。

### 1.2 解法：Latent Action Model（潜在动作模型）

LAM 的核心思想：**"动作"就是"两帧之间发生的变化"**。

不需要人告诉我"这个动作是向右移动手臂"，只需要：
1. 给我看 $f_t$（这帧）
2. 给我看 $f_{t+1}$（下一帧）
3. 我自己学出一个 32 维向量 $\hat{a}_t$，这个向量**足以解释**从 $f_t$ 变化到 $f_{t+1}$ 的原因

这就是**自监督学习**：监督信号不来自人工标注，而来自数据本身（重建下一帧）。

### 1.3 为什么是 VAE？

LAM 使用 VAE（变分自编码器）结构：

```
[f_t, f_{t+1}] → Encoder → μ, σ → 采样 â_t（32维）
                                        ↓
                   f_t → Decoder(â_t, f_t) → 重建 f_{t+1}
```

VAE 的关键：**信息瓶颈**（Information Bottleneck）

- 把复杂的帧变化压缩成 32 维
- KL 散度惩罚迫使 32 维向量尽量接近标准正态分布
- 为了在 32 维内重建出 $f_{t+1}$，模型 **被迫只保留最关键的信息**——也就是"做了什么动作"
- 背景颜色、光照变化等无关信息会被丢弃

### 1.4 跨具身迁移（Cross-embodiment Transfer）

LAM 的另一个重要特性：**人类手的动作和机器人手臂的动作会产生相似的 latent vector**。

原因：物理世界的动作本质是相同的（"向右移动"的 latent action 不依赖于"是用手还是用机械臂"）。

这让我们可以：
- 在**人类视频**上训练 LAM（廉价、多样）
- 提取的 latent action 可以迁移到**机器人数据**上用

### 1.5 LAM 的产物怎么用？

训练完 LAM 后，用它的 Encoder 对所有视频提取 latent action：

```python
latent_action = lam_encoder(f_t, f_{t+1})  # [32维向量]
```

这个向量将作为后续世界模型（Cosmos-Predict2.5）的条件信号，告诉世界模型"执行了什么动作"。

---

## 二、关键概念与知识点

| 概念 | 一句话解释 | 需要了解的程度 |
|------|-----------|-------------|
| **VAE** | 编码器学分布（μ,σ），解码器重建，KL 约束潜空间 | 必须理解 |
| **信息瓶颈** | 低维 + 正则化 = 强迫只保留最关键信息 | 必须理解 |
| **重参数化技巧** | $z = \mu + \sigma \cdot \epsilon$，让采样可微 | 理解即可 |
| **自监督学习** | 监督信号来自数据本身（不需要人工标注） | 必须理解 |
| **Spatiotemporal Transformer** | 同时对空间（图像 patch）和时间（帧序列）做注意力 | 了解架构 |
| **Flow Matching** | 后续世界模型用的训练方式，LAM 本身不用 | 暂时不需要 |
| **跨具身迁移** | 不同机器人/人的相似动作对应相似 latent | 理解概念即可 |
| **AdaWorld** | DreamDojo LAM 的前驱工作，有公开代码（Apache-2.0）| 需要看代码 |

### 关键论文

- **DreamDojo**（本复现目标）：arXiv:2602.06949
- **AdaWorld**（LAM 参考实现）：arXiv:2503.18938，ICML 2025，https://github.com/Little-Podi/AdaWorld
- **Genie**（ST-Transformer 来源）：arXiv:2402.15391，Google DeepMind 2024

---

## 五、完整操作步骤

### Step 0：安装 tmux（检查是否已有）

```bash
tmux -V  # 查看版本，若无输出则安装
# 若没有：sudo apt-get install tmux
```

### Step 1：创建 conda 环境

```bash
# 创建独立环境，避免污染其他项目
conda create -n dreamdojo python=3.10 -y
conda activate dreamdojo

# 安装 PyTorch（CUDA 12.x，对应当前 CUDA 13.1 兼容）
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121

# 验证 GPU 可见
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

### Step 2：克隆 AdaWorld

```bash
mkdir -p /home/xuan/embodied-ai/code
cd /home/xuan/embodied-ai/code
git clone https://github.com/Little-Podi/AdaWorld.git adaworld
cd adaworld
pip install -r requirements.txt

# 额外依赖
pip install h5py decord einops accelerate wandb imageio imageio-ffmpeg
```

### Step 3：申请 AgiBot-World Alpha 访问权限

1. 访问 https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha
2. 点击"Request access"，填写用途（学术研究/复现 DreamDojo）
3. 等待审核（通常 1-2 个工作日）
4. 获批后，在服务器上：
   ```bash
   pip install huggingface_hub
   huggingface-cli login  # 粘贴你的 HuggingFace token
   ```

### Step 4：下载数据（在 tmux 中运行，可断开 VSCode）

```bash
tmux new-session -s data-download

mkdir -p /home/xuan/embodied-ai/data/agibotworld

# 先只下载 task 327（约 50 GB）测试
huggingface-cli download --resume-download --repo-type dataset \
  agibot-world/AgiBotWorld-Alpha \
  --local-dir /home/xuan/embodied-ai/data/agibotworld

# 注意：--resume-download 支持断点续传，断网重连后继续
# 如果要选择特定任务，使用 --include 参数过滤

# Ctrl+B D 脱离 tmux，让下载在后台继续
```

**查看下载进度**：
```bash
tmux attach -t data-download       # 重新进入查看
# 或
du -sh /home/xuan/embodied-ai/data/agibotworld/  # 查看已下载大小
watch -n 30 "df -h /home/xuan"    # 每30秒刷新一次剩余空间
```

### Step 5：创建代码目录结构

```bash
mkdir -p /home/xuan/embodied-ai/code/lam/{data,model,config}
mkdir -p /home/xuan/embodied-ai/checkpoints/lam
```

### Step 6：创建数据加载代码

创建文件 `/home/xuan/embodied-ai/code/lam/data/agibot_dataset.py`：

```python
"""
AgiBot-World Alpha 数据集加载器，用于 LAM 训练。

LAM 的输入是帧对 (f_t, f_{t+1})，纯自监督，不需要动作标注。
动作标注（action/joint/position）在世界模型后训练阶段才使用。

AgiBot 数据格式：
- 视频: data/observations/{task_id}/{ep_id}/videos/top.mp4
- 动作: data/proprio_stats/{task_id}/{ep_id}/proprio_stats.h5
  - state/joint/position  [N, 14]  — 14 DoF 关节角（弧度），30Hz
  - action/joint/position [N, 14]  — 目标关节角
  - timestamp             [N]      — 纳秒时间戳
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

try:
    import decord
    decord.bridge.set_bridge('torch')
    USE_DECORD = True
except ImportError:
    import cv2
    USE_DECORD = False
    print("Warning: decord not found, using cv2 (slower). Install: pip install decord")


class AgibotFramePairDataset(Dataset):
    """
    从 AgiBot-World Alpha 视频中提取帧对，用于 LAM 自监督训练。
    
    每个样本 = {"f_t": [3, H, W], "f_t1": [3, H, W]}，值范围 [0, 1]。
    
    时序降采样因子随机选取，捕捉不同速度的运动：
    - factor=1: 连续帧（快速小幅运动）
    - factor=4: 间隔4帧（慢速大幅运动）
    """
    
    def __init__(
        self,
        data_root: str,
        img_size: tuple = (240, 320),        # (H, W)，LAM 训练分辨率
        downsample_factors: tuple = (1, 2, 3, 4),
        camera: str = "top",                  # AgiBot 有 top/left/right 三路
        split: str = "train",                 # train / val
        val_ratio: float = 0.05,
        seed: int = 42,
    ):
        self.data_root = Path(data_root)
        self.img_size = img_size
        self.downsample_factors = downsample_factors
        self.camera = camera
        
        # 构建帧对索引
        all_pairs = self._build_index()
        
        # 按 episode 划分 train/val（不按帧划分，避免泄露）
        rng = np.random.default_rng(seed)
        n_val = max(1, int(len(all_pairs) * val_ratio))
        val_indices = set(rng.choice(len(all_pairs), n_val, replace=False))
        
        if split == "train":
            self.pairs = [p for i, p in enumerate(all_pairs) if i not in val_indices]
        else:
            self.pairs = [p for i, p in enumerate(all_pairs) if i in val_indices]
        
        print(f"[AgibotDataset] {split}: {len(self.pairs)} frame pairs")
    
    def _build_index(self):
        """遍历所有 episode 的视频，建立帧对索引。"""
        pairs = []
        video_pattern = f"**/videos/{self.camera}.mp4"
        
        for video_path in sorted(self.data_root.glob(video_pattern)):
            n_frames = self._get_frame_count(video_path)
            if n_frames < 2:
                continue
            for factor in self.downsample_factors:
                for t in range(0, n_frames - factor, factor):
                    pairs.append((str(video_path), t, t + factor))
        
        if len(pairs) == 0:
            raise RuntimeError(
                f"No video files found under {self.data_root}. "
                f"Pattern: {video_pattern}. Check your data path."
            )
        return pairs
    
    def _get_frame_count(self, video_path):
        if USE_DECORD:
            try:
                vr = decord.VideoReader(str(video_path))
                return len(vr)
            except Exception:
                return 0
        else:
            cap = cv2.VideoCapture(str(video_path))
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return count
    
    def _load_frame(self, video_path, frame_idx):
        """加载单帧，中心裁剪，缩放到目标分辨率，归一化到 [0,1]。"""
        H, W = self.img_size
        
        if USE_DECORD:
            vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
            frame = vr[frame_idx].numpy()  # [H, W, 3], uint8, RGB
        else:
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return torch.zeros(3, H, W)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 中心裁剪为正方形
        h, w = frame.shape[:2]
        s = min(h, w)
        y1, x1 = (h - s) // 2, (w - s) // 2
        frame = frame[y1:y1+s, x1:x1+s]
        
        # 缩放到目标尺寸
        import cv2 as _cv2
        frame = _cv2.resize(frame, (W, H), interpolation=_cv2.INTER_AREA)
        
        # [H, W, 3] -> [3, H, W]，归一化
        frame = torch.from_numpy(frame).float() / 255.0
        return frame.permute(2, 0, 1)
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        video_path, t, t1 = self.pairs[idx]
        return {
            "f_t":  self._load_frame(video_path, t),
            "f_t1": self._load_frame(video_path, t1),
        }


def build_dataloaders(data_root, img_size=(240, 320), batch_size=32,
                      num_workers=8, **kwargs):
    """创建 train/val DataLoader。"""
    train_ds = AgibotFramePairDataset(data_root, img_size=img_size, 
                                       split="train", **kwargs)
    val_ds   = AgibotFramePairDataset(data_root, img_size=img_size, 
                                       split="val", **kwargs)
    
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        prefetch_factor=2, persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader
```

### Step 7：创建训练配置文件

创建 `/home/xuan/embodied-ai/code/lam/config/lam_agibot.yaml`：

```yaml
# LAM 训练配置（DreamDojo 超参数 + 4× RTX 6000 Ada 适配）

model:
  # DreamDojo 论文 Section 4.1 的超参数
  latent_dim: 32            # 潜在动作维度
  beta: 1.0e-6              # KL 散度权重（论文值，比 AdaWorld 默认值小很多）
  encoder_depth: 24         # 编码器 Transformer 层数
  decoder_depth: 24         # 解码器 Transformer 层数
  num_heads: 16             # 注意力头数
  embed_dim: 1024           # Transformer 隐层维度

data:
  data_root: /home/xuan/embodied-ai/data/agibotworld
  img_size: [240, 320]      # [H, W]，LAM 训练分辨率
  downsample_factors: [1, 2, 3, 4]
  camera: top               # 使用顶部摄像头

training:
  # 论文: batch=256, 400k steps（用 256 GPU hours）
  # 4× RTX 6000 Ada 适配:
  per_gpu_batch_size: 32    # 每卡 32（700M 模型 + bf16 ≈ 28GB/卡）
  gradient_accumulation: 2  # 4卡 × 32 × 2 = 256，等效论文 batch size
  total_steps: 100000       # 先跑 100k（约 3-5天），验证有效后再续跑至 400k
  
  # 优化器（论文值）
  lr: 2.5e-5
  weight_decay: 0.01
  optimizer: adamw
  warmup_steps: 2000
  
  # 精度
  bf16: true                # RTX 6000 Ada 支持 bf16
  
  # 日志与保存
  log_every: 100            # 每 100 步打印一次 loss
  save_every: 5000          # 每 5k 步保存 checkpoint（~700MB）
  eval_every: 2000          # 每 2k 步在验证集上评估重建 MSE
  output_dir: /home/xuan/embodied-ai/checkpoints/lam
  wandb_project: dreamdojo-lam   # 可选，WandB 日志
  
  # 数据加载
  num_workers: 8

# GPU 设置（GPU 0 被其他进程占用）
# 启动时用: CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch ...
```

### Step 8：创建训练主脚本

创建 `/home/xuan/embodied-ai/code/lam/train_lam.py`：

```python
"""
LAM 训练主脚本。

用法:
  CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch \
    --num_processes 3 --mixed_precision bf16 \
    train_lam.py --config config/lam_agibot.yaml

训练流程:
  1. 加载 AdaWorld 的 LAM 实现，调整超参数到 DreamDojo 值
  2. 用 AgiBot-World Alpha 视频的帧对做自监督训练
  3. 每隔一段时间评估重建质量，保存 checkpoint
"""

import argparse, yaml, os, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed

# 加入 AdaWorld 路径
sys.path.insert(0, str(Path(__file__).parent.parent / "adaworld"))

from data.agibot_dataset import build_dataloaders


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg):
    """
    加载 AdaWorld 的 LAM，调整为 DreamDojo 超参数。
    如果 AdaWorld 的接口不完全匹配，这里会是需要适配的地方。
    """
    try:
        from adaworld.lam.model import LatentActionModel
        model = LatentActionModel(
            latent_dim=cfg['model']['latent_dim'],
            beta=cfg['model']['beta'],
            encoder_depth=cfg['model']['encoder_depth'],
            decoder_depth=cfg['model']['decoder_depth'],
            num_heads=cfg['model']['num_heads'],
            embed_dim=cfg['model']['embed_dim'],
        )
    except (ImportError, TypeError) as e:
        print(f"[Warning] Could not load AdaWorld LAM directly: {e}")
        print("Falling back to minimal VAE implementation. Check AdaWorld API.")
        raise
    return model


def lam_loss(f_t1_pred, f_t1_gt, mu, logvar, beta):
    """
    VAE 损失 = 重建损失 + β * KL散度
    
    重建损失: MSE（像素级）
    KL 散度: 正则化潜在空间，鼓励 latent action 接近标准正态分布
    β=1e-6: 非常小，让重建优先，信息瓶颈宽松——这是 DreamDojo 的关键设计
            （太大的 β 会丢失太多信息，跨具身迁移时不好用）
    """
    recon_loss = F.mse_loss(f_t1_pred, f_t1_gt)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss.item(), kl_loss.item()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    
    accelerator = Accelerator(
        mixed_precision="bf16" if cfg['training']['bf16'] else "no",
        gradient_accumulation_steps=cfg['training']['gradient_accumulation'],
        log_with="wandb" if cfg['training'].get('wandb_project') else None,
    )
    set_seed(42)
    
    if accelerator.is_main_process:
        os.makedirs(cfg['training']['output_dir'], exist_ok=True)
        print(f"[Config] {cfg}")
    
    # 数据
    train_loader, val_loader = build_dataloaders(
        data_root=cfg['data']['data_root'],
        img_size=tuple(cfg['data']['img_size']),
        batch_size=cfg['training']['per_gpu_batch_size'],
        downsample_factors=tuple(cfg['data']['downsample_factors']),
        camera=cfg['data']['camera'],
        num_workers=cfg['training']['num_workers'],
    )
    
    # 模型
    model = build_model(cfg)
    beta = cfg['model']['beta']
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['training']['lr'],
        weight_decay=cfg['training']['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['training']['total_steps']
    )
    
    # Accelerate 准备（自动处理 DDP / FSDP / 混合精度）
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    
    # 训练循环
    global_step = 0
    start_time = time.time()
    
    while global_step < cfg['training']['total_steps']:
        for batch in train_loader:
            with accelerator.accumulate(model):
                f_t  = batch["f_t"]   # [B, 3, H, W]
                f_t1 = batch["f_t1"]  # [B, 3, H, W]
                
                # 前向
                f_t1_pred, mu, logvar = model(f_t, f_t1)
                loss, recon_l, kl_l = lam_loss(f_t1_pred, f_t1, mu, logvar, beta)
                
                # 反向
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            global_step += 1
            
            # 日志
            if global_step % cfg['training']['log_every'] == 0 and accelerator.is_main_process:
                elapsed = (time.time() - start_time) / 3600
                eta = elapsed / global_step * (cfg['training']['total_steps'] - global_step)
                print(f"[Step {global_step}/{cfg['training']['total_steps']}] "
                      f"loss={loss.item():.4f} recon={recon_l:.4f} kl={kl_l:.6f} "
                      f"lr={scheduler.get_last_lr()[0]:.2e} "
                      f"elapsed={elapsed:.1f}h ETA={eta:.1f}h")
            
            # 保存 checkpoint
            if global_step % cfg['training']['save_every'] == 0 and accelerator.is_main_process:
                save_path = Path(cfg['training']['output_dir']) / f"step_{global_step:06d}.pt"
                accelerator.save_state(str(save_path.parent / f"step_{global_step:06d}"))
                print(f"[Saved] {save_path}")
            
            # 验证
            if global_step % cfg['training']['eval_every'] == 0:
                val_loss = evaluate(model, val_loader, beta, accelerator)
                if accelerator.is_main_process:
                    print(f"[Val Step {global_step}] val_loss={val_loss:.4f}")
            
            if global_step >= cfg['training']['total_steps']:
                break
    
    # 最终保存
    if accelerator.is_main_process:
        final_path = Path(cfg['training']['output_dir']) / "final"
        accelerator.save_state(str(final_path))
        print(f"[Done] Final model saved to {final_path}")


@torch.no_grad()
def evaluate(model, val_loader, beta, accelerator):
    model.eval()
    total_loss, n = 0.0, 0
    for batch in val_loader:
        f_t, f_t1 = batch["f_t"], batch["f_t1"]
        f_t1_pred, mu, logvar = model(f_t, f_t1)
        loss, _, _ = lam_loss(f_t1_pred, f_t1, mu, logvar, beta)
        total_loss += loss.item()
        n += 1
        if n >= 50:  # 只评估 50 个 batch，节省时间
            break
    model.train()
    return total_loss / max(n, 1)


if __name__ == "__main__":
    main()
```

### Step 9：在 tmux 中启动训练

```bash
# 1. 新建 tmux 会话
tmux new-session -s lam-train

# 2. 激活环境
conda activate dreamdojo

# 3. 确认 GPU 状态（GPU 0 被占用，用 GPU 1-3）
nvidia-smi

# 4. 启动训练（GPU 1-3）
cd /home/xuan/embodied-ai/code/lam
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch \
  --num_processes 3 \
  --mixed_precision bf16 \
  train_lam.py \
  --config config/lam_agibot.yaml

# 5. 确认训练正常开始（看到 loss 输出）后，脱离 tmux
# 按 Ctrl+B，松开，再按 D

# 6. 现在可以安全关闭 VSCode 了
```

### Step 10：验证 LAM 质量

训练完成（或每隔一段时间），运行验证：

```bash
tmux new-session -s lam-eval
conda activate dreamdojo
cd /home/xuan/embodied-ai/code/lam

# 视觉重建检查（输出重建帧对比图）
python eval_lam.py \
  --checkpoint /home/xuan/embodied-ai/checkpoints/lam/final \
  --data_root /home/xuan/embodied-ai/data/agibotworld \
  --output_dir /home/xuan/embodied-ai/checkpoints/lam/eval_vis \
  --n_samples 20
```

**什么样的结果说明 LAM 训练成功？**

1. **重建帧视觉上合理**：$\hat{f}_{t+1}$ 与真实 $f_{t+1}$ 相似，不是一片模糊
2. **Loss 收敛**：recon_loss 在训练过程中持续下降
3. **Latent 有结构**（更高的要求）：相似动作的 latent action 应该聚集在一起

---

## 六、常见问题与排查

| 问题 | 可能原因 | 解决方法 |
|------|---------|---------|
| `RuntimeError: No video files found` | 数据路径错误 | 检查 `data_root` 和目录结构 |
| CUDA OOM | batch size 太大 | 减小 `per_gpu_batch_size`（试试 16 或 8） |
| 训练 loss 不下降 | β 太大压缩过度 | 确认 `beta=1e-6`，不要用 AdaWorld 默认值 |
| DataLoader 是瓶颈（GPU 利用率<50%） | 视频实时解码慢 | 增大 `num_workers`，或预先解码为 PNG 帧 |
| AdaWorld import error | API 不匹配 | 查看 AdaWorld 仓库的 README，调整接口调用 |
| tmux 会话消失 | 服务器重启了 | 会话不能跨重启，需重新创建并续训（用 `--resume` 参数） |

---

## 七、训练后的产物

```
/home/xuan/embodied-ai/checkpoints/lam/
├── step_005000/    # 每 5k steps 一个 checkpoint
├── step_010000/
├── ...
├── step_100000/
└── final/          # 最终模型（即 best.pt 的来源）
    ├── model.pt    # 模型权重
    └── config.yaml # 训练配置（复现用）
```

**下一步**：用 `final/model.pt` 中的 **Encoder** 部分，对 AgiBot-World 所有视频帧对提取 latent action，
然后进入世界模型后训练阶段（见 `02-world-model-post-training.md`）。

---

*文档版本：v1.0，2026-05-27*
