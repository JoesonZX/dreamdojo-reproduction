# DreamDojo 复现与实验指南

**目标**：在 AgiBot-World alpha 数据集上复现 DreamDojo 核心流程：
1. 训练 Latent Action Model（LAM）
2. 用 Cosmos-Predict2.5 进行人类视频预训练（可用 EgoDex 替代）
3. 用 AgiBot-World 机器人关节数据 + 视频进行后训练（Post-Training）
4. 蒸馏（可选，资源充足时进行）

---

## 阶段0：环境配置

### 0.1 硬件需求

| 阶段 | 论文配置 | 最低可行配置 |
|------|---------|------------|
| LAM 训练（400k steps） | 未明确，估计 32+ H100 | 8× A100 80G（可行但慢） |
| 世界模型预训练（140k steps） | 256× H100 | **跳过，直接用 Cosmos-Predict2.5 权重** |
| 后训练（50k steps） | 128× H100 | 8× A100 80G（batch 相应缩小） |
| 蒸馏（13k steps） | 64× H100 | 4× A100 80G |
| 推理 | 1× RTX 5090 | 1× A100 40G |

**实际建议**：跳过全量预训练，直接从 Cosmos-Predict2.5 公开权重出发，重点训练 LAM + 后训练。

### 0.2 软件环境

```bash
# 创建 conda 环境
conda create -n dreamdojo python=3.10 -y
conda activate dreamdojo

# PyTorch（根据 CUDA 版本选择）
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121

# 基础依赖
pip install transformers accelerate diffusers einops
pip install opencv-python imageio imageio-ffmpeg
pip install numpy scipy tqdm wandb

# Cosmos 相关（NVIDIA 开源）
# 参考 NVIDIA Cosmos-Predict 仓库安装说明
pip install cosmos-tokenizer  # WAN2.2 tokenizer
```

### 0.3 目录结构

```
embodied-ai/
├── notes/                          # 笔记（当前目录）
├── code/
│   ├── latent_action_model/        # LAM 训练代码
│   ├── world_model/                # 世界模型训练代码
│   └── distillation/               # 蒸馏代码
└── data/
    ├── egodex/                     # EgoDex 数据集
    ├── agibotworld/                # AgiBot-World alpha
    └── processed/                  # 预处理后数据
```

---

## 阶段1：数据准备

### 1.1 下载 EgoDex（人类视频预训练用）

EgoDex 是公开数据集，包含 829h 第一人称灵巧操作视频（Apple Vision Pro 采集）。

```bash
# 参考 EgoDex 官方仓库获取下载链接
# arXiv: 2505.11709
# 数据包含: 视频 + 精确 3D 手部姿态（MANO格式）
```

**注意**：EgoDex 提供 MANO 手部姿态，可作为潜在动作模型训练的 GT 参考（论文中用于对比实验），但在 LAM 训练中我们只使用视频本身（自监督）。

### 1.2 下载 AgiBot-World Alpha

```bash
# AgiBot-World 官方渠道
# arXiv: 2503.06669
# 数据格式: 视频 + 机器人关节角度（action）+ 语言标注
# 包含: 87 种技能, 1M 条轨迹, 2.9k 小时
```

数据格式通常为：

```
agibotworld/
├── episode_000000/
│   ├── video.mp4              # 第一人称视频（640×480 建议）
│   ├── actions.npy            # 关节角度序列 [T, DoF]
│   └── annotation.json        # 任务语言描述
└── ...
```

### 1.3 数据预处理

#### 视频处理（LAM 训练用）

```python
# 统一处理流程
# 1. 中心裁剪 + 缩放到 320×240（LAM）或 640×480（世界模型）
# 2. 按随机因子 {1,2,3,4} 时序降采样，捕捉不同速度的运动
# 3. 提取连续帧对 (f_t, f_{t+1}) 用于 LAM 训练

import cv2
import numpy as np

def preprocess_video(video_path, target_size=(320, 240), downsample_factor=None):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # 中心裁剪
        h, w = frame.shape[:2]
        crop_size = min(h, w)
        y1, x1 = (h - crop_size) // 2, (w - crop_size) // 2
        frame = frame[y1:y1+crop_size, x1:x1+crop_size]
        frame = cv2.resize(frame, target_size)
        frames.append(frame)
    cap.release()
    
    if downsample_factor is None:
        downsample_factor = np.random.choice([1, 2, 3, 4])
    return frames[::downsample_factor]
```

#### AgiBot 动作处理（后训练用）

```python
# 将绝对关节角度转为相对动作（论文核心改进之一）
def to_relative_actions(abs_actions, chunk_size=4):
    """
    abs_actions: [T, DoF] 绝对关节角度
    返回: [T//chunk_size, chunk_size*DoF] 相对动作 chunks
    """
    # 每个 latent frame（4个时间步）的起始姿态作为基准
    relative_actions = []
    for i in range(0, len(abs_actions) - chunk_size, chunk_size):
        baseline = abs_actions[i]  # 每 chunk 开头的姿态
        chunk = abs_actions[i:i+chunk_size] - baseline  # 相对位移
        relative_actions.append(chunk.flatten())  # [chunk_size * DoF]
    return np.array(relative_actions)
```

---

## 阶段2：训练潜在动作模型（LAM）

### 2.1 LAM 架构实现

LAM 是一个基于时空 Transformer 的 VAE，参考 Genie (Bruce et al., 2024) 的 ST-Transformer 实现。

```python
import torch
import torch.nn as nn
from einops import rearrange

class LatentActionEncoder(nn.Module):
    """
    输入: 连续两帧 (f_t, f_{t+1})，图像大小 320x240
    输出: 32维连续潜在动作 a_hat_t
    """
    def __init__(self, img_size=(240, 320), patch_size=16, dim=1024, 
                 depth=24, heads=16, latent_dim=32):
        super().__init__()
        self.patch_embed = nn.Conv2d(6, dim, patch_size, patch_size)  # 2帧concat=6通道
        num_patches = (img_size[0]//patch_size) * (img_size[1]//patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, dim))
        
        # 时空 Transformer blocks
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, heads, dim*4, batch_first=True),
            num_layers=depth
        )
        # 投影到均值和方差
        self.to_mu = nn.Linear(dim, latent_dim)
        self.to_logvar = nn.Linear(dim, latent_dim)
    
    def forward(self, f_t, f_t1):
        x = torch.cat([f_t, f_t1], dim=1)  # [B, 6, H, W]
        x = self.patch_embed(x)             # [B, dim, h, w]
        x = rearrange(x, 'b d h w -> b (h w) d')
        x = x + self.pos_embed
        x = self.transformer(x)
        x = x.mean(dim=1)  # global pooling
        return self.to_mu(x), self.to_logvar(x)


class LatentActionDecoder(nn.Module):
    """
    输入: a_hat_t (32维) + f_t
    输出: 重建的 f_{t+1}
    """
    def __init__(self, img_size=(240, 320), patch_size=16, dim=1024, 
                 depth=24, heads=16, latent_dim=32):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, patch_size, patch_size)
        num_patches = (img_size[0]//patch_size) * (img_size[1]//patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, dim))
        self.action_proj = nn.Linear(latent_dim, dim)
        
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(dim, heads, dim*4, batch_first=True),
            num_layers=depth
        )
        # 解码回像素
        self.to_pixels = nn.ConvTranspose2d(dim, 3, patch_size, patch_size)
    
    def forward(self, a_hat, f_t):
        # 将 f_t 编码为 patch tokens
        x = self.patch_embed(f_t)
        x = rearrange(x, 'b d h w -> b (h w) d')
        x = x + self.pos_embed
        
        # 将 action 作为 memory
        action_mem = self.action_proj(a_hat).unsqueeze(1)  # [B, 1, dim]
        x = self.transformer(x, action_mem)
        
        # 解码
        h, w = f_t.shape[2] // 16, f_t.shape[3] // 16
        x = rearrange(x, 'b (h w) d -> b d h w', h=h, w=w)
        return self.to_pixels(x)


class LatentActionModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = LatentActionEncoder(**kwargs)
        self.decoder = LatentActionDecoder(**kwargs)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, f_t, f_t1):
        mu, logvar = self.encoder(f_t, f_t1)
        a_hat = self.reparameterize(mu, logvar)
        f_t1_pred = self.decoder(a_hat, f_t)
        return f_t1_pred, mu, logvar
    
    def encode(self, f_t, f_t1):
        """推理时提取 latent action"""
        mu, _ = self.encoder(f_t, f_t1)
        return mu  # 推理时用均值，不采样
```

### 2.2 LAM 损失函数

```python
def lam_loss(f_t1_pred, f_t1_gt, mu, logvar, beta=1e-6):
    # 重建损失（MSE 或感知损失）
    recon_loss = nn.functional.mse_loss(f_t1_pred, f_t1_gt)
    # KL 散度
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss
```

### 2.3 LAM 训练配置

```yaml
# lam_config.yaml
model:
  img_size: [240, 320]
  patch_size: 16
  dim: 1024
  depth: 24
  heads: 16
  latent_dim: 32

training:
  batch_size: 256
  total_steps: 400000
  lr: 2.5e-5
  weight_decay: 0.01
  optimizer: adamw
  beta: 1.0e-6
  
  # 数据
  datasets: [egodex, agibotworld, inlab]  # 根据可用数据调整
  downsample_factors: [1, 2, 3, 4]        # 随机时序降采样

  # 日志
  log_interval: 100
  save_interval: 10000
  eval_interval: 5000
```

```bash
# 启动 LAM 训练
accelerate launch --num_processes 8 \
    train_lam.py \
    --config lam_config.yaml \
    --output_dir ./checkpoints/lam
```

**预期训练时长**：8× A100 80G 约 3-5 天（400k steps, batch=256）。

---

## 阶段3：世界模型后训练（核心阶段）

### 3.1 加载 Cosmos-Predict2.5

```python
# 从 NVIDIA 开源仓库加载 Cosmos-Predict2.5
# 参考: https://github.com/nvidia/Cosmos（2B 或 14B 版本）
from cosmos_predict import CosmosPredict25

model = CosmosPredict25.from_pretrained("nvidia/cosmos-predict2.5-2B")
```

### 3.2 添加动作条件模块

```python
class ActionConditionedWorldModel(nn.Module):
    """
    在 Cosmos-Predict2.5 基础上添加 action conditioning
    """
    def __init__(self, cosmos_model, action_dim, hidden_dim=1024):
        super().__init__()
        self.backbone = cosmos_model
        
        # 动作 MLP：将 relative action chunks 投影到 timestep embedding 维度
        # 最后一层初始化为零（论文关键细节）
        self.action_mlp = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.backbone.timestep_embed_dim)
        )
        # 零初始化最后一层
        nn.init.zeros_(self.action_mlp[-1].weight)
        nn.init.zeros_(self.action_mlp[-1].bias)
    
    def forward(self, x_t, t, condition_frames, actions):
        """
        x_t: 噪声视频 latent [B, T, C, H, W]
        t: 扩散时间步
        condition_frames: 第一帧 [B, 1, C, H, W]
        actions: 相对动作 chunks [B, T, action_chunk_dim]
        """
        # 将 action 投影并注入到每个 latent frame
        action_embeds = self.action_mlp(actions)  # [B, T, embed_dim]
        
        # 通过修改 timestep embedding 实现 chunked action injection
        # action_embeds 与 timestep embed 相加后送入 AdaLN
        return self.backbone(x_t, t, condition_frames, 
                            extra_timestep_embed=action_embeds)
```

### 3.3 后训练数据 Pipeline（AgiBot-World）

```python
class AgibotWorldDataset(torch.utils.data.Dataset):
    """
    加载 AgiBot-World 数据，输出世界模型后训练所需格式
    """
    def __init__(self, data_dir, lam_model, seq_len=13, 
                 img_size=(480, 640), action_freq=10):
        self.data_dir = data_dir
        self.lam = lam_model  # 用于提取 latent actions（验证用）
        self.seq_len = seq_len  # 13帧: 1帧条件 + 12帧预测
        self.img_size = img_size
        self.episodes = self._load_episode_list()
    
    def __getitem__(self, idx):
        episode = self.episodes[idx]
        
        # 加载视频帧（640×480, 采样 ~10Hz）
        frames = self._load_frames(episode['video_path'])  # [T, 3, H, W]
        
        # 加载并处理动作
        abs_actions = np.load(episode['action_path'])   # [T, DoF]
        rel_actions = to_relative_actions(abs_actions)  # [T//4, 4*DoF]
        
        # 随机截取 seq_len 段
        start = np.random.randint(0, len(frames) - self.seq_len)
        video_clip = frames[start:start + self.seq_len]    # [13, 3, H, W]
        action_clip = rel_actions[start//4:(start + self.seq_len)//4]  # [12, ...]
        
        return {
            'condition_frame': video_clip[0],    # 第一帧作为条件
            'target_frames': video_clip[1:],     # 后12帧作为预测目标
            'actions': action_clip,               # 对应的相对动作 chunks
            'task_text': episode['annotation']   # 可选的语言条件
        }
```

### 3.4 后训练损失（Flow Matching + Temporal Consistency）

```python
def temporal_consistency_loss(pred_velocities, gt_velocities):
    """
    pred_velocities, gt_velocities: [B, K, ...]
    K 是视频 latent 序列长度
    """
    pred_diff = pred_velocities[:, 1:] - pred_velocities[:, :-1]
    gt_diff = gt_velocities[:, 1:] - gt_velocities[:, :-1]
    return torch.mean((pred_diff - gt_diff) ** 2)


def world_model_loss(model, batch, lambda_temporal=0.1):
    frames = batch['target_frames']           # [B, T, C, H, W]
    actions = batch['actions']                # [B, T, action_dim]
    condition = batch['condition_frame']      # [B, C, H, W]
    
    # 编码到 latent space
    x = encode_to_latent(frames)             # 用 WAN2.2 tokenizer
    
    # 加噪
    t = torch.rand(x.shape[0])
    eps = torch.randn_like(x)
    x_t = (1 - t) * x + t * eps             # flow matching 插值
    v_gt = eps - x                           # 目标速度
    
    # 前向
    v_pred = model(x_t, t, condition, actions)
    
    # Flow matching 损失
    loss_flow = torch.mean((v_pred - v_gt) ** 2)
    
    # 时序一致性损失
    loss_temporal = temporal_consistency_loss(v_pred, v_gt)
    
    return loss_flow + lambda_temporal * loss_temporal
```

### 3.5 后训练配置与启动

```yaml
# post_training_config.yaml
model:
  base: cosmos-predict2.5-2B      # 或 14B
  action_dim: 48                   # 4 * DoF（AgiBot DoF 需确认，通常 12DoF）
  
training:
  batch_size: 512
  total_steps: 50000
  lr: 1.6e-4
  weight_decay: 0.1
  optimizer: adamw
  lambda_temporal: 0.1
  
  # 后训练时重新初始化 action MLP 第一层
  reinit_action_mlp_first_layer: true
  
  # 输入格式
  seq_len: 13
  img_size: [480, 640]
  sample_freq_hz: 10
  
  # EMA
  use_ema: true
  ema_decay: 0.9999

data:
  robot: agibotworld_alpha
  action_type: relative            # 使用相对动作
  chunk_size: 4                    # 每 chunk 4 个动作
```

```bash
# 启动后训练（8 GPU 版本，batch size 相应缩小）
accelerate launch --num_processes 8 \
    train_world_model.py \
    --config post_training_config.yaml \
    --pretrained_model nvidia/cosmos-predict2.5-2B \
    --lam_checkpoint ./checkpoints/lam/best.pt \
    --output_dir ./checkpoints/world_model
```

---

## 阶段4：蒸馏（可选，用于实时应用）

### 4.1 Warmup 阶段

```python
# 生成 10k 个 Teacher ODE 轨迹
def generate_ode_trajectories(teacher, dataset, n_trajectories=10000):
    trajectories = []
    for batch in dataloader:
        with torch.no_grad():
            x0 = teacher.generate_ode(batch, num_steps=35)  # 35步 ODE
        trajectories.append(x0)
    return trajectories

# Warmup 训练：Student 回归到 Teacher 的 ODE 解
def warmup_loss(student, teacher_x0, batch_xt, t):
    student_pred = student(batch_xt, t, use_causal_attn=True)
    return torch.mean((student_pred - teacher_x0) ** 2)
```

### 4.2 Distillation 阶段

```python
# Student 用自己的历史帧作为上下文（不再用 Teacher 的帧）
# 训练时生成 13~49 帧，只在最后 13 帧计算 KL 损失
def distillation_step(student, teacher, fake_score_model, batch):
    # Student 自回归生成 N' 帧（13~49 随机）
    n_frames = np.random.randint(13, 50)
    with torch.no_grad():
        student_frames = student.generate_autoregressive(
            batch['condition_frame'], 
            batch['actions'], 
            n_frames=n_frames,
            use_own_context=True  # 用自己生成的历史帧
        )
    
    # 只在最后 13 帧计算分布匹配损失
    target_frames = student_frames[:, -13:]
    
    # KL 分布匹配（通过 score 函数梯度）
    x_t = add_noise(target_frames, t)
    s_real = teacher(x_t, t)    # Teacher score
    s_fake = fake_score_model(x_t, t)  # Fake score（在 student 生成数据上训练）
    
    # 分布匹配梯度
    grad = -(s_real - s_fake)
    return grad  # 直接用于反向传播
```

---

## 阶段5：评估

### 5.1 自动指标评估

```python
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips

lpips_fn = lpips.LPIPS(net='alex')

def evaluate_model(model, eval_dataset, n_rounds=3):
    """
    生成 100 个视频（每个 49 帧），计算 PSNR/SSIM/LPIPS
    与论文一致：自回归生成，每次以上一个预测的最后帧作为新的条件帧
    """
    all_psnr, all_ssim, all_lpips = [], [], []
    
    for _ in range(n_rounds):
        for batch in eval_dataset:
            # 自回归生成 49 帧（3 个 chunk，每 chunk 13 帧）
            generated = autoregressive_generate(model, batch, total_frames=49)
            
            psnr = peak_signal_noise_ratio(batch['gt_frames'], generated)
            ssim = structural_similarity(batch['gt_frames'], generated, 
                                         multichannel=True)
            lp = lpips_fn(to_tensor(batch['gt_frames']), 
                          to_tensor(generated)).item()
            
            all_psnr.append(psnr)
            all_ssim.append(ssim)
            all_lpips.append(lp)
    
    return {
        'PSNR': np.mean(all_psnr),
        'SSIM': np.mean(all_ssim),
        'LPIPS': np.mean(all_lpips)
    }
```

### 5.2 Policy Evaluation 验证

```python
# 使用 Pearson 相关系数验证世界模型作为 policy evaluator 的可靠性
from scipy.stats import pearsonr

def policy_eval_correlation(world_model, policy_checkpoints, real_success_rates):
    """
    real_success_rates: 真实机器人上各 checkpoint 的成功率
    返回: Pearson r 和 MMRV
    """
    sim_success_rates = []
    for ckpt in policy_checkpoints:
        policy = load_policy(ckpt)
        sim_rate = evaluate_in_world_model(world_model, policy, n_episodes=20)
        sim_success_rates.append(sim_rate)
    
    r, p = pearsonr(sim_success_rates, real_success_rates)
    mmrv = compute_mmrv(sim_success_rates, real_success_rates)
    return r, mmrv
```

---

## 关键实现注意事项

### 重要细节（容易踩坑的地方）

1. **Action MLP 零初始化**：最后一层必须初始化为全零，否则预训练模型状态在训练初期会被扰动，导致物理建模变差。
   ```python
   nn.init.zeros_(action_mlp[-1].weight)
   nn.init.zeros_(action_mlp[-1].bias)
   ```

2. **相对动作的基准对齐**：相对动作以每个 latent frame（每4个时间步）的开始姿态为基准，不是整个序列的起始姿态。确保数据处理时 chunk 边界与 latent frame 边界对齐。

3. **Chunked 注入的时序对齐**：WAN2.2 tokenizer 时序压缩比为 4，即 latent frame `x^i` 对应像素空间帧 `f_{4i:4i+4}`。动作 chunk 必须与对应的 latent frame 对齐注入，不能全局广播。

4. **EMA 权重**：推理时必须使用 EMA 权重，而不是最后的训练权重。

5. **空文本条件**：预训练阶段论文固定使用空文本（empty prompt），后训练时可以加入任务描述。

6. **数据采样频率**：后训练时 AgiBot 数据采样约 10 Hz，这个频率影响动作 chunk 的时间跨度，需要和评估时保持一致。

### 资源节约方案

如果计算资源有限（比如只有 4-8 块 GPU）：

- **跳过预训练**：直接用 Cosmos-Predict2.5 公开权重，只做 LAM + 后训练
- **减少 LAM 训练步数**：从 400k 减到 100k（会降低 latent action 质量，但可作为初步验证）
- **减小 batch size**：相应增加梯度累积步数维持等效 batch size
- **使用 2B 模型**：14B 效果更好但资源需求约 7× 更高
- **使用 bf16 混合精度**：显著节省显存

### 预期结果与论文对比

复现时可以参考以下基准指标（Table 3 中 In-lab only 行，最容易复现的设置）：

| 指标 | 目标值（In-lab Eval） |
|------|---------------------|
| PSNR | ~20.9 |
| SSIM | ~0.776 |
| LPIPS | ~0.219 |

如果只用 AgiBot 后训练（不做人类视频预训练），对应"w/o pretrain"行：PSNR ~20.3，可作为 baseline。

---

## 参考资源

- **Cosmos-Predict2.5**（基础模型）：NVIDIA 官方 GitHub
- **WAN2.2 tokenizer**：随 Wan 模型开源
- **Genie (Bruce et al., 2024)**：ST-Transformer 的参考实现（LAM 架构参考）
- **AdaWorld (Gao et al., 2025)**：LAM 用于世界模型的直接前驱工作，代码可能部分公开
- **Self Forcing (Huang et al., 2025)**：蒸馏流水线参考实现
- **EgoDex**：arXiv 2505.11709，公开数据集
- **AgiBot-World**：arXiv 2503.06669，alpha 版公开

---

*指南版本：v1.0，2026-05-27*  
*基于 DreamDojo arXiv:2602.06949v1*
