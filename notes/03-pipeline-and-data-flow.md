# DreamDojo Pipeline 理解：数据流与 Rollout 指南

**目标**：在动手训练之前，先理解数据长什么样、怎么流动、模型输出什么。  
**计划顺序**：① 理解流程和数据 → ② Rollout 出结果看效果 → ③ 再做 Post-Training

---

## 1. 整体流程（三步）

```
原始视频（无标注）
      │
      ▼
┌─────────────────────┐
│  Step 1: 训练 LAM   │  ← 我们现在做的
│  输入: (f_t, f_t+1) │
│  输出: 32维 latent  │
│        action â_t   │
└──────────┬──────────┘
           │ â_t 作为"代理动作标签"
           ▼
┌──────────────────────────┐
│  Step 2: Rollout 验证    │  ← 下一步做的
│  2a. LAM rollout:        │
│      给两帧，输出 â_t    │
│      再用 â_t 重建 f_t+1 │
│  2b. Cosmos-Predict2.5   │
│      zero-shot rollout   │
│      (不 post-train 的样子)│
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Step 3: Post-Training   │  ← 之后做的
│  输入: AgiBot 视频 +     │
│        â_t（LAM提取）+   │
│        关节角度 action   │
│  模型: Cosmos-Predict2.5 │
│        2B（NVIDIA开源）  │
│  输出: action-conditioned│
│        世界模型           │
└──────────────────────────┘
```

---

## 2. 数据长什么样

### 2.1 AgiBot-World Alpha 原始数据

**下载形式**：HuggingFace 上以 `.tar` 分发，每个 tar 约 46GB，含约 220 个 episode。

**解压后结构**：
```
extracted/
└── {task_id}/                    # 如 362（某种操作任务）
    └── {episode_id}/             # 如 649552（某次完整操作录像）
        ├── videos/
        │   ├── head_color.mp4         ← 前置 RGB 摄像头，640×480，30fps，AV1编码
        │   ├── hand_left_color.mp4    ← 左手腕摄像头
        │   ├── hand_right_color.mp4   ← 右手腕摄像头
        │   ├── head_left_fisheye_color.mp4
        │   ├── head_right_fisheye_color.mp4
        │   ├── back_left_fisheye_color.mp4
        │   ├── back_right_fisheye_color.mp4
        │   └── head_center_fisheye_color.mp4
        └── depth/
            └── head_depth_XXXXXX.png  ← 深度图（逐帧 PNG）
```

**一个 episode 的规模**：
- 约 1300-1500 帧（约 43-50 秒 @30fps）
- 8 路摄像头视频 + 深度图
- 我们只用 `head_color.mp4`（最接近第一人称视角）

**注意**：AgiBot 的 `proprio_stats.h5`（关节角度）在 observations 外面，LAM 训练不需要它（自监督），Post-Training 才需要。

---

### 2.2 LAM 的数据形态

**LAM 训练的最小单元是"帧对"**：

```
原始视频（T帧）
    │
    │ 按 skip∈{1,2,3,4} 采样
    ▼
(f_t, f_{t+skip})  →  [2, 240, 320, 3]  float32 [0,1]
                                 ↑
                         center crop + resize
                         640×480 → 320×240（正方形裁剪后缩放）
```

**进入模型前的 batch 形状**：
```python
batch["videos"]  # shape: [B, 2, H, W, C] = [32, 2, 240, 320, 3]
                 #         B  T  H   W   C
                 # T=2: index 0 是 f_t，index 1 是 f_{t+1}
```

**LAM 内部数据流**：
```
batch["videos"]  [B, 2, 240, 320, 3]
    │
    │ patchify（patch_size=16）
    ▼
patches          [B, 2, N_patches, patch_dim]
                 N_patches = (240//16) × (320//16) = 15×20 = 300
                 patch_dim = 16×16×3 = 768
    │
    │ + action_prompt token（learnable）
    ▼
padded_patches   [B, 2, 301, 768]
    │
    │ SpatioTemporalTransformer（8层）
    ▼
z                [B, 2, 301, 512]
    │
    │ 取 z[:, 1:, 0]（第2帧的 action token）
    ▼
z_action         [B, 1, 512]
    │
    │ Linear → [mu, logvar]，reparameterize
    ▼
â_t              [B, 32]          ← 32维 latent action
    │
    │ 解码：â_t + f_t patches → 重建 f_{t+1}
    ▼
recon            [B, 1, 240, 320, 3]   ← 预测的下一帧
```

**LAM 的输出**：
```python
outputs = {
    "recon":  [B, 1, H, W, C],   # 重建的 f_{t+1}
    "z_mu":   [B, 32],            # latent action 均值（推理时使用）
    "z_var":  [B, 32],            # latent action 对数方差
}
```

---

### 2.3 Post-Training 的数据形态（世界模型）

Post-Training 比 LAM 复杂，用到视频序列 + 动作序列。

**输入序列**（13帧）：
```
condition frame:  f_0              [1, 3, 480, 640]  ← 第一帧，作为条件
target frames:    f_1...f_12      [12, 3, 480, 640]  ← 预测目标
actions:          a_0...a_11      [12, action_dim]   ← 对应动作
                                       ↑
                               有两种来源：
                               · LAM 提取的 â_t（人类/无标注视频）
                               · AgiBot 关节角度（机器人数据）
```

**动作的处理**（DreamDojo 的关键改进）：
```
AgiBot 原始数据: 绝对关节角度  [T, 14]  (7DoF × 2 臂，弧度)
                                │
                                │ Relative Action + Chunking
                                ▼
Post-Training 动作: 相对角度 chunk  [T//4, 4×14=56]
                    每 chunk = 4 个时间步的相对位移
                    WAN2.2 tokenizer 时序压缩比=4，chunk 对齐 latent frame
```

**在 latent space 里的数据流**：
```
target frames  [12, 3, 480, 640]
    │
    │ WAN2.2 tokenizer（时序压缩比=4）
    ▼
video latents  [3, 3, H/8, W/8]   ← 12帧 → 3个 latent frame
    │
    │ Flow Matching 加噪
    ▼
x_t（噪声 latent）
    │
    │ DiT（Cosmos-Predict2.5）
    │   + condition frame（作为 visual context）
    │   + action chunks（通过 AdaLN 注入）
    ▼
预测速度场 v_pred
    │
    │ 去噪 → 重建 video latents → decode
    ▼
生成的未来帧  [12, 3, 480, 640]
```

---

## 3. Rollout 计划

### Step 2a：LAM Rollout

LAM 训练完后做的验证：给两帧，看重建质量。

**目标**：验证 LAM 学到了有意义的 latent action（重建的 f_{t+1} 应该清晰、接近真实）。

```python
# 伪代码：LAM rollout
model.eval()
f_t   = load_frame(video, t)       # [1, 3, 240, 320]
f_t1  = load_frame(video, t+1)     # ground truth

batch = {"videos": torch.stack([f_t, f_t1], dim=1)}   # [1, 2, 240, 320, 3]
with torch.no_grad():
    out = model(batch)

recon = out["recon"][0, 0]   # [240, 320, 3]  重建的 f_{t+1}
mu    = out["z_mu"][0]        # [32]           latent action

# 可视化：左=f_t，中=f_{t+1}真实，右=重建
show_comparison(f_t, f_t1, recon)
```

**期望看到的**：重建帧与真实帧视觉上接近，PSNR ≥ 20 dB。

---

### Step 2b：Cosmos-Predict2.5 Zero-Shot Rollout

在做 Post-Training 之前，先看 Cosmos-Predict2.5 **原始模型**（未 post-train）的表现，建立 baseline。

**为什么要做**：
- 知道 post-training 前后的差距有多大
- 确认 Cosmos 环境配置正确
- 熟悉推理 API

**推理流程**（无动作条件）：
```
condition frame (f_0)
        │
        ▼
  Cosmos-Predict2.5-2B
  （zero-shot，无动作条件）
        │
        ▼
  生成的未来 12 帧
```

**需要的资源**：
- Cosmos-Predict2.5-2B 权重（NVIDIA 开源，约 15GB）
- 单卡推理需要约 20-30GB VRAM → GPU 1/2/3 任意一张可以

**rollout 命令**（待配置 Cosmos 环境后填写）：
```bash
# TODO: 配置 Cosmos-Predict2.5 环境后补充
# 参考: https://github.com/nvidia/Cosmos
```

---

### Step 2b 的预期效果对比

| 模型 | 动作条件 | 预期表现 |
|------|---------|---------|
| Cosmos-Predict2.5（zero-shot）| 无 | 生成"合理但不受控"的未来帧，动作随机 |
| DreamDojo（post-train 后）| 有（latent/真实动作）| 生成"跟随动作指令"的未来帧，可控 |

这个对比就是论文 Table 2 里 "action-free" vs "latent action" 的直观体现。

---

## 4. 当前已有数据汇总

```
/home/xuan/embodied-ai/data/agibotworld/
├── observations/                  ← 原始 tar 文件
│   ├── 359/648638-681118.tar      46GB，220 episodes（待解压）
│   ├── 362/649552-654138.tar      46GB，220 episodes（待解压）
│   └── 410/686871-686871.tar      198MB，1 episode（已解压）
└── extracted/                     ← 解压 + 转码后
    ├── 359/（162 episodes 已解压，部分完成）
    ├── 410/686871/videos/head_color.mp4   ← h264，已验证可读
    └── 362/（待解压）
```

**数据量评估**：
- 当前可用：~163 episodes，约 20 万帧对
- 目标：解压 362 后达到 ~383 episodes，约 55 万帧对
- 帧对足够做 100k 步 LAM 训练（约 52 遍数据）

---

## 5. 下一步行动顺序

```
现在   →  等待 preprocess_videos.py 完成（解压+转码 362 tar，约 1-2 小时）
            │
            ▼
Step 1  →  LAM 正式训练（100k steps，约 8-12 小时，tmux 后台）
            │
            ▼
Step 2a →  LAM Rollout：给视频帧对，可视化重建质量
            │
            ▼
Step 2b →  配置 Cosmos-Predict2.5 环境，做 zero-shot rollout
            (下载 2B 权重 ~15GB，配置推理脚本)
            │
            ▼
Step 3  →  Post-Training：用 AgiBot 关节数据 fine-tune Cosmos
```
