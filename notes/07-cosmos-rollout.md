# Cosmos-Predict2.5 Zero-Shot Rollout 指南

## 目的

在 Post-Training 之前，用 Cosmos-Predict2.5 **原始权重**（未见过机器人数据）对 AgiBot 视频做推理，建立 baseline。

**作为对照组**：对比 Post-Training 前后模型的表现，量化微调带来的提升。

| 阶段 | 模型 | 动作条件 | 预期表现 |
|------|------|---------|---------|
| Zero-shot rollout（本文）| Cosmos-Predict2.5-2B 原始权重 | 无 | 生成合理但不受控的未来帧 |
| Post-Training 后 rollout | 微调后模型 | 有（AgiBot 关节角度）| 生成跟随动作指令的未来帧 |

---

## 环境要求

Cosmos-Predict2.5 要求 PyTorch ≥ 2.7，与当前 `.venv`（PyTorch 2.3.0）不兼容。
**解决方案：新建独立 conda 环境，不影响 LAM 训练环境。**

服务器环境确认：
- NVIDIA Driver: 590.48.01 ✅（要求 ≥570）
- CUDA: 13.1 ✅（要求 12.8）
- Python: 3.10 ✅

---

## Step 1：安装环境

```bash
# 安装 uv（Cosmos 使用 uv 管理依赖，不用 pip）
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc   # 或重开终端

# 克隆 Cosmos-Predict2.5
cd /home/xuan/embodied-ai/code
git clone https://github.com/nvidia-cosmos/cosmos-predict2.5.git cosmos-predict25
cd cosmos-predict25
git lfs pull       # 需要 git-lfs：sudo apt install git-lfs && git lfs install

# 安装依赖（自动创建 .venv，安装 PyTorch 2.7 + cu128）
uv sync --extra=cu128

# 激活环境
source .venv/bin/activate
```

---

## Step 2：下载模型权重

需要先在 HuggingFace 接受 NVIDIA Open Model License：
`https://huggingface.co/nvidia/Cosmos-Predict2.5-2B`

```bash
# 确认已登录（已登录为 JoeyZX）
hf auth whoami

# 设置缓存目录（避免占用 /home，权重约 15GB）
export HF_HOME=/home/xuan/embodied-ai/data/hf_cache

# 下载 2B 权重（首次推理时自动下载，或手动预下载）
python -c "
from huggingface_hub import snapshot_download
snapshot_download('nvidia/Cosmos-Predict2.5-2B', local_dir='/home/xuan/embodied-ai/data/cosmos-2b')
"
```

---

## Step 3：准备输入帧

从 AgiBot 视频中提取第一帧作为 conditioning frame：

```bash
# 提取 task 362 某 episode 的第一帧
ffmpeg -i /home/xuan/embodied-ai/data/agibotworld/extracted/362/649552/videos/head_color.mp4 \
    -vframes 1 /home/xuan/embodied-ai/results/cosmos-rollout/cond_frame.jpg -y

# 创建推理所需的 JSON 配置
cat > /home/xuan/embodied-ai/results/cosmos-rollout/agibot_input.json << 'EOF'
{
    "inference_type": "image2world",
    "name": "agibot_zero_shot",
    "prompt": "A robot arm performing a manipulation task on a table",
    "input_path": "/home/xuan/embodied-ai/results/cosmos-rollout/cond_frame.jpg"
}
EOF
```

---

## Step 4：运行 Zero-Shot 推理

```bash
cd /home/xuan/embodied-ai/code/cosmos-predict25
source .venv/bin/activate

# Zero-shot image2world（无动作条件）
CUDA_VISIBLE_DEVICES=1 python examples/inference.py \
    -i /home/xuan/embodied-ai/results/cosmos-rollout/agibot_input.json \
    -o /home/xuan/embodied-ai/results/cosmos-rollout/ \
    --inference-type=image2world \
    --model=2B
```

**预计推理时间**：RTX 6000 Ada 48GB，估计 30-90 分钟/clip（参考：L40S 约 43分钟）

---

## Step 5：分析结果

推理完成后，在 `/home/xuan/embodied-ai/results/cosmos-rollout/` 查看生成的 MP4。

**评估角度**：
1. **视觉质量**：生成的帧是否清晰、物理合理？
2. **动作可控性**：在没有动作条件下，生成的未来帧是否随机漂移？
3. **场景理解**：模型是否"理解"机器人操作场景（毕竟预训练没见过机器人）？

---

## 结果记录

> 待填写（推理完成后更新）

- 推理时间：
- 输出视频：`results/cosmos-rollout/agibot_zero_shot.mp4`
- 视觉观察：
- 与 Post-Training 后的对比：（Post-Training 完成后补充）

---

## 下一步

1. ✅ 环境配置
2. ✅ 模型下载
3. ✅ Zero-shot rollout
4. → Post-Training：用 AgiBot 关节数据微调 Cosmos-Predict2.5（见 `08-post-training.md`）
