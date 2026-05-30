# LAM 训练环境排查记录

**时间**：2026-05-27  
**目标**：在服务器（4× RTX 6000 Ada）上用 AgiBot-World Alpha 数据跑通 LAM 训练

---

## 问题1：ModuleNotFoundError: No module named 'torch'

### 现象
```
conda activate dreamdojo
python train_lam.py ...
ModuleNotFoundError: No module named 'torch'
```

### 原因
VSCode Server 启动时会自动激活 `~/.venv`，将 `~/.venv/bin` 插入到 `PATH` 最前面，并设置 `VIRTUAL_ENV=/home/xuan/.venv`。因此即使 `conda activate dreamdojo`，`python` 命令仍指向 `.venv` 里的 Python。而所有依赖（torch、accelerate 等）都被安装到了 `.venv` 中，而不是 conda 环境里。

验证方式：
```bash
echo $VIRTUAL_ENV   # 输出 /home/xuan/.venv
which python        # 输出 /home/xuan/.venv/bin/python（不是 conda 的）
```

### 解决方案
**直接用 `.venv` 的绝对路径**，不依赖 `which python`：

```bash
# 所有 Python 调用改用绝对路径
/home/xuan/.venv/bin/python train_lam.py ...
/home/xuan/.venv/bin/accelerate launch ...

# 安装新包也用绝对路径
/home/xuan/.venv/bin/pip install <package>
```

**不要用**：
```bash
conda activate dreamdojo && python ...   # PATH 被 .venv 劫持
```

---

## 问题2：数据集路径与实际结构不符

### 现象
```
FileNotFoundError: No top.mp4 files found under .../observations/
```

### 原因
代码最初假设数据结构为：
```
observations/{task_id}/{ep_id}/videos/top.mp4
```

实际 AgiBot-World Alpha 的分发格式是 **`.tar` 压缩包**，camera 名也不同：
```
observations/{task_id}/{ep_id_range}.tar   ← HuggingFace 上下载的是 tar
```

解压后结构为：
```
{task_id}/{ep_id}/videos/head_color.mp4         ← 前置摄像头
{task_id}/{ep_id}/videos/hand_left_color.mp4
{task_id}/{ep_id}/videos/hand_right_color.mp4
{task_id}/{ep_id}/videos/head_left_fisheye_color.mp4
{task_id}/{ep_id}/videos/head_right_fisheye_color.mp4
{task_id}/{ep_id}/videos/back_left_fisheye_color.mp4
{task_id}/{ep_id}/videos/back_right_fisheye_color.mp4
{task_id}/{ep_id}/videos/head_center_fisheye_color.mp4
{task_id}/{ep_id}/depth/...
```

另外 task ID 不是 `1/2/3`，而是 `327/352/354/…/410` 等三位数。

### 解决方案
1. 下载后先解压：
```bash
mkdir -p /home/xuan/embodied-ai/data/agibotworld/extracted/{task_id}
tar -xf observations/{task_id}/{range}.tar -C extracted/{task_id}/
```

2. 更新配置：
```yaml
data:
  data_root: /home/xuan/embodied-ai/data/agibotworld/extracted
  camera: head_color   # 不是 top
```

3. 数据集 glob 改为 `*/*/videos/{camera}.mp4`（两层通配）。

---

## 问题3：AV1 视频编码，OpenCV 无法解码

### 现象
```
[av1 @ 0x...] Your platform doesn't support hardware accelerated AV1 decoding.
[av1 @ 0x...] Failed to get pixel format.
[av1 @ 0x...] Get current frame error
```
大量重复错误，`cv2.VideoCapture` 读不出帧。

### 原因
AgiBot-World Alpha 的视频用 **AV1 编码**（`av01` codec）。OpenCV 和 decord 链接的是系统旧版 ffmpeg（Ubuntu 22.04 自带 ffmpeg 4.x），其 OpenCV binding 不支持 AV1 软件解码。

验证：
```bash
ffprobe video.mp4 2>&1 | grep "Video:"
# Stream #0:0: Video: av1 (Main) ...
```

### 解决方案
用系统 `ffmpeg`（已带 libdav1d）将 AV1 转码为 h264，OpenCV 就能正常读取：

```bash
ffmpeg -y -loglevel error \
    -i input_av1.mp4 \
    -c:v libx264 -crf 18 -preset fast -an \
    output_h264.mp4
```

**批量处理**：使用项目中的 `preprocess_videos.py` 一键完成 tar 解压 + AV1→h264 转码：

```bash
/home/xuan/.venv/bin/python preprocess_videos.py \
    --tar_root  /home/xuan/embodied-ai/data/agibotworld/observations \
    --out_root  /home/xuan/embodied-ai/data/agibotworld/extracted \
    --camera    head_color \
    --n_workers 4
```

脚本有断点续传（`.done` marker），重复运行会跳过已处理的 episode。

---

## 问题4：HuggingFace CLI 参数不兼容

### 现象
```
Error: No such option '--resume-download'.
```

### 原因
服务器上安装的是新版 `hf` CLI（v1.16.4），`--resume-download` 是旧版 `huggingface-cli` 的参数，新版不支持。

### 解决方案
去掉 `--resume-download`，新版默认支持断点续传：

```bash
# 正确写法（新版 hf CLI）
hf download \
    --type dataset \
    --local-dir /home/xuan/embodied-ai/data/agibotworld \
    --include "observations/410/**" \
    agibot-world/AgiBotWorld-Alpha
```

---

## 问题5：数据太少时 train/val split 导致 train 为空

### 现象
```
RuntimeError: Dataset is empty after indexing. Found 0 videos but none had enough frames.
```

### 原因
只有 1 个视频时，`n_val = max(1, ...)` 把唯一的视频分给了 val，train 为空。

### 解决方案
当视频数 ≤ 2 时，train 和 val 都使用全部数据（用于 pipeline 验证阶段可接受）：

```python
if len(shuffled) <= 2:
    self.videos = shuffled  # 太少时 train/val 共用
elif split == "val":
    self.videos = shuffled[:max(1, n_val)]
else:
    self.videos = shuffled[max(1, n_val):]
```

---

## 训练时间估算

当前数据量（1 个视频，1298 帧对）只用于验证 pipeline，**不适合正式训练**。正式训练需要至少下载 5-10 个任务的数据。

| 数据量 | 帧对数 | 100k步预计时间（3× RTX 6000 Ada） |
|--------|--------|-----------------------------------|
| 1 视频（现在）| 1,298 | 2-4 小时，但严重过拟合 |
| 5 任务（~50 episodes）| ~65,000 | 6-12 小时 |
| 全量（~400 episodes）| ~500,000 | 1-2 天 |
| 400k 步（论文配置） | 需要大量数据 | 7-10 天 |

**建议**：下载至少 5-10 个任务（选择小的 tar，每个 1-10 GB），再跑 100k 步训练。

---

## 下载更多数据的命令

```bash
# 查看各 tar 大小，选择较小的
hf download --type dataset --dry-run agibot-world/AgiBotWorld-Alpha 2>&1 \
    | grep "\.tar" | sort -t$'\t' -k2 -h | head -20

# 下载几个小 tar（< 10GB）
hf download --type dataset \
    --local-dir /home/xuan/embodied-ai/data/agibotworld \
    --include "observations/410/**" \
    --include "observations/365/**" \
    --include "observations/359/**" \
    agibot-world/AgiBotWorld-Alpha

# 解压 + 转码
/home/xuan/.venv/bin/python /home/xuan/embodied-ai/code/lam/preprocess_videos.py \
    --tar_root /home/xuan/embodied-ai/data/agibotworld/observations \
    --out_root /home/xuan/embodied-ai/data/agibotworld/extracted \
    --camera head_color --n_workers 4
```

---

## 正式训练命令（tmux 中运行）

```bash
# 新建 tmux session（关闭 VSCode 后训练继续）
tmux new-session -s lam_train

# 启动 3 卡训练（GPU 0 被 nextgdt 占用约 44GB）
CUDA_VISIBLE_DEVICES=1,2,3 /home/xuan/.venv/bin/accelerate launch \
    --num_processes 3 --mixed_precision bf16 \
    /home/xuan/embodied-ai/code/lam/train_lam.py \
    --config /home/xuan/embodied-ai/code/lam/config/lam_agibot.yaml

# 分离 session（训练继续后台运行）
# 按 Ctrl+B，再按 D

# 重新连接查看进度
tmux attach -t lam_train
```

---

## 问题6：磁盘空间耗尽，hf download 中断

### 现象
下载 task 359/362/365 时磁盘满了（仅剩 4.4GB），下载中断：
```
df -h → /dev/nvme0n1p3  6.9T  6.6T  4.4G 100% /
```

### 原因分析
磁盘占用来源（从大到小）：
- `observations/362/`：771GB（17 个 tar，每个 46GB）——下载时没有意识到一个任务有多少 tar
- `observations/359/`：139GB（4 个 tar）
- `~/.cache/pip`：36GB（pip 下载缓存，已安装的包不受影响）
- `observations/410/`：190MB（小 tar，已解压用于测试）

**根本原因**：`hf download --include "observations/362/**"` 会下载该任务的**所有** tar 文件（17 个 × 46GB = 771GB），远超预期。在下载前没有用 `--dry-run` 估算大小。

### 解决方案

**第一步：清理 pip cache（零风险，释放 36GB）**
```bash
/home/xuan/.venv/bin/pip cache purge
```

**第二步：评估数据量需求**

每个 46GB tar 约含 220 个 episode，每 episode 约 1500 帧。  
LAM 训练需要足够的帧对但不需要穷举所有 tar：

| 保留数据 | 帧对数 | 100k 步循环次数 | 是否够用 |
|---------|--------|----------------|---------|
| 1 个小 tar（410，190MB）| 1,298 | 22,000 遍 | ❌ 严重过拟合 |
| 1 个大 tar（362 or 359，46GB）| ~333,000 | 86 遍 | ✅ 足够收敛 |
| 2 个大 tar（362+359，各 1 个）| ~555,000 | 52 遍 | ✅ 更好，跨任务多样性 |

**第三步：删除多余 tar，只保留每个任务第 1 个**
```bash
# task 362：删除 16 个多余 tar，保留第 1 个（220 episodes）
cd /home/xuan/embodied-ai/data/agibotworld/observations/362/
ls | grep -v "649552-654138.tar" | xargs rm -f

# task 359：删除后 3 个 tar，保留第 1 个（220 episodes）
cd /home/xuan/embodied-ai/data/agibotworld/observations/359/
rm -f 681122-688735.tar 688742-694016.tar 694024-694109.tar
```

清理后磁盘从 4.4GB 可用恢复到 **843GB 可用**（释放约 800GB）。

### 经验教训：下载前一定先 dry-run 估算

```bash
# 下载任何数据前，先用 --dry-run 确认大小
hf download --type dataset --dry-run \
    --include "observations/362/**" \
    agibot-world/AgiBotWorld-Alpha
# 输出示例：Will download 17 files totalling 771.0G  ← 看到这个就要三思

# 如果太大，改为只下载第 1 个 tar
hf download --type dataset \
    --include "observations/362/649552-654138.tar" \
    agibot-world/AgiBotWorld-Alpha
```

---

## 当前状态（2026-05-27）

- [x] 环境配置完成（`.venv` 中有所有依赖）
- [x] AdaWorld 代码克隆并验证可用
- [x] 数据集代码适配 AgiBot 实际格式
- [x] AV1 转码方案验证
- [x] dry run 通过（5 steps）
- [x] pipeline 验证训练完成（1 个视频，已收敛）
- [x] 磁盘清理完成（释放 800GB，现余 843GB）
- [x] task 362（220 episodes）+ task 359（220 episodes）tar 已就位
- [ ] 解压 + 转码两个 46GB tar（运行 preprocess_videos.py，约 1-2 小时）
- [ ] 用 440 个 episode 正式启动 100k 步训练
- [ ] 验证 LAM 质量（PSNR ≥ 20 dB）
