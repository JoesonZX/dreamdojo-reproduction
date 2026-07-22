# 准备工作记录（LAM 动作解耦实验）

记录在正式训练前已完成的全部调研、决策、实现与验证。设计细节见 [`design.md`](design.md)，
运行命令见 [`../README.md`](../README.md)。

## 1. 调研结论（决定实验可行性的关键事实）

- **LAM 结构**：`code/adaworld/lam/lam/modules/lam.py` 的 VAE。两帧→32维 latent；
  解码器用第 t 帧 + latent 重建第 t+1 帧。损失 = 重建 MSE + β·KL（β=1e-6）。
  连续高斯 VAE（非 VQ）。
- **真实动作标签可用性**（关键）：
  - **EgoDex 可用**：每个 `{task}/{idx}.mp4` 有同名 `.hdf5`，含 `transforms/<joint>`
    形如 `[T,4,4]` 的 SE(3) 位姿（69 个关节，含 `leftHand`/`rightHand`），与 mp4 帧
    **1:1 对齐**。已用 h5py 实测确认。
  - **AgiBot 不可用**：磁盘上只有视频/深度，**没有** proprioception/关节角文件（未下载）。
  - 结论：监督实验**只能用 EgoDex**。
- **真实动作维度**：post-train(Cosmos) 用 7/14/56 维，但那是**已暂缓的独立阶段**，
  不约束 LAM 监督。本实验自定义 **18 维** = 双手各 [Δ平移(3) + 6D旋转(6)]。
  （仓库内不存在"22维"，22 仅出现在无关的 probe 分类数。）

## 2. 已确认决策

| 项目 | 决策 |
|------|------|
| 数据集 | 仅 EgoDex（test split，111 task，~3243 视频） |
| 真实动作 | 18 维，`T_rel = inv(T_t)·T_{t+skip}` 每手 Δp(3)+6D(6)；逐维 z-score |
| latent 维度 | baseline=32、exp1=32、exp2=40(32+8)、exp3=40 |
| 模型规模 | model_dim=512，enc/dec=8，heads=8（小模型，约 60.6M 参数） |
| 训练步数 | 20k，四个 run 一致 |
| 初始化 | 全部从头，seed=42 |
| 下游世界模型(Eval3) | 暂缓，用重建 PSNR 作代理 |
| 代码组织 | 不改原码，独立包 vendoring |

四个 run 每步只改一个变量：baseline→exp1 加动作 L2；exp1→exp2 加 8 维环境子空间；
exp2→exp3 加独立性损失。

## 3. 已实现文件（全部新建，原代码零改动）

```
code/lam_disentangle/
├── README.md
├── run_all.sh                  # tmux 并行启动 / --status / --kill
├── notes/{design.md, preparation.md}
└── code/
    ├── dataset.py              # EgoDex + 18维动作加载（+ 动作统计缓存）
    ├── model.py                # LAM + 环境子空间 + 动作预测头（只读 import 原 blocks）
    ├── train.py                # 训练 + action loss + decorrelation loss
    ├── eval.py                 # reconstruction / visualize / action_probe / perturb
    └── config/{baseline,exp1_action,exp2_split,exp3_indep}.yaml
```

实现要点：
- **dataset.py**：动作在 `__getitem__` 末尾、即 skip/t clamp **之后**用最终 (t,skip) 计算，
  确保与加载的两帧严格对齐；动作均值/方差从全体视频确定性采样后缓存到
  `data/egodex/test/action_stats_18d.npz`，train/eval 共享同一归一化。
- **model.py**：`action_part = latent_dim - env_dim`；`fc` 维度不变（32+8 只是切片解释）；
  `action_predictor` 仅在 `action_head=true` 时存在，且只吃 `z_mu[:, :action_part]`。
- **train.py**：`action_loss = MSE(pred, gt)`；`decorrelation_loss` = 动作/环境子空间的
  交叉协方差平方和（VICReg/Barlow 风格）；按 config 标志接入，日志含 action/indep 项。

## 4. 验证结果（均通过）

1. **数据自检**：action 形状 `[B,18]`、全有限；`_compute_action` 与手算逐位一致
   （skip=1/2 均验证）；统计缓存成功生成。
2. **损失/结构**：baseline **无** `action_predictor`、仅 recon+kl；exp3 有
   `pred_action [B,18]`、action_loss、decorr，三者有限，反向传播正常。
3. **冒烟训练**：exp3 `--dry_run` 跑通（latent=40 action_part=32 env_dim=8，60.6M 参数，
   训练对 73.9 万 / 验证 8.0 万）。
4. **评估管线**：`action_probe`（Ridge 回归 + 2×2 分类矩阵）与 `perturb`（解码网格 PNG）
   均跑通；随机权重下 R² 为负、分类近 chance，符合未训练预期。

## 4b. 数据吞吐瓶颈与修复（重要）

首次并行启动后发现 **GPU 利用率 0–5%，卡在 step 0**。诊断：

- 单 worker `__getitem__` 在四路并行下达 **1540 ms/样本**；隔离测试 42 ms。
- 根因 = EgoDex 视频是 **1920×1080 mpeg4**，随机跳帧解码本就慢，且
  **4 run × 8 worker = 32 个 1080p 解码进程**抢 CPU/内存带宽 + 冷盘缓存，雪上加霜。
- 原始 DreamDojo 用 700M 大模型，前向慢掩盖了数据慢；本实验小模型(60M)前向极快，瓶颈暴露。

**修复（两部分）**：
1. **预转码**：`transcode_240p.sh` 把 3243 个视频一次性转成 **240p 全关键帧(all-intra)
   h264**，居中裁剪到 4:3 后缩放到 320×240；HDF5 用软链接放到新目录。
   结果：15GB→2.7GB，0 失败，逐帧对齐校验通过（src/dst 帧数一致）。
   数据目录 `data/egodex/test_240p/`，4 个 config 的 `data_root` 已指向它。
2. **dataset.py 优化**：decord 用 `native` bridge；`_load_pair` 改为单次打开 +
   `get_batch([t,t+skip])`；帧数缓存进 index，`__getitem__` 不再重复打开视频。

**效果**：`__getitem__` 降到 **8.8 ms/样本（114 samp/s 单 worker）**，index 规模
(739,559) 与动作统计与原始**完全一致**（帧数零变化，结论可比）。8 worker ≈ 900 samp/s/run，
GPU 不再挨饿。

## 5. 运行方式

```bash
cd /home/xuan/embodied-ai
bash code/lam_disentangle/run_all.sh          # 四个 run 并行，各占一卡，各自 tmux 会话
bash code/lam_disentangle/run_all.sh --status # 查看进度
```

输出：`checkpoints/lam-dis/{run}/`（含 `step_*` checkpoint、`train.log`、`log_imgs/`）。
训练后按 README 跑 4 个 eval，汇总
{动作子空间 R²、环境子空间 R²、task acc、ep acc、PSNR} + t-SNE + 扰动网格。
