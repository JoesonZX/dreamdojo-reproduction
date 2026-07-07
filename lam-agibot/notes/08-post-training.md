# Post-Training 指南

用 AgiBot-World Alpha 数据对 Cosmos-Predict2.5-2B 进行 action-conditioned 后训练。

---

## 什么是 Post-Training

在 zero-shot baseline（`07-cosmos-rollout.md`）基础上，加入动作条件让世界模型变得可控：

```
条件帧 f_0 + 关节角度序列 a_{0:12}
              ↓
  Cosmos-Predict2.5-2B（post-trained）
              ↓
  未来帧 f_1...f_12（跟随动作指令）
```

post-training 不改变世界模型的整体结构，只是在 DiT 的每个 block 里加入 action conditioning（通过 AdaLN），并用机器人视频+关节角度数据微调。

---

## Cosmos 框架要求的数据格式

通过阅读 `cosmos_predict2/_src/predict2/action/datasets/dataset_local.py` 确认：

### 目录结构
```
datasets/
└── agibot/
    ├── videos/
    │   ├── episode_000000.mp4    ← 每个 episode 一个 MP4
    │   ├── episode_000001.mp4
    │   └── ...
    └── annotation/
        ├── train/
        │   ├── episode_000000.json
        │   └── ...
        └── val/
            └── ...
```

### JSON Annotation 格式（每个 episode 一个文件）
```json
{
    "episode_metadata": {
        "episode_id": "episode_000000",
        "is_eval": false
    },
    "video_path": "videos/episode_000000.mp4",
    "cam_ids": ["head_color"],
    "state": [
        [x0, y0, z0, rx0, ry0, rz0],
        [x1, y1, z1, rx1, ry1, rz1],
        ...
    ],
    "continuous_gripper_state": [0.0, 0.1, 0.0, ...]
}
```

**关键**：`state` 是末端执行器位姿（xyz + euler angles，6维），**不是关节角度**。AgiBot 提供的是关节角度（14维），需要通过正运动学（forward kinematics）转换。

### Action 的计算方式
框架内部计算相对动作：
```python
# 相邻帧之间的相对末端位移
rel_xyz = prev_rotm.T @ (curr_xyz - prev_xyz)   # 平移
rel_rpy = rotm2euler(prev_rotm.T @ curr_rotm)    # 旋转（euler）
gripper  = curr_gripper                           # 夹爪状态
action = [rel_xyz, rel_rpy, gripper]              # 7维
```

---

## AgiBot 数据格式 vs Cosmos 要求

| 项目 | AgiBot 提供 | Cosmos 需要 | 转换工作 |
|------|------------|------------|---------|
| 视频 | h264 MP4（已转码）✅ | MP4 | 直接用 |
| 动作 | 关节角度 14维（HDF5）| 末端执行器位姿 6维 | 需要正运动学转换 |
| 标注格式 | HDF5 (`proprio_stats.h5`) | JSON per episode | 需要写转换脚本 |
| 夹爪状态 | 包含在 14维关节里 | 单独 `continuous_gripper_state` | 需要提取 |

**最大的工作量**：关节角度 → 末端执行器位姿的正运动学转换，需要 AgiBot 的 URDF 文件。

---

## 两种可行路线

### 路线 A：使用 Cosmos 框架的 action-conditioned 训练（完整复现）

完全使用 Cosmos 的 `scripts.train` + NeMo 框架，需要：
1. 把 AgiBot HDF5 → JSON annotation（含正运动学转换）
2. 配置 `experiment=ac_reason_embeddings_rectified_flow_2b_256_320` 或自定义
3. 用 NeMo 框架 `torchrun -m scripts.train`

**优点**：最接近 DreamDojo 论文的实现  
**工作量**：正运动学转换 + 数据格式转换脚本，约 2-3 天

### 路线 B：直接用关节角度 + 自定义训练脚本（简化路线）

绕过正运动学，直接用关节角度作为 action（14维或 7维，取单臂），写简单的 Accelerate 训练脚本：
1. 参考 `cosmos_predict2/_src/predict2/action/` 的 action conditioning 实现
2. 数据用 AgiBot HDF5 直接读取
3. 用 Accelerate + FSDP 训练

**优点**：快，1 天可以跑通  
**缺点**：动作格式与论文不完全一致，不能直接对比数值

---

## 推荐路线和理由

对于当前复现目标（验证 pipeline，建立 baseline），推荐**路线 B**先跑通，原因：

1. DreamDojo 论文也是用关节角度（只是做了 relative 转换），不是末端位姿
2. AgiBot 的 14DoF 关节角度可以直接相对化，不需要正运动学
3. 先有可运行的 baseline，再考虑与 Cosmos 框架对齐

---

## 下一步行动

1. 确认 AgiBot HDF5 数据可以正常读取（查看 proprio_stats.h5 格式）
2. 写数据转换脚本：HDF5 关节角度 → 相对动作 chunks
3. 写简化训练脚本（参考 Cosmos action conditioning 的 AdaLN 注入方式）
4. 小规模测试（5 episodes，1k steps）

---

## 参考代码位置

| 组件 | 文件 |
|------|------|
| Action conditioning 模型 | `cosmos_predict2/_src/predict2/action/networks/action_conditioned_minimal_v1_lvg_dit.py` |
| Action dataset（JSON格式）| `cosmos_predict2/_src/predict2/action/datasets/dataset_local.py` |
| Experiment config 示例 | `cosmos_predict2/experiments/base/action.py` |
| Post-training 脚本示例 | `examples/posttraining/groot/post_training_groot.py` |
| Action inference | `cosmos_predict2/_src/predict2/action/inference/inference.py` |
