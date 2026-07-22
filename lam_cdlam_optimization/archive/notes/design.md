# LAM 动作解耦实验 — 实施方案

> 本文件最终作为设计笔记放入 `code/lam_disentangle/notes/design.md`。

## 背景与动机

DreamDojo 的 LAM 是一个 VAE：把连续两帧编码成 32 维连续 latent action，解码器用
第 *t* 帧 + latent 重建第 *t+1* 帧。由于训练信号只有重建 MSE 和极小的 KL（β=1e-6），
学到的 latent **不保证是纯动作**——重建第二帧本身就需要环境信息，KL 又几乎不约束，
所以环境/外观信息混入 latent 是必然的。本系列实验验证：加入**动作监督**、**独立的环境
子空间**、**独立性损失**，能否得到更**纯净、更可解耦**的动作表示。

### 设计合理性评估（已确认）
- 动机成立：纯 recon+KL 的无监督解耦是 ill-posed 问题；用真实动作做监督是把它变成
  **有监督锚定**，比无监督解耦靠谱。
- 递进清晰，每步只改一个变量（见下表）。
- **关键认识**：Exp1 的"latent→动作空间 L2"只能让动作信息**可解码**，不能把环境信息
  赶出 latent（解码器仍需它重建画面），所以 32 维里动作与环境共存打架。**Exp3 才是
  核心验证**——独立性损失把环境信息从动作 32 维挤进环境 8 维。Exp2 单独不保证动作维纯净。

### 已确认决策
- **数据集：仅 EgoDex。** 唯一磁盘上带逐帧动作标签的数据集（HDF5 SE(3) 位姿，与 mp4 帧
  1:1 对齐，543 帧），111 个 task 类别适合聚类/probe。AgiBot proprioception 未下载，仅作
  无标签 baseline 路径。
- **真实动作 = 18 维。** 每帧对 (t, t+skip)，每只手：相对平移 Δp(3) + 6D 旋转(6)，来自
  `T_rel = inv(T_t)·T_{t+skip}`，双手共 18 维。逐维 z-score（平移~mm 与旋转~单位差约 100×）。
  **与 post-train 解耦**：post-train(Cosmos) 用 7/14/56 维，是另一个已暂缓阶段，
  **不需要 22 维**。
- **latent 维度**：baseline=32、Exp1=32、Exp2=40(32 动作+8 环境)、Exp3=40。理由见下。
- **规模**：小模型（`model_dim=512`，8 enc/dec blocks，8 heads），**20k 步**，各 run 一致。
- **初始化**：所有 run 从头训练（同 seed=42）。
- **下游世界模型评估(Eval#3)：暂缓。** 无 LAM→Cosmos 桥接，多日独立工作。用 LAM 解码器
  下一帧 **PSNR** 作为代理。
- **代码组织**：不改动原代码，在 `code/lam_disentangle/code/` 重写独立版本（见"实施"）。

## 四个 run（每步只改一个变量）

| Run | latent_dim | env_dim | action_head | λ_action | λ_indep | 相对上一步的唯一变化 |
|-----|-----------|---------|-------------|----------|---------|----------------------|
| Baseline | 32 | 0 | 关 | 0 | 0 | —（忠实 DreamDojo） |
| Exp1 (动作监督) | 32 | 0 | 开（监督全部 32 维） | 1.0 | 0 | +动作 L2（容量仍 32） |
| Exp2 (32+8 拆分) | 40 | 8 | 开（只监督 0:32 维） | 1.0 | 0 | +8 维环境子空间 |
| Exp3 (+独立性) | 40 | 8 | 开（只监督 0:32 维） | 1.0 | 0.02 | +独立性损失 |

约定：维度 `[0:32]`=动作子空间，`[32:40]`=环境子空间。
**为何 baseline/Exp1 用 32 而非 40**：baseline=32 是忠实论文参照；纯净度指标始终只评估
那 32 个动作维（环境维单独评），所以总维数不同不影响跨 run 比较；Exp2 多出的 8 维**是
干预本身**（"预留环境子空间"无法不增维实现），不是混淆。
**可选容量对照**：另跑一个 40 维无监督 baseline，确认 Exp2 增益非单纯多 8 维所致。

跨 run 一致：`seed=42`、`downsample_factors=[1,2]`、`img 240×320`、`per_gpu_batch`、
`grad_accum`、`lr=2.5e-5`、`warmup=1000`、调度、`total_steps=20000`、GPU 数。

## 实施

新建独立实验包 `code/lam_disentangle/`，**不修改任何原有文件**：

```
code/lam_disentangle/
├── notes/
│   └── design.md           # 本设计文档
├── code/
│   ├── model.py            # vendoring 自 adaworld/.../lam.py + 改动(env 拆分/action head)
│   ├── dataset.py          # vendoring 自 lam/data/egodex_dataset.py + 动作标签加载
│   ├── train.py            # vendoring 自 lam/train_lam.py + 新 loss
│   ├── eval.py             # vendoring 自 lam/eval_lam.py + action_probe/perturb
│   └── config/
│       ├── baseline.yaml  exp1_action.yaml  exp2_split.yaml  exp3_indep.yaml
└── README.md
```

**依赖策略**：只 vendoring（复制并修改）我们要改的 4 个文件；**不变的底层模块**
（`adaworld/lam/lam/modules/blocks.py`、`embeddings.py`）通过在脚本顶部把原仓库路径加入
`sys.path` 后**只读 import**，不复制、不修改，避免改动原代码且减少冗余。

### 1. `dataset.py`（基于 egodex_dataset.py）
- 新增 `load_actions: bool=False`、`action_dim: int=18`。
- `_rot_to_6d(R)` 与 `_compute_action(hdf5_path, t, skip)`：打开同名 `.hdf5`，只读
  `transforms/leftHand`、`transforms/rightHand`；每手 `T_rel=inv(T[t])@T[t+skip]`→`[Δp(3),6D(6)]`，
  拼 18 维；用缓存的数据集均值/方差 z-score。
- 索引元组追加 hdf5 路径；缺 hdf5 的 episode 跳过。
- **关键**：动作在 `__getitem__` **最后**、即 `skip`/`t` clamp/fallback **之后**用最终
  `(t,skip)` 计算，确保与实际加载两帧一致；读取失败复用现有随机重采样 fallback。

### 2. `model.py`（基于 lam.py）
- 构造参数 `env_dim=0`、`action_dim=0`、`action_head=False`；`latent_dim`=总维度，
  `self.fc=nn.Linear(model_dim, latent_dim*2)` 不变；`action_part=latent_dim-env_dim`。
- `action_head` 时 `self.action_predictor=Sequential(Linear(action_part,128),GELU,Linear(128,action_dim))`。
- `encode()`/`forward()` 得到 `z_mu` 后，若 `action_head` 则
  `outputs["pred_action"]=self.action_predictor(z_mu[:, :action_part])`；解码器仍用完整 `z_rep`。

### 3. `train.py`（基于 train_lam.py，仿 siglip 段接入）
- `action_loss=F.mse_loss(pred_action, batch["action"])`（T=2 ⇒ 对齐）。
- 独立性（Exp3）VICReg/Barlow 风格交叉协方差：
  ```python
  def decorrelation_loss(A, E):          # A:[B,32] E:[B,8]
      A=(A-A.mean(0))/(A.std(0)+1e-5); E=(E-E.mean(0))/(E.std(0)+1e-5)
      C=(A.T@E)/(A.shape[0]-1); return (C**2).sum()
  ```
- 主循环 `lam_loss` 后：`loss+=λ_action*l_act`，Exp3 再 `+=λ_indep*l_ind`
  （`A=z_mu[:,:32]`,`E=z_mu[:,32:]`）；新增 `running_action`/`running_indep` 日志。

### 4. `config/*.yaml`
按上表 4 个文件；设 `model_dim:512, enc/dec_blocks:8, num_heads:8, total_steps:20000`，
分别 `output_dir`/`run_name`，Exp1–3 设 `load_actions:true`。

### 5. `eval.py`（基于 eval_lam.py，复用 probe/t-SNE/recon）
- `--mode action_probe`（Eval 1+2）：冻结 encoder，收集 `z_mu`+`action`，`Ridge` 拟合
  全 `z_mu`→动作、动作子空间 `[:, :32]`→动作（应**高**）、环境子空间 `[:, 32:]`→动作（应**低**）；
  再对各子空间跑 `task_id`/`ep_id` 分类 probe，凑成 **2×2 解耦矩阵**（动作维预测动作好、
  环境维预测场景好）。
- `--mode perturb`（Eval 4）：对每动作维 vs 环境维加 ±k·σ，经 `forward` 解码存帧网格；
  预期动作维改运动、环境维改外观。
- t-SNE/PCA：复用 `eval_visualize_latents`/`eval_motion_cluster`；加 `mu_arr[:, :32]` 动作子空间面板。
- Eval 3 代理：现有 `eval_reconstruction` PSNR，每 run 一个。

## 验证（端到端）
1. **数据自检**：`EgoDexDataset(load_actions=True)` 取一 batch，断言 `action.shape==[B,18]`、有限。
2. **冒烟**：每 config 跑 ~100 步，确认各 loss 进日志、下降、无 NaN；`action_predictor` 仅在
   `action_head` 时存在。
3. **正式**：4×20k 步（+可选 40 维对照），同 seed/GPU。
4. **评估**：`action_probe`（预期动作子空间 R²≫环境子空间，差距 Baseline→Exp1→Exp2→Exp3 拉大）、
   `reconstruction` PSNR（应基本持平）、`visualize`（按 task 聚类）、`perturb`。
5. **汇总**：4(±1) run 的 {动作子空间 R², 环境子空间 R², task acc, ep acc, PSNR} 表 + t-SNE + 扰动网格。

## 关键文件（全部新建于 code/lam_disentangle/，原代码只读）
- `code/lam_disentangle/notes/design.md`
- `code/lam_disentangle/code/{model,dataset,train,eval}.py`
- `code/lam_disentangle/code/config/{baseline,exp1_action,exp2_split,exp3_indep}.yaml`
- `code/lam_disentangle/README.md`
- 只读参照：`code/adaworld/lam/lam/modules/{lam,blocks,embeddings}.py`、
  `code/lam/{train_lam,eval_lam}.py`、`code/lam/data/egodex_dataset.py`、`config/lam_egodex.yaml`
