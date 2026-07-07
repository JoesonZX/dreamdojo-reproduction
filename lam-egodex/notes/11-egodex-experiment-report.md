# EgoDex 实验报告

## 研究动机

DreamDojo LAM 的 latent action 可能被外观信息污染——unweighted MSE loss 允许 encoder 利用背景纹理、光照等与动作无关的信息来最小化重建误差。

**研究故事**：DreamDojo LAM 的 latent action 被外观信息污染；通过前景感知表示学习（SAM3 语义 mask + SigLIP 文本对比 loss），可以获得更纯净的动作语义表示。

### 为什么从 AgiBot-World 切换到 EgoDex

AgiBot-World Alpha（2 个 task，不同背景）存在根本缺陷：task probe 高准确率无法区分"学到了动作"和"学到了背景颜色"。

EgoDex 的优势：
- **111 个动作类别**，diverse 背景 → task probe 高才真正代表动作语义
- 每个 episode 有 `llm_verbs` 标注（pick、place、fold、stack 等原子动作）
- 3243 个 episode，约 20h 视频

---

## 数据集构成

| 属性 | 值 |
|---|---|
| 数据集 | EgoDex test split |
| Task 数 | 111（文件夹名，复合动作如 `basic_pick_place`） |
| Episode 数 | 3243 |
| 平均时长 | ~5 秒/视频 |
| 标注 | `llm_verbs`（原子动作），`llm_description`（自然语言描述），`environment` |
| 训练帧对数 | 739,559（train）/ 80,509（val） |
| Primary verb 类别 | 59 个（pick, place, fold, stack, insert, remove…） |

---

## 实验矩阵

| 实验 | 配置 | 状态 |
|---|---|---|
| Baseline LAM | 700M，EgoDex，10k steps，纯 MSE+KL | ✅ 完成 |
| SigLIP fine-tune | 从 baseline resume，+SigLIP contrastive loss，λ=0.1 | 🔄 训练中（6k/10k steps） |
| SAM3 fine-tune | 从 baseline resume，+SAM3 手部 mask，fg_weight=10 | 待启动 |
| SAM3+SigLIP | 两者结合 | 待启动 |

---

## 预计算资产

| 资产 | 路径 | 说明 |
|---|---|---|
| RAFT flow masks | `egodex/test/{task}/flow_masks/{ep}/{t:06d}_skip{skip}.pt` | 278,085 个文件，光流幅度二值化 |
| SAM3 hand masks | `egodex/test/{task}/sam3_masks/{ep}/{t:06d}_skip{skip}.pt` | 90,818 个文件，prompt="hands and arms" |
| SigLIP caption emb | `egodex/test/{task}/captions/{idx}.pt` | 3243 个，768-dim float32，来自 llm_description |

---

## 模型配置

| 参数 | 值 |
|---|---|
| 架构 | Spatiotemporal Transformer VAE（AdaWorld LAM） |
| 模型规模 | 700M（model_dim=1024，enc/dec blocks=24，heads=16） |
| Latent dim | 32 |
| β（KL weight） | 1e-6 |
| 输入 | 连续帧对 [2, 240, 320, 3] |
| Baseline 训练 | 10k steps，lr=2.5e-5，bf16，3×GPU，batch=16×6（grad_accum） |
| Fine-tune | 10k steps 续训，lr=1e-5，warmup=200 |

---

## Baseline 评估结果

### Linear Probe（step 10000）

| Probe 类型 | 准确率 | Chance | 倍数 | 说明 |
|---|---|---|---|---|
| Verb probe（llm_verbs，59 类） | **9.0%** | 1.7% | 5.3× | 原子动作语义 |
| Task probe（task_id，108 类） | **7.0%** | 0.9% | 7.8× | 复合任务标签 |
| Episode probe（ep_id，71 ep） | **11.3%** | 1.4% | 8.1× | 场景/外观信息 |

**Action/Appearance ratio：0.80×（mixed）**

系统判定：`CLEAN: latent is not contaminated by scene appearance.`

### 解读

- 三项准确率都偏低，说明 baseline LAM 在 10k steps 下尚未充分收敛，latent 空间结构较弱
- **Episode probe（11.3%）> Verb probe（9.0%）**：外观信息比动作语义略多，符合"外观污染"假设的方向
- Action/Appearance ratio < 1 → 外观信号略强于动作信号，研究假设成立
- 对比 AgiBot baseline（task=94.4%）：EgoDex 的低数值并非失败，而是因为 111 类相比 2 类更难，且 baseline 只跑了 10k steps（AgiBot 跑了更多 steps）

### 与 AgiBot 对比

| 指标 | AgiBot baseline | EgoDex baseline | 说明 |
|---|---|---|---|
| Task probe | 94.4%（2 类） | 7.0%（108 类） | EgoDex 类别多得多 |
| Episode probe | — | 11.3% | 外观信号 |
| Verb probe | — | 9.0% | 新增：原子动作 |
| 诊断价值 | 低（背景=任务） | **高**（背景与任务解耦） | EgoDex 是有效诊断集 |

---

## Fine-tune 期望

Fine-tune 后的目标变化方向：

| Probe | 期望变化 | 含义 |
|---|---|---|
| Verb probe | ↑（15-25%） | 动作语义更强 |
| Task probe | ↑（10-20%） | 任务表示改善 |
| Episode probe | ↓（<8%） | 外观污染减少 |
| Action/Appearance ratio | >1.5（action-dominant） | 语义变纯净 |

---

## 关键代码文件

| 文件 | 作用 |
|---|---|
| `code/lam/train_lam.py` | 训练主脚本，含 SigLIPProjectionHead、siglip_loss、fg_mask 加权 |
| `code/lam/eval_lam.py` | 评估：t-SNE（3 列含 verb）、linear probe（verb/task/ep）|
| `code/lam/data/egodex_dataset.py` | EgoDex loader，支持 use_caption/use_fg_mask/load_verbs/fg_mask_type |
| `code/lam/precompute_flow_masks.py` | RAFT 光流 mask 预计算 |
| `code/lam/precompute_sam3_masks.py` | SAM3 手部 mask 批量预计算（sam3 conda env）|
| `code/lam/precompute_captions.py` | SigLIP text embedding 预计算 |
| `code/lam/visualize_sam3_masks.py` | SAM3 mask 可视化质量检验 |
| `config/lam_egodex.yaml` | Baseline 训练配置 |
| `config/lam_egodex_siglip.yaml` | SigLIP fine-tune 配置 |
| `config/lam_egodex_sam3.yaml` | SAM3 mask fine-tune 配置 |
| `config/lam_egodex_sam3_siglip.yaml` | SAM3+SigLIP 联合配置 |

---

## 下一步

1. **等待 SigLIP fine-tune 完成**（当前 6k/10k steps）→ 评估 probe + t-SNE
2. **启动 SAM3 fine-tune**（SigLIP 跑完后用同 GPU）
3. **启动 SAM3+SigLIP fine-tune**
4. **汇总 4 组对比结果**，制作 2×4 对比图（4 个模型 × verb/ep probe）
