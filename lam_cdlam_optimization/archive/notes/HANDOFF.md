# LAM 动作解耦 —— 工作总结与交接（HANDOFF）

> 用途：把当前进展、代码改动、实验结果、结论与下一步一次性交接给 Codex/后续开发。
> 目录：`/home/xuan/embodied-ai/lam_disentangle/`。Python：`/home/xuan/.venv/bin/python`。
> GPU：4×48GB；700M 模型 batch 需 ≤16（对比法用 micro-batch 8×accum2）。
> 数据：EgoDex `data/egodex/test_240p`（80,509 val framepairs，111 task，~85 可用 verb）。

---

## 0. 一句话背景
LAM 是把连续两帧编码成连续 latent action 的 VAE。目标：让 latent 的**动作子空间 z_a** 干净地
携带"动作"、把"场景/外观"赶到**环境子空间 z_e**。本轮工作回答导师三个追问 + 探索"用对比替代 KL 分离"。

---

## 1. 本轮完成的三个问题（Q1/Q2/Q3）

### Q1 重建 tradeoff（PSNR）—— 已答，写在 `notes/analysis-recon-perdim.md`
- 补测了缺失的 **raw LAM_400k 0 步重建 = 38.29 dB**（此前只有 probe）。
- 结论：动作↔重建的 tradeoff 是**一次性扰动代价**（从零 baseline→exp1 ≈ −1.2dB；微调 frozen→full
  ≈ −3dB），**不随动作 R² 成比例**；叠加解耦（KL 瓶颈）几乎零重建代价。**PSNR 不是解耦指标**
  （失败的 exp3、冻结微调 PSNR 反而最高，因为没动动作内容）。

### Q2 18 维动作的物理含义 —— 已答（纠正了两处误解）
- **不是关节，不是 4+14**。是**两只手腕/末端的 SE(3) 位姿**：每手 `[Δ平移(3)+6D旋转(6)]`，
  左手在前右手在后 → **6 维平移 + 12 维旋转**。见 `code/dataset.py::_pair_action`。
- 逐维 R² 规律（`results/exp2_700m_40k/probe_perdim.txt`）：平移≫旋转（SO(3) 线性探针难拟合）、
  右手≫左手（右手主导操作）、深度轴/6D 末列最低（最不可观测）。

### Q3 用对比替代 KL 分离 —— 导师意见成立，已实现并跑完 v1/v2/v3
- 导师论点：**KL 只约束分布（容量压缩），不做内容路由**；之前 exp4 用非对称 KL(β_a强/β_e弱) 当
  "分离器"是误用。改用 **ConLA（arXiv:2602.00557）** 的对比思路。
- ConLA 两个信号：①**动作中心 SupCon**（同动作类别拉近）②**时间反转线索**（反转帧对→运动翻转但
  外观稳定，用于分离 motion/appearance）。我们保留 recon-MSE + 动作回归头（R² 探针的锚），
  把分离机制从 KL 换成这两个对比项，KL 降为普通正则。**仅微调官方 LAM_400k，不从头训练。**

---

## 2. 代码改动（都在 `code/`，原始 LAM 仓库只读引用）

| 文件 | 改动 |
|---|---|
| `code/model.py` | 加 `contrastive`/`proj_dim` 开关；`action_proj`/`env_proj` 投影头；`encode()` 输出 L2 归一化 `z_a_proj`/`z_e_proj`（仅训练用，eval 探针仍读原始 `z_mu`）。 |
| `code/train.py` | `supcon_loss()`(Khosla SupCon)；时间反转项(**encode-only** 省显存)；配置开关 `lambda_supcon/lambda_temporal/supcon_tau/contrastive_warmup/beta`；PK 采样接线（`pk_sampler/pk_label/classes_per_batch/samples_per_class`）；SupCon 标签按 `pk_label`(task/verb) 取；日志加 `supcon/temporal`。 |
| `code/dataset.py` | verb 解析改用 `which_llm_description` 取**单一执行动词**；新增 `class PKBatchSampler`（按任意 per-index 标签做 P类×K样本 的类平衡批，可 exclude None/unknown）。 |
| `code/eval.py` | 新增 `--mode cluster`：z_a 上按 **task 和 verb 两套**报场景不变性 gap / 分离比 / silhouette；probe 分类矩阵加 `z_a→verb%`；`load_model` 支持 contrastive 头。 |

**关键实现点**：时间反转用 `accelerator.unwrap_model(model).encode(videos.flip(dims=[1]))`
（跳过解码器）；micro-batch 8×accum2 = 有效 batch16，避免第二次前向 OOM；`beta` 从 1e-6→1e-4
可把训练 KL 从 ~143 压到 ~88（限制 ‖z_mu‖ 漂移，但仍偏高）。

---

## 3. 配置与检查点

Configs（`code/config/`）：`ft_l40_full{,_5k}.yaml`(KL基线) ·
`ft_l40_contrastive{,_5k}.yaml`(v1) · `ft_l40_contrastive_v2{,_5k}.yaml`(PK+task) ·
`ft_l40_contrastive_v3{,_5k}.yaml`(PK+verb)。均：latent40(32+8)、全微调 `LAM_400k.ckpt`、lr1e-5。

检查点：`/home/xuan/embodied-ai/checkpoints/lam-dis/<run>/step_*`（5k 的均在）。
源权重：`checkpoints/pretrained/LAM_400k.ckpt`（1024/24/24, latent32, β1e-6）。
结果：`results/<run>/{probe.txt,probe_perdim.txt,recon.txt,cluster.txt}`。
一键脚本：训练+自动评估 `/tmp/run_contrastive.sh <gpu> <run>`（tmux）。

---

## 4. 最终结果（5k 步，raw LAM 为基线，KL 为主对照）

| 指标 | raw LAM(基线) | **KL ft_l40_full_5k** | 对比v1 | 对比v2(task) | 对比v3(verb) |
|---|---:|---:|---:|---:|---:|
| 动作 R²（z_a→真实动作） | 0.127 | **0.346** | 0.291 | 0.280 | 0.281 |
| 重建 PSNR (dB) | 38.29 | 33.23 | **34.62** | 33.58 | 33.54 |
| z_a→task% | 12.9 | 21.1 | 23.2 | 60.0 | 51.1 |
| z_a→ep%（场景泄漏↓） | 13.6 | 18.3 | 20.8 | 53.9 | 48.7 |
| z_a→verb%（动作可分↑） | — | — | — | — | **45.6** |
| cluster gap(task)↓ | 0.049 | 0.030 | **0.016** | 0.043 | 0.074 |
| cluster gap(verb)↓ | **0.045** | 0.051 | 0.050 | 0.135 | 0.088 |
| 分离比(verb)↑ | 1.053 | 1.037 | 1.042 | 1.067 | **1.156** |
| silhouette(verb)↑ | −0.244 | −0.393 | −0.238 | −0.155 | **−0.106** |

---

## 5. 核心结论（三个目标、三个赢家）
1. **动作可分性（z_a 按动作聚类）→ verb-SupCon(v3) 最好**：verb 分离比 1.156 / silhouette −0.106
   ≫ KL；且 R²、PSNR 与 KL 相当。**这是"用对比而非 KL 解耦"方向下最能体现动作结构的 encoder。**
2. **线性动作保真 R² → KL(0.346) 仍最强**，对比法都在 0.28–0.29（SupCon 角度几何与连续回归有张力）。
3. **场景不变性（同动作跨场景最近）→ raw/KL/v1 最好（gap≈0.05）**；SupCon 会收紧 per-episode 子簇
   反而使该 gap 变大。**v2→v3 的教训：task≈场景，必须用 verb 做正样本**（v3 verb-gap 0.088 < v2 0.135）。
- 三者是不同目标，**最终裁决需下游世界模型（Eval#3，未做）**。

---

## 6. 交接给 Codex 的下一步（按优先级）
1. **[裁决性] 下游 Eval#3**：把 z_a 作为世界模型（Cosmos）动作条件，比 KL vs v3 的可控性/迁移——
   这才是选 encoder 的最终依据。当前所有差异都是代理指标。
2. **temporal-only 消融**（`lambda_supcon=0`，留 temporal+动作MSE）：验证"场景不变性"上限、隔离
   temporal 项的贡献。
3. **改 SupCon 正样本采样消除 per-episode 收紧**：对 same-episode 对降权或设为负样本，专门压 verb-gap。
4. **拉高对比法 R²**（如需要）：SupCon 非杠杆；考虑回归感知的对比、或 SupCon 权重/warmup 扫。
5. 稳定性：`beta` 或加 `‖z_mu‖²` 惩罚进一步压制 latent 漂移（当前训练 KL 仍 ~88 偏高）。

## 7. 读这些文件即可复现全貌
- `notes/analysis-recon-perdim.md` —— Q1/Q2/Q3 + v1/v2/v3 全部结果与判定（**主报告**）。
- `notes/results-v1.md`（从零 60M/700M）、`notes/results-v2-finetune.md`（微调 KL 基线）。
- `notes/design.md`（原始四实验设计）。
- 代码：`code/{model,train,dataset,eval}.py` + `code/config/ft_l40_*.yaml`。
