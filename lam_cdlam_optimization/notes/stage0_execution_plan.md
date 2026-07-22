# Stage-0 执行计划(交给新会话在 auto mode 下执行)

你是在 `/home/xuan/embodied-ai/lam_disentangle` 下工作的实验执行代理。目标:完成
"阶段 0(地基)"——建立三 baseline 对照、补齐缺失的 loss 实现、跑出定稿失败清单。
先读三份背景(必读,按序):
1. `notes/research_qa_synthesis.md` — 结论、评估纪律、整体流程
2. `notes/benchmark_v2_findings.md` — 当前 benchmark 结果与四大失败模式
3. `notes/benchmark_v2_changes.md` — 每个指标的语义

## 硬性纪律(违反任何一条 = 实验作废)
1. **共享服务器**:任何 GPU 任务启动前 `nvidia-smi` 确认目标卡空闲(<1GB 且 <10% util);
   绝不动其他用户(thakk100/tang0836/lalit 等)的进程;训练一律 2 卡
   (`CUDA_VISIBLE_DEVICES=X,Y GPU_COUNT=2`),保持与既有 run 相同的有效 batch。
2. **评估纪律**:循环指标(static_response / reverse_cos / shortcut / rev_margin)只做
   机制检查,永不进 headline;每个 loss 的验收指标见文末速查表。
3. **长任务**:nohup + 日志写到 `lam_disentangle/*.log`;5k 步训练约 5.5h/个(2 卡);
   完成标志 = `checkpoints/lam-dis/<run>/step_0005000/model.safetensors` 出现。
4. 不重跑已有结果;benchmark 新输出一律写新目录,**不要覆盖 `results/lam_benchmark_v2`**。
5. 训练与 benchmark 不要同时跑在同一台机(CPU 视频解码互相拖慢 10 倍,已实测)。

## 任务(按序;T1 可与 T2/T3 并行)

### T1 补 SAM3 mask 覆盖(L_emb 的前置)
```bash
CUDA_VISIBLE_DEVICES=<空闲卡> conda run -n sam3 python code/precompute_sam3_hoc_masks.py \
  --data_root /home/xuan/embodied-ai/data/egodex/test_240p --downsample_factors 1 2
```
断点续跑(自动跳过已有文件)。当前覆盖 ~166k 个 .pt(约 20%)。
验收:`find /home/xuan/embodied-ai/data/egodex/test_240p -path "*sam3_hoc_masks*" -name "*.pt" | wc -l`
显著增长;在报告里记录最终覆盖率。T5 之前尽量把覆盖推过 80%;推不完也要记录实际值。

### T2 实现 primitive SigLIP 对比 loss(train.py)
CD-LAM Eq.10 的忠实实现:
- `L_pctr = (1/|P|) Σ_{(i,j)∈P} softplus(−y_ij · (τ · v_i·v_j + b))`
- `v = normalize(proj(z_a))`,proj 复用 model 的 contrastive 投影头
  (`contrastive: true, proj_dim: 64`,参照 supcon 路径的用法);
- τ、b 为**可学习标量**,初始化 τ=10、b=−10(SigLIP 惯例;τ 用 exp(logit_scale) 参数化保证为正);
- 标签:`from primitive_labels import primitive_of`;`primitive_of(verb_id)` 返回 None 的样本
  不参与任何配对;y=+1 同 primitive,y=−1 不同 primitive;排除自身对;
- config 开关:`lambda_primitive_ctr`(线性 warmup `primitive_ctr_warmup` 步),
  初始值见 `ft_l40_cdlam_repro_5k.yaml`;
- 日志:每 log_every 打印 `pctr` loss、正/负 pair 数(与其他 loss 同一行)。
验收(Tier-0):1k 步烟测,日志出现 `pctr` 且 loss 下降、pair 数非零
(pk_label=task 的 P2K4 批次里,同 task 4 样本同 primitive → 每批应有 ≥6 个正对)。
**注意**:`ft_l40_cdlam_repro_5k.yaml` 已引用这些 flag;旧 train.py 会静默忽略——
T4 启动前必须确认日志里有 `pctr` 项,否则训出来的是 Ours-A+L_emb,实验作废。

### T3a(高优先级)Label-free 孪生 run —— 故事定位已定为"自监督+弱标签"
用户已选定故事 (b):主线 loss 不依赖 18D 强标签(caption 动词弱标签允许)。
配置已建好:`ft_l40_full_5k_lf.yaml`(label-free KL trunk)、
`ft_l40_ours_r2_opp_l01_5k_lf.yaml`(label-free Ours-R2:L_zero + opposite margin 保留,
它们只用自监督信号和弱标签)。
```bash
CUDA_VISIBLE_DEVICES=<两张空闲卡> GPU_COUNT=2 nohup bash run_ours.sh lf > lf_twins.log 2>&1 &
```
意义:① 故事 (b) 的可行性检验——若 lf 版 R²_za 相比监督版不塌(掉幅 < 监督版与 raw_lam
差距的一小部分),主线即可 label-free,监督版降为"标签塑形"消融;若塌得厉害,说明当前
路由主要靠监督锚定,故事要重新讨论(结果本身就是量化"路由有多少靠监督"的实验)。
② lf 家族的 mlp_action_r2 是完全外部指标(训练从未见过 18D),与 raw_lam / cdlam_repro
的比较完全公平。
验收:lf 训练日志无 act 项;benchmark 时 R²_za(lf) 记入报告并与监督版对比。

### T3b(可选,分析素材)IDM oracle
故事 (b) 下 IDM 不再是必备对照,降级为 **oracle 上界 + "可解码≠可控" 2×2 分析素材**
(报告的分析章,不进主表)。做法不变:
- train.py 加 `lambda_recon`(默认 1.0),乘在 lam_loss 的 mse 项上(KL 保留);
- **实现之后**再新建 `code/config/ft_l40_idm_5k.yaml`:复制 `ft_l40_full_5k.yaml`,改
  `lambda_recon: 0.0`、输出目录/run_name 改为 ft_l40_idm_5k(顺序不能反——config 先建
  而 flag 未实现的话会静默训成普通 KL run);
- 预期:R² 全场最高、ctrl/swap/zeroact 最差 → "可解码 ≠ 可控"论据。
验收:烟测日志 mse 无下降压力,act 下降。

### ⚠ 故事 (b) 对后续 L_dir 设计的约束(stage-1 预告,本阶段不实施)
计划中的方向 loss(episode 内反平行配对)原设计用 **GT 18D twist** 挖配对——这与
故事 (b) 冲突(强标签进了主线 loss)。stage-1 实施时必须改为自监督运动代理挖配对
(光流方向 / 手部关键点跟踪),GT-twist 版只作为 oracle 消融展示 headroom。

### 架构裁决消融(stage-1/2 backlog,本阶段不实施;背景见对话 2026-07-15)
split(32a+8e)是"分区+接口",不是分离机制;两个消融裁决其真实价值:
1. **no-split 对照**:单一 40 维 latent + 同样压力栈(L_zero+L_emb+方向 loss),
   评测用 probe 找动作子空间。打平 ⇒ split 价值仅接口便利(保留但诚实声明);
   split 胜 ⇒ 拿到 split 的直接证据。
2. **z_e 结构性 typing**:z_e 改为只编码 o_t(E(o_t),从未见第二帧),z_a 仍看两帧。
   使"动作漏进 z_e"与"z_e wormhole 偷载下一帧"构造上不可能(当前 R²_ze=0.15 的
   根治方案)。中等工程量,L_emb 之后若 R²_ze 仍高再上。
z_e 扩容(8→16)排在这两个之后:o_t 通路免费供给场景,z_e 真实职责只是"o_t 说不出的
上下文变化",L_emb 之后 8 维可能已够。

### T4 训练 cdlam_repro(T2 完成后)
```bash
CUDA_VISIBLE_DEVICES=<两张空闲卡> GPU_COUNT=2 nohup bash run_ours.sh cdlam > cdlam_repro.log 2>&1 &
```
启动后 tail 日志确认:`pctr` 项存在且 pair 数非零;`fg` 生效(若 T1 未完成,记录当时 mask 覆盖率)。

### T5 训练 ours_d(T1 覆盖 >80% 后启动;否则先记录覆盖率并在报告注明稀释)
```bash
CUDA_VISIBLE_DEVICES=<两张空闲卡> GPU_COUNT=2 nohup bash run_ours.sh d > ours_d.log 2>&1 &
```

### T6(T3 做了则)训练 idm
```bash
CUDA_VISIBLE_DEVICES=<两张空闲卡> GPU_COUNT=2 nohup bash run_ours.sh idm > idm.log 2>&1 &
```

### T7 Stage-0 benchmark(以上训练全部出 step_0005000 后)
```bash
CUDA_VISIBLE_DEVICES=<空闲卡> /home/xuan/.venv/bin/python code/benchmark_lam.py \
  --runs raw_lam kl_ft_l40_full_5k kl_ft_l40_full_5k_lf cdlam_repro_5k \
         ours_d_r2_fg_5k ours_r2_opp_l01_5k ours_r2_opp_l01_5k_lf idm_5k \
  --n_samples 1500 --perturb_samples 48 --batch_size 32 --num_workers 8 \
  --device cuda:0 --out_dir results/lam_benchmark_stage0
```
RUN_SPECS 已注册全部名字;缺 checkpoint 的 run 会自动跳过(打 [skip])。
本次会包含新指标:`action_r2_trans/rot/left/right`、`action_r2_za_taskheld(+gap)`。

### T8 远端 seeds 到货后(用户会把 model.safetensors 放进
`checkpoints/lam-dis/ft_l40_{ours_r2_opp_l01_5k_s1, ours_r2_opp_l01_5k_s2, ours_a_zero_5k_s1}/step_0005000/`)
```bash
CUDA_VISIBLE_DEVICES=<空闲卡> /home/xuan/.venv/bin/python code/benchmark_lam.py \
  --runs kl_ft_l40_full_5k_s1 kl_ft_l40_full_5k_s2 ours_a_zero_5k_s1 ours_a_zero_5k_s2 \
         ours_r2_opp_l01_5k_s1 ours_r2_opp_l01_5k_s2 \
  --n_samples 1500 --perturb_samples 48 --batch_size 32 --num_workers 8 \
  --device cuda:0 --out_dir results/lam_benchmark_seeds
```
(本地 kl_s1/kl_s2/a_s2 的 checkpoint 已训完在位,上述命令可先跑这三个。)
然后写 `notes/seed_variance.md`:每模型(含 seed42 主 run)的 mlp_action_r2 /
direction_gain / swap_delta 的 mean±range;标注 a-s1 为 1gpu regime、r2-s1/s2 为跨机器。
判定:R2 对 KL 的 +0.057 优势是否 > 各自 range。

### T9 报告 `notes/stage0_report.md`
- 失败清单表:{方向未编码, 外观在 z_a, 双向路由泄漏, 旋转维弱} × 全部模型,
  用指标速查表里对应的指标填值;与 `benchmark_v2_findings.md` 的预测对照。
- 每个新 run 的 Tier-0(机制)与 Tier-1(验收)判定,一句话结论。
- IDM 若成立"R² 最高 + 可控性最差",单独标注(论文 2×2 论据)。

## 指标速查(哪个 loss/claim 看哪个指标)
| 对象 | Tier-0 机制(允许镜像) | Tier-1 验收(裁决) |
|---|---|---|
| L_emb(cdlam_repro / ours_d) | FG 权重生效、camera_shift | **opp_cls_bal_acc_static ↓**、swap_context_cost_psnr ↓、R²(mlp_action_r2_za)持平 |
| L_pctr(cdlam_repro) | pctr loss 下降、pair 数 | 无 audit 预期;记录 R²/kNN-z 即可,价值需 Tier-2,本阶段不判死刑 |
| IDM | act loss 下降 | 预期 R² 最高、ctrl_delta/swap/zeroact 最差 |
| 失败清单 | — | direction_gain、opp_cls_static、camera_shift、R²_ze、action_r2_rot、action_r2_za_taskheld |

## 已就位、无需重做
- benchmark v2 指标全套(含 permutation-null kNN、gt/probe kNN、路由三元组、
  opp_cls 分层采集+静态对照、do(z_a=0)、swap、分组 R²、task-held-out R²);
- `code/primitive_labels.py`(12 类映射,自检:`python code/primitive_labels.py`,val 覆盖 72.6%);
- 配置:`ft_l40_cdlam_repro_5k.yaml`、`ft_l40_ours_d_r2_fg_5k.yaml`、6 个 seed 配置;
- `run_ours.sh` 入口:`cdlam`、`d`、`idm`、`seeds`、`seeds-local`;
- 本地 seed checkpoint:kl_s1、kl_s2、a_s2 已训完;
- dataset 的 3 通道 mask bug 已修,缺 mask 样本回退均匀权重;`training.seed` 已可配置。
