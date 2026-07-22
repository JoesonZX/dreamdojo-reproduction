# notes/ 索引(2026-07-19 重整)

按阅读顺序,不是字母序。**新会话从 §1 开始读。**

## 1. 概念与计划(先读这三份)
| 文件 | 内容 |
|---|---|
| **`concepts_and_pipeline.md`** | ⭐ 概念层单一事实源:好 z_a 的定义与四条性质、三层评估栈(按**循环性**分层)、实验↔指标对齐、相反动作分离的双门设计、失败清单、loss↔验收对照、五个关键问答 |
| `research_qa_synthesis.md` | 五个核心研究问题的定稿结论、涌现问题、论文前待答问题、整体流程(阶段 0–3) |
| **`stage0_execution_plan.md`** | ⭐ 当前执行计划 T1–T9(硬性纪律 + 验收标准)。**动手前必读** |

## 2. 评估栈
| 文件 | 内容 |
|---|---|
| `benchmark_v2_changes.md` | **指标字典**:v2 每个指标的语义与修正动机(permutation-null kNN、路由三元组、opp_cls 静态对照、do(z_a=0)、swap);评估协议纪律 |
| `benchmark_v2_findings.md` | **v2 结果简报**:四个被推翻的旧结论 + 定稿失败清单。数值权威 |
| `cdlam_repro_fidelity.md` | CD-LAM 论文公式 ↔ 我们实现的逐项对照、五处偏离、指标对应(**其 `id_ratio` = 我们的 `static_response_rel_median`**)、官方 code/checkpoint 用法 |

## 2b. 阶段 A 产物(2026-07-20,零训练成本,三项都改了设计)
| 文件 | 内容 |
|---|---|
| **`dir_pair_audit.md`** | ⭐ L_dir 配对挖掘审计:朴素"反平行+幅度"规则不成立(lift 1.24×),富集只在 **Δt 0.5–2s**(lift 2.59×);定稿挖掘规则 + 必配的随机对对照;相机残差 18.3% 必须补偿 |
| **`dirprobe_split_issue.md`** | ⭐ 旧 `opp_cls` 含 **episode 泄漏**(val 里少数 verb 中位仅 1 条 episode)⇒ 失败②证据下调;条件 probe 改用 train 池的理由与判读守则 |

对应代码:`code/audit_dir_pairs{,2,3}.py`、`code/rescore_dirprobe.py`、
`code/test_lr_schedule.py`(LR 调度 bug 的单测 + 旧 bug 回归见证)。
对应结果:`results/dir_pair_audit{,2,3}.json`、`results/dirprobe_rescore.csv`。

## 2c. 阶段 B 产物 + 论文草稿(2026-07-21)
| 文件 | 内容 |
|---|---|
| **`paper_draft.md`** | ⭐ **合并后的诊断论文草稿**(availability≠utilization);十节 + 四条贡献 + 措辞守门。数值权威、可追溯到 results/ |
| **`stage_b_plan.md`** | ⭐ 阶段 B 决策状态单一事实源:五臂设计/H1H2 否决/顾问三轮判定/**H2 分解**(donor 前提被证伪)/**H2 正对照**(硬门✅ 通过,scorer 灵敏度已证)/posterior SNR |
| `stage_b_mechanics.md` | 阶段 B loss/评测的实现细节 |
| **`consult_tier2_prompt.md`** | 第四轮顾问问答(**已回复**):Tier-2 方案 = localize(cross-decode)→transfer(独立轻量 decoder)→scale(2B 仅在可行时);末尾是完整回复 |
| **`tier2_execution_handoff.md`** | ⭐ **新会话执行 prompt**:整段贴进新会话即按顾问 §8 排期执行(Phase 0 零训练硬门 → Tier-1.5 → 最小 Tier-2)|

对应代码:`code/{dirprobe,h2_utilization,h2_decompose,h2_poscontrol,eval_stage_b,eval_h2_decompose,eval_h2_poscontrol}.py`。
对应结果:`results/stage_b_eval.json`、`results/stage_b_h2_decompose.*`、`results/stage_b_h2_poscontrol.*`。
α 消融(decoder-exposure 噪声):`code/config/ft_l40_sb_A{0,1}.yaml`(α=0 vs α=1,matched)+
`model.decoder_noise_alpha` 旋钮;运行中,产物 `results/alpha_ablation.log`、
`checkpoints/lam-dis/ft_l40_sb_A{0,1}/`。

## 3. 长效参考
| 文件 | 内容 |
|---|---|
| `legacy_lessons.md` | 从 `archive/` 打捞的长效事实:18D 动作物理含义、EgoDex 数据管线、LAM_400k 溯源、decorrelation 失败教训、相关工作映射、图表资产 |
| `analysis-recon-perdim.md` | ⚠️ 模型排名已作废(v2 推翻),但 **Q2 的 18D 动作定义 + 逐维 R² 结构**、**Q1 "PSNR 不是解耦指标"** 仍然有效 |
| `throughput-debug.md` | 数据管线性能定案:1080p→240p all-intra 转码(`__getitem__` 1540ms→8.8ms)、grad_accum 陷阱、"训练与 benchmark 不同机"纪律的来源 |

## 4. 结果目录(不在 notes 里)
- `results/lam_benchmark_v2/` — **当前权威 benchmark**,勿覆盖
- `results/lam_benchmark_seeds/` — T8 seed 方差(注:CSV 里 `id_ratio_*` 列是 `scene_knn_*` 改名前的产物)
- `results/lam_benchmark_stage0/` — T7 输出(待跑)
- `results/figs/` — 图表资产,索引见 `legacy_lessons.md` §6

## 5. 阶段结果文档
- **`stage0_report.md`** — ⭐ T9 报告(2026-07-20 草稿,idm/ours_d 待补):失败清单表
  (8 模型实数)、lf 塌缩与 T8 两大发现、各 run Tier-0/1 判定、故事对账、待决策清单
- `seed_variance.md` — T8 定稿:三家族×三 seed,九行同码;R2 无 headline 优势的裁决
  + 与 v2 的复现性对照(delta 全 0)
- **`stage_b_plan.md`** — ⭐ 阶段 B(2026-07-21):五臂 successive-halving 设计、
  H1/H2 confirmatory 结果、决策树判定(**方向线双指标否决,止损**)、转向 utilization 线索
- **`stage_b_mechanics.md`** — 阶段 B 机制说明:五臂/两指标/三对照的证伪逻辑、配对挖掘怎么做
- `dir_pair_audit.md` — L_dir 配对挖掘审计(Δt 窗口 1.24×→2.59× 的发现)
- `consult_story_b_prompt.md` — 与顾问两轮问答 + 阶段 B 同步(末尾四个待回复问题,#3(b) 是零成本 z_mu 诊断)
- 结果:`results/stage_b_eval.json`(H1 per_arm/paired + H2 h2_per_arm/h2_paired);
  `results/stage_b_eval_5task_STALE.json`(GroupKFold bug 的 5-task 假象存档)

## 6. `archive/`
12 份历史文档:从零训练 exp1–4 与 700M 系、ft_l32/l40 微调对比、contrastive v1–v3、
Ours-B/B2/C 时期的 handoff 与审计。**模型排名与推荐已全部作废**(多数 checkpoint 已删,
结论被 v2 推翻);其中仍有效的部分已打捞进 `legacy_lessons.md`。
查历史再进 `archive/`,**不要引用其结论**。

## 7. 本地文件
`Causal_Debiased_LAM*.pdf` = CD-LAM 论文(权威源,公式以它为准)、`method_figure.drawio`。
体积大,仅本地保留。
