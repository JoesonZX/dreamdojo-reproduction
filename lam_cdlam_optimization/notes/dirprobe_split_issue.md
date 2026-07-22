# ⚠️ 可逆-task 评测的 episode 泄漏问题(2026-07-20 发现,影响失败模式②的解读)

实现条件方向 probe 时发现的**协议级问题**,它同时影响旧的 `opp_cls` 家族。

## 1. 现象

条件 probe 在 val split 上跑 raw_lam,给出自相矛盾的结果:
```
gain=+0.267  CI=[+0.085,+0.453](排除 0)  perm p=0.25(不显著)
AUC base=0.392  full=0.475     ← 两个 probe 都低于 0.5,比随机还差
```
两个"比随机还差"的 probe 之间的差值不能解释为方向信息。

## 2. 根因:val split 的可逆 task 里,少数 verb 的 episode 数中位是 **1**

按 episode 分组统计(`ds._verb_map` + `ds._index`):

| split | 可逆 task | 少数 verb 的 ep 数中位 | 少数 verb ≥3 ep 的 task | 总 ep ≥8 的 task |
|---|---|---|---|---|
| **val** | 28 | **1** | **5** | 7 |
| **train** | 68 | **8** | **57** | 59 |

val 只占 10%,可逆 task 的每个 verb 往往只有一条 episode。
⇒ 按 episode 分组做 CV 时,留出该 episode 就等于**留出整个类别**,
fold 退化;GroupKFold 在这种数据上无法给出有意义的估计。

## 3. 更重要的推论:旧 `opp_cls` 的 0.96–0.99 主要是 **episode 记忆**

旧实现用的是 **task 内随机切分**(`train_test_split(..., stratify=y)`),
**同一条 episode 的帧同时出现在训练和测试**。当一个 verb 只有 1–3 条 episode 时,
"分类 verb"退化成"认出这帧来自哪条 episode"——
**`opp_cls_bal_acc_static = 0.96–0.99` 衡量的很大程度是 episode 身份泄漏,
而不是"初始外观→verb"的规则**。

这与"天花板效应"叠加,共同解释了为什么 `static_shortcut_gap` 恒为负:
静态 probe 靠记忆 episode 拿到近满分,transition probe 因为运动信息冲淡了
episode 身份线索反而更低。**两者都不是"外观 vs 方向"的干净对比。**

⇒ 失败模式②("外观在 z_a")的**证据强度必须下调**:
现有数字不足以支撑"70% 的相反动作可分性全部来自初始帧外观"这一强表述。
准确表述:"在允许 episode 泄漏的协议下,静态帧 latent 已能近乎完美地分开两个 verb;
该协议无法区分'外观规则'与'episode 记忆'。"

## 4. 处置

1. **条件 probe 改用 train split 的 episode 池**(`--split train --min_eps_per_verb 4`),
   并在 `_collect_opposite` 里**按 episode 配额采样**(避免一条长 episode 主导一个 verb)。
   代价:encoder 训练时见过这些帧。**这是表征诊断,不是 encoder 泛化主张**;
   全部模型同协议,家族内可比;probe 本身仍在 episode 级严格留出。
2. 新增判读守则:**`dirprobe_nll_gain` 只有在 `dirprobe_auc_full > 0.5` 时才可解释**
   ——两个无预测力的 probe 之间的差值无意义。写进 summary.md 注释。
3. 旧 `opp_cls_*` 与 `static_shortcut_gap` 保留为诊断,但在报告里**必须标注
   "随机切分,含 episode 泄漏"**,不得作为外观混淆的定量证据。
4. `results/lam_benchmark_*` 已有的 `opp_cls` 列全部继承此问题(协议未变),
   引用时一律加注。

## 5. 对论文主张的影响

动机链的三条腿现在是:
- ①**方向**:待条件 probe(train 池)重评后定论 —— **这是当前最关键的未知数**;
- ②**外观在 z_a**:证据下调(见上),需改用 episode-grouped 协议重测才能定量;
- ③**路由泄漏**(R²_ze)、④**旋转弱**(R²_rot):**不受影响**
  ——它们的目标是 18D 连续动作、按样本回归,不涉及 verb 标签与 episode 分组。

③④ 仍是干净的失败证据;①② 的表述必须按上述收窄。
