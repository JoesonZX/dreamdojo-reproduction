# SAM3 前景提取调查 + hand+ring 设计结论

日期:2026-07-21
背景:为 fine-grained causal LAM 的前景加权 loss 用 SAM3 提取 hand/object/contact
三通道 mask,发现 object 通道系统性失败,由此重新审视"前景该是什么"。

## 1. SAM3 全量提取的实测结果(EgoDex test_240p, 3243 视频)

- **hand 通道 99.6% 完整**(仅 13 视频空),质量高,精确到手部轮廓、正确排除前臂。
- **object 通道只有 59.8% 有内容,40.2% 空**,失败与任务强相关,集中在**小物体/透明物**
  (usb、钥匙、骰子、耳机盒、橡皮筋、耙子、史莱姆、电池、螺丝、簸箕、透明盒…)。
- 存储:`{task}/sam3_hoc_frames/{ep}.npy`,uint8 `[2,n,H,W//8]` packbits(ch0=hand,ch1=object),
  全量 15G(旧 per-pair 格式要 229G)。contact 在 dataloader 现算 = dilate(hand)∩dilate(object)。

## 2. object 失败的三层根因(逐一实验证实)

1. **anchor 帧不是问题**(曾假设第 0 帧手没碰物→grounding 失败;A/B 证伪:mid-clip 双向传播只救回极少)。
2. **prompt 必须是具体类名**:关系性短语("objects being manipulated"/"held object")和泛化名词
   ("object")几乎全废;具体名词("cup"/"domino")立刻命中。SAM3 是概念驱动。
3. **小物体天生难**:即使换对具体名词(2 轮诊断,每 task 3 视频),usb/钥匙/骰子/耳机盒等
   仍 0/3——SAM3 在 240p 上对小/透明物 grounding 有固有天花板,换名词无效。
   - 可修的(清晰可见的容器/板/箱/勺/书/篮/布):connect_four→game board、tupperware→plastic
     container、vertical_pick_place→plastic bin、scoop_dump_ice→scoop、color→book、
     boil_serve_egg→frying pan、utensils→wire basket、shirt_in_tube→cloth 等,已 --object_only 补跑。

## 3. hand-seeded 分割实验(试图绕过"按名词 grounding")

想法:hand 通道可靠,用它当空间种子分割"手底下那个物体",不给物体起名。
**SAM3 三种 prompt 的真实行为(实测,关键参考):**

| prompt | 行为 | 证据 |
|---|---|---|
| **text** | 概念分割,需具体类名;小物体失败 | 全量数据 |
| **box** (`bounding_boxes`) | **不是** SAM2 式"分割框内物体",是概念的**视觉示例(exemplar)** | 框住大 cloth 都返回空;代码走 `_get_visual_prompt`→geometric_prompt,且仍和 text 绑定 |
| **point** (`points`+`point_labels`) | **是** SAM2 式空间分割,但**必须传 `obj_id`** | cloth 点质心→cov 0.085(与 text 一致) |

**hand-seeded 结论:不能救小物体,但原因不是 SAM3 缺空间能力。**
point 路径确实是空间分割,障碍是**没有能可靠落在小物体上的种子点**:
- 点打在手上 → 返回几乎空(0.002)
- 点打在手外一点(抓握点上方)→ 抓到整片背景桌布(0.26)
- U 盘夹在手指间、又小又被遮挡,任何从手几何推出的单点都落不到它上面(overlay 已确认)。

## 4. 设计结论:前景 = hand + 手周围一圈(dilation ring)

**动机**:fg 加权 loss 的目的是让 LAM 聚焦"产生动作的部分"。CD-LAM 的 thesis 是 embodiment-centric,
debias 掉 action-irrelevant 的 background 和**未接触物体**。

**关键辨析**(避免过度简化):
- **未接触物体 / 背景** → action-irrelevant → 应排除(支持"物体前景无用"的直觉)。
- **被交互物体** → action-**relevant**(其运动是动作的**效果**:推 vs 提 → 物体轨迹不同)。
  CD-LAM 的 `interact` mask 明确**包含**"interacted-object regions",FDCE 指标就是测被交互物的位移。
- **但**被交互物按定义就在手边,所以 **dilate(hand) 的 ring 从几何上就覆盖了被交互物贴近手的部分**
  ——即动作相关区域——**无需分割物体**。ring 是"embodiment + 交互区"的鲁棒几何代理。

**所以 hand+ring ≈ CD-LAM 的 interact 区,但用几何得到、对分割误差天生鲁棒、无任务偏差、100% 覆盖。**

**局限(诚实记录)**:ring 半径是超参。太小漏掉物体延展(如手挥动的长工具的远端);太大漏进背景。
EgoDex 多为手边小物,中等半径够用;长工具/大物体的远端动作 ring 会低估——若将来需要完整被交互物
运动(如大物体的 action-following 评估),ring 不如真 object mask,那时需回到分割或改用点追踪(CoTracker)。

## 5. 工程影响:不用重跑 SAM3,也不强制重训

- **无需重新提取**:hand 通道已全量提取(99.6%),ring = `dilate(hand)`,在 dataloader 现算。
  现有 `sam3_hoc_frames/*.npy` 的 ch0 直接可用。
- **无需从头重训**:`fg_mask` 只是 `lam_loss(...)` 的参数(见 train.py:994),**不进 `model.forward`**。
  改前景定义 = 改训练 loss 的加权信号,**不改模型架构/输入维度**。可 warm-start;为干净 ablation 可重训,但非架构强制。
- **可廉价 ablation**:现数据同时含 hand + object 通道,ring 是 dilation。只改 dataloader 的
  `_load_fg_mask` 组合方式,即可对比 **hand-only / hand+ring / HOC-3ch** 三种前景定义,无需任何重新提取。
  → 别 a priori 争论"物体有没有用",直接 ablate,用 CD-LAM 的 LAM-bias/action-following 指标裁决。

## 建议下一步
1. dataloader 加一个 `fg_mask_type: "hand_ring"`,返回 `dilate(ch0, radius=r)` 单通道(ring 半径设为超参)。
2. 跑三臂 ablation(hand-only / hand+ring / HOC-3ch),看 debias 指标。
3. object 补跑(--object_only,正在 GPU 0 跑)保留 HOC 臂用,跑完不删——为 ablation 留选项。
