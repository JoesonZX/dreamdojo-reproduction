# 多视角 Latent Action Model（EgoExo4D）

用 Ego-Exo4D 的**同步多视角**人类操作视频训练 latent action model：latent action 从一个视角编码，但必须能重建**另一个视角**的下一帧。视角特有的信息会伤害跨视角重建、被主动挤出去，剩下的只有跨视角不变的东西——动作本身。

```
z^v_t   = Enc(o^v_t, o^v_{t+1})
L_self  = ‖ Dec(o^v_t, z^v_t) − o^v_{t+1} ‖²
L_cross = ‖ Dec(o^w_t, z^v_t) − o^w_{t+1} ‖²      (w ≠ v)
L       = L_self + λ_cross · L_cross + β · KL
```

**为什么改动这么小**：解码器是 `Dec(patchify(o_t), z)`，视角**只通过 context patch 进入**。所以「跨视角重建」的实现就是把 context 换成另一个视角的帧，encoder 一行不用改。

---

## 训练臂

| 配置 | mode | λ_cross | num_views | 作用 |
|---|---|---|---|---|
| `sv.yaml` | single | 0 | 0 | **L0** 单视角基线 |
| `mv_data.yaml` | multi | 0 | 0 | **L1 控制臂**：同样的多视角数据，但只自重建。没有它，L2 优于 L0 无法排除"只是数据变多了" |
| `mv_cross.yaml` | cross | 1.0 | 0 | **L2 主线** |
| `view_prompt.yaml` | multi | 0 | 8 | **L4 X-VLA 式对照**：用 per-view embedding **吸收**视角差异，而不是**消除**它 |
| `mv_cross_hand.yaml` | cross | 1.0 | 0 | **L3** 加手部姿态锚定（λ_hand=0.1）。⚠️ 它的 loss 消费了手部动作，所以 action probe 对**这个臂**只是诊断，不能当 headline |

除了上表标出的字段，**所有配置必须完全一致**（尤其 `data.split_seed`，train/eval 共用同一个划分函数）。

---

## 评价指标

`eval_crossview.py` 全部在**留出相机**（`heldout_cam`，任何臂都没训练过）上计算：

1. **`crossview_gain` = 匹配 cosine 中位数 − 错配 cosine 中位数**
   ⚠️ **只看 gain，不要看裸 cosine**。未训练模型的裸 cosine 也是 0.9999（latent 有主导方向），错配 null 是唯一有意义的参照。
2. **`action_r2_drop`** = 在训练视角上拟合 `z → 手腕 delta` 的 ridge，读留出视角时 R² 的掉幅。掉幅越小，动作信息越不依赖视角。

---

## 完整流程

### 0. 传代码到训练机

```bash
# 在本机
bash fast_wam/wam/scripts/transfer.sh xuan@<A6000_HOST> /home/xuan/embodied-ai
```
传两个目录：`fast_wam/wam`（本项目）和 `code/adaworld`（`model.py` 只读导入的 transformer 模块）。**不传数据、不传 checkpoint。**

如果远端根目录不同，脚本会打印出需要执行的 `sed` 替换命令。

跑完之后把权重拉回来（只拉 `model.safetensors`，optimizer 状态几个 GB 不用拉）：
```bash
bash fast_wam/wam/scripts/transfer.sh xuan@<A6000_HOST> /home/xuan/embodied-ai --pull
```

### 1. 配环境（在训练机上）

```bash
cd /home/xuan/embodied-ai
bash fast_wam/wam/scripts/setup_env.sh          # 默认建在 ~/.venv-wam
source ~/.venv-wam/bin/activate
python fast_wam/wam/scripts/smoke_test.py       # 验收：必须打印 ALL SMOKE TESTS PASSED
```

`smoke_test.py` 用合成数据把 dataset 三种模式、模型前向、cross-view、loss、反向、view-prompt 全跑一遍，**不需要下载任何数据**。它通过了，后面的失败就只可能是数据路径和显存。

> ⚠️ 需要 GPU：vendored 的 AdaWorld `PositionalEncoding` 里硬编码了 `.cuda()`（上游代码，未修改）。

### 2. 下数据

**先做一次性手工步骤**：在 https://ego4ddataset.com/egoexo-license/ 签许可，拿到 AWS keys，然后 `aws configure`。

```bash
bash fast_wam/wam/scripts/download_egoexo.sh /home/xuan/embodied-ai/data/egoexo4d
```

体积（官方文档）：`metadata` 0.05G + `annotations` 10.5G（3D 手部姿态在这里）+ `downscaled_takes/448` **438.6G 全量**。按场景过滤到操作类（cooking / bike repair / health）后约 **150–200G**。原分辨率 `takes`（10.5T）、`take_vrs`（12.3T）、`take_point_cloud`（6.2T）**都不下**。

先看有哪些场景：
```bash
python - <<'PY'
import json, collections
d = json.load(open('data/egoexo4d/takes.json'))
c = collections.Counter(t.get('parent_task_name') or t.get('task_name') for t in d)
[print(f'{n:5d}  {k}') for k, n in c.most_common()]
PY
```

### 3. 预处理

```bash
# 3a. 先 inspect，确认目录 glob 对得上真实布局
python fast_wam/wam/prepare/build_index.py \
    --egoexo_root data/egoexo4d --out data/egoexo4d/raw_index.json --inspect

# 3b. 建索引（只保留 >=3 路 exo 的 take，这样留一路做 heldout 后还剩 >=2 路可训）
python fast_wam/wam/prepare/build_index.py \
    --egoexo_root data/egoexo4d \
    --out         data/egoexo4d/raw_index.json \
    --scenarios cooking "bike repair" health \
    --min_exo 3

# 3c. 转码成 320x240 全 I 帧 h264（-g 1，随机读两帧才不会被 GOP 拖死）
#     Step A 先不带 ego（--skip_ego），exo-exo 配对更简单
python fast_wam/wam/prepare/transcode.py \
    --raw_index data/egoexo4d/raw_index.json \
    --out       data/egoexo4d_mv240 \
    --jobs 16 --skip_ego

# 3d. 手部姿态 -> 手腕 SE(3)。⚠️ 先 --inspect 确认 JSON schema
python fast_wam/wam/prepare/hand_actions.py --inspect \
    --annotations data/egoexo4d/annotations --processed data/egoexo4d_mv240

python fast_wam/wam/prepare/hand_actions.py \
    --annotations data/egoexo4d/annotations \
    --processed   data/egoexo4d_mv240 --prefer automatic
```

**验收**：随机抽几个 take，把同一时刻各路 exo 帧拼图肉眼确认同步；`index.json` 里 `heldout_cam` 不为空的 take 数量足够。

### 4. 训练

一条命令跑全部臂（3 卡机器：GPU 0+1 双卡跑，逐个串行）：
```bash
bash fast_wam/wam/scripts/train_all.sh
```

跑 seed 副本（headline 结论前必须有）：
```bash
SEEDS="42 1 2" bash fast_wam/wam/scripts/train_all.sh "sv mv_data mv_cross"
```

单个臂的原始命令：
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m accelerate.commands.launch \
    --num_processes 2 --mixed_precision bf16 --main_process_port 29540 \
    fast_wam/wam/train.py --config fast_wam/wam/config/mv_cross.yaml \
    --override training.seed=42
```

单卡（accum 翻倍以保持有效 batch）：
```bash
CUDA_VISIBLE_DEVICES=2 python -m accelerate.commands.launch \
    --num_processes 1 --mixed_precision bf16 --main_process_port 29541 \
    fast_wam/wam/train.py --config fast_wam/wam/config/sv.yaml \
    --override training.gradient_accumulation_steps=2
```

**共享机器：起训练前先 `nvidia-smi` 确认没别人在跑。**

训练可断点续跑——再执行同一条命令会自动从 `output_dir` 里最新的 `step_*` 恢复。

### 5. 评测

```bash
python fast_wam/wam/eval_crossview.py \
    --checkpoint fast_wam/checkpoints/mv_cross_s42/step_0010000 \
    --config     fast_wam/wam/config/mv_cross.yaml \
    --n_samples  4000 \
    --out        fast_wam/results/mv_cross_s42.json
```
`train_all.sh` 每个臂训完会自动跑这一步，最后打印汇总表。

---

## 判读

**机制成立的条件**：`crossview_gain` 上 `mv_cross > mv_data > sv`，且差距**大于 seed 之间的波动**。

- `mv_cross ≈ mv_data` ⇒ cross-view 项没起作用，多视角只是数据增强 → 调 `λ_cross`，或换 `view_pairs: any` 加 ego–exo 配对（约束更强）
- `view_prompt` 赢 ⇒ "吸收视角"胜过"消除视角"，这是对 MVP-LAM 路线的反例，照实报
- `action_r2_drop` 应该 `mv_cross < sv`（动作信息更不依赖视角）

只有这一步过了，才值得往下游 Efficient-WAM + LIBERO 投入。

---

## 文件

```
fast_wam/wam/
  model.py               MultiViewLAM（encode / decode / forward）+ lam_loss
  dataset.py             EgoExoMultiViewDataset（single|multi|cross 三种采样）
  train.py               accelerate 训练循环，YAML 驱动
  eval_crossview.py      跨视角一致性 + 动作 probe（在 heldout 相机上）
  prepare/
    build_index.py       扫描原始 Ego-Exo4D -> raw_index.json
    transcode.py         -> 320x240 全 I 帧 + index.json（分配 heldout_cam）
    hand_actions.py      3D 手部关键点 -> 手腕 SE(3) -> hand_poses.npz
  config/                _base.yaml + 五个臂
  scripts/
    setup_env.sh         建环境
    smoke_test.py        合成数据端到端自检（先跑这个）
    download_egoexo.sh   下数据
    transfer.sh          推代码 / 拉权重
    train_all.sh         全部臂 + 自动评测 + 汇总表
```

**唯一的外部依赖**：`code/adaworld/lam`（`patchify` / `unpatchify` / `SpatioTemporalTransformer` / `SpatioTransformer`），只读导入，未修改。
