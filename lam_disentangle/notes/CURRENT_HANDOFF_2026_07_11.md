# Current Handoff: Fine-Grained Causal LAM

Date: 2026-07-11  
Project root: `/home/xuan/embodied-ai/lam_disentangle`  
Main Python env: `/home/xuan/.venv/bin/python`  
SAM3 env: `conda run -n sam3 ...`

## 1. Do Not Put Large Artifacts On GitHub

Do not push these to GitHub:

```text
/home/xuan/embodied-ai/checkpoints/
/home/xuan/embodied-ai/data/
notes/*.pdf
results with large media/masks/checkpoints
```

GitHub should only carry code, configs, notes, and small CSV/markdown results. Checkpoints and SAM3 masks are too large and should be moved with `rsync`/`scp` or a dataset/artifact store.

Recommended transfer method:

```bash
rsync -avh --partial --progress \
  /home/xuan/embodied-ai/lam_disentangle/ \
  USER@NEW_HOST:/home/xuan/embodied-ai/lam_disentangle/

rsync -avh --partial --progress \
  /home/xuan/embodied-ai/checkpoints/lam-dis/ \
  USER@NEW_HOST:/home/xuan/embodied-ai/checkpoints/lam-dis/

rsync -avh --partial --progress \
  /home/xuan/embodied-ai/checkpoints/pretrained/ \
  USER@NEW_HOST:/home/xuan/embodied-ai/checkpoints/pretrained/
```

Only transfer SAM3 masks if foreground experiments will be run:

```bash
rsync -avh --partial --progress \
  /home/xuan/embodied-ai/data/egodex/test_240p/*/sam3_hoc_masks \
  USER@NEW_HOST:/home/xuan/embodied-ai/data/egodex/test_240p/
```

If the target machine already has EgoDex videos/HDF5 files, do not transfer the full dataset again.

## 2. Current Research Framing

Core claim:

```text
The useful innovation is a world-model interface latent split:
32D action subspace + 8D environment subspace.
The goal is to keep z_a action-pure and prevent scene/background/context from leaking into the action condition.
```

The paper story should be:

```text
phenomenon -> loss design -> metric effect
```

Current evidence supports:

```text
L_zero strongly fixes zero-transition calibration.
Temporal reverse is useful for action direction, but env reverse causes context leakage.
Naive semantic hard-negative contrast did not help in the current implementation.
Foreground weighting should be Ours-D, not part of Ours-C.
```

## 3. Important Files

Plans and analysis:

```text
notes/fine_grained_causal_lam_plan.md
notes/stage1_lam_side_audit.md
notes/CURRENT_HANDOFF_2026_07_11.md
```

Code:

```text
code/train.py
code/dataset.py
code/benchmark_lam.py
code/audit_reversible_episodes.py
code/precompute_sam3_hoc_masks.py
run_ours.sh
```

Key configs:

```text
code/config/ft_l40_ours_a_zero_5k.yaml
code/config/ft_l40_ours_b_zero_reverse_5k.yaml
code/config/ft_l40_ours_b2_zero_action_reverse_5k.yaml
code/config/ft_l40_ours_c_zero_reverse_hardneg_5k.yaml
code/config/ft_l40_ours_c_zero_reverse_hardneg_fg_5k.yaml
```

## 4. Completed Experiments And Results

LAM-side benchmark output:

```text
results/lam_benchmark/summary.md
results/lam_benchmark/summary.csv
results/lam_benchmark/per_dim_*.csv
```

Main benchmark table as of 2026-07-08:

| run | action R2 ↑ | static ↓ | camera h/v ↓ | task shortcut ↓ | rev margin ↑ | reverse z_a cos ↓ | reverse z_e cos ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| KL baseline | 0.331 | 0.358 | 0.310 / 0.286 | -0.0446 | +0.0199 | -0.220 | 0.216 |
| Ours-A | 0.333 | **0.035** | **0.293 / 0.286** | -0.0442 | **+0.0259** | -0.239 | -0.418 |
| Ours-B | **0.339** | **0.031** | 0.354 / 0.371 | -0.0012 | -0.0017 | **-0.508** | **0.954** |
| Ours-C | 0.320 | 0.040 | 0.411 / 0.414 | +0.0117 | -0.0044 | -0.451 | 0.951 |

Interpretation:

```text
Ours-A is the current best all-around model.
Ours-B confirms reverse direction learning but hurts camera/task/reversible geometry because env reverse likely encourages context caching.
Ours-C is a negative result: current hard-negative mining hurts action R2 and does not improve reverse hard negatives.
```

## 5. Current Status: B2 And SAM3

Ours-B2:

```text
Config: code/config/ft_l40_ours_b2_zero_action_reverse_5k.yaml
Checkpoint status: only step_0001000 exists as of 2026-07-11.
Target final checkpoint: /home/xuan/embodied-ai/checkpoints/lam-dis/ft_l40_ours_b2_zero_action_reverse_5k/step_0005000
```

Continue B2:

```bash
CUDA_VISIBLE_DEVICES=1 GPU_COUNT=1 bash run_ours.sh b2
```

Benchmark B2 after step 5000 exists:

```bash
CUDA_VISIBLE_DEVICES=1 /home/xuan/.venv/bin/python code/benchmark_lam.py \
  --runs kl_ft_l40_full_5k ours_a_zero_5k ours_b_zero_reverse_5k ours_b2_zero_action_reverse_5k \
  --n_samples 2000 \
  --perturb_samples 64 \
  --batch_size 16 \
  --num_workers 4 \
  --device cuda:0 \
  --out_dir results/lam_benchmark_b2
```

SAM3 H/O/C masks:

```text
Current status as of 2026-07-11:
8 / 111 tasks have sam3_hoc_masks
94349 .pt mask files exist
No active SAM3 process was detected
```

Continue SAM3:

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n sam3 python -u code/precompute_sam3_hoc_masks.py \
  --data_root /home/xuan/embodied-ai/data/egodex/test_240p \
  --downsample_factors 1 2
```

Monitor SAM3:

```bash
watch -n 60 "find /home/xuan/embodied-ai/data/egodex/test_240p -path '*/sam3_hoc_masks/*/*.pt' -type f | wc -l"
```

Known SAM3 issue:

```text
hand masks are mostly stable.
object/contact masks are unreliable for some tasks, e.g. add_remove_lid and arrange_topple_dominoes often have empty object/contact channels.
Do not train Ours-D on H/O/C masks until manual preview confirms quality or the object prompt strategy is improved.
```

Preview images:

```text
results/sam3_hoc_preview/*.jpg
```

## 6. Recommended Next Steps

1. Finish Ours-B2 to 5k.
2. Benchmark Ours-B2 against KL, Ours-A, and Ours-B.
3. If B2 keeps Ours-B's low `reverse_za_cos` while recovering A's camera/task/rev-margin metrics, use B2 as the main reverse variant.
4. Do not continue current Ours-C as the main line. If revisiting C, implement a sampler-aware opposite-pair miner first.
5. Resume SAM3 only if Ours-D foreground ablation is still needed.
6. Before Ours-D, decide whether to use:

```text
D1: CD-LAM-style foreground/background weighting
D2: H/O/C interaction-aware weighting
```

Given current H/O/C quality, D1 is safer than D2.

## 7. Useful Commands

Check experiment status:

```bash
find /home/xuan/embodied-ai/checkpoints/lam-dis -maxdepth 2 -type d -name 'step_*' | sort
find /home/xuan/embodied-ai/data/egodex/test_240p -path '*/sam3_hoc_masks/*/*.pt' -type f | wc -l
pgrep -af "train.py|benchmark_lam|precompute_sam3_hoc_masks"
```

Run Ours variants:

```bash
CUDA_VISIBLE_DEVICES=1 GPU_COUNT=1 bash run_ours.sh a
CUDA_VISIBLE_DEVICES=1 GPU_COUNT=1 bash run_ours.sh b
CUDA_VISIBLE_DEVICES=1 GPU_COUNT=1 bash run_ours.sh b2
CUDA_VISIBLE_DEVICES=1 GPU_COUNT=1 bash run_ours.sh c
```

Git guidance:

```text
Push code/configs/notes/small CSV summaries.
Do not push PDFs, checkpoints, data, SAM3 masks, or large media artifacts.
```
