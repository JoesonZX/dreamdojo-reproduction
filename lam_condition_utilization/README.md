# lam_condition_utilization (Project B)

Diagnostic study of the distinction between action-information **availability**
(is a factor linearly readable from the latent `z`?) and decoder **utilization**
(is it causally used?), in a DreamDojo/CD-LAM latent-action model.

Central result — the paper draft *"Availability is not Utilization: Diagnosing —
and Removing — a Training-Noise Bottleneck in the Latent-Action-to-Decoder Channel"*:
direction information is largely *available* but not *utilized*; a decoder-exposure
noise setting (**α=0**) restores utilization while barely moving the encoder `μ`.
Covers: α=0/1 exposure, paired-latent, 2×2 cross-decoding, Phase 0 / 0.3, Tier-1.5
K-step/sign intervention, Route-C GT18 vs zmu-A0/A1 independent consumer, and the
Tier-2 preregistration.

- **Read first:** [`notes/README.md`](notes/README.md) → `notes/paper_draft.md`, `notes/stage_b_plan.md`.
- **Full record:** [`PROJECT_MANIFEST.md`](PROJECT_MANIFEST.md).
- **Lineage:** A ([`../lam_cdlam_optimization`](../lam_cdlam_optimization)) → **B** → C ([`../lam_world_model_control`](../lam_world_model_control)).

## Dependency on Project A
This project **imports A's shared LAM substrate** (`train/model/dataset/eval/benchmark_lam/
primitive_labels/dirprobe`) as a read-only dependency. Every evaluator that needs it
does `import _apath` first — see [`code/_apath.py`](code/_apath.py). Its α/utilization
LAM arms are trained *through A's* `code/train.py`. B is therefore **not standalone**;
`../lam_cdlam_optimization` must be present. See `PROJECT_MANIFEST.md` §6.

## Layout
```
code/            B-specific diagnostics (consumer, cross-decode, h2, tier15, phase0, α, ood) + _apath.py shim
code/config/     ft_l40_sb_*.yaml Stage-B arms (fed to A's train.py)
notes/           paper_draft.md, stage_b_*, tier2_* (index in notes/README.md)
results/         stage_b_*.json/.npz, consumer_*, alpha_* (authoritative diagnostic data)
logs/            sb_*.log, stage_b_*.log
data/            eval_manifest.json (Eval-A), eval_manifest_B.json (Eval-B), consumer_exclude.json — all FROZEN
checkpoints/     (logical) — physical store is the shared /checkpoints/lam-dis/ft_l40_sb_* (see manifest)
```

## Quickstart (run from this directory)
```bash
# train a Stage-B / alpha arm (through A's shared trainer)
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --mixed_precision bf16 \
    ../lam_cdlam_optimization/code/train.py --config code/config/ft_l40_sb_A0.yaml
# Stage-B confirmatory eval  /  Route-C consumer
python code/eval_stage_b.py
python code/train_consumer.py --cond zmu --arm sb_A0 --seed_tag s0
```
