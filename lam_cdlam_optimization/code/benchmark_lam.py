"""
LAM-side representation benchmark.

This script evaluates existing LAM checkpoints without training. It is meant to
turn the CD-LAM-style audits into local, repeatable numbers:

  - static-pair response: E(o_t, o_t) should be near the zero-action reference
  - latent health: effective rank, active units, per-dim KL/norm/std
  - shortcut leakage: same-episode/different-action vs different-episode/same-action
  - per-dim effect: action information, decoder perturb effect, label leakage
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from dataset import EgoDexDataset  # noqa: E402
from eval import _collate, load_model  # noqa: E402
from model import LatentActionModel, unpatchify  # noqa: E402
from train import load_pretrained_lam  # noqa: E402


@dataclass(frozen=True)
class RunSpec:
    name: str
    config: Path
    checkpoint: Path | None = None
    init_from: Path | None = None


RUN_SPECS = [
    RunSpec(
        name="raw_lam",
        config=Path("code/config/ft_l32_full_5k.yaml"),
        init_from=Path("/home/xuan/embodied-ai/checkpoints/pretrained/LAM_400k.ckpt"),
    ),
    # EXTERNAL ANCHOR: the released CD-LAM Stage-1 LAM (HF yufanwei/CD-LAM). Never
    # trained here. latent 32 / no split, so `_ze` metrics are nan and `_za` == full,
    # exactly like raw_lam. Its direction_gain is official-checkpoint-grade evidence
    # for the motivation, and mlp_action_r2 is fully external for it (no 18-D
    # supervision was consumed). See notes/cdlam_repro_fidelity.md.
    RunSpec(
        name="cdlam_official",
        config=Path("code/config/cdlam_official.yaml"),
        init_from=Path("/home/xuan/embodied-ai/checkpoints/pretrained/CDLAM_official_lam.pt"),
    ),
    RunSpec(
        name="kl_ft_l40_full_5k",
        config=Path("code/config/ft_l40_full_5k.yaml"),
        checkpoint=Path("/home/xuan/embodied-ai/checkpoints/lam-dis/ft_l40_full_5k/step_0005000"),
    ),
    RunSpec(
        name="contrastive_v3_5k",
        config=Path("code/config/ft_l40_contrastive_v3_5k.yaml"),
        checkpoint=Path("/home/xuan/embodied-ai/checkpoints/lam-dis/ft_l40_contrastive_v3_5k/step_0005000"),
    ),
    RunSpec(
        name="ours_a_zero_5k",
        config=Path("code/config/ft_l40_ours_a_zero_5k.yaml"),
        checkpoint=Path("/home/xuan/embodied-ai/checkpoints/lam-dis/ft_l40_ours_a_zero_5k/step_0005000"),
    ),
    RunSpec(
        name="ours_b_zero_reverse_5k",
        config=Path("code/config/ft_l40_ours_b_zero_reverse_5k.yaml"),
        checkpoint=Path("/home/xuan/embodied-ai/checkpoints/lam-dis/ft_l40_ours_b_zero_reverse_5k/step_0005000"),
    ),
    RunSpec(
        name="ours_b2_zero_action_reverse_5k",
        config=Path("code/config/ft_l40_ours_b2_zero_action_reverse_5k.yaml"),
        checkpoint=Path("/home/xuan/embodied-ai/checkpoints/lam-dis/ft_l40_ours_b2_zero_action_reverse_5k/step_0005000"),
    ),
    RunSpec(
        name="ours_c_zero_reverse_hardneg_5k",
        config=Path("code/config/ft_l40_ours_c_zero_reverse_hardneg_5k.yaml"),
        checkpoint=Path("/home/xuan/embodied-ai/checkpoints/lam-dis/ft_l40_ours_c_zero_reverse_hardneg_5k/step_0005000"),
    ),
    RunSpec(
        name="ours_r_zero_opposite_5k",
        config=Path("code/config/ft_l40_ours_r_zero_opposite_5k.yaml"),
        checkpoint=Path("/home/xuan/embodied-ai/checkpoints/lam-dis/ft_l40_ours_r_zero_opposite_5k/step_0005000"),
    ),
    RunSpec(
        name="ours_r2_opp_l01_5k",
        config=Path("code/config/ft_l40_ours_r2_opp_l01_5k.yaml"),
        checkpoint=Path("/home/xuan/embodied-ai/checkpoints/lam-dis/ft_l40_ours_r2_opp_l01_5k/step_0005000"),
    ),
    RunSpec(
        name="ours_r2_opp_l03_5k",
        config=Path("code/config/ft_l40_ours_r2_opp_l03_5k.yaml"),
        checkpoint=Path("/home/xuan/embodied-ai/checkpoints/lam-dis/ft_l40_ours_r2_opp_l03_5k/step_0005000"),
    ),
    RunSpec(
        name="ours_r2_opp_l05_5k",
        config=Path("code/config/ft_l40_ours_r2_opp_l05_5k.yaml"),
        checkpoint=Path("/home/xuan/embodied-ai/checkpoints/lam-dis/ft_l40_ours_r2_opp_l05_5k/step_0005000"),
    ),
    RunSpec(
        name="ours_r2_opp_l08_5k",
        config=Path("code/config/ft_l40_ours_r2_opp_l08_5k.yaml"),
        checkpoint=Path("/home/xuan/embodied-ai/checkpoints/lam-dis/ft_l40_ours_r2_opp_l08_5k/step_0005000"),
    ),
    RunSpec(
        name="ours_d_r2_fg_5k",
        config=Path("code/config/ft_l40_ours_d_r2_fg_5k.yaml"),
        checkpoint=Path("/home/xuan/embodied-ai/checkpoints/lam-dis/ft_l40_ours_d_r2_fg_5k/step_0005000"),
    ),
    RunSpec(
        name="cdlam_repro_5k",
        config=Path("code/config/ft_l40_cdlam_repro_5k.yaml"),
        checkpoint=Path("/home/xuan/embodied-ai/checkpoints/lam-dis/ft_l40_cdlam_repro_5k/step_0005000"),
    ),
    RunSpec(
        name="idm_5k",
        config=Path("code/config/ft_l40_idm_5k.yaml"),
        checkpoint=Path("/home/xuan/embodied-ai/checkpoints/lam-dis/ft_l40_idm_5k/step_0005000"),
    ),
    # Label-free twins (story b): no 18-D action loss — mlp_action_r2 is fully
    # external for these, and comparison with raw_lam / CD-LAM-style is fair.
    RunSpec(
        name="kl_ft_l40_full_5k_lf",
        config=Path("code/config/ft_l40_full_5k_lf.yaml"),
        checkpoint=Path("/home/xuan/embodied-ai/checkpoints/lam-dis/ft_l40_full_5k_lf/step_0005000"),
    ),
    RunSpec(
        name="ours_r2_opp_l01_5k_lf",
        config=Path("code/config/ft_l40_ours_r2_opp_l01_5k_lf.yaml"),
        checkpoint=Path("/home/xuan/embodied-ai/checkpoints/lam-dis/ft_l40_ours_r2_opp_l01_5k_lf/step_0005000"),
    ),
]
# Stage-B arms (1k steps, corrected LR schedule, frozen eval manifest excluded).
# B baseline / R matched non-directional repulsion / D anti-parallel L_dir /
# U latent-usage hinge only / S same-direction adverse sentinel (300 steps).
for _arm, _steps in [("sb_B", 1000), ("sb_R", 1000), ("sb_D", 1000),
                     ("sb_U", 1000), ("sb_S", 300),
                     ("sb_A0", 1000), ("sb_A1", 1000),          # α ablation (seed 42)
                     ("sb_A0_s1", 1000), ("sb_A1_s1", 1000)]:   # α ablation seed-variance (seed 1)
    RUN_SPECS.append(RunSpec(
        name=_arm,
        config=Path(f"code/config/ft_l40_{_arm}.yaml"),
        checkpoint=Path(f"/home/xuan/embodied-ai/checkpoints/lam-dis/ft_l40_{_arm}/step_{_steps:07d}"),
    ))
# Seed-variance runs (same recipe, training.seed 1/2) for the three headline models.
for _base, _cfg in [
    ("kl_ft_l40_full_5k", "ft_l40_full_5k"),
    ("ours_a_zero_5k", "ft_l40_ours_a_zero_5k"),
    ("ours_r2_opp_l01_5k", "ft_l40_ours_r2_opp_l01_5k"),
]:
    for _s in (1, 2):
        RUN_SPECS.append(RunSpec(
            name=f"{_base}_s{_s}",
            config=Path(f"code/config/{_cfg}_s{_s}.yaml"),
            checkpoint=Path(f"/home/xuan/embodied-ai/checkpoints/lam-dis/{_cfg}_s{_s}/step_0005000"),
        ))
DEFAULT_RUN_NAMES = ["raw_lam", "kl_ft_l40_full_5k", "contrastive_v3_5k"]


def _resolve(path: Path) -> Path:
    # Config paths in RUN_SPECS are relative (e.g. "code/config/<arm>.yaml").
    # After the lam_disentangle split, A-arm configs live under this project
    # (lam_cdlam_optimization/) and B's Stage-B (sb_*) configs live under the
    # sibling lam_condition_utilization/. Resolve against cwd first (back-compat),
    # then A's root, then B's root, so any arm resolves from either project.
    if path.is_absolute():
        return path
    _a_root = _HERE.parent                                    # lam_cdlam_optimization/
    _b_root = _a_root.parent / "lam_condition_utilization"    # sibling project (B configs)
    for root in (Path.cwd(), _a_root, _b_root):
        cand = root / path
        if cand.exists():
            return cand
    return _a_root / path


def _load_cfg(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _build_raw_model(cfg: dict, init_from: Path, device: torch.device) -> LatentActionModel:
    mcfg = cfg["model"]
    model = LatentActionModel(
        in_dim=mcfg.get("image_channels", 3),
        model_dim=mcfg.get("lam_model_dim", 512),
        latent_dim=mcfg.get("lam_latent_dim", 32),
        patch_size=mcfg.get("lam_patch_size", 16),
        enc_blocks=mcfg.get("lam_enc_blocks", 8),
        dec_blocks=mcfg.get("lam_dec_blocks", 8),
        num_heads=mcfg.get("lam_num_heads", 8),
        dropout=0.0,
        env_dim=mcfg.get("env_dim", 0),
        action_dim=mcfg.get("action_dim", 0),
        action_head=mcfg.get("action_head", False),
        contrastive=mcfg.get("contrastive", False),
        proj_dim=mcfg.get("proj_dim", 64),
    )

    class _Printer:
        def print(self, *args):
            print(*args)

    load_pretrained_lam(model, str(init_from), _Printer())
    model.to(device).eval()
    return model


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device_arg}, but CUDA is not available.")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


def _make_loader(cfg: dict, n_workers: int, batch_size: int, split: str = "val") -> DataLoader:
    dcfg = cfg["data"]
    ds = EgoDexDataset(
        data_root=dcfg["data_root"],
        img_h=dcfg.get("img_h", 240),
        img_w=dcfg.get("img_w", 320),
        downsample_factors=(1,),
        split=split,
        val_ratio=dcfg.get("val_ratio", 0.1),
        seed=dcfg.get("seed", 42),
        load_verbs=True,
        load_actions=True,
        action_dim=18,
        exclude_manifest=dcfg.get("exclude_manifest", None),
        only_manifest=dcfg.get("only_manifest", False),
    )
    g = torch.Generator()
    g.manual_seed(42)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        collate_fn=_collate,
        generator=g,
    )


def _encode_mu_var(model: LatentActionModel, videos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    out = model.encode(videos)
    return out["z_mu"], out["z_var"]


def _decode_from_mu(model: LatentActionModel, videos: torch.Tensor, z_mu: torch.Tensor) -> torch.Tensor:
    H, W = videos.shape[2:4]
    enc = model.encode(videos)
    patches = enc["patches"]
    z_rep = z_mu.reshape(videos.shape[0], videos.shape[1] - 1, 1, model.latent_dim)
    video_patches = model.patch_up(patches[:, :-1])
    action_patches = model.action_up(z_rep)
    recon_patches = model.decoder(video_patches + action_patches)
    return torch.sigmoid(unpatchify(recon_patches, model.patch_size, H, W))


def _effective_rank(x: np.ndarray) -> Tuple[float, float]:
    xc = x - x.mean(axis=0, keepdims=True)
    cov = (xc.T @ xc) / max(1, len(xc) - 1)
    eig = np.linalg.eigvalsh(cov)
    eig = np.clip(eig, 0.0, None)
    total = float(eig.sum())
    if total <= 1e-12:
        return 0.0, 0.0
    p = eig / total
    entropy = -float(np.sum(p[p > 0] * np.log(p[p > 0])))
    top5 = float(np.sort(eig)[-min(5, len(eig)):].sum() / total)
    return float(np.exp(entropy)), top5


def _eta2_per_dim(x: np.ndarray, labels: Iterable[str], min_count: int = 2) -> np.ndarray:
    labels = np.asarray(list(labels))
    total = ((x - x.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    between = np.zeros(x.shape[1], dtype=np.float64)
    for lab in np.unique(labels):
        idx = labels == lab
        if idx.sum() < min_count:
            continue
        delta = x[idx].mean(axis=0) - x.mean(axis=0)
        between += idx.sum() * (delta ** 2)
    return between / np.maximum(total, 1e-12)


def _shortcut_metrics(z: np.ndarray, ep: List[str], action_labels: List[str]) -> Dict[str, float]:
    ep_arr = np.asarray(ep)
    act_arr = np.asarray(action_labels)
    keep = act_arr != "unknown"
    if keep.sum() < 20:
        keep = np.ones(len(act_arr), dtype=bool)
    z = z[keep].astype(np.float64)
    ep_arr = ep_arr[keep]
    act_arr = act_arr[keep]
    z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
    d = 1.0 - z @ z.T
    off = ~np.eye(len(z), dtype=bool)
    same_ep = ep_arr[:, None] == ep_arr[None, :]
    same_action = act_arr[:, None] == act_arr[None, :]
    same_ep_diff_action = off & same_ep & (~same_action)
    diff_ep_same_action = off & (~same_ep) & same_action
    diff_action = off & (~same_action)

    def mean_or_nan(mask: np.ndarray) -> float:
        return float(d[mask].mean()) if mask.any() else float("nan")

    d_seda = mean_or_nan(same_ep_diff_action)
    d_desa = mean_or_nan(diff_ep_same_action)
    d_diff = mean_or_nan(diff_action)
    return {
        "d_same_ep_diff_action": d_seda,
        "d_diff_ep_same_action": d_desa,
        # Positive means visual/episode shortcut is closer than same-action across episodes.
        "episode_shortcut_leakage": d_desa - d_seda,
        "action_separation_ratio": d_diff / d_desa if d_desa and not math.isnan(d_desa) else float("nan"),
    }


def _task_shortcut_metrics(z: np.ndarray, task: List[str], action_labels: List[str]) -> Dict[str, float]:
    """Context shortcut diagnostic with non-empty pools on EgoDex.

    Episode-level verbs make same-episode/different-action pairs rare or empty.
    This variant asks whether same-task/different-action pairs are closer than
    different-task/same-action pairs. Positive leakage means task/context is
    pulling z_a more strongly than shared action.
    """
    task_arr = np.asarray(task)
    act_arr = np.asarray(action_labels)
    keep = act_arr != "unknown"
    if keep.sum() < 20:
        return {
            "task_shortcut_n": int(keep.sum()),
            "d_same_task_diff_action": float("nan"),
            "d_diff_task_same_action": float("nan"),
            "task_shortcut_leakage": float("nan"),
            "task_shortcut_ratio": float("nan"),
        }
    z = z[keep].astype(np.float64)
    task_arr = task_arr[keep]
    act_arr = act_arr[keep]
    z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
    d = 1.0 - z @ z.T
    off = ~np.eye(len(z), dtype=bool)
    same_task = task_arr[:, None] == task_arr[None, :]
    same_action = act_arr[:, None] == act_arr[None, :]
    same_task_diff_action = off & same_task & (~same_action)
    diff_task_same_action = off & (~same_task) & same_action

    def mean_or_nan(mask: np.ndarray) -> float:
        return float(d[mask].mean()) if mask.any() else float("nan")

    d_stda = mean_or_nan(same_task_diff_action)
    d_dtsa = mean_or_nan(diff_task_same_action)
    return {
        "task_shortcut_n": int(keep.sum()),
        "d_same_task_diff_action": d_stda,
        "d_diff_task_same_action": d_dtsa,
        "task_shortcut_leakage": d_dtsa - d_stda
        if not math.isnan(d_stda) and not math.isnan(d_dtsa) else float("nan"),
        "task_shortcut_ratio": d_stda / d_dtsa
        if d_dtsa and not math.isnan(d_stda) and not math.isnan(d_dtsa) else float("nan"),
    }


_SCENE_KNN_KEYS = (
    "scene_knn_ep_top1", "scene_knn_ep_top5", "scene_knn_ep_null",
    "scene_knn_task_top1", "scene_knn_task_top5", "scene_knn_task_null",
)


def _scene_knn_metrics(z: np.ndarray, ep: List[str], task: List[str]) -> Dict[str, float]:
    """Scene/identity retrieval confound on z_a: does the action code retrieve
    "same scene" instead of "same action"?

    Fraction of cosine nearest neighbours (self excluded) sharing the query's
    episode / task. `*_null` is the analytic random-retrieval expectation
    Σ_g n_g(n_g−1) / (N(N−1)) — compare against it, not against zero. Fills the
    slot left by `episode_shortcut_leakage`, whose same-episode/different-action
    pool is empty under episode-level verbs (all-nan in v2).

    NOT CD-LAM's `id_ratio`: that label (project page, 0.527 -> 0.047) is their
    median ZERO-TRANSITION response (id = identical frames), i.e. our
    `static_response_rel_median` — see notes/cdlam_repro_fidelity.md. This metric
    is ours; no CD-LAM number is comparable to it.
    """
    n = len(z)
    if n < 10:
        return {k: float("nan") for k in _SCENE_KNN_KEYS}
    z = z.astype(np.float64)
    z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
    d = 1.0 - z @ z.T
    np.fill_diagonal(d, np.inf)
    order = np.argsort(d, axis=1)[:, :5]
    out: Dict[str, float] = {}
    for name, labels in (("ep", ep), ("task", task)):
        arr = np.asarray(labels)
        same = arr[order] == arr[:, None]                       # [n, 5]
        counts = np.unique(arr, return_counts=True)[1].astype(np.float64)
        out[f"scene_knn_{name}_top1"] = float(same[:, 0].mean())
        out[f"scene_knn_{name}_top5"] = float(same.mean())
        out[f"scene_knn_{name}_null"] = float((counts * (counts - 1)).sum() / (n * (n - 1)))
    return out


def _opposite_verb_map() -> Dict[str, str]:
    pairs = [
        ("insert", "remove"),
        ("assemble", "disassemble"),
        ("stack", "unstack"),
        ("charge", "uncharge"),
        ("open", "close"),
        ("screw", "unscrew"),
        ("tie", "untie"),
        ("zip", "unzip"),
        ("fold", "unfold"),
        ("stock", "unstock"),
        ("pick", "put"),
        ("scoop", "dump"),
        ("lock", "unlock"),
        ("add", "remove"),
        ("wrap", "unwrap"),
        ("push", "pull"),
    ]
    out = {}
    for a, b in pairs:
        out[a] = b
        out[b] = a
    return out


def _norm_verb(v: str) -> str:
    return str(v).strip().lower().replace("_", " ")


def _reversible_metrics(z: np.ndarray, task: List[str], ep: List[str], verb: List[str]) -> Dict[str, float]:
    """Hard-negative geometry for reversible primitives.

    Good z_a geometry should keep same-verb/different-episode pairs closer than
    visually similar opposite-verb pairs. The most diagnostic pool is
    same-task/opposite-verb because task is a strong scene/context proxy in EgoDex.
    """
    verb_arr = np.asarray([_norm_verb(v) for v in verb])
    task_arr = np.asarray(task)
    ep_arr = np.asarray(ep)
    # Dataset-native opposite: within a reversible EgoDex task the two resolved verbs
    # ARE the opposite pair (llm_verbs); "same task + different verb" = opposite. No
    # hardcoded opposite-verb map — this covers ALL reversible tasks, not just the 41
    # whose pairs happened to be listed.
    known = (verb_arr != "unknown")
    if known.sum() < 8:
        return {
            "rev_n": int(known.sum()),
            "rev_d_same_verb_diff_ep": float("nan"),
            "rev_d_same_task_opposite": float("nan"),
            "rev_d_any_opposite": float("nan"),
            "rev_hard_margin": float("nan"),
            "rev_nearest_pos_beats_opp": float("nan"),
        }

    z = z[known].astype(np.float64)
    z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
    verb_arr = verb_arr[known]
    task_arr = task_arr[known]
    ep_arr = ep_arr[known]
    d = 1.0 - z @ z.T
    n = len(z)
    off = ~np.eye(n, dtype=bool)
    same_verb = verb_arr[:, None] == verb_arr[None, :]
    same_ep = ep_arr[:, None] == ep_arr[None, :]
    same_task = task_arr[:, None] == task_arr[None, :]

    pos = same_verb & (~same_ep) & off
    same_task_opp = same_task & (~same_verb) & off   # opposite = same task, different resolved verb
    any_opp = same_task_opp                          # cross-task different-verb is not "opposite"

    def mean_or_nan(mask: np.ndarray) -> float:
        return float(d[mask].mean()) if mask.any() else float("nan")

    d_pos = mean_or_nan(pos)
    d_same_task_opp = mean_or_nan(same_task_opp)
    d_any_opp = mean_or_nan(any_opp)

    wins = []
    for i in range(n):
        pos_i = pos[i]
        opp_i = same_task_opp[i] if same_task_opp[i].any() else any_opp[i]
        if pos_i.any() and opp_i.any():
            wins.append(float(d[i, pos_i].min() < d[i, opp_i].min()))
    return {
        "rev_n": int(n),
        "rev_d_same_verb_diff_ep": d_pos,
        "rev_d_same_task_opposite": d_same_task_opp,
        "rev_d_any_opposite": d_any_opp,
        # Positive is good: opposite actions are farther than same action across episodes.
        "rev_hard_margin": d_same_task_opp - d_pos
        if not math.isnan(d_same_task_opp) and not math.isnan(d_pos) else float("nan"),
        "rev_nearest_pos_beats_opp": float(np.mean(wins)) if wins else float("nan"),
    }


def _ridge_action_scores(x: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    if len(x) < 20:
        nan_dims = np.full(x.shape[1], float("nan"), dtype=np.float64)
        return float("nan"), nan_dims.copy(), nan_dims.copy()

    idx_train, idx_test = train_test_split(np.arange(len(x)), test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(x[idx_train])
    xs = scaler.transform(x)
    reg = Ridge(alpha=1.0).fit(xs[idx_train], y[idx_train])
    pred = reg.predict(xs[idx_test])
    full_r2 = float(r2_score(y[idx_test], pred))

    drops = np.zeros(x.shape[1], dtype=np.float64)
    for j in range(x.shape[1]):
        x_abl = xs[idx_test].copy()
        x_abl[:, j] = 0.0
        pred_abl = reg.predict(x_abl)
        drops[j] = full_r2 - float(r2_score(y[idx_test], pred_abl))

    single = np.zeros(x.shape[1], dtype=np.float64)
    for j in range(x.shape[1]):
        sj = xs[:, [j]]
        reg_j = Ridge(alpha=1.0).fit(sj[idx_train], y[idx_train])
        pred_j = reg_j.predict(sj[idx_test])
        single[j] = float(r2_score(y[idx_test], pred_j))
    return full_r2, drops, single


def _translate_videos(videos: torch.Tensor, dx: int = 0, dy: int = 0) -> torch.Tensor:
    """Zero-filled image translation for [B,T,H,W,C] videos."""
    out = torch.zeros_like(videos)
    h, w = videos.shape[2:4]
    src_y0 = max(0, -dy)
    src_y1 = h - max(0, dy)
    dst_y0 = max(0, dy)
    dst_y1 = h - max(0, -dy)
    src_x0 = max(0, -dx)
    src_x1 = w - max(0, dx)
    dst_x0 = max(0, dx)
    dst_x1 = w - max(0, -dx)
    if src_y1 > src_y0 and src_x1 > src_x0:
        out[:, :, dst_y0:dst_y1, dst_x0:dst_x1, :] = videos[:, :, src_y0:src_y1, src_x0:src_x1, :]
    return out


def _camera_shift_metrics(
    model: LatentActionModel,
    loader: DataLoader,
    device: torch.device,
    n_samples: int,
    shift_px: int = 8,
) -> Dict[str, float]:
    rel_h, rel_v = [], []
    seen = 0
    with torch.no_grad():
        for batch in loader:
            videos = batch["videos"].to(device)
            mu, _ = _encode_mu_var(model, videos)
            h_mu, _ = _encode_mu_var(model, _translate_videos(videos, dx=shift_px))
            v_mu, _ = _encode_mu_var(model, _translate_videos(videos, dy=shift_px))
            denom = torch.linalg.norm(mu, dim=1).clamp(min=1e-8)
            rel_h.append((torch.linalg.norm(h_mu - mu, dim=1) / denom).cpu().numpy())
            rel_v.append((torch.linalg.norm(v_mu - mu, dim=1) / denom).cpu().numpy())
            seen += videos.shape[0]
            if seen >= n_samples:
                break
    h = np.concatenate(rel_h, axis=0)[:n_samples]
    v = np.concatenate(rel_v, axis=0)[:n_samples]
    return {
        "camera_shift_px": shift_px,
        "camera_shift_h_mean": float(h.mean()),
        "camera_shift_h_median": float(np.median(h)),
        "camera_shift_v_mean": float(v.mean()),
        "camera_shift_v_median": float(np.median(v)),
    }


def _temporal_reverse_metrics(
    model: LatentActionModel,
    loader: DataLoader,
    device: torch.device,
    n_samples: int,
) -> Dict[str, float]:
    za_cos, ze_cos, za_delta_rel, ze_delta_rel = [], [], [], []
    action_part = model.action_part
    seen = 0
    with torch.no_grad():
        for batch in loader:
            videos = batch["videos"].to(device)
            mu, _ = _encode_mu_var(model, videos)
            rmu, _ = _encode_mu_var(model, videos.flip(dims=[1]))
            za = torch.nn.functional.normalize(mu[:, :action_part], dim=1)
            rza = torch.nn.functional.normalize(rmu[:, :action_part], dim=1)
            za_cos.append((za * rza).sum(1).cpu().numpy())
            za_delta_rel.append(
                (torch.linalg.norm(rmu[:, :action_part] - mu[:, :action_part], dim=1)
                 / torch.linalg.norm(mu[:, :action_part], dim=1).clamp(min=1e-8)).cpu().numpy()
            )
            if model.env_dim > 0:
                ze = torch.nn.functional.normalize(mu[:, action_part:], dim=1)
                rze = torch.nn.functional.normalize(rmu[:, action_part:], dim=1)
                ze_cos.append((ze * rze).sum(1).cpu().numpy())
                ze_delta_rel.append(
                    (torch.linalg.norm(rmu[:, action_part:] - mu[:, action_part:], dim=1)
                     / torch.linalg.norm(mu[:, action_part:], dim=1).clamp(min=1e-8)).cpu().numpy()
                )
            seen += videos.shape[0]
            if seen >= n_samples:
                break

    za_c = np.concatenate(za_cos, axis=0)[:n_samples]
    za_d = np.concatenate(za_delta_rel, axis=0)[:n_samples]
    if ze_cos:
        ze_c = np.concatenate(ze_cos, axis=0)[:n_samples]
        ze_d = np.concatenate(ze_delta_rel, axis=0)[:n_samples]
        ze_cos_mean = float(np.nanmean(ze_c))
        ze_cos_median = float(np.nanmedian(ze_c))
        ze_delta_median = float(np.nanmedian(ze_d))
    else:
        ze_cos_mean = float("nan")
        ze_cos_median = float("nan")
        ze_delta_median = float("nan")
    return {
        # Lower z_a cosine is better for directional sensitivity; higher z_e cosine is better invariance.
        "reverse_za_cos_mean": float(np.nanmean(za_c)),
        "reverse_za_cos_median": float(np.nanmedian(za_c)),
        "reverse_ze_cos_mean": ze_cos_mean,
        "reverse_ze_cos_median": ze_cos_median,
        "reverse_za_delta_rel_median": float(np.nanmedian(za_d)),
        "reverse_ze_delta_rel_median": ze_delta_median,
    }


def _percentile_threshold(values: np.ndarray, q: float) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return float("inf")
    return float(np.quantile(finite, q))


def _dim_labels(rows: List[dict]) -> None:
    action_thr = _percentile_threshold(
        np.asarray([max(r["action_r2_drop"], r["single_dim_action_r2"]) for r in rows]), 0.75
    )
    decoder_thr = _percentile_threshold(np.asarray([r["decoder_total_effect"] for r in rows]), 0.75)
    leak_thr = _percentile_threshold(np.asarray([r["leakage_score"] for r in rows]), 0.75)
    active_thr = _percentile_threshold(np.asarray([r["std"] for r in rows]), 0.20)
    for r in rows:
        action_score = max(r["action_r2_drop"], r["single_dim_action_r2"])
        if r["std"] <= active_thr:
            label = "inactive_or_low_var"
        elif action_score >= action_thr and r["decoder_total_effect"] >= decoder_thr:
            label = "action_effective"
        elif r["leakage_score"] >= leak_thr and action_score < action_thr:
            label = "scene_episode_leaky"
        elif action_score >= action_thr:
            label = "action_informative"
        elif r["decoder_total_effect"] >= decoder_thr:
            label = "decoder_effect"
        else:
            label = "mixed_or_weak"
        r["label"] = label


def _collect(
    model: LatentActionModel,
    loader: DataLoader,
    device: torch.device,
    n_samples: int,
) -> dict:
    normal_mu, normal_var = [], []
    static_mu = []
    actions, tasks, eps, verbs = [], [], [], []
    seen = 0
    with torch.no_grad():
        for batch in loader:
            videos = batch["videos"].to(device)
            b = videos.shape[0]
            mu, var = _encode_mu_var(model, videos)

            static_videos = videos.clone()
            static_videos[:, 1] = static_videos[:, 0]
            smu, _ = _encode_mu_var(model, static_videos)

            normal_mu.append(mu.cpu().numpy())
            normal_var.append(var.cpu().numpy())
            static_mu.append(smu.cpu().numpy())
            actions.append(batch["action"].cpu().numpy())
            tasks.extend(batch["task_id"])
            eps.extend(batch["ep_id"])
            verbs.extend(batch.get("verb_id", ["unknown"] * b))
            seen += b
            if seen >= n_samples:
                break

    def cat(xs):
        return np.concatenate(xs, axis=0)[:n_samples]

    return {
        "mu": cat(normal_mu),
        "logvar": cat(normal_var),
        "static_mu": cat(static_mu),
        "action": cat(actions),
        "task": tasks[:n_samples],
        "ep": eps[:n_samples],
        "verb": verbs[:n_samples],
    }


def _decoder_effects(
    model: LatentActionModel,
    loader: DataLoader,
    device: torch.device,
    per_dim_std: np.ndarray,
    n_samples: int,
    perturb_scale: float,
) -> np.ndarray:
    videos_all = []
    seen = 0
    for batch in loader:
        videos_all.append(batch["videos"])
        seen += batch["videos"].shape[0]
        if seen >= n_samples:
            break
    videos = torch.cat(videos_all, dim=0)[:n_samples].to(device)
    effects = []

    with torch.no_grad():
        mu, _ = _encode_mu_var(model, videos)
        base = _decode_from_mu(model, videos, mu)
        for j in range(model.latent_dim):
            step = float(per_dim_std[j] * perturb_scale)
            if step <= 1e-8 or not np.isfinite(step):
                effects.append(0.0)
                continue
            pert = mu.clone()
            pert[:, j] += step
            recon = _decode_from_mu(model, videos, pert)
            diff = (recon - base).abs().mean(dim=[1, 2, 3, 4])  # [B]
            effects.append(float(diff.mean().item()))
    return np.asarray(effects, dtype=np.float64)


# ── Tier-A standard (non-circular) metrics ───────────────────────────────────
# Anti-circularity rule: freeze the encoder, read out with a lightweight probe,
# and score against an EXTERNAL target (true 18-D SE(3) action or a decoded
# rollout) — never a quantity a loss directly shapes. See the literature basis:
# villa-X (2507.23682) frozen MLP L1; LAPO (2312.10812)/Genie (2402.15391)
# label-efficiency; Genie Δt PSNR controllability; DINO (2104.14294) kNN.

def _mlp_action_probe(x: np.ndarray, y: np.ndarray, seed: int = 42,
                      with_shuffle: bool = True) -> Dict[str, float]:
    """Frozen-latent -> true 18-D action via a small MLP (villa-X style).

    Reports R2, mean L1, and max per-dim L1. A label-permutation control
    (`_shuffled`) must collapse to ~0 R2 — proof the probe reads out rather than
    relearns. The encoder is never updated (x is the frozen z_mu).
    `with_shuffle=False` skips the (costly) permutation fit — used for the z_a/z_e
    subspace probes whose control is already established on the full latent.
    """
    out = {
        "mlp_action_r2": float("nan"), "mlp_action_l1": float("nan"),
        "mlp_action_max_l1": float("nan"), "mlp_action_r2_shuffled": float("nan"),
    }
    if len(x) < 50 or x.shape[1] == 0:
        return out
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score

    idx_tr, idx_te = train_test_split(np.arange(len(x)), test_size=0.2, random_state=seed)
    xs = StandardScaler().fit(x[idx_tr]).transform(x)

    def fit_eval(xtr, ytr, xte, yte):
        reg = MLPRegressor(hidden_layer_sizes=(256, 256), activation="relu",
                           max_iter=400, early_stopping=True, random_state=seed)
        reg.fit(xtr, ytr)
        pred = reg.predict(xte)
        return (float(r2_score(yte, pred)),
                float(np.abs(pred - yte).mean()),
                float(np.abs(pred - yte).mean(axis=0).max()))

    r2, l1, max_l1 = fit_eval(xs[idx_tr], y[idx_tr], xs[idx_te], y[idx_te])
    out.update(mlp_action_r2=r2, mlp_action_l1=l1, mlp_action_max_l1=max_l1)
    if with_shuffle:
        # label-permutation control (break x<->y correspondence)
        perm = np.random.RandomState(seed).permutation(len(y))
        yp = y[perm]
        r2s, _, _ = fit_eval(xs[idx_tr], yp[idx_tr], xs[idx_te], yp[idx_te])
        out.update(mlp_action_r2_shuffled=r2s)
    return out


def _suffixed(d: Dict[str, float], suffix: str) -> Dict[str, float]:
    return {f"{k}{suffix}": v for k, v in d.items()}


def _action_r2_groups(x: np.ndarray, y: np.ndarray, seed: int = 42) -> Dict[str, float]:
    """Per-target-group linear R2 on z_a. EgoDex 18-D layout (dataset._pair_action):
    [L dp 0:3, L rot6d 3:9, R dp 9:12, R rot6d 12:18], motion in the hand's own frame.
    Tracks the 'rotation decodes worse than translation' failure mode with an external
    target — the fine-grained-control claim lives or dies on the rot groups."""
    out = {"action_r2_trans": float("nan"), "action_r2_rot": float("nan"),
           "action_r2_left": float("nan"), "action_r2_right": float("nan")}
    if len(x) < 20 or x.shape[1] == 0 or y.shape[1] != 18:
        return out
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    idx_tr, idx_te = train_test_split(np.arange(len(x)), test_size=0.2, random_state=seed)
    xs = StandardScaler().fit(x[idx_tr]).transform(x)
    reg = Ridge(alpha=1.0).fit(xs[idx_tr], y[idx_tr])
    pred = reg.predict(xs[idx_te])
    per_dim = np.asarray([r2_score(y[idx_te][:, j], pred[:, j]) for j in range(18)])
    trans = np.r_[0:3, 9:12]
    rot = np.r_[3:9, 12:18]
    out.update(
        action_r2_trans=float(per_dim[trans].mean()),
        action_r2_rot=float(per_dim[rot].mean()),
        action_r2_left=float(per_dim[:9].mean()),
        action_r2_right=float(per_dim[9:].mean()),
    )
    return out


def _ridge_r2_taskheld(x: np.ndarray, y: np.ndarray, tasks: List[str],
                       n_splits: int = 5, seed: int = 42) -> float:
    """Linear-probe R2 with HELD-OUT TASKS (GroupKFold over task id): can a readout
    trained on some tasks decode true actions on unseen tasks? The gap to the
    standard shuffled-split R2 is an EXTERNAL measure of context dependence of the
    action code (a context-invariant z_a generalizes across tasks; a scene-tangled
    one does not)."""
    if len(x) < 100 or x.shape[1] == 0:
        return float("nan")
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    groups = np.asarray(tasks)
    if len(np.unique(groups)) < n_splits:
        return float("nan")
    scores = []
    for tr, te in GroupKFold(n_splits=n_splits).split(x, y, groups):
        sc = StandardScaler().fit(x[tr])
        reg = Ridge(alpha=1.0).fit(sc.transform(x[tr]), y[tr])
        scores.append(r2_score(y[te], reg.predict(sc.transform(x[te]))))
    return float(np.mean(scores))


def _ridge_r2(x: np.ndarray, y: np.ndarray, seed: int = 42) -> float:
    """Held-out linear-probe R2 (frozen latent -> true action). Subspace counterpart
    of `action_r2_full_latent`; the z_a/z_e/full triple measures ROUTING with an
    external target: R2(z_e) is action leaked into the env subspace, and
    R2(full) − R2(z_a) is action info z_a failed to absorb."""
    if len(x) < 20 or x.shape[1] == 0:
        return float("nan")
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    idx_tr, idx_te = train_test_split(np.arange(len(x)), test_size=0.2, random_state=seed)
    xs = StandardScaler().fit(x[idx_tr]).transform(x)
    reg = Ridge(alpha=1.0).fit(xs[idx_tr], y[idx_tr])
    return float(r2_score(y[idx_te], reg.predict(xs[idx_te])))


def _label_efficiency(x: np.ndarray, y: np.ndarray,
                      fracs=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0), seed: int = 42) -> Dict[str, float]:
    """Linear-probe R2 vs labeled-fraction curve (LAPO/Genie). Sharpest
    discriminator between a latent that encodes actions and one that memorizes
    appearance; report per-fraction R2 + area under the log-fraction curve.

    Probe is RidgeCV (LOO-CV over a wide alpha grid) — a fixed alpha=1.0 overfits
    badly at the small fractions (e.g. 12 samples x 40 dims at 1%), which used to
    drive every run's curve negative and made the metric pure probe variance."""
    res = {f"labeleff_r2_{int(round(f*100))}pct": float("nan") for f in fracs}
    res["labeleff_auc"] = float("nan")
    if len(x) < 50:
        return res
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score

    idx_tr, idx_te = train_test_split(np.arange(len(x)), test_size=0.2, random_state=seed)
    xs = StandardScaler().fit(x[idx_tr]).transform(x)
    rng = np.random.RandomState(seed)
    alphas = np.logspace(-2, 5, 15)
    r2s = []
    for f in fracs:
        k = min(len(idx_tr), max(10, int(len(idx_tr) * f)))
        sub = rng.choice(idx_tr, size=k, replace=False)
        reg = RidgeCV(alphas=alphas).fit(xs[sub], y[sub])
        r2 = float(r2_score(y[idx_te], reg.predict(xs[idx_te])))
        r2s.append(r2)
        res[f"labeleff_r2_{int(round(f*100))}pct"] = r2
    lf = np.log10(np.asarray(fracs))
    _trapz = getattr(np, "trapezoid", np.trapz)
    res["labeleff_auc"] = float(_trapz(r2s, lf) / (lf[-1] - lf[0]))
    return res


def _action_knn(z: np.ndarray, verb: List[str], ep: List[str], k: int = 5,
                prefix: str = "action_knn", standardize: bool = False,
                n_null: int = 20, seed: int = 42) -> Dict[str, float]:
    """kNN in a representation space (DINO/SigLIP zero-trained-readout spirit): does
    the CROSS-episode nearest neighbour share the same action verb?

    Null model: the majority-verb frequency is the baseline of a classifier that
    always predicts the majority class — the WRONG null for retrieval agreement
    (whose analytic null is ~sum_c p_c^2, much lower for a long-tail verb set).
    `{prefix}_null_top1_mean/std` is the proper null: verb labels permuted at the
    EPISODE level (preserving the verb histogram, the episode structure, and the
    retrieval graph), scored with the SAME neighbours. Read
    `{prefix}_top1_z = (top1 − null_mean) / null_std` — |z| < 2 is chance-level.
    The old majority baseline is kept as `{prefix}_chance_majority` for reference.

    `standardize=True` z-scores columns before cosine (use for physical action
    spaces whose columns have heterogeneous units, e.g. the 18-D SE(3) target)."""
    out = {f"{prefix}_verb_top1": float("nan"), f"{prefix}_verb_maj5": float("nan"),
           f"{prefix}_chance_majority": float("nan"),
           f"{prefix}_null_top1_mean": float("nan"), f"{prefix}_null_top1_std": float("nan"),
           f"{prefix}_null_maj5_mean": float("nan"), f"{prefix}_top1_z": float("nan")}
    verb_arr = np.asarray([_norm_verb(v) for v in verb])
    ep_arr = np.asarray(ep)
    known = verb_arr != "unknown"
    if known.sum() < 20:
        return out
    zk = z[known].astype(np.float64)
    if standardize:
        zk = (zk - zk.mean(axis=0, keepdims=True)) / (zk.std(axis=0, keepdims=True) + 1e-8)
    zk = zk / (np.linalg.norm(zk, axis=1, keepdims=True) + 1e-8)
    vk = verb_arr[known]
    ek = ep_arr[known]
    n = len(zk)
    d = 1.0 - zk @ zk.T
    same_ep = ek[:, None] == ek[None, :]
    d[same_ep] = np.inf                       # cross-episode retrieval only
    # Neighbour graph is computed ONCE; observed score and permutation nulls reuse it.
    nn_lists: List[np.ndarray] = []
    for i in range(n):
        order = np.argsort(d[i])
        nn = np.asarray([j for j in order[:k] if np.isfinite(d[i, j])], dtype=np.int64)
        nn_lists.append(nn)

    def _score(labels: np.ndarray) -> Tuple[float, float]:
        top1, maj5 = [], []
        for i, nn in enumerate(nn_lists):
            if len(nn) == 0:
                continue
            top1.append(float(labels[nn[0]] == labels[i]))
            maj5.append(float(np.mean(labels[nn] == labels[i]) >= 0.5))
        return (float(np.mean(top1)) if top1 else float("nan"),
                float(np.mean(maj5)) if maj5 else float("nan"))

    t1, m5 = _score(vk)
    # Episode-level permutation null (episodes are verb-pure in EgoDex).
    uniq_eps = np.unique(ek)
    ep_verb = np.asarray([vk[ek == e][0] for e in uniq_eps])
    rng = np.random.RandomState(seed)
    null_t1, null_m5 = [], []
    for _ in range(n_null):
        perm_verbs = ep_verb[rng.permutation(len(uniq_eps))]
        remap = dict(zip(uniq_eps.tolist(), perm_verbs.tolist()))
        vperm = np.asarray([remap[e] for e in ek])
        a, b = _score(vperm)
        null_t1.append(a)
        null_m5.append(b)
    null_t1 = np.asarray(null_t1, dtype=np.float64)
    null_m5 = np.asarray(null_m5, dtype=np.float64)
    vals, counts = np.unique(vk, return_counts=True)
    null_std = float(null_t1.std())
    out.update({
        f"{prefix}_verb_top1": t1,
        f"{prefix}_verb_maj5": m5,
        f"{prefix}_chance_majority": float(counts.max() / counts.sum()),
        f"{prefix}_null_top1_mean": float(null_t1.mean()),
        f"{prefix}_null_top1_std": null_std,
        f"{prefix}_null_maj5_mean": float(null_m5.mean()),
        f"{prefix}_top1_z": float((t1 - null_t1.mean()) / null_std) if null_std > 1e-12 else float("nan"),
    })
    return out


def _ridge_cross_pred(x: np.ndarray, y: np.ndarray, seed: int = 42) -> np.ndarray:
    """Out-of-fold ridge predictions of the true action from z_a (5-fold). Used to
    re-run the verb kNN in the PROBE-PREDICTED action space: if kNN jumps there while
    raw-z_a kNN sits at chance, action info exists but is variance-dominated by
    nuisance directions (metric misallocation, not information absence)."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    return cross_val_predict(pipe, x, y, cv=cv)


def _collect_za_labels(model: LatentActionModel, cfg: dict, device: torch.device,
                       num_workers: int, batch_size: int, max_total: int = 8000):
    """Encode-only pass gathering z_a + task + verb over a larger, task-diverse val
    sample. Needed because per-task opposite-verb classification requires enough
    samples per reversible task (~8/verb), which the main n_samples pool is too thin
    to provide (1500 → ~2 tasks; 8000 → ~21 tasks)."""
    loader = _make_loader(cfg, num_workers, batch_size)
    action_part = model.action_part
    zs, tasks, verbs = [], [], []
    seen = 0
    with torch.no_grad():
        for batch in loader:
            videos = batch["videos"].to(device)
            mu, _ = _encode_mu_var(model, videos)
            zs.append(mu[:, :action_part].cpu().numpy())
            tasks.extend(batch["task_id"])
            verbs.extend(batch.get("verb_id", ["unknown"] * videos.shape[0]))
            seen += videos.shape[0]
            if seen >= max_total:
                break
    return np.concatenate(zs, axis=0)[:max_total], tasks[:max_total], verbs[:max_total]


def _opposite_verb_classification(z: np.ndarray, task: List[str], verb: List[str],
                                  seed: int = 42) -> Dict[str, float]:
    """FAIR, non-circular opposite-action metric: within each reversible task (its two
    resolved verbs = the opposite pair), train a frozen-latent logistic-regression probe
    on z_a to classify verb A vs B; report mean held-out BALANCED ACCURACY (chance 0.5).

    Unlike `rev_hard_margin` (a raw cosine-distance margin = the loss's own objective),
    this is a trained discriminative probe on external verb labels with a held-out split —
    a downstream consequence of separability, not the loss quantity itself. Scene is held
    constant (within-task), so it isolates pull/push discriminability. `_shuffled` is a
    label-permutation control that MUST sit at ~0.5.

    CONFOUND WARNING: within a reversible task the INITIAL scene state correlates
    perfectly with the verb (a socket about to be `remove`d is occupied; one about to
    receive `insert` is empty), so a z_a that only encodes initial-frame appearance can
    also score high. run_one therefore runs the SAME probe on the static-pair latent
    E(o_t, o_t) (`_static` keys): credit direction encoding only to the extent
    `opp_cls_bal_acc` exceeds `opp_cls_bal_acc_static` (`opp_cls_direction_gain`).
    """
    out = {"opp_cls_bal_acc": float("nan"), "opp_cls_bal_acc_shuffled": float("nan"),
           "opp_cls_n_tasks": 0.0}
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    verb_arr = np.asarray([_norm_verb(v) for v in verb])
    task_arr = np.asarray(task)
    rng = np.random.RandomState(seed)
    accs, accs_shuf = [], []
    for t in np.unique(task_arr):
        m = task_arr == t
        vt, zt = verb_arr[m], z[m].astype(np.float64)
        uv = [v for v in np.unique(vt) if v != "unknown"]
        if len(uv) != 2:                       # only reversible tasks (exactly 2 verbs)
            continue
        keep = np.isin(vt, uv)
        zt, yt = zt[keep], (vt[keep] == uv[0]).astype(int)
        if yt.sum() < 8 or (len(yt) - yt.sum()) < 8:
            continue
        try:
            itr, ite = train_test_split(np.arange(len(zt)), test_size=0.3,
                                        random_state=seed, stratify=yt)
        except ValueError:
            continue
        sc = StandardScaler().fit(zt[itr])
        zs = sc.transform(zt)
        clf = LogisticRegression(max_iter=1000, C=1.0).fit(zs[itr], yt[itr])
        accs.append(balanced_accuracy_score(yt[ite], clf.predict(zs[ite])))
        ysh = rng.permutation(yt)              # label-permutation control
        clf2 = LogisticRegression(max_iter=1000, C=1.0).fit(zs[itr], ysh[itr])
        accs_shuf.append(balanced_accuracy_score(ysh[ite], clf2.predict(zs[ite])))
    if accs:
        out.update(opp_cls_bal_acc=float(np.mean(accs)),
                   opp_cls_bal_acc_shuffled=float(np.mean(accs_shuf)),
                   opp_cls_n_tasks=float(len(accs)))
    return out


_DIRPROBE_KEYS = (
    "dirprobe_nll_base", "dirprobe_nll_full", "dirprobe_nll_gain",
    "dirprobe_gain_ci_lo", "dirprobe_gain_ci_hi", "dirprobe_gain_perm_p",
    "dirprobe_null_mean", "dirprobe_null_std", "dirprobe_gain_vs_null",
    "dirprobe_auc_base", "dirprobe_auc_full", "dirprobe_n_tasks", "dirprobe_n_samples",
)


def _conditional_direction_probe(z_trans: np.ndarray, z_static: np.ndarray,
                                 task: List[str], verb: List[str], ep: List[str],
                                 seed: int = 42, n_boot: int = 1000,
                                 n_perm: int = 20) -> Dict[str, float]:
    """HEADLINE direction metric: does the transition latent add information about the
    opposite-verb label BEYOND the static-appearance latent?

    Replaces `opp_cls_direction_gain` (kept as `static_shortcut_gap`, a leakage
    diagnostic). That metric was a difference of two INDEPENDENT probe accuracies,
    not a conditional information gain: with the static probe at 0.96-0.99 the
    headroom for a positive value is 0.006-0.040, so a latent that both sheds
    appearance and gains direction can still score negative.

    Here both probes share one feature layout and differ only in information:
        baseline : [Z_static, 0]
        full     : [Z_static, Z_trans - Z_static]
    Splits are GroupKFold over EPISODE (labels are episode-level, so a random split
    would leak). We report held-out NLL rather than balanced accuracy because
    accuracy saturates near the appearance ceiling. `gain = NLL_base - NLL_full`,
    so >0 means the transition latent carries conditional information.

    CI is a cluster bootstrap over episodes (samples inside an episode are not
    independent); the null permutes the episode->verb assignment within each task.
    """
    out = {k: float("nan") for k in _DIRPROBE_KEYS}
    out["dirprobe_n_tasks"], out["dirprobe_n_samples"] = 0.0, 0.0
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    verb_arr = np.asarray([_norm_verb(v) for v in verb])
    task_arr, ep_arr = np.asarray(task), np.asarray(ep)
    rng = np.random.RandomState(seed)

    def fit_task(zs, zt, y, groups, y_override=None):
        """Return per-sample held-out NLL for (base, full) plus pooled probs."""
        yy = y if y_override is None else y_override
        n_groups = len(np.unique(groups))
        if n_groups < 3:
            return None
        n_splits = min(5, n_groups)
        base_nll, full_nll, ys, pb, pf, te_used = [], [], [], [], [], []
        delta = zt - zs
        X_full = np.concatenate([zs, delta], axis=1)
        X_base = np.concatenate([zs, np.zeros_like(delta)], axis=1)
        for tr, te in GroupKFold(n_splits=n_splits).split(X_full, yy, groups):
            if len(np.unique(yy[tr])) < 2 or len(np.unique(yy[te])) < 2:
                continue
            for X, bucket, probs in ((X_base, base_nll, pb), (X_full, full_nll, pf)):
                sc = StandardScaler().fit(X[tr])
                clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X[tr]), yy[tr])
                p = clf.predict_proba(sc.transform(X[te]))[:, 1]
                p = np.clip(p, 1e-6, 1 - 1e-6)
                bucket.append(-(yy[te] * np.log(p) + (1 - yy[te]) * np.log(1 - p)))
                probs.append(p)
            ys.append(yy[te])
            te_used.append(te)
        if not ys:
            return None
        # te_used lets the caller align per-sample results with episode ids: folds
        # with a single class are skipped, so the fold order cannot be rebuilt outside.
        return (np.concatenate(base_nll), np.concatenate(full_nll),
                np.concatenate(ys), np.concatenate(pb), np.concatenate(pf),
                np.concatenate(te_used))

    per_sample, n_tasks = [], 0
    for t in np.unique(task_arr):
        m = task_arr == t
        uv = [v for v in np.unique(verb_arr[m]) if v != "unknown"]
        if len(uv) != 2:
            continue
        keep = m & np.isin(verb_arr, uv)
        y = (verb_arr[keep] == uv[0]).astype(int)
        if y.sum() < 8 or (len(y) - y.sum()) < 8:
            continue
        res = fit_task(z_static[keep].astype(np.float64), z_trans[keep].astype(np.float64),
                       y, ep_arr[keep])
        if res is None:
            continue
        n_tasks += 1
        # episode id is only unique within a task; namespace it for the bootstrap
        g = np.asarray([f"{t}/{e}" for e in ep_arr[keep]])
        per_sample.append((res[0], res[1], res[2], res[3], res[4], g[res[5]]))
    if not per_sample:
        return out

    nb = np.concatenate([p[0] for p in per_sample])
    nf = np.concatenate([p[1] for p in per_sample])
    ys = np.concatenate([p[2] for p in per_sample])
    pb = np.concatenate([p[3] for p in per_sample])
    pf = np.concatenate([p[4] for p in per_sample])
    gs = np.concatenate([p[5] for p in per_sample])
    out.update(
        dirprobe_nll_base=float(nb.mean()), dirprobe_nll_full=float(nf.mean()),
        dirprobe_nll_gain=float(nb.mean() - nf.mean()),
        dirprobe_n_tasks=float(n_tasks), dirprobe_n_samples=float(len(nb)),
    )
    try:
        out["dirprobe_auc_base"] = float(roc_auc_score(ys, pb))
        out["dirprobe_auc_full"] = float(roc_auc_score(ys, pf))
    except ValueError:
        pass

    # cluster bootstrap over episodes on the PAIRED per-sample difference
    uniq_g = np.unique(gs)
    idx_by_g = {g: np.flatnonzero(gs == g) for g in uniq_g}
    diff = nb - nf
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(len(uniq_g), len(uniq_g), replace=True)
        sel = np.concatenate([idx_by_g[uniq_g[k]] for k in pick])
        boots[b] = diff[sel].mean()
    out["dirprobe_gain_ci_lo"] = float(np.percentile(boots, 2.5))
    out["dirprobe_gain_ci_hi"] = float(np.percentile(boots, 97.5))

    # permutation null: shuffle the episode->verb assignment inside each task
    if n_perm > 0:
        null = []
        for _ in range(n_perm):
            gains = []
            for t in np.unique(task_arr):
                m = task_arr == t
                uv = [v for v in np.unique(verb_arr[m]) if v != "unknown"]
                if len(uv) != 2:
                    continue
                keep = m & np.isin(verb_arr, uv)
                y = (verb_arr[keep] == uv[0]).astype(int)
                e = ep_arr[keep]
                ue = np.unique(e)
                if y.sum() < 8 or (len(y) - y.sum()) < 8 or len(ue) < 3:
                    continue
                lab = {x: int(v) for x, v in zip(ue, rng.permutation(
                    [y[e == x][0] for x in ue]))}
                y_perm = np.asarray([lab[x] for x in e])
                if len(np.unique(y_perm)) < 2:
                    continue
                r = fit_task(z_static[keep].astype(np.float64),
                             z_trans[keep].astype(np.float64), y, e, y_override=y_perm)
                if r is not None:
                    gains.append(r[0].mean() - r[1].mean())
            if gains:
                null.append(float(np.mean(gains)))
        if null:
            null = np.asarray(null)
            out["dirprobe_gain_perm_p"] = float((null >= out["dirprobe_nll_gain"]).mean())
            out["dirprobe_null_mean"] = float(null.mean())
            out["dirprobe_null_std"] = float(null.std())
            # THE interpretable number. The raw gain is biased NEGATIVE: the full
            # probe carries 32 extra real (noisy) features and pays their
            # overfitting cost, while the baseline's zero block is regularized away.
            # So the no-information reference is the permutation null, not 0.
            out["dirprobe_gain_vs_null"] = float(out["dirprobe_nll_gain"] - null.mean())
    return out


def _collect_opposite(
    model: LatentActionModel,
    cfg: dict,
    device: torch.device,
    n_workers: int,
    batch_size: int,
    per_verb: int = 40,
    min_per_verb: int = 12,
    seed: int = 42,
    split: str = "val",
    min_eps_per_verb: int = 1,
) -> dict | None:
    """Task-stratified collection for the opposite-verb probe.

    The uniform val sample gives ~13 clips per task at n_samples=1500, so almost no
    task clears the >=8-per-verb filter — the first run of the 'fair' metric scored
    opp_cls_n_tasks=1 (a single task; its shuffled control sat at 0.39, i.e. the
    number was statistically void). Here we sample up to `per_verb` clips for EACH
    resolved verb of every reversible task (exactly two known verbs, both with
    >= `min_per_verb` clips) directly from the val-split dataset index, and encode
    both the transition latent and the static-pair latent (for the appearance
    control)."""
    dcfg = cfg["data"]
    ds = EgoDexDataset(
        data_root=dcfg["data_root"],
        img_h=dcfg.get("img_h", 240),
        img_w=dcfg.get("img_w", 320),
        downsample_factors=(1,),
        split=split,
        val_ratio=dcfg.get("val_ratio", 0.1),
        seed=dcfg.get("seed", 42),
        load_verbs=True,
        load_actions=True,
        action_dim=18,
        # honour the frozen eval manifest: `only_manifest=True` restricts collection
        # to episodes the encoder never trained on (required for confirmatory probes).
        exclude_manifest=dcfg.get("exclude_manifest", None),
        only_manifest=dcfg.get("only_manifest", False),
    )
    by_task: Dict[str, Dict[str, List[int]]] = {}
    for i, rec in enumerate(ds._index):
        task_id, ep_id = rec[2], rec[3]
        v = ds._verb_map.get(ep_id)
        v = _norm_verb(v) if v is not None else "unknown"
        if v == "unknown":
            continue
        by_task.setdefault(task_id, {}).setdefault(v, []).append(i)
    rng = np.random.RandomState(seed)
    keep: List[int] = []
    ep_of = {i: rec[3] for i, rec in enumerate(ds._index)}
    for task_id, verbs in sorted(by_task.items()):
        if len(verbs) != 2:                    # reversible tasks only
            continue
        if min(len(ix) for ix in verbs.values()) < min_per_verb:
            continue
        # Episode-grouped probes need several episodes PER VERB, else holding out an
        # episode removes a whole class and the CV degenerates. On the val split the
        # minority verb has a median of ONE episode, which is why the grouped probe
        # must be run on a larger pool (see notes/dirprobe_split_issue.md).
        if min(len({ep_of[i] for i in ix}) for ix in verbs.values()) < min_eps_per_verb:
            continue
        for v in sorted(verbs):
            ix = verbs[v]
            # sample per EPISODE so one long episode cannot dominate a verb
            eps_here = sorted({ep_of[i] for i in ix})
            quota = max(1, per_verb // len(eps_here))
            for e in eps_here:
                sub_ix = [i for i in ix if ep_of[i] == e]
                sel = rng.choice(len(sub_ix), size=min(quota, len(sub_ix)), replace=False)
                keep.extend(sub_ix[s] for s in sel)
    if not keep:
        return None
    sub = torch.utils.data.Subset(ds, keep)
    loader = DataLoader(sub, batch_size=batch_size, shuffle=False,
                        num_workers=n_workers, collate_fn=_collate)
    mus, smus, tasks, verbs_out, eps_out = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            videos = batch["videos"].to(device)
            mu, _ = _encode_mu_var(model, videos)
            static_videos = videos.clone()
            static_videos[:, 1] = static_videos[:, 0]
            smu, _ = _encode_mu_var(model, static_videos)
            mus.append(mu.cpu().numpy())
            smus.append(smu.cpu().numpy())
            tasks.extend(batch["task_id"])
            verbs_out.extend(batch.get("verb_id", ["unknown"] * videos.shape[0]))
            eps_out.extend(batch.get("ep_id", ["?"] * videos.shape[0]))
    return {
        "mu": np.concatenate(mus, axis=0),
        "static_mu": np.concatenate(smus, axis=0),
        "task": tasks,
        "verb": verbs_out,
        "ep": eps_out,          # needed for episode-grouped splits / cluster bootstrap
    }


def _psnr_per_sample(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """PSNR for [B,H,W,C] tensors in [0,1]."""
    mse = ((pred - gt) ** 2).mean(dim=[1, 2, 3]).clamp_min(eps)
    return -10.0 * torch.log10(mse)


def _controllability_metrics(
    model: LatentActionModel,
    loader: DataLoader,
    device: torch.device,
    n_samples: int,
) -> Dict[str, float]:
    """Genie-style Δt PSNR controllability using the LAM's OWN decoder:
    Δt PSNR = PSNR(o_{t+1}, D(o_t, z_inferred)) − PSNR(o_{t+1}, D(o_t, z_random)),
    where z_random shuffles ONLY the action subspace across the batch (keeps z_e),
    isolating whether z_a controls the predicted next frame. Positive = z_a is
    genuinely actionable, not ignored."""
    action_part = model.action_part
    p_inf, p_rand = [], []
    seen = 0
    with torch.no_grad():
        for batch in loader:
            videos = batch["videos"].to(device)
            b = videos.shape[0]
            if b < 2:
                continue
            mu, _ = _encode_mu_var(model, videos)
            gt = videos[:, 1]
            rec_inf = _decode_from_mu(model, videos, mu)[:, 0]
            perm = torch.randperm(b, device=device)
            z_rand = mu.clone()
            z_rand[:, :action_part] = mu[perm, :action_part]
            rec_rand = _decode_from_mu(model, videos, z_rand)[:, 0]
            p_inf.append(_psnr_per_sample(rec_inf, gt).cpu().numpy())
            p_rand.append(_psnr_per_sample(rec_rand, gt).cpu().numpy())
            seen += b
            if seen >= n_samples:
                break
    if not p_inf:
        return {"ctrl_psnr_inferred": float("nan"),
                "ctrl_psnr_random_action": float("nan"),
                "ctrl_delta_psnr": float("nan")}
    pi = np.concatenate(p_inf)[:n_samples]
    pr = np.concatenate(p_rand)[:n_samples]
    return {
        "ctrl_psnr_inferred": float(pi.mean()),
        "ctrl_psnr_random_action": float(pr.mean()),
        "ctrl_delta_psnr": float(pi.mean() - pr.mean()),
    }


def _zero_action_decode_metrics(
    model: LatentActionModel,
    loader: DataLoader,
    device: torch.device,
    n_samples: int,
) -> Dict[str, float]:
    """Non-circular do(z_a=0) intervention via the LAM's own decoder (the analogue of
    CD-LAM's do(u=0) rollout test): zero the action subspace, keep z_e, decode.
    If z_a carries the motion, the zero-action decode should collapse toward the
    CURRENT frame — motion suppressed — instead of continuing the transition.

      zeroact_motion_suppression = PSNR(D(o_t, 0⊕z_e), o_t) − PSNR(D(o_t, z), o_t)
        > 0: zeroing z_a pulls the prediction back toward staying still.
      zeroact_still_margin = PSNR(D(o_t, 0⊕z_e), o_t) − PSNR(D(o_t, 0⊕z_e), o_{t+1})
        > 0: the zero-action decode is closer to the current frame than to the
        true next frame — the functional, decoder-side version of what the
        loss-shaped `static_response` diagnostic only measures in latent space.
    """
    action_part = model.action_part
    supp, still = [], []
    seen = 0
    with torch.no_grad():
        for batch in loader:
            videos = batch["videos"].to(device)
            mu, _ = _encode_mu_var(model, videos)
            z0 = mu.clone()
            z0[:, :action_part] = 0.0
            rec_inf = _decode_from_mu(model, videos, mu)[:, 0]
            rec_zero = _decode_from_mu(model, videos, z0)[:, 0]
            ot, ot1 = videos[:, 0], videos[:, 1]
            p_zero_ot = _psnr_per_sample(rec_zero, ot)
            p_zero_ot1 = _psnr_per_sample(rec_zero, ot1)
            p_inf_ot = _psnr_per_sample(rec_inf, ot)
            supp.append((p_zero_ot - p_inf_ot).cpu().numpy())
            still.append((p_zero_ot - p_zero_ot1).cpu().numpy())
            seen += videos.shape[0]
            if seen >= n_samples:
                break
    if not supp:
        return {"zeroact_motion_suppression": float("nan"),
                "zeroact_still_margin": float("nan")}
    s = np.concatenate(supp)[:n_samples]
    m = np.concatenate(still)[:n_samples]
    return {
        "zeroact_motion_suppression": float(s.mean()),
        "zeroact_still_margin": float(m.mean()),
    }


def _opposite_swap_metrics(
    model: LatentActionModel,
    loader: DataLoader,
    device: torch.device,
    n_pool: int = 768,
    decode_bs: int = 16,
    seed: int = 42,
) -> Dict[str, float]:
    """Decoder-side opposite-action transfer (mini version of CD-LAM's target-action
    transfer, PSNR readout): fix (o_t, z_e) and replace z_a with
      (a) a same-task / same-verb / different-episode donor,
      (b) a same-task / OPPOSITE-verb donor,
    then decode and score against the true o_{t+1}.

      swap_opp_delta_psnr = PSNR(a) − PSNR(b)
        > 0: the decoder produces measurably different futures for opposite z_a —
        direction is ACTIONABLE, not merely separable in latent geometry. Both
        donors come from other episodes, so scene mismatch cancels in the delta.
      swap_context_cost_psnr = PSNR(own z_a) − PSNR(a): cross-episode transfer cost
        of the action code (lower = more context-invariant z_a).

    This is the fair external adjudicator for reverse-family losses: it asks what
    the downstream consumer would experience, not what the latent geometry looks like.
    """
    out = {"swap_n": 0.0, "swap_psnr_own": float("nan"),
           "swap_psnr_same_verb": float("nan"), "swap_psnr_opp_verb": float("nan"),
           "swap_opp_delta_psnr": float("nan"), "swap_context_cost_psnr": float("nan")}
    videos_all, tasks, eps, verbs = [], [], [], []
    seen = 0
    for batch in loader:
        b = batch["videos"].shape[0]
        videos_all.append(batch["videos"])
        tasks.extend(batch["task_id"])
        eps.extend(batch["ep_id"])
        verbs.extend(batch.get("verb_id", ["unknown"] * b))
        seen += b
        if seen >= n_pool:
            break
    if not videos_all:
        return out
    videos = torch.cat(videos_all, dim=0)[:n_pool]
    task_arr = np.asarray(tasks[:len(videos)])
    ep_arr = np.asarray(eps[:len(videos)])
    verb_arr = np.asarray([_norm_verb(v) for v in verbs[:len(videos)]])

    mus = []
    with torch.no_grad():
        for i in range(0, len(videos), decode_bs):
            vb = videos[i:i + decode_bs].to(device)
            mu, _ = _encode_mu_var(model, vb)
            mus.append(mu.cpu())
    mu = torch.cat(mus, dim=0)

    rng = np.random.RandomState(seed)
    known = verb_arr != "unknown"
    triples: List[Tuple[int, int, int]] = []
    for i in range(len(videos)):
        if not known[i]:
            continue
        same_task = task_arr == task_arr[i]
        pos_pool = np.where(same_task & known & (verb_arr == verb_arr[i]) & (ep_arr != ep_arr[i]))[0]
        # In EgoDex a reversible task's two resolved verbs ARE the opposite pair, so
        # same task + different resolved verb = opposite (matches _reversible_metrics).
        opp_pool = np.where(same_task & known & (verb_arr != verb_arr[i]))[0]
        if len(pos_pool) and len(opp_pool):
            triples.append((i, int(rng.choice(pos_pool)), int(rng.choice(opp_pool))))
    if not triples:
        return out

    action_part = model.action_part
    own_p, pos_p, opp_p = [], [], []
    with torch.no_grad():
        for s in range(0, len(triples), decode_bs):
            chunk = triples[s:s + decode_bs]
            idx = [c[0] for c in chunk]
            vb = videos[idx].to(device)
            gt = vb[:, 1]
            z_own = mu[idx].to(device)
            z_pos = z_own.clone()
            z_pos[:, :action_part] = mu[[c[1] for c in chunk], :action_part].to(device)
            z_opp = z_own.clone()
            z_opp[:, :action_part] = mu[[c[2] for c in chunk], :action_part].to(device)
            own_p.append(_psnr_per_sample(_decode_from_mu(model, vb, z_own)[:, 0], gt).cpu().numpy())
            pos_p.append(_psnr_per_sample(_decode_from_mu(model, vb, z_pos)[:, 0], gt).cpu().numpy())
            opp_p.append(_psnr_per_sample(_decode_from_mu(model, vb, z_opp)[:, 0], gt).cpu().numpy())
    own = np.concatenate(own_p)
    pos = np.concatenate(pos_p)
    opp = np.concatenate(opp_p)
    return {
        "swap_n": float(len(own)),
        "swap_psnr_own": float(own.mean()),
        "swap_psnr_same_verb": float(pos.mean()),
        "swap_psnr_opp_verb": float(opp.mean()),
        "swap_opp_delta_psnr": float((pos - opp).mean()),
        "swap_context_cost_psnr": float((own - pos).mean()),
    }


def _write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(v: object) -> str:
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        return f"{v:.4g}"
    return str(v)


def run_one(
    spec: RunSpec,
    out_dir: Path,
    n_samples: int,
    perturb_samples: int,
    batch_size: int,
    num_workers: int,
    perturb_scale: float,
    device: torch.device,
) -> dict:
    cfg = _load_cfg(_resolve(spec.config))
    if spec.init_from:
        model = _build_raw_model(cfg, _resolve(spec.init_from), device)
        source = str(_resolve(spec.init_from))
    else:
        model = load_model(str(_resolve(spec.checkpoint)), cfg, device)
        source = str(_resolve(spec.checkpoint))
    model.eval()
    loader = _make_loader(cfg, num_workers, batch_size)
    print(f"[{spec.name}] source={source}")
    print(f"[{spec.name}] val_size={len(loader.dataset):,} latent_dim={model.latent_dim} action_part={model.action_part}")

    data = _collect(model, loader, device, n_samples)
    mu = data["mu"]
    logvar = data["logvar"]
    action_part = model.action_part
    z_for_shortcut = mu[:, :action_part]
    action_labels = data["verb"] if any(v != "unknown" for v in data["verb"]) else data["task"]

    norm = np.linalg.norm(mu, axis=1)
    s_delta = float(np.sqrt(np.mean(norm ** 2)))
    static_norm = np.linalg.norm(data["static_mu"], axis=1)
    kl_dim = 0.5 * (mu ** 2 + np.exp(logvar) - logvar - 1.0)
    kl_mean_dim = kl_dim.mean(axis=0)
    std_dim = mu.std(axis=0)
    mean_abs_dim = np.abs(mu).mean(axis=0)
    eff_rank, top5 = _effective_rank(mu)
    active_units = int((std_dim > 0.05).sum())
    shortcut = _shortcut_metrics(z_for_shortcut, data["ep"], action_labels)
    task_shortcut = _task_shortcut_metrics(z_for_shortcut, data["task"], action_labels)
    scene_knn = _scene_knn_metrics(z_for_shortcut, data["ep"], data["task"])
    reversible = _reversible_metrics(z_for_shortcut, data["task"], data["ep"], data["verb"])
    full_r2, action_drop, single_r2 = _ridge_action_scores(mu, data["action"])
    task_eta = _eta2_per_dim(mu, data["task"])
    ep_eta = _eta2_per_dim(mu, data["ep"], min_count=5)
    verb_eta = _eta2_per_dim(mu, data["verb"])

    camera_loader = _make_loader(cfg, num_workers, batch_size)
    camera = _camera_shift_metrics(model, camera_loader, device, n_samples)
    reverse_loader = _make_loader(cfg, num_workers, batch_size)
    reverse = _temporal_reverse_metrics(model, reverse_loader, device, n_samples)

    # Recreate a fresh loader for perturbation because the first loader was consumed.
    perturb_loader = _make_loader(cfg, num_workers, batch_size)
    dec_eff = _decoder_effects(
        model, perturb_loader, device, std_dim, perturb_samples, perturb_scale
    )

    # ── Tier-A standard (non-circular, external-target) metrics ──
    # 1-3 reuse the already-collected frozen latents + true action labels (no GPU).
    mlp_probe = _mlp_action_probe(mu, data["action"])
    # Routing triple: R2(z_a) / R2(z_e) / R2(full). R2(z_e) is action leaked into the
    # env subspace; R2(full) − R2(z_a) is action info z_a failed to absorb.
    mlp_probe_za = _suffixed(
        _mlp_action_probe(mu[:, :action_part], data["action"], with_shuffle=False), "_za")
    mlp_probe_ze = _suffixed(
        _mlp_action_probe(mu[:, action_part:], data["action"], with_shuffle=False), "_ze")
    ridge_split = {
        "action_r2_za_linear": _ridge_r2(mu[:, :action_part], data["action"]),
        "action_r2_ze_linear": _ridge_r2(mu[:, action_part:], data["action"]),
        "action_r2_za_taskheld": _ridge_r2_taskheld(
            mu[:, :action_part], data["action"], data["task"]),
        **_action_r2_groups(mu[:, :action_part], data["action"]),
    }
    ridge_split["action_r2_taskheld_gap"] = (
        ridge_split["action_r2_za_linear"] - ridge_split["action_r2_za_taskheld"])
    label_eff = _label_efficiency(mu, data["action"])
    action_knn = _action_knn(z_for_shortcut, data["verb"], data["ep"])
    # kNN controls: ceiling in the TRUE action space (is verb-kNN even achievable from
    # perfect action info?) and in the probe-predicted action space (is the info there
    # but variance-misallocated?).
    gt_knn = _action_knn(data["action"], data["verb"], data["ep"],
                         prefix="gtact_knn", standardize=True)
    probe_pred = _ridge_cross_pred(z_for_shortcut, data["action"])
    probe_knn = _action_knn(probe_pred, data["verb"], data["ep"],
                            prefix="probeact_knn", standardize=True)
    # Opposite-verb probe on a TASK-STRATIFIED sample (the uniform sample yields
    # opp_cls_n_tasks≈1 at n=1500 — statistically void). Falls back to the uniform
    # sample if the stratified collection is empty.
    opp_data = _collect_opposite(model, cfg, device, num_workers, batch_size)
    if opp_data is None:
        opp_data = {"mu": mu, "static_mu": data["static_mu"],
                    "task": data["task"], "verb": data["verb"], "ep": data["ep"]}
    opp_cls = _opposite_verb_classification(
        opp_data["mu"][:, :action_part], opp_data["task"], opp_data["verb"])
    # Initial-state appearance control: the same probe on the static-pair latent
    # E(o_t, o_t) — see the confound warning in _opposite_verb_classification.
    opp_cls_static = _suffixed(_opposite_verb_classification(
        opp_data["static_mu"][:, :action_part], opp_data["task"], opp_data["verb"]), "_static")
    # HEADLINE direction metric (conditional on the appearance latent) — the
    # difference-of-accuracies version above is retained only as a leakage diagnostic.
    dirprobe = _conditional_direction_probe(
        opp_data["mu"][:, :action_part], opp_data["static_mu"][:, :action_part],
        opp_data["task"], opp_data["verb"], opp_data.get("ep", ["?"] * len(opp_data["mu"])))
    # 4 needs decoder passes; cap samples like the perturbation pass to stay cheap.
    ctrl_loader = _make_loader(cfg, num_workers, batch_size)
    controllability = _controllability_metrics(
        model, ctrl_loader, device, min(n_samples, 512)
    )
    zeroact = _zero_action_decode_metrics(
        model, _make_loader(cfg, num_workers, batch_size), device, min(n_samples, 512)
    )
    swap = _opposite_swap_metrics(
        model, _make_loader(cfg, num_workers, batch_size), device
    )
    # Collapse guard: a degenerate latent is trivially "invariant"; require the
    # frozen probe to clear an action-decode floor before any invariance is trusted.
    collapse_flag = bool(
        np.isfinite(mlp_probe["mlp_action_r2"]) and mlp_probe["mlp_action_r2"] < 0.1
    )

    per_dim_rows = []
    for j in range(model.latent_dim):
        subspace = "action" if j < action_part else "env"
        row = {
            "run": spec.name,
            "dim": j,
            "subspace": subspace,
            "std": float(std_dim[j]),
            "mean_abs": float(mean_abs_dim[j]),
            "kl_mean": float(kl_mean_dim[j]),
            "active_score": float(std_dim[j]),
            "action_r2_drop": float(action_drop[j]),
            "single_dim_action_r2": float(single_r2[j]),
            "decoder_total_effect": float(dec_eff[j]),
            "task_eta2": float(task_eta[j]),
            "ep_eta2": float(ep_eta[j]),
            "verb_eta2": float(verb_eta[j]),
            "leakage_score": float(max(task_eta[j], ep_eta[j])),
        }
        per_dim_rows.append(row)
    _dim_labels(per_dim_rows)

    per_dim_fields = [
        "run", "dim", "subspace", "std", "mean_abs", "kl_mean", "active_score",
        "action_r2_drop", "single_dim_action_r2", "decoder_total_effect",
        "task_eta2", "ep_eta2", "verb_eta2", "leakage_score", "label",
    ]
    _write_csv(out_dir / f"per_dim_{spec.name}.csv", per_dim_rows, per_dim_fields)

    summary = {
        "run": spec.name,
        "source": source,
        "latent_dim": model.latent_dim,
        "action_part": action_part,
        "n_samples": len(mu),
        "action_r2_full_latent": full_r2,
        "latent_norm_median": float(np.median(norm)),
        "latent_norm_rms": s_delta,
        "static_response_median": float(np.median(static_norm)),
        "static_response_rel_median": float(np.median(static_norm / max(s_delta, 1e-12))),
        "effective_rank": eff_rank,
        "top5_eig_frac": top5,
        "active_units_std_gt_0.05": active_units,
        "kl_mean_total": float(kl_dim.sum(axis=1).mean()),
        # ── Tier-A standard headline metrics (external target, frozen encoder) ──
        **mlp_probe,
        **mlp_probe_za,
        **mlp_probe_ze,
        **ridge_split,
        **label_eff,
        **action_knn,
        **gt_knn,
        **probe_knn,
        **opp_cls,
        **opp_cls_static,
        **dirprobe,
        # DEPRECATED as a direction metric (ceiling effect: static sits at .96-.99).
        # Kept under both names: `static_shortcut_gap` is what it actually measures.
        "static_shortcut_gap": float(
            opp_cls["opp_cls_bal_acc"] - opp_cls_static["opp_cls_bal_acc_static"]
        ),
        "opp_cls_direction_gain": float(
            opp_cls["opp_cls_bal_acc"] - opp_cls_static["opp_cls_bal_acc_static"]
        ),
        **controllability,
        **zeroact,
        **swap,
        "collapse_flag": collapse_flag,
        # ── diagnostics (loss-shaped — explain WHY, not for ranking) ──
        **shortcut,
        **task_shortcut,
        **scene_knn,
        **reversible,
        **camera,
        **reverse,
    }
    return summary


def _md_table(summaries: List[dict], keys: List[str]) -> List[str]:
    lines = [
        "| " + " | ".join(keys) + " |",
        "| " + " | ".join(["---"] * len(keys)) + " |",
    ]
    for row in summaries:
        lines.append("| " + " | ".join(_fmt(row.get(k, "")) for k in keys) + " |")
    return lines


def _write_markdown(path: Path, summaries: List[dict]) -> None:
    # HEADLINE: external-target, frozen-encoder metrics — these adjudicate quality.
    headline_probe = [
        "run", "mlp_action_r2", "mlp_action_r2_za", "mlp_action_r2_ze",
        "mlp_action_r2_shuffled", "mlp_action_l1", "mlp_action_max_l1",
        "action_r2_full_latent", "action_r2_za_linear", "action_r2_ze_linear",
        "action_r2_za_taskheld", "action_r2_taskheld_gap",
        "action_r2_trans", "action_r2_rot", "action_r2_left", "action_r2_right",
        "labeleff_r2_10pct", "labeleff_auc", "collapse_flag",
    ]
    headline_knn = [
        "run", "action_knn_verb_top1", "action_knn_null_top1_mean",
        "action_knn_null_top1_std", "action_knn_top1_z",
        "gtact_knn_verb_top1", "probeact_knn_verb_top1",
        "action_knn_verb_maj5", "action_knn_null_maj5_mean",
        "action_knn_chance_majority",
    ]
    headline_dir = [
        "run", "dirprobe_nll_base", "dirprobe_nll_full", "dirprobe_nll_gain",
        "dirprobe_gain_ci_lo", "dirprobe_gain_ci_hi", "dirprobe_gain_perm_p",
        "dirprobe_auc_base", "dirprobe_auc_full", "dirprobe_n_tasks", "dirprobe_n_samples",
    ]
    headline_interv = [
        "run", "opp_cls_bal_acc", "opp_cls_bal_acc_static", "static_shortcut_gap",
        "opp_cls_bal_acc_shuffled", "opp_cls_n_tasks",
        "ctrl_delta_psnr", "zeroact_motion_suppression", "zeroact_still_margin",
        "swap_opp_delta_psnr", "swap_context_cost_psnr", "swap_n",
    ]
    # DIAGNOSTICS: loss-shaped / geometry metrics — explain WHY, never for ranking.
    diagnostic = [
        "run", "static_response_rel_median",
        "camera_shift_h_median", "camera_shift_v_median",
        "scene_knn_ep_top1", "scene_knn_ep_null",
        "scene_knn_task_top1", "scene_knn_task_null",
        "episode_shortcut_leakage", "task_shortcut_leakage",
        "rev_hard_margin", "rev_nearest_pos_beats_opp",
        "reverse_za_cos_median", "reverse_ze_cos_median",
        "effective_rank", "top5_eig_frac", "active_units_std_gt_0.05",
        "kl_mean_total",
    ]
    lines = [
        "# LAM Benchmark",
        "",
        "## Headline — standard, non-circular (frozen encoder, external target)",
        "These adjudicate LAM quality. Every metric scores against the true 18-D action",
        "or a decoded rollout, with the encoder frozen — none is an image of a training loss.",
        "NOTE: a loss that consumes a signal loses headline rights on that signal's mirror",
        "(e.g. adding an 18-D regression head demotes `mlp_action_r2` to a diagnostic).",
        "",
        "### Probes — action decoding & subspace routing",
        "",
        *_md_table(summaries, headline_probe),
        "",
        "- `mlp_action_r2`: frozen full z_mu -> true 18-D action via a small MLP (villa-X);",
        "  `_za`/`_ze` are the SUBSPACE probes. The routing triple reads: `_ze` = action leaked",
        "  into the env subspace; full − `_za` = action info z_a failed to absorb.",
        "  `mlp_action_r2_shuffled` is a label-permutation control and MUST be ~0.",
        "- `action_r2_*_linear` / `action_r2_full_latent`: linear-probe counterparts.",
        "- `action_r2_za_taskheld`: linear probe with HELD-OUT TASKS (GroupKFold) — external",
        "  invariance measure; `_taskheld_gap` (shuffled − heldout) = context dependence of z_a.",
        "- `action_r2_trans/rot/left/right`: per-target-group R2 — tracks the",
        "  rotation-decodes-worse failure mode the fine-grained-control claim depends on.",
        "- `labeleff_*`: RidgeCV probe R2 at 10% labels + area under the log-fraction curve",
        "  (LAPO/Genie). Values from the old fixed-alpha probe were probe variance; re-based.",
        "",
        "### Retrieval — cross-episode verb kNN with a PROPER null",
        "",
        *_md_table(summaries, headline_knn),
        "",
        "- `action_knn_verb_top1`: cross-episode kNN verb agreement on raw z_a (zero trained",
        "  readout). `action_knn_null_top1_mean/std` is an episode-level label-permutation null",
        "  (the correct one for retrieval agreement); read `action_knn_top1_z` — |z| < 2 is",
        "  chance-level. `action_knn_chance_majority` (the old 'chance') is a majority-class",
        "  CLASSIFIER baseline, kept only for reference — do not compare retrieval against it.",
        "- `gtact_knn_verb_top1`: the same kNN on the TRUE 18-D action (standardized) — the",
        "  ceiling. If this is low, verb-kNN is a category error for a continuous action code.",
        "- `probeact_knn_verb_top1`: kNN in the out-of-fold ridge-predicted action space. High",
        "  here + low on raw z_a = info present but variance-misallocated (metric problem);",
        "  low here too = action info genuinely missing at linear readout.",
        "",
        "### Direction — CONDITIONAL probe (headline)",
        "",
        *_md_table(summaries, headline_dir),
        "",
        "- Both probes share one feature layout and differ only in information:",
        "  baseline `[Z_static, 0]` vs full `[Z_static, Z_trans - Z_static]`, GroupKFold",
        "  over EPISODE (verb labels are episode-level, so a random split would leak).",
        "- `dirprobe_nll_gain` = NLL_base - NLL_full, in nats: >0 means the transition",
        "  latent carries information about the opposite verb BEYOND appearance. Read it",
        "  with `dirprobe_gain_ci_lo` (cluster bootstrap over episodes) — the claim is",
        "  real only if the CI excludes 0 — and `dirprobe_gain_perm_p` (episode->verb",
        "  permutation within task).",
        "- NLL, not accuracy: the appearance probe already sits at .96-.99, where",
        "  accuracy saturates and cannot resolve added information.",
        "- This REPLACES `opp_cls_direction_gain` as the direction claim. That metric is",
        "  a difference of two independent probe accuracies with a .006-.040 headroom;",
        "  it is retained below as `static_shortcut_gap`, a leakage diagnostic only.",
        "",
        "### Interventions & opposite actions — decoder-side, external",
        "",
        *_md_table(summaries, headline_interv),
        "",
        "- `opp_cls_bal_acc`: within-task frozen logistic probe classifying the two reversible",
        "  verbs (chance 0.5), on a TASK-STRATIFIED sample (up to 40 clips per verb per",
        "  reversible task; the uniform sample yields n_tasks≈1 = statistically void).",
        "  `_static` is the SAME probe on E(o_t,o_t) latents — the initial-state appearance",
        "  confound; only `opp_cls_direction_gain` (= bal_acc − static) certifies direction",
        "  encoding. `_shuffled` MUST sit at ~0.5.",
        "- `ctrl_delta_psnr`: Genie Δt PSNR via the LAM's own decoder (z_a shuffled across batch).",
        "- `zeroact_motion_suppression` / `zeroact_still_margin`: do(z_a=0) decode (CD-LAM do(u=0)",
        "  analogue). >0 = zeroing z_a pulls the prediction toward staying still — the external",
        "  version of the loss-shaped `static_response`.",
        "- `swap_opp_delta_psnr`: same-task same-verb vs OPPOSITE-verb z_a swap decode, PSNR vs the",
        "  true next frame (scene mismatch cancels in the delta). >0 = opposite actions decode to",
        "  measurably different futures — the fair external adjudicator for reverse-family losses.",
        "",
        "## Diagnostics — loss-shaped (explain WHY; NOT for ranking)",
        "These largely mirror training losses (static↔L_zero, rev↔L_reverse, shortcut↔hardneg), so a",
        "model improving on them may just be optimizing its own objective. Read as failure-mode triage.",
        "",
        *_md_table(summaries, diagnostic),
        "",
        "Diagnostic notes:",
        "- Static response is normalized by the RMS norm of ordinary transition latents.",
        "- `scene_knn_*_top1`: scene/identity retrieval confound — fraction of cosine",
        "  nearest neighbours in z_a sharing the query's episode/task (self excluded).",
        "  Compare against `*_null` (random-retrieval expectation), not zero. High = z_a",
        "  retrieves 'same scene' over 'same action'. NOT CD-LAM's id_ratio: that is",
        "  their zero-transition response = our `static_response_rel_median`.",
        "- Camera-shift metrics are relative latent changes under a zero-filled image translation of both frames.",
        "- `*_shortcut_leakage > 0` means same-scene/different-action pairs are closer than different-scene/same-action.",
        "- `rev_hard_margin > 0` means same-task opposite verbs are farther than same-verb cross-episode positives.",
        "- Lower `reverse_za_cos_median` = more direction-sensitive z_a; higher `reverse_ze_cos_median` = more time-invariant z_e.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="results/lam_benchmark")
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--perturb_samples", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--perturb_scale", type=float, default=2.0)
    parser.add_argument("--device", default="auto",
                        help="Device for the benchmark, e.g. auto, cpu, cuda, cuda:0, cuda:1.")
    parser.add_argument("--runs", nargs="*", default=DEFAULT_RUN_NAMES,
                        help="Run names, e.g. raw_lam kl_ft_l40_full_5k contrastive_v3_5k ours_a_zero_5k")
    args = parser.parse_args()

    run_map = {r.name: r for r in RUN_SPECS}
    specs = [run_map[name] for name in args.runs]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        # AdaWorld's vendored positional encoding calls Tensor.cuda() directly.
        # Make CPU smoke tests possible; GPU runs keep the original behavior.
        torch.Tensor.cuda = lambda self, *a, **kw: self  # type: ignore[method-assign]
    device = _resolve_device(args.device)
    print(f"[device] {device}")

    def _flush(summaries: List[dict]) -> None:
        fields: List[str] = []
        for s in summaries:
            for k in s.keys():
                if k not in fields:
                    fields.append(k)
        _write_csv(out_dir / "summary.csv", summaries, fields)
        _write_markdown(out_dir / "summary.md", summaries)

    summaries = []
    for spec in specs:
        if spec.checkpoint is not None and not _resolve(spec.checkpoint).exists():
            print(f"[skip] {spec.name}: checkpoint not found at {spec.checkpoint}")
            continue
        summaries.append(
            run_one(
                spec=spec,
                out_dir=out_dir,
                n_samples=args.n_samples,
                perturb_samples=args.perturb_samples,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                perturb_scale=args.perturb_scale,
                device=device,
            )
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        # Flush after EVERY run so a later failure cannot lose completed rows.
        _flush(summaries)
        print(f"[progress] {len(summaries)} run(s) -> {out_dir / 'summary.md'}", flush=True)

    if not summaries:
        print("[done] no runs executed (all checkpoints missing?)")
        return
    print(f"[done] wrote {out_dir / 'summary.md'} and per-dim CSVs")


if __name__ == "__main__":
    main()
