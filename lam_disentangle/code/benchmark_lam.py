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
]
DEFAULT_RUN_NAMES = ["raw_lam", "kl_ft_l40_full_5k", "contrastive_v3_5k"]


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (Path.cwd() / path)


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


def _make_loader(cfg: dict, n_workers: int, batch_size: int) -> DataLoader:
    dcfg = cfg["data"]
    ds = EgoDexDataset(
        data_root=dcfg["data_root"],
        img_h=dcfg.get("img_h", 240),
        img_w=dcfg.get("img_w", 320),
        downsample_factors=(1,),
        split="val",
        val_ratio=dcfg.get("val_ratio", 0.1),
        seed=dcfg.get("seed", 42),
        load_verbs=True,
        load_actions=True,
        action_dim=18,
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
    opp = _opposite_verb_map()
    verb_arr = np.asarray([_norm_verb(v) for v in verb])
    task_arr = np.asarray(task)
    ep_arr = np.asarray(ep)
    known = (verb_arr != "unknown") & np.asarray([v in opp for v in verb_arr])
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
    opposite = np.zeros((n, n), dtype=bool)
    for i, v in enumerate(verb_arr):
        opposite[i] = verb_arr == opp.get(v, "")

    pos = same_verb & (~same_ep) & off
    same_task_opp = same_task & opposite & off
    any_opp = opposite & off

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
        **shortcut,
        **task_shortcut,
        **reversible,
        **camera,
        **reverse,
    }
    return summary


def _write_markdown(path: Path, summaries: List[dict]) -> None:
    keys = [
        "run", "action_r2_full_latent", "static_response_rel_median",
        "camera_shift_h_median", "camera_shift_v_median",
        "episode_shortcut_leakage", "task_shortcut_leakage",
        "action_separation_ratio",
        "rev_hard_margin", "rev_nearest_pos_beats_opp",
        "reverse_za_cos_median", "reverse_ze_cos_median",
        "effective_rank", "top5_eig_frac", "active_units_std_gt_0.05",
        "kl_mean_total",
    ]
    lines = [
        "# LAM-Side Benchmark",
        "",
        "| " + " | ".join(keys) + " |",
        "| " + " | ".join(["---"] * len(keys)) + " |",
    ]
    for row in summaries:
        lines.append("| " + " | ".join(_fmt(row.get(k, "")) for k in keys) + " |")
    lines += [
        "",
        "Notes:",
        "- Static response is normalized by the RMS norm of ordinary transition latents.",
        "- Camera-shift metrics are relative latent changes after applying the same zero-filled image translation to both frames.",
        "- `episode_shortcut_leakage > 0` means same-episode/different-action pairs are closer than different-episode/same-action pairs.",
        "- `task_shortcut_leakage > 0` means same-task/different-action pairs are closer than different-task/same-action pairs.",
        "- `rev_hard_margin > 0` means same-task opposite verbs are farther than same-verb cross-episode positives.",
        "- Lower `reverse_za_cos_median` means the action subspace is more direction-sensitive; higher `reverse_ze_cos_median` means the env subspace is more time-reversal invariant.",
        "- Per-dim labels are heuristic relative rankings for triage, not claims of fixed physical semantics.",
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

    summaries = []
    for spec in specs:
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

    fields = list(summaries[0].keys())
    _write_csv(out_dir / "summary.csv", summaries, fields)
    _write_markdown(out_dir / "summary.md", summaries)
    print(f"[done] wrote {out_dir / 'summary.md'} and per-dim CSVs")


if __name__ == "__main__":
    main()
