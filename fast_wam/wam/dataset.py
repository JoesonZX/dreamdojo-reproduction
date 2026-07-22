"""
EgoExo4D multi-view frame-pair dataset for the multi-view LAM.

Reads the *preprocessed* tree produced by ``prepare/`` (240x320 all-intra mp4,
one file per camera per take):

    <processed_root>/
        index.json
        <take_uid>/ego.mp4  exo0.mp4  exo1.mp4 ...  [hand_poses.npz]

Sampling modes (one per training arm):

    "single" : encode view fixed to the take's first training exo camera.
               No cross-view target.                       -> arm L0 (sv)
    "multi"  : encode view sampled uniformly from the take's training cameras.
               No cross-view target.                       -> arm L1 (mv-data)
    "cross"  : encode view sampled uniformly; cross view sampled uniformly from
               the remaining training cameras.             -> arm L2 (mv-cross)

``view_pairs`` stages the difficulty (see plan Step A/B):
    "exo_exo" : only static third-person cameras are used.
    "any"     : the moving ego (Aria) camera participates as well.

The camera marked ``heldout_cam`` in index.json is NEVER returned here; it is
reserved for eval_crossview.py.

Sample dict:
    videos           float32 (2, H, W, 3) in [0,1]      -- encode view
    cross_videos     float32 (2, H, W, 3)               -- only in "cross" mode
    view_ids         int64 scalar                        -- view slot of encode view
    cross_view_ids   int64 scalar                        -- view slot of cross view
    action           float32 (18,)                       -- only if load_actions
    take_uid         str
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

try:
    import decord
    decord.bridge.set_bridge("native")
    _HAVE_DECORD = True
except ImportError:  # pragma: no cover
    _HAVE_DECORD = False


# View-slot vocabulary. NOTE: "exo{k}" is a *slot* index within a take, not a
# stable physical camera across takes (Ego-Exo4D re-places the GoPros per take).
# That is fine for the view-prompt ablation -- the prompt can still learn
# ego-vs-exo and average exo statistics -- but do not over-interpret it.
MAX_VIEW_SLOTS = 8          # must be <= model.num_views in the view-prompt arm


def view_slot_id(cam_name: str) -> int:
    if cam_name == "ego":
        return 0
    slot = 1 + int(cam_name.replace("exo", ""))
    if slot >= MAX_VIEW_SLOTS:
        raise ValueError(
            f"camera {cam_name} maps to view slot {slot} >= MAX_VIEW_SLOTS="
            f"{MAX_VIEW_SLOTS}. Raise MAX_VIEW_SLOTS and model.num_views together."
        )
    return slot


def split_take_uids(uids: List[str], val_ratio: float, seed: int) -> Tuple[set, set]:
    """Deterministic take-level split. MUST be identical in train and eval, so
    both call this one function."""
    rng = random.Random(seed)
    uids = sorted(uids)
    rng.shuffle(uids)
    n_val = max(1, int(len(uids) * val_ratio))
    val = set(uids[:n_val])
    return set(uids) - val, val


# ── action geometry (wrist delta, same convention as the EgoDex 18-D action) ──

def _rot_to_6d(R: np.ndarray) -> np.ndarray:
    """3x3 rotation -> 6D continuous representation (Zhou et al. 2019)."""
    return R[:, :2].T.reshape(-1).astype(np.float32)


def pair_action(left: np.ndarray, right: np.ndarray, t: int, skip: int) -> np.ndarray:
    """Raw 18-dim action for one frame pair. left/right: (T,4,4) SE(3)."""
    out = []
    for T_all in (left, right):
        T_rel = np.linalg.inv(T_all[t]) @ T_all[t + skip]
        out.append(T_rel[:3, 3].astype(np.float32))     # delta translation (3)
        out.append(_rot_to_6d(T_rel[:3, :3]))           # 6D rotation (6)
    return np.concatenate(out, axis=0)                  # (18,)


class EgoExoMultiViewDataset(Dataset):
    """Time-synchronised multi-view frame pairs from preprocessed Ego-Exo4D."""

    ACTION_DIM = 18

    def __init__(
        self,
        processed_root: str,
        mode: str = "cross",                 # single | multi | cross
        view_pairs: str = "exo_exo",         # exo_exo | any
        split: str = "train",                # train | val
        val_ratio: float = 0.05,
        seed: int = 42,
        img_h: int = 240,
        img_w: int = 320,
        downsample_factors: Tuple[int, ...] = (1, 2),
        pairs_per_take: int = 64,            # virtual epoch length multiplier
        load_actions: bool = False,
        min_frames: int = 16,
    ) -> None:
        if mode not in ("single", "multi", "cross"):
            raise ValueError(f"unknown mode: {mode}")
        if view_pairs not in ("exo_exo", "any"):
            raise ValueError(f"unknown view_pairs: {view_pairs}")

        self.root = Path(processed_root)
        self.mode = mode
        self.view_pairs = view_pairs
        self.img_h, self.img_w = img_h, img_w
        self.downsample_factors = tuple(downsample_factors)
        self.load_actions = load_actions
        self.max_skip = max(self.downsample_factors)

        with open(self.root / "index.json") as f:
            index = json.load(f)

        takes = [t for t in index["takes"] if t["n_frames"] >= max(min_frames, self.max_skip + 2)]
        if load_actions:
            takes = [t for t in takes if t.get("has_hand")]

        # Deterministic take-level train/val split (never mix frames of a take).
        train_uids, val_uids = split_take_uids([t["uid"] for t in takes], val_ratio, seed)
        keep = val_uids if split == "val" else train_uids
        self.takes = [t for t in takes if t["uid"] in keep]
        if not self.takes:
            raise RuntimeError(f"no takes left for split={split} under {self.root}")

        # Resolve the usable camera list per take (heldout camera excluded).
        self.entries: List[dict] = []
        for t in self.takes:
            cams = [c for c in t["train_cams"] if c != t.get("heldout_cam")]
            if self.view_pairs == "exo_exo":
                cams = [c for c in cams if c != "ego"]
            need = 2 if self.mode == "cross" else 1
            if len(cams) < need:
                continue
            self.entries.append({
                "uid": t["uid"],
                "dir": self.root / t["uid"],
                "cams": sorted(cams),
                "n_frames": t["n_frames"],
                "has_hand": bool(t.get("has_hand")),
            })
        if not self.entries:
            raise RuntimeError(
                f"no take has >= {2 if mode=='cross' else 1} usable cameras "
                f"(view_pairs={view_pairs}). Check index.json."
            )

        self.pairs_per_take = pairs_per_take
        self._epoch_seed = seed
        self._pose_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._reader_cache: Dict[str, "decord.VideoReader"] = {}

        self.action_mean = None
        self.action_std = None
        if load_actions:
            self._init_action_stats(seed)

    def __len__(self) -> int:
        return len(self.entries) * self.pairs_per_take

    # ── action normalisation ─────────────────────────────────────────────────

    def _load_poses(self, take_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
        key = str(take_dir)
        cached = self._pose_cache.get(key)
        if cached is not None:
            return cached
        d = np.load(take_dir / "hand_poses.npz")
        pair = (d["left"].astype(np.float32), d["right"].astype(np.float32))
        self._pose_cache[key] = pair
        return pair

    def _init_action_stats(self, seed: int) -> None:
        cache = self.root / f"action_stats_{self.ACTION_DIM}d.npz"
        if cache.exists():
            d = np.load(cache)
            self.action_mean = torch.from_numpy(d["mean"].astype(np.float32))
            self.action_std = torch.from_numpy(d["std"].astype(np.float32))
            return
        rng = random.Random(seed + 9999)
        acc = []
        ents = [e for e in self.entries if e["has_hand"]]
        rng.shuffle(ents)
        for e in ents:
            if len(acc) >= 4000:
                break
            try:
                left, right = self._load_poses(e["dir"])
            except Exception:
                continue
            T = min(len(left), len(right))
            if T < self.max_skip + 2:
                continue
            for _ in range(8):
                skip = rng.choice(self.downsample_factors)
                t = rng.randint(0, T - skip - 1)
                acc.append(pair_action(left, right, t, skip))
        if not acc:
            raise RuntimeError("load_actions=True but no hand poses found")
        arr = np.stack(acc, 0)
        mean, std = arr.mean(0), arr.std(0)
        std[std < 1e-6] = 1.0
        try:
            np.savez(cache, mean=mean, std=std)
        except Exception:
            pass
        self.action_mean = torch.from_numpy(mean.astype(np.float32))
        self.action_std = torch.from_numpy(std.astype(np.float32))

    # ── frame IO ─────────────────────────────────────────────────────────────

    def _reader(self, path: Path):
        key = str(path)
        r = self._reader_cache.get(key)
        if r is None:
            r = decord.VideoReader(key, ctx=decord.cpu(0))
            if len(self._reader_cache) > 32:          # bound per-worker memory
                self._reader_cache.clear()
            self._reader_cache[key] = r
        return r

    def _load_frame_cv2(self, path: Path, idx: int) -> np.ndarray:
        cap = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"cannot read frame {idx} of {path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _load_pair(self, path: Path, t: int, skip: int) -> Tensor:
        f0 = f1 = None
        if _HAVE_DECORD:
            try:
                fr = self._reader(path).get_batch([t, t + skip]).asnumpy()
                f0, f1 = fr[0], fr[1]
            except Exception:
                f0 = f1 = None
        if f0 is None:
            f0 = self._load_frame_cv2(path, t)
            f1 = self._load_frame_cv2(path, t + skip)
        out = []
        for f in (f0, f1):
            if f.shape[0] != self.img_h or f.shape[1] != self.img_w:
                f = cv2.resize(f, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)
            out.append(f)
        arr = np.stack(out, 0).astype(np.float32) / 255.0     # (2,H,W,3)
        return torch.from_numpy(arr)

    # ── sampling ─────────────────────────────────────────────────────────────

    def __getitem__(self, idx: int) -> Dict:
        e = self.entries[idx % len(self.entries)]
        # Deterministic per-index RNG so runs are reproducible across workers.
        rng = random.Random((self._epoch_seed * 1_000_003 + idx) & 0x7FFFFFFF)

        skip = rng.choice(self.downsample_factors)
        n = e["n_frames"]
        t = rng.randint(0, max(0, n - skip - 2))

        cams = e["cams"]
        if self.mode == "single":
            enc_cam = cams[0]
            cross_cam = None
        elif self.mode == "multi":
            enc_cam = rng.choice(cams)
            cross_cam = None
        else:  # cross
            enc_cam = rng.choice(cams)
            others = [c for c in cams if c != enc_cam]
            cross_cam = rng.choice(others)

        sample: Dict = {
            "videos": self._load_pair(e["dir"] / f"{enc_cam}.mp4", t, skip),
            "view_ids": torch.tensor(view_slot_id(enc_cam), dtype=torch.long),
            "take_uid": e["uid"],
        }
        if cross_cam is not None:
            sample["cross_videos"] = self._load_pair(e["dir"] / f"{cross_cam}.mp4", t, skip)
            sample["cross_view_ids"] = torch.tensor(view_slot_id(cross_cam), dtype=torch.long)

        if self.load_actions:
            left, right = self._load_poses(e["dir"])
            T = min(len(left), len(right))
            tt = min(t, T - skip - 1)
            a = torch.from_numpy(pair_action(left, right, tt, skip))
            sample["action"] = ((a - self.action_mean) / self.action_std).float()
        return sample


def collate(samples: List[Dict]) -> Dict:
    """Stack tensors, keep take_uid as a plain list."""
    out: Dict = {}
    for k in samples[0]:
        if k == "take_uid":
            out[k] = [s[k] for s in samples]
        else:
            out[k] = torch.stack([s[k] for s in samples], dim=0)
    return out
