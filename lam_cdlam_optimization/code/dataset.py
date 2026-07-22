"""
EgoDex dataset loader for the LAM disentanglement experiments.

Vendored from code/lam/data/egodex_dataset.py and EXTENDED to optionally load
real action labels aligned to each frame pair. The original file is NOT modified.

Real action (when ``load_actions=True``): for a frame pair (t, t+skip) we read the
sibling HDF5 ``transforms/leftHand`` and ``transforms/rightHand`` SE(3) poses
([T,4,4]) and build, per hand:

    T_rel = inv(T_t) @ T_{t+skip}
    [ delta_translation(3) , 6D_rotation(6) ]            # 9 dims/hand

Both hands -> 18-dim action. The action is computed from the SAME (t, skip) that
produced the image pair (after the clamp/fallback logic), and is z-scored with
dataset-level mean/std cached to disk so train/eval share identical normalization.

Sample (with load_actions): {"videos":[2,H,W,C], "task_id":str, "ep_id":str,
                             "phase_pos": float, "action": float32[18]}
"""

import random
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, Sampler

try:
    import decord
    decord.bridge.set_bridge("native")   # we go straight to numpy
    _HAVE_DECORD = True
except ImportError:
    _HAVE_DECORD = False


# ── Action geometry helpers ──────────────────────────────────────────────────

def _rot_to_6d(R: np.ndarray) -> np.ndarray:
    """3x3 rotation -> 6D continuous representation (Zhou et al. 2019).

    Returns the first two columns of R flattened: [r11,r21,r31, r12,r22,r32].
    """
    return R[:, :2].T.reshape(-1).astype(np.float32)


def _pair_action(left: np.ndarray, right: np.ndarray, t: int, skip: int) -> np.ndarray:
    """Raw (un-normalized) 18-dim action for one frame pair.

    left/right: [T,4,4] SE(3) poses of leftHand/rightHand.
    """
    out = []
    for T in (left, right):
        T_t = T[t]
        T_n = T[t + skip]
        T_rel = np.linalg.inv(T_t) @ T_n            # motion in hand's own frame at t
        dp = T_rel[:3, 3].astype(np.float32)        # delta translation (3)
        r6 = _rot_to_6d(T_rel[:3, :3])              # 6D rotation (6)
        out.append(dp)
        out.append(r6)
    return np.concatenate(out, axis=0)              # [18]


class EgoDexDataset(Dataset):
    """Consecutive frame pairs from EgoDex, optionally with 18-dim hand actions."""

    ACTION_DIM = 18

    def __init__(
        self,
        data_root: str,
        img_h: int = 240,
        img_w: int = 320,
        downsample_factors: Tuple[int, ...] = (1, 2),
        split: str = "train",
        val_ratio: float = 0.1,
        seed: int = 42,
        use_fg_mask: bool = False,
        fg_mask_type: str = "raft",
        contact_radius: int = 5,           # NEW: derives the sam3_hoc contact band
        hoc_cache_size: int = 64,          # NEW: per-worker mmap LRU for sam3_hoc_frames
        use_caption: bool = False,
        use_verb_emb: bool = False,
        load_verbs: bool = False,
        load_actions: bool = False,        # NEW: load 18-dim hand action labels
        action_dim: int = 18,              # NEW: must stay 18 for current spec
        min_frames: int = 10,
        exclude_manifest: str = None,      # NEW: frozen eval episodes the LAM must not see
        only_manifest: bool = False,       # NEW: invert — keep ONLY the manifest episodes
    ) -> None:
        self.data_root = Path(data_root)
        self.img_h = img_h
        self.img_w = img_w
        self.downsample_factors = downsample_factors
        self.use_fg_mask = use_fg_mask
        self.fg_mask_type = fg_mask_type
        self.contact_radius = contact_radius
        self._hoc_cache_size = hoc_cache_size
        self._hoc_cache: OrderedDict = OrderedDict()  # video path -> mmap or None
        self.min_frames = min_frames
        self.use_caption = use_caption
        self.use_verb_emb = use_verb_emb
        self.load_verbs = load_verbs
        self.load_actions = load_actions
        self.action_dim = action_dim
        self._verb_map: dict = {}
        self._pose_cache: dict = {}        # hdf5 path str -> (left[T,4,4], right[T,4,4])

        if load_actions and action_dim != self.ACTION_DIM:
            raise ValueError(
                f"action_dim must be {self.ACTION_DIM} (both hands dp3+6d). got {action_dim}"
            )

        # Frozen evaluation manifest (data/eval_manifest.json). Episodes listed there
        # are dropped from training so probe results can be confirmatory rather than
        # exploratory; `only_manifest=True` selects exactly that held-out pool for
        # evaluation. See code/make_eval_manifest.py — the file must never be
        # regenerated once a run has trained against it.
        self._excluded_eps: set = set()
        if exclude_manifest:
            import json
            with open(exclude_manifest) as f:
                self._excluded_eps = set(json.load(f)["episodes"])
        self._only_manifest = bool(only_manifest)
        if only_manifest and not self._excluded_eps:
            raise ValueError("only_manifest=True requires exclude_manifest to be given")

        all_videos: List[Path] = sorted(self.data_root.glob("*/*.mp4"))
        if not all_videos:
            raise FileNotFoundError(
                f"No .mp4 files found under {data_root}.\n"
                f"Expected layout: {{task_name}}/{{idx}}.mp4"
            )

        from collections import defaultdict
        task_to_videos = defaultdict(list)
        for vp in all_videos:
            task_to_videos[vp.parent.name].append(vp)

        rng = random.Random(seed)
        selected: List[Path] = []
        for task, vids in task_to_videos.items():
            shuffled = vids[:]
            rng.shuffle(shuffled)
            n_val = max(1, int(len(shuffled) * val_ratio))
            if split == "val":
                selected.extend(shuffled[:n_val])
            else:
                selected.extend(shuffled[n_val:])

        if load_verbs:
            import h5py
            for vp in selected:
                hdf5 = vp.with_suffix(".hdf5")
                task_id = vp.parent.name
                ep_id   = f"{task_id}/{vp.stem}"
                verb = None
                if hdf5.exists():
                    try:
                        with h5py.File(hdf5, "r") as hf:
                            verbs = [str(x) for x in hf.attrs.get("llm_verbs", [])]
                            w = hf.attrs.get("which_llm_description", None)
                            # Resolve the SINGLE performed verb. Most EgoDex episodes are
                            # reversible (llm_verbs=['insert','remove']); which_llm_description
                            # (1/2) says which one the video actually shows. Scene-independent:
                            # the same verb recurs across many tasks/environments.
                            if verbs:
                                if w is not None:
                                    wi = int(w)
                                    verb = verbs[wi - 1] if 1 <= wi <= len(verbs) else (
                                        verbs[0] if len(verbs) == 1 else None)
                                elif len(verbs) == 1:
                                    verb = verbs[0]
                    except Exception:
                        pass
                self._verb_map[ep_id] = verb

        # Build flat index: (video_path, frame_t, task_id, ep_id, hdf5_path)
        max_skip = max(self.downsample_factors)
        self._index: List[Tuple[Path, int, str, str, Path, int]] = []
        for vp in selected:
            hdf5 = vp.with_suffix(".hdf5")
            if load_actions and not hdf5.exists():
                continue                                   # need labels -> skip
            n = self._frame_count(vp)
            if n < max(self.min_frames, max_skip + 1):
                continue
            task_id = vp.parent.name
            ep_id   = f"{task_id}/{vp.stem}"
            if self._excluded_eps:
                in_manifest = ep_id in self._excluded_eps
                if in_manifest != self._only_manifest:
                    continue
            for t in range(n - max_skip):
                self._index.append((vp, t, task_id, ep_id, hdf5, n))

        if len(self._index) == 0:
            raise RuntimeError(
                f"Dataset is empty after indexing. "
                f"Found {len(selected)} videos but none usable (≥{max_skip+1} frames"
                f"{' and an HDF5 file' if load_actions else ''})."
            )

        # Action normalization stats (shared across train/val via on-disk cache)
        self.action_mean = None
        self.action_std = None
        if load_actions:
            self._init_action_stats(seed)

    # ── Action stats ──────────────────────────────────────────────────────────

    def _load_poses(self, hdf5_path: Path):
        """Return (left[T,4,4], right[T,4,4]) float32, cached per worker."""
        key = str(hdf5_path)
        cached = self._pose_cache.get(key)
        if cached is not None:
            return cached
        import h5py
        with h5py.File(hdf5_path, "r") as hf:
            left  = hf["transforms"]["leftHand"][:].astype(np.float32)
            right = hf["transforms"]["rightHand"][:].astype(np.float32)
        self._pose_cache[key] = (left, right)
        return left, right

    def _init_action_stats(self, seed: int):
        """Compute (or load) per-dim mean/std over a deterministic sample of
        actions drawn from ALL videos under data_root, so train/eval match."""
        cache = self.data_root / f"action_stats_{self.ACTION_DIM}d.npz"
        if cache.exists():
            d = np.load(cache)
            self.action_mean = torch.from_numpy(d["mean"].astype(np.float32))
            self.action_std  = torch.from_numpy(d["std"].astype(np.float32))
            return

        rng = random.Random(seed + 9999)
        all_videos = sorted(self.data_root.glob("*/*.mp4"))
        rng.shuffle(all_videos)
        max_skip = max(self.downsample_factors)
        acc = []
        for vp in all_videos:
            if len(acc) >= 4000:
                break
            hdf5 = vp.with_suffix(".hdf5")
            if not hdf5.exists():
                continue
            try:
                left, right = self._load_poses(hdf5)
            except Exception:
                continue
            T = min(len(left), len(right))
            if T < max_skip + 1:
                continue
            for _ in range(8):                              # a few pairs per video
                skip = rng.choice(self.downsample_factors)
                t = rng.randint(0, T - skip - 1)
                acc.append(_pair_action(left, right, t, skip))
        arr = np.stack(acc, axis=0)                         # [N,18]
        mean = arr.mean(0)
        std = arr.std(0)
        std[std < 1e-6] = 1.0
        try:
            np.savez(cache, mean=mean, std=std)
        except Exception:
            pass                                            # read-only fs: keep in-mem
        self.action_mean = torch.from_numpy(mean.astype(np.float32))
        self.action_std  = torch.from_numpy(std.astype(np.float32))

    def _compute_action(self, hdf5_path: Path, t: int, skip: int) -> Tensor:
        left, right = self._load_poses(hdf5_path)
        raw = _pair_action(left, right, t, skip)            # [18]
        a = torch.from_numpy(raw)
        a = (a - self.action_mean) / self.action_std
        return a.float()

    # ── Internal helpers (verbatim from original) ──────────────────────────────

    @staticmethod
    def _frame_count(video_path: Path) -> int:
        if _HAVE_DECORD:
            try:
                vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
                return len(vr)
            except Exception:
                pass
        cap = cv2.VideoCapture(str(video_path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return n

    def _load_pair(self, video_path: Path, t: int, skip: int) -> Tensor:
        if _HAVE_DECORD:
            try:
                vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
                fr = vr.get_batch([t, t + skip]).asnumpy()   # one open, batched fetch
                f0, f1 = fr[0], fr[1]
            except Exception:
                f0 = self._load_frame_opencv(video_path, t)
                f1 = self._load_frame_opencv(video_path, t + skip)
        else:
            f0 = self._load_frame_opencv(video_path, t)
            f1 = self._load_frame_opencv(video_path, t + skip)

        f0 = self._center_crop_resize(f0).astype(np.float32) / 255.0
        f1 = self._center_crop_resize(f1).astype(np.float32) / 255.0
        return torch.from_numpy(np.stack([f0, f1], axis=0))

    def _load_frame_opencv(self, video_path: Path, idx: int) -> np.ndarray:
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Failed to read frame {idx} from {video_path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _center_crop_resize(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        target_ratio = self.img_w / self.img_h
        frame_ratio = w / h
        if frame_ratio > target_ratio:
            new_w = int(h * target_ratio)
            x0 = (w - new_w) // 2
            frame = frame[:, x0:x0 + new_w]
        elif frame_ratio < target_ratio:
            new_h = int(w / target_ratio)
            y0 = (h - new_h) // 2
            frame = frame[y0:y0 + new_h, :]
        return cv2.resize(frame, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)

    def _open_hoc_frames(self, video_path: Path):
        """mmap the per-frame SAM3 hand/object file for one episode (LRU cached).

        Cached per dataset instance, i.e. per DataLoader worker. Misses are
        cached as None so a dataset without the new layout does not stat once
        per __getitem__.
        """
        key = str(video_path)
        if key in self._hoc_cache:
            self._hoc_cache.move_to_end(key)
            return self._hoc_cache[key]

        path = video_path.parent / "sam3_hoc_frames" / f"{video_path.stem}.npy"
        try:
            arr = np.load(path, mmap_mode="r") if path.exists() else None
        except Exception:
            arr = None

        self._hoc_cache[key] = arr
        if len(self._hoc_cache) > self._hoc_cache_size:
            self._hoc_cache.popitem(last=False)
        return arr

    def _contact_band(self, hand: np.ndarray, obj: np.ndarray) -> np.ndarray:
        r = self.contact_radius
        if r <= 0:
            return (hand & obj).astype(bool)
        k = np.ones((2 * r + 1, 2 * r + 1), dtype=np.uint8)
        return (cv2.dilate(hand, k).astype(bool) & cv2.dilate(obj, k).astype(bool))

    def _load_hoc_from_frames(self, video_path: Path, t: int, skip: int):
        """Rebuild the [3,H,W] hand/object/contact tensor from per-frame masks.

        Mirrors precompute_sam3_hoc_masks.process_video: both endpoints of the
        pair are OR'd (the target is t+skip but the action evidence is the
        transition), then the contact band is the intersection of the dilated
        hand and object masks.
        """
        arr = self._open_hoc_frames(video_path)
        if arr is None:
            return None
        n = arr.shape[1]
        if n == 0:
            return None
        t0, t1 = min(t, n - 1), min(t + skip, n - 1)

        # unpackbits yields uint8 0/1, which cv2.dilate takes directly.
        hand = np.unpackbits(arr[0, t0], axis=-1) | np.unpackbits(arr[0, t1], axis=-1)
        obj = np.unpackbits(arr[1, t0], axis=-1) | np.unpackbits(arr[1, t1], axis=-1)
        contact = self._contact_band(hand, obj)
        return torch.from_numpy(np.stack([hand.astype(bool), obj.astype(bool), contact]))

    def _load_fg_mask(self, video_path: Path, t: int, skip: int):
        if self.fg_mask_type == "sam3":
            subdir = "sam3_masks"
        elif self.fg_mask_type in {"sam3_hoc", "sam3_hoc_masks"}:
            # Preferred layout: one per-frame file per episode. Falls through to
            # the legacy per-pair .pt tree when an episode has not been redone.
            mask = self._load_hoc_from_frames(video_path, t, skip)
            if mask is not None:
                return mask
            subdir = "sam3_hoc_masks"
        else:
            subdir = "flow_masks"
        mask_dir  = video_path.parent / subdir / video_path.stem
        mask_path = mask_dir / f"{t:06d}_skip{skip}.pt"
        if mask_path.exists():
            return torch.load(mask_path, map_location="cpu", weights_only=True)
        mask_s1 = mask_dir / f"{t:06d}_skip1.pt"
        if mask_s1.exists():
            return torch.load(mask_s1, map_location="cpu", weights_only=True)
        return None

    # ── Dataset API ─────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        video_path, t, task_id, ep_id, hdf5_path, n = self._index[idx]
        skip = random.choice(self.downsample_factors)

        if t + skip >= n:
            skip = 1
        if t + skip >= n:
            t = max(0, n - 2)
            skip = 1

        try:
            videos = self._load_pair(video_path, t, skip)
        except Exception:
            fallback = random.randint(0, len(self) - 1)
            return self.__getitem__(fallback)

        phase_pos = float(t + 0.5 * skip) / float(max(1, n - 1))
        item = {
            "videos": videos,
            "task_id": task_id,
            "ep_id": ep_id,
            "phase_pos": torch.tensor(phase_pos, dtype=torch.float32),
            "pair_skip": torch.tensor(skip, dtype=torch.long),
        }
        if self.load_verbs:
            verb = self._verb_map.get(ep_id)
            item["verb_id"] = verb if verb is not None else "unknown"

        if self.use_caption:
            cap_path = self.data_root / task_id / "captions" / f"{video_path.stem}.pt"
            if cap_path.exists():
                item["caption_emb"] = torch.load(cap_path, map_location="cpu", weights_only=True)

        if self.use_verb_emb:
            verb_path = self.data_root / task_id / "verb_embs" / f"{video_path.stem}.pt"
            if verb_path.exists():
                item["verb_emb"] = torch.load(verb_path, map_location="cpu", weights_only=True)

        if self.use_fg_mask:
            mask = self._load_fg_mask(video_path, t, skip)
            n_ch = 3 if self.fg_mask_type in {"sam3_hoc", "sam3_hoc_masks"} else 0
            if mask is None:
                # Missing mask -> all-background. lam_loss floors weights at 1x, so
                # the sample degrades to uniform reconstruction weighting, and every
                # sample carries the key (the plain stacking collate needs that).
                shape = (n_ch, self.img_h, self.img_w) if n_ch else (self.img_h, self.img_w)
                mask = torch.zeros(shape, dtype=torch.bool)
            elif mask.ndim == 2:
                if mask.shape != (self.img_h, self.img_w):
                    mask = torch.nn.functional.interpolate(
                        mask.float().unsqueeze(0).unsqueeze(0),
                        size=(self.img_h, self.img_w),
                        mode="nearest",
                    ).squeeze(0).squeeze(0).bool()
            else:  # [C,H,W] multi-channel (sam3_hoc: hand/object/contact)
                if mask.shape[-2:] != (self.img_h, self.img_w):
                    mask = torch.nn.functional.interpolate(
                        mask.float().unsqueeze(0),
                        size=(self.img_h, self.img_w),
                        mode="nearest",
                    ).squeeze(0).bool()
            item["fg_mask"] = mask

        # ── Action label: computed with the FINAL (t, skip) to match the images ──
        if self.load_actions:
            try:
                item["action"] = self._compute_action(hdf5_path, t, skip)
            except Exception:
                fallback = random.randint(0, len(self) - 1)
                return self.__getitem__(fallback)

        return item


# ── PK class-balanced batch sampler (for supervised contrastive) ──────────────

class PKBatchSampler(Sampler):
    """Yield batches of P label-classes x K samples each, so supervised-contrastive
    has guaranteed same-class positives (random sampling over ~100 classes gives ~0).

    ``index_labels[i]`` is the class label of dataset index ``i`` (e.g. task_id or the
    resolved verb). Labels in ``exclude`` (or None) are dropped — used to skip
    'unknown'/unresolved samples. Each batch has ``P*K`` indices; each anchor sees
    ``K-1`` positives and ``(P-1)*K`` negatives.
    """

    def __init__(self, index_labels, classes_per_batch: int, samples_per_class: int,
                 num_batches: int, seed: int = 42, exclude=None):
        self.P = classes_per_batch
        self.K = samples_per_class
        self.num_batches = num_batches
        self.rng = random.Random(seed)
        exclude = set(exclude or [])
        from collections import defaultdict
        groups = defaultdict(list)
        for i, lab in enumerate(index_labels):
            if lab is None or lab in exclude:
                continue
            groups[lab].append(i)
        self.groups = dict(groups)
        self.tasks = list(self.groups.keys())              # class names (kept attr name)
        if len(self.tasks) < self.P:
            raise ValueError(f"PKBatchSampler: only {len(self.tasks)} classes, need >= P={self.P}")

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            chosen = self.rng.sample(self.tasks, self.P)
            batch = []
            for t in chosen:
                pool = self.groups[t]
                # sample K (with replacement if the task is smaller than K)
                if len(pool) >= self.K:
                    batch.extend(self.rng.sample(pool, self.K))
                else:
                    batch.extend(self.rng.choices(pool, k=self.K))
            yield batch


class DirPairDataset(Dataset):
    """Pre-mined anti-parallel frame-pair COUPLES for L_dir.

    Each item returns TWO transition clips from the same episode whose 3-D motions
    are anti-parallel (mined offline by code/mine_dir_pairs.py — see
    notes/dir_pair_audit.md for why the Δt window is mandatory).

    `kind` selects the arm:
      0 = anti-parallel pairs (the real L_dir signal)
      1 = matched control: same count, same Δt/magnitude gates, cos IGNORED.
          If L_dir and this control behave identically, the direction claim is
          empty and the loss is only a generic repulsion term.

    Returns videos of shape [2, 2, H, W, C]: (member, frame, ...) matching the
    layout EgoDexDataset produces for a single clip.
    """

    def __init__(self, npz_path, data_root, img_h: int = 240, img_w: int = 320,
                 kind: int = 0):
        d = np.load(npz_path, allow_pickle=True)
        m = d["kind"] == kind
        if not m.any():
            raise ValueError(f"{npz_path} has no pairs with kind={kind}")
        self.video = d["video"][m]
        self.t_i = d["t_i"][m]
        self.t_j = d["t_j"][m]
        self.skip = d["skip"][m]
        self.cos = d["cos"][m]
        self.n_frames = d["n_frames"][m]
        self.kind = kind
        self.data_root = Path(data_root)
        # reuse EgoDexDataset's frame decoding / cropping without its indexing
        self._io = EgoDexDataset.__new__(EgoDexDataset)
        self._io.img_h, self._io.img_w = img_h, img_w

    def __len__(self) -> int:
        return len(self.t_i)

    def __getitem__(self, idx: int) -> dict:
        vp = self.data_root / str(self.video[idx])
        skip = int(self.skip[idx])
        n = int(self.n_frames[idx])
        clips = []
        for t in (int(self.t_i[idx]), int(self.t_j[idx])):
            t = min(max(0, t), max(0, n - skip - 1))
            try:
                clips.append(self._io._load_pair(vp, t, skip))
            except Exception:
                return self.__getitem__(random.randint(0, len(self) - 1))
        return {
            "videos": torch.stack(clips, dim=0),           # [2, 2, H, W, C]
            "pair_cos": torch.tensor(float(self.cos[idx]), dtype=torch.float32),
        }


def dir_pair_collate(samples):
    return {
        "videos": torch.stack([s["videos"] for s in samples]),      # [B,2,2,H,W,C]
        "pair_cos": torch.stack([s["pair_cos"] for s in samples]),  # [B]
    }
