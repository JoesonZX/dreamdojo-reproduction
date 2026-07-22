"""光流 displacement / signed-direction scorer(prereg §5:覆盖全 clip 的光流位移指标)。

不依赖 SAM3/CoTracker——用 dense Farneback 光流的前景位移场,支持:
  following_error  : 生成序列 vs 真值序列 的前景 endpoint 位移误差(越小越跟随)
  direction_cosine : 生成位移方向 vs 真值/donor 位移方向 的 signed cosine(测符号方向)
canonical-FDCE@K(SAM3 前景点 + track)后续接入;当前用光流位移作等价 displacement 代理。

帧输入 [H,W,3] float∈[0,1]。为速度默认在低分辨率算光流。
"""
from __future__ import annotations

import cv2
import numpy as np


def _gray(f: np.ndarray, size=None) -> np.ndarray:
    x = (np.clip(f, 0, 1) * 255).astype(np.uint8)
    g = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    if size is not None:
        g = cv2.resize(g, size, interpolation=cv2.INTER_AREA)
    return g


def dense_flow(a: np.ndarray, b: np.ndarray, size=(160, 120)) -> np.ndarray:
    """Farneback flow a→b at reduced resolution. Returns [h,w,2] (dx,dy)."""
    ga, gb = _gray(a, size), _gray(b, size)
    return cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 3, 15, 3, 5, 1.2, 0)


def fg_mask(flow: np.ndarray, q: float = 0.8) -> np.ndarray:
    mag = np.linalg.norm(flow, axis=2)
    thr = np.quantile(mag, q)
    return (mag >= max(thr, 1e-3))


def displacement(a: np.ndarray, b: np.ndarray, size=(160, 120)):
    """Flow field + foreground mask + mean foreground displacement vector."""
    fl = dense_flow(a, b, size)
    fg = fg_mask(fl)
    mean_disp = fl[fg].mean(0) if fg.any() else np.zeros(2)
    return fl, fg, mean_disp


def following_error(pred_o0, pred_oK, true_o0, true_oK, size=(160, 120)) -> dict:
    """Compare a generated displacement to the TRUE displacement over the TRUE foreground.

    fg is defined from the reference (true) motion only (prereg §9: mask from reference).
    endpoint_err = mean ‖flow_pred − flow_true‖ over fg;
    dir_cos      = mean cos(flow_pred, flow_true) over fg (signed direction following).
    """
    fl_true = dense_flow(true_o0, true_oK, size)
    fl_pred = dense_flow(pred_o0, pred_oK, size)
    fg = fg_mask(fl_true, q=0.8)
    if not fg.any():
        return {"endpoint_err": float("nan"), "dir_cos": float("nan"),
                "true_mag": 0.0, "pred_mag": 0.0, "n_fg": 0}
    pt, pp = fl_true[fg], fl_pred[fg]
    endpoint = float(np.linalg.norm(pp - pt, axis=1).mean())
    cos = (pt * pp).sum(1) / (np.linalg.norm(pt, axis=1) * np.linalg.norm(pp, axis=1) + 1e-8)
    return {"endpoint_err": endpoint, "dir_cos": float(np.mean(cos)),
            "true_mag": float(np.linalg.norm(pt, axis=1).mean()),
            "pred_mag": float(np.linalg.norm(pp, axis=1).mean()), "n_fg": int(fg.sum())}


def batch_following(pred_o0, pred_oK, true_o0, true_oK, size=(160, 120)) -> dict:
    """Vectorized over a list/array of clips ([N,H,W,3] each). Returns mean scalars."""
    rows = [following_error(pred_o0[i], pred_oK[i], true_o0[i], true_oK[i], size)
            for i in range(len(pred_o0))]
    ep = np.array([r["endpoint_err"] for r in rows], dtype=np.float64)
    dc = np.array([r["dir_cos"] for r in rows], dtype=np.float64)
    tm = np.array([r["true_mag"] for r in rows], dtype=np.float64)
    ok = ~np.isnan(ep)
    return {"endpoint_err": float(ep[ok].mean()), "dir_cos": float(dc[ok].mean()),
            "true_mag": float(tm[ok].mean()), "n": int(ok.sum()),
            "per_clip_endpoint": ep.tolist(), "per_clip_dircos": dc.tolist()}
