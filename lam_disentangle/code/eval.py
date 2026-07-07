"""
Evaluation for the LAM disentanglement experiments.

Modes:
  reconstruction   MSE / PSNR on val (world-model-quality proxy, Eval #3 stand-in)
  visualize        t-SNE of z_mu by task_id / ep_id / verb (+ action-subspace panel)
  probe            classification probes z_mu -> task_id / ep_id (action purity)
  action_probe     frozen-encoder Ridge regression to 18-dim real action, reported
                   for full z_mu / action-subspace / env-subspace, plus the 2x2
                   subspace x {task, ep} classification matrix  (Eval #1 + #2)
  perturb          decode while perturbing single action vs env dims (Eval #4)

Usage:
  python lam_disentangle/code/eval.py \
      --checkpoint <out_dir>/step_0020000 \
      --config lam_disentangle/code/config/exp3_indep.yaml \
      --mode action_probe --n_samples 4000 --out_path results/exp3_probe.txt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from model import LatentActionModel       # noqa: E402
from dataset import EgoDexDataset         # noqa: E402


def _collate(samples):
    batch = {}
    for key in samples[0]:
        if isinstance(samples[0][key], torch.Tensor):
            batch[key] = torch.stack([s[key] for s in samples])
        else:
            batch[key] = [s[key] for s in samples]
    return batch


def load_model(checkpoint_dir, cfg, device):
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
    ckpt = Path(checkpoint_dir)
    model_bin = ckpt / "pytorch_model.bin"
    if not model_bin.exists():
        cands = list(ckpt.glob("*.safetensors")) + list(ckpt.glob("*.bin"))
        if not cands:
            raise FileNotFoundError(f"No model weights in {checkpoint_dir}")
        model_bin = cands[0]
    if str(model_bin).endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(str(model_bin), device="cpu")
    else:
        state_dict = torch.load(model_bin, map_location="cpu")
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model


def _gather(model, loader, device, n_samples, want_action=False):
    """Collect z_mu (+ labels, + actions) over the val loader."""
    mu, task, ep, verb, act = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            videos = batch["videos"].to(device)
            out = model({"videos": videos})
            mu.extend(out["z_mu"].cpu().numpy())
            task.extend(batch.get("task_id", ["?"] * videos.shape[0]))
            ep.extend(batch.get("ep_id", ["?"] * videos.shape[0]))
            verb.extend(batch.get("verb_id", ["unknown"] * videos.shape[0]))
            if want_action and "action" in batch:
                act.extend(batch["action"].cpu().numpy())
            if len(mu) >= n_samples:
                break
    n = min(n_samples, len(mu))
    res = {"mu": np.array(mu[:n]), "task": task[:n], "ep": ep[:n], "verb": verb[:n]}
    if want_action and act:
        res["action"] = np.array(act[:n])
    return res


# ── reconstruction ──────────────────────────────────────────────────────────

def eval_reconstruction(model, loader, device, n_samples=1000):
    total_mse, total_n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            videos = batch["videos"].to(device)
            out = model({"videos": videos})
            mse = ((videos[:, 1:] - out["recon"]) ** 2).mean(dim=[1, 2, 3, 4])
            total_mse += mse.sum().item()
            total_n += videos.shape[0]
            if total_n >= n_samples:
                break
    avg_mse = total_mse / total_n
    avg_psnr = 10 * np.log10(1.0 / (avg_mse + 1e-8))
    print(f"Eval on {total_n} samples:  MSE={avg_mse:.6f}  PSNR={avg_psnr:.2f} dB")
    return avg_mse, avg_psnr


# ── action_probe : regression + 2x2 classification matrix ─────────────────────

def eval_action_probe(model, loader, device, n_samples, action_part, out_path="", label=""):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, accuracy_score
    from collections import Counter

    data = _gather(model, loader, device, n_samples, want_action=True)
    mu = data["mu"]
    lines = [f"Action Probe — {label}", "=" * 56,
             f"n={len(mu)}  latent_dim={mu.shape[1]}  action_part={action_part}"]

    # ---- Regression: subspace -> 18-dim real action (frozen encoder) ----
    # Report PER-DIMENSION R^2 and MSE (sklearn default uniform_average over 18 dims
    # mixes translation & 6D-rotation, which have very different scales/predictability;
    # the per-dim view is what we actually report). 18-dim layout:
    #   leftHand  Δp [0:3]   leftHand  6D-rot [3:9]
    #   rightHand Δp [9:12]  rightHand 6D-rot [12:18]
    GROUPS = [("Lhand Δpos", range(0, 3)),  ("Lhand 6Drot", range(3, 9)),
              ("Rhand Δpos", range(9, 12)), ("Rhand 6Drot", range(12, 18))]
    if "action" in data:
        y = data["action"]
        slices = {
            "full z_mu  ": mu,
            "action-sub ": mu[:, :action_part],
            "env-sub    ": mu[:, action_part:],
        }
        Xtr_idx, Xte_idx = train_test_split(np.arange(len(mu)), test_size=0.2, random_state=42)
        lines.append("\n== Real-action regression (Ridge), PER-DIM R^2 (higher=more action info) ==")
        for name, X in slices.items():
            if X.shape[1] == 0:
                lines.append(f"\n[{name.strip()}]: (no dims)")
                continue
            sc = StandardScaler().fit(X[Xtr_idx])
            Xs = sc.transform(X)
            reg = Ridge(alpha=1.0).fit(Xs[Xtr_idx], y[Xtr_idx])
            pred = reg.predict(Xs[Xte_idx])
            r2_dim = r2_score(y[Xte_idx], pred, multioutput="raw_values")   # [18]
            mse_dim = ((y[Xte_idx] - pred) ** 2).mean(axis=0)                # [18]
            overall = r2_score(y[Xte_idx], pred)                            # uniform avg (compat)
            lines.append(f"\n[{name.strip()}]  overall(uniform) R2={overall:6.3f}")
            for gname, idxs in GROUPS:
                idxs = list(idxs)
                lines.append(f"  {gname:12s}: R2(mean)={r2_dim[idxs].mean():6.3f}  "
                             f"MSE(mean)={mse_dim[idxs].mean():.4f}  "
                             f"per-dim R2=[{', '.join(f'{r2_dim[i]:.3f}' for i in idxs)}]")
    else:
        lines.append("\n(No action labels found in batch — skipping regression.)")

    # ---- Classification 2x2: subspace x {task_id, ep_id} ----
    def clf_probe(X, labels, min_count=2):
        le = LabelEncoder()
        yv = le.fit_transform(labels)
        counts = Counter(yv)
        keep = np.array([i for i, v in enumerate(yv) if counts[v] >= min_count])
        if len(keep) < 20 or len(le.classes_) < 2:
            return None, None
        Xk, yk = X[keep], yv[keep]
        cnt = np.bincount(yk)
        strat = yk if cnt.min() >= 2 else None
        Xtr, Xte, ytr, yte = train_test_split(Xk, yk, test_size=0.2,
                                              random_state=42, stratify=strat)
        clf = LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1)
        clf.fit(Xtr, ytr)
        acc = accuracy_score(yte, clf.predict(Xte))
        return acc, 1.0 / len(le.classes_)

    lines.append("\n-- Classification matrix: subspace x target (acc / chance) --")
    subspaces = {"action-sub": mu[:, :action_part], "env-sub": mu[:, action_part:]}
    for sname, X in subspaces.items():
        if X.shape[1] == 0:
            lines.append(f"  {sname}: (no dims)")
            continue
        ta, tc = clf_probe(X, data["task"])
        ea, ec = clf_probe(X, data["ep"], min_count=5)
        va, vc = clf_probe(X, data.get("verb", ["unknown"] * len(X)))
        ts = f"task {ta*100:5.1f}% (chance {tc*100:.1f}%)" if ta is not None else "task n/a"
        es = f"ep {ea*100:5.1f}% (chance {ec*100:.2f}%)" if ea is not None else "ep n/a"
        vs = f"verb {va*100:5.1f}% (chance {vc*100:.1f}%)" if va is not None else "verb n/a"
        lines.append(f"  {sname:10s}: {ts}  |  {es}  |  {vs}")

    lines.append("\nExpected (good disentanglement): action-sub predicts action & task"
                 " well; env-sub predicts action poorly but ep/scene comparatively better.")
    text = "\n".join(lines)
    print(text)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(text)
        print(f"Saved -> {out_path}")


# ── cluster : quantify scene-invariance of the action subspace ────────────────

def eval_cluster(model, loader, device, n_samples, action_part, out_path="", label=""):
    """Quantitative version of the t-SNE picture.

    In the ACTION subspace z_a = z_mu[:, :action_part], measure mean cosine distance
    within three pools and report whether same-action latents stay close ACROSS
    episodes (scene-invariant action) vs only within the same episode (scene leak):

      d_ss = same action (task), same episode
      d_sd = same action (task), DIFFERENT episode
      d_dd = different action (task)

    Good disentanglement:  d_sd ≈ d_ss  ≪  d_dd.
      scene-invariance gap = d_sd - d_ss   (→0 means z_a ignores the episode/scene)
      separation ratio     = d_dd / d_sd   (↑ means actions are well separated)
    """
    from sklearn.metrics import silhouette_score

    data = _gather(model, loader, device, n_samples)
    mu = data["mu"]
    ep = np.asarray(data["ep"])
    za = mu[:, :action_part].astype(np.float64)
    za /= (np.linalg.norm(za, axis=1, keepdims=True) + 1e-8)   # unit norm -> cosine
    D = 1.0 - za @ za.T                                        # cosine distance [n,n]
    n = len(za)
    off     = ~np.eye(n, dtype=bool)
    same_ep = ep[:, None] == ep[None, :]

    lines = [f"Action-subspace clustering — {label}", "=" * 56,
             f"n={n}  action_part={action_part}  (cosine distance on z_a, unit-normed)"]

    # Report the scene-invariance geometry with BOTH action labels:
    #   task = original task (≈ scene in EgoDex; a task can MIX verbs, e.g. add/remove)
    #   verb = the scene-independent action verb — the faithful "same-action" label
    out = {}
    for aname in ("task", "verb"):
        if aname not in data:
            continue
        a = np.asarray(data[aname])
        if aname == "verb":                       # drop unresolved verbs from this view
            keep = a != "unknown"
            if keep.sum() < 20:
                continue
        else:
            keep = np.ones(n, dtype=bool)
        a_k = a[keep]
        Dk = D[np.ix_(keep, keep)]
        offk = ~np.eye(keep.sum(), dtype=bool)
        same_ak = a_k[:, None] == a_k[None, :]
        same_epk = same_ep[np.ix_(keep, keep)]
        pools = {
            "same-action same-episode": same_ak & same_epk & offk,
            "same-action diff-episode": same_ak & ~same_epk,
            "diff-action":              ~same_ak,
        }
        means = {k: (float(Dk[m].mean()) if m.any() else float("nan")) for k, m in pools.items()}
        d_ss, d_sd, d_dd = (means["same-action same-episode"],
                            means["same-action diff-episode"], means["diff-action"])
        gap = d_sd - d_ss
        sep = d_dd / d_sd if d_sd == d_sd and d_sd else float("nan")
        try:
            sil = float(silhouette_score(za[keep], a_k, metric="cosine")) if len(set(a_k)) > 1 else float("nan")
        except Exception:
            sil = float("nan")
        out[aname] = (gap, sep, sil)
        lines += [f"\n[action label = {aname}]  (n={int(keep.sum())})"]
        for k in pools:
            lines.append(f"  d[{k:26s}] = {means[k]:.4f}")
        lines += [
            f"  scene-invariance gap (d_sd - d_ss) = {gap:+.4f}   (→0 = z_a ignores scene within an action)",
            f"  separation ratio     (d_dd / d_sd) = {sep:.3f}    (↑ = actions better separated)",
            f"  silhouette                          = {sil:+.4f}   (↑ = tighter action clusters)"]
    text = "\n".join(lines)
    print(text)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(text)
        print(f"Saved -> {out_path}")
    return out


# ── visualize (t-SNE) ─────────────────────────────────────────────────────────

def eval_visualize(model, loader, device, n_samples, action_part, out_path, label=""):
    from sklearn.manifold import TSNE
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    data = _gather(model, loader, device, n_samples)
    mu, task, ep = data["mu"], data["task"], data["ep"]

    def to_int(labels):
        uniq = sorted(set(labels)); m = {v: i for i, v in enumerate(uniq)}
        return np.array([m[l] for l in labels]), uniq

    panels_data = [
        ("Full z_mu by task", TSNE(2, random_state=42, perplexity=40).fit_transform(mu), *to_int(task)),
        ("Full z_mu by ep",   None, *to_int(ep)),
        ("Action-sub by task",
         TSNE(2, random_state=42, perplexity=40).fit_transform(mu[:, :action_part]), *to_int(task)),
    ]
    # reuse first embedding for the ep panel
    panels_data[1] = ("Full z_mu by ep", panels_data[0][1], *to_int(ep))

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    for ax, (title, emb, color_int, uniq) in zip(axes, panels_data):
        n_colors = len(uniq)
        cmap = cm.get_cmap("tab20" if n_colors <= 20 else "hsv", n_colors)
        ax.scatter(emb[:, 0], emb[:, 1], c=color_int, cmap=cmap, s=3, alpha=0.6,
                   vmin=0, vmax=max(n_colors - 1, 1))
        ax.set_title(title, fontsize=11); ax.axis("off")
    fig.suptitle(f"LAM latent t-SNE{' — ' + label if label else ''}", fontsize=13)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved t-SNE -> {out_path}")


# ── perturb : decode while nudging single dims ────────────────────────────────

def eval_perturb(model, loader, device, action_part, out_path, label="", n_dims=4, mags=(-2, -1, 0, 1, 2)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Estimate per-dim std of z_mu over a batch for sensible perturbation scale.
    data = _gather(model, loader, device, 512)
    z_std = torch.tensor(data["mu"].std(0), device=device).clamp(min=1e-3)

    batch = next(iter(loader))
    videos = batch["videos"][:1].to(device)             # one sample
    with torch.no_grad():
        enc = model.encode(videos)
        z_mu = enc["z_mu"][:1].clone()                  # [1, latent_dim]
        patches = enc["patches"]

    latent_dim = z_mu.shape[1]
    env_dim = latent_dim - action_part
    action_dims = list(range(0, min(n_dims, action_part)))
    env_dims = list(range(action_part, action_part + min(n_dims, env_dim)))
    rows = [("A%d" % d, d) for d in action_dims] + [("E%d" % d, d) for d in env_dims]

    def decode(zvec):
        z_rep = zvec.reshape(1, 1, 1, latent_dim)
        vp = model.patch_up(patches[:1, :-1])
        ap = model.action_up(z_rep)
        rec = torch.sigmoid(model.decoder(vp + ap))
        H, W = videos.shape[2:4]
        from model import unpatchify
        return unpatchify(rec, model.patch_size, H, W)[0, 0].cpu().clamp(0, 1).numpy()

    fig, axes = plt.subplots(len(rows), len(mags), figsize=(2.2 * len(mags), 2.2 * len(rows)))
    if len(rows) == 1:
        axes = axes[None, :]
    with torch.no_grad():
        for r, (name, d) in enumerate(rows):
            for c, m in enumerate(mags):
                z = z_mu.clone()
                z[0, d] = z[0, d] + m * z_std[d]
                img = decode(z)
                ax = axes[r, c]
                ax.imshow(img); ax.axis("off")
                if c == 0:
                    ax.set_ylabel(name, rotation=0, labelpad=18, fontsize=10)
                if r == 0:
                    ax.set_title(f"{m:+d}σ", fontsize=10)
    fig.suptitle(f"Latent perturbation (rows: A=action dims, E=env dims){' — ' + label if label else ''}",
                 fontsize=12)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved perturbation grid -> {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None,
                        help="Accelerate checkpoint dir (trained run). Mutually exclusive with --init_from.")
    parser.add_argument("--init_from", default=None,
                        help="Raw pretrained LAM ckpt (.ckpt/.safetensors), e.g. LAM_400k — the 0-step anchor. "
                             "Loaded with load_pretrained_lam (strips prefixes, strict=False).")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", default="reconstruction",
                        choices=["reconstruction", "visualize", "action_probe", "perturb", "cluster"])
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--out_path", default="results/out.png")
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    dcfg = cfg["data"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.init_from:
        # 0-step anchor: build model from config arch, load raw pretrained weights.
        import sys as _sys
        _sys.path.insert(0, str(_HERE))
        from train import load_pretrained_lam
        mcfg = cfg["model"]
        model = LatentActionModel(
            in_dim=mcfg.get("image_channels", 3), model_dim=mcfg.get("lam_model_dim", 512),
            latent_dim=mcfg.get("lam_latent_dim", 32), patch_size=mcfg.get("lam_patch_size", 16),
            enc_blocks=mcfg.get("lam_enc_blocks", 8), dec_blocks=mcfg.get("lam_dec_blocks", 8),
            num_heads=mcfg.get("lam_num_heads", 8), dropout=0.0,
            env_dim=mcfg.get("env_dim", 0), action_dim=mcfg.get("action_dim", 0),
            action_head=mcfg.get("action_head", False),
        )
        class _A:
            def print(self, *a): print(*a)
        load_pretrained_lam(model, args.init_from, _A())
        model.to(device).eval()
        src = args.init_from
    else:
        if not args.checkpoint:
            raise SystemExit("Provide --checkpoint <dir> or --init_from <ckpt>")
        model = load_model(args.checkpoint, cfg, device)
        src = args.checkpoint
    action_part = model.action_part
    print(f"Loaded {src} | latent_dim={model.latent_dim} action_part={action_part}")

    need_action = args.mode == "action_probe"
    val_ds = EgoDexDataset(
        data_root=dcfg["data_root"],
        img_h=dcfg.get("img_h", 240), img_w=dcfg.get("img_w", 320),
        downsample_factors=(1,), split="val",
        val_ratio=dcfg.get("val_ratio", 0.1), seed=dcfg.get("seed", 42),
        load_verbs=True, load_actions=need_action, action_dim=18 if need_action else 18,
    )
    g = torch.Generator(); g.manual_seed(42)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=True, num_workers=4,
                            collate_fn=_collate, generator=g)
    print(f"Val size: {len(val_ds):,}")

    if args.mode == "reconstruction":
        eval_reconstruction(model, val_loader, device, n_samples=args.n_samples)
    elif args.mode == "visualize":
        eval_visualize(model, val_loader, device, args.n_samples, action_part,
                       Path(args.out_path).with_suffix(".png"), args.label)
    elif args.mode == "action_probe":
        eval_action_probe(model, val_loader, device, args.n_samples, action_part,
                          str(Path(args.out_path).with_suffix(".txt")), args.label)
    elif args.mode == "perturb":
        eval_perturb(model, val_loader, device, action_part,
                     Path(args.out_path).with_suffix(".png"), args.label)
    elif args.mode == "cluster":
        eval_cluster(model, val_loader, device, args.n_samples, action_part,
                     str(Path(args.out_path).with_suffix(".txt")), args.label)


if __name__ == "__main__":
    main()
