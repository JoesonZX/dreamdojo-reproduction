"""Phase 2 route-C:训练独立 from-scratch conditional predictor + dev-set following gate(prereg §4/§6）。

**Eval-B 密封**——本脚本只在 consumer-train 池(全部 − Eval-A − 14 Eval-B task,
`data/consumer_exclude.json`)上训练,并在一个 held-out **dev** 子集(仍非 Eval-B)上跑
following gate。最终 Eval-B 解封是**单独**一步,不在此。

cond 来源:
  gt_action —— GT 18D action(**正对照**:已知强可控,必须把 target 与 shuffle/antiparallel 分开)
  zmu       —— 冻结 encoder 的 z_μ action 子空间(route-C 主对象,--arm 指定 encoder)

one-step teacher-forced 训练(o_t,c_t→ô_{t+1},残差),推理自回归 K 步。dev gate 用光流
following_error(prereg §5)比 target / shuffle / antiparallel / zero 四种 condition。

    CUDA_VISIBLE_DEVICES=<free> python -u code/train_consumer.py --cond gt_action --steps 3000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np                                                          # noqa: E402
import torch                                                                # noqa: E402
import torch.nn.functional as F                                             # noqa: E402

import _apath  # noqa: F401  # shared A (lam_cdlam_optimization) substrate on sys.path
from benchmark_lam import RUN_SPECS, _load_cfg, _resolve                    # noqa: E402
from dataset import EgoDexDataset                                           # noqa: E402
from eval import load_model                                                 # noqa: E402
from consumer_model import ConditionalUNet                                  # noqa: E402
from rollout_tier15 import sample_clips, load_clip, encode_seq             # noqa: E402
from flow_scorer import batch_following                                     # noqa: E402

CONSUMER_EXCLUDE = "/home/xuan/embodied-ai/lam_condition_utilization/data/consumer_exclude.json"
SEED_ARMS = {"42": "sb_A0", "1": "sb_A0_s1"}   # only used for zmu cond (encoder identity)


def cache_clips(ds, specs):
    frames, actions = [], []
    for c in specs:
        cl = load_clip(ds, c)
        frames.append(cl["frames"]); actions.append(cl["actions"])
    return torch.stack(frames), torch.stack(actions)               # [N,K+1,H,W,3], [N,K,18]


@torch.no_grad()
def dev_gate(model, frames_dev, cond_dev, device, horizons=(1, 4, 8), size=(160, 120)):
    """Roll out under target/shuffle/antiparallel/zero; flow-score following at each horizon.

    Returns {cond: {h: {endpoint_err, dir_cos, true_mag, n}}}. h=1 is drift-free single-step
    following (cleanest test of "does the model use the condition"); h=K is the pre-registered
    horizon. dir_cos is robust to the static-copy trap (a copy → ~0 motion → ill-defined cos).
    """
    N, K = cond_dev.shape[0], cond_dev.shape[1]
    hs = [h for h in horizons if h <= K]
    o0 = frames_dev[:, 0].permute(0, 3, 1, 2).to(device)           # [N,3,H,W]
    true_o0 = frames_dev[:, 0].numpy()
    perm = torch.roll(torch.arange(N), 1)                          # shuffle: shift clip assignment
    conds = {"target": cond_dev, "shuffle": cond_dev[perm],
             "antiparallel": -cond_dev, "zero": torch.zeros_like(cond_dev)}
    out = {}
    for name, c in conds.items():
        roll = model.rollout(o0, c.to(device)).permute(0, 1, 3, 4, 2).cpu().numpy()  # [N,K,H,W,3]
        out[name] = {}
        for h in hs:
            r = batch_following(true_o0, roll[:, h - 1], true_o0, frames_dev[:, h].numpy(), size)
            out[name][h] = {"endpoint_err": r["endpoint_err"], "dir_cos": r["dir_cos"], "n": r["n"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", choices=["gt_action", "zmu"], default="gt_action")
    ap.add_argument("--arm", default=None, help="encoder arm for zmu cond (e.g. sb_A0)")
    ap.add_argument("--seed_tag", default="42", help="42 or 1 (encoder seed for zmu)")
    ap.add_argument("--n_train_clips", type=int, default=500)
    ap.add_argument("--n_dev_clips", type=int, default=80)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--skip", type=int, default=2)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--motion_lambda", type=float, default=12.0,
                    help="motion-weighted L1: weight moving pixels to break the persistence shortcut")
    ap.add_argument("--train_seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.train_seed); np.random.seed(args.train_seed)
    spec_map = {s.name: s for s in RUN_SPECS}
    cfg = _load_cfg(_resolve(spec_map["sb_A0"].config))
    d = cfg["data"]
    ds = EgoDexDataset(data_root=d["data_root"], img_h=d.get("img_h", 240), img_w=d.get("img_w", 320),
                       downsample_factors=(1, 2), split="train", val_ratio=d.get("val_ratio", 0.1),
                       seed=d.get("seed", 42), load_verbs=True, load_actions=True, action_dim=18,
                       exclude_manifest=CONSUMER_EXCLUDE, only_manifest=False)

    specs = sample_clips(ds, args.n_train_clips + args.n_dev_clips, args.K, args.skip,
                         seed=100 + args.train_seed, prefer_reversible=True)
    tr_specs, dev_specs = specs[:args.n_train_clips], specs[args.n_train_clips:]
    print(f"train clips={len(tr_specs)}  dev clips={len(dev_specs)}  K={args.K} skip={args.skip} cond={args.cond}")
    fr_tr, act_tr = cache_clips(ds, tr_specs)
    fr_dev, act_dev = cache_clips(ds, dev_specs)
    print(f"cached train {tuple(fr_tr.shape)} dev {tuple(fr_dev.shape)}")

    if args.cond == "gt_action":
        cond_dim = 18
        cond_tr, cond_dev = act_tr.clone(), act_dev.clone()          # already normalized
    else:
        arm = args.arm or SEED_ARMS[args.seed_tag]
        spec = spec_map[arm]; acfg = _load_cfg(_resolve(spec.config))
        acfg["data"]["exclude_manifest"] = CONSUMER_EXCLUDE; acfg["data"]["only_manifest"] = False
        enc = load_model(str(_resolve(spec.checkpoint)), acfg, device); enc.eval()
        ap_dim = enc.action_part
        cond_tr = torch.stack([encode_seq(enc, f, device)[:, :ap_dim] for f in fr_tr])   # [N,K,ap]
        cond_dev = torch.stack([encode_seq(enc, f, device)[:, :ap_dim] for f in fr_dev])
        mu = cond_tr.reshape(-1, ap_dim).mean(0); sd = cond_tr.reshape(-1, ap_dim).std(0) + 1e-6
        cond_tr = (cond_tr - mu) / sd; cond_dev = (cond_dev - mu) / sd     # standardize for FiLM
        cond_dim = ap_dim
        del enc; torch.cuda.empty_cache()
        print(f"zmu cond arm={arm} dim={cond_dim}")

    model = ConditionalUNet(cond_dim=cond_dim, base=args.base).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"consumer params={n_par/1e6:.1f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    N, K = fr_tr.shape[0], args.K
    rng = np.random.default_rng(args.train_seed)

    model.train()
    for step in range(1, args.steps + 1):
        ci = rng.integers(0, N, args.bs)
        ti = rng.integers(0, K, args.bs)                            # random step per clip (teacher-forced)
        o_t = torch.stack([fr_tr[ci[b], ti[b]] for b in range(args.bs)]).permute(0, 3, 1, 2).to(device)
        o_n = torch.stack([fr_tr[ci[b], ti[b] + 1] for b in range(args.bs)]).permute(0, 3, 1, 2).to(device)
        c_t = torch.stack([cond_tr[ci[b], ti[b]] for b in range(args.bs)]).to(device)
        pred = model(o_t, c_t)
        # motion-weighted L1: emphasize pixels that actually move (o_t→o_{t+1}), so the model
        # cannot minimize loss by copying o_t (the persistence shortcut) — it must use the action.
        w = 1.0 + args.motion_lambda * (o_n - o_t).abs()
        loss = (w * (pred - o_n).abs()).sum() / w.sum()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 250 == 0 or step == 1:
            model.eval()
            g = dev_gate(model, fr_dev, cond_dev, device)
            model.train()
            hK = max(g["target"].keys())
            def e(cond, h): return g[cond][h]["endpoint_err"]
            def dc(cond, h): return g[cond][h]["dir_cos"]
            print(f"step {step:5d} L1w={loss.item():.4f} | h1 endpoint tgt={e('target',1):.3f} "
                  f"shuf={e('shuffle',1):.3f} anti={e('antiparallel',1):.3f} zero={e('zero',1):.3f} "
                  f"| h1 dir_cos tgt={dc('target',1):+.3f} shuf={dc('shuffle',1):+.3f} anti={dc('antiparallel',1):+.3f} "
                  f"| h{hK} dir_cos tgt={dc('target',hK):+.3f} anti={dc('antiparallel',hK):+.3f}")

    model.eval()
    gate = dev_gate(model, fr_dev, cond_dev, device)
    tag = args.cond if args.cond == "gt_action" else f"zmu_{args.arm or SEED_ARMS[args.seed_tag]}"
    out = args.out or f"results/consumer_{tag}_s{args.train_seed}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    payload = {"cond": args.cond, "arm": args.arm, "K": K, "skip": args.skip, "steps": args.steps,
               "n_train": N, "n_dev": int(fr_dev.shape[0]), "params_M": n_par / 1e6, "gate": gate}
    Path(out).write_text(json.dumps(payload, indent=2))
    ckpt = out.replace(".json", ".pt")
    torch.save(model.state_dict(), ckpt)
    hs = sorted(gate["target"].keys())
    print(f"\nGATE ({tag}):  [horizons {hs}]")
    for cond in ["target", "shuffle", "antiparallel", "zero"]:
        row = "  ".join(f"h{h}: ep={gate[cond][h]['endpoint_err']:.3f} dc={gate[cond][h]['dir_cos']:+.3f}" for h in hs)
        print(f"  {cond:12} {row}")
    # PASS = at h=1 (drift-free) target follows direction better than shuffle AND antiparallel,
    # and target endpoint beats antiparallel. dir_cos is the persistence-robust discriminator.
    h1 = 1 if 1 in gate["target"] else hs[0]
    dct, dcs, dca = (gate[c][h1]["dir_cos"] for c in ["target", "shuffle", "antiparallel"])
    ept, epa = gate["target"][h1]["endpoint_err"], gate["antiparallel"][h1]["endpoint_err"]
    passed = (dct > dcs) and (dct > dca) and (ept < epa)
    print(f"  => positive-control gate {'PASS' if passed else 'FAIL'} @h{h1}: "
          f"dir_cos target {dct:+.3f} vs shuffle {dcs:+.3f} vs antiparallel {dca:+.3f}; "
          f"endpoint target {ept:.3f} vs antiparallel {epa:.3f}")
    print(f"wrote {out} + {ckpt}")


if __name__ == "__main__":
    main()
