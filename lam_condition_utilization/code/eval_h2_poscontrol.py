"""H2 正对照编排(顾问第三轮 §1 硬门)。见 code/h2_poscontrol.py 头注。

在冻结 manifest 上取一次 pool(帧 + 每模型 mu + 18D 动作 + 标签),构建 verb/dist/random
三种 donor 方案,分别用 frame-delta oracle(无模型)与真实 LAM decoder(B/D)算 C。

    CUDA_VISIBLE_DEVICES=<free> python code/eval_h2_poscontrol.py --device cuda:0 --arms sb_B sb_D
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np                                                    # noqa: E402
import torch                                                          # noqa: E402

import _apath  # noqa: F401  # shared A (lam_cdlam_optimization) substrate on sys.path
from benchmark_lam import RUN_SPECS, _encode_mu_var, _load_cfg, _make_loader, _norm_verb, _resolve  # noqa: E402
from eval import load_model                                           # noqa: E402
from h2_poscontrol import _build_schemes, _pool_stats, lam_c, oracle_c  # noqa: E402

MANIFEST = "/home/xuan/embodied-ai/lam_condition_utilization/data/eval_manifest.json"


def _load_pool(cfg, num_workers, batch_size, n_pool):
    """One pass over the frozen manifest — frames + labels + 18D actions (model-independent)."""
    loader = _make_loader(cfg, num_workers, batch_size, split="train")
    videos_all, tasks, eps, verbs, actions = [], [], [], [], []
    seen = 0
    for batch in loader:
        videos_all.append(batch["videos"])
        tasks.extend(batch["task_id"]); eps.extend(batch["ep_id"])
        verbs.extend(batch.get("verb_id", ["unknown"] * batch["videos"].shape[0]))
        actions.append(batch["action"])
        seen += batch["videos"].shape[0]
        if seen >= n_pool:
            break
    videos = torch.cat(videos_all, dim=0)[:n_pool]
    N = len(videos)
    return (videos,
            np.asarray(tasks[:N]), np.asarray(eps[:N]),
            np.asarray([_norm_verb(v) for v in verbs[:N]]),
            torch.cat(actions, dim=0)[:N].numpy())


@torch.no_grad()
def _encode_pool(model, videos, device, bs=32):
    mus = []
    for i in range(0, len(videos), bs):
        mu, _ = _encode_mu_var(model, videos[i:i + bs].to(device))
        mus.append(mu.cpu())
    return torch.cat(mus, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=["sb_B", "sb_D"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--n_pool", type=int, default=1500)
    ap.add_argument("--out", default="results/stage_b_h2_poscontrol.json")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    spec_map = {s.name: s for s in RUN_SPECS}

    # pool config: honour the frozen manifest (encoder never trained on it)
    base_spec = spec_map[args.arms[0]]
    cfg = _load_cfg(_resolve(base_spec.config))
    cfg["data"]["exclude_manifest"] = MANIFEST
    cfg["data"]["only_manifest"] = True
    videos, task_arr, ep_arr, verb_arr, act = _load_pool(cfg, args.num_workers, args.batch_size, args.n_pool)
    schemes = _build_schemes(task_arr, ep_arr, verb_arr, act)
    print(f"pool N={len(videos)}  anchors={len(schemes['verb'])}")

    out: dict = {"pool": {"N": int(len(videos)), "n_anchors": int(len(schemes["verb"]))},
                 "donor_stats": {}, "oracle": {}, "lam": {}}

    print("\n=== donor 方案排序统计(18D 动作:距离 / 余弦 / 幅度比) ===")
    for name, sch in schemes.items():
        st = _pool_stats(sch, act)
        out["donor_stats"][name] = st
        print(f"  {name:12} n={st['n']:4d}  d_same={st['d_same']:.3f} d_opp={st['d_opp']:.3f} "
              f"frac(ds<do)={st['frac_same_lt_opp']:.3f} | cos_same={st['cos_same']:+.3f} "
              f"cos_opp={st['cos_opp']:+.3f} anti_frac={st['frac_opp_antiparallel']:.3f} | "
              f"mag_s/i={st['magratio_same']:.2f} mag_o/i={st['magratio_opp']:.2f}")

    print("\n=== frame-delta ORACLE(已知强可控,无模型)C_pooled ===")
    for name, sch in schemes.items():
        oc = oracle_c(videos, sch, device)
        out["oracle"][name] = oc
        print(f"  oracle × {name:12}  C_pooled={oc['C_pooled']:+.4f}  (median {oc['C_median']:+.4f})")

    for arm in args.arms:
        spec = spec_map.get(arm)
        ck = _resolve(spec.checkpoint)
        if not any(ck.glob("*.safetensors")):
            print(f"[skip] {arm}: no checkpoint"); continue
        acfg = _load_cfg(_resolve(spec.config))
        acfg["data"]["exclude_manifest"] = MANIFEST; acfg["data"]["only_manifest"] = True
        model = load_model(str(ck), acfg, device); model.eval()
        mu = _encode_pool(model, videos, device)
        print(f"\n=== LAM {arm}({ck.parent.name})C_pooled ===")
        out["lam"][arm] = {}
        for name, sch in schemes.items():
            lc = lam_c(model, videos, mu, sch, device)
            out["lam"][arm][name] = lc
            print(f"  {arm} × {name:12}  C_pooled={lc['C_pooled']:+.4f}  (median {lc['C_median']:+.4f}, "
                  f"raw {lc['raw_opp_minus_same']:+.2e})")
        del model
        torch.cuda.empty_cache()

    o = Path(args.out); o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps(out, indent=2))

    print("\n=== 判读(C_pooled)===")
    # MAGNITUDE: dist oracle is a valid ceiling (same-task donors, pixel-delta transfers).
    print(f"  [MAGNITUDE] oracle×dist={out['oracle']['dist']['C_pooled']:+.4f}, "
          f"oracle×random={out['oracle']['random']['C_pooled']:+.4f}")
    for arm in out["lam"]:
        lc = out["lam"][arm]["dist"]["C_pooled"]; oc = out["oracle"]["dist"]["C_pooled"]
        print(f"     {arm}×dist C_pooled={lc:+.4f} ({lc/oc*100:.0f}% oracle)  "
              f"{'→ 有 MAGNITUDE 可用性' if lc >= 0.3*oc else '→ MAGNITUDE deficit'}")
    # DIRECTION: donors are cross-scene → the pixel frame-delta oracle does NOT transfer
    # (its low C is a property of the pixel oracle, not scorer insensitivity — dist already
    # proved the scorer is sensitive). Read as an A0-vs-A1 contrast on magnitude-matched,
    # direction-flipped donors; analytic perfect-sign ceiling is C=4 on these anchors.
    print(f"  [DIRECTION] (frame-delta oracle scene-limited here: "
          f"{out['oracle']['antiparallel']['C_pooled']:+.4f}; perfect-sign ceiling≈4.0)")
    for arm in out["lam"]:
        st = out["donor_stats"]["antiparallel"]
        lc = out["lam"][arm]["antiparallel"]["C_pooled"]
        print(f"     {arm}×antiparallel C_pooled={lc:+.4f}  "
              f"(donors: cos_same={st['cos_same']:+.2f} cos_opp={st['cos_opp']:+.2f} "
              f"mag_o/i={st['magratio_opp']:.2f})")
    print(f"\nwrote {o}")


if __name__ == "__main__":
    main()
