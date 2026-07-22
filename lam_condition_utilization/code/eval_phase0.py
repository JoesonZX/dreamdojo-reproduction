"""Phase 0 编排(顾问 §0 零训练硬门):0.1 paired-latent audit + 0.2 four-cell cross-decoding.

在冻结 manifest(Eval-A)上取**一次**共享 pool(帧 + 每模型 z_mu + 18D 动作 + 标签),对每个
seed 的 matched pair(A0=α0 / A1=α1)做:
  0.1  逐 transition 的 paired-latent audit(动作坐标是否保持)
  0.2  D_{A0/A1} × E_{A0/A1} 四格 cross-decoding(修复发生在 encoder 还是 decoder 侧)

对角格 (D_A0×E_A0 / D_A1×E_A1) 的 dist C_pooled 必须复现 poscontrol_alpha(≈0.830 / 0.039)
——correctness gate。donor 方案与 H2 正对照完全一致(_build_schemes)。

    CUDA_VISIBLE_DEVICES=<free> python -u code/eval_phase0.py --device cuda:0 --seeds 42 1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np                                                          # noqa: E402
import torch                                                                # noqa: E402

import _apath  # noqa: F401  # shared A (lam_cdlam_optimization) substrate on sys.path
from benchmark_lam import RUN_SPECS, _encode_mu_var, _load_cfg, _make_loader, _norm_verb, _resolve  # noqa: E402
from eval import load_model                                                 # noqa: E402
from h2_poscontrol import _build_schemes, _pool_stats, oracle_c            # noqa: E402
from cross_decode import cell_metrics                                       # noqa: E402
from paired_latent import paired_audit                                      # noqa: E402

MANIFEST = "/home/xuan/embodied-ai/lam_condition_utilization/data/eval_manifest.json"
# seed -> (A0 arm, A1 arm) checkpoint names in RUN_SPECS
SEED_ARMS = {"42": ("sb_A0", "sb_A1"), "1": ("sb_A0_s1", "sb_A1_s1")}


def _load_pool(cfg, num_workers, batch_size, n_pool):
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
    ap.add_argument("--seeds", nargs="*", default=["42", "1"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--n_pool", type=int, default=1500)
    ap.add_argument("--k_knn", type=int, default=10)
    ap.add_argument("--out_audit", default="results/stage_b_phase0_paired_latent.json")
    ap.add_argument("--out_cross", default="results/stage_b_phase0_cross_decode.json")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    spec_map = {s.name: s for s in RUN_SPECS}

    # ---- shared pool on the frozen manifest (encoder never trained on it) ----
    base_spec = spec_map[SEED_ARMS[args.seeds[0]][0]]
    cfg = _load_cfg(_resolve(base_spec.config))
    cfg["data"]["exclude_manifest"] = MANIFEST
    cfg["data"]["only_manifest"] = True
    videos, task_arr, ep_arr, verb_arr, act = _load_pool(cfg, args.num_workers, args.batch_size, args.n_pool)
    schemes = _build_schemes(task_arr, ep_arr, verb_arr, act)
    n_anchors = len(schemes["dist"])
    print(f"pool N={len(videos)}  anchors={n_anchors}  video shape={tuple(videos.shape)}")

    donor_stats = {name: _pool_stats(sch, act) for name, sch in schemes.items()}
    print("\n=== donor 方案排序统计(18D 动作)===")
    for name, st in donor_stats.items():
        print(f"  {name:12} n={st['n']:4d}  frac(ds<do)={st['frac_same_lt_opp']:.3f} "
              f"cos_same={st['cos_same']:+.2f} cos_opp={st['cos_opp']:+.2f} "
              f"mag_o/i={st['magratio_opp']:.2f}")

    print("\n=== frame-delta ORACLE(参考上限)C_pooled ===")
    oracle = {name: oracle_c(videos, sch, device) for name, sch in schemes.items()}
    for name, oc in oracle.items():
        print(f"  oracle × {name:12}  C_pooled={oc['C_pooled']:+.4f}")

    audit_out = {"pool": {"N": int(len(videos)), "n_anchors": int(n_anchors)},
                 "donor_stats": donor_stats, "seeds": {}}
    cross_out = {"pool": {"N": int(len(videos)), "n_anchors": int(n_anchors)},
                 "donor_stats": donor_stats,
                 "oracle": {k: v["C_pooled"] for k, v in oracle.items()},
                 "seeds": {}}

    for seed in args.seeds:
        a0_name, a1_name = SEED_ARMS[seed]
        print(f"\n{'='*70}\nSEED {seed}:  A0={a0_name}  A1={a1_name}\n{'='*70}")

        def _load(arm):
            spec = spec_map[arm]
            acfg = _load_cfg(_resolve(spec.config))
            acfg["data"]["exclude_manifest"] = MANIFEST; acfg["data"]["only_manifest"] = True
            m = load_model(str(_resolve(spec.checkpoint)), acfg, device); m.eval()
            return m

        m_a0, m_a1 = _load(a0_name), _load(a1_name)
        ap_dim = m_a0.action_part
        mu_a0 = _encode_pool(m_a0, videos, device, args.batch_size)
        mu_a1 = _encode_pool(m_a1, videos, device, args.batch_size)

        # ---- 0.1 paired-latent audit ----
        print("\n--- 0.1 paired-latent audit (A0 vs A1) ---")
        aud = paired_audit(mu_a0.numpy(), mu_a1.numpy(), act, ep_arr, schemes, ap_dim,
                           k_knn=args.k_knn, seed=int(seed))
        audit_out["seeds"][seed] = aud
        ro = aud["readout_r2"]
        print(f"  ‖Δμ‖ rel={aud['l2_diff_rel_mean']:.3f}  cosine={aud['cosine_mean']:.3f}  "
              f"per-dim r(action)={aud['per_dim_corr_action_mean']:.3f}")
        print(f"  CKA={aud['cka_action']:.3f}  Procrustes-R²={aud['procrustes_r2_action']:.3f}  "
              f"kNN-overlap@{args.k_knn}={aud['knn_overlap_action']:.3f}")
        print(f"  readout R²:  self A0={ro['fitA0_evalA0']:.3f}  cross A0→A1={ro['fitA0_evalA1']:.3f}  |  "
              f"self A1={ro['fitA1_evalA1']:.3f}  cross A1→A0={ro['fitA1_evalA0']:.3f}")

        # ---- 0.2 four-cell cross-decoding ----
        print("\n--- 0.2 four-cell cross-decoding (rows=decoder, cols=encoder μ) ---")
        cells = {}
        for dname, dmodel in [("D_A0", m_a0), ("D_A1", m_a1)]:
            for ename, emu in [("E_A0", mu_a0), ("E_A1", mu_a1)]:
                cm = cell_metrics(dmodel, emu, videos, schemes, device)
                cells[f"{dname}x{ename}"] = cm
                print(f"  {dname}×{ename}:  dist C={cm['dist']['C_pooled']:+.4f}  "
                      f"antipar C={cm['antiparallel']['C_pooled']:+.4f}  "
                      f"random C={cm['random']['C_pooled']:+.4f}  |  "
                      f"R_out(dist)={cm['dist']['R_out_fg']:.4f}  PSNR={cm['psnr']:.2f}")
        cross_out["seeds"][seed] = cells

        # compact judgment print (dist C matrix)
        d = lambda c: cells[c]["dist"]["C_pooled"]
        print(f"\n  [dist C matrix]      E_A0     E_A1")
        print(f"    D_A0            {d('D_A0xE_A0'):+.4f}  {d('D_A0xE_A1'):+.4f}")
        print(f"    D_A1            {d('D_A1xE_A0'):+.4f}  {d('D_A1xE_A1'):+.4f}")

        del m_a0, m_a1
        torch.cuda.empty_cache()

    Path(args.out_audit).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_audit).write_text(json.dumps(audit_out, indent=2))
    Path(args.out_cross).write_text(json.dumps(cross_out, indent=2))
    print(f"\nwrote {args.out_audit}\nwrote {args.out_cross}")


if __name__ == "__main__":
    main()
