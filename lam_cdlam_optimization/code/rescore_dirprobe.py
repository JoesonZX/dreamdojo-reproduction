"""对所有已有 checkpoint 离线重评"条件方向 probe"(阶段 A1)。

只跑 `_collect_opposite` 那一段(任务分层采样 + 编码 transition/static 两种 latent),
再算新的条件 probe 与旧的 static_shortcut_gap —— 比整跑 benchmark 快得多。

    CUDA_VISIBLE_DEVICES=0 python code/rescore_dirprobe.py --out results/dirprobe_rescore.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
import yaml

from benchmark_lam import (RUN_SPECS, _build_raw_model, _collect_opposite,  # noqa: E402
                           _conditional_direction_probe, _load_cfg,
                           _opposite_verb_classification, _resolve, _suffixed)
from eval import load_model  # noqa: E402

DEFAULT_RUNS = [
    "raw_lam", "cdlam_official", "cdlam_repro_5k",
    "kl_ft_l40_full_5k", "ours_a_zero_5k", "ours_r2_opp_l01_5k",
    "kl_ft_l40_full_5k_lf", "ours_r2_opp_l01_5k_lf", "idm_5k",
    "kl_ft_l40_full_5k_s1", "kl_ft_l40_full_5k_s2",
    "ours_a_zero_5k_s1", "ours_a_zero_5k_s2",
    "ours_r2_opp_l01_5k_s1", "ours_r2_opp_l01_5k_s2",
]
FIELDS = ["run", "dirprobe_nll_base", "dirprobe_nll_full", "dirprobe_nll_gain",
          "dirprobe_null_mean", "dirprobe_null_std", "dirprobe_gain_vs_null",
          "dirprobe_gain_ci_lo", "dirprobe_gain_ci_hi", "dirprobe_gain_perm_p",
          "dirprobe_auc_base", "dirprobe_auc_full", "dirprobe_n_tasks",
          "dirprobe_n_samples", "opp_cls_bal_acc", "opp_cls_bal_acc_static",
          "static_shortcut_gap"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=DEFAULT_RUNS)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--n_perm", type=int, default=20)
    ap.add_argument("--split", default="train",
                    help="episode pool for the grouped probe. The val split has a MEDIAN "
                         "of 1 episode for the minority verb, which makes episode-grouped "
                         "CV degenerate; train has a median of 8.")
    ap.add_argument("--min_eps_per_verb", type=int, default=4)
    ap.add_argument("--out", default="results/dirprobe_rescore.csv")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    spec_map = {s.name: s for s in RUN_SPECS}
    rows = []
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for name in args.runs:
        spec = spec_map.get(name)
        if spec is None:
            print(f"[skip] {name}: no RunSpec")
            continue
        cfg = _load_cfg(_resolve(spec.config))
        try:
            if spec.init_from:
                model = _build_raw_model(cfg, _resolve(spec.init_from), device)
                src = str(_resolve(spec.init_from))
            else:
                ck = _resolve(spec.checkpoint)
                if not any(ck.glob("*.safetensors")) and not (ck / "pytorch_model.bin").exists():
                    print(f"[skip] {name}: no checkpoint at {ck}")
                    continue
                model = load_model(str(ck), cfg, device)
                src = str(ck)
        except Exception as e:
            print(f"[skip] {name}: load failed ({type(e).__name__}: {e})")
            continue
        model.eval()
        ap_ = model.action_part
        print(f"[{name}] source={src} action_part={ap_}", flush=True)

        d = _collect_opposite(model, cfg, device, args.num_workers, args.batch_size,
                              split=args.split, min_eps_per_verb=args.min_eps_per_verb)
        if d is None:
            print(f"[skip] {name}: opposite collection empty")
            del model
            torch.cuda.empty_cache()
            continue

        zt, zs = d["mu"][:, :ap_], d["static_mu"][:, :ap_]
        row = {"run": name}
        row.update(_conditional_direction_probe(
            zt, zs, d["task"], d["verb"], d["ep"],
            n_boot=args.n_boot, n_perm=args.n_perm))
        occ = _opposite_verb_classification(zt, d["task"], d["verb"])
        ocs = _suffixed(_opposite_verb_classification(zs, d["task"], d["verb"]), "_static")
        row["opp_cls_bal_acc"] = occ["opp_cls_bal_acc"]
        row["opp_cls_bal_acc_static"] = ocs["opp_cls_bal_acc_static"]
        row["static_shortcut_gap"] = occ["opp_cls_bal_acc"] - ocs["opp_cls_bal_acc_static"]
        rows.append({k: row.get(k, "") for k in FIELDS})

        # readable verdict: the gain is biased negative, so judge it against the
        # permutation null, and only trust it when the full probe actually predicts.
        usable = row["dirprobe_auc_full"] == row["dirprobe_auc_full"] and row["dirprobe_auc_full"] > 0.5
        beats_null = row["dirprobe_gain_perm_p"] <= 0.05
        verdict = ("DIRECTION-INFO" if (usable and beats_null)
                   else "no-info" if usable else "probe-unusable(auc<=.5)")
        print(f"  gain={row['dirprobe_nll_gain']:+.4f} null={row['dirprobe_null_mean']:+.4f} "
              f"vs_null={row['dirprobe_gain_vs_null']:+.4f} p={row['dirprobe_gain_perm_p']:.2f} "
              f"auc {row['dirprobe_auc_base']:.3f}->{row['dirprobe_auc_full']:.3f} "
              f"| old_gap={row['static_shortcut_gap']:+.3f} -> {verdict}", flush=True)

        with open(out_path, "w", newline="") as f:          # incremental save
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        del model
        torch.cuda.empty_cache()

    print(f"\nwrote {out_path} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
