"""冻结 Tier-2 route-C 的 held-out **Eval-B** manifest(顾问第四轮 §5 数据独立;Phase 0.3 预注册)。

与 Eval-A(`data/eval_manifest.json`,episode 级 holdout,H2 模型选择用)的关系:
  - Eval-B 是**held-out TASK** 级(整个 task 保留),比 episode 级更强,支撑"跨 task transfer";
  - Eval-B ∩ Eval-A = ∅(构造时减去 Eval-A 的 episode);
  - Phase-2 独立 consumer **训练时必须排除这些 task**(既排 Eval-A 也排 Eval-B task)。

⚠️ 注意:frozen encoder 在预训练时**见过** Eval-B 的 episode——这是已声明的 limitation,
不是 Tier-2 的 consumer-independence 轴。Tier-2 的数据独立指:consumer 的 test(Eval-B)与
consumer 的 train、以及 H2 scorer-validation(Eval-A)分离。

选择规则(确定性,seed):可逆 task(2 verb,signed-direction 用)优先,总 episode 在
[min_eps, max_eps] 带内(太小→donor 池不足;太大→饿死 consumer 训练);另加少量非可逆 task。

⚠️ 一旦冻结(且 Phase-2 结果已解封),不可重生成。
    python code/make_eval_manifest_B.py --n_rev 10 --n_plain 4 --out data/eval_manifest_B.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _apath  # noqa: F401  # shared A (lam_cdlam_optimization) substrate on sys.path
from dataset import EgoDexDataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/home/xuan/embodied-ai/data/egodex/test_240p")
    ap.add_argument("--eval_a", default="data/eval_manifest.json")
    ap.add_argument("--n_rev", type=int, default=10, help="reversible tasks to reserve")
    ap.add_argument("--n_plain", type=int, default=4, help="non-reversible tasks to reserve")
    ap.add_argument("--min_eps", type=int, default=8)
    ap.add_argument("--max_eps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/eval_manifest_B.json")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} 已存在。Eval-B 一旦冻结且解封过就不可重生成。要重来请手动删除并作废相关 Tier-2 run。")

    ds = EgoDexDataset(data_root=args.data_root, img_h=240, img_w=320,
                       downsample_factors=(1,), split="train", val_ratio=0.1,
                       seed=42, load_verbs=True, load_actions=False)
    by_task_verb = defaultdict(lambda: defaultdict(set))
    all_eps = defaultdict(set)
    for rec in ds._index:
        task, ep = rec[2], rec[3]
        all_eps[task].add(ep)
        v = ds._verb_map.get(ep)
        v = str(v).strip().lower().replace("_", " ") if v else None
        if v and v != "unknown":
            by_task_verb[task][v].add(ep)

    evalA = set(json.load(open(args.eval_a))["episodes"])
    evalA_by_task = defaultdict(set)
    for e in evalA:
        evalA_by_task[e.split("/", 1)[0]].add(e)

    tasks = sorted(all_eps)
    rev = [t for t in tasks if len(by_task_verb[t]) == 2]
    plain = [t for t in tasks if len(by_task_verb[t]) != 2]

    def eligible(t):
        return args.min_eps <= len(all_eps[t]) <= args.max_eps

    rev_cand = sorted(t for t in rev if eligible(t))
    plain_cand = sorted(t for t in plain if eligible(t))
    rng = np.random.default_rng(args.seed)
    rev_pick = sorted(rng.choice(rev_cand, size=min(args.n_rev, len(rev_cand)), replace=False).tolist())
    plain_pick = sorted(rng.choice(plain_cand, size=min(args.n_plain, len(plain_cand)), replace=False).tolist())
    held_tasks = sorted(rev_pick + plain_pick)

    # Eval-B episodes = all episodes of held-out tasks MINUS Eval-A (disjoint by construction)
    held_eps = []
    for t in held_tasks:
        held_eps.extend(f"{t}/{ep}" for ep in sorted(all_eps[t]))
    held_eps = sorted(set(held_eps) - evalA)
    assert not (set(held_eps) & evalA), "Eval-B overlaps Eval-A"

    n_total = sum(len(v) for v in all_eps.values())
    payload = {
        "note": "HELD-OUT TASK split for Tier-2 route-C. FROZEN — unseal ONCE for the final "
                "route-C A0-vs-A1 comparison. The Phase-2 consumer MUST exclude these tasks from "
                "training. Disjoint from Eval-A (data/eval_manifest.json). Encoder pretrained on "
                "these episodes (declared limitation, not the consumer-independence axis).",
        "created_for": "Tier-2 route-C independent-consumer final evaluation (Phase 0.3 prereg)",
        "config": vars(args),
        "held_out_tasks": held_tasks,
        "reversible_tasks": rev_pick,
        "plain_tasks": plain_pick,
        "n_tasks": len(held_tasks),
        "n_episodes": len(held_eps),
        "n_total_train_episodes": n_total,
        "episodes": held_eps,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"Eval-B: {len(held_eps)} episodes over {len(held_tasks)} held-out tasks "
          f"({len(rev_pick)} reversible + {len(plain_pick)} plain)")
    print(f"  reversible tasks: {rev_pick}")
    print(f"  plain tasks:      {plain_pick}")
    for t in held_tasks:
        nb = len([e for e in held_eps if e.split('/',1)[0] == t])
        na = len(evalA_by_task.get(t, []))
        vs = {k: len(v) for k, v in by_task_verb[t].items()}
        print(f"    {t:52} evalB={nb:3d}  (evalA held {na:2d}, verbs {vs})")
    print(f"  ⚠ FROZEN — Phase-2 consumer excludes BOTH Eval-A and these {len(held_tasks)} tasks from training.")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
