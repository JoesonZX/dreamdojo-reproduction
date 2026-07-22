"""冻结一份 LAM 永不训练的评测 manifest(阶段 B0 第 2 项)。

为什么需要:条件 probe 的 GroupKFold 只能阻止 **probe 自身**的 episode 泄漏,
**撤销不了 encoder 训练时已经看过这些 episode** 这一事实。因此阶段 A 在 train 池上
得到的所有 probe 数字只能是 exploratory。要得到 confirmatory 结论,必须有一批
encoder 从未见过的 episode。

策略:按 task 分层抽 `--frac` 的 episode;**可逆 task 优先保证两个 verb 都有代表**
(否则该 task 在相反动词 probe 里直接作废)。输出 JSON,由 `EgoDexDataset(
exclude_manifest=...)` 在建索引时剔除。

⚠️ 一旦有 run 用它训练过,就**不能再改**——改了等于评测集被污染。
    python code/make_eval_manifest.py --frac 0.10 --out data/eval_manifest.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import EgoDexDataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/home/xuan/embodied-ai/data/egodex/test_240p")
    ap.add_argument("--frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--min_per_verb", type=int, default=2,
                    help="reversible tasks: hold out at least this many episodes per verb")
    ap.add_argument("--out", default="data/eval_manifest.json")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} 已存在。评测 manifest 一旦冻结就不可重生成"
                         f"(否则已训练的 run 会看见新评测集)。要重来请先手动删除并作废相关 run。")

    ds = EgoDexDataset(data_root=args.data_root, img_h=240, img_w=320,
                       downsample_factors=(1,), split="train", val_ratio=0.1,
                       seed=42, load_verbs=True, load_actions=False)
    by_task = defaultdict(lambda: defaultdict(set))
    all_eps = defaultdict(set)
    for rec in ds._index:
        task, ep = rec[2], rec[3]
        all_eps[task].add(ep)
        v = ds._verb_map.get(ep)
        v = str(v).strip().lower().replace("_", " ") if v else None
        if v and v != "unknown":
            by_task[task][v].add(ep)

    rng = np.random.default_rng(args.seed)
    held, stats = [], {"reversible_balanced": 0, "reversible_partial": 0, "plain": 0}
    for task in sorted(all_eps):
        eps = sorted(all_eps[task])
        quota = max(1, int(round(len(eps) * args.frac)))
        verbs = by_task.get(task, {})
        picked = set()
        if len(verbs) == 2:                       # reversible: keep both verbs present
            ok = True
            for v in sorted(verbs):
                pool = sorted(verbs[v])
                k = min(args.min_per_verb, max(1, len(pool) - 1))   # never take a whole verb
                if len(pool) <= 1:
                    ok = False
                    continue
                picked.update(rng.choice(pool, size=k, replace=False).tolist())
            stats["reversible_balanced" if ok else "reversible_partial"] += 1
        else:
            stats["plain"] += 1
        remaining = [e for e in eps if e not in picked]
        if len(picked) < quota and remaining:
            extra = min(quota - len(picked), len(remaining) - 1)    # never empty a task
            if extra > 0:
                picked.update(rng.choice(remaining, size=extra, replace=False).tolist())
        held.extend(sorted(picked))

    total = sum(len(v) for v in all_eps.values())
    payload = {
        "note": "Episodes excluded from LAM training so probes have an unseen pool. "
                "FROZEN — do not regenerate once any run has trained with it.",
        "created_for": "stage B confirmatory conditional-direction probe",
        "config": vars(args),
        "n_episodes": len(held),
        "n_total_train_episodes": total,
        "episodes": sorted(held),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"held out {len(held)}/{total} episodes ({len(held)/total:.1%}) over {len(all_eps)} tasks")
    print(f"  可逆 task 两 verb 均有代表: {stats['reversible_balanced']}  "
          f"部分: {stats['reversible_partial']}  非可逆: {stats['plain']}")
    rev_held = sum(1 for t in by_task if len(by_task[t]) == 2
                   and any(e in held for v in by_task[t].values() for e in v))
    print(f"  被覆盖的可逆 task: {rev_held}/{sum(1 for t in by_task if len(by_task[t]) == 2)}")
    print(f"wrote {out}  (⚠ 冻结,勿重生成)")


if __name__ == "__main__":
    main()
