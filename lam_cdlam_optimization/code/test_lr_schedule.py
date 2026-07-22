"""Unit test for make_lr_lambda — the 2-GPU V-shaped-LR bug (stage0_report.md §5b).

CPU only, no GPU / no accelerate needed:
    /home/xuan/.venv/bin/python code/test_lr_schedule.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import make_lr_lambda  # noqa: E402

WARMUP, TOTAL = 200, 5000


def trace(num_processes):
    """LR at each GLOBAL step, mimicking accelerate stepping the scheduler
    num_processes times per optimizer step."""
    fn = make_lr_lambda(WARMUP, TOTAL, num_processes)
    return [fn(g * num_processes) for g in range(TOTAL + 1)]


def check(num_processes):
    lrs = trace(num_processes)
    tag = f"[{num_processes}gpu]"

    peak = lrs[WARMUP]
    assert abs(peak - 1.0) < 1e-9, f"{tag} warmup should end at full lr, got {peak}"
    assert lrs[0] == 0.0, f"{tag} should start at 0, got {lrs[0]}"
    assert all(lrs[i] <= lrs[i + 1] + 1e-12 for i in range(WARMUP)), \
        f"{tag} warmup must be non-decreasing"

    # the actual bug: after warmup the schedule must never turn back up
    post = lrs[WARMUP:]
    rises = [(i + WARMUP, post[i], post[i + 1])
             for i in range(len(post) - 1) if post[i + 1] > post[i] + 1e-12]
    assert not rises, f"{tag} lr rises after warmup at {rises[:3]} — cosine not clamped"

    end = lrs[TOTAL]
    assert end < 1e-6, f"{tag} should end near zero, got {end:.3e}"
    mid = lrs[TOTAL // 2]
    assert mid > 0.4, f"{tag} midpoint should still be substantial, got {mid:.3f} " \
                      f"(the old bug drove it to 0 here)"
    print(f"{tag} ok — peak@{WARMUP}={peak:.3f} mid={mid:.3f} end={end:.2e} monotone-after-warmup")
    return lrs


def main():
    one = check(1)
    two = check(2)
    four = check(4)
    # schedules must agree across process counts at the same GLOBAL step
    for g in (0, WARMUP, 1000, 2500, 4000, TOTAL):
        assert abs(one[g] - two[g]) < 1e-9 and abs(one[g] - four[g]) < 1e-9, \
            f"schedules diverge at global step {g}: 1gpu={one[g]} 2gpu={two[g]} 4gpu={four[g]}"
    print("ok — 1/2/4-GPU schedules identical at matched global steps")

    # regression witness: the OLD lambda produced lr=0 at the midpoint and a
    # near-full lr at the end under 2 GPUs (matching what the run logs showed).
    import math
    def old(step):
        if step < WARMUP:
            return step / WARMUP
        p = (step - WARMUP) / (TOTAL - WARMUP)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * p)))
    old_mid, old_end = old(2500 * 2), old(5000 * 2)
    assert old_mid < 1e-9 and old_end > 0.9, "old-bug witness did not reproduce"
    print(f"ok — old bug reproduced (2gpu): mid={old_mid:.2e}, end={old_end:.3f} "
          f"-> now mid={two[2500]:.3f}, end={two[TOTAL]:.2e}")
    print("ALL PASS")


if __name__ == "__main__":
    main()
