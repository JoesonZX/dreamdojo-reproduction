"""Put project A's shared LAM substrate on sys.path.

lam_condition_utilization (Project B) is a diagnostic study built on the CD-LAM
trainer/benchmark substrate OWNED by lam_cdlam_optimization (Project A). B's
evaluators import that substrate (benchmark_lam, eval, dataset, model, train,
primitive_labels, dirprobe) as a documented, read-only cross-project dependency.

Importing this module first inserts A's code/ directory onto sys.path so those
imports resolve no matter what the current working directory is. See
PROJECT_MANIFEST.md ("Dependencies") for the ownership rationale.
"""
import os
import sys

_A_CODE = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), os.pardir, os.pardir,
        "lam_cdlam_optimization", "code",
    )
)
if _A_CODE not in sys.path:
    sys.path.insert(0, _A_CODE)
