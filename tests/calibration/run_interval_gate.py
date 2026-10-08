"""Full interval calibration gate (pre-registered criteria in
docs/superpowers/plans/2026-10-04-bias-corrected-cramers-v.md, Task 4).

    cd tests/calibration && python run_interval_gate.py [--jobs N]
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from interval_harness import joint_probs, run_interval_cell
from parallel import run_cells

SHAPES = {
    "2x2": (np.full(2, 0.5), np.full(2, 0.5)),
    "2x5 skewed": (np.array([0.67, 0.33]), np.array([0.85, 0.10, 0.03, 0.01, 0.01])),
    "5x5": (np.full(5, 0.2), np.full(5, 0.2)),
    "16x2": (np.full(16, 1 / 16), np.array([0.67, 0.33])),
    "14x5 skewed": (np.full(14, 1 / 14), np.array([0.85, 0.10, 0.03, 0.01, 0.01])),
}
N_VALUES = [200, 500, 2000]
W_VALUES = [0.0, 0.1, 0.2]
SESOI = 0.1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()

    start = time.time()
    names, cells = [], []
    for name, (r, c) in SHAPES.items():
        for n in N_VALUES:
            for w in W_VALUES:
                if joint_probs(r, c, w) is None:
                    print(f"skip unreachable: {name} n={n} w={w}")
                    continue
                names.append(name)
                cells.append(dict(row_shares=r, col_shares=c, n=n, true_w=w,
                                  sesoi=SESOI, replicates=1000, seed=n))

    rows = run_cells(run_interval_cell, cells, jobs=args.jobs)
    for name, row in zip(names, rows):
        row["shape"] = name
    df = pd.DataFrame(rows)
    out = Path("results") / "interval_calibration_results.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    g1 = df[df.true_w == SESOI].equivalent_rate.max()
    g2 = df[df.true_w == 0].spurious_disparity_rate.max()
    print(f"\nG1 max EQUIVALENT rate at w = SESOI: {g1:.3f}  (gate <= 0.05)")
    print(f"G2 max spurious-disparity rate at w = 0: {g2:.3f}  (gate <= 0.05)")
    print(f"   old interval, same G2 statistic:  {df[df.true_w == 0].raw_spurious_disparity_rate.max():.3f}")
    print(f"GATE {'PASSED' if g1 <= 0.05 and g2 <= 0.05 else 'FAILED'}  ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
