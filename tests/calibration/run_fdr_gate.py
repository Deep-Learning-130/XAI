"""Full FDR gate (pre-registered criterion in
docs/superpowers/plans/2026-10-04-hierarchical-fdr.md, Task 4).

    cd tests/calibration && python run_fdr_gate.py [--jobs N]

Runs the default per-family procedure and writes
results/fdr_calibration_results_family.csv. The committed
results/fdr_calibration_results.csv also holds the hierarchical rows from
commit 649da32, where that procedure existed; it is the record and is not rewritten.
"""
import argparse
import time
from pathlib import Path

import pandas as pd

from fdr_harness import SCENARIOS, passes_gate, run_fdr_cell
from parallel import run_cells

ALPHA = 0.05


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()

    start = time.time()
    cells = [
        dict(scenario=name, n=2000, replicates=500, fdr=mode, alpha=ALPHA, seed=11)
        for name in SCENARIOS
        for mode in ("family",)
    ]
    rows = run_cells(run_fdr_cell, cells, jobs=args.jobs)
    df = pd.DataFrame(rows)
    out = Path("results") / "fdr_calibration_results_family.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    failed = [f"{r['scenario']}/{r['fdr_mode']}" for r in rows if not passes_gate(r, ALPHA)]
    print(f"\nGATE {'PASSED' if not failed else 'FAILED ' + ', '.join(failed)}: "
          f"every family's FDR <= alpha + 2 SE ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
