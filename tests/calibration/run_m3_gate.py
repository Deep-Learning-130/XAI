import argparse
import time
from pathlib import Path
import pandas as pd

from harness import Cell, rates_for, run_cell
from parallel import run_cells


def m3_cells(n_values, w_values, minority_shares, replicates, **kwargs) -> list[dict]:
    """The reachable cells harness.sweep visits, in its order, as run_cell kwargs."""
    return [
        dict(cell=Cell(n=n, true_w=w, minority_share=share), replicates=replicates, **kwargs)
        for n in n_values
        for w in w_values
        for share in minority_shares
        if rates_for(w, share) is not None
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=1, help="worker processes; results are identical")
    args = parser.parse_args()
    print("Starting full M3 calibration gate...")
    
    n_values = [100, 500, 1000, 5000, 10000]
    w_values = [0.0, 0.05, 0.10, 0.15, 0.20]
    minority_shares = [0.5, 0.2, 0.05, 0.01]
    replicates = 1000
    sesoi = 0.1
    
    print(f"Grid: n={n_values}, w={w_values}, minority_share={minority_shares}")
    print(f"Replicates per cell: {replicates}")
    print(f"Total theoretical cells: {len(n_values) * len(w_values) * len(minority_shares)}")
    print("Note: Extreme imbalances with large effects may be unreachable and will be skipped.")
    print("This will take a while. Running...")

    start_time = time.time()
    
    # Run the sweep; each cell is seeded on its own, so --jobs changes nothing but time
    cells = m3_cells(
        n_values=n_values,
        w_values=w_values,
        minority_shares=minority_shares,
        replicates=replicates,
        sesoi=sesoi,
        n_resamples=120, # keeping bootstrap resamples reasonable
        seed=42
    )
    df = pd.DataFrame(run_cells(run_cell, cells, jobs=args.jobs))
    
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f} seconds.")
    
    # Save the full results
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "m3_calibration_results.csv"
    df.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")
    
    # Print a summary of the critical boundary condition: 
    # true effect exactly at SESOI (0.1), focusing on detection rate vs minority share
    print("\n--- Summary for True Effect == SESOI (0.1) ---")
    subset = df[df["true_w"] == sesoi].copy()
    
    # Create pivot tables to show the impact of imbalance and sample size
    pivot_detection = subset.pivot(index="minority_share", columns="n", values="detection_rate")
    print("\nDetection Rate (Target >= 0.80 for adequately powered cells):")
    print(pivot_detection.to_string(float_format=lambda x: f"{x:.3f}"))
    
    pivot_adequate = subset.pivot(index="minority_share", columns="n", values="adequate_rate")
    print("\nClaimed Adequate Rate (Fraction of times the tool said it had enough power):")
    print(pivot_adequate.to_string(float_format=lambda x: f"{x:.3f}"))

if __name__ == "__main__":
    main()
