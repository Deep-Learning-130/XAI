import time
from pathlib import Path
import pandas as pd

from harness import sweep

def main():
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
    
    # Run the sweep
    df = sweep(
        n_values=n_values,
        w_values=w_values,
        minority_shares=minority_shares,
        replicates=replicates,
        sesoi=sesoi,
        n_resamples=120, # keeping bootstrap resamples reasonable
        seed=42
    )
    
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
