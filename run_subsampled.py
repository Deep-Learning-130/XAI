"""Run the audit pipeline on a subsampled Adult dataset."""
import argparse
from dbias.ingestion.loaders import fetch_adult
from dbias.ingestion.schema_inference import infer_schema
from dbias.ingestion.sensitive_detection import suggest_sensitive_attributes
from dbias.audit import audit
from dbias.report.json_export import write_json
from dbias.report.coverage_plot import plot_coverage_map

def main():
    print("Fetching Adult dataset...")
    df = fetch_adult()
    print(f"Loaded {len(df)} rows.")
    
    # Subsample to simulate a small-N scenario
    N_SUBSAMPLE = 500
    print(f"Subsampling to {N_SUBSAMPLE} rows to test small-N detectability...")
    df = df.sample(n=N_SUBSAMPLE, random_state=42).copy()
    
    schema = infer_schema(df)
    
    sensitive_attrs = ["sex", "race"]
    
    # SESOI of w=0.1
    print("Running audit...")
    result = audit(
        df=df,
        sensitive_cols=sensitive_attrs,
        target_col="income",
        sesoi=0.1
    )
    
    print(f"Audit complete. Found {len(result.findings)} total findings.")
    
    blind_spots = result.blind_spots
    print(f"Found {len(blind_spots)} blind spots out of {len(result.findings)} tests!")
    
    for bs in blind_spots:
        print(f" - BLIND SPOT: {bs.id} (Category: {bs.category.name}, CI: [{bs.effect_size_ci[0]:.3f}, {bs.effect_size_ci[1]:.3f}])")
        
    write_json(result, "subsampled_audit.json")
    print("Exported to subsampled_audit.json")
    
    plot_coverage_map(result, "subsampled_coverage.png")
    print("Exported coverage plot to subsampled_coverage.png")

if __name__ == "__main__":
    main()
