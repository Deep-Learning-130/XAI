"""Run the audit pipeline on the Adult dataset."""
import argparse
from dbias.ingestion.loaders import fetch_adult
from dbias.ingestion.schema_inference import infer_schema
from dbias.ingestion.sensitive_detection import suggest_sensitive_attributes
from dbias.audit import audit
from dbias.report.json_export import write_json

def main():
    print("Fetching Adult dataset...")
    df = fetch_adult()
    print(f"Loaded {len(df)} rows.")
    
    schema = infer_schema(df)
    suggestions = suggest_sensitive_attributes(df)
    print(f"Suggested sensitive attributes: {suggestions}")
    
    sensitive_attrs = ["sex", "race"]
    
    # SESOI of w=0.1
    print("Running audit...")
    result = audit(
        df=df,
        sensitive_cols=sensitive_attrs,
        target_col="income",
        sesoi=0.1,
        seed=0,
    )
    
    print(f"Audit complete. Found {len(result.findings)} total findings.")
    write_json(result, "adult_audit.json")
    print("Exported to adult_audit.json")

if __name__ == "__main__":
    main()
