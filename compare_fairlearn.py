"""Direct comparison between Fairlearn and dbias on the small-N benchmark.

Needs the optional benchmark extra: pip install -e .[bench]
"""

from fairlearn.metrics import MetricFrame, demographic_parity_difference

from dbias.ingestion.loaders import fetch_adult
from dbias.audit import audit
from dbias.report.json_export import blind_spot_reading

def main():
    print("Fetching Adult dataset and subsampling to N=500...")
    df = fetch_adult()
    df = df.sample(n=500, random_state=42).copy()
    
    # We found a blind spot in MAR_WORKCLASS_RACE
    # Let's create a binary target: 1 if workclass is missing, 0 otherwise
    df['workclass_missing'] = df['workclass'].isna().astype(int)
    
    print("\n--- FAIRLEARN OUTPUT ---")
    # Fairlearn MetricFrame for missingness by race
    # We use a dummy metric (mean) since we just want the rates
    mf = MetricFrame(
        metrics=lambda y_true, y_pred: y_true.mean(),
        y_true=df['workclass_missing'],
        y_pred=df['workclass_missing'],
        sensitive_features=df['race']
    )
    print("Missingness rate by race:")
    print(mf.by_group)
    
    dp_diff = demographic_parity_difference(
        y_true=df['workclass_missing'], 
        y_pred=df['workclass_missing'], 
        sensitive_features=df['race']
    )
    print(f"\nDemographic Parity Difference: {dp_diff:.3f}")
    print(f"Conclusion a practitioner draws: 'The maximum difference in missingness rates is {dp_diff:.1%}. The groups are small, so it's not statistically significant. We pass the fairness check!'")
    
    print("\n--- DBIAS OUTPUT ---")
    result = audit(
        df=df,
        sensitive_cols=["race"],
        target_col="income",
        sesoi=0.1,
        seed=0,
    )
    
    # Find the missingness finding for workclass by race
    finding = next((f for f in result.findings if f.category.name == "MISSINGNESS" and f.target_feature == "workclass" and f.sensitive_attribute == "race"), None)
    
    if finding:
        print(f"Severity: {finding.severity.name}")
        print(f"Detectability: {finding.detectability.name}")
        if finding.effect_size_ci is not None:
            print(f"Confidence Interval for Effect Size: [{finding.effect_size_ci[0]:.3f}, {finding.effect_size_ci[1]:.3f}]")
        else:
            print("Confidence Interval for Effect Size: not computed")
        print(f"Verdict: {finding.equivalence_verdict}")
        if finding.severity.name == "BLIND_SPOT":
            print(f"Conclusion dbias draws: 'BLIND SPOT. {blind_spot_reading(finding)}'")
    else:
        print("Finding not found! Available findings:")
        for f in result.findings:
            print(f" - {f.id} (Category: {f.category.name}, Target: {f.target_feature}, Sensitive: {f.sensitive_attribute})")

if __name__ == "__main__":
    main()
