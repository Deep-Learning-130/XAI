# MVP Specification & Roadmap

## Minimum Viable Product (MVP) Definition

The MVP is a **headless Python library/CLI tool**. It does not include the dashboard or post-training ML fairness metrics.

**Goal**: Prove the statistical math works on tabular data.
* **Input**: A Pandas DataFrame and a configuration dictionary (defining target and sensitive columns).
* **Processing**: 
  * Calculate Representation Imbalance.
  * Calculate Missingness Bias.
  * Calculate Feature/Label Disparities.
  * Apply BH-FDR correction and Effect Size thresholds.
* **Output**: A JSON array of fully scored `Finding` objects.

## Development Roadmap

### Phase 1: Foundations & Schemas
* **Tasks**: Define the `Finding` dataclass. Set up `pytest`. Set up Pandas ingestion.
* **Definition of Done**: Project scaffolding exists, data can be loaded, schema inferred.

### Phase 2: The Statistical Core
* **Tasks**: Implement `scipy.stats` wrappers. Implement Cohen's $w$, Cramer's V, Rank-Biserial calculators. Implement BH-FDR correction.
* **Definition of Done**: Unit tests pass confirming accurate $p$-values and effect sizes matching known R/Python packages.

### Phase 3: The Analyzers
* **Tasks**: Build `RepresentationAnalyzer`, `MissingnessAnalyzer`, `FeatureDisparityAnalyzer`.
* **Definition of Done**: Analyzers can consume data and output raw, un-scored findings.

### Phase 4: Rules Engine & Synthetic Evaluation
* **Tasks**: Implement the Risk/Severity framework. Build the Synthetic Data Generator.
* **Definition of Done**: The engine accurately detects injected bias in synthetic data and ignores large-$N$ noise. **(MVP Complete)**

### Phase 5: The Dashboard (V1)
* **Tasks**: Build Streamlit UX. Integrate Plotly for KDEs and Radar charts.
* **Definition of Done**: User can upload a CSV, configure attributes, and view the Findings Feed.

### Phase 6: Post-Training Fairness & XAI (V2)
* **Tasks**: Allow ingestion of model predictions. Integrate `Fairlearn` metrics (Equal Opportunity). Integrate `SHAP` to explain disparities.
* **Definition of Done**: Dashboard shows Model Fairness risk alongside Dataset bias risk.

### Phase 7: Polish & Documentation
* **Tasks**: Export to PDF features. Write final academic report/thesis.
* **Definition of Done**: Code is production-ready and fully documented.
