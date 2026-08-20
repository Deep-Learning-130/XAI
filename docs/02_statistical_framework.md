# Statistical Framework

This project's credibility relies entirely on its statistical rigor. This document defines the mathematical methodologies required for the tool.

## 1. The Core Problem: The $P$-Value Trap
In large datasets ($N > 10,000$), standard errors approach zero. Consequently, almost any statistical test will yield $p < 0.05$ for trivially small, practically meaningless differences. 
**Rule**: The system must NEVER flag a disparity based solely on a $p$-value. **Effect Sizes are mandatory.**

## 2. Hypothesis Testing Decision Tree

The system must automatically select the correct statistical test based on the data types of the variables being compared.

| Target Variable | Predictor Variable | Recommended Test | Effect Size Metric |
| :--- | :--- | :--- | :--- |
| Categorical | Categorical | Chi-Square Test of Independence (if expected counts > 5) | **Cramer's V** |
| Categorical | Categorical | Fisher's Exact Test (if expected counts < 5) | **Odds Ratio** |
| Continuous | Categorical (2 groups) | Mann-Whitney U (non-parametric) | **Rank-Biserial Correlation** |
| Continuous | Categorical (>2 groups)| Kruskal-Wallis (non-parametric ANOVA)| **Eta-Squared ($\eta^2$)** |
| Empirical Dist | Empirical Dist | 2-sample Kolmogorov-Smirnov (KS) Test | **KS Statistic (D)** |

*Note: We prefer non-parametric tests (Mann-Whitney over T-Test) because real-world tabular data is rarely perfectly normally distributed.*

## 3. Multiple Testing Correction
When analyzing a dataset with 50 features and 3 protected attributes, the system will execute at least 150 hypothesis tests. Using a standard $\alpha = 0.05$, we expect ~7-8 false positives purely by chance.

* **Requirement**: Implement the **Benjamini-Hochberg (BH) False Discovery Rate (FDR)** procedure.
* **Workflow**:
  1. Collect all raw $p$-values from the profiling phase.
  2. Rank them from smallest to largest.
  3. Apply the BH step-up procedure to generate $q$-values (adjusted $p$-values).
  4. Use these adjusted values for all downstream severity calculations.

## 4. Effect Size Definitions

### A. Cramer's V (Categorical vs Categorical)
Measures association between two nominal variables.
$$ V = \sqrt{\frac{\chi^2}{n \cdot \min(k-1, r-1)}} $$
* Where $\chi^2$ is the Chi-square statistic, $n$ is sample size, $k$ is columns, $r$ is rows.
* **Interpretation**: $< 0.1$ (Trivial/Small), $0.1 - 0.3$ (Medium), $> 0.3$ (Large).

### B. Cohen's $w$ (Goodness of Fit)
Used for representation imbalance against a baseline.
$$ w = \sqrt{\sum_{i=1}^{m} \frac{(p_{0i} - p_{1i})^2}{p_{0i}}} $$
* **Interpretation**: $< 0.1$ (Small), $0.3$ (Medium), $> 0.5$ (Large).

### C. Rank-Biserial Correlation (Continuous vs Categorical, 2 groups)
Used alongside Mann-Whitney U.
$$ r = \frac{U_1}{n_1 n_2} - \frac{U_2}{n_1 n_2} $$
* **Interpretation**: Scale from -1 to 1. Magnitudes $> 0.3$ are considered practically significant.
