# Bias Detection Methodology

This document outlines *how* the Analyzers utilize the Statistical Engine to detect specific categories of bias.

## 1. Representation Imbalance
**Concept**: Are certain groups severely under- or over-represented relative to an expected baseline (e.g., census data) or a uniform distribution?
* **Method**: Chi-Square Goodness-of-Fit test.
* **Inputs**:
  * Observed group counts (e.g., Male: 8000, Female: 2000)
  * Expected probabilities (e.g., [0.5, 0.5] for uniform)
* **Outputs**: $p$-value, Cohen's $w$.
* **Interpretation**: A high Cohen's $w$ indicates a severe skew. This does not necessarily mean "bias" in the societal sense, but rather a risk of poor model generalization for the minority group.

## 2. Missingness Bias (MNAR - Missing Not At Random)
**Concept**: Are data points missing at a systematically higher rate for specific protected groups?
* **Method**:
  1. For a target feature $X$, create a boolean mask $M$ where $M_i = 1$ if $X_i$ is NaN, else $0$.
  2. Run a Chi-Square test of independence between $M$ and the sensitive attribute $S$.
* **Outputs**: Adjusted $p$-value, Cramer's V.
* **Interpretation**: If significant, missingness is correlated with group membership. Imputing this data naively (e.g., mean imputation) could introduce severe bias.

## 3. Feature Disparity (Proxy Variables)
**Concept**: Do feature distributions differ significantly across groups? This detects potential "redundant encodings" where a supposedly neutral feature (like Zip Code) acts as a proxy for a protected attribute (like Race).
* **Continuous Features**: Mann-Whitney U or KS-Test.
* **Categorical Features**: Chi-Square Test.
* **Interpretation**: Large effect sizes here indicate that dropping the protected attribute from model training will *not* prevent the model from learning the protected group's identity.

## 4. Label Distribution Disparity
**Concept**: Does the ground truth label vary significantly depending on the protected attribute?
* **Method**: Treat the label as a target feature and apply the tests from Section 3.
* **Special Case**: For binary classification labels, this checks for the statistical equivalent of the **Four-Fifths Rule** (Disparate Impact).
  * Calculate success rates: $P(Y=1 | S=unprivileged)$ vs $P(Y=1 | S=privileged)$.
  * Ratio $= \frac{P(Y=1 | S=unprivileged)}{P(Y=1 | S=privileged)}$.

## 5. Conditional Imbalance (Intersectionality)
**Concept**: Imbalances that only appear when examining subgroups (e.g., "Black Females" vs just "Black" or "Female").
* **Method**: Generate composite categories (e.g., Gender_Race) and apply Representation and Label Disparity methodologies to the combinatorial space.
* **Caution**: This creates a combinatorial explosion. Careful application of Benjamini-Hochberg FDR is absolutely critical here to avoid overwhelming false positives.
