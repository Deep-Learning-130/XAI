# Risk & Severity Framework

This document defines how raw statistical findings are escalated into actionable alerts. Severity is assigned by the Rules Engine, not by the Analyzers.

**Core Principle**: Severity is a function of both **Statistical Significance** (Adjusted $p$-value $< \alpha$) AND **Effect Size Magnitude**.

## 1. Severity Levels & Thresholds

Let $\alpha = 0.05$ (after BH-FDR correction).
Let $E$ be the effect size (e.g., Cramer's V, Rank-Biserial).

| Severity Level | Definition / Logic | Action Required |
| :--- | :--- | :--- |
| **Informational** | $p \ge \alpha$ OR $E$ is "Small" ($E < 0.1$). Imbalance exists but is likely statistical noise or trivially small. | None. Documented for transparency. |
| **Low** | $p < \alpha$ AND $E$ is "Small/Medium" ($0.1 \le E < 0.15$). | Review if related to highly sensitive features. |
| **Medium** | $p < \alpha$ AND $E$ is "Medium" ($0.15 \le E < 0.25$). | Consider investigating feature collection methodologies. |
| **High** | $p < \alpha$ AND $E$ is "Large" ($E \ge 0.25$) on any non-label feature. | Strong risk of proxy variables or MNAR. Must be addressed before modeling. |
| **Critical** | High severity on the **Target Label** OR violates legal thresholds (e.g., Disparate Impact $< 0.8$). | STOP. Training a model on this data *will* result in biased outcomes. |

*Note: Effect size thresholds ($0.1, 0.15, 0.25$) are standard starting points based on Cohen's rules of thumb, but should be configurable by the user.*

## 2. The Risk Vector
Do NOT output a single "Dataset Bias Score" (e.g., "This dataset is 80% biased"). This is mathematically meaningless and misleading.

Instead, output a **Risk Vector** that summarizes the highest severity finding in each category:
* **Representation Risk**: `[Medium]`
* **Missingness Risk**: `[High]`
* **Label Disparity Risk**: `[Critical]`
* **Feature Proxy Risk**: `[Low]`

This allows data scientists to pinpoint exactly which part of the data pipeline requires intervention.
