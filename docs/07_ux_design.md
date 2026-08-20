# UX & Dashboard Design

The frontend should be built with **Streamlit** (or Dash) for rapid prototyping and Python integration. The design must prioritize readability and narrative over raw data dumps.

## 1. Global Layout

* **Sidebar**: 
  * File uploader (CSV/Parquet).
  * Global Configuration: Select `Target Label`, select `Protected Attributes` (multi-select).
  * Confidence Level slider ($\alpha$, default 0.05).
* **Main Area Tabs**:
  1. Executive Summary
  2. The Findings Feed
  3. Deep Dives (Representation, Missingness, Feature Disparities)
  4. Export Report

## 2. Core Views

### A. Executive Summary
* **Dataset Profile**: Basic stats ($N$ rows, $M$ columns).
* **The Risk Radar**: A Plotly Radar/Spider chart visualizing the Risk Vector (axes: Representation, Missingness, Label Disparity, Proxy Features).
* **Call to Action**: A high-level NLG summary (e.g., "Critical disparities found in the Target Label. High risk of Missingness Bias in 'Income'").

### B. The Findings Feed (Crucial Component)
A prioritized feed of `Finding` cards, sorted by severity (Critical $\to$ Informational).

**Card Design (Expandable UI)**:
* **Header**: [Severity Icon] [Category] - [Short Description] (e.g., 🔴 *Critical: Label Disparity across Gender*).
* **Expanded View**:
  * *The Evidence*: A small Plotly bar chart showing the disparity.
  * *The Math*: "Chi-Square $p < 0.01$, Cramer's V: 0.35 (Large Effect)".
  * *The Impact*: "Models trained on this data may heavily penalize the minority group."
  * *Recommendations*: "Investigate historical decision-making processes for label assignment."

### C. Deep Dives
For data scientists who want to verify the findings.
* **Feature Disparities Tab**: 
  * User selects a feature. 
  * UI renders **KDE plots** (for continuous) or stacked bar charts (for categorical) partitioned by the protected attribute. 
  * Displays Empirical CDFs (Cumulative Distribution Functions) alongside KS-Test statistics to visually prove distance between distributions.
