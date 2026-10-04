# Project Results and Scientific Claims

This document tracks the core scientific validations of the `dbias` audit pipeline as of Milestone M7.

Every number below comes from a seeded run (`seed=0`, SESOI `w=0.1`) of the
scripts in the repository root, so re-running them reproduces it exactly.

## 1. M3: Calibration Gate (Passed)
**Hypothesis:** The statistical inference layer must respect its nominal false positive rate (alpha) even under extreme marginal skew.
**Result** (`tests/calibration/run_m3_gate.py`, results in `tests/calibration/results/m3_calibration_results.csv`):
<!-- M3_RESULTS -->
- **Meaning:** When the tool flags a disparity, it holds its nominal error rate. It does not falsely accuse clean datasets at more than the rate it advertises.

## 2. M4 & M6: The "Headline Claim"
**Hypothesis:** Incumbent tools (AIF360, Fairlearn) certify small-N datasets as "fair" by misinterpreting `p > 0.05` (absence of evidence) as evidence of absence. `dbias` will flag these as "Blind Spots" when statistical power is too low to detect the Smallest Effect Size of Interest (SESOI).

### Known findings reproduce
The M6 gate requires the analyzers to recover the established disparities first:
- **Adult** — high-income rate by sex is 10.9% (Female) vs 30.6% (Male), V = 0.216, significant after correction.
- **COMPAS** — recidivism rate by race is 55.1% (African-American) vs 41.8% (Caucasian), V = 0.146, significant after correction.

### The "Big-N" Benchmark (Adult N=32,561 & COMPAS N=7,214)
| Dataset | Findings | Blind spots | High |
|---|---|---|---|
| Adult (`run_adult.py`) | 35 | **0** | 3 |
| COMPAS (`run_compas.py`) | 116 | **3** | 3 |

**Adult** has no blind spots at `w=0.1`: at N=32k every test is powered to see an effect of that size.

**COMPAS is not clean.** All three blind spots are on `vr_charge_degree` (by sex, by race, and by sex + race). That column is only populated for the 819 people with a violent recidivism charge, and within it two race groups have 4 people each. A full-size benchmark can therefore still hide a blind spot when a feature is only recorded for a small subset of rows. The original framing — *"the incumbent tools' failure mode is fundamentally a small-N problem"* — holds only if "small N" includes small effective N within a large dataset.

These are candidates for the M6 headline cell, but they do not yet meet the gate: the gate requires running the incumbent tool on the cell, and Fairlearn has no test for the distribution of a multi-level categorical feature, so that comparison is still to be designed.

The High findings are representation against a uniform reference (race and sex shares) and, for Adult, relationship by sex (husband/wife); all are expected properties of these datasets.

### The "Small-N" Simulation (Subsampled Adult, N=500)
**Result** (`run_subsampled.py`): **19 blind spots** out of 41 findings.
- 4 missingness (MAR): `workclass`, `occupation` and `native-country` by race; `native-country` by sex.
- 10 feature disparity: e.g. `education`, `workclass`, `occupation` and `capital-loss` by sex or race.
- 5 intersectional (`sex + race`), skipped by power-guided descent (Section 3).

`education` and `education-num` are the same variable encoded twice in Adult, so they appear as two blind spots.

In the `MAR_WORKCLASS_RACE` missingness test, we ran both Fairlearn's `demographic_parity_difference` and `dbias.audit` side by side (`compare_fairlearn.py`):

**Fairlearn Output:**
```text
Demographic Parity Difference: 0.167
Conclusion a practitioner draws: 'The maximum difference in missingness rates is 16.7%. The groups are small, so it's not statistically significant. We pass the fairness check!'
```

**dbias Output:**
```text
Severity: BLIND_SPOT
Detectability: UNDERPOWERED
Confidence Interval for Effect Size: [0.052, 0.207]
Verdict: inconclusive
Conclusion dbias draws: 'BLIND SPOT. You do not have the power to claim this is fair. The CI straddles the SESOI of 0.1.'
```

**Meaning:** A `p > 0.05`, or a difference reported without power bounds, does not show a dataset is fair. The sample was too small to rule out an effect as large as 0.207, and `dbias` refuses the false certification.

**Caveat on the intervals.** For several sparse feature tables the bootstrap interval on Cramér's V sits *above* the SESOI even though the test is not significant (e.g. `DISP_OCCUPATION_RACE`: V = 0.157, CI [0.170, 0.287]). Near the null, V is biased upward and the percentile bootstrap inherits that bias, so the lower bound carries no inference (see `stats/intervals.py`). These findings are correctly scored as blind spots — the interval did not rule out the SESOI — but their intervals should not be read as evidence *of* a disparity. A bias-corrected interval is future work.

## 3. M7: Intersectional Gating (Partial)
**Hypothesis:** Intersectional subgroups (e.g., "Black Females") suffer from sample size collapse. Testing all intersections blindly inflates the number of hypotheses and wastes compute on tests that cannot reach adequate power.
**Result:**
- Depth-2 intersections (`correction/hierarchy.py`) and **power-guided descent** (`correction/gating.py`) are implemented.
- On the N=500 subsampled Adult run, the tool generated 13 intersectional findings (`sex + race`). For 5 of them the parent attribute was underpowered *and* the intersection itself was underpowered, so the bootstrap was skipped.
- Skipped intersections are reported as explicit blind spots with **no interval** (e.g. `MAR_WORKCLASS_SEX_AND_RACE`), rather than an invented one.

**Not yet done** — the roadmap's "done when" criteria for M7 are not all met:
- Hierarchical FDR (Yekutieli-style) is **not implemented**. Each intersection is corrected as its own (category, attribute) family, the same rule as for parent attributes, so FDR is controlled within a family but not across the audit.
- The audit's FDR with intersections has **not been verified empirically** in the M3 harness.

**Meaning:** Gating bounds compute and keeps untested intersections visible as blind spots instead of quietly passing them. It does not, by itself, control multiplicity.
