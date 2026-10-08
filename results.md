# Project Results and Scientific Claims

This document tracks the core scientific validations of the `dbias` audit pipeline as of Milestone M7.

Every number below comes from a seeded run (`seed=0`, SESOI `w=0.1`) of the
scripts in the repository root, so re-running them reproduces it exactly.

## 1. M3: Calibration Gate (Passed)
**Hypothesis:** The statistical inference layer must respect its nominal false positive rate (alpha) even under extreme marginal skew.
**Result** (`tests/calibration/run_m3_gate.py`, results in `tests/calibration/results/m3_calibration_results.csv`):
- 85,000 simulated datasets: 85 cells (n from 100 to 10,000, true w from 0 to 0.2, minority share 50% down to 1%) x 1,000 replicates, 2x2 tables only. Re-run after the fixes in this branch; detection, adequacy and false-all-clear rates were unchanged in every cell.
- **False positive rate** (true w = 0) averaged **5.2%** at `alpha = 0.05`, ranging from 0.2% (n=100 with a 1% minority, where the test is conservative) to 6.8%.
- **Power claims hold.** In all 51 cells where the tool declared itself adequately powered, it detected an effect at or above the SESOI at least 90.3% of the time, above the 80% target.
- **False all-clears** occurred only when the true effect sat exactly at the SESOI (136 of 15,000 such datasets; at most 3.2% per cell) and never elsewhere. That is the known type-I error of an equivalence test at its boundary (`docs/09` Sec 8).
- **Not covered:** n above 10,000 and tables larger than 2x2.
- **Meaning:** When the tool flags a disparity, its error rate stays close to the nominal 5%. Individual cells run up to 6.8% (about 2.5 standard errors above alpha at 1,000 replicates), but the mean over all 20 null cells, 5.2%, is within 1.5 standard errors of 5%, which is what Monte Carlo noise alone would produce.

## 2. M4 & M6: The "Headline Claim"
**Hypothesis:** Incumbent tools (AIF360, Fairlearn) certify small-N datasets as "fair" by misinterpreting `p > 0.05` (absence of evidence) as evidence of absence. `dbias` will flag these as "Blind Spots" when statistical power is too low to detect the Smallest Effect Size of Interest (SESOI).

### Known findings reproduce
The M6 gate requires the analyzers to recover the established disparities first:
- **Adult** — high-income rate by sex is 10.9% (Female) vs 30.6% (Male), V = 0.216, significant after correction.
- **COMPAS** — recidivism rate by race is 55.1% (African-American) vs 41.8% (Caucasian), V = 0.146, significant after correction.

### The "Big-N" Benchmark (Adult N=32,561 & COMPAS N=7,214)
| Dataset | Findings | Blind spots | High |
|---|---|---|---|
| Adult (`run_adult.py`) | 35 | **0** | 4 |
| COMPAS (`run_compas.py`) | 116 | **3** | 3 |

**Adult** has no blind spots at `w=0.1`: at N=32k every test is powered to see an effect of that size.

**COMPAS is not clean.** All three blind spots are on `vr_charge_degree` (by sex, by race, and by sex + race). That column is only populated for the 819 people with a violent recidivism charge, and within it two race groups have 4 people each. A full-size benchmark can therefore still hide a blind spot when a feature is only recorded for a small subset of rows. The original framing — *"the incumbent tools' failure mode is fundamentally a small-N problem"* — holds only if "small N" includes small effective N within a large dataset.

**The COMPAS cells do not meet the M6 gate.** Under a four-fifths criterion fixed before the run (every level of the feature must keep each group's rate within 80% of the highest; `benchmarks/compare_compas.py`), Fairlearn's demographic-parity metrics flag all three `vr_charge_degree` cells as *unfair* rather than clean. In the sex cell, level `(F1)` has rates of 1.9% for women and 5.0% for men (ratio 0.378) — 2 of 105 women against 36 of 714 men. In the race cell the same level ranges from 0% (Hispanic, Native American) to 25% (Asian: 1 of 4 people), ratio 0.0; in the sex + race cell the largest rate gap is 0.788. The incumbent does not certify these cells as clean; it raises alarms from counts this small, which is a different failure from the one M6 targets. The M6 headline therefore rests on the subsampled Adult cell (`MAR_WORKCLASS_RACE`).

The High findings are representation against a uniform reference (race and sex shares) and, for Adult, relationship by sex and by sex + race (husband/wife); all are expected properties of these datasets.

### The "Small-N" Simulation (Subsampled Adult, N=500)
**Result** (`run_subsampled.py`): **19 blind spots** out of 41 findings, and 12 High.
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

**Caveat on the intervals.** For several sparse feature tables the bootstrap interval on Cramér's V sits *above* the SESOI even though the test is not significant (e.g. `DISP_OCCUPATION_RACE`: V = 0.157, CI [0.170, 0.287]). Near the null, V is biased upward and the percentile bootstrap inherits that bias, so the lower bound carries no inference (see `stats/intervals.py`). These findings are correctly scored as blind spots — the interval did not rule out the SESOI — but their intervals should not be read as evidence *of* a disparity. A bias-corrected interval was tried and rejected: the Bergsma (2013) estimator with a bias-recentred bootstrap removed the spurious disparities (at most 2.3% of null tables, against up to 100% for the current interval) but over-corrected, clearing effects that sat exactly at the SESOI in up to 13.2% of tables — including 12.4% for ordinary 2x2 tables at n = 500 — against a pre-registered limit of 5%. Through the full pipeline it tripled false all-clears in the M3 grid (136 to 404 of 15,000). The current interval errs toward blind spots; the corrected one erred toward false all-clears, so it was reverted (`tests/calibration/run_interval_gate.py`, `results/interval_calibration_results.csv`).

**Caveat on magnitudes.** The same upward bias inflates the *point* estimate of V on sparse tables, so severity grades there run high. Of the 12 High findings at N=500, several are large tables such as `education` x `sex + race` (16 x 10 cells over 500 rows), where V would be about 0.17 with no association at all. On the full dataset `education` by race measures V = 0.075 (Low) against V = 0.265 (High) at N=500, and `education` by sex + race 0.062 against 0.234. (`native-country` appears only at N=500 because the full data has more than 20 countries, above the analyzer's level cap.) The bias-corrected point estimate is implemented (`cramers_v_corrected`) but not reported, because its interval failed the safety gate above; reporting a corrected point estimate beside an uncorrected interval would be inconsistent.

## 3. M7: Intersectional Gating (Partial)
**Hypothesis:** Intersectional subgroups (e.g., "Black Females") suffer from sample size collapse. Testing all intersections blindly inflates the number of hypotheses and wastes compute on tests that cannot reach adequate power.
**Result:**
- Depth-2 intersections (`correction/hierarchy.py`) and **power-guided descent** (`correction/gating.py`) are implemented.
- On the N=500 subsampled Adult run, the tool generated 13 intersectional findings (`sex + race`). For 5 of them the parent attribute was underpowered *and* the intersection itself was underpowered, so the bootstrap was skipped.
- Skipped intersections are reported as explicit blind spots with **no interval** (e.g. `MAR_WORKCLASS_SEX_AND_RACE`), rather than an invented one.

**FDR, measured** (`tests/calibration/run_fdr_gate.py`; 500 simulated audits per row, n = 2,000, two binary attributes, ten features; results in `tests/calibration/results/fdr_calibration_results.csv`). `family` is the default: BH within each (category, attribute) family, intersections included. `hierarchical` (opt-in, `fdr="hierarchical"`) tests an intersection only below a parent with a disparity.

| Scenario | Procedure | Worst per-family FDR (± SE) | Audit-wide FDR (± SE) | Intersection power | Parent power |
|---|---|---|---|---|---|
| global null | family | 0.042 (0.009) | 0.098 (0.013) | — | — |
| global null | hierarchical | **0.082 (0.012)** | 0.082 (0.012) | — | — |
| main effect | family | 0.032 (0.008) | 0.029 (0.003) | 1.000 | 1.000 |
| main effect | hierarchical | 0.032 (0.008) | 0.025 (0.003) | 1.000 | 1.000 |
| xor | family | 0.064 (0.011) | 0.040 (0.004) | 1.000 | — |
| xor | hierarchical | 0.064 (0.011) | 0.074 (0.010) | **0.017** | — |
| mixed | family | 0.042 (0.004) | 0.031 (0.003) | 1.000 | 1.000 |
| mixed | hierarchical | 0.042 (0.004) | 0.033 (0.003) | 0.516 | 1.000 |

**The pre-registered gate (every family's FDR ≤ α + 2 SE) passed for `family` and failed for `hierarchical`**: under the global null its intersection family reached 0.082 against a limit of 0.074. The reason is structural. An intersection's table refines its parent's, so the two tests are positively dependent; admitting an intersection only after a parent rejection selects exactly the data whose chance imbalance also makes the intersection look significant. Yekutieli's guarantee assumes independence between levels, which this tree does not have. FDR is controlled within each (category, attribute) family, not across the audit; the audit-wide column shows how far apart those are.

**The trade-off, measured.** In the `xor` scenario — missingness that depends on race × sex with no marginal disparity — the default per-family correction detected the intersection effect in 100% of audits; hierarchical correction in 1.7%, and its audit-wide FDR was higher, not lower (0.074 against 0.040). Hierarchical audits report those cells as blind spots, never as clean, but on this evidence the opt-in procedure buys a small drop in audit-wide FDR under the global null (0.098 to 0.082) at a large cost, and the default stays `family`.

**Meaning:** Gating bounds compute and keeps untested intersections visible as blind spots instead of quietly passing them. Per-family BH holds its FDR, including for intersections; nothing here controls FDR across the whole audit.
