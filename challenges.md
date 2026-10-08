# Project Challenges and Technical Hurdles

This document outlines the unexpected scientific and engineering challenges encountered while building `dbias`.

## 1. The Marginal Skew Approximation Failure
**The Problem:** The analytic Minimum Detectable Effect (MDE) calculation for chi-square tests relies on a non-centrality parameter approximation (`n` and `df`). However, during the M3 calibration phase, we discovered this approximation errs *optimistically* on heavily skewed subgroups (e.g., a 95/5 group split). It treats Cohen's *w* over the total *n* as sufficient, ignoring that the smaller cell bottlenecks statistical power. 
**The Danger:** An optimistic MDE is fatal for our tool, as it would grant a false "adequate power" label and suppress blind spots.
**The Fix:** We implemented an empirical simulation fallback (`simulate_power_2x2` / `simulate_mde_2x2`) for 2x2 contingency tables when the expected cell counts are `< 5` or the margin ratio exceeds `4:1`. It draws binomial samples under the observed margins and runs the same uncorrected chi-square the audit runs, so it measures the power of the test actually performed. Tables larger than 2x2 still use the analytic approximation and are flagged `mde_is_approximate`.
**A bug on the way:** the first version searched for the MDE up to `w = 5`, which no 2x2 table can reach, so every simulated cell reported `MDE = 5.0`. The search now stops at the largest effect the margins allow.

## 2. "Deceptively Simple" Statistics
**The Problem:** Implementing effect sizes and confidence intervals is prone to subtle bugs. For example, standard library outputs for chi-square and effect sizes often don't properly handle one-sided vs two-sided bounding, or they pool variances incorrectly.
**The Fix:** Strict adherence to the `stats/` vs `detectability/` architectural boundary, and rigorous unit testing against known R/SciPy outputs.

## 3. The "Big-N" Benchmark Paradox
**The Problem:** The original hypothesis assumed that standard AI fairness benchmarks (Adult, COMPAS, German Credit) would easily yield "Blind Spots" when tested. Testing Adult (N=32k) with a SESOI of `w=0.1` found **0 blind spots**: the sample is large enough to fully power every test.
**The Pivot:** We subsampled Adult to `N=500` to simulate regional clinics or small startups, at which point 19 blind spots appeared.
**The Correction:** Once the feature-disparity analyzer was added, full-size COMPAS (N=7k) turned out to have **3 blind spots**, all on `vr_charge_degree` — a column recorded for only 819 rows, with two race groups of 4 people. The failure mode is a small *effective* N problem: it appears in large benchmarks wherever a feature is only recorded for a small subset. An earlier write-up reported 0 blind spots for COMPAS; that run predates the feature-disparity analyzer.

## 4. The Intersectional Compute and Multiplicity Explosion
**The Problem:** Generating depth-2 intersections (like `Race + Sex`) geometrically expands the number of hypotheses tested. Running an expensive 2000-resample bootstrap calculation for every tiny intersectional subgroup drastically slows down the audit, and simultaneously inflates the False Discovery Rate (FDR).
**The Fix:** We implemented "Power-Guided Descent" in M7. If a parent attribute (e.g., `Race`) lacks the power to detect an effect at the SESOI, and the intersectional child (e.g., `Race + Sex`) is itself underpowered, the tool skips the child's bootstrap and reports it as a `BLIND_SPOT` with no interval. This bounds compute time. It does not control FDR: intersections are still corrected as separate families, and hierarchical (Yekutieli) FDR is not implemented.
**Bugs on the way:** the first version reported a hardcoded `[0.0, 1.0]` interval for skipped children, as if it were measured, and skipped them even when the child itself was adequately powered. It also tested each attribute against intersection columns built from that same attribute — a column against a function of itself, which shows `V = 1.0` on any data — and turned missing values into a group called `"nan"`. All four are fixed and covered by regression tests.

## 5. Bootstrap Intervals Near the Null
**The Problem:** Cramér's V is bounded below by zero and biased upward near the null, and the percentile bootstrap inherits that bias. On the sparse multi-level tables the feature-disparity analyzer produces, the interval can sit entirely above the SESOI — and even above the observed V — while the test itself is not significant.
**The Danger:** The severity rules used to let such a finding fall back on the power calculation, so an adequately powered one would have been scored `INFORMATIONAL`, the same label as an earned all-clear, although its interval ruled nothing out.
**The Fix:** Only an `EQUIVALENT` interval earns the all-clear; any other interval on a non-significant finding is a `BLIND_SPOT`. **An attempted fix that failed its gate:** the Bergsma (2013) bias-corrected V with a bias-recentred bootstrap fixed the spurious disparities but pushed intervals too low: at a true effect equal to the SESOI it cleared up to 13.2% of tables (2x2 at n = 500: 12.4%) against a pre-registered 5% limit, and tripled false all-clears in the M3 grid. An interval that errs toward blind spots is safe; one that errs toward all-clears is not, so it was reverted. A correct interval for V near the null (e.g. inverting the non-central chi-square) remains open.
