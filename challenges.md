# Project Challenges and Technical Hurdles

This document outlines the unexpected scientific and engineering challenges encountered while building `dbias`.

## 1. The Marginal Skew Approximation Failure
**The Problem:** The analytic Minimum Detectable Effect (MDE) calculation for chi-square tests relies on a non-centrality parameter approximation (`n` and `df`). However, during the M3 calibration phase, we discovered this approximation errs *optimistically* on heavily skewed subgroups (e.g., a 95/5 group split). It treats Cohen's *w* over the total *n* as sufficient, ignoring that the smaller cell bottlenecks statistical power. 
**The Danger:** An optimistic MDE is fatal for our tool, as it would grant a false "adequate power" label and suppress blind spots.
**The Fix:** We implemented a rigorous empirical simulation fallback (`simulate_mde_2x2`) using exact binomial tests for 2x2 contingency tables when the expected cell counts are `< 5` or the group ratio exceeds `4:1`.

## 2. "Deceptively Simple" Statistics
**The Problem:** Implementing effect sizes and confidence intervals is prone to subtle bugs. For example, standard library outputs for chi-square and effect sizes often don't properly handle one-sided vs two-sided bounding, or they pool variances incorrectly.
**The Fix:** Strict adherence to the `stats/` vs `detectability/` architectural boundary, and rigorous unit testing against known R/SciPy outputs.

## 3. The "Big-N" Benchmark Paradox
**The Problem:** The original hypothesis assumed that standard AI fairness benchmarks (Adult, COMPAS, German Credit) would easily yield "Blind Spots" when tested. However, upon testing Adult (N=32k) and COMPAS (N=7k) with a SESOI of `w=0.1`, we found **0 blind spots**. The sample sizes were large enough to fully power the tests, rendering the confidence intervals extremely tight.
**The Pivot:** We realized the incumbent tools' failure mode (reporting "no disparity" due to absence of evidence) is strictly a **Small-N problem**. We had to artificially subsample the datasets to `N=500` to simulate regional clinics or small startups, at which point the blind spots were successfully exposed.
