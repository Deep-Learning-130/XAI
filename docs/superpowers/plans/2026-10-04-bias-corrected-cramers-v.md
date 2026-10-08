# Bias-Corrected Cramér's V — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the plug-in Cramér's V with the bias-corrected estimator of Bergsma (2013), for both the point estimate and the bootstrap interval, so that sparse tables stop reporting inflated effects and intervals that sit above the SESOI under the null.

**Architecture:** The estimator lives in `stats/effect_sizes.py` as a scalar function plus a vectorised batch version (the bootstrap needs thousands of evaluations). `stats/intervals.py` resamples as before but evaluates the corrected estimator and recentres the percentile interval by the bootstrap's estimated bias. `chi_square_test` reports the corrected V, so every analyzer picks it up without per-analyzer changes. A new interval calibration harness verifies the change empirically on tables larger than 2x2 before anything downstream is trusted.

**Tech Stack:** Python 3.11+, NumPy, SciPy, pandas, pytest.

**Spec:** [results.md](../../../results.md) §2 ("Caveat on the intervals", "Caveat on magnitudes"), [challenges.md](../../../challenges.md) §5, [stats/intervals.py](../../../src/dbias/stats/intervals.py) module docstring, [plan.md](../../../plan.md) §3.2 (the interval is the inference layer). Estimator: Bergsma, W. (2013). "A bias-correction for Cramér's V and Tschuprow's T." *Journal of the Korean Statistical Society* 42(3), 323–328.

**Why this matters, in one paragraph.** Under independence, E[chi²] ≈ (r−1)(k−1), so the plug-in V is about sqrt((r−1)(k−1) / (n·min(r−1, k−1))) even when there is no association at all — 0.17 for a 16x2 table at n = 500. Seeded re-runs showed `education` by race grading HIGH at N = 500 (V = 0.265) and LOW on the full data (V = 0.075), and nine non-significant findings whose 95% interval sat entirely above the SESOI. One (`DISP_OCCUPATION_RACE`, V = 0.157, CI [0.170, 0.287]) did not even contain its own point estimate.

## Global Constraints

- **Python 3.11 minimum.**
- **Test oracles must be independent.** Assert against hand-computed closed forms or simulation against a known population value — never against this code's own output.
- **Effect sizes are mandatory.** No function returns a p-value without an effect size.
- **alpha = 0.05 and target power = 0.80 are parameters with those defaults.** Never inline them.
- **The SESOI is declared on the Cohen's w scale.** Anything compared against it on the V scale uses `v_from_w(sesoi, df_min)`.
- **Do not change Cohen's w for goodness of fit** (`chi_square_goodness_of_fit`, `bootstrap_ci_gof_w`). Representation is out of scope.
- **Do not loosen a safety test to make it pass.** If a test in `tests/unit/detectability/test_classify.py`, `tests/unit/rules/test_severity.py`, `tests/unit/stats/test_intervals.py::test_small_n_null_interval_cannot_exclude_a_sesoi_of_one_tenth`, `tests/synthetic/` or `tests/calibration/` fails after a change, the correction has moved a verdict. Stop and report; do not edit the assertion.
- **Commands run from the repo root on Windows, in Git Bash** (`python -m pytest ...`). Do not paste the `python -c "..."` one-liners into PowerShell 5.1: it strips the inner quotes.

## Review Focus

1. **Tiny tables (n = 2 or 3).** The correction's denominator `min(r̃, k̃) − 1` can reach 0 or below. Expected: V = 0.0, never `ZeroDivisionError` or NaN. Pinned in Task 1.
2. **A bootstrap resample with an all-zero row or column.** Rare categories vanish in some resamples. Expected: a finite value in [0, 1], no NaN. Pinned in Task 1 (batch test).
3. **Shifting the interval down makes all-clears easier.** A corrected interval that is too low turns real effects into false all-clears. Expected: at a true effect equal to the SESOI, the EQUIVALENT rate stays ≤ 5% in every calibrated cell. Pinned by the Task 4 gate.
4. **Small-n null must still be a blind spot.** Expected: `[[25,25],[25,25]]`-sized nulls still cannot exclude w = 0.1. Pinned by the existing `test_small_n_null_interval_cannot_exclude_a_sesoi_of_one_tenth`, kept unchanged in Task 2.
5. **Perfect association is still 1.0.** Expected: the correction never pushes a perfect table below 1 or above 1. Pinned in Task 1.

---

### Task 1: The corrected estimator

**Files:**
- Modify: `src/dbias/stats/effect_sizes.py` (append after `cramers_v`)
- Modify: `src/dbias/models/enums.py:28-34` (`EffectSizeMetric`)
- Modify: `src/dbias/stats/thresholds.py:31` (`_DF_ADJUSTED`)
- Test: `tests/unit/stats/test_effect_sizes.py`

**Interfaces:**
- Produces: `cramers_v_corrected(table: ArrayLike) -> float` and `cramers_v_corrected_many(tables: np.ndarray) -> np.ndarray` (input shape `(B, r, k)`, output shape `(B,)`), both in `dbias.stats.effect_sizes`; `EffectSizeMetric.CRAMERS_V_CORRECTED = "Cramer's V (bias-corrected)"`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/stats/test_effect_sizes.py` (and add `cramers_v_corrected, cramers_v_corrected_many` to the existing import from `dbias.stats.effect_sizes`):

```python
# --- bias-corrected V (Bergsma 2013) ------------------------------------------
#
# phi2~ = max(0, chi2/n - (k-1)(r-1)/(n-1)),  r~ = r - (r-1)^2/(n-1),
# k~ = k - (k-1)^2/(n-1),  V~ = sqrt(phi2~ / min(k~-1, r~-1)).
#
# BALANCED_2X2: n = 60, phi2 = 1/9, phi2~ = 1/9 - 1/59 = 50/531,
# min(r~, k~) - 1 = 58/59, so V~^2 = (50/531)(59/58) = 50/522 = 25/261.

def test_corrected_v_matches_hand_computed_2x2():
    assert cramers_v_corrected(BALANCED_2X2) == pytest.approx(5 / np.sqrt(261), abs=1e-12)


def test_corrected_v_is_zero_under_exact_independence():
    assert cramers_v_corrected(INDEPENDENT_2X3) == pytest.approx(0.0, abs=1e-12)


def test_corrected_v_keeps_perfect_association_at_one():
    """PERFECT_3X3: phi2 = 2, phi2~ = 2 - 4/59 = 114/59, r~ - 1 = 114/59."""
    assert cramers_v_corrected(PERFECT_3X3) == pytest.approx(1.0, abs=1e-12)


def test_corrected_v_is_zero_when_the_correction_leaves_nothing_to_normalise():
    """n = 2: r~ = 2 - 1/1 = 1, so the denominator is 0. No effect is measurable."""
    assert cramers_v_corrected(np.array([[1, 0], [0, 1]])) == 0.0


def test_corrected_v_needs_two_observations():
    with pytest.raises(ValueError, match="at least two"):
        cramers_v_corrected(np.array([[1, 0], [0, 0]]))


def test_corrected_v_is_nearly_unbiased_under_independence():
    """The bias being fixed: for a 16x2 null at n = 500, plug-in V averages
    about sqrt(15/500) = 0.17. The corrected estimator is truncated at 0, so
    it keeps a small positive bias, but well under half the plug-in's."""
    rng = np.random.default_rng(0)
    draws = rng.multinomial(500, np.full(32, 1 / 32), size=400).reshape(400, 16, 2)
    raw = np.mean([cramers_v(t) for t in draws])
    corrected = float(np.mean(cramers_v_corrected_many(draws)))
    assert raw > 0.15
    assert corrected < 0.5 * raw


def test_batch_matches_scalar_and_survives_an_empty_row():
    """Bootstrap resamples can lose a rare category entirely."""
    tables = np.array([[[10, 20], [20, 10]], [[0, 0], [5, 7]], [[3, 9], [8, 1]]])
    batch = cramers_v_corrected_many(tables)
    assert np.all(np.isfinite(batch)) and np.all((0 <= batch) & (batch <= 1))
    assert batch[0] == pytest.approx(cramers_v_corrected(tables[0]), abs=1e-12)
    assert batch[2] == pytest.approx(cramers_v_corrected(tables[2]), abs=1e-12)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/unit/stats/test_effect_sizes.py -q`
Expected: collection ERROR, `ImportError: cannot import name 'cramers_v_corrected'`.

- [ ] **Step 3: Implement**

Append to `src/dbias/stats/effect_sizes.py`:

```python
def cramers_v_corrected_many(tables: np.ndarray) -> np.ndarray:
    """Bias-corrected Cramer's V over a stack of same-shape tables, (B, r, k).

    Vectorised because the bootstrap evaluates it thousands of times. The
    shape (r, k) is the table's, not its non-empty margins', so every resample
    of one table is corrected the same way. Expects every table to hold at
    least two observations; :func:`cramers_v_corrected` validates that.
    """
    t = np.asarray(tables, dtype=float)
    _, r, k = t.shape
    n = t.sum(axis=(1, 2))
    rows = t.sum(axis=2)
    cols = t.sum(axis=1)
    expected = rows[:, :, None] * cols[:, None, :] / n[:, None, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.where(expected > 0, (t - expected) ** 2 / expected, 0.0).sum(axis=(1, 2))
    phi2 = np.maximum(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    denom = np.minimum(k - (k - 1) ** 2 / (n - 1), r - (r - 1) ** 2 / (n - 1)) - 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(denom > 0, np.sqrt(phi2 / denom), 0.0)
    return np.minimum(v, 1.0)


def cramers_v_corrected(table: ArrayLike) -> float:
    """Bergsma (2013) bias-corrected Cramer's V. Bounded [0, 1].

    The plug-in V is biased upward: under independence E[chi2] is about
    (r-1)(k-1), so a 16x2 table at n = 500 shows V near 0.17 with no
    association at all. This subtracts the null expectation from phi^2 and
    shrinks r and k to match, which is what every finding now reports.
    """
    arr = _as_table(table)
    if min(arr.shape) < 2:
        raise ValueError("Cramer's V is undefined for a table with a single row or column")
    if arr.sum() < 2:
        raise ValueError("the bias correction needs at least two observations")
    return float(cramers_v_corrected_many(arr[None, :, :])[0])
```

In `src/dbias/models/enums.py`, add to `EffectSizeMetric` directly under `CRAMERS_V`:

```python
    CRAMERS_V_CORRECTED = "Cramer's V (bias-corrected)"
```

In `src/dbias/stats/thresholds.py`, replace the `_DF_ADJUSTED` line with:

```python
_DF_ADJUSTED = {
    EffectSizeMetric.CRAMERS_V: EffectSizeMetric.COHENS_W,
    EffectSizeMetric.CRAMERS_V_CORRECTED: EffectSizeMetric.COHENS_W,
}
```

- [ ] **Step 4: Run them to verify they pass**

Run: `python -m pytest tests/unit/stats -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/dbias/stats/effect_sizes.py src/dbias/models/enums.py src/dbias/stats/thresholds.py tests/unit/stats/test_effect_sizes.py
git commit -m "feat(stats): Bergsma bias-corrected Cramer's V, scalar and batched"
```

---

### Task 2: The bias-corrected bootstrap interval

**Files:**
- Modify: `src/dbias/stats/intervals.py:1-51` (module docstring and `bootstrap_ci_cramers_v`)
- Test: `tests/unit/stats/test_intervals.py`

**Interfaces:**
- Consumes: `cramers_v_corrected`, `cramers_v_corrected_many` from Task 1.
- Produces: `bootstrap_ci_cramers_v(table, n_resamples=DEFAULT_RESAMPLES, confidence=0.95, seed=None) -> tuple[float, float]` — same signature as today; the interval is now for the corrected V.

- [ ] **Step 1: Write the failing tests**

In `tests/unit/stats/test_intervals.py`, change the import line `from dbias.stats.effect_sizes import cramers_v` to `from dbias.stats.effect_sizes import cramers_v_corrected`, change `test_interval_brackets_the_point_estimate_for_a_strong_effect` to compare against `cramers_v_corrected(STRONG_2X2)`, and append:

```python
def _sparse_null(seed: int = 0) -> np.ndarray:
    """A 14x5 table at n = 500 drawn under exact independence -- the shape of
    the Adult occupation-by-race table that exposed the bias."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(500, np.full(70, 1 / 70)).reshape(14, 5)


def test_sparse_null_interval_contains_its_point_estimate():
    """Regression: DISP_OCCUPATION_RACE reported V = 0.157 with CI [0.170, 0.287]."""
    table = _sparse_null()
    lo, hi = bootstrap_ci_cramers_v(table, n_resamples=2000, seed=1)
    assert lo <= cramers_v_corrected(table) <= hi


def test_sparse_null_interval_does_not_sit_above_the_sesoi():
    """Regression: nine non-significant findings had a lower bound above the
    SESOI. For a 14x5 table the SESOI w = 0.1 is V = 0.1 / sqrt(4) = 0.05."""
    lo, _ = bootstrap_ci_cramers_v(_sparse_null(), n_resamples=2000, seed=1)
    assert lo < 0.05
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/unit/stats/test_intervals.py -q`
Expected: the two new tests FAIL (the current percentile interval on plug-in V sits above both bounds).

- [ ] **Step 3: Implement**

Replace `bootstrap_ci_cramers_v` in `src/dbias/stats/intervals.py`:

```python
def bootstrap_ci_cramers_v(
    table: ArrayLike,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float]:
    """Bias-corrected percentile bootstrap for the bias-corrected Cramer's V.

    Resamples are drawn from the observed cell shares, whose own association
    is the *plug-in* V. The corrected estimator is close to unbiased for that,
    so the resamples centre on the plug-in V rather than on the corrected
    point estimate. Shifting the percentile interval back by the bootstrap's
    estimated bias recentres it on the estimate it is an interval for.
    """
    arr = _as_table(table)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")

    n = int(arr.sum())
    point = cramers_v_corrected(arr)
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(n, (arr / n).ravel(), size=n_resamples)
    estimates = cramers_v_corrected_many(draws.reshape(n_resamples, *arr.shape))

    bias = float(estimates.mean()) - point
    tail = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(estimates, [tail, 1.0 - tail]) - bias
    return (float(np.clip(lo, 0.0, 1.0)), float(np.clip(hi, 0.0, 1.0)))
```

Change its import to `from dbias.stats.effect_sizes import _as_table, cramers_v_corrected, cramers_v_corrected_many`.

Replace the "Method:" paragraph of the module docstring with:

```text
Method: non-parametric bootstrap over the multinomial defined by the observed
cell proportions, holding n fixed. Contingency tables use the Bergsma (2013)
bias-corrected Cramer's V and a percentile interval recentred by the
bootstrap's estimated bias; the plug-in V is biased upward near the null,
which used to put whole intervals above the SESOI on sparse tables. The
estimator is still truncated at zero, so near the null the lower bound is
reported for completeness and the upper bound carries the inference.
```

- [ ] **Step 4: Run the stats and detectability tests**

Run: `python -m pytest tests/unit/stats tests/unit/detectability -q`
Expected: all pass, including the unchanged `test_small_n_null_interval_cannot_exclude_a_sesoi_of_one_tenth` and `test_large_n_null_interval_excludes_a_sesoi_of_one_tenth`. If either of those fails, stop: see Global Constraints.

- [ ] **Step 5: Commit**

```bash
git add src/dbias/stats/intervals.py tests/unit/stats/test_intervals.py
git commit -m "feat(stats): bias-corrected bootstrap interval for corrected V"
```

---

### Task 3: Report the corrected V everywhere

**Files:**
- Modify: `src/dbias/stats/chi_square.py:49` (field comment), `:70-97` (`chi_square_test`)
- Modify: `src/dbias/analyzers/missingness.py:62`, `src/dbias/analyzers/feature_disparity.py:59`, `src/dbias/analyzers/label_disparity.py:54` (`effect_size_metric=`)
- Test: `tests/unit/stats/test_chi_square.py:50`, `tests/unit/analyzers/test_analyzers.py`

**Interfaces:**
- Consumes: `cramers_v_corrected`, `EffectSizeMetric.CRAMERS_V_CORRECTED` from Task 1.
- Produces: `ChiSquareResult.effect_size` is the corrected V; every contingency-table finding has `effect_size_metric == EffectSizeMetric.CRAMERS_V_CORRECTED`.

- [ ] **Step 1: Write the failing tests**

In `tests/unit/stats/test_chi_square.py`, change line 50 to the corrected oracle (the plug-in value 1/3 stays pinned by `test_cramers_v_equals_cohens_w_for_2x2` in `test_effect_sizes.py`):

```python
    assert chi_square_test(BALANCED_2X2).effect_size == pytest.approx(5 / np.sqrt(261), abs=1e-12)
```

Append to `tests/unit/analyzers/test_analyzers.py`:

```python
def test_contingency_findings_report_the_bias_corrected_v(frame):
    from dbias.models.enums import EffectSizeMetric

    hypotheses = MissingnessAnalyzer().analyze(frame, ["gender"])
    assert hypotheses
    assert all(
        h.finding.effect_size_metric is EffectSizeMetric.CRAMERS_V_CORRECTED
        for h in hypotheses
    )
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/unit/stats/test_chi_square.py tests/unit/analyzers -q`
Expected: 2 FAIL (effect size is still 1/3; metric is still `CRAMERS_V`).

- [ ] **Step 3: Implement**

In `src/dbias/stats/chi_square.py`: import `cramers_v_corrected` alongside `cramers_v`; in `chi_square_test` set `effect_size=cramers_v_corrected(arr),`; change the field comment on line 49 to `effect_size: float  # bias-corrected Cramer's V (Bergsma 2013) for independence; Cohen's w for GoF`; and add this sentence to the end of the `chi_square_test` docstring: `The effect size is the Bergsma (2013) bias-corrected V; the plug-in V overstates association on sparse tables.`

In each of the three analyzers, change `effect_size_metric=EffectSizeMetric.CRAMERS_V,` to `effect_size_metric=EffectSizeMetric.CRAMERS_V_CORRECTED,`.

- [ ] **Step 4: Run the full fast suite**

Run: `python -m pytest -m "not slow" -q`
Expected: all pass. A failure in `test_classify.py`, `test_severity.py` or `tests/synthetic/` means a verdict moved: stop and report it (Global Constraints).

- [ ] **Step 5: Commit**

```bash
git add src/dbias/stats/chi_square.py src/dbias/analyzers tests/unit/stats/test_chi_square.py tests/unit/analyzers/test_analyzers.py
git commit -m "feat: findings report the bias-corrected Cramer's V"
```

---

### Task 4: Interval calibration gate for tables larger than 2x2

The M3 gate covers 2x2 only, and the bias this plan fixes is worst on large sparse tables. This task is the empirical check that the corrected interval is safe before any result is reported. The criteria below are fixed **before** the run.

**Gate criteria (pre-registered):**
- **G1 — no new false all-clears.** In every reachable cell with true w = SESOI, the share of replicates whose interval upper bound falls below the SESOI (an EQUIVALENT verdict) is **≤ 0.05**.
- **G2 — no spurious disparities.** In every cell with true w = 0, the share whose interval lower bound exceeds the SESOI is **≤ 0.05**.
- **Reported, not gated:** coverage of the true V, the mean corrected and plug-in estimates, and the same G2 rate for the old plug-in percentile interval (to show what changed).

**Files:**
- Create: `tests/calibration/interval_harness.py`
- Create: `tests/calibration/run_interval_gate.py`
- Create: `tests/calibration/test_interval_calibration.py`

**Interfaces:**
- Consumes: `bootstrap_ci_cramers_v`, `cramers_v_corrected`, `cramers_v` (plug-in, for comparison).
- Produces: `joint_probs(row_shares, col_shares, true_w) -> np.ndarray | None`, `run_interval_cell(row_shares, col_shares, n, true_w, *, sesoi=0.1, alpha=0.05, replicates=1000, n_resamples=200, seed=0) -> dict`.

- [ ] **Step 1: Write the harness**

Create `tests/calibration/interval_harness.py`:

```python
"""Empirical calibration of the contingency-table interval beyond 2x2.

A table with chosen margins and Cohen's w is built as a rank-one departure
from independence: p_ij = r_i c_j (1 + w u_i v_j), where u and v are
contrasts with zero mean and unit variance under the row and column shares.
Then the margins are exactly r and c, and

    w^2 = sum_ij (p_ij - r_i c_j)^2 / (r_i c_j) = w^2 (sum r u^2)(sum c v^2) = w^2,

so the population Cramer's V is w / sqrt(min(R, C) - 1). Where some p_ij
would go negative the cell is unreachable and is skipped, never clipped.
"""
import numpy as np

from dbias.stats.effect_sizes import cramers_v, cramers_v_corrected, v_from_w
from dbias.stats.intervals import bootstrap_ci_cramers_v


def _contrast(shares: np.ndarray) -> np.ndarray:
    u = np.arange(len(shares), dtype=float)
    u -= (shares * u).sum()
    return u / np.sqrt((shares * u**2).sum())


def joint_probs(row_shares, col_shares, true_w: float) -> np.ndarray | None:
    r = np.asarray(row_shares, dtype=float)
    c = np.asarray(col_shares, dtype=float)
    p = np.outer(r, c) * (1.0 + true_w * np.outer(_contrast(r), _contrast(c)))
    return None if np.any(p < 0) else p


def run_interval_cell(
    row_shares,
    col_shares,
    n: int,
    true_w: float,
    *,
    sesoi: float = 0.1,
    alpha: float = 0.05,
    replicates: int = 1000,
    n_resamples: int = 200,
    seed: int = 0,
) -> dict:
    p = joint_probs(row_shares, col_shares, true_w)
    if p is None:
        raise ValueError("unreachable cell")
    shape = p.shape
    df_min = min(shape) - 1
    true_v = v_from_w(true_w, df_min)
    sesoi_v = v_from_w(sesoi, df_min)

    rng = np.random.default_rng(seed)
    covered = equivalent = spurious = raw_spurious = usable = 0
    corrected_sum = raw_sum = 0.0
    for _ in range(replicates):
        table = rng.multinomial(n, p.ravel()).reshape(shape)
        if min((table.sum(axis=1) > 0).sum(), (table.sum(axis=0) > 0).sum()) < 2:
            continue  # a degenerate draw is not a test
        usable += 1
        lo, hi = bootstrap_ci_cramers_v(
            table, n_resamples=n_resamples, confidence=1.0 - alpha,
            seed=int(rng.integers(1 << 31)),
        )
        covered += lo <= true_v <= hi
        equivalent += hi < sesoi_v
        spurious += lo > sesoi_v
        corrected_sum += cramers_v_corrected(table)
        raw = cramers_v(table)
        raw_sum += raw
        # The old interval: plain percentile bootstrap of the plug-in V.
        draws = rng.multinomial(n, (table / n).ravel(), size=n_resamples)
        raw_lo = np.quantile([cramers_v(d.reshape(shape)) for d in draws], alpha / 2)
        raw_spurious += raw_lo > sesoi_v

    return {
        "shape": f"{shape[0]}x{shape[1]}",
        "n": n,
        "true_w": true_w,
        "true_v": true_v,
        "replicates": usable,
        "coverage": covered / usable,
        "equivalent_rate": equivalent / usable,
        "spurious_disparity_rate": spurious / usable,
        "raw_spurious_disparity_rate": raw_spurious / usable,
        "mean_corrected_v": corrected_sum / usable,
        "mean_raw_v": raw_sum / usable,
    }
```

- [ ] **Step 2: Write the slow smoke test**

Create `tests/calibration/test_interval_calibration.py`:

```python
"""Reduced interval calibration: two cells, 200 replicates. Loose bounds --
at 200 replicates a rate near 0.05 has a standard error near 0.015."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from interval_harness import joint_probs, run_interval_cell  # noqa: E402

pytestmark = pytest.mark.slow


def test_construction_hits_the_requested_effect():
    from dbias.stats.effect_sizes import cohens_w

    p = joint_probs(np.full(5, 0.2), [0.7, 0.3], 0.2)
    assert cohens_w(p * 1e6) == pytest.approx(0.2, abs=1e-6)
    assert np.allclose(p.sum(axis=1), 0.2) and np.allclose(p.sum(axis=0), [0.7, 0.3])


def test_sparse_null_produces_no_spurious_disparities():
    row = run_interval_cell(np.full(16, 1 / 16), [0.5, 0.5], 500, 0.0, replicates=200, seed=1)
    assert row["spurious_disparity_rate"] <= 0.08


def test_effect_at_the_sesoi_is_not_cleared():
    row = run_interval_cell(np.full(5, 0.2), np.full(5, 0.2), 500, 0.1, replicates=200, seed=2)
    assert row["equivalent_rate"] <= 0.08
```

- [ ] **Step 3: Run the smoke test**

Run: `python -m pytest tests/calibration/test_interval_calibration.py -q`
Expected: 3 pass. If `test_effect_at_the_sesoi_is_not_cleared` fails, the recentred interval is too low: stop and report.

- [ ] **Step 4: Write the full gate script**

Create `tests/calibration/run_interval_gate.py`:

```python
"""Full interval calibration gate (pre-registered criteria in the plan).

    cd tests/calibration && python run_interval_gate.py
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

from interval_harness import joint_probs, run_interval_cell

SHAPES = {
    "2x2": (np.full(2, 0.5), np.full(2, 0.5)),
    "2x5 skewed": (np.array([0.67, 0.33]), np.array([0.85, 0.10, 0.03, 0.01, 0.01])),
    "5x5": (np.full(5, 0.2), np.full(5, 0.2)),
    "16x2": (np.full(16, 1 / 16), np.array([0.67, 0.33])),
    "14x5 skewed": (np.full(14, 1 / 14), np.array([0.85, 0.10, 0.03, 0.01, 0.01])),
}
N_VALUES = [200, 500, 2000]
W_VALUES = [0.0, 0.1, 0.2]
SESOI = 0.1


def main() -> None:
    start = time.time()
    rows = []
    for name, (r, c) in SHAPES.items():
        for n in N_VALUES:
            for w in W_VALUES:
                if joint_probs(r, c, w) is None:
                    print(f"skip unreachable: {name} n={n} w={w}")
                    continue
                row = run_interval_cell(r, c, n, w, sesoi=SESOI, replicates=1000, seed=n)
                row["shape"] = name
                rows.append(row)
                print(row)
    df = pd.DataFrame(rows)
    out = Path("results") / "interval_calibration_results.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)

    g1 = df[df.true_w == SESOI].equivalent_rate.max()
    g2 = df[df.true_w == 0].spurious_disparity_rate.max()
    print(f"\nG1 max EQUIVALENT rate at w = SESOI: {g1:.3f}  (gate <= 0.05)")
    print(f"G2 max spurious-disparity rate at w = 0: {g2:.3f}  (gate <= 0.05)")
    print(f"   old interval, same G2 statistic:  {df[df.true_w == 0].raw_spurious_disparity_rate.max():.3f}")
    print(f"GATE {'PASSED' if g1 <= 0.05 and g2 <= 0.05 else 'FAILED'}  ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the gate**

Run: `cd tests/calibration && python run_interval_gate.py` (minutes, not hours — every bootstrap is vectorised except the plug-in comparison).
Expected: `GATE PASSED`. If it prints `GATE FAILED`, do not continue to Task 5: commit the harness and the CSV, and report which cells failed.

- [ ] **Step 6: Re-run the 2x2 M3 gate on the new estimator**

Run: `cd tests/calibration && PYTHONPATH=. python run_m3_gate.py` (about 30 minutes; it writes `results/m3_calibration_results.csv` in place).
Expected: false positive rate and detection rate per cell match the current `results/m3_calibration_results.csv` within Monte Carlo error (the significance test is unchanged); false all-clears at w = 0.1 stay at or below 3.2% per cell. Compare with:

```bash
git diff --stat tests/calibration/results/m3_calibration_results.csv
```

- [ ] **Step 7: Commit**

```bash
git add tests/calibration/interval_harness.py tests/calibration/run_interval_gate.py tests/calibration/test_interval_calibration.py tests/calibration/results/
git commit -m "test(calibration): interval calibration gate for tables beyond 2x2"
```

---

### Task 5: Re-run the benchmarks and correct the write-up

**Files:**
- Modify: `results.md` (§1 if Step 6 above moved anything, §2 tables and both caveats)
- Modify: `challenges.md` (§5)

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Re-run the seeded benchmarks**

Run, from the repo root:

```bash
python run_adult.py && python run_compas.py && python run_subsampled.py && python compare_fairlearn.py
```

Expected: all exit 0.

- [ ] **Step 2: Extract the numbers the write-up quotes**

```bash
python -c "import json,collections; [print(n, len(f), dict(collections.Counter(x['severity'] for x in f)), [x['id'] for x in f if x['severity']=='Blind Spot']) for n in ['adult','compas','subsampled'] for f in [json.load(open(n+'_audit.json'))['findings']]]"
python -c "import json; f={x['id']:x for x in json.load(open('subsampled_audit.json'))['findings']}; g={x['id']:x for x in json.load(open('adult_audit.json'))['findings']}; [print(i, round(f[i]['effect_size_value'],3), f[i]['severity'], round(g[i]['effect_size_value'],3), g[i]['severity']) for i in ['DISP_EDUCATION_RACE','DISP_EDUCATION_SEX_AND_RACE','DISP_OCCUPATION_RACE'] if i in f and i in g]"
```

- [ ] **Step 3: Update `results.md`**

- In §2, replace the Big-N table, the COMPAS blind-spot paragraph and the Small-N result line with the counts printed in Step 2. If the COMPAS blind-spot ids changed, say so in one sentence and name the new ids — Plan `2026-10-04-m6-compas-incumbent.md` depends on them.
- Replace the paragraph starting **"Caveat on the intervals."** with: `**Intervals.** Contingency-table intervals use the Bergsma (2013) bias-corrected V with a bias-recentred bootstrap. On the interval calibration grid (tables up to 16x2 and 14x5, n from 200 to 2,000), no cell cleared an effect at the SESOI more than` followed by the G1 number from Task 4 Step 5, `of the time, and no null cell produced a spurious disparity more than` followed by the G2 number, `(the old plug-in interval:` followed by the old G2 number `).`
- Replace the paragraph starting **"Caveat on magnitudes."** with one sentence per row printed by the second command of Step 2, in the form `` `education` by race now measures V = <N=500 value> at N=500 and <full value> on the full data. ``

- [ ] **Step 4: Update `challenges.md` §5**

Replace the last line (`**The Fix:** Only an EQUIVALENT ...`) with:

```markdown
**The Fix:** Only an `EQUIVALENT` interval earns the all-clear; any other interval on a non-significant finding is a `BLIND_SPOT`. Effect sizes and intervals then moved to the Bergsma (2013) bias-corrected V, with the bootstrap interval recentred by its own estimated bias, and an interval calibration gate on tables up to 16x2 checks that the change creates no false all-clears (`tests/calibration/run_interval_gate.py`).
```

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add results.md challenges.md
git commit -m "docs: results on the bias-corrected V"
```
