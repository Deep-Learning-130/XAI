# Detectability-Aware Statistical Core — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the statistical core of the dataset bias auditor such that every test reports not only whether a disparity was found, but whether the dataset had enough power to have found one — making "we checked and it is clean" distinguishable from "we could not see."

**Architecture:** Three stacked layers with strictly one-directional dependencies. `models/` holds inert dataclasses and enums. `stats/` executes mathematically valid tests on arrays and knows nothing about bias, fairness, or protected attributes. `detectability/` wraps `stats/` to answer the inverse question — given this sample size, what is the smallest effect this test could have caught? No layer imports from a layer above it. Analyzers, rules, and reporting are out of scope for this plan and consume these modules in later plans.

**Tech Stack:** Python 3.11+, NumPy, SciPy (`scipy.stats`), statsmodels (`statsmodels.stats.multitest`, used as an independent cross-check oracle in tests), pandas, pytest.

**Spec:** [docs/02_statistical_framework.md](../../02_statistical_framework.md), [docs/05_risk_and_severity.md](../../05_risk_and_severity.md), [docs/04_data_model_and_api.md](../../04_data_model_and_api.md), [docs/10_repository_structure.md](../../10_repository_structure.md)

> **Note on spec drift:** This plan deliberately *overrides* three decisions in the specs above. Those overrides are listed verbatim in Global Constraints and are not open questions — implement what is written here, not what docs 02 and 05 say. A follow-up doc (`docs/09_detectability.md`) will backfill the design rationale; it is not required to execute this plan.

## Global Constraints

- **Python 3.11 minimum.** The plan uses `X | None` union syntax and `StrEnum`, both 3.11+.
- **Effect sizes are mandatory.** No function in `stats/` may return a p-value without an accompanying effect size. A p-value alone is never a valid result object.
- **No single scalar bias score, ever.** Per docs/05, outputs are per-category vectors. Nothing in this plan may compute a weighted mean across categories.
- **OVERRIDE 1 — per-metric thresholds.** docs/05 applies one small/medium/large ladder (`0.10 / 0.15 / 0.25`) to every effect size metric. This is invalid: Cramer's V, Cohen's *w*, rank-biserial, eta-squared, KS-D, and odds ratio are not on a common scale. Replace with the per-metric registry defined in Task 3. The `0.10 / 0.15 / 0.25` ladder is **deleted**.
- **OVERRIDE 2 — Cramer's V thresholds are df-adjusted.** docs/02 gives V thresholds of `0.1 / 0.3`; docs/05 gives `0.1 / 0.15 / 0.25`. Both are wrong for tables larger than 2x2. Since `w = V * sqrt(min(r-1, k-1))`, V thresholds are Cohen's `w` thresholds divided by `sqrt(min(r-1, k-1))`. Implement the division; do not hardcode either published ladder.
- **OVERRIDE 3 — severity is optional at construction.** docs/04 declares `severity` as a required `Finding` field, but docs/01 requires analyzers to emit *un-scored* findings. `severity` defaults to `Severity.UNDETERMINED` and is set only by the (out-of-scope) rules engine.
- **Blind spots are a first-class outcome.** A non-significant result is never reported as a clean result unless the test was adequately powered. This is the central novelty of the project and must not be optimised away.
- **Power target is 0.80 and alpha is 0.05,** both configurable via function parameters with those defaults. Never hardcode them inline.
- **Test oracles must be independent.** Assert against hand-computed closed-form values or statsmodels output — never against the implementation's own output.

---

### Task 1: Project scaffolding, enums, and core dataclasses

**Files:**
- Create: `pyproject.toml`
- Create: `src/dbias/__init__.py`
- Create: `src/dbias/models/__init__.py`
- Create: `src/dbias/models/enums.py`
- Create: `src/dbias/models/finding.py`
- Create: `src/dbias/models/profile.py`
- Test: `tests/unit/models/test_finding.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `Severity`, `Category`, `EffectSizeMetric`, `VariableKind`, `Detectability`, `Magnitude` (all `StrEnum`); `Finding` dataclass; `DatasetProfile` dataclass. Every later task imports from `dbias.models`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/models/test_finding.py
from dbias.models import Category, Detectability, EffectSizeMetric, Finding, Severity


def test_finding_defaults_to_undetermined_severity():
    """Analyzers emit un-scored findings; only the rules engine assigns severity."""
    finding = Finding(
        id="MNAR_INCOME_GENDER",
        category=Category.MISSINGNESS,
        sensitive_attribute="gender",
        target_feature="income",
        metric_name="Missingness Rate Disparity",
        observed_values={"male": 0.02, "female": 0.15},
        statistical_test="Chi-Square Test of Independence",
        p_value_raw=0.001,
        effect_size_metric=EffectSizeMetric.CRAMERS_V,
        effect_size_value=0.35,
        n_per_group={"male": 5000, "female": 5000},
    )
    assert finding.severity is Severity.UNDETERMINED
    assert finding.p_value_corrected is None
    assert finding.detectability is Detectability.UNKNOWN
    assert finding.is_significant is False


def test_finding_total_n_sums_group_counts():
    finding = Finding(
        id="REP_GENDER",
        category=Category.REPRESENTATION,
        sensitive_attribute="gender",
        target_feature=None,
        metric_name="Representation Imbalance",
        observed_values={"male": 0.8, "female": 0.2},
        statistical_test="Chi-Square Goodness of Fit",
        p_value_raw=0.0,
        effect_size_metric=EffectSizeMetric.COHENS_W,
        effect_size_value=0.6,
        n_per_group={"male": 8000, "female": 2000},
    )
    assert finding.total_n == 10000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/models/test_finding.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dbias'`

- [ ] **Step 3: Write minimal implementation**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "dbias"
version = "0.1.0"
description = "Detectability-aware dataset bias auditing"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "pandas>=2.1",
    "scipy>=1.11",
    "statsmodels>=0.14",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

```python
# src/dbias/models/enums.py
from enum import StrEnum


class Severity(StrEnum):
    UNDETERMINED = "Undetermined"
    BLIND_SPOT = "Blind Spot"
    INFORMATIONAL = "Informational"
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class Category(StrEnum):
    REPRESENTATION = "Representation"
    MISSINGNESS = "Missingness"
    FEATURE_DISPARITY = "Feature_Disparity"
    LABEL_DISPARITY = "Label_Disparity"


class EffectSizeMetric(StrEnum):
    CRAMERS_V = "Cramer's V"
    COHENS_W = "Cohen's w"
    RANK_BISERIAL = "Rank-Biserial Correlation"
    ETA_SQUARED = "Eta-Squared"
    KS_STATISTIC = "KS Statistic (D)"
    ODDS_RATIO = "Odds Ratio"


class VariableKind(StrEnum):
    CATEGORICAL = "categorical"
    CONTINUOUS = "continuous"
    BOOLEAN = "boolean"


class Detectability(StrEnum):
    UNKNOWN = "unknown"            # power analysis not yet run
    ADEQUATE = "adequate"          # could have caught a medium effect
    UNDERPOWERED = "underpowered"  # could not; a null result here means nothing


class Magnitude(StrEnum):
    TRIVIAL = "trivial"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
```

```python
# src/dbias/models/finding.py
from dataclasses import dataclass, field
from typing import Any

from dbias.models.enums import Category, Detectability, EffectSizeMetric, Severity


@dataclass
class Finding:
    """Universal contract for one audited hypothesis.

    Carries both directions of error: `p_value_corrected` / `effect_size_value`
    describe what was found, while `statistical_power` /
    `minimum_detectable_effect` describe what *could* have been found. A
    Finding with no disparity is only meaningful alongside its detectability.
    """

    id: str
    category: Category
    sensitive_attribute: str
    target_feature: str | None
    metric_name: str
    observed_values: dict[str, Any]

    # Statistical evidence
    statistical_test: str
    p_value_raw: float
    effect_size_metric: EffectSizeMetric
    effect_size_value: float
    n_per_group: dict[str, int]

    # Populated by the correction pass (later plan)
    p_value_corrected: float | None = None
    is_significant: bool = False

    # Populated by the detectability engine (Tasks 6-7)
    statistical_power: float | None = None
    minimum_detectable_effect: float | None = None
    detectability: Detectability = Detectability.UNKNOWN

    # Populated by the rules engine (later plan)
    severity: Severity = Severity.UNDETERMINED

    # Populated by the NLG layer (later plan)
    evidence_text: str = ""
    potential_impact: str = ""
    recommendations: list[str] = field(default_factory=list)

    @property
    def total_n(self) -> int:
        return sum(self.n_per_group.values())
```

```python
# src/dbias/models/profile.py
from dataclasses import dataclass, field


@dataclass
class DatasetProfile:
    total_rows: int
    categorical_cols: list[str] = field(default_factory=list)
    continuous_cols: list[str] = field(default_factory=list)
    boolean_cols: list[str] = field(default_factory=list)
    sensitive_cols: list[str] = field(default_factory=list)
    target_col: str | None = None
```

```python
# src/dbias/models/__init__.py
from dbias.models.enums import (
    Category,
    Detectability,
    EffectSizeMetric,
    Magnitude,
    Severity,
    VariableKind,
)
from dbias.models.finding import Finding
from dbias.models.profile import DatasetProfile

__all__ = [
    "Category",
    "DatasetProfile",
    "Detectability",
    "EffectSizeMetric",
    "Finding",
    "Magnitude",
    "Severity",
    "VariableKind",
]
```

Create `src/dbias/__init__.py` as an empty file.

- [ ] **Step 4: Run test to verify it passes**

Run: `pip install -e ".[dev]"` then `pytest tests/unit/models/test_finding.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/dbias tests/unit/models
git commit -m "feat: add Finding and DatasetProfile models with detectability fields"
```

---

### Task 2: Effect size calculators

**Files:**
- Create: `src/dbias/stats/__init__.py`
- Create: `src/dbias/stats/effect_sizes.py`
- Test: `tests/unit/stats/test_effect_sizes.py`

**Interfaces:**
- Consumes: nothing from Task 1 (pure numeric functions).
- Produces: `cramers_v(table) -> float`, `cohens_w(observed_props, expected_props) -> float`, `rank_biserial(u_statistic, n1, n2) -> float`, `eta_squared_h(h_statistic, k_groups, n_total) -> float`. Tasks 4 and 6 call these.

- [ ] **Step 1: Write the failing test**

Each expected value below is hand-computed from the closed form, not read off the implementation.

```python
# tests/unit/stats/test_effect_sizes.py
import numpy as np
import pytest

from dbias.stats.effect_sizes import cohens_w, cramers_v, eta_squared_h, rank_biserial


def test_cramers_v_matches_closed_form():
    # table [[10,20],[20,10]]: all expected counts 15, chi2 = 4*(25/15) = 6.6667,
    # n = 60, min(r-1, k-1) = 1  ->  V = sqrt(6.6667/60) = 1/3
    table = np.array([[10, 20], [20, 10]])
    assert cramers_v(table) == pytest.approx(1 / 3, abs=1e-9)


def test_cramers_v_is_zero_for_independent_table():
    table = np.array([[25, 25], [25, 25]])
    assert cramers_v(table) == pytest.approx(0.0, abs=1e-9)


def test_cohens_w_matches_closed_form():
    # observed [0.8, 0.2] vs expected [0.5, 0.5]
    # w = sqrt(0.09/0.5 + 0.09/0.5) = sqrt(0.36) = 0.6
    assert cohens_w([0.8, 0.2], [0.5, 0.5]) == pytest.approx(0.6, abs=1e-9)


def test_cohens_w_is_zero_when_observed_equals_expected():
    assert cohens_w([0.25, 0.75], [0.25, 0.75]) == pytest.approx(0.0, abs=1e-9)


def test_cohens_w_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        cohens_w([0.5, 0.5], [0.3, 0.3, 0.4])


def test_rank_biserial_is_one_under_complete_separation():
    # group 1 entirely above group 2 -> U1 = n1*n2 = 9 -> r = 2*9/9 - 1 = 1
    assert rank_biserial(u_statistic=9, n1=3, n2=3) == pytest.approx(1.0)


def test_rank_biserial_is_negative_one_under_reverse_separation():
    assert rank_biserial(u_statistic=0, n1=3, n2=3) == pytest.approx(-1.0)


def test_rank_biserial_is_zero_at_perfect_tie():
    assert rank_biserial(u_statistic=4.5, n1=3, n2=3) == pytest.approx(0.0)


def test_eta_squared_h_matches_closed_form():
    # (H - k + 1) / (n - k) = (10 - 3 + 1) / (30 - 3) = 8/27
    assert eta_squared_h(h_statistic=10.0, k_groups=3, n_total=30) == pytest.approx(
        8 / 27, abs=1e-9
    )


def test_eta_squared_h_clamps_negative_to_zero():
    # H below k-1 yields a negative raw value, which is not a valid proportion
    assert eta_squared_h(h_statistic=0.5, k_groups=3, n_total=30) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/stats/test_effect_sizes.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dbias.stats'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/dbias/stats/effect_sizes.py
"""Effect size calculators.

This module is deliberately ignorant of bias, fairness, and protected
attributes. It converts test statistics into standardised magnitudes.
"""

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import chi2_contingency


def cramers_v(table: ArrayLike) -> float:
    """Association between two nominal variables.

    V = sqrt(chi2 / (n * min(k-1, r-1)))
    """
    table = np.asarray(table, dtype=float)
    if table.ndim != 2:
        raise ValueError("Cramer's V requires a 2-D contingency table")
    chi2 = chi2_contingency(table, correction=False).statistic
    n = table.sum()
    df_min = min(table.shape[0] - 1, table.shape[1] - 1)
    if n == 0 or df_min == 0:
        return 0.0
    return float(np.sqrt(chi2 / (n * df_min)))


def cohens_w(observed_props: ArrayLike, expected_props: ArrayLike) -> float:
    """Goodness-of-fit effect size for representation imbalance.

    w = sqrt(sum((p0i - p1i)^2 / p0i)) where p0 is expected, p1 is observed.
    """
    observed = np.asarray(observed_props, dtype=float)
    expected = np.asarray(expected_props, dtype=float)
    if observed.shape != expected.shape:
        raise ValueError("observed and expected must have the same length")
    if np.any(expected <= 0):
        raise ValueError("expected proportions must all be strictly positive")
    return float(np.sqrt(np.sum((expected - observed) ** 2 / expected)))


def rank_biserial(u_statistic: float, n1: int, n2: int) -> float:
    """Effect size companion to Mann-Whitney U, on [-1, 1].

    `u_statistic` must be the U value reported for group 1.
    """
    if n1 <= 0 or n2 <= 0:
        raise ValueError("both groups must be non-empty")
    return float(2.0 * u_statistic / (n1 * n2) - 1.0)


def eta_squared_h(h_statistic: float, k_groups: int, n_total: int) -> float:
    """Effect size companion to Kruskal-Wallis H.

    eta^2 = (H - k + 1) / (n - k), clamped to [0, 1].
    """
    if n_total <= k_groups:
        raise ValueError("n_total must exceed the number of groups")
    value = (h_statistic - k_groups + 1) / (n_total - k_groups)
    return float(min(max(value, 0.0), 1.0))
```

Create `src/dbias/stats/__init__.py` as an empty file.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/stats/test_effect_sizes.py -v`
Expected: PASS (10 passed)

- [ ] **Step 5: Commit**

```bash
git add src/dbias/stats tests/unit/stats
git commit -m "feat: add Cramer's V, Cohen's w, rank-biserial and eta-squared calculators"
```

---

### Task 3: Per-metric threshold registry

This task implements OVERRIDE 1 and OVERRIDE 2. It is the fix for docs/05 applying a single ladder to metrics that live on different scales.

**Files:**
- Create: `src/dbias/stats/thresholds.py`
- Test: `tests/unit/stats/test_thresholds.py`

**Interfaces:**
- Consumes: `EffectSizeMetric`, `Magnitude` from `dbias.models` (Task 1).
- Produces: `ThresholdSet` frozen dataclass with fields `small: float, medium: float, large: float`; `thresholds_for(metric, df_min=1) -> ThresholdSet`; `classify_magnitude(metric, value, df_min=1) -> Magnitude`. Task 7 calls `thresholds_for`; the later rules engine calls both.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/stats/test_thresholds.py
import math

import pytest

from dbias.models import EffectSizeMetric, Magnitude
from dbias.stats.thresholds import classify_magnitude, thresholds_for


def test_cramers_v_thresholds_equal_cohens_w_for_2x2():
    # df_min = 1  ->  V and w coincide, so Cohen's 0.1/0.3/0.5 applies unchanged
    ts = thresholds_for(EffectSizeMetric.CRAMERS_V, df_min=1)
    assert (ts.small, ts.medium, ts.large) == pytest.approx((0.1, 0.3, 0.5))


def test_cramers_v_thresholds_shrink_with_table_size():
    # w = V * sqrt(df_min)  ->  V thresholds are w thresholds / sqrt(df_min)
    ts = thresholds_for(EffectSizeMetric.CRAMERS_V, df_min=2)
    assert ts.medium == pytest.approx(0.3 / math.sqrt(2), abs=1e-9)


def test_eta_squared_uses_its_own_convention_not_the_cohen_ladder():
    ts = thresholds_for(EffectSizeMetric.ETA_SQUARED)
    assert (ts.small, ts.medium, ts.large) == pytest.approx((0.01, 0.06, 0.14))


def test_odds_ratio_is_classified_on_the_log_scale():
    # OR = 1 is the null; reciprocal pairs must classify identically
    assert classify_magnitude(EffectSizeMetric.ODDS_RATIO, 1.0) is Magnitude.TRIVIAL
    assert classify_magnitude(
        EffectSizeMetric.ODDS_RATIO, 4.0
    ) is classify_magnitude(EffectSizeMetric.ODDS_RATIO, 0.25)


def test_classify_magnitude_uses_absolute_value_for_signed_metrics():
    assert classify_magnitude(EffectSizeMetric.RANK_BISERIAL, -0.6) is Magnitude.LARGE


def test_classify_magnitude_boundaries_are_inclusive_lower():
    assert classify_magnitude(EffectSizeMetric.COHENS_W, 0.30) is Magnitude.MEDIUM
    assert classify_magnitude(EffectSizeMetric.COHENS_W, 0.2999) is Magnitude.SMALL
    assert classify_magnitude(EffectSizeMetric.COHENS_W, 0.05) is Magnitude.TRIVIAL


def test_unknown_metric_raises():
    with pytest.raises(KeyError):
        thresholds_for("not-a-metric")  # type: ignore[arg-type]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/stats/test_thresholds.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dbias.stats.thresholds'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/dbias/stats/thresholds.py
"""Per-metric effect size thresholds.

docs/05 originally applied a single 0.10/0.15/0.25 ladder to every metric.
That is invalid: these metrics are not on a common scale. Each metric family
gets its own published convention here, and Cramer's V is additionally
adjusted for table size because w = V * sqrt(min(r-1, k-1)).
"""

import math
from dataclasses import dataclass

from dbias.models import EffectSizeMetric, Magnitude


@dataclass(frozen=True)
class ThresholdSet:
    small: float
    medium: float
    large: float


_BASE: dict[EffectSizeMetric, ThresholdSet] = {
    EffectSizeMetric.COHENS_W: ThresholdSet(0.10, 0.30, 0.50),
    EffectSizeMetric.CRAMERS_V: ThresholdSet(0.10, 0.30, 0.50),
    EffectSizeMetric.RANK_BISERIAL: ThresholdSet(0.10, 0.30, 0.50),
    EffectSizeMetric.ETA_SQUARED: ThresholdSet(0.01, 0.06, 0.14),
    # KS-D has no established convention; these are provisional and are
    # expected to be overridden by user configuration.
    EffectSizeMetric.KS_STATISTIC: ThresholdSet(0.10, 0.20, 0.30),
    # Odds ratio is classified on |log(OR)| converted to Cohen's d via
    # d = log(OR) * sqrt(3) / pi, so the ladder below is d's 0.2/0.5/0.8.
    EffectSizeMetric.ODDS_RATIO: ThresholdSet(0.20, 0.50, 0.80),
}

_LOG_ODDS_TO_D = math.sqrt(3) / math.pi


def thresholds_for(metric: EffectSizeMetric, df_min: int = 1) -> ThresholdSet:
    """Return small/medium/large cutoffs for `metric`.

    `df_min` is min(rows-1, cols-1) and only affects Cramer's V.
    """
    base = _BASE[metric]
    if metric is EffectSizeMetric.CRAMERS_V and df_min > 1:
        scale = math.sqrt(df_min)
        return ThresholdSet(base.small / scale, base.medium / scale, base.large / scale)
    return base


def _normalise(metric: EffectSizeMetric, value: float) -> float:
    """Map a raw effect size onto the scale its thresholds are defined on."""
    if metric is EffectSizeMetric.ODDS_RATIO:
        if value <= 0:
            raise ValueError("odds ratio must be strictly positive")
        return abs(math.log(value)) * _LOG_ODDS_TO_D
    return abs(value)


def classify_magnitude(
    metric: EffectSizeMetric, value: float, df_min: int = 1
) -> Magnitude:
    """Bucket an effect size. Lower bounds are inclusive."""
    ts = thresholds_for(metric, df_min)
    magnitude = _normalise(metric, value)
    if magnitude >= ts.large:
        return Magnitude.LARGE
    if magnitude >= ts.medium:
        return Magnitude.MEDIUM
    if magnitude >= ts.small:
        return Magnitude.SMALL
    return Magnitude.TRIVIAL
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/stats/test_thresholds.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add src/dbias/stats/thresholds.py tests/unit/stats/test_thresholds.py
git commit -m "feat: add per-metric threshold registry with df-adjusted Cramer's V"
```

---

### Task 4: Test dispatch (the decision tree from docs/02)

**Files:**
- Create: `src/dbias/stats/dispatch.py`
- Test: `tests/unit/stats/test_dispatch.py`

**Interfaces:**
- Consumes: `EffectSizeMetric`, `VariableKind` from `dbias.models` (Task 1).
- Produces: `TestSpec` frozen dataclass with fields `test_name: str, effect_size_metric: EffectSizeMetric`; `select_test(target_kind, predictor_kind, n_groups, min_expected_count=None) -> TestSpec`. The later analyzers call `select_test`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/stats/test_dispatch.py
import pytest

from dbias.models import EffectSizeMetric, VariableKind
from dbias.stats.dispatch import select_test


def test_categorical_pair_with_adequate_counts_uses_chi_square():
    spec = select_test(
        VariableKind.CATEGORICAL,
        VariableKind.CATEGORICAL,
        n_groups=2,
        min_expected_count=12.0,
    )
    assert spec.test_name == "Chi-Square Test of Independence"
    assert spec.effect_size_metric is EffectSizeMetric.CRAMERS_V


def test_categorical_pair_with_sparse_counts_uses_fishers_exact():
    spec = select_test(
        VariableKind.CATEGORICAL,
        VariableKind.CATEGORICAL,
        n_groups=2,
        min_expected_count=3.0,
    )
    assert spec.test_name == "Fisher's Exact Test"
    assert spec.effect_size_metric is EffectSizeMetric.ODDS_RATIO


def test_fishers_exact_is_refused_beyond_two_groups():
    """The odds ratio is undefined for RxC, so sparse multi-group falls back."""
    spec = select_test(
        VariableKind.CATEGORICAL,
        VariableKind.CATEGORICAL,
        n_groups=4,
        min_expected_count=2.0,
    )
    assert spec.test_name == "Chi-Square Test of Independence"
    assert spec.effect_size_metric is EffectSizeMetric.CRAMERS_V


def test_continuous_by_two_groups_uses_mann_whitney():
    spec = select_test(VariableKind.CONTINUOUS, VariableKind.CATEGORICAL, n_groups=2)
    assert spec.test_name == "Mann-Whitney U"
    assert spec.effect_size_metric is EffectSizeMetric.RANK_BISERIAL


def test_continuous_by_many_groups_uses_kruskal_wallis():
    spec = select_test(VariableKind.CONTINUOUS, VariableKind.CATEGORICAL, n_groups=5)
    assert spec.test_name == "Kruskal-Wallis"
    assert spec.effect_size_metric is EffectSizeMetric.ETA_SQUARED


def test_boolean_target_is_treated_as_categorical():
    spec = select_test(
        VariableKind.BOOLEAN,
        VariableKind.CATEGORICAL,
        n_groups=2,
        min_expected_count=50.0,
    )
    assert spec.effect_size_metric is EffectSizeMetric.CRAMERS_V


def test_single_group_is_rejected():
    with pytest.raises(ValueError, match="at least two groups"):
        select_test(VariableKind.CONTINUOUS, VariableKind.CATEGORICAL, n_groups=1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/stats/test_dispatch.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dbias.stats.dispatch'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/dbias/stats/dispatch.py
"""Automatic test selection.

Encodes the decision tree in docs/02. Non-parametric tests are preferred
throughout because real tabular data is rarely normally distributed.
"""

from dataclasses import dataclass

from dbias.models import EffectSizeMetric, VariableKind

# Below this expected cell count the chi-square approximation degrades.
SPARSE_CELL_THRESHOLD = 5.0


@dataclass(frozen=True)
class TestSpec:
    test_name: str
    effect_size_metric: EffectSizeMetric


def _is_discrete(kind: VariableKind) -> bool:
    return kind in (VariableKind.CATEGORICAL, VariableKind.BOOLEAN)


def select_test(
    target_kind: VariableKind,
    predictor_kind: VariableKind,
    n_groups: int,
    min_expected_count: float | None = None,
) -> TestSpec:
    """Choose a valid test for the given variable pair.

    `min_expected_count` is the smallest expected cell count of the
    contingency table; it is only consulted for discrete/discrete pairs.
    """
    if n_groups < 2:
        raise ValueError("a comparison needs at least two groups")

    if _is_discrete(target_kind) and _is_discrete(predictor_kind):
        sparse = (
            min_expected_count is not None
            and min_expected_count < SPARSE_CELL_THRESHOLD
        )
        # The odds ratio is only defined for 2x2, so Fisher's exact is not
        # offered beyond two groups even when cells are sparse.
        if sparse and n_groups == 2:
            return TestSpec("Fisher's Exact Test", EffectSizeMetric.ODDS_RATIO)
        return TestSpec("Chi-Square Test of Independence", EffectSizeMetric.CRAMERS_V)

    if target_kind is VariableKind.CONTINUOUS and _is_discrete(predictor_kind):
        if n_groups == 2:
            return TestSpec("Mann-Whitney U", EffectSizeMetric.RANK_BISERIAL)
        return TestSpec("Kruskal-Wallis", EffectSizeMetric.ETA_SQUARED)

    if (
        target_kind is VariableKind.CONTINUOUS
        and predictor_kind is VariableKind.CONTINUOUS
    ):
        return TestSpec("Two-Sample Kolmogorov-Smirnov", EffectSizeMetric.KS_STATISTIC)

    raise ValueError(
        f"no valid test for target={target_kind}, predictor={predictor_kind}"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/stats/test_dispatch.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add src/dbias/stats/dispatch.py tests/unit/stats/test_dispatch.py
git commit -m "feat: add automatic statistical test dispatch"
```

---

### Task 5: Benjamini-Hochberg FDR correction

**Files:**
- Create: `src/dbias/stats/correction.py`
- Test: `tests/unit/stats/test_correction.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (operates on plain float sequences).
- Produces: `benjamini_hochberg(p_values) -> list[float]`. The later hierarchical-gating plan wraps this; it is the flat single-family case.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/stats/test_correction.py
import pytest
from statsmodels.stats.multitest import multipletests

from dbias.stats.correction import benjamini_hochberg


def test_uniform_ramp_yields_identical_q_values():
    # p_i * m / i equals 0.05 for every i in this sequence
    q = benjamini_hochberg([0.01, 0.02, 0.03, 0.04, 0.05])
    assert q == pytest.approx([0.05] * 5)


def test_matches_hand_computed_classic_example():
    p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.060, 0.074, 0.205]
    expected = [0.008, 0.032, 0.0672, 0.0672, 0.0672, 0.080, 0.08457142857, 0.205]
    assert benjamini_hochberg(p) == pytest.approx(expected, abs=1e-8)


def test_matches_statsmodels_on_unsorted_input():
    """Independent oracle, and confirms input order is preserved."""
    p = [0.9, 0.001, 0.31, 0.042, 0.6, 0.008]
    expected = multipletests(p, method="fdr_bh")[1]
    assert benjamini_hochberg(p) == pytest.approx(list(expected), abs=1e-9)


def test_q_values_are_capped_at_one():
    assert all(v <= 1.0 for v in benjamini_hochberg([0.7, 0.8, 0.99]))


def test_empty_input_returns_empty_list():
    assert benjamini_hochberg([]) == []


def test_rejects_out_of_range_p_values():
    with pytest.raises(ValueError, match="between 0 and 1"):
        benjamini_hochberg([0.5, 1.4])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/stats/test_correction.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dbias.stats.correction'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/dbias/stats/correction.py
"""Multiple-testing correction.

An audit of 50 features against 3 protected attributes runs 150+ tests; at
alpha=0.05 roughly 8 of those are false positives by chance alone. Every
p-value that reaches a severity decision must be corrected first.
"""

from collections.abc import Sequence

import numpy as np


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg step-up FDR correction.

    Returns q-values in the same order as the input.
    """
    if len(p_values) == 0:
        return []

    p = np.asarray(p_values, dtype=float)
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError("p-values must be between 0 and 1")

    m = p.size
    order = np.argsort(p)
    ranks = np.arange(1, m + 1)

    # Step up: scale by m/i, then enforce monotonicity from the largest down.
    scaled = p[order] * m / ranks
    monotone = np.minimum.accumulate(scaled[::-1])[::-1]
    monotone = np.minimum(monotone, 1.0)

    q = np.empty(m, dtype=float)
    q[order] = monotone
    return q.tolist()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/stats/test_correction.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/dbias/stats/correction.py tests/unit/stats/test_correction.py
git commit -m "feat: add Benjamini-Hochberg FDR correction"
```

---

### Task 6: Power and minimum detectable effect engine

This is the flagship task. Everything before it exists in other toolkits; this does not.

**Files:**
- Create: `src/dbias/detectability/__init__.py`
- Create: `src/dbias/detectability/power.py`
- Test: `tests/unit/detectability/test_power.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (pure numeric functions over SciPy).
- Produces: `DEFAULT_ALPHA = 0.05`, `DEFAULT_POWER = 0.80`; `mde_chi_square(n, df, alpha=0.05, power=0.80) -> float` (returns Cohen's *w*); `mde_rank_biserial(n1, n2, alpha=0.05, power=0.80) -> float`; `achieved_power_chi_square(effect_size, n, df, alpha=0.05) -> float`. Task 7 calls all three.

Two closed-form anchors make these testable without trusting the implementation:
- For `df=1, alpha=0.05, power=0.80`, the required non-centrality is `(1.96 + 0.8416)^2 = 7.849`, so `w = sqrt(7.849/n)`. At `n=100` that is `0.2802`.
- Under normality, Cohen's `d` converts exactly to rank-biserial via `r = 2*Phi(d/sqrt(2)) - 1`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/detectability/test_power.py
import math

import pytest

from dbias.detectability.power import (
    achieved_power_chi_square,
    mde_chi_square,
    mde_rank_biserial,
)


def test_mde_chi_square_matches_closed_form_at_df_one():
    # required non-centrality for 80% power at df=1, alpha=.05 is 7.849
    expected = math.sqrt(7.849 / 100)
    assert mde_chi_square(n=100, df=1) == pytest.approx(expected, abs=1e-3)


def test_mde_chi_square_scales_as_inverse_root_n():
    """Quadrupling n should halve the minimum detectable effect."""
    assert mde_chi_square(n=400, df=1) == pytest.approx(
        mde_chi_square(n=100, df=1) / 2, rel=1e-6
    )


def test_small_sample_cannot_detect_even_a_large_effect():
    """The headline claim: at n=40 nothing below a huge effect is visible."""
    assert mde_chi_square(n=40, df=1) > 0.40  # Cohen's 'large' for w is 0.50


def test_large_sample_can_detect_a_trivial_effect():
    """The mirror case: at n=1e6 even noise is detectable, hence the effect gate."""
    assert mde_chi_square(n=1_000_000, df=1) < 0.01


def test_achieved_power_is_high_when_effect_far_exceeds_mde():
    mde = mde_chi_square(n=100, df=1)
    assert achieved_power_chi_square(effect_size=mde * 3, n=100, df=1) > 0.99


def test_achieved_power_equals_target_at_the_mde():
    mde = mde_chi_square(n=250, df=1)
    assert achieved_power_chi_square(effect_size=mde, n=250, df=1) == pytest.approx(
        0.80, abs=1e-3
    )


def test_achieved_power_at_null_effect_equals_alpha():
    assert achieved_power_chi_square(effect_size=0.0, n=500, df=1) == pytest.approx(
        0.05, abs=1e-6
    )


def test_mde_rank_biserial_is_bounded_and_shrinks_with_n():
    small = mde_rank_biserial(n1=20, n2=20)
    large = mde_rank_biserial(n1=2000, n2=2000)
    assert 0.0 < large < small < 1.0
    assert small > 0.40  # 20 per group is effectively blind


def test_mde_rank_biserial_rejects_degenerate_groups():
    with pytest.raises(ValueError, match="at least two observations"):
        mde_rank_biserial(n1=1, n2=50)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/detectability/test_power.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dbias.detectability'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/dbias/detectability/power.py
"""Power and minimum detectable effect.

Every other auditing tool answers "is there a disparity?". This module
answers the inverse: given this sample size, what is the smallest disparity
this test could possibly have found? A non-significant result from an
underpowered test is not evidence of fairness, and reporting it as though it
were is the failure mode this project exists to fix.
"""

import math

from scipy.optimize import brentq
from scipy.stats import chi2, ncx2, norm

DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.80


def achieved_power_chi_square(
    effect_size: float, n: int, df: int, alpha: float = DEFAULT_ALPHA
) -> float:
    """Power of a chi-square test to detect Cohen's `w` at sample size `n`.

    Non-centrality is lambda = w^2 * n.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if df < 1:
        raise ValueError("df must be at least 1")
    critical = chi2.ppf(1 - alpha, df)
    ncp = (effect_size**2) * n
    return float(ncx2.sf(critical, df, ncp))


def mde_chi_square(
    n: int, df: int, alpha: float = DEFAULT_ALPHA, power: float = DEFAULT_POWER
) -> float:
    """Smallest Cohen's `w` detectable at `power` given `n`."""
    if n <= 0:
        raise ValueError("n must be positive")

    def shortfall(w: float) -> float:
        return achieved_power_chi_square(w, n, df, alpha) - power

    # w = 2.0 is far past anything observable in practice, so it brackets the root.
    return float(brentq(shortfall, 1e-9, 2.0, xtol=1e-10))


def _d_to_rank_biserial(d: float) -> float:
    """Exact conversion under normality: r = 2*Phi(d/sqrt(2)) - 1."""
    return 2.0 * norm.cdf(d / math.sqrt(2.0)) - 1.0


def mde_rank_biserial(
    n1: int, n2: int, alpha: float = DEFAULT_ALPHA, power: float = DEFAULT_POWER
) -> float:
    """Smallest rank-biserial correlation detectable at `power`.

    Solved on the Cohen's `d` scale (a two-sided two-sample location test)
    and converted. Mann-Whitney has ~95% asymptotic relative efficiency
    versus the t-test under normality, so this is very slightly optimistic.
    """
    if n1 < 2 or n2 < 2:
        raise ValueError("each group needs at least two observations")

    z_alpha = norm.ppf(1 - alpha / 2)
    z_power = norm.ppf(power)
    d = (z_alpha + z_power) * math.sqrt(1.0 / n1 + 1.0 / n2)
    return float(_d_to_rank_biserial(d))
```

Create `src/dbias/detectability/__init__.py` as an empty file for now; Task 7 fills it in.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/detectability/test_power.py -v`
Expected: PASS (9 passed)

- [ ] **Step 5: Commit**

```bash
git add src/dbias/detectability tests/unit/detectability
git commit -m "feat: add power and minimum detectable effect engine"
```

---

### Task 7: Detectability classification and Finding annotation

Wires Tasks 3 and 6 together into the decision that gives the project its name: a null result is either an earned all-clear or a blind spot, and the two must never be conflated.

**Files:**
- Create: `src/dbias/detectability/classify.py`
- Modify: `src/dbias/detectability/__init__.py`
- Test: `tests/unit/detectability/test_classify.py`

**Interfaces:**
- Consumes: `Finding`, `Detectability`, `EffectSizeMetric` (Task 1); `thresholds_for` (Task 3); `DEFAULT_ALPHA`, `DEFAULT_POWER`, `mde_chi_square`, `mde_rank_biserial`, `achieved_power_chi_square` (Task 6).
- Produces: `annotate_detectability(finding, df=1, alpha=0.05, power_target=0.80) -> Finding` — mutates and returns the same `Finding`, setting `statistical_power`, `minimum_detectable_effect`, and `detectability`. The later rules engine reads `finding.detectability` to choose between `Severity.INFORMATIONAL` and `Severity.BLIND_SPOT`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/detectability/test_classify.py
import pytest

from dbias.detectability.classify import annotate_detectability
from dbias.models import Category, Detectability, EffectSizeMetric, Finding


def _finding(effect: float, metric: EffectSizeMetric, n_per_group: dict[str, int]):
    return Finding(
        id="TEST",
        category=Category.MISSINGNESS,
        sensitive_attribute="group",
        target_feature="income",
        metric_name="Missingness Rate Disparity",
        observed_values={},
        statistical_test="Chi-Square Test of Independence",
        p_value_raw=0.5,
        effect_size_metric=metric,
        effect_size_value=effect,
        n_per_group=n_per_group,
    )


def test_tiny_subgroup_is_flagged_underpowered():
    """Scenario D: 40 rows cannot rule out even a large disparity."""
    finding = _finding(0.05, EffectSizeMetric.CRAMERS_V, {"a": 20, "b": 20})
    annotate_detectability(finding, df=1)
    assert finding.detectability is Detectability.UNDERPOWERED
    assert finding.minimum_detectable_effect > 0.40


def test_large_sample_null_is_an_earned_all_clear():
    """Same null effect, enough data: this one genuinely means 'clean'."""
    finding = _finding(0.005, EffectSizeMetric.CRAMERS_V, {"a": 50_000, "b": 50_000})
    annotate_detectability(finding, df=1)
    assert finding.detectability is Detectability.ADEQUATE
    assert finding.minimum_detectable_effect < 0.10


def test_adequacy_is_judged_against_the_medium_threshold():
    """A test is adequate iff its MDE sits at or below a 'medium' effect."""
    finding = _finding(0.01, EffectSizeMetric.CRAMERS_V, {"a": 60, "b": 60})
    annotate_detectability(finding, df=1)
    # MDE at n=120, df=1 is ~0.256, below the 0.30 medium cutoff for V
    assert finding.minimum_detectable_effect == pytest.approx(0.256, abs=0.01)
    assert finding.detectability is Detectability.ADEQUATE


def test_power_is_recorded_against_the_observed_effect():
    finding = _finding(0.35, EffectSizeMetric.CRAMERS_V, {"a": 500, "b": 500})
    annotate_detectability(finding, df=1)
    assert finding.statistical_power > 0.99


def test_continuous_metric_routes_to_the_rank_biserial_solver():
    finding = _finding(0.02, EffectSizeMetric.RANK_BISERIAL, {"a": 20, "b": 20})
    annotate_detectability(finding)
    assert finding.detectability is Detectability.UNDERPOWERED
    assert finding.minimum_detectable_effect > 0.40


def test_df_adjustment_flows_through_to_adequacy():
    """A wider table has a stricter medium cutoff, so the same n is blinder."""
    finding = _finding(0.01, EffectSizeMetric.CRAMERS_V, {"a": 60, "b": 60})
    annotate_detectability(finding, df=4)
    assert finding.detectability is Detectability.UNDERPOWERED


def test_unsupported_metric_leaves_detectability_unknown():
    finding = _finding(1.5, EffectSizeMetric.ODDS_RATIO, {"a": 50, "b": 50})
    annotate_detectability(finding)
    assert finding.detectability is Detectability.UNKNOWN
    assert finding.minimum_detectable_effect is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/detectability/test_classify.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dbias.detectability.classify'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/dbias/detectability/classify.py
"""Turn a power calculation into a reportable verdict.

A test is ADEQUATE when its minimum detectable effect sits at or below the
"medium" threshold for its own metric — i.e. a disparity a practitioner
would care about would have been caught. Otherwise it is UNDERPOWERED and
its null result carries no information.
"""

from dbias.detectability.power import (
    DEFAULT_ALPHA,
    DEFAULT_POWER,
    achieved_power_chi_square,
    mde_chi_square,
    mde_rank_biserial,
)
from dbias.models import Detectability, EffectSizeMetric, Finding
from dbias.stats.thresholds import thresholds_for

# Metrics with a solver in detectability.power. Others stay UNKNOWN rather
# than silently reporting a fabricated MDE.
_CHI_SQUARE_METRICS = (EffectSizeMetric.CRAMERS_V, EffectSizeMetric.COHENS_W)
_LOCATION_METRICS = (EffectSizeMetric.RANK_BISERIAL,)


def annotate_detectability(
    finding: Finding,
    df: int = 1,
    alpha: float = DEFAULT_ALPHA,
    power_target: float = DEFAULT_POWER,
) -> Finding:
    """Populate power, MDE and detectability on `finding`, in place."""
    metric = finding.effect_size_metric
    counts = sorted(finding.n_per_group.values(), reverse=True)

    if metric in _CHI_SQUARE_METRICS:
        mde = mde_chi_square(n=finding.total_n, df=df, alpha=alpha, power=power_target)
        finding.statistical_power = achieved_power_chi_square(
            effect_size=abs(finding.effect_size_value),
            n=finding.total_n,
            df=df,
            alpha=alpha,
        )
    elif metric in _LOCATION_METRICS and len(counts) >= 2:
        mde = mde_rank_biserial(
            n1=counts[0], n2=counts[1], alpha=alpha, power=power_target
        )
        finding.statistical_power = None
    else:
        finding.minimum_detectable_effect = None
        finding.detectability = Detectability.UNKNOWN
        return finding

    finding.minimum_detectable_effect = mde
    medium_cutoff = thresholds_for(metric, df_min=df).medium
    finding.detectability = (
        Detectability.ADEQUATE if mde <= medium_cutoff else Detectability.UNDERPOWERED
    )
    return finding
```

```python
# src/dbias/detectability/__init__.py
from dbias.detectability.classify import annotate_detectability
from dbias.detectability.power import (
    DEFAULT_ALPHA,
    DEFAULT_POWER,
    achieved_power_chi_square,
    mde_chi_square,
    mde_rank_biserial,
)

__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_POWER",
    "achieved_power_chi_square",
    "annotate_detectability",
    "mde_chi_square",
    "mde_rank_biserial",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/detectability/test_classify.py -v` then `pytest -v`
Expected: PASS — 7 passed in the new file, 48 passed overall.

- [ ] **Step 5: Commit**

```bash
git add src/dbias/detectability tests/unit/detectability
git commit -m "feat: classify null results as adequate or blind spot"
```

---

## Out of Scope for This Plan

These are separate plans, each producing working software on its own. They are listed so an executor knows where the boundary is — do not start them from this document.

1. **Ingestion and profiling** — loaders, semantic dtype inference, sensitive-column detection, `DatasetProfile` population.
2. **Analyzers** — `BaseAnalyzer` plus representation, missingness, feature-disparity and label-disparity implementations, emitting un-scored `Finding` objects.
3. **Gated hierarchical FDR** — power-guided descent into intersectional subgroups, wrapping Task 5's flat correction. Depends on this plan's Task 6 for the gating signal.
4. **Rules engine** — maps `(is_significant, magnitude, detectability)` onto `Severity`, and assembles the Risk + Coverage vector.
5. **Synthetic evaluation harness** — Scenarios A (run twice: well-powered and underpowered), B, C, and the new D.
6. **Propagation harness** — counterfactual dataset repair, surrogate model training, downstream fairness delta attribution.
7. **Dashboard** — Streamlit UI, findings feed, coverage map.

## Self-Review Notes

- **Spec coverage:** docs/02's decision tree is Task 4; its BH-FDR requirement is Task 5; its effect size definitions are Task 2. docs/04's `Finding` and `DatasetProfile` are Task 1. docs/05's severity ladder is deliberately *not* implemented here — it belongs to the out-of-scope rules engine, and its threshold defect is fixed in Task 3.
- **Known gap:** `docs/09_detectability.md` does not exist yet. This plan is self-contained without it (the design decisions are inlined in Global Constraints), but the rationale should be written up before the rules-engine plan consumes `Detectability`.
- **Type consistency:** `EffectSizeMetric` members are referenced identically across Tasks 2, 3, 4, 6 and 7. `mde_chi_square(n, df, alpha, power)` and `mde_rank_biserial(n1, n2, alpha, power)` keep the signatures declared in Task 6's Interfaces block when called in Task 7. `thresholds_for(metric, df_min)` is called with the keyword `df_min=df` in Task 7, matching Task 3.
- **Deliberate asymmetry:** `statistical_power` is set to `None` on the rank-biserial path because no achieved-power solver is defined for it in Task 6. Task 7's tests assert this rather than papering over it.
