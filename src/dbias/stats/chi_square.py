"""Pearson chi-square test of independence, packaged with its effect size.

Deliberately does not know: what the rows and columns mean. No parameter here
is named sensitive, protected, or group (docs/10 Sec 2, invariant 1).

Per the plan's global constraints, no function returns a p-value without an
accompanying effect size, so the result object carries both.
"""
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from dbias.stats.effect_sizes import (
    _as_table,
    chi_square_statistic,
    cramers_v_corrected,
    df_min,
)

# Below this expected cell count the chi-square approximation degrades.
MIN_EXPECTED_COUNT = 5.0
# Above this worst-case margin ratio the analytic MDE errs optimistic.
MAX_MARGIN_RATIO = 4.0


@dataclass(frozen=True)
class GofSample:
    """A one-dimensional sample: category counts against a reference.

    Exists so that callers can hand a goodness-of-fit question to the same
    detectability machinery that handles contingency tables, without the
    machinery having to guess which kind of question a bare array represents.
    """

    counts: tuple[float, ...]
    expected_probs: tuple[float, ...] | None = None


@dataclass(frozen=True)
class ChiSquareResult:
    """What the sample showed, plus the diagnostics detectability/ needs."""

    statistic: float
    p_value: float
    dof: int
    df_min: int
    effect_size: float  # bias-corrected Cramer's V (Bergsma 2013) for independence; Cohen's w for GoF
    n: int
    min_expected: float
    margin_ratio: float

    @property
    def is_sparse(self) -> bool:
        """True when the chi-square approximation is unreliable."""
        return self.min_expected < MIN_EXPECTED_COUNT

    @property
    def is_skewed(self) -> bool:
        """True when one margin dwarfs another.

        In this regime effective power is governed by the smaller cell rather
        than by the total, and the analytic MDE is optimistic (plan.md Sec 3.4).
        """
        return self.margin_ratio > MAX_MARGIN_RATIO


def chi_square_test(table: ArrayLike) -> ChiSquareResult:
    """Run the test without Yates' correction.

    The continuity correction is deliberately omitted: Cramer's V and the
    non-central chi-square used for the MDE are both defined on the
    uncorrected statistic, and mixing the two would make the reported effect
    size inconsistent with the reported detectability.

    The effect size is the Bergsma (2013) bias-corrected V; the plug-in V
    overstates association on sparse tables.
    """
    arr = _as_table(table)
    n = int(arr.sum())
    rows, cols = arr.shape
    dof = (rows - 1) * (cols - 1)
    if dof < 1:
        raise ValueError("chi-square independence needs at least a 2x2 table")

    statistic = chi_square_statistic(arr)
    p_value = float(stats.chi2.sf(statistic, dof))

    expected = np.outer(arr.sum(axis=1), arr.sum(axis=0)) / n
    margins = np.concatenate([arr.sum(axis=1), arr.sum(axis=0)])
    positive = margins[margins > 0]
    margin_ratio = float(positive.max() / positive.min()) if positive.size else float("inf")

    return ChiSquareResult(
        statistic=statistic,
        p_value=p_value,
        dof=dof,
        df_min=df_min(arr),
        effect_size=cramers_v_corrected(arr),
        n=n,
        min_expected=float(expected.min()),
        margin_ratio=margin_ratio,
    )


def chi_square_goodness_of_fit(
    counts: ArrayLike, expected_probs: ArrayLike | None = None
) -> ChiSquareResult:
    """Test observed category counts against a reference distribution.

    Representation is a one-dimensional question -- are these shares what they
    should be? -- so it has no second margin to normalise against. The reported
    effect size is therefore Cohen's w unadjusted, and `df_min` is 1 so that
    downstream code treating the effect size as Cramer's V gets the same number.

    `expected_probs` defaults to uniform. A uniform reference is a convention,
    not a claim about what a population looks like; callers auditing against a
    known population should pass its shares explicitly.
    """
    observed = np.asarray(counts, dtype=float)
    if observed.ndim != 1:
        raise ValueError(f"expected a 1-D vector of counts, got ndim={observed.ndim}")
    if observed.size < 2:
        raise ValueError("goodness of fit needs at least two categories")
    if np.any(observed < 0):
        raise ValueError("counts cannot be negative")
    n = observed.sum()
    if n <= 0:
        raise ValueError("no observations; there is nothing to measure")

    if expected_probs is None:
        probs = np.full(observed.size, 1.0 / observed.size)
    else:
        probs = np.asarray(expected_probs, dtype=float)
        if probs.shape != observed.shape:
            raise ValueError("expected_probs must have one entry per category")
        if np.any(probs <= 0):
            raise ValueError("expected_probs must be strictly positive")
        if not np.isclose(probs.sum(), 1.0):
            raise ValueError(f"expected_probs must sum to 1, got {probs.sum()}")

    expected = probs * n
    statistic = float((((observed - expected) ** 2) / expected).sum())
    dof = observed.size - 1

    positive = observed[observed > 0]
    margin_ratio = float(positive.max() / positive.min()) if positive.size else float("inf")

    return ChiSquareResult(
        statistic=statistic,
        p_value=float(stats.chi2.sf(statistic, dof)),
        dof=dof,
        df_min=1,
        effect_size=float(np.sqrt(statistic / n)),
        n=int(n),
        min_expected=float(expected.min()),
        margin_ratio=margin_ratio,
    )
