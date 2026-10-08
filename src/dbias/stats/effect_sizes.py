"""Effect sizes for contingency tables.

Deliberately does not know: what a protected attribute is, which group is
disadvantaged, or whether any of this constitutes bias. It receives a table of
counts and returns a number (docs/10 Sec 2, invariant 1).
"""
import numpy as np
from numpy.typing import ArrayLike


def _as_table(table: ArrayLike) -> np.ndarray:
    arr = np.asarray(table, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D contingency table, got ndim={arr.ndim}")
    if np.any(arr < 0):
        raise ValueError("contingency tables cannot contain negative counts")
    if arr.sum() <= 0:
        raise ValueError("contingency table is empty; there is nothing to measure")
    return arr


def chi_square_statistic(table: ArrayLike) -> float:
    """Pearson chi-square against the independence model of the observed margins."""
    arr = _as_table(table)
    n = arr.sum()
    expected = np.outer(arr.sum(axis=1), arr.sum(axis=0)) / n
    # A zero margin contributes a zero row/column to both observed and expected.
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(expected > 0, (arr - expected) ** 2 / expected, 0.0)
    return float(terms.sum())


def cohens_w(table: ArrayLike) -> float:
    """w = sqrt(chi2 / n). Unbounded above for tables larger than 2x2."""
    arr = _as_table(table)
    return float(np.sqrt(chi_square_statistic(arr) / arr.sum()))


def df_min(table: ArrayLike) -> int:
    """min(rows - 1, cols - 1): the divisor in the Cramer's V normalisation."""
    arr = _as_table(table)
    return int(min(arr.shape[0] - 1, arr.shape[1] - 1))


def cramers_v(table: ArrayLike) -> float:
    """V = w / sqrt(min(r-1, k-1)). Bounded [0, 1]; 1 means perfect association."""
    arr = _as_table(table)
    k = df_min(arr)
    if k < 1:
        raise ValueError("Cramer's V is undefined for a table with a single row or column")
    return float(cohens_w(arr) / np.sqrt(k))


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


def w_from_v(v: float, df_min: int) -> float:
    """Convert Cramer's V to Cohen's w. OVERRIDE 2 in the implementation plan.

    Power calculations are defined on w, thresholds are quoted on V, and the
    two differ by sqrt(min(r-1, k-1)) for anything bigger than a 2x2.
    """
    if df_min < 1:
        raise ValueError("df_min must be at least 1")
    return float(v * np.sqrt(df_min))


def v_from_w(w: float, df_min: int) -> float:
    """Inverse of :func:`w_from_v`."""
    if df_min < 1:
        raise ValueError("df_min must be at least 1")
    return float(w / np.sqrt(df_min))
