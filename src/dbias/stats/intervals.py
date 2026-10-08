"""Bootstrap confidence intervals for contingency-table effect sizes.

Deliberately does not know: what a disparity is, or what counts as a
meaningful effect. It returns an interval; deciding whether that interval
rules something out is detectability/classify.py's job.

plan.md Sec 3.2: the CI is the inference layer of this project. A CI whose
upper bound sits below the SESOI is an earned all-clear; one whose upper bound
sits above it is a blind spot. MDE remains the communication layer.

Method: non-parametric bootstrap over the multinomial defined by the observed
cell proportions, holding n fixed. Contingency tables use the Bergsma (2013)
bias-corrected Cramer's V and a percentile interval recentred by the
bootstrap's estimated bias; the plug-in V is biased upward near the null,
which used to put whole intervals above the SESOI on sparse tables. The
estimator is still truncated at zero, so near the null the lower bound is
reported for completeness and the upper bound carries the inference.
"""
import numpy as np
from numpy.typing import ArrayLike

from dbias.stats.effect_sizes import _as_table, cramers_v_corrected, cramers_v_corrected_many

DEFAULT_RESAMPLES = 2000


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


def bootstrap_ci_gof_w(
    counts: ArrayLike,
    expected_probs: ArrayLike | None = None,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap interval for Cohen's w on a goodness-of-fit test.

    Resamples the n observations from the observed category shares, holding the
    reference distribution fixed -- the reference is an assumption, not data,
    so it does not carry sampling error.
    """
    observed = np.asarray(counts, dtype=float)
    if observed.ndim != 1 or observed.size < 2:
        raise ValueError("expected a 1-D vector of at least two category counts")
    n = int(observed.sum())
    if n <= 0:
        raise ValueError("no observations; there is nothing to measure")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")

    if expected_probs is None:
        probs = np.full(observed.size, 1.0 / observed.size)
    else:
        probs = np.asarray(expected_probs, dtype=float)

    rng = np.random.default_rng(seed)
    draws = rng.multinomial(n, observed / n, size=n_resamples)

    expected = probs * n
    statistics = (((draws - expected) ** 2) / expected).sum(axis=1)
    estimates = np.sqrt(statistics / n)

    tail = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(estimates, [tail, 1.0 - tail])
    return (float(lo), float(hi))
