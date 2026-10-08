"""Bootstrap confidence intervals on Cramer's V.

plan.md Sec 3.2: the CI is the inference layer. Every property asserted here
is a distributional fact about the bootstrap, checked against closed forms
where one exists.
"""
import numpy as np
import pytest

from dbias.stats.effect_sizes import cramers_v_corrected
from dbias.stats.intervals import bootstrap_ci_cramers_v

INDEPENDENT_LARGE = np.array([[2500, 2500], [2500, 2500]])
INDEPENDENT_SMALL = np.array([[25, 25], [25, 25]])
STRONG_2X2 = np.array([[400, 100], [100, 400]])


def test_interval_is_deterministic_under_a_fixed_seed():
    a = bootstrap_ci_cramers_v(STRONG_2X2, n_resamples=400, seed=7)
    b = bootstrap_ci_cramers_v(STRONG_2X2, n_resamples=400, seed=7)
    assert a == b


def test_different_seeds_give_different_intervals():
    a = bootstrap_ci_cramers_v(STRONG_2X2, n_resamples=400, seed=1)
    b = bootstrap_ci_cramers_v(STRONG_2X2, n_resamples=400, seed=2)
    assert a != b


def test_interval_brackets_the_point_estimate_for_a_strong_effect():
    lo, hi = bootstrap_ci_cramers_v(STRONG_2X2, n_resamples=1000, seed=11)
    assert lo < cramers_v_corrected(STRONG_2X2) < hi


def test_interval_is_ordered_and_non_negative():
    lo, hi = bootstrap_ci_cramers_v(INDEPENDENT_SMALL, n_resamples=1000, seed=3)
    assert 0.0 <= lo <= hi <= 1.0


def test_large_n_null_interval_excludes_a_sesoi_of_one_tenth():
    """n = 10,000 under exact independence: V-hat^2 * n is roughly chi2_1, so
    the 97.5th percentile sits near sqrt(5.02 / 10000) = 0.022."""
    lo, hi = bootstrap_ci_cramers_v(INDEPENDENT_LARGE, n_resamples=2000, seed=5)
    assert hi < 0.05


def test_small_n_null_interval_cannot_exclude_a_sesoi_of_one_tenth():
    """The same proportions at n = 100: the upper bound is near
    sqrt(5.02 / 100) = 0.224, so a null result here rules out nothing.
    This pair is the entire argument of the project."""
    lo, hi = bootstrap_ci_cramers_v(INDEPENDENT_SMALL, n_resamples=2000, seed=5)
    assert hi > 0.1


def test_interval_narrows_as_n_grows():
    narrow = bootstrap_ci_cramers_v(INDEPENDENT_LARGE, n_resamples=1000, seed=9)
    wide = bootstrap_ci_cramers_v(INDEPENDENT_SMALL, n_resamples=1000, seed=9)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_confidence_level_widens_the_interval():
    lo95, hi95 = bootstrap_ci_cramers_v(STRONG_2X2, n_resamples=1000, seed=4, confidence=0.95)
    lo99, hi99 = bootstrap_ci_cramers_v(STRONG_2X2, n_resamples=1000, seed=4, confidence=0.99)
    assert (hi99 - lo99) > (hi95 - lo95)


def test_rejects_an_empty_table():
    with pytest.raises(ValueError):
        bootstrap_ci_cramers_v(np.zeros((2, 2), dtype=int), n_resamples=100, seed=1)


# --- goodness of fit ---------------------------------------------------------

from dbias.stats.intervals import bootstrap_ci_gof_w  # noqa: E402


def test_gof_interval_is_deterministic_under_a_fixed_seed():
    a = bootstrap_ci_gof_w([300, 100], n_resamples=400, seed=7)
    b = bootstrap_ci_gof_w([300, 100], n_resamples=400, seed=7)
    assert a == b


def test_gof_interval_brackets_a_strong_imbalance():
    """Shares of 0.75/0.25 against a uniform reference give w = 0.5."""
    lo, hi = bootstrap_ci_gof_w([300, 100], n_resamples=1000, seed=11)
    assert lo < 0.5 < hi


def test_gof_large_n_balanced_sample_excludes_a_sesoi_of_one_tenth():
    lo, hi = bootstrap_ci_gof_w([5000, 5000], n_resamples=1000, seed=5)
    assert hi < 0.1


def test_gof_small_n_balanced_sample_cannot_exclude_it():
    lo, hi = bootstrap_ci_gof_w([50, 50], n_resamples=1000, seed=5)
    assert hi > 0.1


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
