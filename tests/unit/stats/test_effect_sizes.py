"""Effect size oracles are hand-computed closed forms, never this code's output.

Worked example used throughout, a 2x2 table [[10, 20], [20, 10]]:
  n = 60, every expected cell = 15, chi2 = 4 * (5^2 / 15) = 20/3
  Cohen's w  = sqrt((20/3) / 60) = 1/3
  Cramer's V = w / sqrt(min(r-1, k-1)) = 1/3
"""
import numpy as np
import pytest

from dbias.stats.effect_sizes import (
    cohens_w,
    cramers_v,
    cramers_v_corrected,
    cramers_v_corrected_many,
    w_from_v,
)

BALANCED_2X2 = np.array([[10, 20], [20, 10]])
PERFECT_3X3 = np.array([[20, 0, 0], [0, 20, 0], [0, 0, 20]])
INDEPENDENT_2X3 = np.array([[10, 10, 10], [20, 20, 20]])


def test_cohens_w_matches_hand_computed_2x2():
    assert cohens_w(BALANCED_2X2) == pytest.approx(1 / 3, abs=1e-12)


def test_cramers_v_equals_cohens_w_for_2x2():
    """min(r-1, k-1) == 1, so the df adjustment is the identity here."""
    assert cramers_v(BALANCED_2X2) == pytest.approx(1 / 3, abs=1e-12)


def test_cramers_v_is_one_for_perfect_association():
    assert cramers_v(PERFECT_3X3) == pytest.approx(1.0, abs=1e-12)


def test_cohens_w_exceeds_one_where_cramers_v_is_capped():
    """chi2 = 120, n = 60, so w = sqrt(2) -- w is not bounded by 1, V is."""
    assert cohens_w(PERFECT_3X3) == pytest.approx(np.sqrt(2), abs=1e-12)


def test_effect_sizes_are_zero_under_exact_independence():
    assert cramers_v(INDEPENDENT_2X3) == pytest.approx(0.0, abs=1e-12)
    assert cohens_w(INDEPENDENT_2X3) == pytest.approx(0.0, abs=1e-12)


def test_w_from_v_applies_the_df_adjustment():
    """OVERRIDE 2: w = V * sqrt(min(r-1, k-1))."""
    assert w_from_v(0.5, df_min=1) == pytest.approx(0.5)
    assert w_from_v(0.5, df_min=4) == pytest.approx(1.0)


def test_cramers_v_is_invariant_to_transposition():
    assert cramers_v(INDEPENDENT_2X3) == pytest.approx(cramers_v(INDEPENDENT_2X3.T))
    assert cramers_v(BALANCED_2X2) == pytest.approx(cramers_v(BALANCED_2X2.T))


def test_empty_table_raises_rather_than_returning_zero():
    """A table with no rows is an absence, not an effect size of zero."""
    with pytest.raises(ValueError):
        cramers_v(np.zeros((2, 2), dtype=int))


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
