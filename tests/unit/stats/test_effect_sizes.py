"""Effect size oracles are hand-computed closed forms, never this code's output.

Worked example used throughout, a 2x2 table [[10, 20], [20, 10]]:
  n = 60, every expected cell = 15, chi2 = 4 * (5^2 / 15) = 20/3
  Cohen's w  = sqrt((20/3) / 60) = 1/3
  Cramer's V = w / sqrt(min(r-1, k-1)) = 1/3
"""
import numpy as np
import pytest

from dbias.stats.effect_sizes import cohens_w, cramers_v, w_from_v

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
