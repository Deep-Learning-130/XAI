"""Chi-square test of independence.

Oracles: the statistic is hand-computed; the p-value is checked against
published chi-square table values (chi2 = 6.635 at df = 1 corresponds to
p = 0.01), not against a recomputation.
"""
import numpy as np
import pytest

from dbias.stats.chi_square import chi_square_test

BALANCED_2X2 = np.array([[10, 20], [20, 10]])
INDEPENDENT_2X3 = np.array([[10, 10, 10], [20, 20, 20]])
SKEWED_2X2 = np.array([[950, 50], [45, 5]])


def test_statistic_matches_hand_computed_value():
    assert chi_square_test(BALANCED_2X2).statistic == pytest.approx(20 / 3, abs=1e-12)


def test_no_yates_correction_is_applied():
    """Yates would give 4*(4.5^2/15) = 5.4, not 20/3. Cramer's V and the MDE
    are both defined on the uncorrected statistic, so the two must agree."""
    assert chi_square_test(BALANCED_2X2).statistic != pytest.approx(5.4)


def test_p_value_brackets_the_published_table_value():
    """chi2 = 6.667 sits just above the df=1 critical value of 6.635 (p=0.01)."""
    p = chi_square_test(BALANCED_2X2).p_value
    assert 0.009 < p < 0.011


def test_degrees_of_freedom_are_row_minus_one_times_col_minus_one():
    assert chi_square_test(INDEPENDENT_2X3).dof == 2
    assert chi_square_test(BALANCED_2X2).dof == 1


def test_df_min_is_the_cramers_v_divisor():
    assert chi_square_test(INDEPENDENT_2X3).df_min == 1
    assert chi_square_test(np.array([[9, 1, 1], [1, 9, 1], [1, 1, 9]])).df_min == 2


def test_exact_independence_gives_p_of_one():
    result = chi_square_test(INDEPENDENT_2X3)
    assert result.statistic == pytest.approx(0.0, abs=1e-12)
    assert result.p_value == pytest.approx(1.0)


def test_effect_size_is_cramers_v():
    assert chi_square_test(BALANCED_2X2).effect_size == pytest.approx(1 / 3, abs=1e-12)


def test_sparse_table_is_flagged_for_the_chi_square_approximation():
    """min(expected) = 50*50/1050 = 2.38 < 5, so the approximation degrades."""
    assert chi_square_test(SKEWED_2X2).min_expected < 5
    assert chi_square_test(SKEWED_2X2).is_sparse is True
    assert chi_square_test(BALANCED_2X2).is_sparse is False


def test_margin_ratio_reports_the_worst_row_imbalance():
    """Rows are 1000 and 50, so the ratio is 20:1 -- the regime where the
    analytic MDE errs optimistic (plan.md Sec 3.4)."""
    assert chi_square_test(SKEWED_2X2).margin_ratio == pytest.approx(20.0)
    assert chi_square_test(BALANCED_2X2).margin_ratio == pytest.approx(1.0)


def test_skewed_margins_are_flagged():
    assert chi_square_test(SKEWED_2X2).is_skewed is True
    assert chi_square_test(BALANCED_2X2).is_skewed is False


# --- goodness of fit ---------------------------------------------------------
# Representation is a one-dimensional question -- "are these group shares what
# they should be?" -- so it needs a goodness-of-fit test rather than a test of
# independence. Oracle: counts [30, 10] against a uniform reference gives
# expected [20, 20], chi2 = 2 * (10^2 / 20) = 10, n = 40, w = sqrt(10/40) = 0.5.

from dbias.stats.chi_square import chi_square_goodness_of_fit  # noqa: E402


def test_gof_statistic_matches_hand_computed_value():
    assert chi_square_goodness_of_fit([30, 10]).statistic == pytest.approx(10.0, abs=1e-12)


def test_gof_effect_size_is_cohens_w():
    assert chi_square_goodness_of_fit([30, 10]).effect_size == pytest.approx(0.5, abs=1e-12)


def test_gof_degrees_of_freedom_are_k_minus_one():
    assert chi_square_goodness_of_fit([10, 10, 10, 10]).dof == 3


def test_gof_reports_df_min_of_one_so_v_and_w_coincide():
    """There is no second dimension to normalise against, so the reported
    effect size is Cohen's w unadjusted."""
    assert chi_square_goodness_of_fit([10, 10, 10, 10]).df_min == 1


def test_gof_against_a_uniform_reference_by_default():
    assert chi_square_goodness_of_fit([25, 25, 25, 25]).statistic == pytest.approx(0.0)


def test_gof_accepts_an_explicit_reference_distribution():
    """Expected [36, 4]; chi2 = 6^2/36 + 6^2/4 = 1 + 9 = 10."""
    result = chi_square_goodness_of_fit([30, 10], expected_probs=[0.9, 0.1])
    assert result.statistic == pytest.approx(10.0, abs=1e-12)


def test_gof_rejects_a_reference_that_does_not_sum_to_one():
    with pytest.raises(ValueError):
        chi_square_goodness_of_fit([30, 10], expected_probs=[0.9, 0.9])


def test_gof_flags_sparse_expected_cells():
    """Expected [98, 2]: the 2 is below the count where chi2 is trustworthy."""
    assert chi_square_goodness_of_fit([50, 50], expected_probs=[0.98, 0.02]).is_sparse
