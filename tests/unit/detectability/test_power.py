"""Power and minimum detectable effect for the chi-square test.

Independent oracles used here:
  * Under w = 0 the non-centrality is zero, so power collapses to exactly
    alpha. Closed form, no table needed.
  * Cohen (1988) Table 7.3.6 / G*Power: for w = 0.30, df = 1, alpha = 0.05,
    n = 87 gives power near 0.80.
  * The non-centrality is n * w^2 and the required non-centrality for a fixed
    (power, df, alpha) is a constant, so MDE scales exactly as n^(-1/2).
    Quadrupling n must halve the MDE. This is an exact algebraic oracle.
"""
import pytest

from dbias.detectability.power import (
    max_reachable_w_2x2,
    mde_chi_square,
    mde_cramers_v,
    power_chi_square,
    simulate_mde_2x2,
    simulate_power_2x2,
)


def test_zero_effect_gives_power_equal_to_alpha():
    assert power_chi_square(w=0.0, n=1000, dof=1, alpha=0.05) == pytest.approx(0.05, abs=1e-9)
    assert power_chi_square(w=0.0, n=50, dof=4, alpha=0.01) == pytest.approx(0.01, abs=1e-9)


def test_matches_cohens_published_sample_size():
    """w = 0.30, df = 1, n = 87 is the textbook 0.80-power point."""
    assert power_chi_square(w=0.30, n=87, dof=1) == pytest.approx(0.80, abs=0.02)


def test_power_increases_with_sample_size():
    small = power_chi_square(w=0.2, n=100, dof=1)
    large = power_chi_square(w=0.2, n=1000, dof=1)
    assert small < large < 1.0


def test_power_increases_with_effect_size():
    assert power_chi_square(w=0.1, n=200, dof=1) < power_chi_square(w=0.4, n=200, dof=1)


def test_power_decreases_as_degrees_of_freedom_grow():
    """Spreading the same non-centrality over more cells costs power."""
    assert power_chi_square(w=0.3, n=100, dof=1) > power_chi_square(w=0.3, n=100, dof=6)


def test_mde_round_trips_through_the_power_function():
    w = mde_chi_square(n=250, dof=2, target_power=0.80)
    assert power_chi_square(w=w, n=250, dof=2) == pytest.approx(0.80, abs=1e-6)


def test_mde_matches_cohens_published_value():
    assert mde_chi_square(n=87, dof=1, target_power=0.80) == pytest.approx(0.30, abs=0.01)


def test_mde_scales_as_inverse_square_root_of_n():
    """Exact algebraic oracle: w_mde = sqrt(lambda* / n)."""
    at_100 = mde_chi_square(n=100, dof=1)
    at_400 = mde_chi_square(n=400, dof=1)
    assert at_400 == pytest.approx(at_100 / 2.0, rel=1e-6)


def test_mde_shrinks_as_n_grows():
    assert mde_chi_square(n=10_000, dof=1) < mde_chi_square(n=100, dof=1)


def test_mde_in_cramers_v_units_applies_the_df_adjustment():
    """V = w / sqrt(df_min)."""
    w = mde_chi_square(n=500, dof=4)
    assert mde_cramers_v(n=500, dof=4, df_min=2) == pytest.approx(w / 2**0.5, rel=1e-9)


def test_mde_is_identical_in_both_units_for_a_two_by_two():
    assert mde_cramers_v(n=500, dof=1, df_min=1) == pytest.approx(
        mde_chi_square(n=500, dof=1), rel=1e-12
    )


def test_higher_target_power_demands_a_larger_effect():
    assert mde_chi_square(n=200, dof=1, target_power=0.95) > mde_chi_square(
        n=200, dof=1, target_power=0.80
    )


def test_zero_sample_size_is_rejected():
    with pytest.raises(ValueError):
        mde_chi_square(n=0, dof=1)


def test_power_function_takes_no_observed_effect():
    """plan.md Sec 3.1: power is computed against a pre-specified effect only.
    There is no signature by which an observed effect could be passed in."""
    import inspect

    params = set(inspect.signature(power_chi_square).parameters)
    assert params == {"w", "n", "dof", "alpha"}


# --- the 2x2 simulation fallback ---------------------------------------------

def test_reachable_w_is_one_for_balanced_margins():
    """phi = 1 is the perfect 2x2 association, reachable only at 50/50."""
    assert max_reachable_w_2x2(0.5, 0.5) == pytest.approx(1.0)
    assert max_reachable_w_2x2(0.05, 0.2) < 1.0


def test_simulated_mde_is_a_real_effect_not_the_search_ceiling():
    """Regression: searching up to w = 5 always failed, so every skewed 2x2
    reported MDE = 5.0."""
    mde = simulate_mde_2x2(n=8_000, minority_share=0.2, base_rate=0.5, seed=0)
    assert mde < 0.1
    assert simulate_power_2x2(mde, 8_000, 0.2, 0.5, seed=0) == pytest.approx(0.80, abs=0.03)


def test_simulated_mde_agrees_with_the_analytic_one_where_both_hold():
    analytic = mde_chi_square(n=2_000, dof=1)
    simulated = simulate_mde_2x2(n=2_000, minority_share=0.5, base_rate=0.5, seed=0)
    assert simulated == pytest.approx(analytic, rel=0.05)


def test_simulated_mde_reports_the_ceiling_when_nothing_reachable_is_detectable():
    assert simulate_mde_2x2(n=20, minority_share=0.05, base_rate=0.1, seed=0) == 5.0
