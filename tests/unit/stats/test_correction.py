"""Benjamini-Hochberg, cross-checked against scipy -- never against itself."""
import numpy as np
import pytest
from scipy.stats import false_discovery_control

from dbias.stats.correction import benjamini_hochberg


def test_matches_hand_computed_uniform_case():
    """p_i * m / i is 0.05 for every i here, so all adjusted values collapse."""
    adjusted = benjamini_hochberg([0.01, 0.02, 0.03, 0.04, 0.05])
    assert adjusted == pytest.approx([0.05] * 5)


def test_matches_scipy_on_a_random_vector():
    rng = np.random.default_rng(20260820)
    p = rng.uniform(0, 1, size=50)
    assert benjamini_hochberg(p) == pytest.approx(false_discovery_control(p, method="bh"))


def test_matches_scipy_on_a_mixture_of_signal_and_null():
    p = [1e-6, 1e-4, 0.003, 0.02, 0.2, 0.5, 0.7, 0.9, 0.95, 0.99]
    assert benjamini_hochberg(p) == pytest.approx(false_discovery_control(p, method="bh"))


def test_preserves_input_order():
    p = [0.9, 0.01, 0.5]
    adjusted = benjamini_hochberg(p)
    assert adjusted[1] < adjusted[2] < adjusted[0] or adjusted[1] < adjusted[2] == adjusted[0]
    assert adjusted == pytest.approx(false_discovery_control(p, method="bh"))


def test_adjusted_values_never_exceed_one():
    assert max(benjamini_hochberg([0.9, 0.95, 0.99])) <= 1.0


def test_empty_input_returns_empty():
    assert benjamini_hochberg([]) == []


def test_rejects_values_outside_the_unit_interval():
    with pytest.raises(ValueError):
        benjamini_hochberg([0.5, 1.5])
