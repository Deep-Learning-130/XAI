"""Scenario A, underpowered: the same null, too little data to claim it.

Ground truth is identical to test_scenario_a_well_powered.py -- the same true
null, the same 50/50 proportions, the same missingness rates. Only n differs.

An incumbent tool prints the same sentence for both files. That is the defect
this project exists to fix, so the assertion here is the mirror image of the
assertion there: not "a disparity was found", but "this result rules nothing
out and must not be read as clean".
"""
from conftest import SESOI, audit_frame, missingness_finding

from dbias.models.enums import Detectability, EquivalenceVerdict, Severity
from dbias.synthetic import make_missingness_frame

N = 80
TRUE_EFFECT = 0.0


def result():
    frame = make_missingness_frame(
        n=N,
        group_shares={"a": 0.5, "b": 0.5},
        missing_rates={"a": 0.2, "b": 0.2},
        seed=101,
    )
    return audit_frame(frame)


def test_no_disparity_is_reported():
    """Exactly as in the well-powered twin."""
    assert missingness_finding(result()).is_significant is False


def test_the_null_is_reported_as_a_blind_spot():
    assert missingness_finding(result()).severity is Severity.BLIND_SPOT


def test_the_audit_cannot_rule_out_an_effect_at_the_sesoi():
    finding = missingness_finding(result())
    assert finding.equivalence_verdict is EquivalenceVerdict.INCONCLUSIVE
    assert finding.effect_size_ci[1] > SESOI


def test_the_test_was_underpowered():
    finding = missingness_finding(result())
    assert finding.detectability is Detectability.UNDERPOWERED
    assert finding.minimum_detectable_effect > SESOI
    assert finding.power_to_detect_sesoi < 0.80


def test_the_p_value_is_indistinguishable_from_the_well_powered_case():
    """Both are non-significant. The p-value cannot tell these two datasets
    apart; only the detectability annotation can."""
    assert missingness_finding(result()).is_significant is False
