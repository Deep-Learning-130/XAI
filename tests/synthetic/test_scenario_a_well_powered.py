"""Scenario A, well powered: a true null the audit is entitled to report.

Ground truth: missingness rates are genuinely identical across groups, and n
is large enough that an effect at the SESOI would have been caught. The
correct output is an *earned* all-clear -- a positive claim that effects at or
above the SESOI are ruled out, not merely an absence of evidence.

Its twin lives in test_scenario_a_underpowered.py. Same null hypothesis, same
proportions, different n, opposite correct answers. docs/10 Sec 3 keeps them
in separate files because that pair is the clearest statement of what this
tool does that others do not.
"""
from conftest import SESOI, audit_frame, missingness_finding

from dbias.models.enums import Detectability, EquivalenceVerdict, Severity
from dbias.synthetic import make_missingness_frame

N = 8_000
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
    assert missingness_finding(result()).is_significant is False


def test_the_null_is_an_earned_all_clear_not_a_blind_spot():
    assert missingness_finding(result()).severity is Severity.INFORMATIONAL


def test_the_audit_can_rule_out_an_effect_at_the_sesoi():
    finding = missingness_finding(result())
    assert finding.equivalence_verdict is EquivalenceVerdict.EQUIVALENT
    assert finding.effect_size_ci[1] < SESOI


def test_the_test_was_adequately_powered():
    finding = missingness_finding(result())
    assert finding.detectability is Detectability.ADEQUATE
    assert finding.minimum_detectable_effect < SESOI
    assert finding.power_to_detect_sesoi >= 0.80
