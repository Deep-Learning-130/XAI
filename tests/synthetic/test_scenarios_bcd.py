"""Scenarios B, C and D from the evaluation protocol (docs/06).

  B  an injected MAR disparity, large enough and with enough data  -> found
  C  a trivial effect at enormous n                                -> not alarmed at
  D  a real effect above the SESOI, with too little power          -> blind spot

D is the one that matters. B and C establish that the tool is not merely
timid; D establishes that when it fails to find something real, it says so.
"""
import numpy as np
import pytest
from conftest import SESOI, audit_frame, missingness_finding

from dbias.models.enums import Detectability, EquivalenceVerdict, Severity
from dbias.synthetic import make_missingness_frame


# --- Scenario B: a real disparity, adequately powered ------------------------

def test_b_injected_mar_disparity_is_detected():
    frame = make_missingness_frame(
        n=4_000,
        group_shares={"a": 0.5, "b": 0.5},
        missing_rates={"a": 0.10, "b": 0.30},
        seed=202,
    )
    finding = missingness_finding(audit_frame(frame))
    assert finding.is_significant
    assert finding.equivalence_verdict is EquivalenceVerdict.DISPARITY
    assert finding.severity in {Severity.LOW, Severity.MEDIUM, Severity.HIGH}


def test_b_reports_the_observed_rates_that_drove_it():
    frame = make_missingness_frame(
        n=4_000,
        group_shares={"a": 0.5, "b": 0.5},
        missing_rates={"a": 0.10, "b": 0.30},
        seed=202,
    )
    observed = missingness_finding(audit_frame(frame)).observed_values
    assert observed["a"] == pytest.approx(0.10, abs=0.03)
    assert observed["b"] == pytest.approx(0.30, abs=0.03)


# --- Scenario C: significant, and meaningless --------------------------------

def test_c_trivial_effect_at_huge_n_does_not_raise_an_alarm():
    """plan.md Sec 6. A 0.5-point rate difference over a million rows is
    overwhelmingly significant and completely uninteresting. Significance at
    this n is a statement about n."""
    frame = make_missingness_frame(
        n=1_000_000,
        group_shares={"a": 0.5, "b": 0.5},
        missing_rates={"a": 0.200, "b": 0.205},
        seed=303,
    )
    finding = missingness_finding(audit_frame(frame, n_resamples=200))
    assert finding.severity is Severity.INFORMATIONAL
    assert finding.effect_size_value < SESOI


def test_c_huge_n_earns_a_genuine_equivalence_claim():
    frame = make_missingness_frame(
        n=1_000_000,
        group_shares={"a": 0.5, "b": 0.5},
        missing_rates={"a": 0.200, "b": 0.205},
        seed=303,
    )
    finding = missingness_finding(audit_frame(frame, n_resamples=200))
    assert finding.equivalence_verdict is EquivalenceVerdict.EQUIVALENT
    assert finding.detectability is Detectability.ADEQUATE


# --- Scenario D: the one the project exists for ------------------------------

D_RATES = {"a": 0.20, "b": 0.32}  # a genuine, non-trivial disparity
D_N = 90


def test_d_the_injected_effect_genuinely_exceeds_the_sesoi():
    """Establish the ground truth before asserting anything about the verdict.
    Cohen's w for this 2x2 at equal group sizes."""
    p = np.array(list(D_RATES.values()))
    observed = np.vstack([p, 1 - p]) / 2
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    w = float(np.sqrt((((observed - expected) ** 2) / expected).sum()))
    assert w > SESOI


def _scenario_d_runs(replicates: int = 60):
    """Repeat scenario D across seeds, returning one finding per replicate.

    Asserting on a single seed would be both fragile and dishonest: at n = 90
    this effect is sometimes caught and sometimes missed, and which happens on
    one arbitrary seed is not the claim. The claim is about what the tool says
    on the runs where it misses.
    """
    for seed in range(400, 400 + replicates):
        frame = make_missingness_frame(
            n=D_N, group_shares={"a": 0.5, "b": 0.5}, missing_rates=D_RATES, seed=seed
        )
        yield missingness_finding(audit_frame(frame, n_resamples=200))


def test_d_a_real_effect_is_missed_often_at_this_sample_size():
    """Establish that the scenario is actually the hard case it claims to be."""
    findings = list(_scenario_d_runs())
    missed = [f for f in findings if not f.is_significant]
    assert len(missed) > len(findings) * 0.3


def test_d_a_missed_real_effect_is_never_reported_as_clean():
    """The whole argument, stated as a rate rather than a single seed.

    On every run where the audit fails to detect a disparity that genuinely
    exceeds the declared SESOI, it must say it could not see -- never that
    there was nothing there. One INFORMATIONAL here is a false all-clear on a
    real disparity, which is the exact failure incumbent tools ship with.
    """
    missed = [f for f in _scenario_d_runs() if not f.is_significant]
    assert missed, "scenario D never missed; it is not testing what it claims"
    assert all(f.severity is Severity.BLIND_SPOT for f in missed)
    assert not any(f.severity is Severity.INFORMATIONAL for f in missed)


def test_d_every_missed_run_reports_an_mde_above_the_sesoi():
    missed = [f for f in _scenario_d_runs() if not f.is_significant]
    assert all(f.minimum_detectable_effect > SESOI for f in missed)


def test_d_the_same_effect_is_found_once_there_is_enough_data():
    """The effect was always real; only the sample size changed."""
    frame = make_missingness_frame(
        n=4_000, group_shares={"a": 0.5, "b": 0.5}, missing_rates=D_RATES, seed=404
    )
    assert missingness_finding(audit_frame(frame)).is_significant
