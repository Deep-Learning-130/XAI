"""Per-category risk roll-up, and the coverage vector beside it.

docs/05 Sec 2 forbids a single aggregate bias score. This module must never
produce one -- and it must publish a coverage vector alongside the risk
vector, because a risk vector on its own cannot distinguish a category that
was checked and found clean from one that could not be checked at all.
"""
import pytest

from dbias.models.enums import (
    Category,
    Detectability,
    EffectSizeMetric,
    Severity,
)
from dbias.models.finding import Finding
from dbias.rules.risk_vector import coverage_vector, risk_vector, summarise


def finding(category, severity, detectability) -> Finding:
    return Finding(
        id=f"{category}_{severity}",
        category=category,
        sensitive_attribute="gender",
        target_feature="income",
        metric_name="metric",
        observed_values={},
        statistical_test="Chi-square test of independence",
        p_value_raw=0.5,
        effect_size_metric=EffectSizeMetric.CRAMERS_V,
        effect_size_value=0.1,
        n_per_group={"m": 50, "f": 50},
        severity=severity,
        detectability=detectability,
    )


SAMPLE = [
    finding(Category.MISSINGNESS, Severity.HIGH, Detectability.ADEQUATE),
    finding(Category.MISSINGNESS, Severity.INFORMATIONAL, Detectability.ADEQUATE),
    finding(Category.LABEL_DISPARITY, Severity.BLIND_SPOT, Detectability.UNDERPOWERED),
    finding(Category.REPRESENTATION, Severity.LOW, Detectability.ADEQUATE),
]


# --- risk --------------------------------------------------------------------

def test_risk_vector_reports_the_worst_severity_per_category():
    risks = risk_vector(SAMPLE)
    assert risks[Category.MISSINGNESS] is Severity.HIGH
    assert risks[Category.REPRESENTATION] is Severity.LOW


def test_categories_with_no_findings_are_absent_rather_than_clean():
    """Reporting FEATURE_DISPARITY as INFORMATIONAL when it was never tested
    would be exactly the false reassurance this tool exists to prevent."""
    assert Category.FEATURE_DISPARITY not in risk_vector(SAMPLE)


def test_blind_spot_is_the_reported_risk_when_nothing_else_was_found():
    assert risk_vector(SAMPLE)[Category.LABEL_DISPARITY] is Severity.BLIND_SPOT


def test_there_is_no_aggregate_score():
    """docs/05 Sec 2: a single dataset bias score is mathematically
    meaningless and is never produced."""
    summary = summarise(SAMPLE)
    assert "overall_score" not in summary
    assert "bias_score" not in summary
    assert not any(
        isinstance(v, float) and k.endswith("score") for k, v in summary.items()
    )


# --- coverage ----------------------------------------------------------------

def test_coverage_vector_reports_the_adequately_powered_share_per_category():
    coverage = coverage_vector(SAMPLE)
    assert coverage[Category.MISSINGNESS] == pytest.approx(1.0)
    assert coverage[Category.LABEL_DISPARITY] == pytest.approx(0.0)


def test_coverage_of_a_half_seen_category():
    findings = [
        finding(Category.MISSINGNESS, Severity.INFORMATIONAL, Detectability.ADEQUATE),
        finding(Category.MISSINGNESS, Severity.BLIND_SPOT, Detectability.UNDERPOWERED),
    ]
    assert coverage_vector(findings)[Category.MISSINGNESS] == pytest.approx(0.5)


def test_empty_cells_do_not_count_against_coverage():
    """A cell with no rows was never a hypothesis; it neither confirms nor
    denies that the category was checkable."""
    findings = [
        finding(Category.MISSINGNESS, Severity.INFORMATIONAL, Detectability.ADEQUATE),
        finding(Category.MISSINGNESS, Severity.INFORMATIONAL, Detectability.EMPTY),
    ]
    assert coverage_vector(findings)[Category.MISSINGNESS] == pytest.approx(1.0)


def test_summary_carries_risk_and_coverage_together():
    summary = summarise(SAMPLE)
    assert set(summary["risk"]) == set(summary["coverage"])
    assert summary["blind_spot_count"] == 1
    assert summary["finding_count"] == 4


def test_empty_input_yields_empty_vectors():
    assert risk_vector([]) == {}
    assert coverage_vector([]) == {}


def test_an_adequately_powered_blind_spot_is_not_covered():
    """Regression: power looked adequate but the interval ruled nothing out.
    Counting it printed 'risk=Blind Spot coverage=100%'."""
    findings = [finding(Category.MISSINGNESS, Severity.BLIND_SPOT, Detectability.ADEQUATE)]
    assert coverage_vector(findings)[Category.MISSINGNESS] == pytest.approx(0.0)


def test_an_underpowered_cell_whose_interval_earned_an_all_clear_is_covered():
    """The interval outranks the power calculation in both directions."""
    findings = [finding(Category.MISSINGNESS, Severity.INFORMATIONAL, Detectability.UNDERPOWERED)]
    assert coverage_vector(findings)[Category.MISSINGNESS] == pytest.approx(1.0)
