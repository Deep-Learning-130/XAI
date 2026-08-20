"""Contract tests for the Finding dataclass.

Oracles here are the specification itself (roadmap M0 "done when", plan.md
Sec 3.1), not computed values -- this layer is inert data with zero logic.
"""
import dataclasses

import pytest

from dbias.models.enums import (
    Category,
    Detectability,
    EffectSizeMetric,
    EquivalenceVerdict,
    Severity,
)
from dbias.models.finding import Finding


def make_finding(**overrides) -> Finding:
    kwargs = dict(
        id="REPRESENTATION_GENDER",
        category=Category.REPRESENTATION,
        sensitive_attribute="gender",
        target_feature=None,
        metric_name="Group share vs reference",
        observed_values={"male": 0.67, "female": 0.33},
        statistical_test="Chi-square goodness of fit",
        p_value_raw=0.42,
        effect_size_metric=EffectSizeMetric.CRAMERS_V,
        effect_size_value=0.03,
        n_per_group={"male": 670, "female": 330},
    )
    kwargs.update(overrides)
    return Finding(**kwargs)


def test_finding_without_severity_is_undetermined():
    assert make_finding().severity is Severity.UNDETERMINED


def test_finding_without_detectability_pass_is_unknown():
    assert make_finding().detectability is Detectability.UNKNOWN


def test_total_n_sums_group_counts():
    assert make_finding(n_per_group={"a": 40, "b": 60, "c": 5}).total_n == 105


def test_equivalence_fields_default_to_none():
    f = make_finding()
    assert f.effect_size_ci is None
    assert f.equivalence_verdict is None
    assert f.sesoi is None


def test_mde_defaults_to_none_and_not_flagged_approximate():
    f = make_finding()
    assert f.minimum_detectable_effect is None
    assert f.mde_is_approximate is False


def test_observed_power_field_does_not_exist():
    """plan.md Sec 3.1: post-hoc observed power is a fallacy and the code path
    is deleted, not deprecated. Guard against it being reintroduced."""
    names = {f.name for f in dataclasses.fields(Finding)}
    assert "statistical_power" not in names
    assert "achieved_power" not in names
    assert "power_to_detect_sesoi" in names


def test_equivalence_verdict_accepts_enum_values():
    f = make_finding(equivalence_verdict=EquivalenceVerdict.EQUIVALENT, sesoi=0.1)
    assert f.equivalence_verdict is EquivalenceVerdict.EQUIVALENT
    assert f.sesoi == pytest.approx(0.1)
