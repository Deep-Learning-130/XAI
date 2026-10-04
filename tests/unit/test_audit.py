"""The audit pipeline: analyzers -> detectability -> correction -> rules.

The ordering matters and is asserted here. Detectability runs before severity
because severity depends on it; correction runs before severity because
significance depends on it; and nothing may skip the detectability pass
(docs/10 Sec 2, invariant 3).
"""
import numpy as np
import pandas as pd
import pytest

from dbias.audit import audit
from dbias.models.enums import Category, Detectability, Severity
from dbias.synthetic import make_audit_scenario


@pytest.fixture(scope="module")
def result():
    df, truth = make_audit_scenario(n=600, seed=11)
    return audit(
        df,
        sensitive_cols=truth["sensitive_cols"],
        target_col=truth["target_col"],
        sesoi=0.1,
        seed=7,
        n_resamples=300,
    )


def test_audit_produces_findings(result):
    assert len(result.findings) > 0


def test_every_finding_is_annotated_with_detectability(result):
    """The invariant: UNKNOWN must never reach the report layer."""
    assert all(f.detectability is not Detectability.UNKNOWN for f in result.findings)


def test_every_finding_is_scored(result):
    assert all(f.severity is not Severity.UNDETERMINED for f in result.findings)


def test_every_finding_carries_a_corrected_p_value(result):
    assert all(f.p_value_corrected is not None for f in result.findings)


def test_every_finding_carries_an_interval_and_an_mde(result):
    """The one exception is power-guided descent: an underpowered intersection
    may skip its bootstrap, and then reports no interval rather than a fake one."""
    for finding in result.findings:
        if finding.detectability is Detectability.EMPTY:
            continue
        assert finding.minimum_detectable_effect is not None
        if finding.effect_size_ci is None:
            assert "_AND_" in finding.sensitive_attribute
            assert finding.detectability is Detectability.UNDERPOWERED
            assert finding.equivalence_verdict is None


def test_the_declared_sesoi_is_recorded_on_every_finding(result):
    """plan.md Sec 3.6: a tool whose verdict is 'we could have caught anything
    that matters' must say what matters, everywhere it says it."""
    assert all(f.sesoi == pytest.approx(0.1) for f in result.findings)
    assert result.sesoi == pytest.approx(0.1)


def test_all_four_analyzers_contributed(result):
    categories = {f.category for f in result.findings}
    assert categories == {
        Category.REPRESENTATION,
        Category.MISSINGNESS,
        Category.FEATURE_DISPARITY,
        Category.LABEL_DISPARITY,
    }


# --- intersections -----------------------------------------------------------

def _independent_frame(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """race, sex and the label are mutually independent, by construction."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "race": rng.choice(["a", "b", "c"], n),
            "sex": rng.choice(["F", "M"], n),
            "y": rng.integers(0, 2, n),
        }
    )
    df.loc[rng.choice(n, 6, replace=False), "race"] = np.nan
    return df


@pytest.fixture(scope="module")
def independent_result():
    return audit(
        _independent_frame(), ["race", "sex"], "y", sesoi=0.1, seed=3, n_resamples=200
    )


def test_no_attribute_is_tested_against_an_intersection_built_from_it(independent_result):
    """race vs race_AND_sex is a column against a function of itself: V = 1 on
    any data. Neither direction of that comparison may appear."""
    for f in independent_result.findings:
        if f.target_feature is None:
            continue
        assert "_AND_" not in f.target_feature
        if "_AND_" in f.sensitive_attribute:
            assert f.target_feature not in f.sensitive_attribute.split("_AND_")


def test_missing_parent_values_are_not_an_intersectional_group(independent_result):
    rep = next(
        f for f in independent_result.findings
        if f.id == "REPRESENTATION_RACE_AND_SEX"
    )
    assert not any("nan" in level for level in rep.n_per_group)
    assert rep.total_n == 2000 - 6


def test_independent_data_raises_no_high_findings(independent_result):
    assert not [
        f for f in independent_result.findings
        if f.severity in (Severity.HIGH, Severity.CRITICAL)
    ]


def test_finding_order_is_stable_across_intersections():
    df = _independent_frame(n=400)
    df["age"] = np.random.default_rng(1).choice(["young", "old"], len(df))
    kwargs = dict(sesoi=0.1, seed=3, n_resamples=50)
    a = audit(df, ["race", "sex", "age"], "y", **kwargs)
    attrs = list(dict.fromkeys(f.sensitive_attribute for f in a.findings))
    assert attrs == ["race", "sex", "age", "race_AND_sex", "race_AND_age", "sex_AND_age"]


def test_summary_reports_risk_and_coverage(result):
    assert "risk" in result.summary
    assert "coverage" in result.summary


def test_summary_has_no_aggregate_bias_score(result):
    assert "overall_score" not in result.summary
    assert "bias_score" not in result.summary


# --- FDR family boundary -----------------------------------------------------

def test_correction_families_are_per_category_and_attribute(result):
    """Resolves the open question in docs/10 Sec 5 / handoff Sec 7.1.

    Pooling every p-value in the audit into one family would let an unrelated
    representation test change the verdict on a missingness test. One family
    per (category, attribute) keeps the correction interpretable.
    """
    assert result.correction_families
    for (category, attribute), size in result.correction_families.items():
        assert isinstance(category, Category)
        assert isinstance(attribute, str)
        assert size >= 1
    assert sum(result.correction_families.values()) == len(result.findings)


def test_a_single_test_family_is_left_uncorrected():
    """BH over one p-value returns it unchanged; anything else is a bug."""
    df = pd.DataFrame({"g": ["a"] * 100 + ["b"] * 100, "y": [0, 1] * 100})
    out = audit(df, sensitive_cols=["g"], target_col="y", sesoi=0.1, seed=1, n_resamples=200)
    label = [f for f in out.findings if f.category is Category.LABEL_DISPARITY]
    assert len(label) == 1
    assert label[0].p_value_corrected == pytest.approx(label[0].p_value_raw)


# --- configuration -----------------------------------------------------------

def test_sesoi_is_required():
    df, truth = make_audit_scenario(n=200, seed=1)
    with pytest.raises(TypeError):
        audit(df, sensitive_cols=["gender"], target_col="hired")


def test_audit_is_deterministic_under_a_fixed_seed():
    df, truth = make_audit_scenario(n=300, seed=2)
    kwargs = dict(
        sensitive_cols=["gender"], target_col="hired", sesoi=0.1, seed=5, n_resamples=200
    )
    a = audit(df, **kwargs)
    b = audit(df, **kwargs)
    assert [f.effect_size_ci for f in a.findings] == [f.effect_size_ci for f in b.findings]


def test_an_unknown_sensitive_column_is_rejected():
    df, _ = make_audit_scenario(n=100, seed=1)
    with pytest.raises(ValueError, match="not in the dataframe"):
        audit(df, sensitive_cols=["nonexistent"], target_col="hired", sesoi=0.1)
