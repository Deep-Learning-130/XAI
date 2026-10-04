"""Severity assignment, including the row docs/05 does not have.

docs/05 grades severity on significance and magnitude alone. That table has no
way to express "we found nothing and that means nothing", which is the whole
reason this project exists. The BLIND_SPOT row is the addition.
"""
import pytest

from dbias.models.enums import (
    Category,
    Detectability,
    EffectSizeMetric,
    EquivalenceVerdict,
    Severity,
)
from dbias.models.finding import Finding
from dbias.rules.severity import assign_severity


def finding(**overrides) -> Finding:
    kwargs = dict(
        id="TEST",
        category=Category.MISSINGNESS,
        sensitive_attribute="gender",
        target_feature="income",
        metric_name="Missingness rate disparity",
        observed_values={},
        statistical_test="Chi-square test of independence",
        p_value_raw=0.5,
        effect_size_metric=EffectSizeMetric.CRAMERS_V,
        effect_size_value=0.02,
        n_per_group={"m": 50, "f": 50},
        p_value_corrected=0.5,
        is_significant=False,
        detectability=Detectability.ADEQUATE,
        equivalence_verdict=EquivalenceVerdict.EQUIVALENT,
        sesoi=0.1,
        effect_size_ci=(0.0, 0.03),
        minimum_detectable_effect=0.05,
    )
    kwargs.update(overrides)
    return assign_severity(Finding(**kwargs))


# --- the row docs/05 is missing ---------------------------------------------

def test_null_result_in_an_underpowered_test_is_a_blind_spot():
    f = finding(
        is_significant=False,
        detectability=Detectability.UNDERPOWERED,
        equivalence_verdict=EquivalenceVerdict.INCONCLUSIVE,
    )
    assert f.severity is Severity.BLIND_SPOT


def test_null_result_in_an_adequate_test_is_informational():
    """An earned all-clear. Same p-value as the blind spot above, opposite meaning."""
    assert finding(is_significant=False).severity is Severity.INFORMATIONAL


def test_blind_spot_and_all_clear_differ_only_in_detectability():
    blind = finding(
        is_significant=False,
        detectability=Detectability.UNDERPOWERED,
        equivalence_verdict=EquivalenceVerdict.INCONCLUSIVE,
    )
    clear = finding(is_significant=False, detectability=Detectability.ADEQUATE)
    assert blind.p_value_corrected == clear.p_value_corrected
    assert blind.severity is not clear.severity


# --- the docs/05 ladder, preserved ------------------------------------------

@pytest.mark.parametrize(
    "effect,expected",
    [
        (0.02, Severity.INFORMATIONAL),  # trivial, below the SESOI
        (0.15, Severity.LOW),            # small
        (0.35, Severity.MEDIUM),         # medium
        (0.60, Severity.HIGH),           # large
    ],
)
def test_significant_findings_are_graded_by_magnitude(effect, expected):
    f = finding(
        is_significant=True,
        p_value_corrected=0.001,
        effect_size_value=effect,
        equivalence_verdict=EquivalenceVerdict.DISPARITY,
    )
    assert f.severity is expected


def test_a_large_effect_on_the_label_is_critical():
    f = finding(
        category=Category.LABEL_DISPARITY,
        is_significant=True,
        p_value_corrected=0.001,
        effect_size_value=0.60,
        equivalence_verdict=EquivalenceVerdict.DISPARITY,
    )
    assert f.severity is Severity.CRITICAL


def test_a_large_effect_on_a_non_label_feature_is_not_critical():
    f = finding(
        category=Category.MISSINGNESS,
        is_significant=True,
        p_value_corrected=0.001,
        effect_size_value=0.60,
        equivalence_verdict=EquivalenceVerdict.DISPARITY,
    )
    assert f.severity is Severity.HIGH


# --- scenario C --------------------------------------------------------------

def test_statistically_significant_but_trivial_effect_is_informational():
    """plan.md Sec 6: n = 10^6 with a 0.01 effect must not raise an alarm.
    Significance at huge n is a statement about n, not about the effect."""
    f = finding(
        is_significant=True,
        p_value_corrected=1e-12,
        effect_size_value=0.01,
        n_per_group={"m": 500_000, "f": 500_000},
        equivalence_verdict=EquivalenceVerdict.EQUIVALENT,
    )
    assert f.severity is Severity.INFORMATIONAL


# --- absences are not blind spots -------------------------------------------

def test_an_empty_cell_is_informational_not_a_blind_spot():
    """A cell with no rows is an absence, not something we failed to see."""
    f = finding(detectability=Detectability.EMPTY, equivalence_verdict=None)
    assert f.severity is Severity.INFORMATIONAL


# --- the invariant -----------------------------------------------------------

def test_unannotated_finding_raises_rather_than_being_graded():
    """docs/10 invariant 3: detectability is not optional at runtime. A
    finding reaching rules/ with UNKNOWN detectability is a bug."""
    with pytest.raises(ValueError, match="detectability"):
        finding(detectability=Detectability.UNKNOWN)


# --- which of the two signals decides ----------------------------------------
# plan.md Sec 3.2: TOST is the inference layer, MDE is the communication layer.
# When the a-priori power calculation and the realised interval disagree, the
# interval wins -- it is computed from the data that actually arrived, while
# the MDE is a design-stage approximation that ignores marginal structure.

def test_equivalence_beats_a_pessimistic_power_calculation():
    """The interval rules out the SESOI even though the MDE says the test was
    underpowered. The audit did in fact see clearly, so this is an earned
    all-clear, not a blind spot."""
    f = finding(
        is_significant=False,
        detectability=Detectability.UNDERPOWERED,
        equivalence_verdict=EquivalenceVerdict.EQUIVALENT,
        effect_size_ci=(0.009, 0.088),
        minimum_detectable_effect=0.103,
    )
    assert f.severity is Severity.INFORMATIONAL


def test_inconclusive_interval_is_a_blind_spot_even_when_power_looked_adequate():
    """The mirror case: the MDE promised adequacy, the interval could not
    deliver it. The honest answer is that we could not see."""
    f = finding(
        is_significant=False,
        detectability=Detectability.ADEQUATE,
        equivalence_verdict=EquivalenceVerdict.INCONCLUSIVE,
        effect_size_ci=(0.02, 0.14),
        minimum_detectable_effect=0.09,
    )
    assert f.severity is Severity.BLIND_SPOT


def test_detectability_alone_decides_when_no_interval_exists():
    """Fallback only. Every supported metric produces an interval."""
    f = finding(
        is_significant=False,
        detectability=Detectability.UNDERPOWERED,
        equivalence_verdict=None,
    )
    assert f.severity is Severity.BLIND_SPOT


def test_a_nonsignificant_interval_above_the_sesoi_is_never_an_all_clear():
    """Regression: a non-significant finding with a DISPARITY verdict fell
    through to detectability and, if power looked adequate, was scored
    INFORMATIONAL -- the earned all-clear -- although its interval ruled
    nothing out. Seen on sparse feature tables in the N=500 Adult run."""
    f = finding(
        is_significant=False,
        detectability=Detectability.ADEQUATE,
        equivalence_verdict=EquivalenceVerdict.DISPARITY,
        effect_size_value=0.157,
        effect_size_ci=(0.170, 0.287),
    )
    assert f.severity is Severity.BLIND_SPOT


def test_an_adequately_powered_null_without_an_interval_is_informational():
    f = finding(
        is_significant=False,
        detectability=Detectability.ADEQUATE,
        equivalence_verdict=None,
        effect_size_ci=None,
    )
    assert f.severity is Severity.INFORMATIONAL
