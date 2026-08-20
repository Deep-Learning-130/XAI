"""The verdict: an equivalence decision against a declared SESOI.

plan.md Sec 3.2 -- the blind-spot verdict is a two-one-sided-tests style
decision on the confidence interval, not an MDE comparison. MDE survives as
the human-facing communication device only.

The two tables below carry *identical proportions* and differ only in n. They
must receive different verdicts. That pair is the entire argument of the
project, so it is asserted directly.
"""
import numpy as np
import pytest

from dbias.models.enums import Category, Detectability, EffectSizeMetric, EquivalenceVerdict
from dbias.models.finding import Finding
from dbias.detectability.classify import annotate_detectability

NULL_LARGE = np.array([[2500, 2500], [2500, 2500]])   # n = 10,000
NULL_SMALL = np.array([[25, 25], [25, 25]])           # n = 100, same proportions
STRONG = np.array([[400, 100], [100, 400]])           # n = 1,000, V = 0.6
TRIVIAL_HUGE = np.array([[252_500, 247_500], [247_500, 252_500]])  # n = 10^6, V = 0.005


def bare_finding(table: np.ndarray, effect: float, p: float = 0.5) -> Finding:
    return Finding(
        id="TEST",
        category=Category.LABEL_DISPARITY,
        sensitive_attribute="attr",
        target_feature="target",
        metric_name="Chi-square independence",
        observed_values={},
        statistical_test="Chi-square test of independence",
        p_value_raw=p,
        effect_size_metric=EffectSizeMetric.CRAMERS_V,
        effect_size_value=effect,
        n_per_group={"a": int(table.sum(axis=1)[0]), "b": int(table.sum(axis=1)[1])},
    )


def annotate(table, effect=0.0, sesoi=0.1, p=0.5, **kw):
    return annotate_detectability(
        bare_finding(table, effect, p), table, sesoi=sesoi, seed=42, n_resamples=800, **kw
    )


# --- the headline pair -------------------------------------------------------

def test_well_powered_null_is_an_earned_all_clear():
    f = annotate(NULL_LARGE)
    assert f.equivalence_verdict is EquivalenceVerdict.EQUIVALENT
    assert f.detectability is Detectability.ADEQUATE


def test_underpowered_null_is_a_blind_spot_not_a_clean_result():
    f = annotate(NULL_SMALL)
    assert f.equivalence_verdict is EquivalenceVerdict.INCONCLUSIVE
    assert f.detectability is Detectability.UNDERPOWERED


def test_the_pair_differs_only_in_sample_size():
    """Same proportions, opposite verdicts. This is the contribution."""
    assert np.allclose(NULL_LARGE / NULL_LARGE.sum(), NULL_SMALL / NULL_SMALL.sum())
    assert annotate(NULL_LARGE).detectability is not annotate(NULL_SMALL).detectability


# --- the other quadrants -----------------------------------------------------

def test_large_effect_at_adequate_power_reports_a_disparity():
    f = annotate(STRONG, effect=0.6, p=1e-40)
    assert f.equivalence_verdict is EquivalenceVerdict.DISPARITY
    assert f.detectability is Detectability.ADEQUATE


def test_trivial_effect_at_huge_n_is_ruled_equivalent():
    """Scenario C (plan.md Sec 6): n = 10^6 with a 0.005 effect is
    statistically significant and practically nothing. The equivalence verdict
    must say so rather than raising an alarm."""
    f = annotate(TRIVIAL_HUGE, effect=0.005, p=1e-9)
    assert f.equivalence_verdict is EquivalenceVerdict.EQUIVALENT
    assert f.detectability is Detectability.ADEQUATE


# --- fields populated --------------------------------------------------------

def test_annotation_populates_every_detectability_field():
    f = annotate(NULL_SMALL)
    assert f.sesoi == pytest.approx(0.1)
    assert f.effect_size_ci is not None and f.effect_size_ci[0] <= f.effect_size_ci[1]
    assert 0.0 <= f.power_to_detect_sesoi <= 1.0
    assert f.minimum_detectable_effect > 0


def test_mde_is_reported_in_the_same_units_as_the_effect_size():
    """The finding's effect size is Cramer's V, so its MDE must be too."""
    f = annotate(NULL_SMALL)
    assert f.minimum_detectable_effect == pytest.approx(0.28, abs=0.02)


def test_underpowered_cell_has_an_mde_above_the_sesoi():
    f = annotate(NULL_SMALL, sesoi=0.1)
    assert f.minimum_detectable_effect > f.sesoi


def test_adequate_cell_has_an_mde_at_or_below_the_sesoi():
    f = annotate(NULL_LARGE, sesoi=0.1)
    assert f.minimum_detectable_effect <= f.sesoi


# --- the defect-1 guard ------------------------------------------------------

def test_power_does_not_depend_on_the_observed_effect():
    """plan.md Sec 3.1. Two findings over the same table shape and n, with
    wildly different observed effects, must report identical power -- because
    power is computed against the SESOI, never against what was seen."""
    quiet = annotate(NULL_LARGE, effect=0.001)
    loud = annotate_detectability(
        bare_finding(NULL_LARGE, 0.85), NULL_LARGE, sesoi=0.1, seed=42, n_resamples=800
    )
    assert quiet.power_to_detect_sesoi == pytest.approx(loud.power_to_detect_sesoi)


def test_annotated_finding_never_gains_an_observed_power_attribute():
    assert not hasattr(annotate(NULL_SMALL), "statistical_power")


# --- required configuration --------------------------------------------------

def test_sesoi_is_required():
    with pytest.raises(TypeError):
        annotate_detectability(bare_finding(NULL_SMALL, 0.0), NULL_SMALL)


def test_non_positive_sesoi_is_rejected():
    with pytest.raises(ValueError):
        annotate(NULL_SMALL, sesoi=0.0)


def test_severity_is_untouched_by_the_detectability_pass():
    """docs/10 invariant 2: severity is assigned only in rules/."""
    from dbias.models.enums import Severity

    assert annotate(NULL_SMALL).severity is Severity.UNDETERMINED


# --- the deferred correction is surfaced, not hidden -------------------------

def test_skewed_margins_flag_the_mde_as_approximate():
    """plan.md Sec 3.4 is deferred, so affected cells must be labelled."""
    skewed = np.array([[950, 50], [45, 5]])
    assert annotate(skewed).mde_is_approximate is True


def test_balanced_margins_are_not_flagged():
    assert annotate(NULL_LARGE).mde_is_approximate is False


# --- goodness-of-fit samples take the same path ------------------------------
# Representation asks a one-dimensional question, but the verdict logic is
# identical: an interval that clears the SESOI is an earned all-clear, one that
# straddles it is a blind spot.

from dbias.stats.chi_square import GofSample  # noqa: E402


def gof(counts, effect=0.0, sesoi=0.1, expected_probs=None):
    sample = GofSample(counts=tuple(counts), expected_probs=expected_probs)
    finding = Finding(
        id="REPRESENTATION",
        category=Category.REPRESENTATION,
        sensitive_attribute="attr",
        target_feature=None,
        metric_name="Group share vs reference",
        observed_values={},
        statistical_test="Chi-square goodness of fit",
        p_value_raw=0.5,
        effect_size_metric=EffectSizeMetric.COHENS_W,
        effect_size_value=effect,
        n_per_group={f"g{i}": int(c) for i, c in enumerate(counts)},
    )
    return annotate_detectability(
        finding, sample, sesoi=sesoi, seed=42, n_resamples=800
    )


def test_balanced_large_sample_earns_its_all_clear():
    f = gof([5000, 5000])
    assert f.equivalence_verdict is EquivalenceVerdict.EQUIVALENT
    assert f.detectability is Detectability.ADEQUATE


def test_balanced_small_sample_is_a_blind_spot():
    """Same 50/50 shares, n = 100 instead of 10,000."""
    f = gof([50, 50])
    assert f.equivalence_verdict is EquivalenceVerdict.INCONCLUSIVE
    assert f.detectability is Detectability.UNDERPOWERED


def test_strong_imbalance_is_reported_as_a_disparity():
    f = gof([3000, 1000], effect=0.5)
    assert f.equivalence_verdict is EquivalenceVerdict.DISPARITY


def test_gof_annotation_populates_the_same_fields():
    f = gof([50, 50])
    assert f.sesoi == pytest.approx(0.1)
    assert f.effect_size_ci is not None
    assert f.minimum_detectable_effect > 0
    assert 0.0 <= f.power_to_detect_sesoi <= 1.0
