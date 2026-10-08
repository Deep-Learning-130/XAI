"""Two-level FDR: parents as before, intersections only below a rejected parent."""
import pytest

from dbias.correction.gating import correct_within_families, hierarchical_fdr
from dbias.models.enums import Category, EffectSizeMetric
from dbias.models.finding import Finding

PARENTS = {"race_AND_sex": ["race", "sex"]}


def f(attribute, p, category=Category.MISSINGNESS, feature="x") -> Finding:
    return Finding(
        id=f"{category}_{feature}_{attribute}", category=category,
        sensitive_attribute=attribute, target_feature=feature, metric_name="",
        observed_values={}, statistical_test="", p_value_raw=p,
        effect_size_metric=EffectSizeMetric.CRAMERS_V, effect_size_value=0.0,
        n_per_group={},
    )


def by_attr(findings):
    return {x.sensitive_attribute: x for x in findings}


def test_parents_are_corrected_exactly_as_in_family_mode():
    findings = [f("race", 0.001), f("sex", 0.4), f("race_AND_sex", 0.002)]
    hier, _ = hierarchical_fdr(findings, PARENTS, alpha=0.05)
    flat, _ = correct_within_families(findings, alpha=0.05)
    for attr in ("race", "sex"):
        assert by_attr(hier)[attr] == by_attr(flat)[attr]


def test_a_child_below_a_rejected_parent_is_tested():
    out, _ = hierarchical_fdr([f("race", 0.001), f("sex", 0.4), f("race_AND_sex", 0.002)], PARENTS, 0.05)
    child = by_attr(out)["race_AND_sex"]
    assert child.is_significant is True
    assert child.gated_by_parent is False
    assert child.p_value_corrected == pytest.approx(0.002)


def test_a_child_with_no_rejected_parent_is_not_tested():
    """The XOR case: no marginal disparity, so the intersection is never
    tested -- however small its own p-value."""
    out, families = hierarchical_fdr([f("race", 0.4), f("sex", 0.5), f("race_AND_sex", 1e-9)], PARENTS, 0.05)
    child = by_attr(out)["race_AND_sex"]
    assert child.is_significant is False
    assert child.gated_by_parent is True
    assert child.p_value_corrected == 1.0
    assert (Category.MISSINGNESS, "race_AND_sex") not in families


def test_admission_is_per_category_and_feature():
    """race rejected for feature x does not admit race_AND_sex for feature y."""
    findings = [f("race", 0.001, feature="x"), f("race", 0.6, feature="y"),
                f("sex", 0.6, feature="x"), f("sex", 0.6, feature="y"),
                f("race_AND_sex", 0.001, feature="x"), f("race_AND_sex", 0.001, feature="y")]
    out, families = hierarchical_fdr(findings, PARENTS, 0.05)
    kids = {x.target_feature: x for x in out if x.sensitive_attribute == "race_AND_sex"}
    assert kids["x"].gated_by_parent is False and kids["x"].is_significant is True
    assert kids["y"].gated_by_parent is True and kids["y"].is_significant is False
    assert families[(Category.MISSINGNESS, "race_AND_sex")] == 1


def test_bh_among_children_counts_only_admitted_children():
    """One intersection family across three features. race is rejected for x
    and y only, so z is gated. BH over the m = 2 admitted children keeps both
    (0.045 * 2/2 = 0.045 < 0.05). Counting gated z (p = 0.9) as m = 3 would
    drop both: 0.045 * 3/2 = 0.0675."""
    findings = [f("race", 0.001, feature="x"), f("race", 0.001, feature="y"), f("race", 0.9, feature="z"),
                f("sex", 0.9, feature="x"), f("sex", 0.9, feature="y"), f("sex", 0.9, feature="z"),
                f("race_AND_sex", 0.03, feature="x"), f("race_AND_sex", 0.045, feature="y"),
                f("race_AND_sex", 0.9, feature="z")]
    out, families = hierarchical_fdr(findings, PARENTS, 0.05)
    kids = {x.target_feature: x for x in out if x.sensitive_attribute == "race_AND_sex"}
    assert kids["x"].is_significant and kids["y"].is_significant
    assert kids["z"].gated_by_parent
    assert families[(Category.MISSINGNESS, "race_AND_sex")] == 2


def test_without_intersections_hierarchical_equals_family():
    findings = [f("race", 0.01), f("race", 0.03, feature="y"), f("sex", 0.2)]
    assert hierarchical_fdr(findings, {}, 0.05) == correct_within_families(findings, 0.05)


def test_a_parent_name_containing_the_separator_still_admits():
    parents = {"RACE_AND_ETHNICITY_AND_sex": ["RACE_AND_ETHNICITY", "sex"]}
    out, _ = hierarchical_fdr(
        [f("RACE_AND_ETHNICITY", 0.001), f("sex", 0.9), f("RACE_AND_ETHNICITY_AND_sex", 0.001)],
        parents, 0.05,
    )
    assert by_attr(out)["RACE_AND_ETHNICITY_AND_sex"].gated_by_parent is False


def test_representation_children_are_admitted_by_parent_representation():
    out, _ = hierarchical_fdr(
        [f("race", 0.001, Category.REPRESENTATION, None), f("sex", 0.9, Category.REPRESENTATION, None),
         f("race_AND_sex", 0.001, Category.REPRESENTATION, None)],
        PARENTS, 0.05,
    )
    assert by_attr(out)["race_AND_sex"].is_significant is True


def test_a_missing_parent_finding_does_not_admit():
    """sex was constant, so it has no finding; race alone decides."""
    out, _ = hierarchical_fdr([f("race", 0.7), f("race_AND_sex", 0.001)], PARENTS, 0.05)
    assert by_attr(out)["race_AND_sex"].gated_by_parent is True
