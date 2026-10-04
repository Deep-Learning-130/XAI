"""Intersection columns and the gating that descends into them."""
import numpy as np
import pandas as pd

from dbias.analyzers.base import Hypothesis
from dbias.correction.gating import power_guided_gating
from dbias.correction.hierarchy import build_intersections, parents_of
from dbias.models.enums import Category, Detectability, EffectSizeMetric
from dbias.models.finding import Finding


def test_a_missing_parent_value_is_missing_in_the_intersection():
    df = pd.DataFrame({"race": ["a", None, "b"], "sex": ["F", "M", None]})
    out, _ = build_intersections(df, ["race", "sex"])
    assert out["race_AND_sex"].tolist()[0] == "a + F"
    assert out["race_AND_sex"].isna().tolist() == [False, True, True]


def test_parents_come_from_the_tree_in_build_order():
    df = pd.DataFrame({c: ["x", "y"] for c in ["a", "b", "c"]})
    _, tree = build_intersections(df, ["a", "b", "c"])
    assert parents_of(tree) == {
        "a_AND_b": ["a", "b"],
        "a_AND_c": ["a", "c"],
        "b_AND_c": ["b", "c"],
    }


def _hyp(attribute: str, detectability=Detectability.UNKNOWN) -> Hypothesis:
    finding = Finding(
        id="T",
        category=Category.MISSINGNESS,
        sensitive_attribute=attribute,
        target_feature="income",
        metric_name="",
        observed_values={},
        statistical_test="",
        p_value_raw=0.5,
        effect_size_metric=EffectSizeMetric.CRAMERS_V,
        effect_size_value=0.0,
        n_per_group={},
        detectability=detectability,
    )
    return Hypothesis(finding=finding, sample=np.zeros((2, 2)))


def test_gating_finds_parents_whose_names_contain_the_separator():
    """Regression: parents were recovered by splitting on '_AND_', which
    broke for a column like RACE_AND_ETHNICITY."""
    df = pd.DataFrame({"RACE_AND_ETHNICITY": ["x", "y"], "sex": ["F", "M"]})
    _, tree = build_intersections(df, ["RACE_AND_ETHNICITY", "sex"])
    parents = parents_of(tree)
    child = "RACE_AND_ETHNICITY_AND_sex"
    parent = _hyp("RACE_AND_ETHNICITY", Detectability.UNDERPOWERED).finding
    parent_findings = {"MISSINGNESS_income_RACE_AND_ETHNICITY": parent}

    [(_, skip)] = power_guided_gating([_hyp(child)], parent_findings, parents)
    assert skip is True


def test_gating_leaves_children_of_adequate_parents_alone():
    parents = {"a_AND_b": ["a", "b"]}
    parent_findings = {
        "MISSINGNESS_income_a": _hyp("a", Detectability.ADEQUATE).finding,
        "MISSINGNESS_income_b": _hyp("b", Detectability.ADEQUATE).finding,
    }
    [(_, skip)] = power_guided_gating([_hyp("a_AND_b")], parent_findings, parents)
    assert skip is False
