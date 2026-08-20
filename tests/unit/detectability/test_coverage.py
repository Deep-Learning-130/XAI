"""The attribute x feature grid of minimum detectable effects.

plan.md Sec 2 calls this the single most valuable artifact in the design: the
thing a practitioner screenshots, and the thing that makes the idea legible in
one image. It is built here and drawn in report/coverage_plot.py.
"""
import pytest

from dbias.detectability.coverage import REPRESENTATION_COLUMN, coverage_map
from dbias.models.enums import (
    Category,
    Detectability,
    EffectSizeMetric,
    EquivalenceVerdict,
    Severity,
)
from dbias.models.finding import Finding


def finding(attribute, feature, category, mde, detectability, severity) -> Finding:
    return Finding(
        id=f"{attribute}_{feature}",
        category=category,
        sensitive_attribute=attribute,
        target_feature=feature,
        metric_name="metric",
        observed_values={},
        statistical_test="Chi-square test of independence",
        p_value_raw=0.5,
        effect_size_metric=EffectSizeMetric.CRAMERS_V,
        effect_size_value=0.05,
        n_per_group={"a": 450, "b": 450},
        minimum_detectable_effect=mde,
        detectability=detectability,
        severity=severity,
        sesoi=0.1,
        equivalence_verdict=EquivalenceVerdict.INCONCLUSIVE,
    )


FINDINGS = [
    finding("gender", None, Category.REPRESENTATION, 0.093,
            Detectability.ADEQUATE, Severity.INFORMATIONAL),
    finding("gender", "income", Category.MISSINGNESS, 0.093,
            Detectability.ADEQUATE, Severity.INFORMATIONAL),
    finding("ethnicity", "income", Category.MISSINGNESS, 0.115,
            Detectability.UNDERPOWERED, Severity.BLIND_SPOT),
    finding("ethnicity", "hired", Category.LABEL_DISPARITY, 0.115,
            Detectability.UNDERPOWERED, Severity.BLIND_SPOT),
]


def test_map_lists_every_attribute():
    assert coverage_map(FINDINGS).attributes == ["ethnicity", "gender"]


def test_representation_gets_its_own_column():
    """Representation has no target feature, but it still belongs on the grid."""
    grid = coverage_map(FINDINGS)
    assert REPRESENTATION_COLUMN in grid.features
    assert grid.cell("gender", REPRESENTATION_COLUMN) is not None


def test_cells_carry_the_mde_and_the_verdict():
    cell = coverage_map(FINDINGS).cell("ethnicity", "income")
    assert cell.mde == pytest.approx(0.115)
    assert cell.detectability is Detectability.UNDERPOWERED
    assert cell.severity is Severity.BLIND_SPOT


def test_untested_combinations_are_absent_rather_than_clean():
    """gender x hired was never tested. Rendering it as anything other than
    'not tested' would be the false reassurance this tool exists to prevent."""
    assert coverage_map(FINDINGS).cell("gender", "hired") is None


def test_matrix_shape_matches_attributes_by_features():
    grid = coverage_map(FINDINGS)
    matrix = grid.matrix()
    assert matrix.shape == (len(grid.attributes), len(grid.features))


def test_untested_cells_are_nan_in_the_matrix():
    import numpy as np

    grid = coverage_map(FINDINGS)
    row = grid.attributes.index("gender")
    col = grid.features.index("hired")
    assert np.isnan(grid.matrix()[row, col])


def test_blind_spot_cells_are_enumerable():
    blind = coverage_map(FINDINGS).blind_spots()
    assert {(c.attribute, c.feature) for c in blind} == {
        ("ethnicity", "income"),
        ("ethnicity", "hired"),
    }


def test_worst_mde_per_attribute_summarises_the_row():
    grid = coverage_map(FINDINGS)
    assert grid.worst_mde("ethnicity") == pytest.approx(0.115)
    assert grid.worst_mde("gender") == pytest.approx(0.093)


def test_features_are_ordered_with_representation_first():
    """It is the column that describes the attribute itself, so it reads
    naturally at the left edge of the figure."""
    assert coverage_map(FINDINGS).features[0] == REPRESENTATION_COLUMN


def test_empty_findings_give_an_empty_map():
    grid = coverage_map([])
    assert grid.attributes == []
    assert grid.features == []
