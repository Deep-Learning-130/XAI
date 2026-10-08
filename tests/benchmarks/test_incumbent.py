"""The incumbent's verdict on one cell, against hand-computed rates."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "benchmarks"))

from incumbent import four_fifths_cell  # noqa: E402


def frame() -> pd.DataFrame:
    """Group A: 5 x, 5 y.  Group B: 2 x, 8 y.
    Level x: rates A 0.5, B 0.2 -> ratio 0.4, difference 0.3 (fails).
    Level y: rates A 0.5, B 0.8 -> ratio 0.625, difference 0.3 (fails)."""
    return pd.DataFrame({
        "g": ["A"] * 10 + ["B"] * 10,
        "f": ["x"] * 5 + ["y"] * 5 + ["x"] * 2 + ["y"] * 8,
    })


def test_rates_ratio_and_difference_match_the_hand_computation():
    cell = four_fifths_cell(frame(), "f", "g")
    x = next(level for level in cell.levels if level.level == "x")
    assert x.selection_rates == {"A": pytest.approx(0.5), "B": pytest.approx(0.2)}
    assert x.ratio == pytest.approx(0.4)
    assert x.difference == pytest.approx(0.3)
    assert x.passes_four_fifths is False
    assert cell.reports_clean is False
    assert cell.max_difference == pytest.approx(0.3)


def test_equal_rates_are_reported_clean():
    df = pd.DataFrame({"g": ["A"] * 10 + ["B"] * 10, "f": (["x"] * 3 + ["y"] * 7) * 2})
    cell = four_fifths_cell(df, "f", "g")
    assert cell.reports_clean is True
    assert all(level.ratio == pytest.approx(1.0) for level in cell.levels)


def test_a_group_with_no_rows_at_a_level_has_rate_zero():
    df = pd.DataFrame({"g": ["A", "A", "B", "B"], "f": ["x", "y", "y", "y"]})
    x = next(level for level in four_fifths_cell(df, "f", "g").levels if level.level == "x")
    assert x.selection_rates == {"A": pytest.approx(0.5), "B": 0.0}
    assert x.ratio == 0.0


def test_rows_missing_either_column_are_dropped():
    df = frame()
    df.loc[0, "f"] = np.nan
    df.loc[19, "g"] = np.nan
    assert four_fifths_cell(df, "f", "g").n == 18


def test_numeric_group_labels_are_keyed_as_strings():
    df = pd.DataFrame({"g": [0, 0, 1, 1], "f": ["x", "y", "x", "x"]})
    x = next(level for level in four_fifths_cell(df, "f", "g").levels if level.level == "x")
    assert x.selection_rates == {"0": pytest.approx(0.5), "1": pytest.approx(1.0)}


def test_matches_fairlearn():
    fairlearn = pytest.importorskip("fairlearn.metrics")
    df = frame()
    for level in four_fifths_cell(df, "f", "g").levels:
        indicator = (df["f"] == level.level).astype(int)
        expected = fairlearn.demographic_parity_ratio(
            y_true=indicator, y_pred=indicator, sensitive_features=df["g"]
        )
        assert level.ratio == pytest.approx(expected)
