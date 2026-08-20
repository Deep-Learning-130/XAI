"""OVERRIDE 1 and 2: one ladder per metric, and Cramer's V is df-adjusted.

The docs/05 ladder (0.10 / 0.15 / 0.25 applied to every metric alike) is
deleted -- V, w, and eta-squared are not on a common scale.
"""
import pytest

from dbias.models.enums import EffectSizeMetric, Magnitude
from dbias.stats.thresholds import magnitude, thresholds_for


def test_cohens_w_uses_the_canonical_ladder():
    assert thresholds_for(EffectSizeMetric.COHENS_W) == (0.1, 0.3, 0.5)


def test_cohens_w_magnitude_bands():
    assert magnitude(EffectSizeMetric.COHENS_W, 0.05) is Magnitude.TRIVIAL
    assert magnitude(EffectSizeMetric.COHENS_W, 0.10) is Magnitude.SMALL
    assert magnitude(EffectSizeMetric.COHENS_W, 0.29) is Magnitude.SMALL
    assert magnitude(EffectSizeMetric.COHENS_W, 0.30) is Magnitude.MEDIUM
    assert magnitude(EffectSizeMetric.COHENS_W, 0.50) is Magnitude.LARGE


def test_cramers_v_thresholds_equal_w_thresholds_for_2x2():
    assert thresholds_for(EffectSizeMetric.CRAMERS_V, df_min=1) == (0.1, 0.3, 0.5)


def test_cramers_v_thresholds_shrink_with_df():
    """w = V * sqrt(df_min), so the V ladder is the w ladder over sqrt(df_min)."""
    assert thresholds_for(EffectSizeMetric.CRAMERS_V, df_min=4) == pytest.approx(
        (0.05, 0.15, 0.25)
    )


def test_cramers_v_magnitude_uses_the_adjusted_ladder():
    """V = 0.2 is SMALL in a 2x2 but MEDIUM in a table with df_min = 4."""
    assert magnitude(EffectSizeMetric.CRAMERS_V, 0.2, df_min=1) is Magnitude.SMALL
    assert magnitude(EffectSizeMetric.CRAMERS_V, 0.2, df_min=4) is Magnitude.MEDIUM


def test_cramers_v_requires_df_min():
    with pytest.raises(ValueError):
        thresholds_for(EffectSizeMetric.CRAMERS_V)


def test_unregistered_metric_raises_rather_than_guessing():
    with pytest.raises(KeyError):
        thresholds_for(EffectSizeMetric.ODDS_RATIO)
