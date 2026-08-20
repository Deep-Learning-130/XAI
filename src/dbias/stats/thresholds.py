"""Per-metric magnitude ladders.

Deliberately does not know: whether a large effect is bad, or which direction
counts as harm. It maps a number onto a word.

OVERRIDE 1 (implementation plan): docs/05 applies a single 0.10/0.15/0.25
ladder to every effect size. Cramer's V, Cohen's w, rank-biserial and
eta-squared are not on a common scale, so that ladder is deleted in favour of
this registry.

OVERRIDE 2: Cramer's V thresholds are the Cohen's w thresholds divided by
sqrt(min(r-1, k-1)), because w = V * sqrt(min(r-1, k-1)). Neither the docs/02
(0.1/0.3) nor the docs/05 (0.1/0.15/0.25) published ladder is hardcoded.
"""
import math

from dbias.models.enums import EffectSizeMetric, Magnitude

Ladder = tuple[float, float, float]  # (small, medium, large) lower bounds

# Cohen (1988). These are conventions, not domain judgements -- which is
# exactly why the audit's central verdict uses a user-declared SESOI instead
# (plan.md Sec 3.6). This registry drives reporting language only.
_REGISTRY: dict[EffectSizeMetric, Ladder] = {
    EffectSizeMetric.COHENS_W: (0.1, 0.3, 0.5),
    EffectSizeMetric.RANK_BISERIAL: (0.1, 0.3, 0.5),
    EffectSizeMetric.ETA_SQUARED: (0.01, 0.06, 0.14),
    EffectSizeMetric.KS_STATISTIC: (0.1, 0.2, 0.3),
}

_DF_ADJUSTED = {EffectSizeMetric.CRAMERS_V: EffectSizeMetric.COHENS_W}


def thresholds_for(metric: EffectSizeMetric, df_min: int | None = None) -> Ladder:
    """Return the (small, medium, large) lower bounds for `metric`."""
    if metric in _DF_ADJUSTED:
        if df_min is None:
            raise ValueError(
                f"{metric} thresholds are df-adjusted; pass df_min=min(r-1, k-1)"
            )
        if df_min < 1:
            raise ValueError("df_min must be at least 1")
        base = _REGISTRY[_DF_ADJUSTED[metric]]
        scale = math.sqrt(df_min)
        return (base[0] / scale, base[1] / scale, base[2] / scale)
    if metric not in _REGISTRY:
        raise KeyError(
            f"no threshold ladder registered for {metric}; register one rather "
            "than borrowing another metric's conventions"
        )
    return _REGISTRY[metric]


def magnitude(
    metric: EffectSizeMetric, value: float, df_min: int | None = None
) -> Magnitude:
    small, medium, large = thresholds_for(metric, df_min=df_min)
    v = abs(value)
    if v < small:
        return Magnitude.TRIVIAL
    if v < medium:
        return Magnitude.SMALL
    if v < large:
        return Magnitude.MEDIUM
    return Magnitude.LARGE
