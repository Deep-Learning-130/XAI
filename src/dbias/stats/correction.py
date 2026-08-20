"""Flat Benjamini-Hochberg over a list of p-values.

Deliberately does not know: which hypotheses belong to the same family. The
family boundary is a policy decision and lives in a higher layer (docs/10
Sec 5, open question 1). This function corrects whatever it is handed.
"""
from collections.abc import Sequence

import numpy as np


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """Return BH step-up adjusted p-values, in the input's order.

    Equivalent to `scipy.stats.false_discovery_control(p, method="bh")`, which
    the unit tests use as an independent oracle.
    """
    p = np.asarray(list(p_values), dtype=float)
    if p.size == 0:
        return []
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError("p-values must lie in [0, 1]")

    m = p.size
    order = np.argsort(p)
    ranks = np.arange(1, m + 1)
    scaled = p[order] * m / ranks
    # Step up: enforce monotonicity from the largest p-value downwards.
    adjusted_sorted = np.minimum.accumulate(scaled[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()


def is_significant(adjusted_p: float, alpha: float = 0.05) -> bool:
    return adjusted_p < alpha
