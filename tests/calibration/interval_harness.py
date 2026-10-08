"""Empirical calibration of the contingency-table interval beyond 2x2.

A table with chosen margins and Cohen's w is built as a rank-one departure
from independence: p_ij = r_i c_j (1 + w u_i v_j), where u and v are
contrasts with zero mean and unit variance under the row and column shares.
Then the margins are exactly r and c, and

    w^2 = sum_ij (p_ij - r_i c_j)^2 / (r_i c_j) = w^2 (sum r u^2)(sum c v^2) = w^2,

so the population Cramer's V is w / sqrt(min(R, C) - 1). Where some p_ij
would go negative the cell is unreachable and is skipped, never clipped.
"""
import numpy as np

from dbias.stats.effect_sizes import cramers_v, cramers_v_corrected, v_from_w
from dbias.stats.intervals import bootstrap_ci_cramers_v


def _contrast(shares: np.ndarray) -> np.ndarray:
    u = np.arange(len(shares), dtype=float)
    u -= (shares * u).sum()
    return u / np.sqrt((shares * u**2).sum())


def joint_probs(row_shares, col_shares, true_w: float) -> np.ndarray | None:
    r = np.asarray(row_shares, dtype=float)
    c = np.asarray(col_shares, dtype=float)
    p = np.outer(r, c) * (1.0 + true_w * np.outer(_contrast(r), _contrast(c)))
    return None if np.any(p < 0) else p


def run_interval_cell(
    row_shares,
    col_shares,
    n: int,
    true_w: float,
    *,
    sesoi: float = 0.1,
    alpha: float = 0.05,
    replicates: int = 1000,
    n_resamples: int = 200,
    seed: int = 0,
) -> dict:
    p = joint_probs(row_shares, col_shares, true_w)
    if p is None:
        raise ValueError("unreachable cell")
    shape = p.shape
    df_min = min(shape) - 1
    true_v = v_from_w(true_w, df_min)
    sesoi_v = v_from_w(sesoi, df_min)

    rng = np.random.default_rng(seed)
    covered = equivalent = spurious = raw_spurious = usable = 0
    corrected_sum = raw_sum = 0.0
    for _ in range(replicates):
        table = rng.multinomial(n, p.ravel()).reshape(shape)
        if min((table.sum(axis=1) > 0).sum(), (table.sum(axis=0) > 0).sum()) < 2:
            continue  # a degenerate draw is not a test
        usable += 1
        lo, hi = bootstrap_ci_cramers_v(
            table, n_resamples=n_resamples, confidence=1.0 - alpha,
            seed=int(rng.integers(1 << 31)),
        )
        covered += lo <= true_v <= hi
        equivalent += hi < sesoi_v
        spurious += lo > sesoi_v
        corrected_sum += cramers_v_corrected(table)
        raw = cramers_v(table)
        raw_sum += raw
        # The old interval: plain percentile bootstrap of the plug-in V.
        draws = rng.multinomial(n, (table / n).ravel(), size=n_resamples)
        raw_lo = np.quantile([cramers_v(d.reshape(shape)) for d in draws], alpha / 2)
        raw_spurious += raw_lo > sesoi_v

    return {
        "shape": f"{shape[0]}x{shape[1]}",
        "n": n,
        "true_w": true_w,
        "true_v": true_v,
        "replicates": usable,
        "coverage": covered / usable,
        "equivalent_rate": equivalent / usable,
        "spurious_disparity_rate": spurious / usable,
        "raw_spurious_disparity_rate": raw_spurious / usable,
        "mean_corrected_v": corrected_sum / usable,
        "mean_raw_v": raw_sum / usable,
    }
