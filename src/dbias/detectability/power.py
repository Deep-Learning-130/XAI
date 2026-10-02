"""What this sample *could* have shown.

Deliberately does not know: what was observed. Every function here takes a
pre-specified effect size and a sample size. There is no code path by which an
observed effect can reach a power calculation, because power computed from the
observed effect is a monotone function of the p-value and carries no
information beyond it (Hoenig & Heisey 2001, "The Abuse of Power";
plan.md Sec 3.1).

Known approximation. The non-centrality is taken as n * w^2, which treats
Cohen's w as sufficient for the alternative and ignores the marginal
structure. Under skewed margins -- a 95/5 or 99/1 group split, exactly the
minority subgroups an audit cares most about -- effective power is governed by
the smaller cell rather than by the total, and this approximation errs
*optimistic*. That is the dangerous direction for a tool whose purpose is
honest null results. Callers must propagate `ChiSquareResult.is_skewed` and
`.is_sparse` so affected cells are reported as approximate. A simulation-based
fallback is the correct fix and is deliberately deferred (plan.md Sec 3.4).
"""
import math

from scipy import optimize, stats

DEFAULT_ALPHA = 0.05
DEFAULT_TARGET_POWER = 0.80

# The MDE solver searches Cohen's w over this range. w is unbounded above but
# an effect beyond 5.0 is far past anything a real table produces.
_W_SEARCH = (1e-9, 5.0)


def power_chi_square(w: float, n: int, dof: int, alpha: float = DEFAULT_ALPHA) -> float:
    """Probability of rejecting at `alpha` when the true effect is `w`.

    `w` is a pre-specified effect -- typically the SESOI. It is never the
    observed effect size.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if dof < 1:
        raise ValueError("dof must be at least 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    if w < 0:
        raise ValueError("w cannot be negative")

    critical = stats.chi2.ppf(1.0 - alpha, dof)
    noncentrality = n * w**2
    return float(stats.ncx2.sf(critical, dof, noncentrality))


def simulate_power_2x2(
    w: float,
    n: int,
    minority_share: float,
    base_rate: float,
    alpha: float = DEFAULT_ALPHA,
    n_sims: int = 5000,
    seed: int | None = None,
) -> float:
    """Empirical power for a 2x2 table, conditioning on observed margins.

    Solves the approximation error under skewed margins by directly simulating
    binomial draws.
    """
    import numpy as np

    if w < 0:
        raise ValueError("w cannot be negative")

    g2 = minority_share
    g1 = 1.0 - g2
    if w == 0.0:
        p1, p2 = base_rate, base_rate
    else:
        d = w * math.sqrt(base_rate * (1.0 - base_rate) / (g1 * g2))
        p1 = base_rate + g2 * d
        p2 = base_rate - g1 * d
        if not (0.0 <= p1 <= 1.0 and 0.0 <= p2 <= 1.0):
            # Effect size is unreachable with these margins
            return 0.0

    rng = np.random.default_rng(seed)
    n2 = max(1, int(round(n * minority_share)))
    n1 = n - n2

    events1 = rng.binomial(n1, p1, size=n_sims).astype(float)
    events2 = rng.binomial(n2, p2, size=n_sims).astype(float)

    a, b = events1, n1 - events1
    c, d = events2, n2 - events2

    # Fast 2x2 chi-square statistic: n * (ad - bc)^2 / (row1*row2*col1*col2)
    col1 = a + c
    col2 = b + d
    
    with np.errstate(divide="ignore", invalid="ignore"):
        stat = n * (a * d - b * c) ** 2 / (n1 * n2 * col1 * col2)
    stat[np.isnan(stat)] = 0.0

    critical = stats.chi2.ppf(1.0 - alpha, 1)
    return float(np.mean(stat > critical))


def mde_chi_square(
    n: int,
    dof: int,
    alpha: float = DEFAULT_ALPHA,
    target_power: float = DEFAULT_TARGET_POWER,
) -> float:
    """Smallest Cohen's w this test would detect at `target_power`.

    Scales exactly as n^(-1/2): the required non-centrality is a constant for
    fixed (target_power, dof, alpha), and the non-centrality is n * w^2.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0.0 < target_power < 1.0:
        raise ValueError("target_power must lie strictly between 0 and 1")
    if target_power <= alpha:
        raise ValueError("target_power must exceed alpha to be meaningful")

    def shortfall(w: float) -> float:
        return power_chi_square(w, n, dof, alpha) - target_power

    lo, hi = _W_SEARCH
    if shortfall(hi) < 0:
        # Even an implausibly large effect would not reach target power here.
        return float(hi)
    return float(optimize.brentq(shortfall, lo, hi, xtol=1e-12, rtol=1e-12))


def simulate_mde_2x2(
    n: int,
    minority_share: float,
    base_rate: float,
    alpha: float = DEFAULT_ALPHA,
    target_power: float = DEFAULT_TARGET_POWER,
    n_sims: int = 5000,
    seed: int | None = None,
) -> float:
    """Simulation-based MDE for a 2x2 table."""
    def shortfall(w: float) -> float:
        return simulate_power_2x2(w, n, minority_share, base_rate, alpha, n_sims, seed) - target_power

    lo, hi = _W_SEARCH
    
    # If the effect is unreachable or underpowered even at max w
    if shortfall(hi) < 0:
        return float(hi)
        
    try:
        return float(optimize.brentq(shortfall, lo, hi, xtol=1e-4, rtol=1e-4))
    except ValueError:
        return float(hi)


def mde_cramers_v(
    n: int,
    dof: int,
    df_min: int,
    alpha: float = DEFAULT_ALPHA,
    target_power: float = DEFAULT_TARGET_POWER,
) -> float:
    """The same MDE expressed on the Cramer's V scale, V = w / sqrt(df_min)."""
    if df_min < 1:
        raise ValueError("df_min must be at least 1")
    return mde_chi_square(n, dof, alpha, target_power) / math.sqrt(df_min)
