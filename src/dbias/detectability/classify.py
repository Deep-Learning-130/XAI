"""Turn a finding's confidence interval into a verdict against a declared SESOI.

Deliberately does not know: whether a disparity is harmful, or how severe it
is. Severity is assigned only in rules/ (docs/10 Sec 2, invariant 2).

The design in one paragraph. Every incumbent auditing tool returns "no
significant disparity found" in identical words for a 40-row subgroup and a
400,000-row subgroup. That is an unsafe output. Here a null result is split in
two: if the confidence interval on the effect size sits entirely below the
smallest effect size of interest, the audit has *earned* its all-clear and can
say so as a positive claim -- effects at or above the SESOI are ruled out. If
the interval straddles the SESOI, the sample simply could not see, and the
cell is a blind spot. The MDE is carried alongside as the number a human
reads; it is not what decides the verdict (plan.md Sec 3.2).

The SESOI is required configuration with no silent default. A tool whose
central verdict is "we could have caught anything that matters" must make the
user define what matters, and must echo that number in the report
(plan.md Sec 3.6).
"""
import dataclasses

import numpy as np
from numpy.typing import ArrayLike

from dbias.detectability.power import (
    DEFAULT_ALPHA,
    DEFAULT_TARGET_POWER,
    mde_cramers_v,
    power_chi_square,
    simulate_mde_2x2,
    simulate_power_2x2,
)
from dbias.models.enums import Detectability, EquivalenceVerdict
from dbias.models.finding import Finding
from dbias.stats.chi_square import GofSample, chi_square_goodness_of_fit, chi_square_test
from dbias.stats.effect_sizes import v_from_w
from dbias.stats.intervals import (
    DEFAULT_RESAMPLES,
    bootstrap_ci_cramers_v,
    bootstrap_ci_gof_w,
)

# Seed for the 2x2 power simulation when the caller gives none.
_SIMULATION_SEED = 0


def _measure(sample: ArrayLike | GofSample):
    """Return (ChiSquareResult, total_n, interval_function) for either sample kind."""
    if isinstance(sample, GofSample):
        counts = np.asarray(sample.counts, dtype=float)
        result = chi_square_goodness_of_fit(counts, sample.expected_probs)

        def interval(n_resamples: int, confidence: float, seed: int | None):
            return bootstrap_ci_gof_w(
                counts,
                sample.expected_probs,
                n_resamples=n_resamples,
                confidence=confidence,
                seed=seed,
            )

        return result, float(counts.sum()), interval

    table = np.asarray(sample, dtype=float)
    total = float(table.sum())
    if total == 0:
        return None, 0.0, None
    result = chi_square_test(table)

    def interval(n_resamples: int, confidence: float, seed: int | None):
        return bootstrap_ci_cramers_v(
            table, n_resamples=n_resamples, confidence=confidence, seed=seed
        )

    return result, total, interval


def annotate_detectability(
    finding: Finding,
    sample: ArrayLike | GofSample,
    *,
    sesoi: float,
    alpha: float = DEFAULT_ALPHA,
    target_power: float = DEFAULT_TARGET_POWER,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int | None = None,
    skip_interval: bool = False,
) -> Finding:
    """Return a copy of `finding` carrying its detectability annotation.

    `sample` is either a 2-D contingency table (a test of independence) or a
    :class:`GofSample` (a test of representation against a reference).

    `sesoi` is the smallest effect size of interest, declared on the Cohen's w
    scale so that it is comparable across table shapes. It is converted to the
    Cramer's V scale internally, since that is the scale the finding's effect
    size and interval are on.
    """
    if sesoi <= 0:
        raise ValueError("sesoi must be positive; it is what 'matters' means here")

    result, total, interval = _measure(sample)
    if result is None or total == 0:
        # No rows is an absence, not a blind spot: there is nothing to be blind to.
        return dataclasses.replace(
            finding, sesoi=sesoi, detectability=Detectability.EMPTY
        )

    sesoi_v = v_from_w(sesoi, result.df_min)

    # Fallback to empirical simulation if analytic MDE is optimistic (plan.md Sec 3.4)
    # The 2x2 fallback handles both power and MDE.
    used_approximation = result.is_sparse or result.is_skewed
    if used_approximation and result.dof == 1 and not isinstance(sample, GofSample):
        table = np.asarray(sample, dtype=float)
        row_sums = table.sum(axis=1)
        col_sums = table.sum(axis=0)

        minority_share = float(row_sums.min() / result.n)
        base_rate = float(col_sums[0] / result.n)
        # The simulation must not make the verdict vary between runs, so it
        # never draws from fresh entropy even when the caller passes no seed.
        sim_seed = _SIMULATION_SEED if seed is None else seed

        power = simulate_power_2x2(
            w=sesoi,
            n=result.n,
            minority_share=minority_share,
            base_rate=base_rate,
            alpha=alpha,
            seed=sim_seed,
        )
        mde_v = simulate_mde_2x2(
            n=result.n,
            minority_share=minority_share,
            base_rate=base_rate,
            alpha=alpha,
            target_power=target_power,
            seed=sim_seed,
        ) / np.sqrt(result.df_min)  # convert back to Cramer's V scale

        # The simulation runs the same uncorrected chi-square the audit runs,
        # so its power is the power of the test actually performed, up to
        # Monte Carlo error -- not the marginal-blind n * w^2 approximation.
        used_approximation = False
    else:
        power = power_chi_square(w=sesoi, n=result.n, dof=result.dof, alpha=alpha)
        mde_v = mde_cramers_v(
            n=result.n,
            dof=result.dof,
            df_min=result.df_min,
            alpha=alpha,
            target_power=target_power,
        )

    detectability = (
        Detectability.ADEQUATE if power >= target_power else Detectability.UNDERPOWERED
    )

    if skip_interval and detectability is Detectability.UNDERPOWERED:
        # Power-guided descent: the parent was underpowered and so is this
        # cell, so the bootstrap is skipped. No interval was computed, so none
        # is reported and there is no verdict; rules/ falls back to
        # detectability, which reads an underpowered null as a blind spot.
        effect_size_ci = None
        verdict = None
    else:
        ci_lo, ci_hi = interval(n_resamples, 1.0 - alpha, seed)
        effect_size_ci = (ci_lo, ci_hi)

        if ci_hi < sesoi_v:
            verdict = EquivalenceVerdict.EQUIVALENT
        elif ci_lo > sesoi_v:
            verdict = EquivalenceVerdict.DISPARITY
        else:
            verdict = EquivalenceVerdict.INCONCLUSIVE

    return dataclasses.replace(
        finding,
        sesoi=sesoi,
        effect_size_ci=effect_size_ci,
        power_to_detect_sesoi=power,
        minimum_detectable_effect=mde_v,
        mde_is_approximate=used_approximation,
        equivalence_verdict=verdict,
        detectability=detectability,
    )
