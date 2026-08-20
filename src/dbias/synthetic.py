"""Synthetic datasets whose true effects are known by construction.

This is the only module in the project where ground truth exists. Everything
else measures; this declares. That asymmetry is what makes it possible to ask
whether a verdict is *right* rather than merely self-consistent
(plan.md Sec 3.3).

Group sizes are allocated deterministically rather than sampled, so that a
seeded scenario has exactly the n it claims -- a scenario about statistical
power should not itself vary in power between runs.
"""
from typing import Any

import numpy as np
import pandas as pd

LABEL_UNOBSERVED_SHARE = 0.33
"""Share of rows whose outcome has not landed yet. Realistic, and the reason
label-disparity cells sit on less data than missingness cells."""


def _allocate(n: int, shares: dict[str, float]) -> list[str]:
    """Deterministic group labels honouring `shares` exactly.

    Largest-remainder allocation, so rounding never loses or invents rows.
    """
    total = sum(shares.values())
    if not np.isclose(total, 1.0):
        raise ValueError(f"group_shares must sum to 1, got {total}")

    exact = {name: n * share for name, share in shares.items()}
    counts = {name: int(np.floor(value)) for name, value in exact.items()}
    remainder = n - sum(counts.values())
    by_fraction = sorted(exact, key=lambda name: exact[name] - counts[name], reverse=True)
    for name in by_fraction[:remainder]:
        counts[name] += 1

    labels: list[str] = []
    for name, count in counts.items():
        labels.extend([name] * count)
    return labels


def make_missingness_frame(
    n: int,
    group_shares: dict[str, float],
    missing_rates: dict[str, float] | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """A two-column frame with a controllable missingness disparity.

    `missing_rates` maps each group to the probability its `value` is missing.
    Equal rates across groups is a true null; unequal rates is a true MAR
    disparity of a size the caller chose.
    """
    rng = np.random.default_rng(seed)
    groups = np.array(_allocate(n, group_shares))
    rng.shuffle(groups)

    rates = missing_rates or dict.fromkeys(group_shares, 0.2)
    value = rng.normal(50_000, 12_000, n)
    draws = rng.random(n)
    threshold = np.array([rates[g] for g in groups])
    value[draws < threshold] = np.nan

    return pd.DataFrame({"group": groups, "value": value})


def make_audit_scenario(
    n: int = 900, seed: int | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The flagship scenario: a real disparity the sample cannot see.

    Two attributes, identical sample size, opposite truths:

      * `gender` -- two balanced groups, missingness rates genuinely equal.
        A true null. With n this large and one degree of freedom, the audit
        can rule out effects at the SESOI, so this is an *earned* all-clear.

      * `ethnicity` -- five groups, missingness rates genuinely unequal, with
        a true Cohen's w of 0.102: just above a SESOI of 0.1, so an effect the
        user has declared they care about. Spread over four degrees of freedom
        instead of one, the same n yields only ~0.66 power, so the test misses
        it about a third of the time.

    Be precise about what the demonstration shows, because the honest version
    is the persuasive one. The ethnicity test is *not* guaranteed to miss --
    at n=900 it detects the injected effect roughly two runs in three. The
    claim is about the third run. On that run an incumbent tool prints exactly
    the "no significant disparity found" it printed for gender, and the two
    sentences mean opposite things. This tool has to say so.

    `tests/calibration/` measures that miss rate rather than asserting it.

    Returns the frame and its ground truth.
    """
    rng = np.random.default_rng(seed)

    gender = np.array(_allocate(n, {"male": 0.5, "female": 0.5}))
    rng.shuffle(gender)
    ethnicity = np.array(
        _allocate(n, {"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.2, "E": 0.2})
    )
    rng.shuffle(ethnicity)
    age_band = np.array(_allocate(n, {"18-34": 1 / 3, "35-54": 1 / 3, "55+": 1 / 3}))
    rng.shuffle(age_band)

    # income goes missing at genuinely different rates across ethnicity, and at
    # genuinely identical rates across gender.
    ethnicity_rates = {"A": 0.16, "B": 0.19, "C": 0.22, "D": 0.25, "E": 0.28}
    income = rng.normal(52_000, 14_000, n)
    income[rng.random(n) < np.array([ethnicity_rates[e] for e in ethnicity])] = np.nan

    # education goes missing at a rate independent of every attribute.
    education = rng.choice(["school", "college", "postgrad"], n).astype(object)
    education[rng.random(n) < 0.15] = None

    # The label is independent of every attribute: a true null throughout.
    # It is also unknown for a third of the rows -- outcomes that have not
    # landed yet, which is the ordinary state of an outcome column in real
    # data. That makes every label-disparity test run on fewer rows than the
    # missingness tests, so the coverage map has genuine per-cell variation
    # rather than one number per attribute.
    hired = np.where(rng.random(n) < 0.35, 1.0, 0.0)
    hired[rng.random(n) < LABEL_UNOBSERVED_SHARE] = np.nan

    frame = pd.DataFrame(
        {
            "gender": gender,
            "ethnicity": ethnicity,
            "age_band": age_band,
            "income": income,
            "education": education,
            "hired": hired,
        }
    )
    truth = {
        "n": n,
        "seed": seed,
        "sensitive_cols": ["gender", "ethnicity", "age_band"],
        "target_col": "hired",
        # Cohen's w of the injected missingness disparity, by attribute.
        "true_effects": {
            "gender": 0.0,
            "ethnicity": _cohens_w_of_rates(ethnicity_rates),
            "age_band": 0.0,
        },
        "notes": (
            "income missingness varies genuinely across ethnicity and not at "
            "all across gender or age_band; hired is independent of every "
            f"attribute and unobserved for about {LABEL_UNOBSERVED_SHARE:.0%} "
            "of rows."
        ),
    }
    return frame, truth


def _cohens_w_of_rates(rates: dict[str, float]) -> float:
    """Cohen's w for a 2 x k missingness table with equal group sizes."""
    p = np.array(list(rates.values()))
    k = p.size
    # Joint cell probabilities of (missing?, group), groups equally sized.
    observed = np.vstack([p, 1 - p]) / k
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    return float(np.sqrt((((observed - expected) ** 2) / expected).sum()))
