"""Empirical calibration of the detectability verdict.

plan.md Sec 3.3 is the reason this exists. Asserting that a 40-row null gets
labelled BLIND_SPOT proves nothing: the label and the MDE come from the same
formula, so such a test can only fail if the code contradicts itself. It
verifies the implementation, not the claim.

The claim under test is empirical and can genuinely fail:

    when the tool says ADEQUATE, the real detection rate at the SESOI is
    at least the target power; when it says the test was underpowered, it
    is materially below.

So: simulate datasets with a known true effect, run the real pipeline, and
count what actually happens.

SCOPE. This is a *reduced* version of the roadmap M3 experiment -- fewer
replicates, fewer grid points, 2x2 tables only. It runs in minutes rather than
hours and is a smoke test, not the M3 gate. Passing it does not pass M3.

Constructing a 2x2 with a chosen Cohen's w. For a table of group (shares
g1, g2) against a binary event with overall rate p, the closed form is

    w = |p1 - p2| * sqrt(g1 * g2 / (p * (1 - p)))

so the rate difference delivering a target w is d = w * sqrt(p(1-p)/(g1 g2)),
with the two group rates p1 = p + g2*d and p2 = p - g1*d. Beyond a certain w
those rates leave [0, 1] -- most sharply under extreme imbalance -- and such
cells are reported as unreachable rather than silently clipped.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from dbias.audit import audit
from dbias.models.enums import Detectability, Severity


@dataclass(frozen=True)
class Cell:
    n: int
    true_w: float
    minority_share: float


def rates_for(true_w: float, minority_share: float, base_rate: float = 0.5):
    """Group event rates producing `true_w`, or None if unreachable."""
    g2 = minority_share
    g1 = 1.0 - g2
    if true_w == 0:
        return base_rate, base_rate
    d = true_w * np.sqrt(base_rate * (1 - base_rate) / (g1 * g2))
    p1 = base_rate + g2 * d
    p2 = base_rate - g1 * d
    if not (0.01 <= p1 <= 0.99 and 0.01 <= p2 <= 0.99):
        return None
    return p1, p2


def simulate_frame(cell: Cell, rng: np.random.Generator) -> pd.DataFrame:
    """One dataset with the cell's true effect built in."""
    rates = rates_for(cell.true_w, cell.minority_share)
    if rates is None:
        raise ValueError(f"unreachable cell: {cell}")
    p_major, p_minor = rates

    n_minor = max(2, int(round(cell.n * cell.minority_share)))
    n_major = cell.n - n_minor
    group = np.array(["major"] * n_major + ["minor"] * n_minor)
    threshold = np.where(group == "major", p_major, p_minor)
    event = rng.random(cell.n) < threshold

    value = np.full(cell.n, 1.0)
    value[event] = np.nan  # the event is "this field is missing"
    return pd.DataFrame({"group": group, "value": value})


def run_cell(
    cell: Cell,
    replicates: int,
    sesoi: float = 0.1,
    alpha: float = 0.05,
    seed: int = 0,
    n_resamples: int = 120,
) -> dict:
    """Run the real pipeline `replicates` times and tally what it said."""
    rng = np.random.default_rng(seed)
    detected = 0
    adequate = 0
    blind_spots = 0
    false_all_clears = 0
    usable = 0

    for _ in range(replicates):
        frame = simulate_frame(cell, rng)
        result = audit(
            frame,
            sensitive_cols=["group"],
            sesoi=sesoi,
            alpha=alpha,
            n_resamples=n_resamples,
            seed=int(rng.integers(1 << 31)),
        )
        matches = [f for f in result.findings if f.id == "MAR_VALUE_GROUP"]
        if not matches:
            continue
        finding = matches[0]
        usable += 1

        if finding.is_significant:
            detected += 1
        if finding.detectability is Detectability.ADEQUATE:
            adequate += 1
        if finding.severity is Severity.BLIND_SPOT:
            blind_spots += 1
        # The failure that matters: a real effect at or above the SESOI,
        # missed, and reported as though the data were clean.
        if (
            cell.true_w >= sesoi
            and not finding.is_significant
            and finding.severity is Severity.INFORMATIONAL
        ):
            false_all_clears += 1

    return {
        "n": cell.n,
        "true_w": cell.true_w,
        "minority_share": cell.minority_share,
        "replicates": usable,
        "detection_rate": detected / usable if usable else float("nan"),
        "adequate_rate": adequate / usable if usable else float("nan"),
        "blind_spot_rate": blind_spots / usable if usable else float("nan"),
        "false_all_clears": false_all_clears,
    }


def sweep(
    n_values: list[int],
    w_values: list[float],
    minority_shares: list[float],
    replicates: int,
    **kwargs,
) -> pd.DataFrame:
    rows = []
    for n in n_values:
        for w in w_values:
            for share in minority_shares:
                cell = Cell(n=n, true_w=w, minority_share=share)
                if rates_for(w, share) is None:
                    continue
                rows.append(run_cell(cell, replicates=replicates, **kwargs))
    return pd.DataFrame(rows)
