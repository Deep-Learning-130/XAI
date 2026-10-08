"""What an incumbent fairness toolkit concludes about one audit cell.

Fairlearn reports demographic parity for a binary outcome and leaves the
pass/fail line to the user. The line used here is the four-fifths rule: the
lowest group rate must be at least 80% of the highest. A multi-level feature
becomes one indicator per level, and the cell is "clean" only if every level
passes (docs/superpowers/plans/2026-10-04-m6-compas-incumbent.md, pre-registered).
"""
from dataclasses import dataclass

import pandas as pd

FOUR_FIFTHS = 0.8


@dataclass(frozen=True)
class LevelVerdict:
    level: str
    selection_rates: dict[str, float]  # P(feature == level | group)
    ratio: float  # lowest / highest rate -- Fairlearn's demographic_parity_ratio
    difference: float  # highest - lowest -- demographic_parity_difference
    passes_four_fifths: bool


@dataclass(frozen=True)
class CellVerdict:
    feature: str
    attribute: str
    n: int
    levels: list[LevelVerdict]

    @property
    def reports_clean(self) -> bool:
        return all(level.passes_four_fifths for level in self.levels)

    @property
    def max_difference(self) -> float:
        return max(level.difference for level in self.levels)


def four_fifths_cell(df: pd.DataFrame, feature: str, attribute: str) -> CellVerdict:
    rows = df[[feature, attribute]].dropna()
    groups = rows[attribute].astype(str)
    levels = []
    for level in sorted(rows[feature].unique(), key=str):
        rates = (rows[feature] == level).groupby(groups).mean()
        high, low = float(rates.max()), float(rates.min())
        ratio = 1.0 if high == 0 else low / high
        levels.append(
            LevelVerdict(
                level=str(level),
                selection_rates={str(g): float(r) for g, r in rates.items()},
                ratio=ratio,
                difference=high - low,
                passes_four_fifths=ratio >= FOUR_FIFTHS,
            )
        )
    return CellVerdict(feature=feature, attribute=attribute, n=len(rows), levels=levels)
