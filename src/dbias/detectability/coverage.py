"""The coverage map: what each cell of the audit could have seen.

An ordinary audit report answers "what did we find?". This grid answers the
question underneath it -- "where could we have found anything?" -- for every
(attribute, feature) pair the audit touched. A cell holds the smallest effect
that pair could have detected, so a reader can see at a glance which parts of
the dataset the audit is entitled to speak about.

Cells that were never tested stay empty. Filling them with a default would
reintroduce exactly the false reassurance the map exists to remove.
"""
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from dbias.models.enums import Category, Detectability, Severity
from dbias.models.finding import Finding
from dbias.stats.effect_sizes import w_from_v

# Representation has no target feature -- it is a statement about the
# attribute itself -- but it still earns a column on the grid.
REPRESENTATION_COLUMN = "(representation)"

# Several categories can test the same column -- missingness of `workclass`
# and the distribution of `workclass` are different hypotheses -- so every
# category but missingness tags its column, and no cell overwrites another.
_COLUMN_SUFFIX = {
    Category.LABEL_DISPARITY: " (label)",
    Category.FEATURE_DISPARITY: " (distribution)",
}


def column_for(finding: Finding) -> str:
    """The grid column a finding occupies."""
    if finding.category is Category.REPRESENTATION or finding.target_feature is None:
        return REPRESENTATION_COLUMN
    return finding.target_feature + _COLUMN_SUFFIX.get(finding.category, "")


@dataclass(frozen=True)
class CoverageCell:
    attribute: str
    feature: str
    mde: float | None  # Cramer's V, as on the finding
    df_min: int | None
    detectability: Detectability
    severity: Severity
    n: int
    approximate: bool
    finding_id: str

    @property
    def mde_w(self) -> float | None:
        """The MDE on the SESOI's (Cohen's w) scale, so cells of different
        table shapes can be compared against one declared SESOI."""
        return None if self.mde is None else w_from_v(self.mde, self.df_min or 1)


@dataclass(frozen=True)
class CoverageMap:
    attributes: list[str]
    features: list[str]
    cells: dict[tuple[str, str], CoverageCell]

    def cell(self, attribute: str, feature: str) -> CoverageCell | None:
        return self.cells.get((attribute, feature))

    def matrix(self) -> np.ndarray:
        """MDE per cell on the Cohen's w scale, NaN where never tested."""
        out = np.full((len(self.attributes), len(self.features)), np.nan)
        for row, attribute in enumerate(self.attributes):
            for col, feature in enumerate(self.features):
                found = self.cells.get((attribute, feature))
                if found is not None and found.mde_w is not None:
                    out[row, col] = found.mde_w
        return out

    def blind_spots(self) -> list[CoverageCell]:
        return [c for c in self.cells.values() if c.severity is Severity.BLIND_SPOT]

    def worst_mde(self, attribute: str) -> float | None:
        """Largest MDE (Cohen's w) across an attribute's row -- its weakest cell."""
        values = [
            c.mde_w
            for (a, _), c in self.cells.items()
            if a == attribute and c.mde_w is not None
        ]
        return max(values) if values else None


def coverage_map(findings: Iterable[Finding]) -> CoverageMap:
    cells: dict[tuple[str, str], CoverageCell] = {}
    for finding in findings:
        feature = column_for(finding)
        key = (finding.sensitive_attribute, feature)
        cells[key] = CoverageCell(
            attribute=finding.sensitive_attribute,
            feature=feature,
            mde=finding.minimum_detectable_effect,
            df_min=finding.df_min,
            detectability=finding.detectability,
            severity=finding.severity,
            n=finding.total_n,
            approximate=finding.mde_is_approximate,
            finding_id=finding.id,
        )

    attributes = sorted({a for a, _ in cells})
    features = sorted({f for _, f in cells if f != REPRESENTATION_COLUMN})
    if any(f == REPRESENTATION_COLUMN for _, f in cells):
        features.insert(0, REPRESENTATION_COLUMN)
    return CoverageMap(attributes=attributes, features=features, cells=cells)
