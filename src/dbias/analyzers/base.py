"""What an analyzer is, and what it is forbidden from doing.

An analyzer turns a dataframe into hypotheses. It does not score them, does not
correct them, and does not decide whether they matter. Severity is assigned
only in rules/ (docs/10 Sec 2, invariant 2), and every finding produced here
leaves with `Severity.UNDETERMINED` and `Detectability.UNKNOWN`.
"""
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from dbias.models.finding import Finding
from dbias.stats.chi_square import GofSample


@dataclass(frozen=True)
class Hypothesis:
    """A finding plus the sample it came from.

    The sample travels with the finding because detectability/ needs to
    re-examine the data to answer "what could this have shown?" -- a question
    the finding's summary statistics alone cannot answer.
    """

    finding: Finding
    sample: np.ndarray | GofSample


class BaseAnalyzer(ABC):
    @abstractmethod
    def analyze(
        self,
        df: pd.DataFrame,
        sensitive_cols: list[str],
        target_col: str | None = None,
    ) -> list[Hypothesis]:
        """Return un-scored, un-annotated hypotheses. Never assigns severity."""


def excluded_features(
    attribute: str, intersections: Mapping[str, Sequence[str]]
) -> set[str]:
    """Columns that must not be tested as features against `attribute`.

    An attribute is never tested against itself. Intersection columns are
    derived from the sensitive attributes, never features in their own right,
    and an intersection is never tested against the parents it was built from
    -- each of those comparisons is a column against a function of itself, and
    shows a maximal association on any data.
    """
    return {attribute, *intersections, *intersections.get(attribute, ())}


def slug(name: str) -> str:
    """Normalise a column name for use inside a finding id."""
    return re.sub(r"[^A-Z0-9]+", "_", str(name).upper()).strip("_")
