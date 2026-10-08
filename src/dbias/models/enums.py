"""Inert vocabulary shared by every layer.

Deliberately knows nothing: no thresholds, no policy, no statistics. Anything
that decides *which* member applies lives in stats/, detectability/ or rules/.
"""
from enum import StrEnum


class Severity(StrEnum):
    """Assigned only in rules/. Analyzers emit UNDETERMINED (docs/10 Sec 2)."""

    UNDETERMINED = "Undetermined"
    BLIND_SPOT = "Blind Spot"
    INFORMATIONAL = "Informational"
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class Category(StrEnum):
    REPRESENTATION = "Representation"
    MISSINGNESS = "Missingness"
    FEATURE_DISPARITY = "Feature_Disparity"
    LABEL_DISPARITY = "Label_Disparity"


class EffectSizeMetric(StrEnum):
    CRAMERS_V = "Cramer's V"
    COHENS_W = "Cohen's w"
    RANK_BISERIAL = "Rank-Biserial Correlation"
    ETA_SQUARED = "Eta-Squared"
    KS_STATISTIC = "KS Statistic (D)"
    ODDS_RATIO = "Odds Ratio"


class VariableKind(StrEnum):
    CATEGORICAL = "categorical"
    CONTINUOUS = "continuous"
    BOOLEAN = "boolean"


class Detectability(StrEnum):
    """What the sample *could* have shown, independent of what it did show."""

    UNKNOWN = "unknown"            # detectability pass has not run
    ADEQUATE = "adequate"          # could have caught an effect at the SESOI
    UNDERPOWERED = "underpowered"  # could not; a null result here means nothing
    EMPTY = "empty"                # no rows to speak from; an absence, not a blind spot


class Magnitude(StrEnum):
    TRIVIAL = "trivial"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class EquivalenceVerdict(StrEnum):
    """Outcome of the TOST-style decision against a declared SESOI.

    plan.md Sec 3.2: the verdict is an equivalence decision, not an MDE
    comparison. MDE remains the human-facing communication device.
    """

    EQUIVALENT = "equivalent"          # CI rules out effects at or above the SESOI
    DISPARITY = "disparity"            # CI excludes zero; a real effect is present
    INCONCLUSIVE = "inconclusive"      # CI spans both; the sample cannot decide
