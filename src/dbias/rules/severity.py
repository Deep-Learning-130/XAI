"""The only place in the codebase that assigns severity (docs/10 invariant 2).

The table below is the docs/05 ladder plus one row that docs/05 cannot express.
docs/05 grades on significance and magnitude alone, so a null result always
lands on INFORMATIONAL -- whether the audit looked at four hundred thousand
rows or forty. This module splits that case in two:

    not significant  +  EQUIVALENT    ->  INFORMATIONAL   (an earned all-clear)
    not significant  +  INCONCLUSIVE  ->  BLIND_SPOT      (we could not see)

Those two findings carry the same p-value and opposite meanings, and telling
them apart is the entire contribution of this project.
"""
from dataclasses import replace

from dbias.models.enums import (
    Category,
    Detectability,
    EquivalenceVerdict,
    Magnitude,
    Severity,
)
from dbias.models.finding import Finding
from dbias.stats.thresholds import magnitude as classify_magnitude

_MAGNITUDE_LADDER: dict[Magnitude, Severity] = {
    Magnitude.TRIVIAL: Severity.INFORMATIONAL,
    Magnitude.SMALL: Severity.LOW,
    Magnitude.MEDIUM: Severity.MEDIUM,
    Magnitude.LARGE: Severity.HIGH,
}


def _magnitude_of(finding: Finding) -> Magnitude:
    """Grade the effect against the declared SESOI, not against a 1988 convention.

    plan.md Sec 3.6: the SESOI is what "matters" means for this audit, so it
    anchors the ladder. Bands are multiples of it -- an effect below the SESOI
    is trivial by the user's own definition, and the higher bands follow the
    3x / 5x spacing of Cohen's small/medium/large.
    """
    # The effect size is Cramer's V; the SESOI is declared as Cohen's w.
    sesoi = finding.sesoi_v
    if sesoi is None:
        return classify_magnitude(
            finding.effect_size_metric,
            finding.effect_size_value,
            df_min=1,
        )
    effect = abs(finding.effect_size_value)
    if effect < sesoi:
        return Magnitude.TRIVIAL
    if effect < 3 * sesoi:
        return Magnitude.SMALL
    if effect < 5 * sesoi:
        return Magnitude.MEDIUM
    return Magnitude.LARGE


def _null_result_severity(finding: Finding) -> Severity:
    """Split a null result into an earned all-clear and a blind spot.

    The equivalence verdict decides, not the MDE. plan.md Sec 3.2 puts TOST at
    the inference layer and keeps MDE as the communication layer, and the two
    can disagree: the MDE is a design-stage approximation computed before the
    data arrived, blind to the marginal structure, while the interval is
    computed from the sample that actually turned up. When they conflict the
    interval is the better evidence in both directions -- it can vindicate a
    test the power calculation pessimistically wrote off, and it can withdraw
    an adequacy the power calculation promised.

    Only EQUIVALENT earns the all-clear. A non-significant finding whose
    interval sits above the SESOI (DISPARITY) has not ruled the SESOI out
    either -- near the null the bootstrap interval on V is biased upward and
    its lower bound carries no inference (stats/intervals.py) -- so it is a
    blind spot, never an all-clear on the strength of the power calculation.

    `detectability` is the fallback for metrics with no interval, and remains
    what the coverage map reports.
    """
    if finding.equivalence_verdict is EquivalenceVerdict.EQUIVALENT:
        return Severity.INFORMATIONAL
    if finding.equivalence_verdict is not None:
        return Severity.BLIND_SPOT
    if finding.detectability is Detectability.ADEQUATE:
        return Severity.INFORMATIONAL
    return Severity.BLIND_SPOT


def assign_severity(finding: Finding) -> Finding:
    """Return a copy of `finding` with its severity set."""
    if finding.detectability is Detectability.UNKNOWN:
        raise ValueError(
            f"finding {finding.id!r} reached the rules engine with "
            "detectability=UNKNOWN; the detectability pass is not optional "
            "(docs/10 Sec 2, invariant 3)"
        )

    if finding.detectability is Detectability.EMPTY:
        # No rows is an absence, not a failure to see.
        return replace(finding, severity=Severity.INFORMATIONAL)

    if not finding.is_significant:
        return replace(finding, severity=_null_result_severity(finding))

    severity = _MAGNITUDE_LADDER[_magnitude_of(finding)]
    if severity is Severity.HIGH and finding.category is Category.LABEL_DISPARITY:
        # A large, real disparity in the outcome itself: training on this data
        # will reproduce it.
        severity = Severity.CRITICAL
    return replace(finding, severity=severity)


def assign_all(findings: list[Finding]) -> list[Finding]:
    return [assign_severity(f) for f in findings]
