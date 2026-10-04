"""Per-category risk, and the coverage vector that must travel with it.

docs/05 Sec 2 forbids a single aggregate bias score, and nothing here computes
one. Two reviewers reached that conclusion independently; the pressure to
produce one dashboard number is exactly what this module refuses.

The coverage vector is the second half of the answer. A risk vector alone
cannot distinguish "we checked this category and it is clean" from "we could
not check this category", and reporting the two identically is the failure
mode the project exists to fix.
"""
from collections.abc import Iterable
from typing import Any

from dbias.models.enums import Category, Detectability, Severity
from dbias.models.finding import Finding

# Worst-first. BLIND_SPOT sits above INFORMATIONAL because "we could not see"
# demands more attention than "we looked and there was nothing", but below any
# confirmed disparity because it is an absence of evidence, not evidence.
_RANK: dict[Severity, int] = {
    Severity.UNDETERMINED: -1,
    Severity.INFORMATIONAL: 0,
    Severity.BLIND_SPOT: 1,
    Severity.LOW: 2,
    Severity.MEDIUM: 3,
    Severity.HIGH: 4,
    Severity.CRITICAL: 5,
}


def risk_vector(findings: Iterable[Finding]) -> dict[Category, Severity]:
    """Worst severity per category.

    Categories with no findings are omitted rather than reported as clean.
    """
    worst: dict[Category, Severity] = {}
    for finding in findings:
        current = worst.get(finding.category)
        if current is None or _RANK[finding.severity] > _RANK[current]:
            worst[finding.category] = finding.severity
    return worst


def coverage_vector(findings: Iterable[Finding]) -> dict[Category, float]:
    """Share of each category's testable cells the audit can speak about.

    A cell counts unless it is a blind spot. The interval outranks the power
    calculation in both directions (rules/severity.py): a cell whose power
    looked adequate but whose interval ruled nothing out is not covered --
    counting it would print "coverage=100%" beside a blind spot -- and an
    underpowered cell whose interval still earned an all-clear is.

    Cells with no rows are excluded from both numerator and denominator: they
    were never hypotheses, so they neither demonstrate nor undermine coverage.
    """
    seen: dict[Category, int] = {}
    total: dict[Category, int] = {}
    for finding in findings:
        if finding.detectability is Detectability.EMPTY:
            continue
        total[finding.category] = total.get(finding.category, 0) + 1
        if finding.severity is not Severity.BLIND_SPOT:
            seen[finding.category] = seen.get(finding.category, 0) + 1
    return {
        category: seen.get(category, 0) / count for category, count in total.items()
    }


def summarise(findings: Iterable[Finding]) -> dict[str, Any]:
    """Risk and coverage side by side. Deliberately produces no scalar score."""
    findings = list(findings)
    risk = risk_vector(findings)
    coverage = coverage_vector(findings)
    # Every category that has a risk entry gets a coverage entry, so the two
    # vectors are always read together.
    for category in risk:
        coverage.setdefault(category, 0.0)
    return {
        "risk": {str(k): str(v) for k, v in risk.items()},
        "coverage": {str(k): v for k, v in coverage.items()},
        "finding_count": len(findings),
        "blind_spot_count": sum(
            1 for f in findings if f.severity is Severity.BLIND_SPOT
        ),
    }
