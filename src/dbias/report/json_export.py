"""Data Card-shaped JSON, with the audit's own assumptions on the front.

Two things this file is careful about.

The SESOI appears verbatim in `configuration`, on every report, with the scale
it is measured on. Every verdict in the document is conditional on it, and a
reader who does not know the number cannot read the document (plan.md Sec 3.6).

The limitations are part of the artifact, not the paper. This tool measures
statistical association between observed columns. It does not detect
discrimination, it does not establish cause, and it does not mitigate
anything. A JSON file gets pasted into compliance documents long after its
caveats have been forgotten, so the caveats travel with it.
"""
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dbias.audit import AuditResult
from dbias.detectability.coverage import coverage_map
from dbias.models.enums import Detectability, Severity
from dbias.models.finding import Finding

LIMITATIONS = [
    "Findings are statistical associations between observed columns. They are "
    "not causal claims and are not evidence of discrimination.",
    "Every verdict is conditional on the declared SESOI. A different SESOI "
    "would move cells between 'earned all-clear' and 'blind spot'.",
    "Missingness findings detect MAR -- missingness that depends on an "
    "observed attribute. True MNAR is not identifiable from observed data.",
    "Minimum detectable effects are computed from a non-central chi-square "
    "that ignores marginal structure, except for skewed or sparse 2x2 tables, "
    "which are simulated. Cells flagged mde_is_approximate are larger tables "
    "in the skewed or sparse regime, where the approximation errs optimistic.",
    "This tool audits data. It does not audit models, and a clean dataset "
    "does not imply a fair model.",
]


def blind_spot_reading(finding: Finding) -> str:
    """Why a null result rules nothing out, in the SESOI's own units.

    A blind spot has one of two causes, and saying the wrong one misleads: the
    test was underpowered, or power looked adequate but the realised interval
    still did not rule out an effect at the SESOI.
    """
    return f"No disparity was detected, but {blind_spot_reason(finding)}. This is not a clean result."


def blind_spot_reason(finding: Finding) -> str:
    """The cause clause of :func:`blind_spot_reading`, also used by the CLI.

    The cause is read from `detectability`, which is what decided it -- not
    re-derived from the MDE, which on the simulated 2x2 path comes from a
    separate Monte Carlo search and can disagree with the power by a hair.
    """
    sesoi = "the SESOI" if finding.sesoi is None else f"the SESOI (w = {finding.sesoi:.3f})"
    if finding.detectability is Detectability.UNDERPOWERED:
        mde_w = finding.mde_w
        reach = "" if mde_w is None else f"; it could only have caught w = {mde_w:.3f} or larger"
        return f"the test was not powered to detect {sesoi}{reach}"
    if finding.effect_size_ci is None:
        return f"no interval was computed, so {sesoi} was not ruled out"
    return (
        f"although the test was powered for {sesoi}, the realised interval "
        f"did not rule out an effect that large"
    )


def _finding_to_dict(finding: Finding) -> dict[str, Any]:
    record = asdict(finding)
    record["total_n"] = finding.total_n
    return {k: (str(v) if hasattr(v, "value") else v) for k, v in record.items()}


def to_dict(result: AuditResult) -> dict[str, Any]:
    grid = coverage_map(result.findings)
    blind = [f for f in result.findings if f.severity is Severity.BLIND_SPOT]

    return {
        "tool": "dbias",
        "configuration": {
            "sesoi": result.sesoi,
            "sesoi_scale": "Cohen's w (converted to Cramer's V per table shape)",
            "alpha": result.alpha,
            "target_power": result.target_power,
            "correction": "Benjamini-Hochberg within each (category, attribute) family",
            "correction_families": {
                f"{category}|{attribute}": size
                for (category, attribute), size in result.correction_families.items()
            },
        },
        "summary": result.summary,
        "coverage_map": {
            "attributes": grid.attributes,
            "features": grid.features,
            "cells": [
                {
                    "attribute": cell.attribute,
                    "feature": cell.feature,
                    "minimum_detectable_effect": cell.mde,
                    "minimum_detectable_effect_w": cell.mde_w,
                    "detectability": str(cell.detectability),
                    "severity": str(cell.severity),
                    "n": cell.n,
                    "mde_is_approximate": cell.approximate,
                }
                for cell in grid.cells.values()
            ],
        },
        "blind_spots": [
            {
                "id": f.id,
                "attribute": f.sensitive_attribute,
                "feature": f.target_feature,
                "reading": blind_spot_reading(f),
            }
            for f in blind
        ],
        "findings": [_finding_to_dict(f) for f in result.findings],
        "limitations": LIMITATIONS,
    }


def write_json(result: AuditResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_dict(result), indent=2), encoding="utf-8")
    return path
