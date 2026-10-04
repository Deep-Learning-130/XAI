"""Power-guided descent for intersectional subgroups.

This is a compute optimisation, not a multiple-testing procedure. Intersection
findings are corrected in their own (category, attribute) families exactly as
parent attributes are (see audit.py); hierarchical FDR in the Yekutieli sense
-- conditioning a child's rejection on its parent's -- is not implemented.
"""
from typing import Sequence, Any
from dbias.models.enums import Detectability
from dbias.models.finding import Finding

def power_guided_gating(
    hypotheses: Sequence,  # Sequence[Hypothesis]
    parent_findings: dict[str, Finding],
    parents: dict[str, list[str]],
) -> list[tuple[Any, bool]]:
    """Mark which intersection hypotheses may skip their bootstrap interval.

    Returns a list of (Hypothesis, skip_interval) tuples. `parents` maps each
    intersection column to the attributes it was built from (see
    `hierarchy.parents_of`). If a parent attribute (e.g. 'race') was
    underpowered for the same category and feature, the intersection
    (e.g. 'race_AND_sex') is a candidate for skipping the expensive bootstrap.
    The skip only takes effect if the child is itself underpowered;
    `annotate_detectability` checks that.
    """
    gated = []
    for h in hypotheses:
        skip = False
        for parent in parents.get(h.finding.sensitive_attribute, []):
            parent_key = f"{h.finding.category.name}_{h.finding.target_feature}_{parent}"
            parent_finding = parent_findings.get(parent_key)

            if parent_finding and parent_finding.detectability == Detectability.UNDERPOWERED:
                skip = True
                break

        gated.append((h, skip))
    return gated
