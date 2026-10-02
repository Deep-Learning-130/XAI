"""Power-guided descent and Hierarchical FDR.

Implements Yekutieli-style hierarchical testing and power-based pruning
for intersectional subgroups.
"""
from typing import Sequence, Any
from dbias.models.enums import Detectability
from dbias.models.finding import Finding

def power_guided_gating(
    hypotheses: Sequence,  # Sequence[Hypothesis]
    parent_findings: dict[str, Finding], 
    sesoi: float
) -> list[tuple[Any, bool]]:
    """Filter hypotheses based on parent power.
    
    Returns a list of (Hypothesis, skip_interval) tuples.
    If a parent attribute (e.g. 'race') was underpowered (MDE > SESOI),
    then any intersection (e.g. 'race_AND_sex') will mathematically have even
    less power (smaller margins). We mark it skip_interval=True to skip expensive bootstraps.
    """
    gated = []
    for h in hypotheses:
        skip = False
        if "_AND_" in h.finding.sensitive_attribute:
            parents = h.finding.sensitive_attribute.split("_AND_")
            
            for parent in parents:
                parent_key = f"{h.finding.category.name}_{h.finding.target_feature}_{parent}"
                parent_finding = parent_findings.get(parent_key)
                
                if parent_finding and parent_finding.detectability == Detectability.UNDERPOWERED:
                    skip = True
                    break
                    
        gated.append((h, skip))
    return gated
