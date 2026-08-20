"""The universal contract for one audited hypothesis."""
from dataclasses import dataclass, field
from typing import Any

from dbias.models.enums import (
    Category,
    Detectability,
    EffectSizeMetric,
    EquivalenceVerdict,
    Severity,
)


@dataclass
class Finding:
    """One hypothesis, carrying both directions of error.

    `p_value_raw` / `effect_size_value` describe what was found;
    `power_to_detect_sesoi` / `minimum_detectable_effect` / `equivalence_verdict`
    describe what *could* have been found. A null result is meaningless without
    the second half, which is the entire point of this project.

    There is deliberately no observed/achieved power field. Power computed from
    the observed effect is a monotone function of the p-value and carries no
    information (Hoenig & Heisey 2001; plan.md Sec 3.1).
    """

    id: str
    category: Category
    sensitive_attribute: str
    target_feature: str | None
    metric_name: str
    observed_values: dict[str, Any]

    # What was found
    statistical_test: str
    p_value_raw: float
    effect_size_metric: EffectSizeMetric
    effect_size_value: float
    n_per_group: dict[str, int]

    # Populated by stats/correction.py
    p_value_corrected: float | None = None
    is_significant: bool = False

    # What could have been found -- populated by detectability/
    effect_size_ci: tuple[float, float] | None = None
    sesoi: float | None = None
    power_to_detect_sesoi: float | None = None
    minimum_detectable_effect: float | None = None
    mde_is_approximate: bool = False
    equivalence_verdict: EquivalenceVerdict | None = None
    detectability: Detectability = Detectability.UNKNOWN

    # Populated by rules/
    severity: Severity = Severity.UNDETERMINED

    # Populated by the NLG layer (out of scope)
    evidence_text: str = ""
    potential_impact: str = ""
    recommendations: list[str] = field(default_factory=list)

    @property
    def total_n(self) -> int:
        return sum(self.n_per_group.values())
