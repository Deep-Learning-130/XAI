"""The pipeline that ties the layers together.

    analyzers  ->  detectability  ->  correction  ->  rules

Each arrow is a hard ordering. Severity depends on both detectability and
significance, so rules runs last. Detectability is not a plugin and has no
bypass: a finding that reached the rules engine unannotated raises rather than
being scored as though the audit had seen clearly (docs/10 Sec 2, invariant 3).

FDR family boundary. docs/02 says to pool every p-value in the audit into one
family; this module does not. One family per (category, attribute) means an
unrelated representation test cannot change the verdict on a missingness test,
which is what pooling globally would allow.

Intersections (depth 2, e.g. race_AND_sex) are treated as attributes in their
own right, so each one is its own family per category. That is the same rule
as for parent attributes, and the same caveat applies: FDR is controlled
within a family, not across the audit, and adding intersections adds
families. Hierarchical FDR -- conditioning a child's rejection on its
parent's (Yekutieli) -- is not implemented; the family boundary is chosen so
that it can be added without redefining it. Gating (correction/gating.py)
only decides which underpowered children may skip their bootstrap.
"""
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from dbias.analyzers.base import Hypothesis
from dbias.analyzers.label_disparity import LabelDisparityAnalyzer
from dbias.analyzers.missingness import MissingnessAnalyzer
from dbias.analyzers.representation import RepresentationAnalyzer
from dbias.analyzers.feature_disparity import FeatureDisparityAnalyzer
from dbias.correction.hierarchy import build_intersections, parents_of
from dbias.correction.gating import correct_within_families, power_guided_gating
from dbias.detectability.classify import annotate_detectability
from dbias.detectability.power import DEFAULT_ALPHA, DEFAULT_TARGET_POWER
from dbias.models.enums import Category
from dbias.models.finding import Finding
from dbias.rules.risk_vector import summarise
from dbias.rules.severity import assign_severity
from dbias.stats.intervals import DEFAULT_RESAMPLES


@dataclass(frozen=True)
class AuditResult:
    """Everything the report layer needs, and nothing it has to recompute."""

    findings: list[Finding]
    summary: dict[str, Any]
    sesoi: float
    alpha: float
    target_power: float
    correction_families: dict[tuple[Category, str], int] = field(default_factory=dict)

    @property
    def blind_spots(self) -> list[Finding]:
        from dbias.models.enums import Severity

        return [f for f in self.findings if f.severity is Severity.BLIND_SPOT]


def audit(
    df: pd.DataFrame,
    sensitive_cols: list[str],
    target_col: str | None = None,
    *,
    sesoi: float,
    alpha: float = DEFAULT_ALPHA,
    target_power: float = DEFAULT_TARGET_POWER,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int | None = None,
    reference: dict[str, dict[str, float]] | None = None,
) -> AuditResult:
    """Audit `df` for disparities, and for what it could not have seen.

    `sesoi` -- the smallest effect size of interest, on the Cohen's w scale --
    is required. There is no default, because the audit's central claim is
    phrased in terms of it and a silent convention would put a 1988 rule of
    thumb at the centre of the user's compliance document.
    """
    missing = [c for c in sensitive_cols if c not in df.columns]
    if missing:
        raise ValueError(f"sensitive columns {missing} not in the dataframe")
    if target_col is not None and target_col not in df.columns:
        raise ValueError(f"target column {target_col!r} not in the dataframe")

    # Step 1: Intersections
    df, tree = build_intersections(df, sensitive_cols)
    parents = parents_of(tree)
    child_cols = list(parents)

    def _run_analyzers(cols: list[str]) -> list[Hypothesis]:
        if not cols:
            return []
        hyps = []
        hyps += RepresentationAnalyzer(reference).analyze(df, cols)
        hyps += MissingnessAnalyzer(parents).analyze(df, cols)
        hyps += FeatureDisparityAnalyzer(parents).analyze(df, cols, target_col=target_col)
        if target_col is not None:
            hyps += LabelDisparityAnalyzer().analyze(df, cols, target_col=target_col)
        return hyps

    # Step 2: Base hypotheses
    base_hypotheses = _run_analyzers(sensitive_cols)
    
    base_annotated = [
        annotate_detectability(
            h.finding,
            h.sample,
            sesoi=sesoi,
            alpha=alpha,
            target_power=target_power,
            n_resamples=n_resamples,
            seed=seed,
        )
        for h in base_hypotheses
    ]
    
    # Step 3: Gating child hypotheses
    parent_findings = {
        f"{f.category.name}_{f.target_feature}_{f.sensitive_attribute}": f 
        for f in base_annotated
    }
    
    child_hypotheses = _run_analyzers(child_cols)
    gated_children = power_guided_gating(child_hypotheses, parent_findings, parents)
    
    child_annotated = [
        annotate_detectability(
            h.finding,
            h.sample,
            sesoi=sesoi,
            alpha=alpha,
            target_power=target_power,
            n_resamples=n_resamples,
            seed=seed,
            skip_interval=skip,
        )
        for (h, skip) in gated_children
    ]
    
    annotated = base_annotated + child_annotated

    corrected, families = correct_within_families(annotated, alpha=alpha)
    scored = [assign_severity(f) for f in corrected]

    return AuditResult(
        findings=scored,
        summary=summarise(scored),
        sesoi=sesoi,
        alpha=alpha,
        target_power=target_power,
        correction_families=families,
    )
