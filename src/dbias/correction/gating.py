"""Multiple-testing correction across the attribute -> intersection tree.

Two procedures:

* ``correct_within_families`` -- Benjamini-Hochberg within each (category,
  attribute) family, intersections included as attributes in their own right.
  The audit's default.
* ``hierarchical_fdr`` -- the same rule for parent attributes; an intersection
  is tested only if one of its parents was rejected for the same (category,
  target feature), and BH runs over the admitted intersections of each
  family. Yekutieli (2008) proves FDR control for a tree whose levels are
  independent; here an intersection's table refines its parents', so the
  levels are positively dependent. Measured empirically
  (tests/calibration/run_fdr_gate.py), it FAILED its gate: under the null the
  intersection family reached FDR 0.082 against a 0.074 limit. Experimental.

``power_guided_gating`` is unrelated to either: it only decides which
underpowered intersections may skip their bootstrap.
"""
from typing import Sequence, Any
from dataclasses import replace

from dbias.models.enums import Category, Detectability
from dbias.models.finding import Finding
from dbias.stats.correction import benjamini_hochberg

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


Families = dict[tuple[Category, str], int]


def _apply_bh(out: list[Finding], indices: list[int], alpha: float) -> None:
    adjusted = benjamini_hochberg([out[i].p_value_raw for i in indices])
    for i, p in zip(indices, adjusted):
        out[i] = replace(out[i], p_value_corrected=p, is_significant=bool(p < alpha))


def _families(findings: list[Finding]) -> dict[tuple[Category, str], list[int]]:
    groups: dict[tuple[Category, str], list[int]] = {}
    for index, finding in enumerate(findings):
        groups.setdefault((finding.category, finding.sensitive_attribute), []).append(index)
    return groups


def correct_within_families(
    findings: list[Finding], alpha: float
) -> tuple[list[Finding], Families]:
    """Apply BH within each (category, attribute) family, preserving order."""
    out = list(findings)
    groups = _families(findings)
    for indices in groups.values():
        _apply_bh(out, indices, alpha)
    return out, {key: len(indices) for key, indices in groups.items()}


def hierarchical_fdr(
    findings: list[Finding], parents: dict[str, list[str]], alpha: float
) -> tuple[list[Finding], Families]:
    """Parents as in ``correct_within_families``; intersections only below a
    rejected parent. Family sizes count only what was actually tested."""
    out = list(findings)
    sizes: Families = {}
    groups = _families(findings)

    for key, indices in groups.items():
        if key[1] not in parents:
            _apply_bh(out, indices, alpha)
            sizes[key] = len(indices)

    rejected = {
        (f.category, f.target_feature, f.sensitive_attribute)
        for f in out
        if f.sensitive_attribute not in parents and f.is_significant
    }
    for key, indices in groups.items():
        if key[1] not in parents:
            continue
        admitted = []
        for i in indices:
            f = out[i]
            if any((f.category, f.target_feature, p) in rejected for p in parents[key[1]]):
                admitted.append(i)
            else:
                out[i] = replace(f, p_value_corrected=1.0, is_significant=False, gated_by_parent=True)
        if admitted:
            _apply_bh(out, admitted, alpha)
            sizes[key] = len(admitted)
    return out, sizes
