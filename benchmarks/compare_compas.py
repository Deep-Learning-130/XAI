"""M6 gate on full-size COMPAS: incumbent verdict vs dbias, per cell.

    python benchmarks/compare_compas.py

Criterion pre-registered in docs/superpowers/plans/2026-10-04-m6-compas-incumbent.md.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from incumbent import four_fifths_cell  # noqa: E402

from dbias.audit import audit  # noqa: E402
from dbias.correction.hierarchy import build_intersections  # noqa: E402
from dbias.ingestion.loaders import fetch_compas  # noqa: E402
from dbias.models.enums import Severity  # noqa: E402
from dbias.report.json_export import blind_spot_reading  # noqa: E402

CELLS = {
    "DISP_VR_CHARGE_DEGREE_SEX": ("vr_charge_degree", "sex"),
    "DISP_VR_CHARGE_DEGREE_RACE": ("vr_charge_degree", "race"),
    "DISP_VR_CHARGE_DEGREE_SEX_AND_RACE": ("vr_charge_degree", "sex_AND_race"),
}


def main() -> int:
    try:
        import fairlearn  # noqa: F401  -- the cross-check in tests/benchmarks needs it
    except ImportError:
        print("Fairlearn is required: pip install -e .[bench]", file=sys.stderr)
        return 2

    df = fetch_compas()
    result = audit(df, ["sex", "race"], "is_recid", sesoi=0.1, seed=0)
    findings = {f.id: f for f in result.findings}

    stale = [i for i in CELLS if i not in findings or findings[i].severity is not Severity.BLIND_SPOT]
    if stale:
        current = sorted(f.id for f in result.findings if f.severity is Severity.BLIND_SPOT)
        print(f"These cells are no longer blind spots: {stale}\nCurrent blind spots: {current}\n"
              "Update CELLS (and results.md) before comparing.", file=sys.stderr)
        return 1

    with_intersections, _ = build_intersections(df, ["sex", "race"])
    report = []
    for finding_id, (feature, attribute) in CELLS.items():
        cell = four_fifths_cell(with_intersections, feature, attribute)
        meets = cell.reports_clean  # dbias side already checked: BLIND_SPOT
        report.append({
            "finding": finding_id,
            "n": cell.n,
            "dbias": blind_spot_reading(findings[finding_id]),
            "incumbent_reports_clean": cell.reports_clean,
            "max_dp_difference": cell.max_difference,
            "failing_levels": [
                {"level": lv.level, "ratio": lv.ratio, "rates": lv.selection_rates}
                for lv in cell.levels if not lv.passes_four_fifths
            ],
            "meets_m6_gate": meets,
        })
        print(f"{finding_id:36} n={cell.n:4}  incumbent clean={cell.reports_clean!s:5}  "
              f"max DP diff={cell.max_difference:.3f}  "
              f"failing levels={[lv.level for lv in cell.levels if not lv.passes_four_fifths]}  "
              f"M6={'YES' if meets else 'no'}")

    out = Path("out") / "compas_incumbent.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nM6 headline cells: {[r['finding'] for r in report if r['meets_m6_gate']] or 'none'}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
