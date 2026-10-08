"""Command line entry point.

    dbias audit data.csv --sensitive gender --sesoi 0.1 --target income
    dbias demo

`--sensitive` and `--sesoi` are both required and neither has a default. The
tool suggests nothing about which attributes are sensitive and assumes nothing
about what size of effect matters -- those are the two judgements that belong
to the person running the audit, and defaulting either of them would put the
tool's opinion into a document that carries the user's name.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from dbias.audit import audit
from dbias.detectability.coverage import coverage_map
from dbias.models.enums import Severity
from dbias.report.coverage_plot import plot_coverage_map
from dbias.report.json_export import blind_spot_reason, write_json
from dbias.synthetic import make_audit_scenario

DEMO_SEED = 1
"""The scenario seed used by `dbias demo`.

At n=900 the injected ethnicity disparity is detected about two runs in three.
This seed is one of the runs where it is missed -- which is the case worth
demonstrating, since it is the one every other tool reports as clean. The
scenario docstring says so, and `tests/calibration/` measures the rate.
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dbias",
        description="Audit a dataset for disparities, and for what it could not have seen.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("audit", help="audit a CSV file")
    run.add_argument("path", help="CSV file to audit")
    run.add_argument(
        "--sensitive", action="append", required=True, metavar="COLUMN",
        help="a column to treat as a sensitive attribute; repeat for several",
    )
    run.add_argument("--target", default=None, help="outcome label column")
    run.add_argument(
        "--sesoi", type=float, required=True,
        help="smallest effect size of interest, on the Cohen's w scale. "
             "Required: every verdict is conditional on it.",
    )
    run.add_argument("--alpha", type=float, default=0.05)
    run.add_argument("--power", type=float, default=0.80, dest="target_power")
    run.add_argument("--resamples", type=int, default=2000)
    run.add_argument("--seed", type=int, default=None)
    run.add_argument(
        "--fdr", choices=["family", "hierarchical"], default="family",
        help="multiple-testing procedure. 'hierarchical' tests an intersection "
             "only below a parent with a disparity, and so cannot find "
             "intersection-only effects.",
    )
    run.add_argument("--out", default="out", help="directory for the report and figure")

    demo = sub.add_parser(
        "demo", help="generate the synthetic scenario and audit it end to end"
    )
    demo.add_argument("--out", default="out/demo")
    demo.add_argument("--n", type=int, default=900)
    demo.add_argument("--seed", type=int, default=DEMO_SEED)
    demo.add_argument("--sesoi", type=float, default=0.1)
    demo.add_argument("--resamples", type=int, default=2000)
    return parser


def _emit(result, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = write_json(result, out_dir / "audit.json")
    grid = coverage_map(result.findings)
    # No testable cells: there is no map to draw, and that is reported below
    # rather than crashing after the JSON is already written.
    png_path = (
        plot_coverage_map(result, out_dir / "coverage_map.png")
        if grid.attributes and grid.features
        else None
    )

    blind = [f for f in result.findings if f.severity is Severity.BLIND_SPOT]
    print(f"\n  {len(result.findings)} findings, SESOI = {result.sesoi:g} (Cohen's w)")
    for category, severity in result.summary["risk"].items():
        coverage = result.summary["coverage"][category]
        print(f"    {category:<18} risk={severity:<13} coverage={coverage:.0%}")

    if blind:
        print(f"\n  {len(blind)} blind spot(s) - null results that rule nothing out:")
        for finding in blind:
            print(f"    {finding.id:<34} {blind_spot_reason(finding)}")
    else:
        print("\n  No blind spots: every null result here is an earned all-clear.")

    print(f"\n  wrote {json_path}")
    if png_path is None:
        print("  no coverage map: the audit produced no testable cells\n")
    else:
        print(f"  wrote {png_path}\n")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    out_dir = Path(args.out)

    if args.command == "demo":
        frame, truth = make_audit_scenario(n=args.n, seed=args.seed)
        out_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out_dir / "scenario.csv", index=False)
        (out_dir / "ground_truth.json").write_text(
            json.dumps(truth, indent=2), encoding="utf-8"
        )
        print(
            f"\n  Synthetic scenario: n={truth['n']}, seed={truth['seed']}\n"
            f"  True missingness effect (Cohen's w) by attribute:"
        )
        for attribute, effect in truth["true_effects"].items():
            note = "a real disparity" if effect > 0 else "a true null"
            print(f"    {attribute:<12} w = {effect:.4f}   ({note})")

        result = audit(
            frame,
            sensitive_cols=truth["sensitive_cols"],
            target_col=truth["target_col"],
            sesoi=args.sesoi,
            n_resamples=args.resamples,
            seed=args.seed,
        )
        _emit(result, out_dir)
        return 0

    try:
        frame = pd.read_csv(args.path)
    except (OSError, ValueError) as error:
        # ValueError covers pandas' ParserError / EmptyDataError and
        # UnicodeDecodeError.
        print(f"dbias: cannot read {args.path}: {error}", file=sys.stderr)
        return 2

    try:
        result = audit(
            frame,
            sensitive_cols=args.sensitive,
            target_col=args.target,
            sesoi=args.sesoi,
            alpha=args.alpha,
            target_power=args.target_power,
            n_resamples=args.resamples,
            seed=args.seed,
            fdr=args.fdr,
        )
    except ValueError as error:
        print(f"dbias: {error}", file=sys.stderr)
        return 2

    _emit(result, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
