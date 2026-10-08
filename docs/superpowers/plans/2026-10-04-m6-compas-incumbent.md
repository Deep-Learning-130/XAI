# M6 Gate on Full-Size COMPAS — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Settle whether the COMPAS `vr_charge_degree` blind spots meet the roadmap M6 headline criterion — a cell in a *real, full-size* benchmark that an incumbent tool reports as clean and `dbias` marks as a blind spot — by running Fairlearn on those cells under a criterion fixed before the run.

**Architecture:** A small, tested helper (`benchmarks/incumbent.py`) computes what Fairlearn's demographic-parity metrics conclude about one (feature, attribute) cell under the four-fifths rule, one indicator per feature level. A script (`benchmarks/compare_compas.py`) runs the seeded `dbias` audit and the incumbent check on the same cells and prints, per cell, whether it meets the gate. The write-up then records whichever outcome occurs; both are written out in Task 3.

**Tech Stack:** Python 3.11+, pandas, Fairlearn ≥ 0.10 (the `bench` extra: `pip install -e .[bench]`), pytest.

**Spec:** [roadmap.md](../../../roadmap.md) M6 ("identify at least one (attribute, feature) cell in a *real* benchmark that AIF360 or Fairlearn reports as clean and this tool marks as a blind spot. Verify the claim by running the incumbent tool, not by assuming."), [results.md](../../../results.md) §2 (COMPAS has 3 blind spots, all on `vr_charge_degree`).

**Depends on:** run this **after** `2026-10-04-bias-corrected-cramers-v.md` if that plan is being executed: it changes effect sizes and intervals, so it can change which COMPAS cells are blind spots. Task 2 reads the blind-spot ids from a fresh run, and stops if a pinned cell is no longer a blind spot.

## Pre-registered criterion — fixed before any run

Fairlearn has no pass/fail verdict and no test for a multi-level categorical feature, so the comparison needs a stated rule. This one is fixed here, before the numbers exist, so the outcome cannot steer it:

- For each level L of the feature, build the indicator `feature == L` and compute its rate per group of the attribute — Fairlearn's `MetricFrame` selection rate, equivalently `demographic_parity_ratio(y_true=ind, y_pred=ind, sensitive_features=attr)`.
- A level **passes** if the ratio of the lowest to the highest group rate is ≥ 0.8 — the four-fifths rule, the most widely used pass/fail threshold for demographic parity.
- The incumbent **reports the cell clean** iff **every** level passes. Rows missing the feature or the attribute are dropped, as `dbias` drops them. All groups are kept, however small, because that is Fairlearn's default.
- The cell **meets the M6 gate** iff the incumbent reports it clean **and** `dbias` marks it `BLIND_SPOT`.

Secondary, reported but not used for the gate: the largest demographic-parity *difference* across levels, against the common 0.1 threshold.

**Expect either outcome.** Groups of 4 people make some level rates 0, which drives the ratio to 0 and *fails* the rule. In that case the incumbent does not call the cell clean — it raises an alarm from noise, which is a different failure from the one M6 is about — and the honest result is that full-size COMPAS does not meet the gate. Task 3 has the wording for both outcomes. Do not change the rule after seeing the numbers.

## Global Constraints

- **Python 3.11 minimum.**
- **Seeded runs only** (`seed=0`, `sesoi=0.1`), identical to `run_compas.py`.
- **Test oracles are hand-computed**, with Fairlearn as an independent cross-check, never this code's own output.
- **Commands run from the repo root on Windows, in Git Bash.**
- **Outputs go to `out/`**, which is gitignored.

## Review Focus

1. **A group with no rows at some level** (e.g. no women with charge `(F5)`). Expected: rate 0.0 and ratio 0.0 — a number, not NaN or a missing key. Pinned in Task 1.
2. **NaN in the feature or the attribute.** Expected: those rows are dropped before any rate is computed, and `n` reports the rows kept. Pinned in Task 1.
3. **Numeric group labels** (an attribute coded 0/1, as in many real datasets). Expected: rates keyed by the string labels `"0"`, `"1"`, matching how `dbias` names groups in `n_per_group`. Pinned in Task 1.
4. **The pinned blind spots changed** (e.g. after the bias-corrected V plan). Expected: the script stops with a message naming the cells, rather than comparing against stale ids. Pinned in Task 2.
5. **Fairlearn not installed.** Expected: the cross-check test is skipped, and the script fails with an instruction to run `pip install -e .[bench]`. Pinned in Tasks 1 and 2.

---

### Task 1: The incumbent verdict for one cell

**Files:**
- Create: `benchmarks/incumbent.py`
- Create: `tests/benchmarks/test_incumbent.py`

**Interfaces:**
- Produces: `FOUR_FIFTHS = 0.8`; `LevelVerdict(level: str, selection_rates: dict[str, float], ratio: float, difference: float, passes_four_fifths: bool)`; `CellVerdict(feature: str, attribute: str, n: int, levels: list[LevelVerdict])` with properties `reports_clean -> bool` and `max_difference -> float`; `four_fifths_cell(df: pd.DataFrame, feature: str, attribute: str) -> CellVerdict`.

- [ ] **Step 1: Write the failing tests**

Create `tests/benchmarks/test_incumbent.py`:

```python
"""The incumbent's verdict on one cell, against hand-computed rates."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "benchmarks"))

from incumbent import four_fifths_cell  # noqa: E402


def frame() -> pd.DataFrame:
    """Group A: 5 x, 5 y.  Group B: 2 x, 8 y.
    Level x: rates A 0.5, B 0.2 -> ratio 0.4, difference 0.3 (fails).
    Level y: rates A 0.5, B 0.8 -> ratio 0.625, difference 0.3 (fails)."""
    return pd.DataFrame({
        "g": ["A"] * 10 + ["B"] * 10,
        "f": ["x"] * 5 + ["y"] * 5 + ["x"] * 2 + ["y"] * 8,
    })


def test_rates_ratio_and_difference_match_the_hand_computation():
    cell = four_fifths_cell(frame(), "f", "g")
    x = next(level for level in cell.levels if level.level == "x")
    assert x.selection_rates == {"A": pytest.approx(0.5), "B": pytest.approx(0.2)}
    assert x.ratio == pytest.approx(0.4)
    assert x.difference == pytest.approx(0.3)
    assert x.passes_four_fifths is False
    assert cell.reports_clean is False
    assert cell.max_difference == pytest.approx(0.3)


def test_equal_rates_are_reported_clean():
    df = pd.DataFrame({"g": ["A"] * 10 + ["B"] * 10, "f": (["x"] * 3 + ["y"] * 7) * 2})
    cell = four_fifths_cell(df, "f", "g")
    assert cell.reports_clean is True
    assert all(level.ratio == pytest.approx(1.0) for level in cell.levels)


def test_a_group_with_no_rows_at_a_level_has_rate_zero():
    df = pd.DataFrame({"g": ["A", "A", "B", "B"], "f": ["x", "y", "y", "y"]})
    x = next(level for level in four_fifths_cell(df, "f", "g").levels if level.level == "x")
    assert x.selection_rates == {"A": pytest.approx(0.5), "B": 0.0}
    assert x.ratio == 0.0


def test_rows_missing_either_column_are_dropped():
    df = frame()
    df.loc[0, "f"] = np.nan
    df.loc[19, "g"] = np.nan
    assert four_fifths_cell(df, "f", "g").n == 18


def test_numeric_group_labels_are_keyed_as_strings():
    df = pd.DataFrame({"g": [0, 0, 1, 1], "f": ["x", "y", "x", "x"]})
    x = next(level for level in four_fifths_cell(df, "f", "g").levels if level.level == "x")
    assert x.selection_rates == {"0": pytest.approx(0.5), "1": pytest.approx(1.0)}


def test_matches_fairlearn():
    fairlearn = pytest.importorskip("fairlearn.metrics")
    df = frame()
    for level in four_fifths_cell(df, "f", "g").levels:
        indicator = (df["f"] == level.level).astype(int)
        expected = fairlearn.demographic_parity_ratio(
            y_true=indicator, y_pred=indicator, sensitive_features=df["g"]
        )
        assert level.ratio == pytest.approx(expected)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/benchmarks -q`
Expected: collection ERROR, `ModuleNotFoundError: No module named 'incumbent'`.

- [ ] **Step 3: Implement**

Create `benchmarks/incumbent.py`:

```python
"""What an incumbent fairness toolkit concludes about one audit cell.

Fairlearn reports demographic parity for a binary outcome and leaves the
pass/fail line to the user. The line used here is the four-fifths rule: the
lowest group rate must be at least 80% of the highest. A multi-level feature
becomes one indicator per level, and the cell is "clean" only if every level
passes (docs/superpowers/plans/2026-10-04-m6-compas-incumbent.md, pre-registered).
"""
from dataclasses import dataclass

import pandas as pd

FOUR_FIFTHS = 0.8


@dataclass(frozen=True)
class LevelVerdict:
    level: str
    selection_rates: dict[str, float]  # P(feature == level | group)
    ratio: float  # lowest / highest rate -- Fairlearn's demographic_parity_ratio
    difference: float  # highest - lowest -- demographic_parity_difference
    passes_four_fifths: bool


@dataclass(frozen=True)
class CellVerdict:
    feature: str
    attribute: str
    n: int
    levels: list[LevelVerdict]

    @property
    def reports_clean(self) -> bool:
        return all(level.passes_four_fifths for level in self.levels)

    @property
    def max_difference(self) -> float:
        return max(level.difference for level in self.levels)


def four_fifths_cell(df: pd.DataFrame, feature: str, attribute: str) -> CellVerdict:
    rows = df[[feature, attribute]].dropna()
    groups = rows[attribute].astype(str)
    levels = []
    for level in sorted(rows[feature].unique(), key=str):
        rates = (rows[feature] == level).groupby(groups).mean()
        high, low = float(rates.max()), float(rates.min())
        ratio = 1.0 if high == 0 else low / high
        levels.append(
            LevelVerdict(
                level=str(level),
                selection_rates={str(g): float(r) for g, r in rates.items()},
                ratio=ratio,
                difference=high - low,
                passes_four_fifths=ratio >= FOUR_FIFTHS,
            )
        )
    return CellVerdict(feature=feature, attribute=attribute, n=len(rows), levels=levels)
```

- [ ] **Step 4: Run them to verify they pass**

Run: `python -m pytest tests/benchmarks -q`
Expected: 6 pass (`test_matches_fairlearn` skipped if Fairlearn is absent; install it with `pip install -e .[bench]` so it runs).

- [ ] **Step 5: Commit**

```bash
git add benchmarks/incumbent.py tests/benchmarks/test_incumbent.py
git commit -m "feat(benchmarks): four-fifths incumbent verdict for one audit cell"
```

---

### Task 2: Compare on COMPAS

**Files:**
- Create: `benchmarks/compare_compas.py`

**Interfaces:**
- Consumes: `four_fifths_cell`, `CellVerdict` (Task 1); `dbias.audit.audit`, `dbias.correction.hierarchy.build_intersections`, `dbias.ingestion.loaders.fetch_compas`, `dbias.report.json_export.blind_spot_reading`.
- Produces: `out/compas_incumbent.json` and a printed table.

- [ ] **Step 1: Write the script**

Create `benchmarks/compare_compas.py`:

```python
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
```

- [ ] **Step 2: Run it**

Run: `python benchmarks/compare_compas.py` (about 40 seconds; the dataset is cached in `datasets/`).
Expected: exit 0, three rows, a final `M6 headline cells:` line. Exit 1 means the blind-spot set changed: update `CELLS` from the printed list and `results.md` §2, then re-run. Do not change the criterion.

- [ ] **Step 3: Commit**

```bash
git add benchmarks/compare_compas.py
git commit -m "feat(benchmarks): M6 incumbent comparison on full-size COMPAS"
```

---

### Task 3: Record the outcome

**Files:**
- Modify: `results.md` §2 (the paragraph beginning "These are candidates for the M6 headline cell")
- Modify: `roadmap.md` (the M6 **Result:** line)

- [ ] **Step 1: Write the outcome into `results.md`**

Replace the paragraph beginning **"These are candidates for the M6 headline cell"** with exactly one of the following, using the numbers from `out/compas_incumbent.json`.

If at least one cell has `meets_m6_gate: true`:

```markdown
**M6 headline cell found in a full-size benchmark.** Under a four-fifths criterion fixed before the run (every level of the feature must keep each group's rate within 80% of the highest), Fairlearn's demographic-parity metrics report <finding id> as clean (largest difference <max_dp_difference>), while `dbias` marks it a blind spot: <dbias reading>. See `benchmarks/compare_compas.py`.
```

If none does:

```markdown
**The COMPAS cells do not meet the M6 gate.** Under a four-fifths criterion fixed before the run, Fairlearn's demographic-parity metrics flag all three `vr_charge_degree` cells as *unfair* — for example level <first failing level of the sex cell> has rates <its rates> (ratio <its ratio>) — driven by groups as small as 4 people. The incumbent does not certify these cells as clean; it raises an alarm from noise, which is a different failure from the one M6 targets. The M6 headline therefore rests on the subsampled Adult cell (`MAR_WORKCLASS_RACE`). See `benchmarks/compare_compas.py`.
```

Replace every `<...>` with the value from the JSON.

- [ ] **Step 2: Update the M6 result line in `roadmap.md`**

Replace the sentence `The COMPAS cells are candidates for the headline cell in a full-size benchmark, but still need an incumbent-tool comparison.` with either `Full-size COMPAS also meets the gate: <finding id> is reported clean by Fairlearn under the four-fifths rule and is a dbias blind spot.` or `Full-size COMPAS does not meet the gate: Fairlearn flags its blind-spot cells as unfair rather than clean (small-group noise).` — matching Step 1.

- [ ] **Step 3: Commit**

```bash
git add results.md roadmap.md
git commit -m "docs: M6 incumbent comparison on full-size COMPAS"
```
