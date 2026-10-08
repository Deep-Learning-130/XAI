# Hierarchical FDR for Intersections — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in hierarchical FDR procedure in which an intersection (e.g. `race_AND_sex`) is tested only below a parent attribute that showed a disparity, and verify empirically — for both the existing per-family rule and the new one — what false discovery rate the audit actually delivers. This closes roadmap M7's two outstanding criteria.

**Architecture:** The FDR step moves out of `audit.py` into `correction/gating.py`, where `docs/10` places it. It offers two procedures. `family` (the default, today's behaviour) runs Benjamini–Hochberg within each (category, attribute) family. `hierarchical` keeps that rule for parent attributes, then admits an intersection hypothesis only if one of its parents was rejected for the same (category, target feature), and runs BH over the admitted intersections within their own family. Rejected-from-testing intersections are marked `gated_by_parent`, are never significant, and fall through to the interval-based null-result rule — so they become blind spots, never false all-clears, unless their interval earns equivalence. A simulation harness measures realised FDR and intersection power under four ground-truth scenarios.

**Tech Stack:** Python 3.11+, NumPy, pandas, pytest.

**Spec:** [roadmap.md](../../../roadmap.md) M7 ("Hierarchical FDR (Yekutieli-style) over the tree … its FDR is verified empirically in the M3 harness"), [plan.md](../../../plan.md) §1 line 34 and the decision table ("Gated hierarchical FDR over intersections — Build, but reduced"), [docs/11_prototype_report.md](../../11_prototype_report.md) §9.5 ("hierarchical descent must nest inside" the per-(category, attribute) families), [docs/10_repository_structure.md](../../10_repository_structure.md) §5 (FDR family boundary). Method: Yekutieli, D. (2008), "Hierarchical false discovery rate–controlling methodology", *JASA* 103(481).

## Decision recorded in this plan — read before executing

Hierarchical testing **hides intersection-only effects.** If missingness depends on race × sex in an XOR pattern, neither race nor sex shows a marginal disparity, so the intersection is never tested. That is the price of the FDR guarantee, and it is why this plan ships `hierarchical` as **opt-in** with the default unchanged (`family`). Task 4 measures the trade-off on both procedures. Whether to change the default is a decision for the project owner after reading Task 4's numbers — not for the implementer. A gated intersection with a real effect is still reported as a blind spot, not as clean (Task 2 pins this).

Yekutieli's theorem is for a tree; here an intersection has two parents (a DAG). Admitting a child when **either** parent is rejected is the natural extension, but it is not covered by the theorem — which is exactly why Task 4's empirical check is the gate, not an afterthought.

**What is and is not controlled.** The audit's FDR unit is the (category, attribute) family (`docs/11` §9.5: hierarchical descent "must nest inside them"). Neither procedure controls FDR *across* the audit: under a global null with two parent families at α = 0.05, the chance of at least one false discovery somewhere is already about 1 − 0.95² ≈ 10%. So the gate is **per-family** FDR. Audit-wide FDR is measured and reported so the write-up can state it, but it is not a pass/fail criterion. What hierarchical testing buys is that adding intersections adds almost nothing to audit-wide false discoveries under the null.

## Global Constraints

- **Python 3.11 minimum.**
- **The default must not change behaviour.** With `fdr="family"` every finding, p-value and severity must be identical to before this plan. Task 2 asserts this.
- **alpha = 0.05 is a parameter with that default.** Never inline it.
- **Parents come from the intersection tree** (`hierarchy.parents_of`), never from splitting a column name on `"_AND_"`.
- **Every finding carries a corrected p-value.** A gated intersection gets `p_value_corrected = 1.0` (it was not tested) and `gated_by_parent = True`.
- **Commands run from the repo root on Windows, in Git Bash** (`python -m pytest ...`).

## Review Focus

1. **A single sensitive column** (no intersections). Expected: `hierarchical` gives exactly the `family` result. Pinned in Task 1.
2. **A user column whose name already contains `_AND_`.** Expected: its intersections find their parents. Pinned in Task 1.
3. **Representation findings** (`target_feature is None`). Expected: a child representation test is admitted below a rejected parent representation test. Pinned in Task 1.
4. **A parent attribute that produced no finding** (e.g. constant, so never tested). Expected: its children are gated by that parent; the other parent can still admit them. Pinned in Task 1.
5. **An invalid `--fdr` value on the command line.** Expected: argparse error, exit 2, no traceback. Pinned in Task 3.

---

### Task 1: The hierarchical procedure

**Files:**
- Modify: `src/dbias/models/finding.py` (one new field)
- Modify: `src/dbias/correction/gating.py` (module docstring; add `correct_within_families`, `hierarchical_fdr`)
- Modify: `src/dbias/audit.py:159-175` (delete `_correct_within_families`, import the moved one)
- Test: `tests/unit/test_hierarchical_fdr.py` (create)

**Interfaces:**
- Consumes: `benjamini_hochberg(p_values) -> list[float]` from `dbias.stats.correction`.
- Produces, in `dbias.correction.gating`:
  - `correct_within_families(findings: list[Finding], alpha: float) -> tuple[list[Finding], dict[tuple[Category, str], int]]` — today's `audit._correct_within_families`, moved verbatim.
  - `hierarchical_fdr(findings: list[Finding], parents: dict[str, list[str]], alpha: float) -> tuple[list[Finding], dict[tuple[Category, str], int]]`.
  - `Finding.gated_by_parent: bool = False`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_hierarchical_fdr.py`:

```python
"""Two-level FDR: parents as before, intersections only below a rejected parent."""
import pytest

from dbias.correction.gating import correct_within_families, hierarchical_fdr
from dbias.models.enums import Category, EffectSizeMetric
from dbias.models.finding import Finding

PARENTS = {"race_AND_sex": ["race", "sex"]}


def f(attribute, p, category=Category.MISSINGNESS, feature="x") -> Finding:
    return Finding(
        id=f"{category}_{feature}_{attribute}", category=category,
        sensitive_attribute=attribute, target_feature=feature, metric_name="",
        observed_values={}, statistical_test="", p_value_raw=p,
        effect_size_metric=EffectSizeMetric.CRAMERS_V, effect_size_value=0.0,
        n_per_group={},
    )


def by_attr(findings):
    return {x.sensitive_attribute: x for x in findings}


def test_parents_are_corrected_exactly_as_in_family_mode():
    findings = [f("race", 0.001), f("sex", 0.4), f("race_AND_sex", 0.002)]
    hier, _ = hierarchical_fdr(findings, PARENTS, alpha=0.05)
    flat, _ = correct_within_families(findings, alpha=0.05)
    for attr in ("race", "sex"):
        assert by_attr(hier)[attr] == by_attr(flat)[attr]


def test_a_child_below_a_rejected_parent_is_tested():
    out, _ = hierarchical_fdr([f("race", 0.001), f("sex", 0.4), f("race_AND_sex", 0.002)], PARENTS, 0.05)
    child = by_attr(out)["race_AND_sex"]
    assert child.is_significant is True
    assert child.gated_by_parent is False
    assert child.p_value_corrected == pytest.approx(0.002)


def test_a_child_with_no_rejected_parent_is_not_tested():
    """The XOR case: no marginal disparity, so the intersection is never
    tested -- however small its own p-value."""
    out, families = hierarchical_fdr([f("race", 0.4), f("sex", 0.5), f("race_AND_sex", 1e-9)], PARENTS, 0.05)
    child = by_attr(out)["race_AND_sex"]
    assert child.is_significant is False
    assert child.gated_by_parent is True
    assert child.p_value_corrected == 1.0
    assert (Category.MISSINGNESS, "race_AND_sex") not in families


def test_admission_is_per_category_and_feature():
    """race rejected for feature x does not admit race_AND_sex for feature y."""
    findings = [f("race", 0.001, feature="x"), f("race", 0.6, feature="y"),
                f("sex", 0.6, feature="x"), f("sex", 0.6, feature="y"),
                f("race_AND_sex", 0.001, feature="x"), f("race_AND_sex", 0.001, feature="y")]
    out, families = hierarchical_fdr(findings, PARENTS, 0.05)
    kids = {x.target_feature: x for x in out if x.sensitive_attribute == "race_AND_sex"}
    assert kids["x"].gated_by_parent is False and kids["x"].is_significant is True
    assert kids["y"].gated_by_parent is True and kids["y"].is_significant is False
    assert families[(Category.MISSINGNESS, "race_AND_sex")] == 1


def test_bh_among_children_counts_only_admitted_children():
    """One intersection family across three features. race is rejected for x
    and y only, so z is gated. BH over the m = 2 admitted children keeps both
    (0.045 * 2/2 = 0.045 < 0.05). Counting gated z (p = 0.9) as m = 3 would
    drop both: 0.045 * 3/2 = 0.0675."""
    findings = [f("race", 0.001, feature="x"), f("race", 0.001, feature="y"), f("race", 0.9, feature="z"),
                f("sex", 0.9, feature="x"), f("sex", 0.9, feature="y"), f("sex", 0.9, feature="z"),
                f("race_AND_sex", 0.03, feature="x"), f("race_AND_sex", 0.045, feature="y"),
                f("race_AND_sex", 0.9, feature="z")]
    out, families = hierarchical_fdr(findings, PARENTS, 0.05)
    kids = {x.target_feature: x for x in out if x.sensitive_attribute == "race_AND_sex"}
    assert kids["x"].is_significant and kids["y"].is_significant
    assert kids["z"].gated_by_parent
    assert families[(Category.MISSINGNESS, "race_AND_sex")] == 2


def test_without_intersections_hierarchical_equals_family():
    findings = [f("race", 0.01), f("race", 0.03, feature="y"), f("sex", 0.2)]
    assert hierarchical_fdr(findings, {}, 0.05) == correct_within_families(findings, 0.05)


def test_a_parent_name_containing_the_separator_still_admits():
    parents = {"RACE_AND_ETHNICITY_AND_sex": ["RACE_AND_ETHNICITY", "sex"]}
    out, _ = hierarchical_fdr(
        [f("RACE_AND_ETHNICITY", 0.001), f("sex", 0.9), f("RACE_AND_ETHNICITY_AND_sex", 0.001)],
        parents, 0.05,
    )
    assert by_attr(out)["RACE_AND_ETHNICITY_AND_sex"].gated_by_parent is False


def test_representation_children_are_admitted_by_parent_representation():
    out, _ = hierarchical_fdr(
        [f("race", 0.001, Category.REPRESENTATION, None), f("sex", 0.9, Category.REPRESENTATION, None),
         f("race_AND_sex", 0.001, Category.REPRESENTATION, None)],
        PARENTS, 0.05,
    )
    assert by_attr(out)["race_AND_sex"].is_significant is True


def test_a_missing_parent_finding_does_not_admit():
    """sex was constant, so it has no finding; race alone decides."""
    out, _ = hierarchical_fdr([f("race", 0.7), f("race_AND_sex", 0.001)], PARENTS, 0.05)
    assert by_attr(out)["race_AND_sex"].gated_by_parent is True
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/unit/test_hierarchical_fdr.py -q`
Expected: collection ERROR, `ImportError: cannot import name 'correct_within_families'`.

- [ ] **Step 3: Implement**

In `src/dbias/models/finding.py`, directly under `is_significant: bool = False`, add:

```python
    # Hierarchical FDR only: True when no parent attribute was rejected, so
    # this intersection was never tested for significance.
    gated_by_parent: bool = False
```

In `src/dbias/correction/gating.py`, replace the module docstring with:

```python
"""Multiple-testing correction across the attribute -> intersection tree.

Two procedures:

* ``correct_within_families`` -- Benjamini-Hochberg within each (category,
  attribute) family, intersections included as attributes in their own right.
  The audit's default.
* ``hierarchical_fdr`` -- the same rule for parent attributes; an intersection
  is tested only if one of its parents was rejected for the same (category,
  target feature), and BH runs over the admitted intersections of each
  family. Yekutieli (2008) proves FDR control for a tree; an intersection has
  two parents, so its control here is verified empirically
  (tests/calibration/run_fdr_gate.py), not assumed.

``power_guided_gating`` is unrelated to either: it only decides which
underpowered intersections may skip their bootstrap.
"""
```

Add `from dataclasses import replace` and `from dbias.stats.correction import benjamini_hochberg` to the imports, and extend the existing `from dbias.models.enums import Detectability` to `from dbias.models.enums import Category, Detectability`. Then append:

```python
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
```

In `src/dbias/audit.py`: delete `_correct_within_families` (lines 159–175); remove the `benjamini_hochberg` import (line 42) and drop `replace` from line 24 if a search shows nothing else in the file uses it; change `from dbias.correction.gating import power_guided_gating` to `from dbias.correction.gating import correct_within_families, power_guided_gating`; change the call to `corrected, families = correct_within_families(annotated, alpha=alpha)`.

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/unit/test_hierarchical_fdr.py tests/unit/test_audit.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/dbias/models/finding.py src/dbias/correction/gating.py src/dbias/audit.py tests/unit/test_hierarchical_fdr.py
git commit -m "feat(correction): hierarchical FDR over the intersection tree"
```

---

### Task 2: `audit(..., fdr=...)`

**Files:**
- Modify: `src/dbias/audit.py` (signature, `AuditResult`, module docstring)
- Modify: `src/dbias/report/json_export.py:88` (`"correction"` text)
- Test: `tests/unit/test_audit.py`

**Interfaces:**
- Consumes: `correct_within_families`, `hierarchical_fdr` (Task 1); `parents` already computed in `audit()`.
- Produces: `audit(..., fdr: Literal["family", "hierarchical"] = "family")`; `AuditResult.fdr: str = "family"`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_audit.py`:

```python
# --- hierarchical FDR ----------------------------------------------------------

def _xor_frame(n_per_cell: int = 1000) -> pd.DataFrame:
    """Missingness of x depends on race x sex in an XOR pattern: exactly 10%
    missing in (a, F) and (b, M), exactly 40% otherwise. Built deterministically,
    so each marginal rate is exactly 25%: both parent tests have chi2 = 0 and
    p = 1, and only the intersection is real. (Random draws would let a parent
    reach significance by chance and make the hierarchical tests flaky.)"""
    cells = []
    for race in ("a", "b"):
        for sex in ("F", "M"):
            rate = 0.10 if (race == "a") == (sex == "F") else 0.40
            k = int(round(rate * n_per_cell))
            x = np.r_[np.full(k, np.nan), np.ones(n_per_cell - k)]
            cells.append(pd.DataFrame({"race": race, "sex": sex, "x": x}))
    return pd.concat(cells, ignore_index=True)


def _mar(result, attribute):
    return next(f for f in result.findings if f.id == f"MAR_X_{attribute}")


def test_family_mode_is_the_default_and_unchanged():
    df = _xor_frame()
    kwargs = dict(sesoi=0.1, seed=1, n_resamples=100)
    default = audit(df, ["race", "sex"], **kwargs)
    explicit = audit(df, ["race", "sex"], fdr="family", **kwargs)
    assert default.findings == explicit.findings
    assert default.fdr == "family"


def test_family_mode_finds_an_intersection_only_effect():
    result = audit(_xor_frame(), ["race", "sex"], sesoi=0.1, seed=1, n_resamples=100)
    assert _mar(result, "RACE_AND_SEX").is_significant is True


def test_hierarchical_mode_gates_it_but_never_calls_it_clean():
    """The documented cost: the intersection-only effect is not tested. The
    safety property: it is reported as a blind spot, not an all-clear."""
    result = audit(_xor_frame(), ["race", "sex"], sesoi=0.1, seed=1, n_resamples=100, fdr="hierarchical")
    child = _mar(result, "RACE_AND_SEX")
    assert child.gated_by_parent is True
    assert child.is_significant is False
    assert child.severity is Severity.BLIND_SPOT
    assert result.fdr == "hierarchical"


def test_an_unknown_fdr_mode_is_rejected():
    with pytest.raises(ValueError, match="fdr"):
        audit(_xor_frame(n_per_cell=50), ["race"], sesoi=0.1, fdr="global")
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/unit/test_audit.py -q -k "fdr or family or hierarchical"`
Expected: FAIL — `audit() got an unexpected keyword argument 'fdr'` / `AuditResult has no attribute 'fdr'`.

- [ ] **Step 3: Implement**

In `src/dbias/audit.py`:

- Add `from typing import Literal` and `FdrMode = Literal["family", "hierarchical"]` below the imports; import `hierarchical_fdr` alongside `correct_within_families`.
- Add to `AuditResult` after `correction_families`: `fdr: str = "family"`.
- Add `fdr: FdrMode = "family",` as the last keyword parameter of `audit()`, and as the first lines of its body:

```python
    if fdr not in ("family", "hierarchical"):
        raise ValueError(f"fdr must be 'family' or 'hierarchical', got {fdr!r}")
```

- Replace the correction call with:

```python
    if fdr == "hierarchical":
        corrected, families = hierarchical_fdr(annotated, parents, alpha=alpha)
    else:
        corrected, families = correct_within_families(annotated, alpha=alpha)
```

- Pass `fdr=fdr,` into the `AuditResult(...)` constructor.
- In the module docstring, replace the sentence beginning `Hierarchical FDR -- conditioning a child's rejection on its parent's (Yekutieli) -- is not implemented;` up to the end of that sentence with: `fdr="hierarchical" adds that conditioning (correction/gating.py): an intersection is tested only below a rejected parent. It is opt-in because it hides intersection-only effects.`

In `src/dbias/report/json_export.py`, replace the `"correction"` line with:

```python
            "correction": (
                "Benjamini-Hochberg within each (category, attribute) family"
                if result.fdr == "family"
                else "Hierarchical: Benjamini-Hochberg within each parent "
                "(category, attribute) family; intersections tested only "
                "below a rejected parent"
            ),
```

- [ ] **Step 4: Run the full fast suite**

Run: `python -m pytest -m "not slow" -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/dbias/audit.py src/dbias/report/json_export.py tests/unit/test_audit.py
git commit -m "feat(audit): opt-in hierarchical FDR (fdr='hierarchical')"
```

---

### Task 3: Surface it — CLI flag and blind-spot wording

**Files:**
- Modify: `src/dbias/cli.py` (`_build_parser`, the `audit(...)` call in `main`)
- Modify: `src/dbias/report/json_export.py` (`blind_spot_reason`)
- Test: `tests/unit/test_cli.py`, `tests/unit/test_report.py`

**Interfaces:**
- Consumes: `audit(..., fdr=...)`, `Finding.gated_by_parent`.
- Produces: CLI option `--fdr {family,hierarchical}`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_cli.py`:

```python
def test_hierarchical_fdr_is_selectable_and_recorded(csv_path, tmp_path):
    out = tmp_path / "out"
    code = main([
        "audit", str(csv_path), "--sensitive", "gender", "--sensitive", "ethnicity",
        "--target", "hired", "--sesoi", "0.1", "--fdr", "hierarchical",
        "--out", str(out), "--resamples", "50", "--seed", "7",
    ])
    assert code == 0
    report = json.loads((out / "audit.json").read_text())
    assert report["configuration"]["correction"].startswith("Hierarchical")


def test_an_unknown_fdr_flag_is_an_argument_error(csv_path):
    with pytest.raises(SystemExit) as exit_info:
        main(["audit", str(csv_path), "--sensitive", "gender", "--sesoi", "0.1", "--fdr", "global"])
    assert exit_info.value.code == 2
```

Append to `tests/unit/test_report.py`:

```python
def test_a_gated_blind_spot_says_it_was_never_tested():
    from dbias.models.enums import Category, Detectability, EffectSizeMetric
    from dbias.models.finding import Finding
    from dbias.report.json_export import blind_spot_reading

    gated = Finding(
        id="T", category=Category.MISSINGNESS, sensitive_attribute="a_AND_b",
        target_feature="x", metric_name="", observed_values={}, statistical_test="",
        p_value_raw=1e-6, effect_size_metric=EffectSizeMetric.CRAMERS_V,
        effect_size_value=0.3, n_per_group={}, sesoi=0.1, df_min=1,
        minimum_detectable_effect=0.05, detectability=Detectability.ADEQUATE,
        effect_size_ci=(0.2, 0.4), gated_by_parent=True,
    )
    assert "neither parent attribute" in blind_spot_reading(gated)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/unit/test_cli.py tests/unit/test_report.py -q`
Expected: 3 FAIL (unrecognised `--fdr`; wording missing).

- [ ] **Step 3: Implement**

In `src/dbias/cli.py` `_build_parser`, after the `--seed` argument of the `audit` subcommand:

```python
    run.add_argument(
        "--fdr", choices=["family", "hierarchical"], default="family",
        help="multiple-testing procedure. 'hierarchical' tests an intersection "
             "only below a parent with a disparity, and so cannot find "
             "intersection-only effects.",
    )
```

and pass `fdr=args.fdr,` in the `audit(...)` call inside `main` (the CSV path, not the demo).

In `src/dbias/report/json_export.py`, make this the first branch of `blind_spot_reason`, right after the `sesoi = ...` line:

```python
    if finding.gated_by_parent:
        return (
            f"it was never tested for significance, because neither parent "
            f"attribute showed a disparity (hierarchical FDR), and its interval "
            f"did not rule out {sesoi}"
        )
```

and change the body of `blind_spot_reading` so a gated finding does not open with "No disparity was detected" (nothing was tested):

```python
    if finding.gated_by_parent:
        return f"Not tested: {blind_spot_reason(finding)}. This is not a clean result."
    return f"No disparity was detected, but {blind_spot_reason(finding)}. This is not a clean result."
```

Also extend the new test in `tests/unit/test_report.py` with `assert blind_spot_reading(gated).startswith("Not tested")`.

- [ ] **Step 4: Run the full fast suite**

Run: `python -m pytest -m "not slow" -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/dbias/cli.py src/dbias/report/json_export.py tests/unit/test_cli.py tests/unit/test_report.py
git commit -m "feat(cli): --fdr flag; gated blind spots say they were never tested"
```

---

### Task 4: Empirical FDR gate

The criteria are fixed **before** the run.

**Gate criterion (pre-registered):** for **both** procedures, in every scenario, the realised FDR of **each** missingness family (`race`, `sex`, `race_AND_sex`) is **≤ alpha + 2·SE**, where SE is the standard error of that family's per-audit false discovery proportion.

**Reported, not gated:** audit-wide FDR (all missingness findings pooled) for both procedures — expected to exceed alpha under the null for both, see "What is and is not controlled" — and the power to detect real intersection and real parent effects.

**Files:**
- Create: `tests/calibration/fdr_harness.py`
- Create: `tests/calibration/run_fdr_gate.py`
- Create: `tests/calibration/test_fdr_calibration.py`

**Interfaces:**
- Consumes: `audit(..., fdr=...)`.
- Produces: `SCENARIOS: dict[str, dict[str, np.ndarray]]`, `FAMILIES = ("race", "sex", "race_AND_sex")`, `truth(rates) -> dict[tuple[str, str], bool]`, `run_fdr_cell(scenario: str, n: int, replicates: int, fdr: str, alpha: float = 0.05, seed: int = 0) -> dict` (keys `fdr_<family>`, `fdr_<family>_se`, `fdr_audit_wide`, `fdr_audit_wide_se`, `intersection_power`, `parent_power`), `passes_gate(row: dict, alpha: float = 0.05) -> bool`.

- [ ] **Step 1: Write the harness**

Create `tests/calibration/fdr_harness.py`:

```python
"""Realised FDR of the audit's missingness findings, with known ground truth.

Two binary sensitive attributes, race in {a, b} and sex in {F, M}, each
50/50 and independent. Every feature x_j goes missing with a probability
set per (race, sex) cell by a 2x2 rate table. Ground truth follows from the
table: race is a real effect for x_j iff its marginal rates differ, sex
likewise, and race_AND_sex iff the four cell rates are not all equal.
"""
import numpy as np
import pandas as pd

from dbias.audit import audit

NULL = np.full((2, 2), 0.2)
RACE = np.array([[0.1, 0.1], [0.3, 0.3]])  # rows: race a/b; cols: sex F/M
XOR = np.array([[0.1, 0.3], [0.3, 0.1]])


def _features(*kinds: np.ndarray) -> dict[str, np.ndarray]:
    return {f"x{i}": rates for i, rates in enumerate(kinds)}


SCENARIOS: dict[str, dict[str, np.ndarray]] = {
    "global_null": _features(*[NULL] * 10),
    "main_effect": _features(*[RACE] * 5, *[NULL] * 5),
    "xor": _features(*[XOR] * 5, *[NULL] * 5),
    "mixed": _features(*[RACE] * 3, *[XOR] * 3, *[NULL] * 4),
}


def truth(rates: dict[str, np.ndarray]) -> dict[tuple[str, str], bool]:
    """(feature, attribute) -> True if the effect is real."""
    out = {}
    for feature, table in rates.items():
        out[(feature, "race")] = not np.isclose(table[0].mean(), table[1].mean())
        out[(feature, "sex")] = not np.isclose(table[:, 0].mean(), table[:, 1].mean())
        out[(feature, "race_AND_sex")] = not np.allclose(table, table[0, 0])
    return out


def simulate(rates: dict[str, np.ndarray], n: int, rng: np.random.Generator) -> pd.DataFrame:
    race = rng.integers(0, 2, n)
    sex = rng.integers(0, 2, n)
    cols = {"race": np.array(["a", "b"])[race], "sex": np.array(["F", "M"])[sex]}
    for feature, table in rates.items():
        value = np.ones(n)
        value[rng.random(n) < table[race, sex]] = np.nan
        cols[feature] = value
    return pd.DataFrame(cols)


FAMILIES = ("race", "sex", "race_AND_sex")


def _fdp(findings, real) -> float:
    rejected = [f for f in findings if f.is_significant]
    false = [f for f in rejected if not real[(f.target_feature, f.sensitive_attribute)]]
    return len(false) / max(1, len(rejected))


def _mean_se(values: list[float]) -> tuple[float, float]:
    a = np.asarray(values)
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else float("nan")


def run_fdr_cell(scenario: str, n: int, replicates: int, fdr: str, alpha: float = 0.05, seed: int = 0) -> dict:
    """One row: per-family FDR (gated), audit-wide FDR and power (reported)."""
    rates = SCENARIOS[scenario]
    real = truth(rates)
    rng = np.random.default_rng(seed)
    family_fdp = {name: [] for name in FAMILIES}
    pooled_fdp = []
    hits = {"race_AND_sex": 0, "parent": 0}
    real_count = {"race_AND_sex": 0, "parent": 0}
    for _ in range(replicates):
        result = audit(
            simulate(rates, n, rng), ["race", "sex"], sesoi=0.1, alpha=alpha,
            n_resamples=20, seed=int(rng.integers(1 << 31)), fdr=fdr,
        )
        missing = [f for f in result.findings if f.id.startswith("MAR_")]
        pooled_fdp.append(_fdp(missing, real))
        for name in FAMILIES:
            family_fdp[name].append(_fdp([f for f in missing if f.sensitive_attribute == name], real))
        for f in missing:
            kind = "race_AND_sex" if f.sensitive_attribute == "race_AND_sex" else "parent"
            if real[(f.target_feature, f.sensitive_attribute)]:
                real_count[kind] += 1
                hits[kind] += f.is_significant

    row = {"scenario": scenario, "fdr_mode": fdr, "n": n, "replicates": replicates}
    for name in FAMILIES:
        row[f"fdr_{name}"], row[f"fdr_{name}_se"] = _mean_se(family_fdp[name])
    row["fdr_audit_wide"], row["fdr_audit_wide_se"] = _mean_se(pooled_fdp)
    for kind in ("race_AND_sex", "parent"):
        row[f"{'intersection' if kind == 'race_AND_sex' else 'parent'}_power"] = (
            hits[kind] / real_count[kind] if real_count[kind] else float("nan")
        )
    return row


def passes_gate(row: dict, alpha: float = 0.05) -> bool:
    """Every family's FDR within alpha + 2 SE."""
    return all(row[f"fdr_{name}"] <= alpha + 2 * row[f"fdr_{name}_se"] for name in FAMILIES)
```

- [ ] **Step 2: Write the slow smoke test**

Create `tests/calibration/test_fdr_calibration.py`:

```python
"""Reduced FDR check: 100 replicates per cell, loose bounds."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fdr_harness import FAMILIES, RACE, XOR, run_fdr_cell, truth  # noqa: E402

pytestmark = pytest.mark.slow


def test_ground_truth_matches_the_construction():
    real = truth({"x": XOR, "y": RACE})
    assert real[("x", "race")] is False and real[("x", "sex")] is False
    assert real[("x", "race_AND_sex")] is True
    assert real[("y", "race")] is True and real[("y", "sex")] is False


@pytest.mark.parametrize("fdr", ["family", "hierarchical"])
@pytest.mark.parametrize("scenario", ["global_null", "xor"])
def test_every_family_stays_near_alpha(scenario, fdr):
    row = run_fdr_cell(scenario, n=2000, replicates=100, fdr=fdr, seed=3)
    for name in FAMILIES:
        assert row[f"fdr_{name}"] <= 0.10, name


def test_audit_wide_fdr_is_reported_not_controlled():
    """Under the global null, two parent families at alpha = 0.05 already give
    roughly a 10% chance of some false discovery. Recorded so the write-up
    cannot claim audit-wide control."""
    row = run_fdr_cell("global_null", n=2000, replicates=100, fdr="family", seed=5)
    assert row["fdr_audit_wide"] >= row["fdr_race"]


def test_hierarchical_cannot_see_xor_effects_and_family_can():
    """The documented trade-off, measured rather than asserted in prose."""
    hier = run_fdr_cell("xor", n=2000, replicates=50, fdr="hierarchical", seed=4)
    flat = run_fdr_cell("xor", n=2000, replicates=50, fdr="family", seed=4)
    assert hier["intersection_power"] < 0.2
    assert flat["intersection_power"] > 0.8
```

- [ ] **Step 3: Run the smoke test**

Run: `python -m pytest tests/calibration/test_fdr_calibration.py -q`
Expected: 7 pass (several minutes).

- [ ] **Step 4: Write the full gate script**

Create `tests/calibration/run_fdr_gate.py`:

```python
"""Full FDR gate (pre-registered criterion in the plan).

    cd tests/calibration && python run_fdr_gate.py
"""
import time
from pathlib import Path

import pandas as pd

from fdr_harness import SCENARIOS, passes_gate, run_fdr_cell

ALPHA = 0.05


def main() -> None:
    start = time.time()
    rows = [
        run_fdr_cell(name, n=2000, replicates=500, fdr=mode, alpha=ALPHA, seed=11)
        for name in SCENARIOS
        for mode in ("family", "hierarchical")
    ]
    df = pd.DataFrame(rows)
    out = Path("results") / "fdr_calibration_results.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    failed = [f"{r['scenario']}/{r['fdr_mode']}" for r in rows if not passes_gate(r, ALPHA)]
    print(f"\nGATE {'PASSED' if not failed else 'FAILED ' + ', '.join(failed)}: "
          f"every family's FDR <= alpha + 2 SE, both procedures ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the gate**

Run: `cd tests/calibration && python run_fdr_gate.py` (expect tens of minutes: 4,000 audits).
Expected: `GATE PASSED`. If `FAILED`, commit the harness and CSV anyway and report which scenario failed; do not tune the procedure to pass.

- [ ] **Step 6: Commit**

```bash
git add tests/calibration/fdr_harness.py tests/calibration/run_fdr_gate.py tests/calibration/test_fdr_calibration.py tests/calibration/results/fdr_calibration_results.csv
git commit -m "test(calibration): empirical FDR gate for family and hierarchical correction"
```

---

### Task 5: Record the result

**Files:**
- Modify: `results.md` §3, `roadmap.md` (M7 heading), `challenges.md` §4

- [ ] **Step 1: Update `results.md` §3**

Replace the **"Not yet done"** bullet list with a table built from `tests/calibration/results/fdr_calibration_results.csv`, one row per (scenario, procedure), columns `worst per-family FDR (± SE)`, `audit-wide FDR (± SE)`, `intersection power`, `parent power`, followed by one sentence stating whether the pre-registered gate passed and this sentence: `FDR is controlled within each (category, attribute) family, not across the audit; the audit-wide column shows how far apart those are.` Then add:

```markdown
**The trade-off, measured.** In the `xor` scenario — missingness that depends on race × sex with no marginal disparity — the default per-family correction detected the intersection effect in <intersection_power for family/xor> of audits; hierarchical correction in <intersection_power for hierarchical/xor>. Hierarchical audits report those cells as blind spots, never as clean.
```

replacing the two `<...>` values with the CSV numbers.

- [ ] **Step 2: Update `roadmap.md`**

If the gate passed, change the M7 heading status to `**[PASSED: hierarchical FDR opt-in; per-family FDR verified empirically]**`. If it failed, change it to `**[PARTIAL: hierarchical FDR implemented; empirical FDR gate failed — see results.md §3]**`.

- [ ] **Step 3: Update `challenges.md` §4**

Replace the sentence `It does not control FDR: intersections are still corrected as separate families, and hierarchical (Yekutieli) FDR is not implemented.` with `FDR control across the tree is a separate, opt-in procedure (fdr="hierarchical"): an intersection is tested only below a parent with a disparity. Its realised per-family and audit-wide FDR are measured in tests/calibration/run_fdr_gate.py; the price is that intersection-only effects go untested and are reported as blind spots.`

- [ ] **Step 4: Run the whole suite**

Run: `python -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add results.md roadmap.md challenges.md
git commit -m "docs: M7 hierarchical FDR results"
```

- [ ] **Step 6: Hand the default decision back**

Do not change `fdr`'s default. Report the Task 4 table to the project owner with two lines: how much audit-wide FDR under `global_null` drops from `family` to `hierarchical`, and how much intersection power `hierarchical` gives up in `xor` and `mixed`. Those two numbers are the trade-off the default decision rests on.
