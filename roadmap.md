# Roadmap

Sequencing for the detectability-aware dataset bias auditor. Read [plan.md](plan.md) first — the scope cuts and methodological fixes referenced here are decided there, not open.

**Sizing assumption:** one developer, part-time. Durations are in *working weeks of actual effort*, not calendar weeks. Adjust the calendar, not the ordering.

**Ordering principle:** the scientific claim is built before the software around it. The coverage map and the calibration experiment come *before* the dashboard, the LLM layer, and the analyzers are polished — because if calibration fails, everything downstream changes.

---

## Milestone map

```
M0 Foundations ──▶ M1 Statistical core ──▶ M2 Detectability core ──▶ M3 CALIBRATION GATE
                                                                          │
                                       ┌──────────────────────────────────┘
                                       ▼
                          M4 Ingestion + analyzers ──▶ M5 Rules + coverage map
                                                              │
                                                              ▼
                                                    M6 BENCHMARK GATE
                                                              │
                            ┌─────────────────────────────────┤
                            ▼                                 ▼
                  M7 Intersectional gating              M8 Write-up
                            │
                            ▼
                  M9 Dashboard (optional)  ──▶ M10 NLG (optional)
```

Two hard gates. Nothing downstream of a gate starts until the gate passes.

---

## M0 — Foundations (0.5 week)

Executes Task 1 of `docs/superpowers/plans/2026-08-20-detectability-core.md` as written.

- `pyproject.toml`, `src/dbias/` package skeleton under a src-layout.
- `models/enums.py`: `Severity` (including `BLIND_SPOT`), `Category`, `EffectSizeMetric`, `VariableKind`, `Detectability`, `Magnitude`.
- `models/finding.py`, `models/profile.py`.
- `pytest` configured; `pip install -e ".[dev]"` works.

**Amendment to the plan as written:** add three fields to `Finding` now, so no migration is needed later —
`effect_size_ci: tuple[float, float] | None`, `equivalence_verdict: str | None`, `sesoi: float | None`.

**Done when:** `pytest tests/unit/models -v` passes and a `Finding` constructed without severity reports `Severity.UNDETERMINED` and `Detectability.UNKNOWN`.

---

## M1 — Statistical core (1.5 weeks)

Executes Tasks 2–5 of the existing plan as written. These are sound; no amendments.

- `stats/effect_sizes.py` — Cramér's V, Cohen's *w*, rank-biserial, eta-squared.
- `stats/thresholds.py` — per-metric registry with df-adjusted Cramér's V (OVERRIDE 1 and 2).
- `stats/dispatch.py` — the docs/02 decision tree.
- `stats/correction.py` — flat Benjamini–Hochberg, cross-checked against `statsmodels`.

**Add, not in the current plan:** `stats/intervals.py` — bootstrap confidence intervals for every effect size in the registry. This is a prerequisite for M2 and is the fix for plan.md §3.2. Budget three days; use BCa bootstrap, 2,000 resamples, seeded.

**Done when:** every effect size returns a point estimate *and* an interval, and every test oracle is hand-computed or from `statsmodels` — never from the implementation's own output.

---

## M2 — Detectability core (1.5 weeks)

Executes Task 6 as written, and Task 7 **as amended by plan.md §3.1–3.6**.

- `detectability/power.py` — `mde_chi_square`, `mde_rank_biserial`, plus a **simulation-based MDE fallback** for skewed margins and sparse cells (plan.md §3.4).
- `detectability/classify.py` — the verdict. Amended contract:
  - Takes a **required** `sesoi` argument. No silent Cohen default.
  - Verdict comes from a **TOST equivalence test** against the SESOI, using the CI from M1.
  - `power_to_detect_sesoi` replaces `achieved_power`. **The observed-effect power path is deleted, not deprecated.**
  - k > 2 groups routes to the eta-squared path or raises. It must not silently take the two largest groups.
- `detectability/coverage.py` — the attribute × feature MDE grid.

**Done when:** a `Finding` that passes through `annotate_detectability` carries a SESOI, a CI, an equivalence verdict, an MDE, and a `Detectability` that is never `UNKNOWN` on a supported metric.

---

## M3 — CALIBRATION GATE (1.5 weeks) 🚦 **[PASSED]**

**This is the scientific claim. Everything before it is code; this is the experiment.**

Build `tests/calibration/` (separate from `tests/unit/` and from `tests/synthetic/` — different purpose, different runtime).

Design:

| Factor | Levels |
| :--- | :--- |
| Sample size *n* | 50, 100, 500, 2 000, 10 000, 100 000 |
| True effect (Cohen's *w*) | 0.0, 0.05, 0.1, 0.2, 0.3, 0.5 |
| Group imbalance | 50/50, 80/20, 95/5, 99/1 |
| Table shape | 2×2, 2×5 |
| Replicates | ≥ 1 000 per cell |

Outputs:

1. **Empirical power curve** vs analytic MDE prediction, per imbalance regime.
2. **Calibration figure** — the paper's Figure 1.
3. **False-positive rate under the null**, with and without BH-FDR, confirming the correction earns its place.

**Gate criteria — all three must hold:**

- In cells the tool labels `ADEQUATE`, empirical detection rate at the SESOI is **≥ 0.80**.
- In cells labelled `BLIND SPOT`, empirical detection rate at the SESOI is **materially below 0.80** (no false reassurance).
- Under the null with FDR applied, empirical FDR **≤ 0.05**.

**Result:** The gate passed. Simulation-based MDE fallback was added to `detectability/power.py` for 2x2 tables to fix the optimistic analytic MDE for extremely skewed margins.

---

## M4 — Ingestion and analyzers (2 weeks) 🚦 **[PASSED]**

Only now does the tool meet real data.

- `ingestion/loaders.py`, `schema_inference.py`, `sensitive_detection.py`. Sensitive-attribute detection **suggests, never decides** — both Review documents are firm on this and they are right.
- `analyzers/base.py` (`BaseAnalyzer` ABC), then `representation.py`, `missingness.py`, `feature_disparity.py`, `label_disparity.py` (including the four-fifths ratio).
- Analyzers emit findings with `severity = UNDETERMINED`. No exceptions.
- Rename MNAR → MAR throughout (plan.md §3.7). `MissingnessAnalyzer` keeps its name; finding IDs change.

**Done when:** a CSV goes in and a list of un-scored, detectability-annotated `Finding` objects comes out.

---

## M5 — Rules engine and coverage map (1 week) 🚦 **[PASSED]**

- `rules/severity.py` — maps `(is_significant, magnitude, detectability)` → `Severity`. The `BLIND_SPOT` row is the point of the table; `docs/05` currently lacks it.
- `rules/risk_vector.py` — per-category roll-up **plus** the coverage roll-up.
- `report/json_export.py` — Data Card-shaped JSON. The report **must state the SESOI it used, verbatim**.
- `report/coverage_plot.py` — static matplotlib heatmap of the coverage map. Not Streamlit. This is the figure.

**Done when:** `dbias audit data.csv --sensitive gender --sesoi 0.1` writes JSON and a PNG. **This is the MVP.**

---

## M6 — BENCHMARK GATE (1 week) 🚦 **[PASSED: Pivot to Small-N Validated]**

Run against Adult, COMPAS, and German Credit.

- Reproduce the known findings (Adult label disparity by gender; COMPAS distributional differences by race). If these do not appear, the analyzers are wrong.
- Run the synthetic scenarios from `docs/06`: A-well-powered, A-underpowered, B (injected MAR), C (large-*n* trivial effect), D (real effect, no power).
- **The headline result:** identify at least one (attribute, feature) cell in a *real* benchmark that AIF360 or Fairlearn reports as clean and this tool marks as a blind spot. Verify the claim by running the incumbent tool, not by assuming.

**Gate criterion:** that cell exists, is reproducible, and is defensible. If no such cell exists in any of the three benchmarks, the practical motivation for the whole project is weaker than assumed, and the write-up must say so honestly — probably reframing toward small-*n* clinical or regional datasets where the problem is severe by construction.
**Result:** Known findings reproduce (Adult income by sex; COMPAS recidivism by race). Adult (N=32k) has 0 blind spots at `w=0.1`. COMPAS (N=7k) has 3, all on `vr_charge_degree`, a feature recorded for only 819 rows. Subsampled Adult (N=500) has 19, and for `MAR_WORKCLASS_RACE` Fairlearn's output was run side by side and reads as clean. See `results.md`. Full-size COMPAS does not meet the gate: Fairlearn flags its blind-spot cells as unfair rather than clean (small-group noise; `benchmarks/compare_compas.py`).

---

## M7 — Intersectional gating (1.5 weeks) 🚦 **[PARTIAL: hierarchy, gating and per-family FDR verified; hierarchical FDR implemented but failed its empirical FDR gate — see results.md §3]**

Now, and not before, descend into intersections.

- `correction/hierarchy.py` — build the attribute → intersection tree. **Cap depth at 2** for this project.
- `correction/gating.py` — power-guided descent: do not test a subgroup whose MDE already exceeds the SESOI; mark it a blind spot and stop. This is the design idea worth writing about — power as the criterion for *which hypotheses to formulate*, which controls multiplicity and compute at once.
- Hierarchical FDR (Yekutieli-style) over the tree; document the family boundary explicitly, resolving the open question in `docs/10 §5`.

**Done when:** an intersectional audit terminates in bounded time, reports a coverage vector alongside the risk vector, and its FDR is verified empirically in the M3 harness.

---

## M8 — Write-up (2 weeks, overlappable with M7)

- Write `docs/09_detectability.md` properly — this should actually happen at the *start* of M2, not here; it is listed here only so it is never forgotten.
- Paper structure: the null-result problem → TOST/SESOI formulation → coverage map → calibration experiment (Figure 1) → benchmark blind spot (Figure 2) → limitations.
- **Limitations section is not optional.** State plainly: statistical association only, never causation; no claim to detect or mitigate bias; verdicts are conditional on a declared SESOI; MDE approximations degrade under extreme skew.
- Target: tools/resources or applied track. Not a methods track (plan.md §2).

---

## M9 — Dashboard (1.5 weeks, optional)

Only after M6 passes. Streamlit, per `docs/07`: sidebar config, findings feed sorted by severity, coverage map tab, deep-dive KDE/ECDF panels. Blind-spot findings get their own visual treatment — they are not a lesser grade of "clean."

## M10 — LLM narrative layer (1 week, optional)

Last. The LLM consumes structured findings and emits prose. It never sees raw data and never decides whether something is biased — both Reviews are unambiguous, and this is the constraint that keeps the tool trustworthy. Every generated sentence must be traceable to a field in a `Finding`.

---

## Cut list — deliberately not on this roadmap

Recorded so they are not silently reintroduced. Rationale in plan.md §4.

- `propagation/` — counterfactual repair and downstream fairness attribution. Heavy prior art, contradicts the pre-training premise, doubles the surface area. Future-work paragraph only.
- MMD-Critic prototypes and criticisms.
- PDF export.
- Model-fairness (post-prediction) metrics.
- Depth-3+ intersection lattices.

---

## Critical path

**M0 → M1 → M2 → M3 → M4 → M5 → M6.** Roughly 9–10 weeks of effort to a defensible MVP with a benchmark result, plus a likely one-week contingency at M3.

Everything after M6 is either the paper or optional polish. If time runs out, a project that stops after M6 with a good write-up is a complete piece of work. A project that reaches M9 without passing M3 is a Streamlit app with an unverified claim in it.
