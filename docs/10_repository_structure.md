# Repository Structure

This document supersedes the "Recommended Repository Layout" section of [README.md](../README.md). It adds the three modules introduced by the detectability reframe — `detectability/`, `correction/`, and `propagation/` — and makes the dependency rules between layers explicit and enforceable.

## 1. Layout

```text
.
├── README.md
├── pyproject.toml                    # packaging, deps, pytest config
├── docs/
│   ├── 01_architecture_spec.md
│   ├── 02_statistical_framework.md
│   ├── 03_bias_methodology.md
│   ├── 04_data_model_and_api.md
│   ├── 05_risk_and_severity.md
│   ├── 06_evaluation_protocol.md
│   ├── 07_ux_design.md
│   ├── 08_mvp_and_roadmap.md
│   ├── 09_detectability.md           # NOT YET WRITTEN — see §5
│   ├── 10_repository_structure.md    # this file
│   └── superpowers/plans/            # dated, executable implementation plans
│
├── src/dbias/
│   ├── __init__.py
│   ├── cli.py                        # `dbias audit data.csv --sensitive gender`
│   │
│   ├── models/                       # LAYER 0 — inert data, zero logic
│   │   ├── enums.py                  # Severity, Category, EffectSizeMetric,
│   │   │                             #   VariableKind, Detectability, Magnitude
│   │   ├── finding.py                # Finding
│   │   ├── profile.py                # DatasetProfile
│   │   └── risk.py                   # RiskVector, CoverageVector
│   │
│   ├── stats/                        # LAYER 1 — maths only, bias-blind
│   │   ├── effect_sizes.py           # Cramer's V, Cohen's w, rank-biserial, eta^2
│   │   ├── tests.py                  # scipy wrappers returning (stat, p, effect)
│   │   ├── dispatch.py               # decision tree: dtypes -> TestSpec
│   │   ├── thresholds.py             # per-metric small/medium/large registry
│   │   └── correction.py             # flat Benjamini-Hochberg
│   │
│   ├── detectability/                # LAYER 2 — the inverse question
│   │   ├── power.py                  # achieved power, minimum detectable effect
│   │   ├── classify.py               # ADEQUATE vs UNDERPOWERED verdict
│   │   └── coverage.py               # attribute x feature MDE grid
│   │
│   ├── correction/                   # LAYER 2 — structured multiplicity
│   │   ├── hierarchy.py              # builds the attribute -> intersection tree
│   │   └── gating.py                 # power-guided descent + hierarchical FDR
│   │
│   ├── ingestion/                    # LAYER 2 — dataframe in, profile out
│   │   ├── loaders.py                # CSV / Parquet via pandas
│   │   ├── schema_inference.py       # semantic dtype inference + user override
│   │   └── sensitive_detection.py    # heuristic protected-attribute detection
│   │
│   ├── analyzers/                    # LAYER 3 — domain hypotheses
│   │   ├── base.py                   # BaseAnalyzer ABC
│   │   ├── representation.py
│   │   ├── missingness.py
│   │   ├── feature_disparity.py
│   │   ├── label_disparity.py        # includes the four-fifths rule
│   │   └── intersectional.py         # drives correction/gating.py
│   │
│   ├── rules/                        # LAYER 4 — the only place severity exists
│   │   ├── severity.py               # (significance, magnitude, detectability) -> Severity
│   │   └── risk_vector.py            # per-category roll-up + coverage roll-up
│   │
│   ├── propagation/                  # LAYER 5 — research tier, optional
│   │   ├── repair.py                 # counterfactual dataset repairs per finding
│   │   ├── surrogate.py              # cheap model trained on original vs repaired
│   │   └── attribution.py            # downstream fairness delta per finding
│   │
│   ├── explainability/               # LAYER 5 — humanising
│   │   ├── nlg.py                    # evidence_text / potential_impact / recs
│   │   └── templates/                # per-category phrasing templates
│   │
│   ├── report/                       # LAYER 5 — serialisation
│   │   ├── json_export.py            # the MVP deliverable
│   │   └── pdf.py
│   │
│   └── app/                          # LAYER 6 — Streamlit
│       ├── main.py
│       └── components/
│           ├── findings_feed.py
│           ├── coverage_map.py       # the blind-spot heatmap
│           └── deep_dives.py         # KDE / ECDF panels
│
├── tests/
│   ├── unit/                         # mirrors src/dbias/ one-to-one
│   │   ├── models/
│   │   ├── stats/
│   │   ├── detectability/
│   │   ├── correction/
│   │   ├── ingestion/
│   │   ├── analyzers/
│   │   ├── rules/
│   │   └── propagation/
│   ├── synthetic/                    # ground-truth evaluation (docs/06)
│   │   ├── generators.py             # causal DAG data generators
│   │   ├── test_scenario_a_null.py       # negative control, well-powered
│   │   ├── test_scenario_a_underpowered.py  # SAME null, small n -> BLIND_SPOT
│   │   ├── test_scenario_b_mnar.py
│   │   ├── test_scenario_c_large_n.py
│   │   └── test_scenario_d_blind_spot.py    # real effect, no power to see it
│   ├── benchmarks/                   # vs incumbent tools (docs/06 §2)
│   │   └── test_incumbent_comparison.py
│   └── e2e/
│       └── test_cli_audit.py
│
└── datasets/                         # gitignored except for a small fixture
    ├── adult/
    ├── compas/
    └── german_credit/
```

## 2. Dependency Rules

Layers import downward only. A module may import from any lower-numbered layer and never from a higher one. This is what keeps the statistical core reusable and testable in isolation — and it is the constraint doc/01 already states informally.

| Layer | Package | May import from | Must never import |
| :--- | :--- | :--- | :--- |
| 0 | `models/` | stdlib only | anything in `dbias` |
| 1 | `stats/` | `models` | `analyzers`, `rules`, `app` |
| 2 | `detectability/`, `correction/`, `ingestion/` | `models`, `stats` | `analyzers`, `rules`, `app` |
| 3 | `analyzers/` | `models`, `stats`, `detectability`, `correction`, `ingestion` | `rules`, `app` |
| 4 | `rules/` | `models` + layers 1–3 | `app`, `propagation` |
| 5 | `propagation/`, `explainability/`, `report/` | `models` + layers 1–4 | `app` |
| 6 | `app/`, `cli.py` | anything | — |

Three invariants worth stating loudly, because violating them is how this project would quietly become another metrics calculator:

1. **`stats/` never learns what bias is.** No parameter in that package may be named `sensitive`, `protected`, or `group`. It receives arrays and returns numbers. If a function there needs to know which group is disadvantaged, it belongs in `analyzers/`.
2. **Severity exists only in `rules/`.** Analyzers emit `Finding` objects with `severity = UNDETERMINED`. Nothing outside `rules/` may assign it.
3. **`detectability/` is not optional and not a plugin.** Every `Finding` passes through it before reaching `rules/`. A `Finding` with `detectability == UNKNOWN` reaching the report layer is a bug, not a degraded mode.

## 3. Why These Splits

**`detectability/` is separate from `stats/`** even though both are pure maths. They answer opposite questions — "what does this sample show?" versus "what could this sample have shown?" — and the second is the project's differentiator. Keeping it in its own package makes it independently citable, independently testable, and hard to quietly drop under deadline pressure.

**`correction/` is separate from `stats/correction.py`.** The latter is the flat, stateless BH procedure over a list of p-values. The former is the stateful, tree-shaped, power-gated descent that decides *which tests to run at all*. Different concerns, different test strategies; the flat one is a pure function, the tree one needs fixtures.

**`propagation/` is quarantined at layer 5** because it is the only module that trains models, the only one with heavy compute, and the only one that could be cut entirely without breaking the MVP. It depends on everything and nothing depends on it.

**`tests/synthetic/` is not under `tests/unit/`.** These are not unit tests — they are the evaluation protocol from docs/06, measuring the tool's own detection accuracy against known ground truth. They are slow, they are the scientific claim, and they should be runnable and reportable on their own: `pytest tests/synthetic -v`.

**Scenario A appears twice.** Same null hypothesis, two sample sizes, two different correct answers (`INFORMATIONAL` vs `BLIND_SPOT`). That pair is the single clearest demonstration of what this tool does that others do not, so it gets two files rather than one parametrised test.

## 4. Conventions

- Package name is `dbias`, under `src/` layout so tests run against the installed package rather than the working tree.
- One public concept per module; if a file exceeds roughly 300 lines, split by responsibility rather than by technical layer.
- Test files mirror source paths exactly: `src/dbias/stats/thresholds.py` is tested by `tests/unit/stats/test_thresholds.py`.
- Test oracles are hand-computed closed forms or an independent library (statsmodels, R output). Never assert against the implementation's own output.
- Every module in `stats/` and `detectability/` carries a docstring stating what it deliberately does *not* know.

## 5. Known Gaps

- **`docs/09_detectability.md` does not exist.** The detectability design currently lives inlined in the Global Constraints of [the first implementation plan](superpowers/plans/2026-08-20-detectability-core.md). It should be written up as a proper spec before the rules-engine plan consumes `Detectability`.
- **docs/02 and docs/05 are stale.** Both carry effect-size threshold ladders that the threshold registry deliberately overrides, and docs/05's severity table has no row for `Blind Spot`. Both need revising once the rules engine lands.
- **docs/03 §2 mislabels MAR as MNAR.** Testing a NaN-mask against an *observed* attribute detects MAR; true MNAR is not identifiable from observed data. The method is sound, the name is not.
- **The FDR family boundary is unresolved in docs/02**, which says to pool all p-values globally. `correction/gating.py` is the intended answer, but the policy needs writing down.
