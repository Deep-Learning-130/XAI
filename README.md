# Dataset Bias Detection & Analysis Tool

Welcome to the **Dataset Bias Detection & Analysis Tool** repository. This project is a rigorous, statistically grounded XAI/Responsible AI tool designed to audit datasets for potential biases, representation flaws, and fairness risks *before* models are trained.

## Mission Statement
The industry currently over-indexes on *model fairness* (post-hoc metrics) while neglecting *dataset auditing* (Data-Centric AI). This tool bridges that gap. It distinguishes clearly between **observed imbalances**, **statistically significant disparities**, and **potential biases**. It actively avoids the trap of equating simple class imbalance with societal bias, relying instead on inferential statistics, effect sizes, and rigorous multiple-testing corrections.

---

## Quickstart

A working vertical slice is implemented. See [handoff.md](handoff.md) §2b for
exactly what exists and §2c for what does not.

```bash
pip install -e ".[dev]"
python -m pytest                          # 220 tests, ~35 s
python -m dbias.cli demo --out out/demo   # the demonstration, on synthetic data
```

`demo` writes `audit.json` and `coverage_map.png`. To audit your own CSV:

```bash
dbias audit data.csv --sensitive gender --sensitive ethnicity --target hired --sesoi 0.1 --out out/
```

`--sesoi` — the smallest effect size of interest, on the Cohen's *w* scale —
is required and has no default. Every verdict the tool produces is conditional
on it, and a tool whose central claim is *"we could have caught anything that
matters"* has to make you say what matters.

### What the output looks like

The findings feed splits null results in two, which is the point of the whole
project. Two findings with the same p-value:

```
MAR_INCOME_GENDER      p=0.475  effect=0.029  MDE=0.093  ->  Informational  (earned all-clear)
MAR_INCOME_ETHNICITY   p=0.100  effect=0.108  MDE=0.115  ->  Blind Spot     (we could not see)
```

The second one has a *real* injected disparity of w = 0.102 behind it, above
the declared SESOI of 0.1. Every incumbent tool reports both lines as "no
significant disparity found". Alongside the feed, the coverage map shows the
minimum detectable effect for every (attribute, feature) cell, so a reader can
see which parts of the dataset the audit is entitled to speak about at all.

---

## Project Structure & Documentation

To ensure a smooth hand-off and modular implementation, the complete system specification has been divided into detailed technical documents located in the `docs/` directory:

1. **[Architecture Specification](docs/01_architecture_spec.md)**: System design, data flow, and component responsibilities.
2. **[Statistical Framework](docs/02_statistical_framework.md)**: The mathematical backbone (tests, effect sizes, FDR correction).
3. **[Bias Detection Methodology](docs/03_bias_methodology.md)**: How specific bias types (MNAR, Representation, Label) are identified.
4. **[Data Model & API Contracts](docs/04_data_model_and_api.md)**: Dataclasses (e.g., `Finding`) and abstract base classes for Analyzers.
5. **[Risk & Severity Framework](docs/05_risk_and_severity.md)**: How findings are escalated from 'Informational' to 'Critical'.
6. **[Evaluation Protocol](docs/06_evaluation_protocol.md)**: How to build synthetic datasets and benchmark the tool to ensure it actually works.
7. **[UX & Dashboard Design](docs/07_ux_design.md)**: Frontend specifications for Streamlit/Dash.
8. **[MVP Specification & Roadmap](docs/08_mvp_and_roadmap.md)**: The phased development plan and definition of the Minimum Viable Product. *(superseded by [roadmap.md](roadmap.md))*
9. **[Detectability](docs/09_detectability.md)**: The blind-spot verdict — SESOI, equivalence testing, and the coverage map. **This is the contribution; read it after 02 and 04.**
10. **[Repository Structure](docs/10_repository_structure.md)**: Authoritative layout and the layer dependency rules.
11. **[Prototype Report](docs/11_prototype_report.md)**: What the built slice does, the data it ran on, the exact numbers, and what they imply. **Start here if you want the results.**

## Recommended Repository Layout

```text
.
├── README.md                 # This file
├── docs/                     # Technical specifications and hand-off docs
├── src/                      # Main source code
│   ├── ingestion/            # Pandas/Polars loaders and schema inference
│   ├── stats/                # Hypothesis testing and effect size calculators
│   ├── analyzers/            # Bias detection domain logic
│   ├── rules/                # Severity thresholds and risk logic
│   ├── explainability/       # NLG report generation, XAI integrations
│   ├── models/               # Dataclasses and schemas
│   └── app/                  # Streamlit dashboard
├── tests/                    # Unit and E2E tests
│   ├── unit/
│   └── synthetic/            # Synthetic data generation for evaluation
└── datasets/                 # Sample datasets for testing (Adult, COMPAS)
```

## How to Proceed

Start with **[handoff.md](handoff.md)** — it is written for arriving cold and
supersedes this section. Then [plan.md](plan.md) §3 for the methodological
corrections, and [docs/09_detectability.md](docs/09_detectability.md) for the
design rationale of the part that matters.

Note that the "Recommended Repository Layout" above predates the detectability
reframe. [docs/10_repository_structure.md](docs/10_repository_structure.md) is
the authoritative layout; where the two disagree, docs/10 wins.
