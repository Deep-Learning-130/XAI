# Handoff

Everything needed to pick this project up cold. If you are an agent or a developer arriving with no context, read this file top to bottom, then [plan.md](plan.md) §3, then start at the section "Where to start" below.

**Last updated:** 2026-08-20
**Repository state:** a working vertical slice. `src/dbias/` is installable, `dbias demo` runs end to end, and 237 tests pass. The **M3 calibration gate** has passed for 2x2 tables and the simulation-based MDE fallback for 2x2 tables is implemented. See §2b for exactly what exists and §2c for what does not.

---

## 1. What this project is, in three sentences

A pre-training auditor for tabular datasets. It runs statistical tests for representation imbalance, missingness disparity, feature disparity and label disparity across protected attributes — and, unlike every incumbent tool, it reports for each test whether the dataset had enough statistical power to have found a disparity at all, so that "we checked and it is clean" is never confused with "we could not see."

That second half — **detectability** — is the contribution. The rest is plumbing around it.

---

## 2. Repository as it stands

```
XAI/
├── README.md                          # original framing; predates the detectability reframe
├── plan.md                            # ← assessment, defects, scope decisions  (NEW)
├── roadmap.md                         # ← milestones and two hard gates          (NEW)
├── handoff.md                         # ← this file                              (NEW)
└── docs/
    ├── 01_architecture_spec.md        # component layering and data flow
    ├── 02_statistical_framework.md    # test decision tree, BH-FDR, effect sizes   [STALE §4]
    ├── 03_bias_methodology.md         # per-bias-type methodology                  [MISLABELS MAR AS MNAR]
    ├── 04_data_model_and_api.md       # Finding dataclass, BaseAnalyzer ABC        [SUPERSEDED by plan Task 1]
    ├── 05_risk_and_severity.md        # severity ladder, risk vector               [STALE — no BLIND_SPOT row]
    ├── 06_evaluation_protocol.md      # synthetic scenarios + benchmarks           [INSUFFICIENT — see §5]
    ├── 07_ux_design.md                # Streamlit dashboard spec                   [deferred]
    ├── 08_mvp_and_roadmap.md          # original 7-phase roadmap                   [SUPERSEDED by roadmap.md]
    ├── 09_detectability.md            # design rationale, SESOI/equivalence   [WRITTEN]
    ├── 10_repository_structure.md     # authoritative layout + dependency rules    [CURRENT]
    └── superpowers/plans/
        └── 2026-08-20-detectability-core.md   # TDD plan  [Tasks 1–6 EXECUTED; Task 7 SUPERSEDED, see §2b]
```

---

## 2b. What is implemented

Installable with `pip install -e ".[dev]"`; `python -m pytest` is green (220 tests, ~35 s).

```
src/dbias/
├── models/         enums, Finding (with CI / SESOI / verdict fields), DatasetProfile
├── stats/          chi_square (independence + goodness of fit), effect_sizes,
│                   thresholds (per-metric, df-adjusted), intervals (bootstrap CI),
│                   correction (BH, cross-checked against scipy)
├── detectability/  power (non-central chi-square MDE), classify (the verdict),
│                   coverage (attribute × feature grid)
├── analyzers/      representation, missingness (MAR), label_disparity
├── rules/          severity (incl. the BLIND_SPOT row), risk_vector + coverage_vector
├── report/         json_export (states the SESOI verbatim), coverage_plot (the figure)
├── audit.py        the pipeline; BH families are per (category, attribute)
├── synthetic.py    scenario generators carrying known ground truth
└── cli.py          dbias audit … --sesoi 0.1   |   dbias demo

tests/    unit (193) · synthetic evaluation scenarios (18) · calibration (9)
```

Run the demonstration:

```bash
pip install -e ".[dev]"
python -m dbias.cli demo --out out/demo      # writes audit.json + coverage_map.png
```

**The one statistical path.** Everything runs through chi-square / Cramér's V.
That single path covers representation (goodness of fit), MAR missingness
(NaN-mask × attribute) and label disparity (label × attribute), because all
three are contingency tables. This is the cut that made the slice fit in a
night, and it is why Mann–Whitney, KS, eta-squared and every continuous
feature are absent.

**Corrections from plan.md §3 that are applied:** 1 and 2 (no observed power
anywhere; the verdict is an equivalence decision on a bootstrap CI against a
required SESOI), 6 (SESOI is required configuration, echoed verbatim in the
report), 7 (MAR naming from the first commit), and the BH oracle is
`scipy.stats.false_discovery_control`. Defect 5 is moot — the rank-biserial
path does not exist.

**Correction 4 (skewed margins) is fixed for 2x2 tables.** The simulation-based
MDE is implemented in `detectability/power.py`. Cells with highly skewed margins
drop to empirical binomial simulation rather than relying on the optimistic 
chi-square non-centrality approximation. For tables larger than 2x2 where it is
not implemented, it carries `mde_is_approximate=True` into the JSON and an
asterisk on the figure.

**One design decision was made during implementation and is not in plan.md.**
Where the power-based `detectability` label and the CI-based
`equivalence_verdict` disagree, the interval wins, and `rules/severity.py`
implements that precedence. Rationale in [docs/09](docs/09_detectability.md) §6.

## 2c. What the slice does NOT contain

- **Real benchmark data.** Adult, COMPAS and German Credit are untouched, so
  roadmap M6 — and the headline "an incumbent tool says clean, this says blind
  spot" cell on a *real* dataset — does not exist. Everything is synthetic.
- Intersections, hierarchical FDR, the dashboard, the LLM layer, `propagation/`.
- Continuous features and continuous sensitive attributes.

**Authoritative documents:** `docs/10_repository_structure.md`, the implementation plan, and the three new root-level files. Where anything else disagrees, those win.

Two review documents (`Dataset Bias Auditing Framework Review1.docx`, `...Review3.pdf`) exist outside the repo. They are literature reviews of the *original* framing only and predate the detectability reframe; their "integration and automation contribution" verdict applies to layer one, not to the current design. See plan.md §2.

---

## 3. Decisions already made — do not relitigate

1. **No single aggregate bias score.** Ever. Output is a per-category risk vector plus a coverage vector. Both reviews reached this independently.
2. **Effect sizes are mandatory.** No function in `stats/` returns a p-value without one.
3. **Layers import downward only** (`docs/10 §2`). `stats/` never learns what a protected attribute is — no parameter in that package may be named `sensitive`, `protected`, or `group`.
4. **Severity is assigned only in `rules/`.** Analyzers emit `Severity.UNDETERMINED`.
5. **Human-in-the-loop configuration.** The tool suggests candidate sensitive attributes; the user declares them. It never auto-classifies an attribute as sensitive.
6. **No causal claims.** Outputs are statistical disparities and correlational risks. The tool does not detect discrimination and does not mitigate bias.
7. **The LLM never touches raw data.** It consumes structured `Finding` objects and produces prose. Deferred to last regardless.
8. **`propagation/` is cut** from this project (plan.md §4).
9. **Test oracles must be independent** — hand-computed closed forms or `statsmodels`, never the implementation's own output.

---

## 4. Corrections to apply before writing code

Full detail in plan.md §3. Summary, in order of severity:

| # | Defect | Where | Fix |
| :-- | :--- | :--- | :--- |
| 1 | **Post-hoc observed power is a fallacy** — it is a monotone function of the p-value and carries no information | implementation plan, Task 7, `annotate_detectability` | Compute power against a pre-specified SESOI. **Delete** the observed-effect path. |
| 2 | MDE alone is weaker than a confidence interval; reviewers will say "just report a CI" | whole detectability design | Add bootstrap CIs; make the verdict a **TOST equivalence test** against the SESOI. Keep MDE as the human-facing communication device only. |
| 3 | Evaluation is circular — asserting the formula against itself | `docs/06`, planned Scenario A | Replace with an **empirical calibration** experiment (roadmap M3). **[DONE]** |
| 4 | Chi-square MDE ignores marginal skew; errs *optimistic* exactly where it matters most | `mde_chi_square` | Condition on observed margins, or simulate when `min(expected) < 5` or ratio > 4:1. **[DONE for 2x2]** |
| 5 | `mde_rank_biserial` silently picks the two largest groups when k > 2 | `classify.py` | Route k > 2 to eta-squared, or raise. |
| 6 | Adequacy rule hardcodes Cohen's 1988 conventions into the central claim | `classify.py` | SESOI is required user configuration with a documented default, echoed verbatim in the report. |
| 7 | `docs/03 §2` labels a MAR test as MNAR | `docs/03` | Rename. True MNAR is not identifiable from observed data. |
| 8 | Cleanlab limitation cited to a vendor blog | review docs | Cite Northcutt et al., *Confident Learning*, JAIR 2021. |

---

## 5. Where to start

The vertical slice in §2b exists, so the ordering below has changed from what
this section originally said. Both of its old instructions — write `docs/09`,
execute Tasks 1–6 — are done.

**Do these in order. The first two are the project; the rest is polish.**

**1. Run the full M3 calibration gate (roadmap M3, ~1 week). [DONE for 2x2]**
`tests/calibration/run_m3_gate.py` ran 85,000 simulated datasets (85 cells x 1,000
replicates; n in {100 ... 10,000}, minority share down to 1%, 2x2 tables only).
The false positive rate held near alpha; see `results.md` Sec 1. Still outstanding
from the original spec: n up to 100,000 and tables larger than 2x2.

**2. Build the simulation-based MDE (plan.md §3.4). [DONE for 2x2]**
The simulation-based fallback drops into `detectability/power.py` and is fully integrated for 2x2 tables.

**3. Get onto real data (roadmap M4/M6).** Everything so far is synthetic, so
the project's headline claim — a cell an incumbent tool calls clean and this
one calls a blind spot, on a *real* benchmark — has not been made. Fetch
Adult, COMPAS and German Credit, then **verify the claim by actually running
AIF360 or Fairlearn on that cell**, not by assuming what they would say.

**4. Widen the statistical path.** Mann–Whitney / rank-biserial and
Kruskal–Wallis / eta-squared for continuous features. Route k > 2 groups to the
eta-squared path or raise — do not reintroduce the two-largest-groups bug
(plan.md §3.5). Each new metric needs a threshold ladder in
`stats/thresholds.py` and a bootstrap interval in `stats/intervals.py`, or the
detectability pass will refuse it, which is the correct failure.

**5. Intersections (roadmap M7).** The FDR family boundary is already one
family per (category, attribute), implemented in `audit.py`. Hierarchical
descent must nest *inside* those families rather than redefining them.

**What not to do first:** the dashboard, the LLM layer, or `propagation/`.
§8 below is the test for whether this has gone off track, and it still applies.

---

## 6. Environment

- Python 3.11+ (the plan uses `StrEnum` and `X | None`).
- `numpy>=1.26`, `pandas>=2.1`, `scipy>=1.11`, `statsmodels>=0.14`; `pytest>=8.0` for dev.
- src-layout: package is `dbias` under `src/`, so tests run against the installed package. `pip install -e ".[dev]"`.
- Datasets (Adult, COMPAS, German Credit) are gitignored except for one small fixture. Fetch scripts do not exist yet — write them in M4.

---

## 7. Open questions

Unresolved. Each needs a decision before the milestone that consumes it.

1. **FDR family boundary.** `docs/02` says pool all p-values globally; `docs/10` says `correction/gating.py` is the answer but the policy is unwritten. Needed by M7. Recommendation: one family per (category × attribute), with hierarchical descent inside intersections.
2. **SESOI default.** What number ships when the user declares nothing? Needed by M2. Recommendation: Cohen's *w* = 0.1, stated loudly in every report, with a documented warning that it is a convention and not a domain judgement.
3. **Continuous sensitive attributes.** Age as a continuous variable currently has no path — everything assumes categorical groups. Needed by M4. Simplest answer: require user-declared binning, and say so.
4. **What the tool does with zero-count intersections.** A cell with 0 rows is not a blind spot, it is an absence. It probably deserves its own verdict rather than being folded into `UNDERPOWERED`.
5. **KS-D thresholds are provisional** and admitted as such in the plan. Either find a defensible convention or exclude KS from the detectability path.

---

## 8. How to tell if this is going well

The project is on track if, at any point you check in, the most recent work was on either the statistical core or the calibration experiment. It has gone off track if the most recent work was on the dashboard, the LLM layer, or the propagation module.

The one artifact that determines whether this is a real contribution: **a coverage map of a real benchmark dataset, showing a cell that an incumbent tool reports as clean and this tool reports as a blind spot.** Everything in the roadmap is arranged to produce that image as early as possible.

---

## Sources consulted for the novelty assessment

- [A Brief Tutorial on Sample Size Calculations for Fairness Audits](https://arxiv.org/abs/2312.04745) — nearest prior art; model-level, prospective.
- [Cherian & Candès, Statistical Inference for Fairness Auditing, JMLR 2024](https://jmlr.org/papers/v25/23-0739.html) — simultaneous multiplicity-corrected subgroup inference; model-level.
- [Interpretable Data-Based Explanations for Fairness Debugging (SIGMOD 2022)](https://dl.acm.org/doi/abs/10.1145/3514221.3517886) — prior art for the cut `propagation/` module.
- [Training data debugging for the fairness of machine learning software (ICSE 2022)](https://dl.acm.org/doi/abs/10.1145/3510003.3510091) — likewise.
- [Beyond "non-significant" results: why and how to test for practical equivalence (PNAS)](https://www.pnas.org/doi/abs/10.1073/pnas.2611548123) — basis for the TOST recommendation.
- [Aequitas — Bias and Fairness Audit Toolkit](https://dssg.github.io/aequitas/) — incumbent tool, no power reporting.
