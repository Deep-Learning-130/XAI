# Project Plan — Detectability-Aware Dataset Bias Auditing

**Status:** pre-implementation. No source code exists yet; `src/` is specified but empty.
**Date:** 2026-08-20
**Scope of this document:** an honest assessment of whether the idea is worth building, what specifically is and is not novel about it, the methodological defects that must be fixed before implementation starts, and the decisions that follow.

Companion documents: [roadmap.md](roadmap.md) for sequencing, [handoff.md](handoff.md) for picking the project up cold.

---

## 1. What the project currently is

Two distinct ideas are layered in the existing documents, and they have very different value.

**Layer one — the original framing** (README, `docs/01`–`docs/08`, and both Review documents). A pre-training, dataset-level bias auditor: profile a tabular dataset, run the right hypothesis test per variable pair, compute effect sizes, apply Benjamini–Hochberg, roll findings into a per-category risk vector rather than a single score, emit a Data Card-shaped JSON.

**Layer two — the detectability reframe** (`docs/10_repository_structure.md`, `docs/superpowers/plans/2026-08-20-detectability-core.md`). Every test additionally reports what it *could* have found. A non-significant result is classified as either an earned all-clear or a **blind spot**, and the audit emits a coverage map alongside the findings.

The Review documents assess only layer one. Their conclusion — "primarily an integration and automation contribution" — is correct for what they were assessing, and it is the right instinct to have been conservative. But those documents predate the reframe, and the reframe changes the answer.

---

## 2. Is it a good idea? Verdict

**Layer one, on its own: no.** Not as research. A tool that computes Cramér's V per protected attribute, applies BH-FDR, and prints a Data Card is a weekend of engineering over `scipy.stats` plus a Streamlit front end. AIF360, Fairlearn, Aequitas, Deepchecks, and `ydata-profiling` collectively cover most of it. "There is no unified pipeline" is a true statement that is also true of a thousand unbuilt npm packages; absence of an integration is not a research gap. Build it as infrastructure, do not defend it as a contribution.

**Layer two: yes, with qualifications.** The detectability reframe is a genuine and unusually clean idea, and it has the property good research ideas have — once stated, the absence of it in existing tools looks like an obvious defect rather than a missing feature. Every incumbent auditing tool returns "no significant disparity found" for a 40-row subgroup and a 400,000-row subgroup in exactly the same words. That is an unsafe output, it is the kind of output that gets quoted in a compliance document, and no shipped tool distinguishes the two cases.

**But be precise about what kind of contribution it is.** It is a *framing and systems* contribution, not a *method* contribution. Nothing in the mathematics is new:

- Power analysis and minimum detectable effect are textbook (Cohen, 1988).
- Sample-size calculation specifically for fairness audits is already published: [*A Brief Tutorial on Sample Size Calculations for Fairness Audits*](https://arxiv.org/abs/2312.04745).
- Simultaneous, multiplicity-corrected inference over many subgroups of a model audit is already published and rigorous: [Cherian & Candès, *Statistical Inference for Fairness Auditing*, JMLR 2024](https://jmlr.org/papers/v25/23-0739.html).
- Hierarchical FDR over a tree of hypotheses is Yekutieli (2008), with an R implementation in `structSSI`.
- "Absence of evidence is not evidence of absence" has a standard statistical remedy — equivalence testing / TOST — that is decades old.

So the sentence "we invented detectability-aware auditing" will not survive review. The sentence that will survive is narrower and still worth writing:

> Power-based reasoning is standard in study design and is already available for *model* fairness audits, but it is absent from *dataset-level, pre-training* auditing tooling, where audits are run on data that was never designed as a sample and where subgroup sizes are wildly uneven by construction. We make detectability a mandatory field on every finding, define a blind-spot verdict, and expose an attribute × feature coverage map as a first-class audit artifact — then measure empirically whether the verdict is calibrated.

That is defensible. It is a tools/resources or applied-track paper (FAccT demo track, ECML ADS, JOSS), not a methods paper. Scoped that way it is a strong final-year or masters project. Scoped as "novel algorithm" it will be rejected.

**The single most valuable asset in the whole design is the coverage map** — the attribute × feature grid of minimum detectable effects. It is the artifact a practitioner would screenshot, it is what makes the idea legible in one image, and it is what makes the tool citable. Everything else in the repository is scaffolding around it. Prioritise accordingly.

---

## 3. Methodological defects that must be fixed before coding

These are ordered by severity. Items 1–3 are correctness problems in the current plan, not preferences.

### 3.1 Post-hoc "observed power" is a statistical fallacy — currently baked into Task 7

`detectability/classify.py` in the implementation plan calls:

```python
finding.statistical_power = achieved_power_chi_square(
    effect_size=abs(finding.effect_size_value), n=finding.total_n, df=df, alpha=alpha
)
```

Power computed from the *observed* effect size is a deterministic, monotone function of the p-value. It carries zero information beyond the p-value, and reporting it is a well-documented error (Hoenig & Heisey, 2001, *The Abuse of Power*). A reviewer will find this in the first ten minutes.

**Fix:** power must always be computed against a **pre-specified** effect — the smallest effect size of interest (SESOI) — never against what was observed. Report `power_to_detect_sesoi`, not `achieved_power`. Delete the observed-effect code path entirely rather than leaving it as an option.

### 3.2 MDE is a weaker summary than a confidence interval; the blind-spot verdict should be an equivalence test

The reviewer's obvious objection to the whole reframe is: *"a confidence interval on the effect size already tells you this, and tells you more."* That objection is correct on the statistics. A CI whose upper bound sits below the SESOI *is* an earned all-clear; a CI whose upper bound sits above it *is* a blind spot; and the CI additionally shows where the point estimate sits.

**Fix — and this strengthens the project rather than weakening it:**

- Compute a confidence interval on every effect size (bootstrap where no closed form exists).
- Define the blind-spot verdict as a **two-one-sided-tests (TOST) equivalence decision** against the SESOI, not as an MDE comparison. `ADEQUATE + null` becomes a positive claim ("we can rule out effects above the SESOI at 95%"), which is far stronger than "we had enough rows."
- Keep MDE as the *communication layer*, not the inference layer. MDE is what goes in the coverage map cell and in the sentence a human reads; TOST is what decides the verdict. This is honest and it is what a statistician would ask for.

This turns the headline claim from "we noticed power exists" into "we replace an uninterpretable null with a bounded equivalence claim, at scale, across an intersection lattice." That is a materially better paper.

### 3.3 The evaluation as designed is circular

Planned Scenario A-underpowered asserts that a 40-row null returns `BLIND_SPOT`. But the verdict and the MDE are computed from the same closed-form formula, so this test can only fail if the code contradicts itself. It verifies the implementation, not the claim.

**Fix:** the scientific evaluation must be **empirical calibration**, not formula restatement:

1. For a grid of (n, true effect size, group imbalance, table shape), simulate 1,000+ datasets each.
2. Measure the *actual* detection rate of the pipeline.
3. Plot empirical power against the analytic MDE prediction.
4. The claim under test is: **when the tool says ADEQUATE, the empirical detection rate at the SESOI is ≥ 0.80; when it says BLIND SPOT, it is materially below.**

That is a real experiment with a real possible failure, and it is the figure the paper lives or dies on. It will very likely reveal that the analytic MDE is optimistic under skewed margins (see 3.4) — which is a finding, not a setback.

### 3.4 The chi-square MDE ignores the marginal structure

`mde_chi_square(n=total_n, df=df)` treats Cohen's *w* over total *n* as sufficient. For a 95/5 group split, effective power is governed by the smaller cell, not the total. The audit's most important cases — small minority subgroups — are exactly where this approximation is worst, and it errs *optimistic*, which is the dangerous direction for a tool whose purpose is honest null results.

**Fix:** condition the non-centrality on the observed margins, or fall back to simulation-based MDE when `min(expected_count) < 5` or when the group ratio exceeds ~4:1. Document the approximation in the module docstring.

### 3.5 `mde_rank_biserial` uses the two largest groups

`counts = sorted(..., reverse=True)` then `n1=counts[0], n2=counts[1]` is correct for two groups and silently wrong for k > 2 — and it picks the *best-powered* pair, again erring optimistic. Route k > 2 to a Kruskal–Wallis / eta-squared power path, or raise.

### 3.6 The adequacy rule depends on Cohen's rules of thumb

`ADEQUATE iff MDE ≤ medium threshold` puts an arbitrary 1988 convention at the centre of the project's novel claim. Cohen himself described these cutoffs as a last resort when domain knowledge is absent.

**Fix:** SESOI is a **required, user-supplied configuration value** with a documented default. The audit report must state, verbatim, which SESOI was used. A tool whose central verdict is "we could have caught anything that matters" must define "matters" out loud.

### 3.7 Naming and citation hygiene

- `docs/03 §2` calls the NaN-mask test MNAR. Testing missingness against an *observed* attribute detects **MAR**; true MNAR is not identifiable from observed data. Already noted in `docs/10 §5` — fix it before it propagates into class names (`MissingnessAnalyzer` is fine; `MNAR_INCOME_GENDER` as a finding ID is not).
- The Cleanlab limitation claim is cited to a vendor-comparison blog post. Replace with the primary source (Northcutt, Jiang & Chuang, *Confident Learning*, JAIR 2021).
- Verify every 2026-dated citation resolves before it appears in a submission.

---

## 4. Scope decisions

The current documents describe roughly three projects' worth of work. These cuts are the difference between finishing and not.

| Component | Decision | Reasoning |
| :--- | :--- | :--- |
| `models/`, `stats/`, `detectability/` | **Build first, build well** | This is the contribution. Nothing else matters if this is not right. |
| Coverage map (attribute × feature MDE grid) | **Promote to first-class MVP deliverable** | The single most communicable artifact; currently buried as a dashboard component. |
| Empirical calibration harness | **Promote to MVP** | Without it there is no scientific claim, only code. |
| Analyzers (representation, missingness, feature, label) | **Build, minimally** | Necessary plumbing. Do not gold-plate. |
| Rules engine + risk/coverage vector | **Build** | Small, and it is where `BLIND_SPOT` becomes visible. |
| Gated hierarchical FDR over intersections | **Build, but reduced** | Genuinely interesting; power-gated descent is a real design idea. Cap lattice depth at 2 for the MVP. |
| Streamlit dashboard | **Defer to V1, after the paper figure exists** | A static coverage-map plot delivers 90% of the value at 5% of the cost. |
| `propagation/` (counterfactual repair → downstream fairness delta) | **Cut from this project** | Heavy prior art — [Gopher / *Interpretable Data-Based Explanations for Fairness Debugging*, SIGMOD 2022](https://dl.acm.org/doi/abs/10.1145/3514221.3517886), [training-data debugging for fairness, ICSE 2022](https://dl.acm.org/doi/abs/10.1145/3510003.3510091), FairIF. It requires training models, which contradicts the pre-training premise, and it doubles the surface area. Keep as a "future work" paragraph. |
| MMD-Critic prototypes/criticisms | **Cut from MVP** | Review 1 already deferred it. Correct call. It is a separate visual project. |
| LLM narrative generation | **Defer to last** | Both Reviews correctly constrain the LLM to consuming structured findings only. Nothing depends on it. Ship the JSON first. |
| PDF export | **Cut** | JSON + a plot is enough. |

---

## 5. Architectural decisions that stand

These are already right in `docs/10` and should not be renegotiated:

1. **Layered dependencies, imports downward only.** `stats/` never learns what a protected attribute is. This is what keeps the statistical core independently testable and independently citable.
2. **Severity exists only in `rules/`.** Analyzers emit `UNDETERMINED`.
3. **No aggregate bias score.** Per-category risk vector plus coverage vector. Both Reviews reached this independently; hold the line under pressure to produce a dashboard number.
4. **`detectability/` is a separate package from `stats/`.** They answer opposite questions and the second is the differentiator.
5. **Test oracles are independent** — hand-computed closed forms or statsmodels, never the implementation's own output.

One addition: **`detectability/` must be non-optional at runtime.** A `Finding` reaching the report layer with `detectability == UNKNOWN` is a bug that should raise, not a degraded mode that silently prints.

---

## 6. What "done" means

The project is complete, and the claim is defensible, when all of the following hold:

- [ ] Running the auditor on a dataset produces, for every (attribute, feature) cell, an effect size with a confidence interval, a TOST equivalence verdict against a declared SESOI, and an MDE.
- [ ] A coverage map figure exists showing which cells the dataset can and cannot speak to.
- [ ] The calibration experiment shows empirical detection rate ≥ 0.80 in cells the tool calls ADEQUATE, across at least three group-imbalance regimes.
- [ ] Scenario C (n = 10⁶, difference = 0.01) returns `INFORMATIONAL`, not a finding.
- [ ] Running on Adult and COMPAS produces a coverage map with at least one cell that incumbent tools would have silently reported as clean.
- [ ] That last bullet has a screenshot in the write-up. It is the whole argument in one image.

---

## 7. Honest risk register

| Risk | Likelihood | Mitigation |
| :--- | :--- | :--- |
| Reviewer says "this is just a confidence interval" | **High** | Pre-empt it: implement TOST as the inference layer (§3.2) and say so in the abstract. Frame MDE as communication, not inference. |
| Calibration experiment shows the analytic MDE is badly miscalibrated under skew | Medium | This is a *result*, not a failure. Report it and switch to simulation-based MDE. |
| Scope creep back into propagation / dashboard / LLM | **High** | The cuts in §4 are decisions, not suggestions. Revisit only after §6 is fully checked. |
| Someone publishes the same reframe first | Low–Medium | The nearest neighbours are model-level. Do not sit on it for a year. |
| Effect-size thresholds bikeshedding consumes weeks | Medium | Make SESOI configuration, document the default, move on. |

---

## 8. Immediate next actions

1. Write `docs/09_detectability.md`. The design currently lives inline in a plan file's Global Constraints — that is fragile, and `docs/10 §5` already flags it. It should state the SESOI/TOST decision from §3.2 as the design, not the MDE-comparison version.
2. Amend `docs/superpowers/plans/2026-08-20-detectability-core.md` Task 7 to remove observed power (§3.1) and to add the CI/TOST path (§3.2). **Do not execute Task 7 as currently written.**
3. Tasks 1–6 of that plan are sound and can be executed as-is. Start there.
4. Revise `docs/02` and `docs/05` for the threshold-registry override and add a `BLIND_SPOT` row to the severity table.
5. Rename MNAR → MAR throughout (§3.7).
