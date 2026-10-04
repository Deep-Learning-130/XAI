# Detectability

The design rationale for the `detectability/` package. This document was the
top gap recorded in [10_repository_structure.md](10_repository_structure.md)
§5 — the design previously lived inline in the Global Constraints of an
implementation plan. It describes the **SESOI / equivalence** formulation
decided in [plan.md](../plan.md) §3.2, not the earlier MDE-comparison design,
and it matches the code in `src/dbias/detectability/`.

---

## 1. The problem

Every dataset auditing tool in circulation returns the same sentence for two
datasets that have nothing in common:

> No significant disparity found.

It says this for a subgroup of 40 rows and for a subgroup of 400,000. In the
second case the statement is informative. In the first it is nearly vacuous —
a test with almost no power will fail to reject almost regardless of what is
true. Both sentences get pasted into the same compliance document, and nothing
in the output distinguishes them.

This is not a subtle statistical point. It is the oldest one there is:
*absence of evidence is not evidence of absence*. What is notable is that the
remedy — equivalence testing — is decades old and standard in clinical trial
design, and no shipped dataset-auditing tool applies it.

Audit data makes this worse than usual. A dataset being audited was never
designed as a sample. Nobody chose n. Subgroup sizes are wildly uneven by
construction, and the groups an audit cares about most are systematically the
smallest ones. The cells where the tool is least able to see are exactly the
cells someone is relying on it to check.

## 2. What every finding must carry

A finding therefore reports two things, not one:

| Question | Field |
| :--- | :--- |
| What did this sample show? | `effect_size_value`, `effect_size_ci`, `p_value_corrected` |
| What could this sample have shown? | `power_to_detect_sesoi`, `minimum_detectable_effect`, `equivalence_verdict`, `detectability` |

`detectability/` is not optional and not a plugin. Every `Finding` passes
through it before reaching `rules/`, and one that arrives with
`Detectability.UNKNOWN` raises rather than being scored — `rules/severity.py`
enforces this. A degraded mode that silently prints an unannotated null would
reintroduce the exact defect the package exists to remove.

## 3. The SESOI is required configuration

Every verdict here is conditional on one number: the **smallest effect size of
interest**, declared on the Cohen's *w* scale. `--sesoi` has no default and
`annotate_detectability` raises without it.

This is deliberate and it is a design decision, not an ergonomic oversight.
The audit's central claim is *"we could have caught anything that matters"*. A
tool that makes that claim has to make the user define what matters. Defaulting
it to Cohen's 1988 conventions would put a rule of thumb — one Cohen himself
described as a last resort in the absence of domain knowledge — at the centre
of a document that carries the user's name.

The report states the SESOI verbatim, in `configuration.sesoi`, with the scale
it is measured on. A reader who does not know that number cannot read the
report.

## 4. The verdict is an equivalence decision, not an MDE comparison

The obvious objection to this whole reframe is: *a confidence interval on the
effect size already tells you this, and tells you more.* That objection is
correct, and the design concedes it rather than arguing with it.

So the inference layer is the interval. For each finding, a bootstrap
confidence interval on the effect size is compared against the SESOI:

| Interval relative to the SESOI | Verdict | Reading |
| :--- | :--- | :--- |
| entirely below | `EQUIVALENT` | Effects at or above the SESOI are ruled out. An **earned all-clear**. |
| entirely above | `DISPARITY` | The effect is at least as large as the SESOI. |
| straddles it | `INCONCLUSIVE` | The sample cannot decide. A **blind spot**. |

`EQUIVALENT` is the valuable half. It converts an uninterpretable null into a
positive, bounded claim — which is a far stronger thing to be able to say than
"we had enough rows", and it is what a statistician would ask for.

The interval is a percentile bootstrap over the multinomial defined by the
observed cell proportions (`stats/intervals.py`), seeded and reproducible.
Cramér's V is bounded below by zero and biased upward near the null, so the
interval is effectively one-sided in practice: the **upper** bound carries the
inference and the lower bound is reported for completeness.

## 5. MDE is the communication layer

The minimum detectable effect is still computed and still reported. It is what
goes in a coverage-map cell and in the sentence a human reads:

> No disparity was detected, but this test could only have caught effects of
> 0.115 or larger, against a declared SESOI of 0.100. This is not a clean
> result.

That sentence is legible to a practitioner in a way an interval is not. But it
does not decide anything. Verdict comes from §4; MDE communicates it.

## 6. Two signals that can disagree

`detectability` (from the power calculation) and `equivalence_verdict` (from
the interval) answer nearly the same question by different routes, and they
sometimes conflict. **The interval wins**, and `rules/severity.py` implements
that precedence.

The reason: the MDE is a design-stage quantity computed from n, the degrees of
freedom, α and the target power. It knows nothing about the sample that
actually arrived. The interval is computed from that sample. When they
disagree the interval is better evidence in both directions — it can vindicate
a cell the power calculation pessimistically wrote off, and it can withdraw an
adequacy the power calculation promised.

Both are reported, because the disagreement is itself informative. On the
coverage map, colour is the MDE and hatching is the verdict; a cool cell that
is hatched had less power than its sample size promised.

### Power is never computed from the observed effect

There is no `achieved_power` or `statistical_power` field, and no signature in
`detectability/power.py` accepts an observed effect. Post-hoc observed power is
a deterministic monotone function of the p-value and carries no information
beyond it (Hoenig & Heisey 2001, *The Abuse of Power*). The code path was
deleted rather than deprecated, and `tests/unit/models/test_finding.py` asserts
it has not come back.

## 7. The coverage map

The audit emits an attribute × feature grid of minimum detectable effects
alongside the findings feed. Each cell holds the smallest effect that pair
could have detected, the n it was computed on, and whether the verdict was a
blind spot. Untested pairs stay empty — filling them with a default would
reintroduce the false reassurance the map exists to remove.

This is the most communicable artifact in the design. It answers, in one
image, the question a findings list cannot: *which parts of this dataset is
the audit entitled to speak about at all?*

The risk vector is always published with a **coverage vector** beside it, for
the same reason. A risk vector alone cannot distinguish a category that was
checked and found clean from one that could not be checked.

## 8. What is approximate, stated plainly

**The MDE ignores marginal structure.** The analytic non-centrality is taken as
`n · w²`, which treats Cohen's *w* as sufficient for the alternative. Under a
95/5 or 99/1 group split, effective power is governed by the smaller cell
rather than by the total, and this approximation errs **optimistic** — the
dangerous direction for a tool whose purpose is honest null results, and worst
precisely for the small minority subgroups that matter most.

The correct fix is a simulation-based MDE ([plan.md](../plan.md) §3.4). **This is
implemented for 2x2 tables.** For 2x2 tables with skewed margins
(`min(expected) < 5` or margin ratio > 4:1), the pipeline drops to empirical
binomial simulation in `detectability/power.py`. For tables larger than 2x2, the
simulation fallback is not yet implemented; these carry `mde_is_approximate=True`
into the JSON and an asterisk on the figure. The limitation is surfaced
rather than hidden.

**The equivalence test has a type-I error at the boundary.** When the true
effect sits exactly on the SESOI, the interval will sometimes fall entirely
below it and issue an all-clear on a real effect. This is the type-I error of
the equivalence decision and it is bounded by α by construction. Measured at
roughly 2–3% in `tests/calibration/`. The report's claim should therefore be
read as *"effects above the SESOI are ruled out at 95% confidence"*, not
*"ruled out"*.

## 9. Why the evaluation is empirical

Asserting that a 40-row null returns `BLIND_SPOT` proves nothing. The verdict
and the MDE come from the same formula, so such a test can only fail if the
code contradicts itself — it verifies the implementation, not the claim
([plan.md](../plan.md) §3.3).

The claim under test is empirical and can genuinely fail:

> When the tool says a test was adequately powered, the real detection rate at
> the SESOI is at least the target power. When it says the test was
> underpowered, the real detection rate is materially below it.

`tests/calibration/` simulates datasets with known true effects, runs the real
pipeline, and counts what happens. A reduced grid is implemented; the full
experiment is roadmap M3 and is the figure the write-up lives on.

## 10. Relation to prior art

Nothing in the mathematics here is new, and the write-up should not claim
otherwise. Power analysis and MDE are textbook (Cohen 1988). Sample-size
calculation for fairness audits is published
([arXiv:2312.04745](https://arxiv.org/abs/2312.04745)). Simultaneous
multiplicity-corrected subgroup inference for model audits is published
([Cherian & Candès, JMLR 2024](https://jmlr.org/papers/v25/23-0739.html)).
Equivalence testing is decades old. Hierarchical FDR over a hypothesis tree is
Yekutieli (2008).

The contribution is one of framing and systems, and it is narrower than
"detectability-aware auditing":

> Power-based reasoning is standard in study design and is already available
> for *model* fairness audits, but it is absent from *dataset-level,
> pre-training* auditing tooling — where audits run on data that was never
> designed as a sample and subgroup sizes are uneven by construction. We make
> detectability a mandatory field on every finding, define a blind-spot
> verdict as an equivalence decision against a declared SESOI, expose an
> attribute × feature coverage map as a first-class audit artifact, and
> measure empirically whether the verdict is calibrated.

## 11. Open questions

Carried from [handoff.md](../handoff.md) §7; none is resolved by this document.

1. **SESOI default.** There is none, by design. Whether the tool should ship a
   *suggested* value for users with no domain anchor is unresolved.
2. **Continuous sensitive attributes.** Age as a continuous variable has no
   path; everything assumes categorical groups. Current answer: require
   user-declared binning.
3. **Zero-count intersections.** Implemented as `Detectability.EMPTY` — an
   absence rather than a blind spot, since there is nothing to be blind to.
   Whether that is the right treatment in a coverage roll-up is untested.
4. **KS-D thresholds** remain provisional; the continuous path is not built.
5. **FDR family boundary** — resolved here as one family per
   (category, attribute), implemented in `audit.py`. Intersectional descent
   (M7) must nest inside these families rather than redefining them.
