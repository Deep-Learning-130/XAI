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

FAMILIES = ("race", "sex", "race_AND_sex")


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
