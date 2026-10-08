"""Reduced FDR check: 100 replicates per cell, loose bounds."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fdr_harness import FAMILIES, RACE, XOR, run_fdr_cell, truth  # noqa: E402

pytestmark = pytest.mark.slow


def test_ground_truth_matches_the_construction():
    real = truth({"x": XOR, "y": RACE})
    assert real[("x", "race")] is False and real[("x", "sex")] is False
    assert real[("x", "race_AND_sex")] is True
    assert real[("y", "race")] is True and real[("y", "sex")] is False


HIERARCHICAL_FAILS_UNDER_THE_NULL = pytest.mark.xfail(
    strict=True,
    reason="Hierarchical FDR failed its pre-registered gate: under the global null the "
    "intersection family's FDR is 0.082 (500 audits) against a 0.074 limit. Parent and "
    "child tests are positively dependent, so admitting a child after a chance parent "
    "rejection inflates it. See tests/calibration/results/fdr_calibration_results.csv.",
)


@pytest.mark.parametrize(
    "scenario, fdr",
    [
        ("global_null", "family"),
        pytest.param("global_null", "hierarchical", marks=HIERARCHICAL_FAILS_UNDER_THE_NULL),
        ("xor", "family"),
        ("xor", "hierarchical"),
    ],
)
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
