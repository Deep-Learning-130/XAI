"""Reduced FDR check for the default per-family correction: 100 replicates
per cell, loose bounds. Hierarchical FDR was measured here, failed its gate
and was removed; see fdr_harness.py for where to reproduce it."""
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


@pytest.mark.parametrize("scenario", ["global_null", "xor"])
def test_every_family_stays_near_alpha(scenario):
    row = run_fdr_cell(scenario, n=2000, replicates=100, fdr="family", seed=3)
    for name in FAMILIES:
        assert row[f"fdr_{name}"] <= 0.10, name


def test_audit_wide_fdr_is_reported_not_controlled():
    """Under the global null, two parent families at alpha = 0.05 already give
    roughly a 10% chance of some false discovery. Recorded so the write-up
    cannot claim audit-wide control."""
    row = run_fdr_cell("global_null", n=2000, replicates=100, fdr="family", seed=5)
    assert row["fdr_audit_wide"] >= row["fdr_race"]


def test_family_correction_finds_intersection_only_effects():
    row = run_fdr_cell("xor", n=2000, replicates=50, fdr="family", seed=4)
    assert row["intersection_power"] > 0.8


def test_the_removed_procedure_is_refused_with_a_pointer():
    with pytest.raises(ValueError, match="649da32"):
        run_fdr_cell("xor", n=200, replicates=1, fdr="hierarchical")
