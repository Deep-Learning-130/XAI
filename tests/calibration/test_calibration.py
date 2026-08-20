"""Is the detectability verdict calibrated?

A reduced version of the roadmap M3 experiment: fewer replicates, fewer grid
points, 2x2 tables only. It runs in minutes and is a smoke test. **Passing it
does not pass the M3 gate**, which needs 1,000+ replicates per cell across
imbalance regimes and table shapes.

    pytest tests/calibration -v          # everything, a few minutes
    pytest -m "not slow"                 # skip it

Thresholds here are loose on purpose. At ~150 replicates a rate has a standard
error near 0.04, so asserting 0.80 to two decimal places would produce a test
that fails for arithmetic reasons rather than scientific ones.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from harness import Cell, rates_for, run_cell  # noqa: E402

pytestmark = pytest.mark.slow

SESOI = 0.1
REPLICATES = 150


@pytest.fixture(scope="module")
def adequate_at_sesoi():
    """n = 900, balanced, true effect exactly at the SESOI. The tool calls
    this ADEQUATE, and the analytic power for it is 0.851."""
    return run_cell(
        Cell(n=900, true_w=SESOI, minority_share=0.5),
        replicates=REPLICATES, n_resamples=100, seed=5,
    )


@pytest.fixture(scope="module")
def underpowered_at_sesoi():
    """The same effect at n = 100. The tool calls this UNDERPOWERED."""
    return run_cell(
        Cell(n=100, true_w=SESOI, minority_share=0.5),
        replicates=REPLICATES, n_resamples=100, seed=5,
    )


@pytest.fixture(scope="module")
def under_the_null():
    return run_cell(
        Cell(n=900, true_w=0.0, minority_share=0.5),
        replicates=REPLICATES, n_resamples=100, seed=5,
    )


# --- the gate criteria, in reduced form --------------------------------------

def test_adequate_cells_really_do_detect_effects_at_the_sesoi(adequate_at_sesoi):
    """Gate criterion 1. When the tool claims it could have seen an effect at
    the SESOI, it must actually see one at roughly the target rate."""
    assert adequate_at_sesoi["adequate_rate"] == 1.0
    assert adequate_at_sesoi["detection_rate"] >= 0.75


def test_empirical_power_tracks_the_analytic_prediction(adequate_at_sesoi):
    """The analytic power for this cell is 0.851. Empirical should land near
    it -- this is the comparison the M3 calibration figure plots."""
    assert adequate_at_sesoi["detection_rate"] == pytest.approx(0.851, abs=0.10)


def test_underpowered_cells_really_do_miss(underpowered_at_sesoi):
    """Gate criterion 2. The blind-spot label must not be pessimism theatre --
    detection has to be materially below target, not just a shade under."""
    assert underpowered_at_sesoi["adequate_rate"] == 0.0
    assert underpowered_at_sesoi["detection_rate"] < 0.50


def test_the_two_labels_separate_cleanly(adequate_at_sesoi, underpowered_at_sesoi):
    """The verdict has to carry information: cells labelled ADEQUATE must
    detect far more often than cells labelled UNDERPOWERED."""
    gap = adequate_at_sesoi["detection_rate"] - underpowered_at_sesoi["detection_rate"]
    assert gap > 0.4


def test_false_positive_rate_under_the_null_is_near_alpha(under_the_null):
    """Gate criterion 3, reduced. With one hypothesis per replicate there is
    no multiplicity to correct, so this checks the underlying test rather than
    BH itself."""
    assert under_the_null["detection_rate"] <= 0.12


def test_underpowered_nulls_are_reported_as_blind_spots(underpowered_at_sesoi):
    assert underpowered_at_sesoi["blind_spot_rate"] > 0.5


# --- the failure mode that would matter most ---------------------------------

def test_a_clearly_supra_sesoi_effect_is_never_falsely_cleared():
    """The worst possible output: a real disparity the user cares about,
    missed, and reported as an earned all-clear. For an effect comfortably
    above the SESOI this must not happen at all."""
    cell = run_cell(
        Cell(n=900, true_w=0.15, minority_share=0.5),
        replicates=REPLICATES, n_resamples=100, seed=5,
    )
    assert cell["false_all_clears"] == 0


def test_false_all_clears_at_the_sesoi_boundary_stay_within_the_tost_error_rate(
    adequate_at_sesoi,
):
    """A measured limitation, not a bug.

    When the true effect sits exactly *on* the declared SESOI, the equivalence
    test will sometimes place the whole interval below it and issue an
    all-clear. That is the type-I error of the equivalence decision and it is
    bounded by alpha by construction. The number is recorded here so it is
    known rather than discovered by a reviewer, and so the audit report's
    claim can be read as 'effects above the SESOI are ruled out at 95%'
    rather than 'ruled out'.
    """
    rate = adequate_at_sesoi["false_all_clears"] / adequate_at_sesoi["replicates"]
    assert rate <= 0.08


# --- the construction itself -------------------------------------------------

def test_extreme_imbalance_makes_large_effects_unreachable():
    """A 99/1 split cannot express a large Cohen's w at a 0.5 base rate: the
    implied group rates leave [0, 1]. The harness reports that rather than
    clipping, which would silently weaken the effect it claims to inject."""
    assert rates_for(true_w=0.5, minority_share=0.01) is None
    assert rates_for(true_w=0.1, minority_share=0.5) is not None
