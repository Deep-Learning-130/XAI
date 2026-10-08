"""Reduced interval calibration: two cells, 200 replicates. Loose bounds --
at 200 replicates a rate near 0.05 has a standard error near 0.015."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from interval_harness import joint_probs, run_interval_cell  # noqa: E402
from parallel import run_cells  # noqa: E402

pytestmark = pytest.mark.slow


def test_construction_hits_the_requested_effect():
    from dbias.stats.effect_sizes import cohens_w

    p = joint_probs(np.full(5, 0.2), [0.7, 0.3], 0.2)
    assert cohens_w(p * 1e6) == pytest.approx(0.2, abs=1e-6)
    assert np.allclose(p.sum(axis=1), 0.2) and np.allclose(p.sum(axis=0), [0.7, 0.3])


def test_sparse_null_produces_no_spurious_disparities():
    row = run_interval_cell(np.full(16, 1 / 16), [0.5, 0.5], 500, 0.0, replicates=200, seed=1)
    assert row["spurious_disparity_rate"] <= 0.08


def test_effect_at_the_sesoi_is_not_cleared():
    row = run_interval_cell(np.full(5, 0.2), np.full(5, 0.2), 500, 0.1, replicates=200, seed=2)
    assert row["equivalent_rate"] <= 0.08


def test_parallel_cells_reproduce_the_serial_run_exactly():
    """Every cell carries its own seed, so splitting the grid across
    processes must not change a single number."""
    cells = [
        dict(row_shares=np.full(2, 0.5), col_shares=np.full(2, 0.5), n=200, true_w=0.1,
             replicates=20, n_resamples=50, seed=3),
        dict(row_shares=np.full(5, 0.2), col_shares=[0.7, 0.3], n=300, true_w=0.0,
             replicates=20, n_resamples=50, seed=4),
    ]
    serial = run_cells(run_interval_cell, cells, jobs=1)
    parallel = run_cells(run_interval_cell, cells, jobs=2)
    assert parallel == serial
