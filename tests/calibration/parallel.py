"""Run independent calibration cells across processes.

Every cell in the gates is seeded on its own, so the grid can be split over
cores without changing a single number: `run_cells(fn, cells, jobs=8)` returns
exactly what `[fn(**c) for c in cells]` would, in the same order. The cell
function must live at module level so worker processes can import it.
"""
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor


def _call(job: tuple[Callable[..., dict], dict]) -> dict:
    fn, kwargs = job
    return fn(**kwargs)


def run_cells(fn: Callable[..., dict], cells: list[dict], jobs: int = 1) -> list[dict]:
    if jobs <= 1:
        return [fn(**kwargs) for kwargs in cells]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(_call, [(fn, kwargs) for kwargs in cells]))
