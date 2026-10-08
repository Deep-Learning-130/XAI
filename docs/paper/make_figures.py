"""Regenerate the write-up's figures from committed data and seeded runs.

    python docs/paper/make_figures.py

Figure 1 reads tests/calibration/results/m3_calibration_results.csv (no
simulation is rerun). Figure 2 runs the seeded N = 500 Adult audit exactly as
run_subsampled.py does and draws it with the tool's own coverage-map renderer.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent / "figures"

# Reference categorical palette, slots 1-4 in fixed order (validated on the
# light surface; aqua and yellow sit below 3:1, so every series also has its
# own marker, a shared legend, and its values in the write-up's table).
SURFACE = "#fcfcfb"
INK, INK_MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5e0"
SERIES = {0.50: ("#2a78d6", "o"), 0.20: ("#eb6834", "s"), 0.05: ("#1baf7a", "^"), 0.01: ("#eda100", "D")}


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK, labelsize=9)
    ax.set_xlabel("dataset size n (log scale)", color=INK, fontsize=10)


def _series(ax, df: pd.DataFrame, value: str) -> None:
    for share, (color, marker) in SERIES.items():
        rows = df[df.minority_share == share].sort_values("n")
        if rows.empty:
            continue
        ax.plot(rows.n, rows[value], color=color, linewidth=2, marker=marker,
                markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5,
                label=f"{share:.0%} minority group")


def figure_1() -> Path:
    df = pd.read_csv(ROOT / "tests/calibration/results/m3_calibration_results.csv")
    at_sesoi = df[df.true_w == 0.1]
    null = df[df.true_w == 0.0]

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    for ax in (left, right):
        _style(ax)

    # (a) A real effect exactly at the SESOI: does "ADEQUATE" mean it gets found?
    claimed = at_sesoi.groupby("n").adequate_rate.max()
    adequate_from = claimed[claimed >= 0.999].index.min()
    left.axvspan(adequate_from * 0.8, at_sesoi.n.max() * 1.25, color="#e8f0fb", zorder=0)
    left.text(adequate_from * 0.85, 0.06, "tool reports ADEQUATE power", fontsize=8.5, color=INK_MUTED)
    left.axhline(0.80, color=INK_MUTED, linewidth=1, linestyle=(0, (4, 3)))
    left.text(at_sesoi.n.min(), 0.82, "target power 0.80", fontsize=8.5, color=INK_MUTED)
    _series(left, at_sesoi, "detection_rate")
    left.set_ylim(0, 1.05)
    left.set_xlim(at_sesoi.n.min() * 0.8, at_sesoi.n.max() * 1.25)
    left.set_ylabel("share of datasets where the effect was detected", color=INK, fontsize=10)
    left.set_title("(a) True effect equal to the SESOI (w = 0.1)", loc="left", fontsize=11, color=INK)

    # (b) No effect at all: does the test keep its advertised error rate?
    right.axhline(0.05, color=INK_MUTED, linewidth=1, linestyle=(0, (4, 3)))
    right.text(null.n.max() * 3.2, 0.0515, "alpha = 0.05", fontsize=8.5, color=INK_MUTED, ha="right")
    _series(right, null, "detection_rate")
    right.set_ylim(0, 0.10)
    right.set_xlim(null.n.min() * 0.8, null.n.max() * 3.5)
    right.set_ylabel("false positive rate", color=INK, fontsize=10)
    right.set_title("(b) No effect (w = 0)", loc="left", fontsize=11, color=INK)

    fig.suptitle("Figure 1. Calibration on 85,000 simulated 2x2 datasets (1,000 per point)",
                 x=0.01, ha="left", fontsize=12, color=INK)
    handles, labels = right.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.93), ncol=4,
               frameon=False, fontsize=9, labelcolor=INK, handlelength=2.5)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "fig1_calibration.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


def figure_2() -> Path:
    sys.path.insert(0, str(ROOT / "src"))
    from dbias.audit import audit
    from dbias.ingestion.loaders import fetch_adult
    from dbias.report.coverage_plot import plot_coverage_map

    df = fetch_adult().sample(n=500, random_state=42).copy()
    result = audit(df, ["sex", "race"], "income", sesoi=0.1, seed=0)
    return plot_coverage_map(
        result, OUT / "fig2_coverage_map.png",
        title="Figure 2. Coverage map, Adult subsampled to N = 500 (SESOI w = 0.1)",
    )


if __name__ == "__main__":
    print(figure_1())
    print(figure_2())
