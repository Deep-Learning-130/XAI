"""The figure: an attribute x feature grid of minimum detectable effects.

Static matplotlib, deliberately. A Streamlit dashboard delivers a fraction of
the value at many times the cost, and this image is the one artifact that
makes the idea legible without reading anything (plan.md Sec 4).

Reading the figure: each cell is the smallest effect that pair could have
detected, on the Cohen's w scale the SESOI is declared on -- converting from
Cramer's V per table shape, so one SESOI line serves every cell. Cells at or below the declared SESOI are shaded cool -- the audit
was entitled to speak about them. Cells above it are shaded warm and hatched:
whatever the p-value said, that pair could not have caught an effect the user
declared they care about. A hatched cell with no finding is not a clean cell.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this writes files, it never opens a window

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from dbias.audit import AuditResult  # noqa: E402
from dbias.detectability.coverage import coverage_map  # noqa: E402
from dbias.models.enums import Severity  # noqa: E402


def plot_coverage_map(
    result: AuditResult, path: str | Path, title: str | None = None
) -> Path:
    grid = coverage_map(result.findings)
    if not grid.attributes or not grid.features:
        raise ValueError("nothing to plot: the audit produced no testable cells")

    matrix = grid.matrix()
    sesoi = result.sesoi

    finite = matrix[np.isfinite(matrix)]
    vmax = max(float(finite.max()) if finite.size else sesoi, sesoi * 1.6)
    vmin = min(float(finite.min()) if finite.size else 0.0, sesoi * 0.5)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=sesoi, vmax=vmax)

    height = 3.0 + 0.80 * len(grid.attributes)
    width = 3.4 + 1.6 * len(grid.features)
    fig, ax = plt.subplots(figsize=(width, height))

    image = ax.imshow(matrix, cmap="RdYlBu_r", norm=norm, aspect="auto")

    ax.set_xticks(range(len(grid.features)), grid.features, rotation=18, ha="right")
    ax.set_yticks(range(len(grid.attributes)), grid.attributes)
    ax.set_xlabel("feature audited")
    ax.set_ylabel("declared sensitive attribute")
    ax.set_xticks(np.arange(-0.5, len(grid.features), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(grid.attributes), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)

    for row, attribute in enumerate(grid.attributes):
        for col, feature in enumerate(grid.features):
            cell = grid.cell(attribute, feature)
            if cell is None or cell.mde_w is None:
                ax.text(col, row, "not\ntested", ha="center", va="center",
                        fontsize=7.5, color="0.45", style="italic")
                continue

            blind = cell.severity is Severity.BLIND_SPOT
            if blind:
                # Hatch the border region only, so the number stays readable.
                ax.add_patch(
                    plt.Rectangle(
                        (col - 0.5, row - 0.5), 1, 1,
                        fill=False, hatch="////", edgecolor="black", linewidth=1.8,
                    )
                )
            label = f"{cell.mde_w:.3f}{'*' if cell.approximate else ''}"
            ax.text(
                col, row + 0.06, label, ha="center", va="center",
                fontsize=10.5, fontweight="bold" if blind else "normal", color="black",
                bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                          alpha=0.86, edgecolor="none"),
            )
            ax.text(
                col, row + 0.34, f"n={cell.n}", ha="center", va="center",
                fontsize=7.5, color="0.15",
                bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                          alpha=0.86, edgecolor="none"),
            )

    bar = fig.colorbar(image, ax=ax, shrink=0.9, pad=0.02)
    bar.set_label("minimum detectable effect (Cohen's w)")
    bar.ax.axhline(sesoi, color="black", linewidth=1.8)
    bar.ax.text(0.5, sesoi, "SESOI", va="bottom", ha="center", fontsize=7.5,
                fontweight="bold", transform=bar.ax.get_yaxis_transform())

    ax.set_title(
        title
        or f"Coverage map: what this audit could have seen (SESOI = {sesoi:g})",
        fontsize=12, pad=12,
    )
    fig.legend(
        handles=[
            Patch(facecolor="white", edgecolor="black", hatch="////",
                  label="hatched = blind spot: the realised interval did not rule out "
                        "an effect at the SESOI"),
            Patch(facecolor="white", edgecolor="white",
                  label="colour = the MDE this cell was predicted to reach. A cool cell "
                        "that is hatched had less"),
            Patch(facecolor="white", edgecolor="white",
                  label="             power than its sample size promised."),
            Patch(facecolor="white", edgecolor="white",
                  label="*  MDE approximate (skewed or sparse margins; errs optimistic)"),
        ],
        loc="lower center", bbox_to_anchor=(0.5, 0.0), fontsize=8, frameon=False,
    )
    fig.subplots_adjust(bottom=0.36 if len(grid.attributes) < 4 else 0.28, top=0.88)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path
