"""Plot post-propagation spectral-band contributions for three datasets."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


BANDS = ("1–8", "9–16", "17–32", "33–64")

DATA = {
    "Baby": {
        "original": (0.411645, 0.154208, 0.177321, 0.173552),
        "masked": (0.320079, 0.165058, 0.203709, 0.209000),
    },
    "Sports": {
        "original": (0.298103, 0.148938, 0.199837, 0.222461),
        "masked": (0.241536, 0.156036, 0.215467, 0.242613),
    },
    "Clothing": {
        "original": (0.107030, 0.083289, 0.149118, 0.267510),
        "masked": (0.103313, 0.083097, 0.151405, 0.270104),
    },
}

COLORS = {
    "Baby": "#0072B2",
    "Sports": "#D55E00",
    "Clothing": "#009E73",
}


def create_figure():
    """Create the comparison figure from the fixed experiment values."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    positions = np.arange(len(BANDS))
    figure, (contribution_axis, delta_axis) = plt.subplots(
        1,
        2,
        figsize=(14.5, 6.2),
        gridspec_kw={"width_ratios": (1.55, 1.0)},
        constrained_layout=True,
    )

    for dataset, views in DATA.items():
        color = COLORS[dataset]
        original = 100.0 * np.asarray(views["original"])
        masked = 100.0 * np.asarray(views["masked"])
        delta = masked - original

        contribution_axis.plot(
            positions,
            original,
            color=color,
            linestyle="--",
            linewidth=2.0,
            marker="o",
            markersize=7,
            markerfacecolor="white",
            markeredgewidth=1.8,
        )
        contribution_axis.plot(
            positions,
            masked,
            color=color,
            linestyle="-",
            linewidth=2.4,
            marker="s",
            markersize=6.5,
        )

        delta_axis.plot(
            positions,
            delta,
            color=color,
            linewidth=2.4,
            marker="o",
            markersize=7,
            label=dataset,
        )
        for x_position, value in zip(positions, delta):
            vertical_offset = 7 if value >= 0 else -14
            delta_axis.annotate(
                f"{value:+.2f}",
                (x_position, value),
                xytext=(0, vertical_offset),
                textcoords="offset points",
                ha="center",
                va="bottom" if value >= 0 else "top",
                color=color,
                fontsize=9,
                fontweight="bold",
            )

    contribution_axis.set_title("Original vs. masked spectral contribution")
    contribution_axis.set_xlabel("Singular-value band")
    contribution_axis.set_ylabel("Global spectral-energy contribution (%)")
    contribution_axis.set_xticks(positions, BANDS)
    contribution_axis.set_ylim(0.0, 45.0)
    contribution_axis.grid(axis="y", linestyle=":", alpha=0.45)

    dataset_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[dataset],
            linewidth=2.5,
            label=dataset,
        )
        for dataset in DATA
    ]
    view_handles = [
        Line2D(
            [0],
            [0],
            color="#333333",
            linestyle="--",
            marker="o",
            markerfacecolor="white",
            linewidth=2.0,
            label="Original view",
        ),
        Line2D(
            [0],
            [0],
            color="#333333",
            linestyle="-",
            marker="s",
            linewidth=2.4,
            label="Masked view",
        ),
    ]
    dataset_legend = contribution_axis.legend(
        handles=dataset_handles,
        title="Dataset",
        loc="upper right",
        frameon=False,
    )
    contribution_axis.add_artist(dataset_legend)
    contribution_axis.legend(
        handles=view_handles,
        title="Graph view",
        loc="center right",
        frameon=False,
    )

    delta_axis.axhline(0.0, color="#555555", linewidth=1.2, linestyle="--")
    delta_axis.set_title("Change introduced by masking")
    delta_axis.set_xlabel("Singular-value band")
    delta_axis.set_ylabel("Masked − Original (percentage points)")
    delta_axis.set_xticks(positions, BANDS)
    delta_axis.set_ylim(-10.5, 5.2)
    delta_axis.grid(axis="y", linestyle=":", alpha=0.45)
    delta_axis.legend(title="Dataset", loc="lower right", frameon=False)

    figure.suptitle(
        "Post-propagation User Representations: Spectral-band Comparison",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5,
        -0.02,
        "Values are normalized by each representation's total Frobenius "
        "energy; spectral bands have unequal rank widths.",
        ha="center",
        fontsize=9.5,
        color="#555555",
    )
    return figure


def main():
    output_directory = Path(__file__).resolve().parent
    output_stem = output_directory / "spectral_band_comparison"
    figure = create_figure()
    figure.savefig(output_stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)
    print("Saved {} and {}".format(
        output_stem.with_suffix(".png"),
        output_stem.with_suffix(".svg"),
    ))


if __name__ == "__main__":
    main()
