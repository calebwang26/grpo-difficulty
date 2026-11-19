"""Shared look for every figure, so all plots read as one set."""

import matplotlib

matplotlib.use("Agg")  # draw to files, never open a window
import matplotlib.pyplot as plt

# Chart chrome
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# One fixed colour per series. A colour belongs to the series, never to its rank.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"

BIN_COLORS = {"hard": BLUE, "sweet": ORANGE, "easy": AQUA, "mixed": INK_MUTED}
BIN_LABELS = {
    "hard": "Too hard (<10%)",
    "sweet": "Sweet spot (20–80%)",
    "easy": "Too easy (>90%)",
    "mixed": "Random mix (all tasks)",
}


def apply_style():
    """Set matplotlib defaults: quiet axes, hairline grid, 2px lines."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "legend.frameon": False,
            "legend.fontsize": 9,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
        }
    )
