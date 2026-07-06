from __future__ import annotations

import matplotlib.pyplot as plt

from src.config import THEME


def apply_matplotlib_theme() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": THEME["background_light"],
            "axes.facecolor": THEME["card"],
            "axes.edgecolor": THEME["secondary_surface"],
            "axes.labelcolor": THEME["text_primary"],
            "xtick.color": THEME["text_secondary"],
            "ytick.color": THEME["text_secondary"],
            "text.color": THEME["text_primary"],
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.titlepad": 14,
            "grid.color": "#D4E0EB",
            "grid.alpha": 0.65,
            "savefig.facecolor": THEME["background_light"],
            "savefig.bbox": "tight",
        }
    )


def polish_axis(ax, title: str, xlabel: str = "", ylabel: str = ""):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="-", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax