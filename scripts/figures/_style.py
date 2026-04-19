"""Publication-quality matplotlib + seaborn styling for BayesBreak figures.

Every figure script calls :func:`setup_style` once at the top. This module
enforces consistent typography, colour palette, line weights, and figure sizes
across all figures shipped with the report.

The palette is a colour-blind-safe variation of Paul Tol's bright palette.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="fontTools")
warnings.filterwarnings("ignore", message=".*timestamp.*")
warnings.filterwarnings("ignore", message=".*extra bytes.*")
logging.getLogger("fontTools").setLevel(logging.ERROR)

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

# Column widths (inches) following a ~170 mm text width.
SINGLE_COL_WIDTH = 3.5
DOUBLE_COL_WIDTH = 7.0
FULL_PAGE_WIDTH = 7.0

DPI_SCREEN = 150
DPI_PRINT = 300


# Paul Tol's "bright" palette, colour-blind safe.
_BRIGHT_PALETTE = [
    "#4477AA",  # blue
    "#EE6677",  # red
    "#228833",  # green
    "#CCBB44",  # yellow
    "#66CCEE",  # cyan
    "#AA3377",  # purple
    "#BBBBBB",  # grey
]

COLORS = {
    "blue": _BRIGHT_PALETTE[0],
    "red": _BRIGHT_PALETTE[1],
    "green": _BRIGHT_PALETTE[2],
    "yellow": _BRIGHT_PALETTE[3],
    "cyan": _BRIGHT_PALETTE[4],
    "purple": _BRIGHT_PALETTE[5],
    "grey": _BRIGHT_PALETTE[6],
    "black": "#222222",
    "lightgrey": "#DDDDDD",
    "orange": "#EE7733",
}

COLOR_CYCLE = _BRIGHT_PALETTE[:6]


def setup_style(
    *,
    font_scale: float = 1.0,
    use_tex: bool = False,
    context: str = "paper",
    despine: bool = True,
) -> None:
    """Set matplotlib + seaborn defaults for publication figures.

    Parameters
    ----------
    font_scale : float
        Multiplier applied to all font sizes.
    use_tex : bool
        If True, render text via LaTeX (slower but crisper math).
    context : str
        Seaborn context ("paper", "notebook", "talk", "poster").
    despine : bool
        Remove the top and right spines by default (seaborn-style).
    """

    sns.set_theme(
        style="ticks",
        context=context,
        palette=_BRIGHT_PALETTE,
        font_scale=font_scale,
        rc={
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        },
    )

    base = 9.5 * font_scale
    params: dict = {
        "figure.dpi": DPI_SCREEN,
        "figure.facecolor": "white",
        "figure.constrained_layout.use": True,
        "font.size": base,
        "axes.labelsize": base + 0.5,
        "axes.titlesize": base + 1.5,
        "axes.titleweight": "medium",
        "axes.labelweight": "regular",
        "axes.linewidth": 0.9,
        "axes.edgecolor": "#333333",
        "axes.spines.top": not despine,
        "axes.spines.right": not despine,
        "axes.prop_cycle": mpl.cycler(color=COLOR_CYCLE),
        "xtick.labelsize": base - 0.5,
        "ytick.labelsize": base - 0.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "lines.linewidth": 1.5,
        "lines.markersize": 4.5,
        "legend.fontsize": base - 0.5,
        "legend.frameon": False,
        "legend.borderaxespad": 0.3,
        "legend.handlelength": 1.5,
        "savefig.dpi": DPI_PRINT,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "savefig.transparent": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "pdf.compression": 9,
    }
    if use_tex:
        params["text.usetex"] = True
        params["text.latex.preamble"] = r"\usepackage{amsmath}\usepackage{amssymb}"
        params["font.family"] = "serif"

    mpl.rcParams.update(params)


def get_figsize(
    width: str | float = "single",
    aspect: float = 0.62,
    nrows: int = 1,
    ncols: int = 1,
) -> tuple[float, float]:
    """Compute (width, height) in inches for a grid of subplots."""

    if isinstance(width, str):
        w = {"single": SINGLE_COL_WIDTH, "double": DOUBLE_COL_WIDTH, "full": FULL_PAGE_WIDTH}[width]
    else:
        w = float(width)
    h = w * aspect * (nrows / ncols)
    return (w, h)


def add_panel_label(
    ax,
    label: str,
    title: str | None = None,
    *,
    fontsize: float | None = None,
    offset: tuple[float, float] = (-0.14, 1.05),
) -> None:
    """Add a small bold panel label (``A``, ``B``, …), optionally with a title."""

    fs = fontsize if fontsize is not None else mpl.rcParams["axes.titlesize"] + 1
    ax.text(
        offset[0],
        offset[1],
        label,
        transform=ax.transAxes,
        fontsize=fs,
        fontweight="bold",
        va="top",
        ha="left",
    )
    if title:
        ax.set_title(title, loc="left", fontsize=mpl.rcParams["axes.titlesize"])


def save_figure(
    fig,
    path,
    *,
    formats: tuple[str, ...] = ("png", "pdf"),
    dpi: int | None = None,
    despine: bool = True,
) -> None:
    """Save a figure in multiple formats. PDF is vector; PNG is 300 dpi by default."""

    if despine:
        sns.despine(fig=fig)
    path = Path(path)
    base = path.parent / path.stem
    for fmt in formats:
        out = base.with_suffix(f".{fmt}")
        fig.savefig(out, format=fmt, dpi=(dpi or DPI_PRINT))
    plt.close(fig)


__all__ = [
    "COLORS",
    "COLOR_CYCLE",
    "DPI_PRINT",
    "DPI_SCREEN",
    "DOUBLE_COL_WIDTH",
    "FULL_PAGE_WIDTH",
    "SINGLE_COL_WIDTH",
    "add_panel_label",
    "get_figsize",
    "save_figure",
    "setup_style",
]
