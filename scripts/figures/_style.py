"""Publication-quality matplotlib styling for BayesBreak figures.

This module provides consistent, high-quality styling suitable for academic
publications (e.g., NeurIPS, ICML, ICLR).
"""

from __future__ import annotations

import logging
import warnings

# Suppress fontTools warnings about font metadata timestamps (benign)
warnings.filterwarnings("ignore", category=UserWarning, module="fontTools")
warnings.filterwarnings("ignore", message=".*timestamp.*")
warnings.filterwarnings("ignore", message=".*extra bytes.*")
logging.getLogger("fontTools").setLevel(logging.ERROR)

import matplotlib as mpl  # noqa: E402
import seaborn as sns  # noqa: E402

# ══════════════════════════════════════════════════════════════════════════════
# Publication-quality defaults
# ══════════════════════════════════════════════════════════════════════════════

# Figure sizes (inches) - typical single/double column widths
SINGLE_COL_WIDTH = 3.25  # ~83mm (Nature/Science single column)
DOUBLE_COL_WIDTH = 6.75  # ~171mm (Nature/Science double column)
FULL_PAGE_WIDTH = 7.0

# Color palette - vibrant but professional
COLORS = {
    "blue": "#0077BB",  # Strong blue
    "red": "#CC3311",  # Strong red
    "green": "#009988",  # Teal green
    "orange": "#EE7733",  # Orange
    "purple": "#AA4499",  # Purple
    "cyan": "#33BBEE",  # Cyan
    "grey": "#888888",  # Medium grey
    "lightgrey": "#DDDDDD",  # Light grey for backgrounds
    "black": "#000000",
}

# Ordered color cycle (high contrast)
COLOR_CYCLE = [
    COLORS["blue"],
    COLORS["red"],
    COLORS["green"],
    COLORS["orange"],
    COLORS["purple"],
    COLORS["cyan"],
]

# Standard DPI for different outputs
DPI_SCREEN = 150
DPI_PRINT = 300


def setup_style(
    *,
    font_scale: float = 1.0,
    use_tex: bool = False,
    style: str = "paper",
    context: str = "paper",
) -> None:
    """Configure matplotlib for publication-quality figures.

    Parameters
    ----------
    font_scale : float
        Multiplier for all font sizes.
    use_tex : bool
        If True, use LaTeX for text rendering (slower but better math).
    style : str
        One of 'paper' (tight, minimal), 'presentation' (larger fonts/lines).
    context : str
        Seaborn context: 'paper', 'notebook', 'talk', 'poster'.
    """
    # Use seaborn's clean style as base
    sns.set_theme(
        style="ticks",
        context=context,
        font_scale=font_scale,
        rc={
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        },
    )

    # Custom overrides for publication quality
    if style == "presentation":
        base_fontsize = 14
        linewidth = 2.5
        markersize = 10
    else:  # paper
        base_fontsize = 10
        linewidth = 1.8
        markersize = 6

    base_fontsize *= font_scale

    params = {
        # Figure
        "figure.dpi": DPI_SCREEN,
        "figure.facecolor": "white",
        "figure.edgecolor": "white",
        "figure.autolayout": False,
        "figure.constrained_layout.use": True,
        # Fonts
        "font.size": base_fontsize,
        "axes.labelsize": base_fontsize + 1,
        "axes.titlesize": base_fontsize + 2,
        "axes.titleweight": "bold",
        "xtick.labelsize": base_fontsize,
        "ytick.labelsize": base_fontsize,
        "legend.fontsize": base_fontsize - 1,
        "legend.title_fontsize": base_fontsize,
        # Lines - thicker for visibility
        "lines.linewidth": linewidth,
        "lines.markersize": markersize,
        "patch.linewidth": 1.0,
        # Axes
        "axes.linewidth": 1.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=COLOR_CYCLE),
        "axes.grid": False,
        "axes.axisbelow": True,
        "axes.labelweight": "medium",
        # Ticks
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        # Legend
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.fancybox": False,
        "legend.edgecolor": "0.7",
        "legend.borderpad": 0.4,
        # Savefig
        "savefig.dpi": DPI_PRINT,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "savefig.transparent": False,
        "savefig.facecolor": "white",
        # PDF/PS - use TrueType for editability
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }

    if use_tex:
        params.update(
            {
                "text.usetex": True,
                "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
                "font.family": "serif",
            }
        )

    mpl.rcParams.update(params)


def get_figsize(
    width: str | float = "single",
    aspect: float = 0.65,
    nrows: int = 1,
    ncols: int = 1,
) -> tuple[float, float]:
    """Calculate figure size maintaining aspect ratio.

    Parameters
    ----------
    width : str or float
        'single', 'double', 'full', or a numeric width in inches.
    aspect : float
        Height-to-width ratio for each subplot.
    nrows, ncols : int
        Number of subplot rows/columns.

    Returns
    -------
    tuple[float, float]
        (width, height) in inches.
    """
    if width == "single":
        w = SINGLE_COL_WIDTH
    elif width == "double":
        w = DOUBLE_COL_WIDTH
    elif width == "full":
        w = FULL_PAGE_WIDTH
    else:
        w = float(width)

    h = w * aspect * (nrows / ncols)
    return (w, h)


def add_panel_label(
    ax,
    label: str,
    loc: str = "upper left",
    fontsize: float | None = None,
    fontweight: str = "bold",
    offset: tuple[float, float] = (-0.12, 1.05),
) -> None:
    """Add a panel label (A), (B), etc. to an axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    label : str
        The label text, e.g. 'A', 'B', etc.
    loc : str
        Location hint (currently unused, offset used directly).
    fontsize : float, optional
        Font size; defaults to axes title size + 2.
    fontweight : str
        Font weight.
    offset : tuple[float, float]
        (x, y) in axes coordinates.
    """
    fs = fontsize or (mpl.rcParams["axes.titlesize"] + 2)
    ax.text(
        offset[0],
        offset[1],
        label,
        transform=ax.transAxes,
        fontsize=fs,
        fontweight=fontweight,
        va="top",
        ha="left",
    )


def save_figure(
    fig,
    path,
    *,
    formats: tuple[str, ...] = ("png", "pdf"),
    dpi: int | None = None,
) -> None:
    """Save figure in multiple formats.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    path : Path or str
        Base path without extension.
    formats : tuple of str
        File formats to save (e.g., 'png', 'pdf', 'svg').
    dpi : int, optional
        Override DPI for raster formats.
    """
    from pathlib import Path

    path = Path(path)
    base = path.parent / path.stem

    for fmt in formats:
        out = base.with_suffix(f".{fmt}")
        save_dpi = dpi if dpi else DPI_PRINT
        fig.savefig(out, format=fmt, dpi=save_dpi)
