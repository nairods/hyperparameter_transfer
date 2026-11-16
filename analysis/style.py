"""Shared matplotlib style: ordinal blue ramp for width, fixed categorical order for arms/precisions."""
import matplotlib as mpl

# ordinal ramp (light -> dark = narrow -> wide), steps 250..700 of the reference blue ramp
WIDTH_RAMP = ["#86b6ef", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
              "#0d366b"]
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ARM_ORDER = ["sp", "sp_sqrtD", "mup", "umup"]
ARM_LABEL = {"sp": "SP", "sp_sqrtD": r"SP, $\eta_0\sqrt{d_{base}/D}$", "mup": r"$\mu$P", "umup": r"u-$\mu$P"}
PRECISION_ORDER = ["fp32", "fixed8", "fixed6", "fixed4", "fixed3", "fixed2", "ternary", "binary"]


def width_colors(widths):
    widths = sorted(widths)
    n = len(widths)
    idx = [round(i * (len(WIDTH_RAMP) - 1) / max(1, n - 1)) for i in range(n)]
    return {w: WIDTH_RAMP[i] for w, i in zip(widths, idx)}


def precision_colors(precisions):
    return {p: CATEGORICAL[PRECISION_ORDER.index(p)] for p in precisions}


def apply():
    mpl.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 200, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
        "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
        "lines.linewidth": 2.0, "lines.markersize": 5, "legend.frameon": False, "font.size": 9,
        "axes.titlesize": 10, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    })
