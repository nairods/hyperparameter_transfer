"""Q1: transfer of the learning rate from FP32 to quantised (QAT) networks at fixed width.

Combines several sweeps (FP32 rows are reused as the baseline for every scale_mode).
python -m analysis.precision --sweeps phase1_fp32,phase2_fixed,phase3_precision --widths 64,512
"""
import argparse
import math
import os

import numpy as np
import pandas as pd

from .analyze import PROJECT_DIR, collect, optimum_table


def load(sweeps, widths, arms):
    df = pd.concat([collect(s) for s in sweeps], ignore_index=True)
    df = df[df["width"].isin(widths) & df["arm"].isin(arms)]
    fp = df[df["precision"] == "fp32"]
    q = df[df["precision"] != "fp32"]
    parts = [q]
    for mode in sorted(q["scale_mode"].unique()):
        parts.append(fp.assign(scale_mode=mode))
    return pd.concat(parts, ignore_index=True)


def fp32_transfer(df, opt):
    """Loss when using the FP32 eta* (nearest grid point) in each precision, minus that precision's best."""
    rows = []
    for (arm, width, mode), g in df.groupby(["arm", "width", "scale_mode"]):
        base = opt[(opt.arm == arm) & (opt.width == width) & (opt.scale_mode == mode) &
                   (opt.precision == "fp32")]
        if base.empty:
            continue
        eta_fp = base["eta_star"].iloc[0]
        for prec, gp in g.groupby("precision"):
            m = gp.groupby("eta0")["final_val_loss"].mean()
            j = np.argmin(np.abs(np.log2(m.index.values) - math.log2(eta_fp)))
            o = opt[(opt.arm == arm) & (opt.width == width) & (opt.scale_mode == mode) &
                    (opt.precision == prec)]
            rows.append({"arm": arm, "width": width, "scale_mode": mode, "precision": prec,
                         "eta_star": o["eta_star"].iloc[0] if len(o) else np.nan,
                         "eta_ratio_vs_fp32": (o["eta_star"].iloc[0] / eta_fp) if len(o) else np.nan,
                         "loss_best": np.nanmin(m.values), "loss_at_fp32_eta": m.values[j],
                         "degradation": m.values[j] - np.nanmin(m.values)})
    return pd.DataFrame(rows)


def plot(df, opt, width, mode, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from . import style
    style.apply()
    sub = df[(df.width == width) & (df.scale_mode == mode)]
    arms = [a for a in style.ARM_ORDER if a in set(sub.arm)]
    precs = [p for p in style.PRECISION_ORDER if p in set(sub.precision)]
    colors = style.precision_colors(precs)
    fig, axes = plt.subplots(1, len(arms), figsize=(3.8 * len(arms), 3.5), squeeze=False)
    for ax, arm in zip(axes[0], arms):
        sa = sub[sub.arm == arm]
        lo = sa["final_val_loss"].min()
        for p in precs:
            s = sa[sa.precision == p].groupby("eta0")["final_val_loss"].agg(["mean", "min", "max"])
            if s.empty:
                continue
            ax.plot(s.index, s["mean"], marker="o", ms=3.5, color=colors[p], label=p)
            ax.fill_between(s.index, s["min"], s["max"], color=colors[p], alpha=0.12, lw=0)
            o = opt[(opt.arm == arm) & (opt.width == width) & (opt.scale_mode == mode) & (opt.precision == p)]
            if len(o) and np.isfinite(o.eta_star.iloc[0]):
                ax.axvline(o.eta_star.iloc[0], color=colors[p], lw=0.8, ls=":")
        ax.set_xscale("log", base=2)
        ax.set_ylim(lo - 0.005, lo + 0.2)
        ax.set_title(style.ARM_LABEL[arm])
        ax.set_xlabel(r"base learning rate $\eta_0$")
    axes[0, 0].set_ylabel("final val loss")
    axes[0, -1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    fig.suptitle(f"Precision transfer at D={width}, weight scale mode '{mode}'", fontsize=10)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print("saved", out)


def plot_ratio(tab, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from . import style
    style.apply()
    precs = [p for p in style.PRECISION_ORDER if p in set(tab.precision)]
    modes = sorted(tab.scale_mode.unique())
    fig, axes = plt.subplots(1, len(modes), figsize=(5 * len(modes), 3.6), sharey=True, squeeze=False)
    for ax, mode in zip(axes[0], modes):
        i = 0
        for arm in style.ARM_ORDER:
            for width, ls in zip(sorted(tab.width.unique()), ["-", "--", ":"]):
                t = tab[(tab.arm == arm) & (tab.width == width) & (tab.scale_mode == mode)]
                if t.empty:
                    continue
                t = t.set_index("precision").reindex(precs)
                ax.plot(range(len(precs)), t["eta_ratio_vs_fp32"], marker="o", ls=ls,
                        color=style.CATEGORICAL[style.ARM_ORDER.index(arm)],
                        label=f"{style.ARM_LABEL[arm]}, D={width}")
            i += 1
        ax.axhline(1.0, color="#52514e", lw=0.8)
        ax.set_yscale("log", base=2)
        ax.set_xticks(range(len(precs)), precs, rotation=30)
        ax.set_title(f"scale mode '{mode}'")
    axes[0, 0].set_ylabel(r"$\eta_0^*(\mathrm{precision}) / \eta_0^*(\mathrm{fp32})$")
    axes[0, -1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print("saved", out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweeps", default="phase1_fp32,phase2_fixed,phase3_precision")
    p.add_argument("--widths", default="64,512")
    p.add_argument("--arms", default="sp,mup,umup")
    p.add_argument("--tag", default="q1")
    a = p.parse_args()
    widths = [int(w) for w in a.widths.split(",")]
    df = load(a.sweeps.split(","), widths, a.arms.split(","))
    opt = optimum_table(df, ["arm", "width", "scale_mode", "precision"])
    tab = fp32_transfer(df, opt)
    out_dir = os.path.join(PROJECT_DIR, "plots", a.tag)
    os.makedirs(out_dir, exist_ok=True)
    tab.to_csv(os.path.join(out_dir, "precision_transfer.csv"), index=False)
    with pd.option_context("display.width", 200):
        print(tab.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    for w in widths:
        for mode in sorted(df.scale_mode.unique()):
            plot(df, opt, w, mode, os.path.join(out_dir, f"curves_D{w}_{mode}.png"))
    plot_ratio(tab, os.path.join(out_dir, "eta_ratio_vs_precision.png"))


if __name__ == "__main__":
    main()
