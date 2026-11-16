"""Phase 2 summary figures (width transfer at fixed precision).

1. eta* vs width, one panel per (precision, scale_mode), FP32 (phase1) as dashed reference.
2. Hidden-layer transition rate (fraction of quantised weights changing per step) vs width,
   measured at each width's own optimal LR (nearest grid point).

python -m analysis.phase2_summary
"""
import argparse
import os

import numpy as np
import pandas as pd

from .analyze import PROJECT_DIR, collect, optimum_table


def nearest_eta_rows(df, opt, keys):
    """Rows of df at the grid eta closest to the fitted eta* of their (keys) group."""
    out = []
    for _, o in opt.iterrows():
        sel = df
        for k in keys:
            sel = sel[sel[k] == o[k]]
        if sel.empty or not np.isfinite(o["eta_star"]):
            continue
        etas = np.sort(sel["eta0"].unique())
        e = etas[np.argmin(np.abs(np.log2(etas) - np.log2(o["eta_star"])))]
        out.append(sel[sel["eta0"] == e])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", default="phase2_fixed")
    p.add_argument("--fp32", default="phase1_fp32")
    a = p.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from . import style
    style.apply()
    out_dir = os.path.join(PROJECT_DIR, "plots", a.sweep)
    os.makedirs(out_dir, exist_ok=True)

    df = collect(a.sweep)
    fp = collect(a.fp32)
    arms = [x for x in style.ARM_ORDER if x in set(df.arm)]
    keys = ["arm", "width", "precision", "scale_mode"]
    opt = optimum_table(df, keys)
    opt_fp = optimum_table(fp[fp.arm.isin(arms)], ["arm", "width"])
    opt.to_csv(os.path.join(out_dir, "optimum_all.csv"), index=False)

    # ---- 1. eta* vs width ----
    precs = [x for x in style.PRECISION_ORDER if x in set(df.precision)]
    modes = sorted(df.scale_mode.unique())
    fig, axes = plt.subplots(len(precs), len(modes), figsize=(5.2 * len(modes), 3.6 * len(precs)),
                             sharex=True, squeeze=False)
    slopes = []
    for r, prec in enumerate(precs):
        for c, mode in enumerate(modes):
            ax = axes[r, c]
            for arm in arms:
                col = style.CATEGORICAL[style.ARM_ORDER.index(arm)]
                f = opt_fp[opt_fp.arm == arm].sort_values("width")
                ax.plot(f.width, f.eta_star, ls="--", lw=1.2, color=col, alpha=0.6)
                o = opt[(opt.arm == arm) & (opt.precision == prec) & (opt.scale_mode == mode)]
                o = o.sort_values("width")
                ok = np.isfinite(o.eta_star.values)
                slope = (np.polyfit(np.log2(o.width.values[ok]), np.log2(o.eta_star.values[ok]), 1)[0]
                         if ok.sum() >= 2 else np.nan)
                slopes.append({"arm": arm, "precision": prec, "scale_mode": mode, "slope": slope})
                s = o.log2_eta_star_std.fillna(0).values
                y = o.eta_star.values
                ax.errorbar(o.width, y, yerr=[y - y * 2 ** -s, y * 2 ** s - y], marker="o", capsize=2,
                            color=col, label=f"{style.ARM_LABEL[arm]} (slope {slope:+.2f})")
            ax.set_xscale("log", base=2)
            ax.set_yscale("log", base=2)
            ax.set_title(f"{prec}, scale mode '{mode}'")
            if c == 0:
                ax.set_ylabel(r"optimal $\eta_0^*$")
            if r == len(precs) - 1:
                ax.set_xlabel("width D")
            ax.legend(fontsize=7, loc="lower left")
    fig.suptitle("Optimal LR vs width at fixed precision (solid); FP32 reference (dashed)", fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, "summary_eta_star_vs_width.png")
    fig.savefig(path)
    plt.close(fig)
    print("saved", path)
    sl = pd.DataFrame(slopes)
    print(sl.pivot_table(index=["precision", "scale_mode"], columns="arm", values="slope")
          .to_string(float_format=lambda v: f"{v:+.2f}"))

    # ---- 2. transition rate at eta* vs width ----
    at_opt = nearest_eta_rows(df, opt, keys)
    tr_cols = [c for c in at_opt.columns if c.startswith("tr@") and "hidden" in c]
    if not tr_cols:
        print("no transition-rate columns")
        return
    starts = sorted({int(c.split("@")[1].split(":")[0]) for c in tr_cols})
    mid = starts[len(starts) // 2]
    cols_mid = [c for c in tr_cols if c.startswith(f"tr@{mid}:")]
    at_opt["tr_hidden_mid"] = at_opt[cols_mid].mean(axis=1)
    fig, axes = plt.subplots(len(precs), len(modes), figsize=(5.2 * len(modes), 3.4 * len(precs)),
                             sharex=True, squeeze=False)
    for r, prec in enumerate(precs):
        for c, mode in enumerate(modes):
            ax = axes[r, c]
            for arm in arms:
                g = at_opt[(at_opt.arm == arm) & (at_opt.precision == prec) & (at_opt.scale_mode == mode)]
                m = g.groupby("width")["tr_hidden_mid"].mean()
                ax.plot(m.index, m.values, marker="o", color=style.CATEGORICAL[style.ARM_ORDER.index(arm)],
                        label=style.ARM_LABEL[arm])
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_title(f"{prec}, scale mode '{mode}'")
            if c == 0:
                ax.set_ylabel("hidden-layer transition rate\n(fraction of weights changing / step)")
            if r == len(precs) - 1:
                ax.set_xlabel("width D")
            ax.legend(fontsize=7)
    fig.suptitle(f"Weight transition rate at each width's optimal LR (window at step {mid})", fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, "summary_transition_rate_vs_width.png")
    fig.savefig(path)
    plt.close(fig)
    print("saved", path)


if __name__ == "__main__":
    main()
