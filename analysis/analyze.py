"""Collect sweep results and make the transfer plots.

python -m analysis.analyze phase1_fp32                       # facets = arm
python -m analysis.analyze phase3_precision --facet precision --curve width
"""
import argparse
import glob
import json
import math
import os

import numpy as np
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG_COLS = ["arm", "width", "depth", "precision", "scale_mode", "eta0", "seed", "epochs", "d_base",
            "umup_input_lr", "readout_zero_init", "tag"]


def collect(sweep):
    """Load all run JSONs of a sweep (comma-separated names are merged) into a DataFrame."""
    if "," in sweep:
        parts = [collect(s) for s in sweep.split(",")]
        return pd.concat([p for p in parts if len(p)], ignore_index=True)
    rows = []
    for f in glob.glob(os.path.join(PROJECT_DIR, "results", sweep, "*.json")):
        r = json.load(open(f))
        row = {k: r["config"].get(k) for k in CFG_COLS}
        row.update({k: r.get(k) for k in ("final_val_loss", "final_val_acc", "best_val_loss",
                                          "best_val_acc", "init_val_loss", "diverged", "wall_time_s",
                                          "max_train_loss", "test_loss", "test_acc", "final_val_auc",
                                          "test_auc")})
        row["sweep"] = sweep
        row["tag"] = row.get("tag") or ""
        h = r["history"]
        row["final_train_loss"] = h["train_loss"][-1] if h["train_loss"] and not r["diverged"] else np.nan
        tr = r.get("transition_rate", {})
        for start, vals in (tr.get("windows") or {}).items():
            if vals:
                per_layer = np.mean(np.asarray(vals), axis=0)
                for name, v in zip(tr["layers"], per_layer):
                    row[f"tr@{start}:{name.split('_', 2)[-1]}{name.split('_')[1]}"] = float(v)
        w_last = r["history"]["weights"][-1] if r["history"]["weights"] else None
        if w_last:
            for i, w in enumerate(w_last):
                row[f"wq_zero_frac_{i}"] = w["wq_zero_frac"]
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        bad = df["diverged"].astype(bool) | ~np.isfinite(df["final_val_loss"].astype(float))
        df.loc[bad, ["final_val_loss", "final_val_acc", "final_train_loss"]] = np.nan
        df = df.sort_values(["arm", "precision", "scale_mode", "width", "eta0", "seed"])
        df = df.reset_index(drop=True)
        out = os.path.join(PROJECT_DIR, "results", f"{sweep}.csv")
        df.to_csv(out, index=False)
    return df


def fit_optimum(etas, losses, n_side=2):
    """Vertex of a quadratic in log2(eta) through the points around the argmin of `losses`."""
    etas, losses = np.asarray(etas, float), np.asarray(losses, float)
    ok = np.isfinite(losses)
    if ok.sum() < 3:
        return (etas[ok][np.argmin(losses[ok])] if ok.any() else np.nan), np.nan
    x, y = np.log2(etas[ok]), losses[ok]
    i = int(np.argmin(y))
    lo, hi = max(0, i - n_side), min(len(x), i + n_side + 1)
    xs, ys = x[lo:hi], y[lo:hi]
    if len(xs) >= 3:
        a, b, c = np.polyfit(xs, ys, 2)
        if a > 0:
            xv = -b / (2 * a)
            if xs.min() <= xv <= xs.max():
                return 2 ** xv, c - b * b / (4 * a)
    return 2 ** x[i], y[i]


def optimum_table(df, group_cols, metric="final_val_loss", n_boot=200, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for key, g in df.groupby(group_cols):
        piv = g.pivot_table(index="eta0", columns="seed", values=metric, aggfunc="mean").sort_index()
        etas = piv.index.values
        mean = piv.mean(axis=1, skipna=False).values
        eta_star, l_star = fit_optimum(etas, mean)
        boots = []
        if piv.shape[1] > 1:
            for _ in range(n_boot):
                cols = rng.integers(0, piv.shape[1], piv.shape[1])
                boots.append(math.log2(fit_optimum(etas, piv.values[:, cols].mean(axis=1))[0]))
        rows.append({**dict(zip(group_cols, key if isinstance(key, tuple) else (key,))),
                     "eta_star": eta_star, "loss_star": l_star,
                     "log2_eta_star_std": float(np.nanstd(boots)) if boots else np.nan,
                     "best_grid_loss": np.nanmin(mean) if np.isfinite(mean).any() else np.nan,
                     "n_seeds": piv.shape[1]})
    return pd.DataFrame(rows)


def transfer_degradation(df, opt, group_cols, d_base, metric="final_val_loss"):
    """Loss at width D when using the eta* found at d_base (nearest grid point) minus the best at D."""
    rows = []
    other = [c for c in group_cols if c != "width"]
    for key, g in df.groupby(other):
        key = key if isinstance(key, tuple) else (key,)
        sel = opt
        for c, v in zip(other, key):
            sel = sel[sel[c] == v]
        base = sel[sel["width"] == d_base]
        if base.empty:
            continue
        eta_b = base["eta_star"].iloc[0]
        means = g.groupby(["width", "eta0"])[metric].mean().reset_index()
        for w, gw in means.groupby("width"):
            j = np.argmin(np.abs(np.log2(gw["eta0"].values) - math.log2(eta_b)))
            rows.append({**dict(zip(other, key)), "width": w, "eta_transfer": gw["eta0"].values[j],
                         "loss_transfer": gw[metric].values[j],
                         "loss_best": np.nanmin(gw[metric].values),
                         "degradation": gw[metric].values[j] - np.nanmin(gw[metric].values)})
    return pd.DataFrame(rows)


def plot_curves(df, facet, curve, metric, title, out, opt=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from . import style
    style.apply()
    facets = [f for f in (style.ARM_ORDER if facet == "arm" else style.PRECISION_ORDER if facet == "precision"
                          else sorted(df[facet].unique())) if f in set(df[facet])]
    cvals = sorted(df[curve].unique(), key=lambda v: (style.PRECISION_ORDER.index(v)
                                                       if curve == "precision" else v))
    colors = style.width_colors(cvals) if curve == "width" else style.precision_colors(cvals)
    fig, axes = plt.subplots(1, len(facets), figsize=(3.6 * len(facets), 3.4), sharey=True,
                             squeeze=False)
    finite = df[metric].dropna()
    if metric.endswith("loss"):
        ylo, yhi = finite.min() - 0.005, min(finite.quantile(0.9), finite.min() + 0.25)
    else:
        ylo, yhi = max(finite.quantile(0.1), finite.max() - 0.15), finite.max() + 0.003
    for ax, fv in zip(axes[0], facets):
        sub = df[df[facet] == fv]
        for cv in cvals:
            s = sub[sub[curve] == cv].groupby("eta0")[metric].agg(["mean", "min", "max", "count"])
            if s.empty:
                continue
            lab = f"D={cv}" if curve == "width" else str(cv)
            ax.plot(s.index, s["mean"], marker="o", color=colors[cv], label=lab)
            if s["count"].max() > 1:
                ax.fill_between(s.index, s["min"], s["max"], color=colors[cv], alpha=0.15, lw=0)
            if opt is not None:
                o = opt[(opt[facet] == fv) & (opt[curve] == cv)]
                if len(o) and np.isfinite(o["eta_star"].iloc[0]):
                    ax.axvline(o["eta_star"].iloc[0], color=colors[cv], lw=0.8, ls=":")
        ax.set_xscale("log", base=2)
        ax.set_ylim(ylo, yhi)
        ax.set_title(style.ARM_LABEL.get(fv, fv) if facet == "arm" else fv)
        ax.set_xlabel(r"base learning rate $\eta_0$")
    axes[0, 0].set_ylabel(metric.replace("_", " "))
    axes[0, -1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print("saved", out)


def plot_optimum(opt, facet, out, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from . import style
    style.apply()
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    vals = sorted(opt[facet].unique(), key=lambda v: (style.ARM_ORDER + style.PRECISION_ORDER).index(v)
                  if v in style.ARM_ORDER + style.PRECISION_ORDER else 99)
    for i, fv in enumerate(vals):
        o = opt[opt[facet] == fv].sort_values("width")
        if len(o) < 2:
            continue
        x, y = o["width"].values, o["eta_star"].values
        ok = np.isfinite(y)
        slope = np.polyfit(np.log2(x[ok]), np.log2(y[ok]), 1)[0] if ok.sum() >= 2 else np.nan
        yerr = None
        if o["log2_eta_star_std"].notna().any():
            s = o["log2_eta_star_std"].fillna(0).values
            yerr = [y - y * 2 ** -s, y * 2 ** s - y]
        ax.errorbar(x, y, yerr=yerr, marker="o", color=style.CATEGORICAL[i], capsize=2,
                    label=f"{style.ARM_LABEL.get(fv, fv)}  (slope {slope:+.2f})")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("width D")
    ax.set_ylabel(r"optimal $\eta_0^*$")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print("saved", out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sweep")
    p.add_argument("--facet", default="arm")
    p.add_argument("--curve", default="width")
    p.add_argument("--d-base", type=int, default=64)
    a = p.parse_args()
    df = collect(a.sweep)
    if df.empty:
        print("no results")
        return
    print(f"{len(df)} runs, {int(df['diverged'].sum())} diverged")
    plot_dir = os.path.join(PROJECT_DIR, "plots", a.sweep)
    os.makedirs(plot_dir, exist_ok=True)
    split_cols = [c for c in ("arm", "precision", "scale_mode") if c not in (a.facet, a.curve)
                  and df[c].nunique() > 1]
    groups = df.groupby(split_cols) if split_cols else [((), df)]
    for key, g in groups:
        key = key if isinstance(key, tuple) else (key,)
        tag = "_".join(str(k) for k in key) or "all"
        gcols = [a.facet, a.curve] + split_cols
        opt = optimum_table(g, [c for c in gcols if c in g.columns])
        opt.to_csv(os.path.join(plot_dir, f"optimum_{tag}.csv"), index=False)
        print(f"\n== {tag}\n" + opt.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        for metric in ("final_val_loss", "final_val_acc", "final_train_loss"):
            o = opt if metric == "final_val_loss" else None
            plot_curves(g, a.facet, a.curve, metric, f"{a.sweep} {tag}",
                        os.path.join(plot_dir, f"curves_{metric}_{tag}.png"), o)
        if a.curve == "width":
            plot_optimum(opt, a.facet, os.path.join(plot_dir, f"eta_star_vs_width_{tag}.png"),
                         f"{a.sweep} {tag}")
            deg = transfer_degradation(g, opt, gcols, a.d_base)
            if len(deg):
                deg.to_csv(os.path.join(plot_dir, f"degradation_{tag}.csv"), index=False)
                print(deg.pivot_table(index="width", columns=a.facet, values="degradation")
                      .to_string(float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
