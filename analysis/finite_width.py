"""Finite-width effects: at which width does LR transfer break down?

Merges the LR sweeps at D=32..2048 (phase1_fp32 / phase2_fixed) with the small-width sweep
(phase5_small_width, D=4,8,16) and produces (plots/finite_width/):

  eta_ratio_vs_width.png    eta*(D) / eta*_inf, with eta*_inf = geometric mean of eta* at D>=512.
                            Grey band = +-0.5 octave (measured eta* noise). Breakdown width D_c marked.
  transfer_heatmap_<setting>.png
                            Delta L(s -> t): val loss at target width t with the eta* tuned at source width s
                            (nearest grid point) minus the best val loss at t. Diagonal = 0.
  curves_<setting>.png      val loss vs eta0 for all widths 4..2048
  breakdown.csv             D_c per arm and setting, fitted finite-width correction exponent p

D_c = smallest width such that every width >= D_c has |log2(eta*/eta*_inf)| <= 0.5 AND a loss penalty
for using eta*_inf below max(2 x seed standard error, 0.001).

python -m analysis.finite_width
"""
import math
import os

import numpy as np
import pandas as pd

from .analyze import PROJECT_DIR, collect, optimum_table, plot_curves

SETTINGS = [("fp32", "latent"), ("fixed8", "latent"), ("fixed8", "auto"), ("fixed4", "latent"),
            ("fixed4", "auto")]
LARGE = (512, 1024, 2048)
BAND = 0.5   # octaves


def load():
    fp = collect("phase1_fp32,phase5_small_width")
    q = collect("phase2_fixed,phase5_small_width")
    df = pd.concat([fp[fp.precision == "fp32"], q[q.precision != "fp32"]], ignore_index=True)
    return df[df["tag"].fillna("") == ""]


def nearest(etas, target):
    etas = np.asarray(etas)
    return etas[np.argmin(np.abs(np.log2(etas) - math.log2(target)))]


def analyse_setting(df, opt):
    """Per arm: ratio table, eta*_inf, penalties, D_c."""
    rows, heat = [], {}
    for arm, g in df.groupby("arm"):
        o = opt[opt.arm == arm].set_index("width").sort_index()
        big = o.loc[[w for w in LARGE if w in o.index], "eta_star"]
        eta_inf = float(np.exp(np.mean(np.log(big)))) if len(big) else np.nan
        means = g.groupby(["width", "eta0"])["final_val_loss"].agg(["mean", "std", "count"]).reset_index()
        widths = sorted(g.width.unique())
        pen = {}
        for w in widths:
            m = means[means.width == w]
            if m["mean"].notna().sum() == 0:
                continue
            best = m.loc[m["mean"].idxmin()]
            se = best["std"] / math.sqrt(max(1, best["count"])) if best["count"] > 1 else 0.0
            at_inf = m[m.eta0 == nearest(m.eta0, eta_inf)]["mean"].iloc[0] if np.isfinite(eta_inf) else np.nan
            pen[w] = (at_inf - best["mean"], max(2 * se, 0.001))
        # source -> target heatmap
        H = pd.DataFrame(index=widths, columns=widths, dtype=float)
        for s in widths:
            if s not in o.index or not np.isfinite(o.loc[s, "eta_star"]):
                continue
            for t in widths:
                m = means[means.width == t]
                if m["mean"].notna().sum() == 0:
                    continue
                e = nearest(m.eta0, o.loc[s, "eta_star"])
                H.loc[s, t] = m[m.eta0 == e]["mean"].iloc[0] - m["mean"].min()
        heat[arm] = H
        ok = {}
        for w in widths:
            r = math.log2(o.loc[w, "eta_star"] / eta_inf) if (w in o.index and np.isfinite(eta_inf)) else np.nan
            p_, tol = pen.get(w, (np.nan, np.nan))
            ok[w] = bool(np.isfinite(r) and abs(r) <= BAND and np.isfinite(p_) and p_ <= tol)
            rows.append({"arm": arm, "width": w, "eta_star": o.loc[w, "eta_star"] if w in o.index else np.nan,
                         "log2_ratio": r, "log2_std": o.loc[w, "log2_eta_star_std"] if w in o.index else np.nan,
                         "penalty_at_eta_inf": p_, "tolerance": tol, "eta_inf": eta_inf, "ok": ok[w]})
        dc = None
        for w in widths:
            if all(ok[v] for v in widths if v >= w):
                dc = w
                break
        for r in rows:
            if r["arm"] == arm:
                r["D_c"] = dc
    return pd.DataFrame(rows), heat


def fit_power(tab):
    """log2(eta*/eta*_inf) = a * D^(-p), fitted on all widths; returns (a, p) or nan."""
    from scipy.optimize import curve_fit
    t = tab.dropna(subset=["log2_ratio"])
    if len(t) < 4 or np.all(np.abs(t.log2_ratio) < 0.1):
        return np.nan, np.nan
    try:
        (a, p), _ = curve_fit(lambda d, a, p: a * d ** (-p), t.width.values.astype(float),
                              t.log2_ratio.values, p0=(t.log2_ratio.iloc[0] * 2, 0.5),
                              bounds=([-50, 0.05], [50, 3]), maxfev=20000)
        return a, p
    except Exception:
        return np.nan, np.nan


def plot_ratio(all_tabs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from . import style
    style.apply()
    settings = [s for s in SETTINGS if s in all_tabs]
    fig, axes = plt.subplots(1, len(settings), figsize=(3.9 * len(settings), 3.7), sharey=True, squeeze=False)
    for ax, s in zip(axes[0], settings):
        tab = all_tabs[s]
        ax.axhspan(-BAND, BAND, color="#e4e3df", alpha=0.7, lw=0)
        ax.axhline(0, color="#52514e", lw=0.8)
        for i, arm in enumerate(style.ARM_ORDER):
            t = tab[tab.arm == arm].sort_values("width")
            if t.empty:
                continue
            c = style.CATEGORICAL[style.ARM_ORDER.index(arm)]
            off = 1 + 0.04 * (i - 1.5)
            dc = t["D_c"].iloc[0]
            ax.errorbar(t.width * off, t.log2_ratio, yerr=t.log2_std.fillna(0), marker="o", ms=4, capsize=2,
                        color=c, label=f"{style.ARM_LABEL[arm]}  ($D_c$={dc if dc else '>2048'})")
            if dc:
                ax.axvline(dc, color=c, lw=0.8, ls=":")
        ax.set_xscale("log", base=2)
        ax.set_title("FP32" if s[0] == "fp32" else f"{s[0]}, scale '{s[1]}'")
        ax.set_xlabel("width D")
        ax.legend(fontsize=6.5, loc="best")
    axes[0, 0].set_ylabel(r"$\log_2(\eta_0^*(D) / \eta_0^*(\infty))$  [octaves]")
    fig.suptitle(r"Finite-width deviation of the optimal LR; $\eta^*(\infty)$ = geo. mean of $\eta^*$ at "
                 r"D$\geq$512; band = $\pm$0.5 octave (noise); dotted = breakdown width $D_c$", fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print("saved", out)


def plot_heat(heat, setting, out, vmax=0.02):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from . import style
    style.apply()
    cmap = LinearSegmentedColormap.from_list("blues", ["#fcfcfb"] + style.WIDTH_RAMP)
    arms = [a for a in style.ARM_ORDER if a in heat]
    fig, axes = plt.subplots(1, len(arms), figsize=(4.4 * len(arms), 4.2), squeeze=False)
    for ax, arm in zip(axes[0], arms):
        H = heat[arm].astype(float)
        im = ax.imshow(np.clip(H.values, 0, vmax), cmap=cmap, vmin=0, vmax=vmax, origin="lower")
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                v = H.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{1000 * v:.0f}", ha="center", va="center", fontsize=6,
                            color="#fcfcfb" if v > 0.6 * vmax else "#0b0b0b")
        ax.set_xticks(range(H.shape[1]), [str(w) for w in H.columns], fontsize=7, rotation=45)
        ax.set_yticks(range(H.shape[0]), [str(w) for w in H.index], fontsize=7)
        ax.set_xlabel("target width t")
        ax.set_ylabel(r"source width s (where $\eta^*$ is tuned)")
        ax.set_title(style.ARM_LABEL[arm])
        ax.grid(False)
    cb = fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8)
    cb.set_label(r"$\Delta L$ = val-loss penalty (cells: $\times 10^{-3}$)")
    name = "FP32" if setting[0] == "fp32" else f"{setting[0]}, scale '{setting[1]}'"
    fig.suptitle(f"Transfer penalty when tuning at width s and training at width t ({name})", fontsize=10)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def main():
    out_dir = os.path.join(PROJECT_DIR, "plots", "finite_width")
    os.makedirs(out_dir, exist_ok=True)
    df = load()
    all_tabs, summary = {}, []
    for prec, mode in SETTINGS:
        g = df[(df.precision == prec) & (df.scale_mode == mode)]
        if g.empty:
            continue
        opt = optimum_table(g, ["arm", "width"])
        tab, heat = analyse_setting(g, opt)
        tab["precision"], tab["scale_mode"] = prec, mode
        all_tabs[(prec, mode)] = tab
        tag = prec if prec == "fp32" else f"{prec}_{mode}"
        plot_heat(heat, (prec, mode), os.path.join(out_dir, f"transfer_heatmap_{tag}.png"))
        plot_curves(g, "arm", "width", "final_val_loss", f"val loss vs LR, widths 4-2048 ({tag})",
                    os.path.join(out_dir, f"curves_{tag}.png"), opt)
        for arm, t in tab.groupby("arm"):
            a, p = fit_power(t)
            summary.append({"precision": prec, "scale_mode": mode, "arm": arm, "D_c": t["D_c"].iloc[0],
                            "eta_inf": t["eta_inf"].iloc[0], "fit_a": a, "fit_p": p,
                            "log2_ratio_D4": t[t.width == 4]["log2_ratio"].mean(),
                            "log2_ratio_D32": t[t.width == 32]["log2_ratio"].mean()})
    plot_ratio(all_tabs, os.path.join(out_dir, "eta_ratio_vs_width.png"))
    pd.concat(all_tabs.values()).to_csv(os.path.join(out_dir, "ratio_table.csv"), index=False)
    s = pd.DataFrame(summary)
    s.to_csv(os.path.join(out_dir, "breakdown.csv"), index=False)
    with pd.option_context("display.width", 200):
        print(s.to_string(index=False, float_format=lambda v: f"{v:.3g}"))


if __name__ == "__main__":
    main()
