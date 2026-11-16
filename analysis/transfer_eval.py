"""Performance and time-to-threshold vs width when the LR tuned at a small source width is reused.

Reads results/transfer_eval/*.json (made by sweeps/make_transfer_eval.py):
  tag "transfer32": eta* tuned at D=32 used at every width
  tag "own":        each width's own eta* (per-width-tuned reference)

Figures (plots/transfer_eval/):
  perf_test_acc.png, perf_test_auc.png       test accuracy / macro AUC vs width, with literature baselines
  time_to_<metric>_<thr>.png                 epochs until the validation metric first reaches a threshold

python -m analysis.transfer_eval
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from .analyze import PROJECT_DIR

# literature baselines on the hls4ml jet HLF data (see README "Baseline")
BASELINES = {
    "acc": [(0.760, "76.0%: 6-bit QAT / CERNBox float level"),
            (0.769, "76.9%: best published 16-64-32-32-5 (HGQ, da4ml; OpenML)")],
    "auc": [(0.940, "AUC 0.940: Ps&Qs FP32 mean AUC")],
}
THRESHOLDS = [("acc", 0.760), ("acc", 0.769), ("auc", 0.940)]
SETTINGS = [("fp32", "latent"), ("fixed8", "latent"), ("fixed8", "auto"), ("fixed4", "latent"),
            ("fixed4", "auto")]


def setting_label(prec, mode):
    return "FP32" if prec == "fp32" else f"{prec}, scale '{mode}'"


def load(sweep):
    rows = []
    for f in glob.glob(os.path.join(PROJECT_DIR, "results", sweep, "*.json")):
        r = json.load(open(f))
        c = r["config"]
        evals = r["history"].get("evals") or []
        rows.append({
            "arm": c["arm"], "width": c["width"], "precision": c["precision"],
            "scale_mode": c["scale_mode"], "eta0": c["eta0"], "seed": c["seed"], "tag": c.get("tag", ""),
            "diverged": r["diverged"], "test_acc": r.get("test_acc", np.nan),
            "test_auc": r.get("test_auc", np.nan), "val_acc": r["final_val_acc"],
            "val_auc": r.get("final_val_auc", np.nan),
            "ev_epoch": [e["epoch_frac"] for e in evals],
            "ev_acc": [e["val_acc"] for e in evals], "ev_auc": [e["val_auc"] for e in evals],
        })
    df = pd.DataFrame(rows)
    for col in ("test_acc", "test_auc", "val_acc", "val_auc"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.loc[df.diverged.astype(bool), ["test_acc", "test_auc", "val_acc", "val_auc"]] = np.nan
    return df


def time_to(row, metric, thr):
    """First epoch (fraction) at which the validation metric reaches thr; inf if never."""
    for ep, v in zip(row["ev_epoch"], row[f"ev_{metric}"]):
        if v is not None and np.isfinite(v) and v >= thr:
            return ep
    return np.inf


def plot_perf(df, metric, out, source_tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from . import style
    style.apply()
    settings = [s for s in SETTINGS if ((df.precision == s[0]) & (df.scale_mode == s[1])).any()]
    fig, axes = plt.subplots(1, len(settings), figsize=(3.9 * len(settings), 3.6), sharey=True,
                             squeeze=False)
    col = f"test_{metric}"
    for ax, (prec, mode) in zip(axes[0], settings):
        sub = df[(df.precision == prec) & (df.scale_mode == mode)]
        for arm in style.ARM_ORDER:
            for tag, ls, lab in ((source_tag, "-", r"$\eta^*_{32}$ transferred"), ("own", "--", "own $\\eta^*$")):
                g = sub[(sub.arm == arm) & (sub.tag == tag)]
                if tag == "own":   # at the source width "own" == transferred
                    g = pd.concat([g, sub[(sub.arm == arm) & (sub.tag == source_tag) & (sub.width == 32)]])
                if g.empty:
                    continue
                s = g.groupby("width")[col].agg(["mean", "std", "count"])
                n_div = g.groupby("width")[col].apply(lambda v: v.isna().sum())
                c = style.CATEGORICAL[style.ARM_ORDER.index(arm)]
                ax.errorbar(s.index, s["mean"], yerr=s["std"].fillna(0), ls=ls, marker="o" if ls == "-" else None,
                            ms=4, capsize=2, color=c, alpha=1 if ls == "-" else 0.7,
                            label=f"{style.ARM_LABEL[arm]}" if ls == "-" else None)
                for w, nd in n_div.items():
                    if nd and ls == "-":
                        ax.annotate(f"{nd}×div", (w, ax.get_ylim()[0]), fontsize=6, color=c,
                                    xytext=(0, 3), textcoords="offset points", ha="center")
        for y, lab in BASELINES[metric]:
            ax.axhline(y, color="#52514e", lw=0.8, ls=":")
            ax.annotate(lab, (ax.get_xlim()[0], y), fontsize=6, color="#52514e", xytext=(2, 2),
                        textcoords="offset points")
        ax.set_xscale("log", base=2)
        ax.set_title(setting_label(prec, mode))
        ax.set_xlabel("width D")
    lo = np.nanpercentile(df[col], 5)
    axes[0, 0].set_ylim(min(lo, min(y for y, _ in BASELINES[metric])) - 0.004, np.nanmax(df[col]) + 0.002)
    axes[0, 0].set_ylabel(f"test {metric.upper() if metric == 'auc' else 'accuracy'}")
    axes[0, -1].plot([], [], color="#52514e", ls="-", label=r"$\eta^*$ from D=32 (transferred)")
    axes[0, -1].plot([], [], color="#52514e", ls="--", label=r"each width's own $\eta^*$")
    axes[0, -1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle(f"Test {metric} after 15 epochs vs width (mean ± std over seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print("saved", out)


def plot_time(df, metric, thr, out, source_tag, max_epoch=15):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from . import style
    style.apply()
    settings = [s for s in SETTINGS if ((df.precision == s[0]) & (df.scale_mode == s[1])).any()]
    fig, axes = plt.subplots(1, len(settings), figsize=(3.9 * len(settings), 3.6), sharey=True,
                             squeeze=False)
    top = max_epoch + 1
    rows = []
    for ax, (prec, mode) in zip(axes[0], settings):
        sub = df[(df.precision == prec) & (df.scale_mode == mode) & (df.tag == source_tag)].copy()
        sub["t"] = sub.apply(lambda r: time_to(r, metric, thr), axis=1)
        for i, arm in enumerate(style.ARM_ORDER):
            g = sub[sub.arm == arm]
            if g.empty:
                continue
            c = style.CATEGORICAL[style.ARM_ORDER.index(arm)]
            agg = g.groupby("width")["t"].agg(["median", "min", "max"])
            agg["n_reached"] = g.groupby("width")["t"].apply(lambda v: np.isfinite(v).sum())
            agg["n"] = g.groupby("width")["t"].size()
            for w, r in agg.iterrows():
                rows.append({"precision": prec, "scale_mode": mode, "arm": arm, "width": w,
                             "metric": metric, "threshold": thr, **r.to_dict()})
            reached = agg[np.isfinite(agg["median"])]
            off = 1 + 0.04 * (i - 1.5)   # small horizontal offset per arm
            ax.errorbar(reached.index * off, reached["median"],
                        yerr=[reached["median"] - reached["min"],
                              np.minimum(reached["max"], top) - reached["median"]],
                        marker="o", ms=4, capsize=2, color=c, label=style.ARM_LABEL[arm])
            nr = agg[~np.isfinite(agg["median"])]
            if len(nr):
                ax.scatter(nr.index * off, [top] * len(nr), marker="o", facecolors="none", edgecolors=c, s=30)
        ax.axhline(max_epoch, color="#52514e", lw=0.6, ls=":")
        ax.annotate("not reached ↑", (2, top), fontsize=6, color="#52514e", xytext=(0, -9),
                    textcoords="offset points")
        ax.set_xscale("log", base=2)
        ax.set_ylim(0, top + 0.6)
        ax.set_title(setting_label(prec, mode))
        ax.set_xlabel("width D")
    name = f"val accuracy ≥ {thr:.1%}" if metric == "acc" else f"val AUC ≥ {thr:.3f}"
    axes[0, 0].set_ylabel(f"epochs until {name}")
    axes[0, -1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle(f"Time to threshold ({name}) with the LR transferred from D=32 "
                 f"(median, min–max over seeds; 15-epoch cosine schedule)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print("saved", out)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", default="transfer_eval")
    p.add_argument("--source-tag", default="transfer32")
    a = p.parse_args()
    df = load(a.sweep)
    print(f"{len(df)} runs, {int(df.diverged.sum())} diverged")
    out_dir = os.path.join(PROJECT_DIR, "plots", a.sweep)
    os.makedirs(out_dir, exist_ok=True)
    summ = (df.groupby(["precision", "scale_mode", "arm", "tag", "width"])
            [["test_acc", "test_auc", "val_acc", "val_auc", "eta0"]].agg(["mean", "std"]))
    summ.to_csv(os.path.join(out_dir, "summary.csv"))
    for metric in ("acc", "auc"):
        plot_perf(df, metric, os.path.join(out_dir, f"perf_test_{metric}.png"), a.source_tag)
    tt = [plot_time(df, m, t, os.path.join(out_dir, f"time_to_{m}_{t:g}.png"), a.source_tag)
          for m, t in THRESHOLDS]
    tt = pd.concat(tt, ignore_index=True)
    tt.to_csv(os.path.join(out_dir, "time_to_threshold.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(df[df.tag == a.source_tag].pivot_table(index=["precision", "scale_mode", "width"],
                                                     columns="arm", values="test_acc").round(4))


if __name__ == "__main__":
    main()
