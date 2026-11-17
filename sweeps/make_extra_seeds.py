"""Add seeds to an existing LR sweep, only at the grid points near each fitted optimum.

For every (arm, width, precision, scale_mode, umup_input_lr) group of the sweep, the eta0 grid points
within +-K grid steps of the fitted eta* get runs with the new seeds. The runs go into the same sweep
directory, so all analyses pick them up automatically (means use all available seeds per point).

python sweeps/make_extra_seeds.py phase2_fixed --seeds 2,3 --half-steps 4
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.analyze import collect, optimum_table  # noqa: E402
from sweeps.make_sweep import write_jobs  # noqa: E402

KEYS = ["arm", "width", "precision", "scale_mode", "umup_input_lr"]


def runs(sweep, seeds, half_steps, precisions=None, epochs=15):
    df = collect(sweep)
    df = df[df["tag"].fillna("") == ""]
    if precisions:
        df = df[df.precision.isin(precisions)]
    df["umup_input_lr"] = df["umup_input_lr"].fillna("const")
    opt = optimum_table(df, KEYS)
    for _, o in opt.iterrows():
        if not np.isfinite(o.eta_star):
            continue
        g = df
        for k in KEYS:
            g = g[g[k] == o[k]]
        etas = np.sort(g.eta0.unique())
        i = int(np.argmin(np.abs(np.log2(etas) - np.log2(o.eta_star))))
        for e in etas[max(0, i - half_steps): i + half_steps + 1]:
            for seed in seeds:
                c = {"arm": o.arm, "width": int(o.width), "precision": o.precision,
                     "scale_mode": o.scale_mode, "eta0": float(e), "seed": seed, "epochs": epochs}
                if o.umup_input_lr != "const":
                    c["umup_input_lr"] = o.umup_input_lr
                yield c


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sweep")
    p.add_argument("--seeds", required=True)
    p.add_argument("--half-steps", type=int, default=4)
    p.add_argument("--precisions", default=None, help="comma-separated filter, e.g. fixed8,fixed4")
    a = p.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    precs = a.precisions.split(",") if a.precisions else None
    write_jobs(a.sweep, list(runs(a.sweep, seeds, a.half_steps, precs)))


if __name__ == "__main__":
    main()
