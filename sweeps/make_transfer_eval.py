"""Generate the transfer-evaluation sweep: train every width with (a) the LR tuned at the source width
(default D=32, tag "transfer32") and (b) each width's own optimal LR (tag "own"), for a fixed budget,
with half-epoch validation checkpoints, AUC and a test-set evaluation.

The optimal LRs are fitted from the LR-sweep results (analysis.analyze.optimum_table), so rerunning this
script after more sweeps have finished adds the runs that were not possible before (e.g. "own" runs at
widths 4, 8, 16 once phase5_small_width is done). Finished runs are skipped.

python sweeps/make_transfer_eval.py [--source-width 32]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.analyze import collect, optimum_table  # noqa: E402
from sweeps.make_sweep import write_jobs  # noqa: E402

WIDTHS = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
SETTINGS = ([("fp32", "latent", a) for a in ("sp", "sp_sqrtD", "mup", "umup")] +
            [(p, m, a) for p in ("fixed8", "fixed4") for m in ("latent", "auto")
             for a in ("sp", "mup", "umup")])
FP32_SWEEPS = "phase1_fp32,phase5_small_width"
QUANT_SWEEPS = "phase2_fixed,phase5_small_width"


def optimal_lrs():
    """DataFrame (arm, width, precision, scale_mode, eta_star) from all LR sweeps."""
    import pandas as pd
    tabs = []
    for sweeps in (FP32_SWEEPS, QUANT_SWEEPS):
        df = collect(sweeps)
        df = df[df["tag"].fillna("") == ""]
        tabs.append(optimum_table(df, ["arm", "width", "precision", "scale_mode"]))
    opt = pd.concat(tabs, ignore_index=True)
    return opt.drop_duplicates(["arm", "width", "precision", "scale_mode"])


def runs(opt, source_width, seeds, epochs, evals_per_epoch):
    missing = []
    for prec, mode, arm in SETTINGS:
        o = opt[(opt.arm == arm) & (opt.precision == prec) & (opt.scale_mode == mode)]
        eta = dict(zip(o.width, o.eta_star))
        if not np.isfinite(eta.get(source_width, np.nan)):
            missing.append(f"{arm}/{prec}/{mode} source D={source_width}")
            continue
        for width in WIDTHS:
            choices = [(f"transfer{source_width}", eta[source_width])]
            if width != source_width:
                if np.isfinite(eta.get(width, np.nan)):
                    choices.append(("own", eta[width]))
                else:
                    missing.append(f"{arm}/{prec}/{mode} own D={width}")
            for tag, e in choices:
                for seed in seeds:
                    yield {"arm": arm, "width": width, "precision": prec, "scale_mode": mode,
                           "eta0": float(f"{e:.6g}"), "seed": seed, "epochs": epochs,
                           "evals_per_epoch": evals_per_epoch, "eval_test": True, "tag": tag}
    if missing:
        print(f"skipped (no fitted eta* yet): {len(missing)} entries, e.g. {missing[:4]}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-width", type=int, default=32)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--evals-per-epoch", type=int, default=2)
    p.add_argument("--sweep", default="transfer_eval")
    a = p.parse_args()
    opt = optimal_lrs()
    seeds = [int(s) for s in a.seeds.split(",")]
    write_jobs(a.sweep, list(runs(opt, a.source_width, seeds, a.epochs, a.evals_per_epoch)))


if __name__ == "__main__":
    main()
