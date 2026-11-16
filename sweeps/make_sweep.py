"""Expand a sweep config (configs/<name>.json) into an HTCondor args file, skipping finished runs.

Config format:
{
  "sweep": "phase1_fp32",
  "grid": {"arm": [...], "width": [...], "precision": [...], "scale_mode": [...], "seed": [...]},
  "eta0": {"<arm>": {"center": 2e-3, "k_min": -8, "k_max": 8, "step": 0.5}},   # eta0 = center * 2^(k*step)
  "fixed": {"epochs": 15},                         # any other hpt.train option
  "max_width_per_arm": {"sp": 2048},               # optional
  "extra_grids": [{"grid": {...}, "eta0": {...}}]  # optional additional blocks (same "fixed")
}
python sweeps/make_sweep.py configs/phase1_fp32.json   ->  sweeps/phase1_fp32.args
"""
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hpt.train import CONFIG_KEYS, parse_args, run_id  # noqa: E402

FLAVOUR = os.environ.get("HPT_FLAVOUR")  # force a JobFlavour, e.g. for resubmissions
CHUNK_BUDGET_S = float(os.environ.get("HPT_CHUNK_H", "5")) * 3600   # estimated CPU seconds per condor job (runs are packed into chunks)


def est_cost(width, epochs, precision="fp32"):
    """Estimated wall seconds of one run on a 4-core lxbatch slot, including ~20 s python/TF start-up.
    Calibrated on phase1/phase2 (fp32 D=2048 ~3400 s; quantised D=1024 ~1050 s, D=64 ~65 s)."""
    if precision == "fp32":
        c = 30 + 0.0008 * width ** 2
    else:
        c = 60 + 0.0009 * width ** 2
    return 20 + c * epochs / 15


def eta_grid(spec):
    n = int(round((spec["k_max"] - spec["k_min"]) / 1)) + 1
    ks = [spec["k_min"] + i for i in range(n)]
    return [float(f"{spec['center'] * 2 ** (k * spec.get('step', 0.5)):.6g}") for k in ks]


def to_cli(cfg):
    out = []
    for k, v in cfg.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                out.append(flag)
        elif v is not None:
            out += [flag, str(v)]
    return out


def expand(conf):
    grid = conf["grid"]
    blocks = [{"grid": grid, "eta0": conf["eta0"]}] + conf.get("extra_grids", [])
    for blk in blocks:
        keys = list(blk["grid"])
        for combo in itertools.product(*(blk["grid"][k] for k in keys)):
            c = dict(zip(keys, combo))
            if c["width"] > conf.get("max_width_per_arm", {}).get(c["arm"], 10 ** 9):
                continue
            for eta in eta_grid(blk["eta0"][c["arm"]]):
                yield {**conf.get("fixed", {}), **c, "eta0": eta}


PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def write_jobs(sweep, runs, out_dir=None):
    """Pack runs into condor chunks and write sweeps/<sweep>.args. Skips runs whose result exists.

    runs: iterable of dicts of hpt.train options (without --sweep). Returns (n_total, n_todo, n_jobs).
    """
    res_dir = os.path.join(PROJECT, "results", sweep)
    done = set(os.listdir(res_dir)) if os.path.isdir(res_dir) else set()
    todo, n_total, seen = [], 0, set()
    for c in runs:
        cli = to_cli(c) + ["--sweep", sweep]
        a = parse_args(cli)
        cfg = {k: getattr(a, k) for k in CONFIG_KEYS}
        rid = run_id(cfg)
        if rid in seen:          # identical config listed twice
            continue
        seen.add(rid)
        n_total += 1
        if rid + ".json" in done:
            continue
        todo.append((est_cost(a.width, a.epochs, a.precision), " ".join(cli)))
    # pack runs into chunks of ~CHUNK_BUDGET_S (first-fit decreasing), one condor job per chunk
    todo.sort(key=lambda t: -t[0])
    chunks = []
    for cost, cli in todo:
        for ch in chunks:
            if ch[0] + cost <= CHUNK_BUDGET_S:
                ch[0] += cost
                ch[1].append(cli)
                break
        else:
            chunks.append([cost, [cli]])
    chunk_dir = os.path.join(PROJECT, "sweeps", "chunks", sweep)
    os.makedirs(chunk_dir, exist_ok=True)
    for f in os.listdir(chunk_dir):
        os.remove(os.path.join(chunk_dir, f))
    lines = []
    for i, (cost, clis) in enumerate(chunks):
        path = os.path.join(chunk_dir, f"{i:04d}.txt")
        with open(path, "w") as f:
            f.write("\n".join(clis) + "\n")
        flav = FLAVOUR or ("longlunch" if cost < 1.5 * 3600 else "workday" if cost < 6 * 3600 else "tomorrow")
        lines.append(f"{flav} --chunk {path}")
    out = os.path.join(out_dir or os.path.join(PROJECT, "sweeps"), sweep + ".args")
    with open(out, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    tot_h = sum(c[0] for c in chunks) / 3600
    print(f"{sweep}: {n_total} runs in grid, {n_total - len(todo)} done, {len(todo)} to run in "
          f"{len(chunks)} jobs (~{tot_h:.0f} slot-hours) -> {out}")
    return n_total, len(todo), len(chunks)


def main(path, out_dir=None):
    conf = json.load(open(path))
    return write_jobs(conf["sweep"], expand(conf), out_dir)


if __name__ == "__main__":
    main(*sys.argv[1:])
