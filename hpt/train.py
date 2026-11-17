"""Train one configuration and write a JSON result.

python -m hpt.train --arm mup --width 256 --precision fixed4 --eta0 0.01 --seed 0 --sweep phase1
"""
import argparse
import hashlib
import json
import math
import os
import platform
import socket
import sys
import time

import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True)
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--depth", type=int, default=3,
                   help="mlp: number of hidden layers; resmlp: number of residual blocks")
    p.add_argument("--arch", default="mlp", choices=["mlp", "resmlp"])
    p.add_argument("--depth-scaling", default="none", choices=["none", "sqrt"],
                   help="resmlp: depth-muP block multiplier and LR x sqrt(depth_base/depth)")
    p.add_argument("--depth-base", type=int, default=2)
    p.add_argument("--precision", default="fp32")
    p.add_argument("--scale-mode", default="latent")
    p.add_argument("--eta0", type=float, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--max-steps", type=int, default=None, help="override total steps (smoke tests)")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--final-lr-frac", type=float, default=0.1)
    p.add_argument("--d-base", type=int, default=64)
    p.add_argument("--adam-eps", type=float, default=1e-8)
    p.add_argument("--umup-input-lr", default="const", choices=["const", "fan_out"])
    p.add_argument("--readout-zero-init", action="store_true")
    p.add_argument("--range-mode", default="init", choices=["init", "calib"],
                   help="quantiser ranges from init std (init) or from a trained FP32 base-width model")
    p.add_argument("--calib-file", default=None,
                   help="default: calib/<arm>_D<d_base>.npz (made with --save-calib)")
    p.add_argument("--save-calib", default=None, help="save trained layer stats for calibration")
    p.add_argument("--tr-window", type=int, default=20)
    p.add_argument("--eval-test", action="store_true")
    p.add_argument("--evals-per-epoch", type=int, default=1,
                   help="validation evaluations per epoch, evenly spaced (logged in history['evals'] if > 1)")
    p.add_argument("--tag", default="", help="free label stored in the config (e.g. LR source)")
    p.add_argument("--sweep", default="debug")
    p.add_argument("--out-dir", default=os.path.join(PROJECT_DIR, "results"))
    p.add_argument("--data", default=os.path.join(PROJECT_DIR, "data", "jets_hlf.npz"))
    p.add_argument("--threads", type=int, default=int(os.environ.get("HPT_THREADS", "0")))
    p.add_argument("--force", action="store_true")
    return p.parse_args(argv)


CONFIG_KEYS = ("arm", "width", "depth", "precision", "scale_mode", "eta0", "seed", "epochs",
               "max_steps", "batch_size", "warmup_frac", "final_lr_frac", "d_base", "adam_eps",
               "umup_input_lr", "readout_zero_init", "range_mode", "tag", "arch", "depth_scaling",
               "depth_base")
# keys left out of the run-id hash when they have their default value, so that run ids made before
# the key existed stay valid
HASH_OMIT_DEFAULTS = {"range_mode": "init", "tag": "", "arch": "mlp", "depth_scaling": "none",
                      "depth_base": 2}


def run_id(cfg):
    base = (f"{cfg['arm']}_D{cfg['width']}_L{cfg['depth']}_{cfg['precision']}_{cfg['scale_mode']}"
            f"_lr{cfg['eta0']:.4g}_s{cfg['seed']}")
    hcfg = dict(cfg)
    for k, default in HASH_OMIT_DEFAULTS.items():
        if hcfg.get(k, default) in (default, None):
            hcfg.pop(k, None)
    if cfg.get("arch", "mlp") == "resmlp":
        base = base.replace(f"_L{cfg['depth']}_", f"_R{cfg['depth']}{cfg['depth_scaling']}_")
    h = hashlib.sha1(json.dumps(hcfg, sort_keys=True).encode()).hexdigest()[:8]
    return f"{base}_{h}"


def auc_scores(y, probs):
    """Macro one-vs-rest ROC AUC and per-class AUC from class probabilities."""
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    probs = np.asarray(probs, dtype=np.float64)
    if not np.all(np.isfinite(probs)):
        return float("nan"), [float("nan")] * probs.shape[1]
    per_class = [float(roc_auc_score(y == k, probs[:, k])) for k in range(probs.shape[1])]
    return float(np.mean(per_class)), per_class


def lr_factor(step, total, warmup_frac, final_frac):
    """Linear warmup then cosine decay to final_frac. Pure python (also used for logging)."""
    warm = max(1, int(round(warmup_frac * total)))
    if step < warm:
        return (step + 1) / warm
    t = (step - warm) / max(1, total - warm)
    return final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))


def main(argv=None):
    args = parse_args(argv)
    if args.precision == "fp32":
        args.range_mode = "init"   # no quantisers -> irrelevant
    cfg = {k: getattr(args, k) for k in CONFIG_KEYS}
    rid = run_id(cfg)
    out_dir = os.path.join(args.out_dir, args.sweep)
    out_path = os.path.join(out_dir, rid + ".json")
    if os.path.exists(out_path) and not args.force:
        print(f"exists, skipping: {out_path}")
        return out_path

    import tensorflow as tf
    if args.threads > 0:
        tf.config.threading.set_intra_op_parallelism_threads(args.threads)
        tf.config.threading.set_inter_op_parallelism_threads(min(2, args.threads))
    from .data import load
    from .model import build_mlp

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    d = load(args.data)
    Xtr, ytr, Xva, yva = d["X_train"], d["y_train"], d["X_val"], d["y_val"]

    calib = None
    if args.range_mode == "calib":
        cpath = args.calib_file or os.path.join(PROJECT_DIR, "calib", f"{args.arm}_D{args.d_base}.npz")
        c = np.load(cpath)
        calib = [{"kernel": c[f"kernel_{i}"], "bias": c[f"bias_{i}"],
                  "acts": c[f"acts_{i}"] if f"acts_{i}" in c.files else None}
                 for i in range(args.depth + 1)]
    if args.arch == "resmlp":
        from .model import build_resmlp
        model, layer_info = build_resmlp(args.arm, args.width, args.depth, args.precision,
                                         args.scale_mode, args.d_base, args.seed,
                                         umup_input_lr=args.umup_input_lr,
                                         depth_scaling=args.depth_scaling, depth_base=args.depth_base)
    else:
        model, layer_info = build_mlp(args.arm, args.width, args.depth, args.precision,
                                      args.scale_mode, args.d_base, args.seed,
                                      umup_input_lr=args.umup_input_lr,
                                      readout_zero_init=args.readout_zero_init, calib=calib)

    steps_per_epoch = len(Xtr) // args.batch_size
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    n_epochs = math.ceil(total_steps / steps_per_epoch)

    # one Adam per LR multiplier group; LR is set every step via a shared factor variable
    groups = {}
    for v in model.trainable_variables:
        groups.setdefault(float(v.lr_mult), []).append(v)
    opts = {m: tf.keras.optimizers.Adam(learning_rate=args.eta0 * m, beta_1=0.9, beta_2=0.999,
                                        epsilon=args.adam_eps) for m in groups}
    group_list = [(opts[m], args.eta0 * m, groups[m]) for m in groups]
    all_vars = [v for _, _, vs in group_list for v in vs]
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    @tf.function
    def train_step(x, y, lr_factor_t):
        for opt, base_lr, _ in group_list:
            opt.learning_rate.assign(base_lr * lr_factor_t)
        with tf.GradientTape() as tape:
            loss = loss_fn(y, model(x, training=True))
        grads = tape.gradient(loss, all_vars)
        i = 0
        for opt, _, vs in group_list:
            opt.apply_gradients(zip(grads[i:i + len(vs)], vs))
            i += len(vs)
        return loss

    @tf.function
    def eval_batch(x, y):
        logits = model(x, training=False)
        l = tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True)
        acc = tf.cast(tf.equal(tf.argmax(logits, -1, output_type=tf.int64), y), tf.float32)
        return tf.reduce_sum(l), tf.reduce_sum(acc), tf.nn.softmax(logits)

    def evaluate(X, Y, bs=8192, with_auc=False):
        """(loss, acc) or (loss, acc, auc_macro, auc_per_class)."""
        ls, accs, probs = 0.0, 0.0, []
        for i in range(0, len(X), bs):
            l, a, p = eval_batch(tf.constant(X[i:i + bs]), tf.constant(Y[i:i + bs]))
            ls += float(l)
            accs += float(a)
            if with_auc:
                probs.append(p.numpy())
        if not with_auc:
            return ls / len(X), accs / len(X)
        auc, auc_pc = auc_scores(Y, np.concatenate(probs))
        return ls / len(X), accs / len(X), auc, auc_pc

    qlayers = [l for l in model.dense_layers if l.kernel_quantizer is not None]

    def codes():
        return [l.kernel_codes() for l in qlayers]

    def changed_frac(a, b):
        return [float(tf.reduce_mean(tf.cast(tf.not_equal(x, y), tf.float32))) for x, y in zip(a, b)]

    def weight_stats():
        return [{"w_rms": float(tf.sqrt(tf.reduce_mean(l.kernel ** 2))),
                 "wq_rms": float(tf.sqrt(tf.reduce_mean(l.quantized_kernel() ** 2))),
                 "wq_zero_frac": float(tf.reduce_mean(tf.cast(tf.equal(l.quantized_kernel(), 0.0),
                                                              tf.float32)))}
                for l in model.dense_layers]

    tr_starts = sorted({min(total_steps - args.tr_window, max(0, int(f * total_steps)))
                        for f in (0.02, 0.25, 0.5, 0.75, 0.98)}) if qlayers else []
    tr_windows = {s: [] for s in tr_starts}

    init_val_loss, init_val_acc = evaluate(Xva, yva)
    hist = {"epoch": [], "step": [], "train_loss": [], "val_loss": [], "val_acc": [], "val_auc": [],
            "code_change_epoch": [], "weights": [], "evals": []}
    t0 = time.time()
    step = 0
    diverged = False
    max_train_loss = 0.0
    prev_epoch_codes = codes() if qlayers else None
    rng = np.random.default_rng(args.seed)
    # within-epoch evaluation points (batch counts), the epoch end is always evaluated
    mid_eval_steps = {round(j * steps_per_epoch / args.evals_per_epoch)
                      for j in range(1, max(1, args.evals_per_epoch))}
    for epoch in range(n_epochs):
        perm = rng.permutation(len(Xtr))
        run_loss, n_loss = 0.0, 0
        for b in range(steps_per_epoch):
            if step >= total_steps:
                break
            idx = perm[b * args.batch_size:(b + 1) * args.batch_size]
            f_lr = tf.constant(lr_factor(step, total_steps, args.warmup_frac, args.final_lr_frac),
                               tf.float32)
            in_window = [s for s in tr_starts if s <= step < s + args.tr_window]
            if in_window:
                before = codes()
            loss = float(train_step(tf.constant(Xtr[idx]), tf.constant(ytr[idx]), f_lr))
            if in_window:
                tr_windows[in_window[0]].append(changed_frac(before, codes()))
            step += 1
            if not math.isfinite(loss):
                diverged = True
                break
            if b + 1 in mid_eval_steps:
                el, ea, eu, _ = evaluate(Xva, yva, with_auc=True)
                hist["evals"].append({"step": step, "epoch_frac": step / steps_per_epoch,
                                      "val_loss": el, "val_acc": ea, "val_auc": eu})
            run_loss += loss
            n_loss += 1
            max_train_loss = max(max_train_loss, loss)
        if diverged:
            break
        vl, va, vu, vu_pc = evaluate(Xva, yva, with_auc=True)
        if args.evals_per_epoch > 1:
            hist["evals"].append({"step": step, "epoch_frac": step / steps_per_epoch,
                                  "val_loss": vl, "val_acc": va, "val_auc": vu})
        hist["val_auc"].append(vu)
        hist["epoch"].append(epoch + 1)
        hist["step"].append(step)
        hist["train_loss"].append(run_loss / max(1, n_loss))
        hist["val_loss"].append(vl)
        hist["val_acc"].append(va)
        hist["weights"].append(weight_stats())
        if qlayers:
            c = codes()
            hist["code_change_epoch"].append(changed_frac(prev_epoch_codes, c))
            prev_epoch_codes = c
        print(f"[{rid}] epoch {epoch + 1}/{n_epochs} step {step} train {run_loss / max(1, n_loss):.4f} "
              f"val {vl:.4f} acc {va:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if not math.isfinite(vl):
            diverged = True
            break

    final_val_loss = hist["val_loss"][-1] if (hist["val_loss"] and not diverged) else float("nan")
    final_val_acc = hist["val_acc"][-1] if (hist["val_acc"] and not diverged) else float("nan")
    if not diverged and final_val_loss > init_val_loss:
        diverged = True  # trained to a worse state than init
    result = {
        "run_id": rid, "config": cfg, "layers": layer_info,
        "steps_per_epoch": steps_per_epoch, "total_steps": total_steps, "steps_done": step,
        "init_val_loss": init_val_loss, "init_val_acc": init_val_acc,
        "final_val_loss": final_val_loss, "final_val_acc": final_val_acc,
        "best_val_loss": min(hist["val_loss"]) if hist["val_loss"] else float("nan"),
        "best_val_acc": max(hist["val_acc"]) if hist["val_acc"] else float("nan"),
        "final_val_auc": hist["val_auc"][-1] if (hist["val_auc"] and not diverged) else float("nan"),
        "final_val_auc_per_class": vu_pc if (hist["val_auc"] and not diverged) else None,
        "max_train_loss": max_train_loss, "diverged": diverged,
        "history": hist,
        "transition_rate": {"layers": [l.name for l in qlayers],
                            "windows": {str(s): v for s, v in tr_windows.items()}},
        "wall_time_s": time.time() - t0,
        "env": {"host": socket.gethostname(), "python": platform.python_version(),
                "tf": tf.__version__, "threads": args.threads, "argv": sys.argv},
    }
    if not diverged:   # end-of-training scale statistics (does the network stay at unit scale?)
        _, pre = model(tf.constant(Xva[:8192]), return_hidden=True)
        result["final_stats"] = [{
            "pre_rms": float(tf.sqrt(tf.reduce_mean(z ** 2))), "pre_absmax": float(tf.reduce_max(tf.abs(z))),
            "w_rms": float(tf.sqrt(tf.reduce_mean(l.kernel ** 2))),
            "w_absmax": float(tf.reduce_max(tf.abs(l.kernel))),
            "b_absmax": float(tf.reduce_max(tf.abs(l.bias))),
            "act_clip_frac": (float(tf.reduce_mean(tf.cast(z >= model.act_qs[i].s, tf.float32)))
                              if i < len(model.act_qs) and model.act_qs[i] is not None else None),
        } for i, (l, z) in enumerate(zip(model.dense_layers, pre))]
    if args.save_calib and not diverged:
        _, pre = model(tf.constant(Xva[:8192]), return_hidden=True)
        arrs = {}
        for i, (l, z) in enumerate(zip(model.dense_layers, pre)):
            arrs[f"kernel_{i}"] = l.kernel.numpy()
            arrs[f"bias_{i}"] = l.bias.numpy()
            if i < len(pre) - 1:
                arrs[f"acts_{i}"] = np.maximum(z.numpy(), 0).ravel()
        os.makedirs(os.path.dirname(os.path.abspath(args.save_calib)), exist_ok=True)
        np.savez(args.save_calib, **arrs)
        result["calib_saved"] = args.save_calib
    if args.eval_test and not diverged:
        (result["test_loss"], result["test_acc"], result["test_auc"],
         result["test_auc_per_class"]) = evaluate(d["X_test"], d["y_test"], with_auc=True)

    os.makedirs(out_dir, exist_ok=True)
    tmp = out_path + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(result, f, allow_nan=True)
    os.replace(tmp, out_path)
    print(f"wrote {out_path}  final val loss {final_val_loss:.4f} acc {final_val_acc:.4f} "
          f"diverged={diverged} ({time.time() - t0:.0f}s)")
    return out_path


if __name__ == "__main__":
    main()
