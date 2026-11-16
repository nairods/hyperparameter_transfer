"""Coordinate checks (Tensor Programs V, App. D.1; u-muP App. A.8): per-layer RMS quantities vs width
during the first Adam steps, for the forward and the backward pass.

Metrics recorded per layer after step t (one figure per metric):
  fwd_dz      RMS of the change of the pre-activations z since init, on a fixed eval batch   (forward)
  own_dz      RMS of (dW_eff) h: change of z caused by this layer's own weight update at step t
              (h = layer input on the eval batch). The central muP quantity: O(1) in width.
  upd_rms     RMS of the effective weight update dW_eff = d(multiplier * Q(W)) at step t
              (includes the quantiser: zero if no quantised weight changed level)
  grad_rms    RMS of dL/dW of the stored (latent) kernel
  bwd_signal  RMS of dL/dz (backward signal at the layer output), multiplied by the batch size

Expectations (Adam, after the first few steps): muP / u-muP -> own_dz and fwd_dz flat in width;
SP -> own_dz grows ~ D in hidden/output layers.

python -m hpt.coord_check --precision fp32
python -m hpt.coord_check --precision fixed4 --scale-mode latent
"""
import argparse
import os

import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ETA = {"sp": 1e-2, "sp_sqrtD": 1e-2, "mup": 1e-2, "umup": 1.0}
METRICS = {
    "fwd_dz": r"RMS $\Delta z$ since init",
    "own_dz": r"RMS $(\Delta W_{\rm eff})\,h$ at step t",
    "upd_rms": r"RMS $\Delta W_{\rm eff}$ at step t",
    "grad_rms": r"RMS $\partial L/\partial W$",
    "bwd_signal": r"RMS $\partial L/\partial z$ $\times$ batch",
}
# theoretical width exponent of own_dz per (arm, role) for Adam, used as dashed guide lines
OWN_DZ_SLOPE = {"sp": {"input": 0, "hidden": 1, "output": 1},
                "sp_sqrtD": {"input": -0.5, "hidden": 0.5, "output": 0.5},
                "mup": {"input": 0, "hidden": 0, "output": 0},
                "umup": {"input": 0, "hidden": 0, "output": 0}}


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def run(arm, width, precision, scale_mode, eta0, steps, X, Y, seed=0, batch=1024):
    import tensorflow as tf
    from .model import build_mlp
    tf.random.set_seed(seed)
    model, _ = build_mlp(arm, width, precision=precision, scale_mode=scale_mode, seed=seed)
    layers = model.dense_layers
    groups = {}
    for v in model.trainable_variables:
        groups.setdefault(float(v.lr_mult), []).append(v)
    opts = [(tf.keras.optimizers.Adam(eta0 * m), vs) for m, vs in groups.items()]
    allv = [v for _, vs in opts for v in vs]
    kernels = [l.kernel for l in layers]
    xe = tf.constant(X[:4096])

    def eff_weights():
        return [l.rule.multiplier * l.quantized_kernel().numpy() for l in layers]

    def layer_inputs(x):
        _, pre = model(x, return_hidden=True)
        ins = [x.numpy()] + [model._act(z, i).numpy() for i, z in enumerate(pre[:-1])]
        return ins, [z.numpy() for z in pre]

    _, pre0 = layer_inputs(xe)
    out = {m: [] for m in METRICS}
    for t in range(steps):
        idx = slice(4096 + t * batch, 4096 + (t + 1) * batch)
        ins_e, _ = layer_inputs(xe)
        w_before = eff_weights()
        xb = tf.constant(X[idx])
        with tf.GradientTape() as tape:
            logits, pre = model(xb, return_hidden=True)
            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
                Y[idx], logits, from_logits=True))
        grads = tape.gradient(loss, allv + pre)
        g_vars, g_pre = grads[:len(allv)], grads[len(allv):]
        g_by_var = {id(v): g for v, g in zip(allv, g_vars)}
        i = 0
        for opt, vs in opts:
            opt.apply_gradients(zip(g_vars[i:i + len(vs)], vs))
            i += len(vs)
        w_after = eff_weights()
        _, pre_now = layer_inputs(xe)
        dws = [wa - wb for wa, wb in zip(w_after, w_before)]
        out["fwd_dz"].append([rms(z - z0) for z, z0 in zip(pre_now, pre0)])
        out["own_dz"].append([rms(h @ dw) for h, dw in zip(ins_e, dws)])
        out["upd_rms"].append([rms(dw) for dw in dws])
        out["grad_rms"].append([rms(g_by_var[id(k)].numpy()) for k in kernels])
        out["bwd_signal"].append([rms(g.numpy()) * batch for g in g_pre])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--precision", default="fp32")
    p.add_argument("--scale-mode", default="latent")
    p.add_argument("--widths", default="4,8,16,32,64,128,256,512,1024,2048")
    p.add_argument("--arms", default="sp,sp_sqrtD,mup,umup")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--out-dir", default=os.path.join(PROJECT_DIR, "plots", "coord_check"))
    a = p.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from analysis import style
    from .data import load
    style.apply()
    d = load()
    X, Y = d["X_train"], d["y_train"]
    widths = [int(w) for w in a.widths.split(",")]
    arms = a.arms.split(",")
    if a.precision != "fp32":   # the quantised sweeps cover sp / mup / umup only
        arms = [x for x in arms if x != "sp_sqrtD"]
    names = ["input", "hidden 1", "hidden 2", "output (logits)"]
    roles = ["input", "hidden", "hidden", "output"]
    res = {}
    for arm in arms:
        res[arm] = {w: [run(arm, w, a.precision, a.scale_mode, DEFAULT_ETA[arm], a.steps, X, Y, seed=s)
                        for s in range(a.seeds)] for w in widths}
        print(arm, "done", flush=True)
    steps_show = [1, 2, 5, a.steps]
    step_cols = style.width_colors(steps_show)
    os.makedirs(a.out_dir, exist_ok=True)
    tag = f"{a.precision}_{a.scale_mode}" if a.precision != "fp32" else "fp32"
    for metric, ylabel in METRICS.items():
        fig, axes = plt.subplots(len(arms), 4, figsize=(12, 2.5 * len(arms)), sharex=True, squeeze=False)
        for r, arm in enumerate(arms):
            for li in range(4):
                ax = axes[r, li]
                for t in steps_show:
                    vals = [np.mean([rr[metric][t - 1][li] for rr in res[arm][w]]) for w in widths]
                    ax.plot(widths, vals, marker="o", ms=3.5, color=step_cols[t], label=f"step {t}")
                if metric == "own_dz":
                    slope = OWN_DZ_SLOPE[arm][roles[li]]
                    v_ref = np.mean([rr[metric][a.steps - 1][li] for rr in res[arm][64]]) if 64 in widths else None
                    if v_ref and v_ref > 0:
                        ww = np.array([widths[0], widths[-1]], float)
                        ax.plot(ww, v_ref * (ww / 64) ** slope, ls="--", lw=1, color="#52514e",
                                label=f"theory $\\propto D^{{{slope:g}}}$")
                ax.set_xscale("log", base=2)
                ax.set_yscale("log")
                if r == 0:
                    ax.set_title(names[li])
                if li == 0:
                    ax.set_ylabel(f"{style.ARM_LABEL[arm]}\n{ylabel}", fontsize=8)
                if r == len(arms) - 1:
                    ax.set_xlabel("width D")
                if li == 3:
                    ax.legend(fontsize=6, loc="best")
        fig.suptitle(f"Coordinate check [{metric}] ({a.precision}, {a.scale_mode}); eta0: " +
                     ", ".join(f"{style.ARM_LABEL[x]}={DEFAULT_ETA[x]}" for x in arms), fontsize=9)
        fig.tight_layout()
        out = os.path.join(a.out_dir, f"{metric}_{tag}.png")
        fig.savefig(out)
        plt.close(fig)
        print("saved", out)


if __name__ == "__main__":
    main()
