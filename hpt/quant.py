"""Precision configurations built from QKeras quantisers.

A weight quantiser is applied to the *stored* (latent) parameter as   W_q = s * Q(W / s),
where Q is a unit-format QKeras quantiser (STE gradients) and s is a fixed power of two.

scale_mode
  "latent": s is the MSE-optimal power of two for N(0, sigma_ref^2), where sigma_ref is the init std
            of the stored kernel at the *base* width. The grid is then frozen across widths, like a
            fixed hardware format (e.g. ap_fixed<b, i>).  Under SP / muP the latent std shrinks like
            1/sqrt(D) so the grid becomes relatively coarse; under u-muP the latent std is 1 at every
            width so the grid is matched.
  "auto":   QKeras data-dependent scales (alpha="auto_po2" per channel for fixed point,
            alpha="auto" for ternary / binary), s = 1.  Approximately scale invariant (control).
"""
import functools
import math

import numpy as np
import tensorflow as tf
from qkeras import binary, quantized_bits, quantized_relu, ternary

PRECISIONS = ("fp32", "fixed8", "fixed6", "fixed4", "fixed3", "fixed2", "ternary", "binary")
SCALE_MODES = ("latent", "auto")
EXTREME_EDGE_BITS = 8   # first/last layer + activations for ternary / binary networks
ACT_BITS_FOR_EXTREME = 8
MIN_BIAS_BITS = 8


def _bits(precision):
    return int(precision[len("fixed"):]) if precision.startswith("fixed") else None


def make_unit_quantizer(kind, bits=None, auto=False):
    """Unit-format QKeras quantiser. kind in {"fixed", "ternary", "binary", "relu"}."""
    if kind == "fixed":
        return quantized_bits(bits, 0, alpha="auto_po2" if auto else 1)
    if kind == "ternary":
        return ternary(alpha="auto" if auto else 1)
    if kind == "binary":
        return binary(alpha="auto" if auto else 1)
    if kind == "relu":
        return quantized_relu(bits, 0)
    raise ValueError(kind)


@functools.lru_cache(maxsize=None)
def optimal_po2_scale(kind, bits, sigma, relu=False, n=200_000, seed=0):
    """Power-of-two s minimising E[(s Q(x/s) - x)^2] for x ~ N(0, sigma^2) (or ReLU of it)."""
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, sigma, n).astype(np.float32)
    if relu:
        x = np.maximum(x, 0.0)
    q = make_unit_quantizer(kind, bits)
    xt = tf.constant(x)
    best = None
    k0 = int(round(math.log2(sigma)))
    for k in range(k0 - 6, k0 + 7):
        s = 2.0 ** k
        mse = float(tf.reduce_mean((s * q(xt / s) - xt) ** 2))
        if best is None or mse < best[0]:
            best = (mse, s)
    return best[1]


def optimal_po2_scale_samples(kind, bits, samples, relu=False):
    """Power-of-two s minimising the quantisation MSE of the given (calibration) samples."""
    x = np.asarray(samples, np.float32).ravel()
    if relu:
        x = np.maximum(x, 0.0)
    ref = float(np.sqrt(np.mean(x ** 2))) or 1.0
    q = make_unit_quantizer(kind, bits)
    xt = tf.constant(x)
    best = None
    k0 = int(round(math.log2(ref)))
    for k in range(k0 - 6, k0 + 8):
        sc = 2.0 ** k
        mse = float(tf.reduce_mean((sc * q(xt / sc) - xt) ** 2))
        if best is None or mse < best[0]:
            best = (mse, sc)
    return best[1]


def cover_po2(max_abs, floor=2.0 ** -3):
    """Smallest power of two >= max_abs (range of a signed unit-format quantiser is [-s, s))."""
    return max(floor, 2.0 ** math.ceil(math.log2(max(max_abs, 1e-12) * 1.0001)))


class WeightQuantizer:
    """Callable W -> s * Q(W / s) with STE, plus integer-code extraction for transition rates."""

    def __init__(self, kind, bits=None, scale_mode="latent", sigma_ref=None, samples=None, s=None):
        self.kind, self.bits, self.scale_mode = kind, bits, scale_mode
        auto = scale_mode == "auto"
        self.q = make_unit_quantizer(kind, bits, auto=auto)
        if auto:
            self.s = 1.0
        elif s is not None:
            self.s = float(s)
        elif samples is not None:
            self.s = optimal_po2_scale_samples(kind, bits, samples)
        else:
            if sigma_ref is None:
                raise ValueError("latent scale mode needs sigma_ref")
            self.s = optimal_po2_scale(kind, bits, float(sigma_ref))
        # LSB of the unit grid (codes = Q(w/s) / lsb are integers in latent mode)
        self.lsb = 2.0 ** -(bits - 1) if kind == "fixed" else 1.0

    def __call__(self, w):
        return self.s * self.q(w / self.s)

    def codes(self, w):
        """Discrete codes of the quantised weights (exact integers in latent mode)."""
        return tf.round(self.q(w / self.s) / self.lsb) if self.scale_mode == "latent" else self.q(w / self.s)

    def describe(self):
        return {"kind": self.kind, "bits": self.bits, "scale_mode": self.scale_mode, "s": self.s,
                "lsb_effective": self.s * self.lsb if self.scale_mode == "latent" else None}


class ActQuantizer:
    """Quantised ReLU with a fixed power-of-two range (activations are O(1) in every arm)."""

    def __init__(self, bits, sigma_pre=1.0, samples=None):
        self.bits = bits
        self.q = make_unit_quantizer("relu", bits)
        if samples is not None:
            self.s = optimal_po2_scale_samples("relu", bits, samples, relu=True)
        else:
            self.s = optimal_po2_scale("relu", bits, float(sigma_pre), relu=True)

    def __call__(self, x):
        return self.s * self.q(x / self.s)

    def describe(self):
        return {"bits": self.bits, "s": self.s}


def layer_quantizers(precision, role, scale_mode, sigma_ref, calib=None):
    """(kernel_quantizer, bias_quantizer) for one layer; None means full precision.

    calib: optional dict with the trained base-width FP32 layer ("kernel", "bias" arrays). If given,
    the latent kernel grid is MSE-optimal for the trained kernel and the bias range covers max|b|
    (range_mode "calib"); otherwise the kernel grid is MSE-optimal for the init std (range_mode "init").
    """
    if precision == "fp32":
        return None, None
    wsamp = None if calib is None else calib["kernel"]
    if precision in ("ternary", "binary"):
        if role == "hidden":
            kq = WeightQuantizer(precision, None, scale_mode, sigma_ref, samples=wsamp)
        else:
            kq = WeightQuantizer("fixed", EXTREME_EDGE_BITS, scale_mode, sigma_ref, samples=wsamp)
        bias_bits = MIN_BIAS_BITS
    else:
        b = _bits(precision)
        kq = WeightQuantizer("fixed", b, scale_mode, sigma_ref, samples=wsamp)
        bias_bits = max(b, MIN_BIAS_BITS)
    if calib is None:
        bq = WeightQuantizer("fixed", bias_bits, "latent", sigma_ref=0.25)
    else:
        bq = WeightQuantizer("fixed", bias_bits, "latent", s=cover_po2(np.abs(calib["bias"]).max()))
    return kq, bq


def act_quantizer(precision, calib_acts=None):
    """calib_acts: optional post-ReLU activation samples of this layer from the calibration model."""
    if precision == "fp32":
        return None
    bits = ACT_BITS_FOR_EXTREME if precision in ("ternary", "binary") else _bits(precision)
    return ActQuantizer(bits, samples=calib_acts)
