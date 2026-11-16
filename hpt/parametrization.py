"""Per-tensor rules (init std, forward multiplier, Adam LR multiplier) for SP / muP / u-muP.

Effective layer:  y = multiplier * (x @ Q(W)) + b,   W ~ N(0, init_std^2),   Adam LR = eta0 * lr_mult.

Roles: "input" (16 -> D, fan_in fixed), "hidden" (D -> D), "output" (D -> 5, fan_out fixed).
ReLU gain sqrt(2) is applied to layers whose input is a ReLU output (hidden, output) so that
pre-activations are ~unit RMS at init in every arm.

Arms
----
sp        standard parametrisation, global LR eta0 (He/LeCun init).
sp_sqrtD  SP with global LR eta0 * sqrt(d_base / D)  (the eta ~ 1/sqrt(D) hypothesis).
mup       muP (Tensor Programs V, Adam), written so that mup == sp exactly at D == d_base:
            hidden LR eta0 * d_base/fan_in, readout multiplier d_base/fan_in (mup-package MuReadout style).
umup      u-muP (Blake et al. 2024, Table 2): unit-variance weights, width-dependent scale in the
            forward multiplier; hidden LR eta0 / sqrt(fan_in), output multiplier 1/fan_in.
"""
import math
from dataclasses import dataclass, asdict

ARMS = ("sp", "sp_sqrtD", "mup", "umup")
ROLES = ("input", "hidden", "output")
RELU_GAIN = math.sqrt(2.0)


@dataclass(frozen=True)
class TensorRule:
    init_std: float        # std of the stored (latent) kernel
    multiplier: float      # constant forward multiplier applied after the matmul
    lr_mult: float         # kernel Adam LR = eta0 * lr_mult
    bias_lr_mult: float    # bias Adam LR = eta0 * bias_lr_mult

    def as_dict(self):
        return asdict(self)


def get_tensor_rule(arm, role, fan_in, fan_out, width, d_base=64,
                    umup_input_lr="const", readout_zero_init=False):
    """Return the TensorRule for one dense layer.

    umup_input_lr: "const" (muP / muS style, default) or "fan_out" (u-muP paper's eta/sqrt(fan_out)).
    readout_zero_init: zero-initialise the output kernel (TP-V recommendation) in mup / umup.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; choose from {ARMS}")
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}")
    gain = 1.0 if role == "input" else RELU_GAIN

    if arm in ("sp", "sp_sqrtD"):
        std = (gain if role == "hidden" else 1.0) / math.sqrt(fan_in)
        lr = 1.0 if arm == "sp" else math.sqrt(d_base / width)
        return TensorRule(std, 1.0, lr, lr)

    if arm == "mup":
        if role == "input":
            return TensorRule(1.0 / math.sqrt(fan_in), 1.0, 1.0, 1.0)
        if role == "hidden":
            return TensorRule(gain / math.sqrt(fan_in), 1.0, d_base / fan_in, 1.0)
        std = 0.0 if readout_zero_init else 1.0 / math.sqrt(fan_in)
        return TensorRule(std, d_base / fan_in, 1.0, 1.0)

    # umup
    if role == "input":
        lr = 1.0 if umup_input_lr == "const" else 1.0 / math.sqrt(fan_out)
        return TensorRule(1.0, 1.0 / math.sqrt(fan_in), lr, 1.0)
    if role == "hidden":
        return TensorRule(1.0, gain / math.sqrt(fan_in), 1.0 / math.sqrt(fan_in), 1.0)
    return TensorRule(0.0 if readout_zero_init else 1.0, 1.0 / fan_in, 1.0, 1.0)


def layer_shapes(width, depth=3, n_in=16, n_out=5):
    """[(role, fan_in, fan_out), ...] for an MLP with `depth` hidden layers of size `width`."""
    shapes = [("input", n_in, width)]
    shapes += [("hidden", width, width)] * (depth - 1)
    shapes.append(("output", width, n_out))
    return shapes
