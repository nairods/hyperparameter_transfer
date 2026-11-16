"""MLP builder: 16 -> D x depth -> 5 under a given parametrisation arm and precision."""
import tensorflow as tf

from .layers import ScaledQDense
from .parametrization import get_tensor_rule, layer_shapes
from .quant import act_quantizer, layer_quantizers


class MLP(tf.keras.Model):
    def __init__(self, dense_layers, act_qs=None, **kwargs):
        super().__init__(**kwargs)
        self.dense_layers = dense_layers
        self.act_qs = act_qs or [None] * (len(dense_layers) - 1)

    def _act(self, h, i):
        q = self.act_qs[i]
        return tf.nn.relu(h) if q is None else q(h)

    def call(self, x, return_hidden=False):
        pre = []
        h = x
        for i, layer in enumerate(self.dense_layers):
            z = layer(h)
            pre.append(z)
            h = self._act(z, i) if i < len(self.dense_layers) - 1 else z
        return (h, pre) if return_hidden else h


def build_mlp(arm, width, depth=3, precision="fp32", scale_mode="latent", d_base=64, seed=0,
              n_in=16, n_out=5, umup_input_lr="const", readout_zero_init=False, calib=None):
    """calib: optional list (one per dense layer) of {"kernel", "bias", "acts"} from a trained FP32
    model at d_base; fixes the quantiser ranges ("range_mode calib"). None -> init-based ranges."""
    layers, info, act_qs = [], [], []
    for i, (role, fan_in, fan_out) in enumerate(layer_shapes(width, depth, n_in, n_out)):
        rule = get_tensor_rule(arm, role, fan_in, fan_out, width, d_base, umup_input_lr,
                               readout_zero_init)
        # the latent grid is frozen at the base width: reference std of the stored kernel at d_base
        ref_fan_in = fan_in if role == "input" else d_base
        ref_rule = get_tensor_rule(arm, role, ref_fan_in, fan_out if role == "output" else d_base,
                                   d_base, d_base, umup_input_lr, readout_zero_init=False)
        lc = None if calib is None else calib[i]
        kq, bq = layer_quantizers(precision, role, scale_mode, ref_rule.init_std, lc)
        if role != "output":
            act_qs.append(act_quantizer(precision, None if lc is None else lc["acts"]))
        layers.append(ScaledQDense(fan_out, rule, kq, bq, role=role, seed=seed * 1000 + i,
                                   name=f"dense_{i}_{role}"))
        info.append({"role": role, "fan_in": fan_in, "fan_out": fan_out, **rule.as_dict(),
                     "kernel_quantizer": kq.describe() if kq else None,
                     "bias_s": bq.s if bq else None,
                     "act_s": act_qs[-1].s if role != "output" and act_qs[-1] is not None else None})
    model = MLP(layers, act_qs)
    model.build((None, n_in))
    return model, info
