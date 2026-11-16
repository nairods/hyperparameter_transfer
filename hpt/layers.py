"""Dense layer with explicit (init std, forward multiplier, LR multiplier) and optional QKeras quantisers."""
import tensorflow as tf


class ScaledQDense(tf.keras.layers.Layer):
    """y = multiplier * (x @ Qk(W)) + Qb(b)

    The LR multipliers are not used by the layer itself; they are read by the trainer, which
    assigns each variable to an Adam instance with LR = eta0 * lr_mult.
    """

    def __init__(self, units, rule, kernel_quantizer=None, bias_quantizer=None, role="hidden",
                 seed=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.rule = rule
        self.role = role
        self.kernel_quantizer = kernel_quantizer
        self.bias_quantizer = bias_quantizer
        self.seed = seed

    def build(self, input_shape):
        fan_in = int(input_shape[-1])
        init = (tf.keras.initializers.Zeros() if self.rule.init_std == 0.0 else
                tf.keras.initializers.RandomNormal(stddev=self.rule.init_std, seed=self.seed))
        self.kernel = self.add_weight(name="kernel", shape=(fan_in, self.units), initializer=init,
                                      trainable=True)
        self.bias = self.add_weight(name="bias", shape=(self.units,), initializer="zeros",
                                    trainable=True)
        self.kernel.lr_mult = self.rule.lr_mult
        self.bias.lr_mult = self.rule.bias_lr_mult
        super().build(input_shape)

    def quantized_kernel(self):
        return self.kernel if self.kernel_quantizer is None else self.kernel_quantizer(self.kernel)

    def quantized_bias(self):
        return self.bias if self.bias_quantizer is None else self.bias_quantizer(self.bias)

    def kernel_codes(self):
        if self.kernel_quantizer is None:
            return None
        return self.kernel_quantizer.codes(self.kernel)

    def call(self, x):
        return self.rule.multiplier * tf.matmul(x, self.quantized_kernel()) + self.quantized_bias()
