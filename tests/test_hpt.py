import math

import numpy as np
import pytest
import tensorflow as tf

from hpt.model import build_mlp
from hpt.parametrization import ARMS, get_tensor_rule, layer_shapes
from hpt.quant import WeightQuantizer, optimal_po2_scale


def test_mup_equals_sp_at_base_width():
    for role, fi, fo in layer_shapes(64):
        assert get_tensor_rule("mup", role, fi, fo, 64, 64) == get_tensor_rule("sp", role, fi, fo, 64, 64)
        assert get_tensor_rule("sp_sqrtD", role, fi, fo, 64, 64) == get_tensor_rule("sp", role, fi, fo, 64, 64)


def test_effective_update_scaling():
    """Adam moves the stored weight by ~eta*lr_mult; effective change = multiplier * that.
    muP and u-muP hidden layers must both have effective LR ~ 1/fan_in (relative to base)."""
    for D in (64, 256, 1024):
        m = get_tensor_rule("mup", "hidden", D, D, D, 64)
        u = get_tensor_rule("umup", "hidden", D, D, D, 64)
        assert m.lr_mult * m.multiplier == pytest.approx(64 / D)
        assert u.lr_mult * u.multiplier == pytest.approx(math.sqrt(2) / D)
        # init: effective std of mup and umup hidden weights coincide
        assert m.init_std * m.multiplier == pytest.approx(u.init_std * u.multiplier)


@pytest.mark.parametrize("arm", ["mup", "umup"])
def test_preactivation_rms_width_independent(arm):
    x = tf.constant(np.random.default_rng(0).normal(size=(4096, 16)).astype("float32"))
    rms = []
    for D in (64, 1024):
        model, _ = build_mlp(arm, D, seed=1)
        _, pre = model(x, return_hidden=True)
        rms.append([float(tf.sqrt(tf.reduce_mean(z ** 2))) for z in pre[:-1]])
    np.testing.assert_allclose(rms[0], rms[1], rtol=0.1)
    if arm == "umup":
        np.testing.assert_allclose(rms[0], 1.0, rtol=0.15)


def test_ste_gradient_passes():
    w = tf.Variable(tf.random.normal((32, 8)))
    q = WeightQuantizer("fixed", 4, "latent", sigma_ref=1.0)
    with tf.GradientTape() as t:
        y = tf.reduce_sum(q(w))
    g = t.gradient(y, w).numpy()
    assert np.isfinite(g).all() and np.abs(g).sum() > 0


@pytest.mark.parametrize("kind,bits", [("fixed", 8), ("fixed", 4), ("fixed", 2), ("ternary", None),
                                       ("binary", None)])
def test_codes_are_exact(kind, bits):
    q = WeightQuantizer(kind, bits, "latent", sigma_ref=0.2)
    w = tf.random.normal((64, 64), stddev=0.2)
    c = q.codes(w).numpy()
    np.testing.assert_array_equal(c, np.round(c))
    np.testing.assert_allclose(c * q.s * q.lsb, q(w).numpy(), atol=1e-6)
    assert len(np.unique(c)) <= (2 ** bits if bits else 3)


def test_po2_scale_tracks_sigma():
    s1 = optimal_po2_scale("fixed", 4, 1.0)
    s2 = optimal_po2_scale("fixed", 4, 1.0 / 8)
    assert s1 / s2 == pytest.approx(8.0)


def test_group_lrs_applied():
    """One Adam step moves each variable by ~eta0*lr_mult (Adam's first step is sign-like)."""
    from hpt.train import main  # noqa: F401  (import check)
    model, _ = build_mlp("mup", 256, seed=0)
    eta0 = 1e-3
    x = tf.random.normal((256, 16))
    y = tf.random.uniform((256,), 0, 5, dtype=tf.int64)
    before = [v.numpy().copy() for v in model.trainable_variables]
    for v in model.trainable_variables:
        opt = tf.keras.optimizers.Adam(eta0 * v.lr_mult)
        with tf.GradientTape() as t:
            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
                y, model(x), from_logits=True))
        g = t.gradient(loss, v)
        opt.apply_gradients([(g, v)])
    for b, v in zip(before, model.trainable_variables):
        step = np.abs(v.numpy() - b)
        moved = step[step > 0]
        if moved.size:
            assert np.median(moved) == pytest.approx(eta0 * v.lr_mult, rel=0.05)


def test_auc_scores():
    from hpt.train import auc_scores
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 5000)
    perfect = np.eye(5)[y]
    assert auc_scores(y, perfect)[0] == pytest.approx(1.0)
    rand = rng.dirichlet(np.ones(5), 5000)
    assert auc_scores(y, rand)[0] == pytest.approx(0.5, abs=0.03)


def test_run_id_backward_compatible():
    """Adding optional keys (range_mode='init', tag='') must not change existing run ids."""
    from hpt.train import CONFIG_KEYS, parse_args, run_id
    a = parse_args(["--arm", "mup", "--width", "64", "--eta0", "0.01"])
    cfg = {k: getattr(a, k) for k in CONFIG_KEYS}
    old = {k: v for k, v in cfg.items() if k not in ("range_mode", "tag")}
    assert run_id(cfg) == run_id(old)
    assert run_id({**cfg, "tag": "own"}) != run_id(cfg)


def test_resmlp_depth_scaling():
    """depth-muP: block multiplier and block LR x sqrt(L_base/L); identity at L = L_base."""
    from hpt.model import build_resmlp
    x = tf.random.normal((64, 16))
    m2, info2 = build_resmlp("mup", 32, depth=2, depth_scaling="sqrt", depth_base=2)
    m2n, info2n = build_resmlp("mup", 32, depth=2, depth_scaling="none", depth_base=2)
    assert m2.branch_mult == m2n.branch_mult == 1.0
    m8, info8 = build_resmlp("mup", 32, depth=8, depth_scaling="sqrt", depth_base=2)
    assert m8.branch_mult == pytest.approx(0.5)
    blk = [i for i in info8 if i["role"] == "hidden"]
    assert len(blk) == 8 and blk[0]["lr_mult"] == pytest.approx(0.5 * 64 / 32)
    out, pre = m8(x, return_hidden=True)
    assert out.shape == (64, 5) and len(pre) == 10


def test_new_keys_keep_run_ids():
    from hpt.train import CONFIG_KEYS, parse_args, run_id
    a = parse_args(["--arm", "sp", "--width", "128", "--eta0", "0.004", "--precision", "fixed4"])
    cfg = {k: getattr(a, k) for k in CONFIG_KEYS}
    legacy = {k: v for k, v in cfg.items()
              if k not in ("range_mode", "tag", "arch", "depth_scaling", "depth_base")}
    assert run_id(cfg) == run_id(legacy)
    assert run_id({**cfg, "arch": "resmlp"}) != run_id(cfg)
