"""App A tests: convergence, determinism, module reuse (public API only)."""
import os

import numpy as np

from tensorlang import T, run_file

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_main():
    return run_file(os.path.join(APP, "main.tl"))


def test_converges_to_reference():
    run_main()
    got = np.load(os.path.join(APP, "out_params.npz"))["params"]
    assert np.allclose(got, [2.5, 0.8, 0.3], atol=0.02)


def test_deterministic():
    run_main()
    first = np.load(os.path.join(APP, "out_params.npz"))["params"].copy()
    run_main()
    second = np.load(os.path.join(APP, "out_params.npz"))["params"]
    assert np.array_equal(first, second)


def test_imported_operator_reuse():
    eq = run_file(os.path.join(APP, "models", "equations.tl"))
    out = eq["predict"](T(2.0), T(0.5), T([1.0]), T(0.1)).collect()
    assert np.allclose(np.asarray(out), [2.0 * np.exp(-0.5) + 0.1])


def test_loss_module_standalone():
    loss = run_file(os.path.join(APP, "models", "loss.tl"))
    assert list(loss["mse"].params) == ["A", "K", "T", "C", "Y"]
