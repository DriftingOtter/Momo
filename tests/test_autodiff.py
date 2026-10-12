"""v0.4 tests: reverse-mode automatic differentiation."""
import numpy as np
import polars as pl
import pytest

import tensorlang as tl
from tensorlang import (
    P,
    T,
    TLDiffError,
    TLTypeError,
    grad,
    run_source,
    value_and_grad,
    vjp,
)

autograd = pytest.importorskip("autograd")


@pytest.fixture
def x():
    return T(np.array([1.0, 2.0, 4.0]))


def fd(op, xv, eps=1e-6):
    g = np.zeros_like(xv, dtype=np.float64)
    for i in range(xv.size):
        xp = xv.copy()
        xp.flat[i] += eps
        xm = xv.copy()
        xm.flat[i] -= eps
        g.flat[i] = (op(T(xp)).collect() - op(T(xm)).collect()) / (2 * eps)
    return g


def test_reference_example(x):
    Pm, Y = P("P"), P("Y")
    loss = ((Pm - Y) ** 2).mean()
    d_loss = grad(loss, wrt="P")
    target = T([0.0, 1.0, 1.0])
    assert d_loss(x, target).collect().tolist() == pytest.approx(
        [2 / 3, 2 / 3, 2.0])
    both = value_and_grad(loss, wrt="P")
    value, gradient = both(x, target)
    assert value.collect() == pytest.approx(11 / 3)
    assert gradient.collect().tolist() == pytest.approx([2 / 3, 2 / 3, 2.0])


def test_value_and_grad_shares_forward(x):
    loss = (P("X") ** 2).mean()
    both = value_and_grad(loss)
    value, gradient = both(x)
    assert value.collect() == pytest.approx(7.0)
    assert gradient.collect().tolist() == pytest.approx([2 / 3, 4 / 3, 8 / 3])
    assert gradient.root.shared.forward_runs == 1
    value.collect()
    gradient.collect()
    assert gradient.root.shared.forward_runs == 1


def test_scalar_derivative():
    assert grad(P("X") ** 3)(T(2.0)).collect() == pytest.approx(12.0)


def test_multi_wrt_returns_tuple_in_order():
    X, Y = P("X"), P("Y")
    loss = ((X - Y) ** 2).mean()
    d = grad(loss, wrt=["Y", "X"])
    gy, gx = d(T([1.0, 2.0]), T([0.0, 0.0]))
    assert gy.collect().tolist() == pytest.approx([-1.0, -2.0])
    assert gx.collect().tolist() == pytest.approx([1.0, 2.0])


def test_default_wrt_single_param():
    assert grad(P("Q") * 3)(T(2.0)).collect() == pytest.approx(3.0)
    with pytest.raises(TLDiffError):
        grad(P("A") + P("B"))


def test_reused_nodes_accumulate_once():
    X = P("X")
    y = X * 2
    loss = (y + y).sum()  # y used twice: d = 2 + 2
    assert grad(loss)(T([1.0, 2.0])).collect().tolist() == [4.0, 4.0]


def test_broadcasting_gradients():
    X = P("X")
    M = np.array([[1.0, 2.0], [3.0, 4.0]])
    v = np.array([10.0, 20.0])
    loss = (X * P("Y")).sum()
    d = grad(loss, wrt="X")
    assert d(T(M), T(v)).collect().tolist() == [[10.0, 20.0], [10.0, 20.0]]
    d2 = grad(loss, wrt="Y")
    assert d2(T(M), T(v)).collect().tolist() == pytest.approx([4.0, 6.0])


def test_matmul_gradients():
    rng = np.random.default_rng(2)
    A = rng.random((3, 4)) + 0.5
    B = rng.random((4, 2)) + 0.5
    X, Y = P("X"), P("Y")
    loss = (X @ Y).sum()
    dA, dB = grad(loss, wrt=["X", "Y"])(T(A), T(B))
    assert np.allclose(dA.collect(), np.ones((3, 2)) @ B.T)
    assert np.allclose(dB.collect(), A.T @ np.ones((3, 2)))
    v = rng.random(4) + 0.5
    lv = (X @ Y).sum()
    d = grad(lv, wrt="X")(T(v), T(B))
    assert np.allclose(d.collect(), np.ones(2) @ B.T)


def test_finite_difference_sweep():
    rng = np.random.default_rng(0)
    X = P("X")
    x = rng.random(6) + 0.5
    S = lambda e: e.sum()  # noqa: E731
    cases = [
        S(X + 2.0), S(X * X), S(X / (X + 1.0)), S(X**3), S(-X), S(abs(X)),
        S(tl.sin(X)), S(tl.exp(X / 2)), S(tl.log(X)), S(tl.sqrt(X)),
        S(tl.cos(X)), S(tl.tan(X / 3)), (X * 2).sum(), (X**2).mean(),
        (X**2).min(), (X**2).max(), (X**2).std(), (X**2).var(),
        (tl.reshape(X, (2, 3)) @ tl.transpose(tl.reshape(X, (2, 3)))).sum(),
        (tl.transpose(tl.reshape(X, (2, 3))) ** 2).sum(),
        (tl.reshape(X, (3, 2)) * 2).sum(),
        (tl.flatten(tl.reshape(X, (2, 3))) + 1).sum(),
        tl.concat([X, X * 2]).sum(), tl.stack([X, X]).sum(),
        (X[1:4] ** 2).sum(), tl.where(X > 0.7, X, X * 0).sum(),
        tl.clip(X, 0.6, 1.2).sum(),
        (tl.moveaxis(tl.reshape(X, (2, 3)), 0, 1) ** 2).sum(),
        tl.norm(X),
    ]
    for op in cases:
        assert np.allclose(grad(op)(T(x)).collect(), fd(op, x),
                           rtol=1e-4, atol=1e-6)


def test_sympy_cross_validation():
    import sympy as sp

    x = sp.Symbol("X")
    expr = sp.sin(x) * sp.exp(-x**2)
    f = tl.sin(P("X")) * tl.exp(-P("X") ** 2)
    pts = np.linspace(0.2, 2.0, 9)
    expected = sp.lambdify(x, sp.diff(expr, x), "numpy")(pts)
    assert np.allclose(grad(f.sum())(T(pts)).collect(), expected, rtol=1e-6)


def test_autograd_cross_validation(x):
    import autograd.numpy as anp
    from autograd import grad as _ag

    f = lambda v: anp.mean((v - anp.array([0.0, 1.0, 1.0])) ** 2)
    Pm, Y = P("P"), P("Y")
    loss = ((Pm - Y) ** 2).mean()
    assert np.allclose(
        grad(loss, wrt="P")(x, T([0.0, 1.0, 1.0])).collect(),
        _ag(f)(x.collect()))


def test_composition_differentiates():
    X = P("X")
    pipe = (X**2) >> (P("Y") + 5) >> (P("Z") * 2)
    loss = pipe.sum()
    assert grad(loss)(T(3.0)).collect() == pytest.approx(12.0)  # 2*3*2
    assert grad(loss)(T(np.array([1.0, 2.0]))).collect().tolist() == [4.0, 8.0]


def test_solve_and_norm_gradients():
    A = np.array([[3.0, 1.0], [1.0, 2.0]])
    b = np.array([9.0, 8.0])
    X, Y = P("X"), P("Y")
    got = grad(tl.solve(X, Y).sum(), wrt=["X", "Y"])(T(A), T(b))
    dA, dB = (g.collect() for g in got)
    assert np.allclose(dB, np.linalg.solve(A.T, np.ones(2)))
    assert np.allclose(dA, -np.outer(dB, np.linalg.solve(A, b)))
    v = np.array([3.0, 4.0])
    assert grad(tl.norm(P("X")))(T(v)).collect().tolist() == \
        pytest.approx([0.6, 0.8])


def test_structured_gradients_keep_labels():
    t = T({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    D = P("D")
    loss = ((D.a - D.b) ** 2).mean()
    out = grad(loss)(t).collect()
    assert out.columns == ["a", "b"]
    n = 3
    assert np.allclose(out["a"].to_list(),
                       [2 * (a - b) / n for a, b in zip([1, 2, 3], [4, 5, 6])])
    s = T({"x": [1.0, 2.0, 4.0]})
    gout = grad((P("D").x ** 2).mean())(s).collect()
    assert gout.columns == ["x"]  # frame input → frame gradient, labels kept
    assert gout["x"].to_list() == pytest.approx([2 / 3, 4 / 3, 8 / 3])
    import polars as _pl

    st = T(_pl.Series("x", [1.0, 2.0, 4.0]))
    sout = grad((P("D") ** 2).mean())(st).collect()
    assert isinstance(sout, _pl.Series) and sout.name == "x"
    assert sout.to_list() == pytest.approx([2 / 3, 4 / 3, 8 / 3])


def test_mixed_structured_rejected():
    t = T({"a": [1.0, 2.0], "s": ["x", "y"]})
    with pytest.raises(TLDiffError):
        grad((P("D").a + 1).mean())(t).collect()


def test_unsupported_operations():
    X = P("X")
    x = T([3.0, 1.0, 2.0])
    with pytest.raises(TLDiffError):
        grad((X > 1).sum())(x).collect()
    with pytest.raises(TLDiffError):
        grad(tl.sort(X).sum())(x).collect()
    with pytest.raises(TLDiffError):
        grad(tl.unique(X).sum())(x).collect()
    with pytest.raises(TLDiffError):
        grad(tl.einsum("i,i->", X, X))(x).collect()


def test_domain_and_shape_errors():
    X = P("X")
    with pytest.raises(TLDiffError):
        grad(X + 1)(T([1.0])).collect()  # vector output rejected
    with pytest.raises(TLDiffError):
        grad(X + 1, wrt="Q")
    with pytest.raises(TLDiffError):
        grad(X + 1, wrt=[])
    with pytest.raises(TLTypeError):
        grad(T([1.0]))
    with pytest.raises(TLDiffError):
        grad(X + 1)(T([1, 2])).collect()  # integer input rejected
    with pytest.raises(TLDiffError):
        grad(X + 1)(T([True, False])).collect()  # boolean rejected
    with pytest.raises(TLDiffError):
        grad(X + 1)(T(np.array([1 + 1j]))).collect()  # complex rejected


def test_second_order_rejected(x):
    d = grad((P("X") ** 2).sum())
    with pytest.raises(TLDiffError):
        grad(d)
    with pytest.raises(TLDiffError):
        value_and_grad(d)
    # Reusing a computed gradient as data is first-order (stop-gradient):
    # d2 differentiates (Z^2).sum() evaluated AT the inner value.
    applied = d(x)
    d2 = grad((P("Z") ** 2).sum())
    assert d2(applied).collect().tolist() == [4.0, 8.0, 16.0]


def test_unfitted_and_fitted_estimators(x):
    from tensorlang import standardize, pca, normalize

    with pytest.raises(TLDiffError):
        grad(standardize())(x).collect()  # construction-time rejection
    fitted = standardize().fit(T(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])))
    scale = fitted.parameters["scale_"]
    # Scalar-output composition through the fitted affine map.
    pipe = fitted >> P("S").sum()
    got = grad(pipe)(T(np.array([[7.0, 8.0]]))).collect()
    assert np.allclose(got, np.array([[1 / scale[0], 1 / scale[1]]]))
    # Direct VJP with a ones seed gives the same row-summed gradient.
    v = vjp(fitted)(T(np.array([[7.0, 8.0]])), T(np.ones((1, 2))))
    assert np.allclose(v.collect(), np.array([[1 / scale[0], 1 / scale[1]]]))
    with pytest.raises(TLDiffError):
        grad(normalize("l1"))(x).collect()
    p = pca(1).fit(T(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])))
    seed = T(pl.DataFrame({"pc0": [1.0]}))
    dp = vjp(p)(T(np.array([[1.0, 2.0]])), seed).collect()
    assert dp.shape == (1, 2)
    assert np.allclose(dp, p.parameters["components_"][:1])


def test_vjp_custom_seed():
    X = P("X")
    op = (X**2).sum()
    v = vjp(op)(T([1.0, 2.0]), 3.0)
    assert v.collect().tolist() == pytest.approx([6.0, 12.0])
    with pytest.raises(TLDiffError):
        vjp(op)(T([1.0, 2.0]), T([1.0])).collect()  # seed shape mismatch
    with pytest.raises(Exception):
        vjp(op)(T([1.0, 2.0])).collect()  # missing seed: arity error
    X2, Y2 = P("X"), P("Y")
    both = vjp((X2 - Y2).sum(), wrt=["X", "Y"])(T([3.0]), T([1.0]), 2.0)
    gx, gy = (g.collect() for g in both)
    assert gx.tolist() == [2.0] and gy.tolist() == [-2.0]


def test_mutation_and_snapshot_safety(x):
    X = P("X")
    d = grad((X**2).mean())
    pending = d(x)
    (X + 100)(x).inplace()
    assert pending.collect().tolist() == pytest.approx([2 / 3, 4 / 3, 8 / 3])
    y = T([1.0, 2.0])
    grad((P("Q") ** 2).sum())(y).inplace()
    assert y.collect().tolist() == [2.0, 4.0]
    pending.release()
    assert pending._cached is None


def test_tl_frontend_grad_program():
    ns = run_source(
        "from tensorlang import T, P, grad, value_and_grad\n"
        "loss := ((P('P') - P('Y'))**2).mean()\n"
        "d_loss := grad(loss, wrt='P')\n"
        "pred = T([1.0, 2.0, 4.0])\n"
        "target = T([0.0, 1.0, 1.0])\n"
        "g = d_loss(pred, target)\n"
        "both := value_and_grad(loss, wrt='P')\n"
        "v, gg = both(pred, target)\n", filename="g.tl")
    assert ns["g"].collect().tolist() == pytest.approx([2 / 3, 2 / 3, 2.0])
    assert ns["v"].collect() == pytest.approx(11 / 3)
    with pytest.raises(Exception):
        run_source("from tensorlang import T, P, grad\n"
                   "f := {X} + 1\n"
                   "d := grad(f)\n"
                   "r := d(T([1.0]))\n", filename="g.tl")


def test_unused_param_gives_zeros():
    X, Y = P("X"), P("Y")
    loss = (Y**2).mean() + (X * 0.0).sum()  # X present but contributes nothing
    assert grad(loss, wrt="X")(T([5.0]), T([2.0])).collect() == [0.0]


def test_explain_and_profile(x):
    d = grad((P("X") ** 2).mean())
    text = d.explain()
    assert "grad(wrt=['X'])" in text
    assert "differentiable rules" in text
    prof = d(x).profile()
    assert prof.lowered is False
    assert prof.backend == "materialized"
    assert prof.forward_ms is not None and prof.backward_ms is not None
    assert prof.forward_ms >= 0 and prof.backward_ms >= 0
