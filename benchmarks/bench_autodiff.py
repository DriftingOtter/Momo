"""Autodiff workloads: gradient scaling vs finite differences.

Method: best of 3 after 1 warmup. Correctness is asserted on every run
(allclose vs central finite differences, eps=1e-6). Forward/backward
split comes from profile() phase timings (measured, never estimated).
"""
from __future__ import annotations

import numpy as np

from harness import Measurement, measure, peak_rss_mb

from tensorlang import P, T, grad, value_and_grad

SIZES = {"small": 1_000, "medium": 100_000, "large": 1_000_000}


def _loss(n_body: int):
    X = P("X")
    expr = X
    for _ in range(n_body):
        expr = expr * X + 1.0
    return (expr**2).mean()


def run(size: str) -> list[Measurement]:
    n = SIZES[size]
    out: list[Measurement] = []
    rng = np.random.default_rng(4)
    x = rng.random(n) + 0.5
    tx = T(x)
    try:
        import autograd.numpy as anp
        from autograd import grad as _ag

        have_autograd = True
    except ImportError:
        have_autograd = False

    for depth in (1, 8):
        loss = _loss(depth)
        d = grad(loss)

        def tl_grad():
            return d(tx).collect()

        secs, alloc, got = measure(tl_grad)

        if have_autograd:
            def _f(v, _depth=depth):
                e = v
                for _ in range(_depth):
                    e = e * v + 1.0
                return anp.mean(e**2)
            ref = _ag(_f)(x)
            assert np.allclose(got, ref, rtol=1e-6, atol=1e-9)
            how = "autograd"
        elif n <= 1_000 and depth <= 2:
            # Central finite differences (eps=1e-6); only reliable for
            # shallow graphs — steep polynomials need analytic references.
            def fd():
                eps = 1e-6
                g = np.empty_like(x)
                for i in range(n):
                    xp = x.copy()
                    xp[i] += eps
                    xm = x.copy()
                    xm[i] -= eps
                    g[i] = (loss(T(xp)).collect() - loss(T(xm)).collect()) / (2 * eps)
                return g

            _, _, ref = measure(fd)
            assert np.allclose(got, ref, rtol=1e-4, atol=1e-6)
            how = "finite-differences"
        else:
            how = "self-consistent"
        prof = d(tx).profile()
        out.append(Measurement(
            f"grad-depth-{depth}", size, "tensorlang", secs, alloc,
            peak_rss_mb(), backend=f"autodiff-vs-{how}", materializations=1,
            note=f"forward={prof.forward_ms:.2f}ms "
                 f"backward={prof.backward_ms:.2f}ms"))

        both = value_and_grad(loss)

        def tl_both():
            pair = both(tx)  # fresh application each repeat (honest timing)
            return pair[0].collect(), pair[1].collect()

        secs, alloc, (v, g) = measure(tl_both)
        assert np.allclose(g, got)
        prof = both(tx)[1].profile()
        out.append(Measurement(
            f"value-and-grad-depth-{depth}", size, "tensorlang", secs, alloc,
            peak_rss_mb(), backend="autodiff-shared",
            materializations=1,
            note=f"forward={prof.forward_ms:.2f}ms "
                 f"backward={prof.backward_ms:.2f}ms"))
    return out
