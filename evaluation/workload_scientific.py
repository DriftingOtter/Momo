"""Workload 1 (v2): Lennard-Jones cluster with natural tensor syntax.

v0.6 M2 rewrite of workload_scientific_v1.py: axis reductions replace
matmul-with-ones, `.T` replaces explicit transpose calls, and no manual
reshape scaffolding remains. Same model, data, native reference, and
thresholds as v1.
"""
from __future__ import annotations

import numpy as np

from tensorlang import P, T, diff, eigh, grad, reshape, simplify, where
from workload_scientific_v1 import TINY


def tl_build(n: int):
    """Build (not run) the reusable operators for ``n`` particles."""
    R = P("R")
    D = reshape(R, (n, 1, 3)) - reshape(R, (1, n, 3))
    R2m = (D * D).sum(axis=2)
    safe = where(R2m > TINY, R2m, 1.0)
    inv6 = (1.0 / safe) ** 3
    V0 = where(R2m > TINY, 4.0 * (inv6 * inv6 - inv6), 0.0)
    energy = V0.sum() / 2.0
    gradient = grad(energy, wrt="R")
    C = R.T @ R
    inertia = (C * T(np.eye(3))).sum() * T(np.eye(3)) - C
    com = R.mean(axis=0)
    r = P("r")
    force_law = simplify(diff(4.0 * ((1.0 / r) ** 12 - (1.0 / r) ** 6)))
    return {"energy": energy, "gradient": gradient, "inertia": inertia,
            "com": com, "force_law": force_law}


def tl_energy(ops, positions: np.ndarray) -> float:
    return float(ops["energy"](T(positions)).collect())


def tl_gradient(ops, positions: np.ndarray) -> np.ndarray:
    return np.asarray(ops["gradient"](T(positions)).collect())


def tl_inertia(ops, positions: np.ndarray) -> np.ndarray:
    return np.asarray(ops["inertia"](T(positions)).collect())


def tl_com(ops, positions: np.ndarray) -> np.ndarray:
    return np.asarray(ops["com"](T(positions)).collect()).ravel()


def tl_force_law(ops, points: np.ndarray) -> np.ndarray:
    return np.asarray(ops["force_law"](T(points)).collect())
