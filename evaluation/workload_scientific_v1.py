"""Workload 1 (v1, frozen): Lennard-Jones cluster.

Original M1-era implementation with axis-reduction workarounds
(matmul-with-ones, manual reshapes, explicit transpose calls).
Preserved for version-controlled performance attribution; see
workload_scientific.py (v2, natural syntax) and evaluation reports.

Covers: broadcasting arithmetic, 3-D shape transforms, matmul-based
axis reductions, whole-array reductions, math functions, a symbolic
derivative cross-checked numerically, reverse-mode gradients vs an
analytic native reference, and a symmetric eigendecomposition.

Deliberate workarounds used (recorded as findings, not hidden):
- axis sums via matmul with a ones vector (no axis= reductions);
- newaxis construction via reshape (no None-indexing);
- diagonal handling via where (no triu/diag extraction);
- data enters operators only through parameters ({R}, {O}).
"""
from __future__ import annotations

import numpy as np

from tensorlang import P, T, diff, eigh, grad, reshape, simplify, transpose, where

TINY = 1e-12


# -- TensorLang implementation -------------------------------------------------


def tl_build(n: int):
    """Build (not run) the reusable operators for ``n`` particles."""
    R, O = P("R"), P("O")
    D = reshape(R, (n, 1, 3)) - reshape(R, (1, n, 3))
    R2m = reshape(reshape(D * D, (n * n, 3)) @ O, (n, n))
    safe = where(R2m > TINY, R2m, 1.0)
    inv6 = (1.0 / safe) ** 3
    V0 = where(R2m > TINY, 4.0 * (inv6 * inv6 - inv6), 0.0)
    energy = V0.sum() / 2.0
    gradient = grad(energy, wrt="R")
    C = transpose(R) @ R
    inertia = (C * T(np.eye(3))).sum() * T(np.eye(3)) - C
    com = transpose(O) @ R
    r = P("r")
    force_law = simplify(diff(4.0 * ((1.0 / r) ** 12 - (1.0 / r) ** 6)))
    return {"energy": energy, "gradient": gradient, "inertia": inertia,
            "com": com, "force_law": force_law}


def tl_energy(ops, positions: np.ndarray) -> float:
    ones = T(np.ones((3, 1)))
    return float(ops["energy"](T(positions), ones).collect())


def tl_gradient(ops, positions: np.ndarray) -> np.ndarray:
    n = positions.shape[0]
    ones = T(np.ones((3, 1)))
    return np.asarray(ops["gradient"](T(positions), ones).collect())


def tl_inertia(ops, positions: np.ndarray) -> np.ndarray:
    return np.asarray(ops["inertia"](T(positions)).collect())


def tl_com(ops, positions: np.ndarray) -> np.ndarray:
    n = positions.shape[0]
    ones = T(np.ones((n, 1)) / n)
    return np.asarray(ops["com"](ones, T(positions)).collect()).ravel()


def tl_force_law(ops, points: np.ndarray) -> np.ndarray:
    return np.asarray(ops["force_law"](T(points)).collect())


# -- native NumPy/SciPy reference -------------------------------------------------


def np_energy(positions: np.ndarray) -> float:
    d = positions[:, None, :] - positions[None, :, :]
    r2 = (d ** 2).sum(-1)
    np.fill_diagonal(r2, np.inf)
    inv6 = (1.0 / r2) ** 3
    return float((4.0 * (inv6 ** 2 - inv6)).sum() / 2.0)


def np_gradient(positions: np.ndarray) -> np.ndarray:
    """Analytic LJ forces (hand-derived; the fair O(N^2) reference)."""
    d = positions[:, None, :] - positions[None, :, :]
    r2 = (d ** 2).sum(-1)
    np.fill_diagonal(r2, np.inf)
    inv2 = 1.0 / r2
    inv6 = inv2 ** 3
    magnitude = 24.0 * (2.0 * inv6 ** 2 - inv6) * inv2
    return np.asarray(-(magnitude[..., None] * d).sum(axis=1))


def np_inertia(positions: np.ndarray) -> np.ndarray:
    C = positions.T @ positions
    return np.trace(C) * np.eye(3) - C


def np_com(positions: np.ndarray) -> np.ndarray:
    return positions.mean(axis=0)


def make_positions(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((n, 3)) + 0.5) * 5.0


def make_positions_conditioned(n: int, min_sep: float = 1.5,
                               seed: int = 0) -> np.ndarray:
    """Well-conditioned configuration: jittered cubic lattice with a
    controlled minimum separation (avoids r^-13 blowups so absolute and
    relative gradient discrepancies stay comparable)."""
    rng = np.random.default_rng(seed)
    side = int(np.ceil(n ** (1 / 3)))
    grid = np.stack(np.meshgrid(*[np.arange(side)] * 3,
                                indexing="ij")).reshape(3, -1).T[:n]
    pts = (grid + 0.5 + (rng.random((n, 3)) - 0.5) * 0.5) * min_sep
    d = pts[:, None, :] - pts[None, :, :]
    r2 = (d ** 2).sum(-1)
    np.fill_diagonal(r2, np.inf)
    assert r2.min() >= (min_sep * 0.4) ** 2, np.sqrt(r2.min())
    return pts
