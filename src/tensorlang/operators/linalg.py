"""Numerical linear algebra: SVD, eigendecomposition, solves, norms.

Stable SciPy/NumPy routines throughout.  Multi-output decompositions
return explicit tuple results (``SVDResult`` / ``EighResult``) whose three
components share one computation.
"""
from __future__ import annotations

import numpy as np

from ..core import expression as ir
from ..core.errors import TLShapeError, TLTypeError


class _Shared:
    """Memoized multi-output computation shared by component Lazys."""

    def __init__(self, compute) -> None:
        self._compute = compute
        self._done = False
        self._parts: tuple | None = None

    def get(self) -> tuple:
        if not self._done:
            self._parts = tuple(self._compute())
            self._done = True
        assert self._parts is not None
        return self._parts


class SVDResult(tuple):
    """``(U, S, Vh)`` lazy triple from ``svd()``; unpacks like a tuple."""

    @property
    def U(self):
        return self[0]

    @property
    def S(self):
        return self[1]

    @property
    def Vh(self):
        return self[2]


class EighResult(tuple):
    """``(values, vectors)`` lazy pair from ``eigh()``."""

    @property
    def values(self):
        return self[0]

    @property
    def vectors(self):
        return self[1]


def svd(x, full_matrices: bool = True) -> SVDResult:
    """Singular value decomposition: ``U, S, Vh = svd(A)``.

    All three components share one LAPACK computation.  Sign ambiguity of
    singular vectors is inherent: compare reconstructions or subspaces,
    not raw signs.
    """
    from ..core.tensor import Lazy, expr_of

    if isinstance(x, Lazy):
        root: ir.Expr = x.root
    elif hasattr(x, "_snapshot_node"):
        root = x._snapshot_node()
    else:
        root = expr_of(x, "svd")
    shared = _Shared(lambda: _compute_svd(root, bool(full_matrices)))
    return SVDResult((
        Lazy(ir.SvdPart(shared, 0)),
        Lazy(ir.SvdPart(shared, 1)),
        Lazy(ir.SvdPart(shared, 2)),
    ))


def _compute_svd(root: ir.Expr, full_matrices: bool):
    import scipy.linalg as _la

    from ..runtime.executor import evaluate

    value, kind = evaluate(root)
    if kind != "dense":
        raise TLTypeError(
            f"svd needs dense numerical data, got {kind}; select numeric "
            f"fields explicitly for structured data"
        )
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2:
        raise TLShapeError(f"svd needs a rank-2 matrix, got shape {arr.shape}",
                           code="TL0507",
                           details={"shape": tuple(arr.shape)})
    try:
        u, s, vh = _la.svd(arr, full_matrices=full_matrices)
    except Exception as exc:
        raise TLTypeError(f"svd failed: {exc}") from exc
    return u, s, vh


def evaluate_svd_part(shared: _Shared, index: int):
    return shared.get()[index], "dense"


def eigh(x) -> EighResult:
    """Symmetric/Hermitian eigendecomposition: ``values, vectors = eigh(A)``.

    ``A`` must be square and (within tolerance) symmetric; general
    matrices should use ``svd`` instead.  Eigenvalues ascend.
    """
    from ..core.tensor import Lazy, expr_of

    if isinstance(x, Lazy):
        root = x.root
    elif hasattr(x, "_snapshot_node"):
        root = x._snapshot_node()
    else:
        root = expr_of(x, "eigh")
    shared = _Shared(lambda: _compute_eigh(root))
    return EighResult((Lazy(ir.EighPart(shared, 0)), Lazy(ir.EighPart(shared, 1))))


def _compute_eigh(root: ir.Expr):
    import scipy.linalg as _la

    from ..runtime.executor import evaluate

    value, kind = evaluate(root)
    if kind != "dense":
        raise TLTypeError(f"eigh needs dense numerical data, got {kind}")
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise TLShapeError(f"eigh needs a square matrix, got shape {arr.shape}",
                           code="TL0507",
                           details={"shape": tuple(arr.shape)})
    if not np.allclose(arr, arr.T, rtol=1e-5, atol=1e-8):
        raise TLTypeError(
            "eigh needs a symmetric matrix (only the triangle would be "
            "used otherwise); use svd() for general matrices"
        )
    try:
        values, vectors = _la.eigh(arr)
    except Exception as exc:
        raise TLTypeError(f"eigh failed: {exc}") from exc
    return values, vectors


def evaluate_eigh_part(shared: _Shared, index: int):
    return shared.get()[index], "dense"


def solve(a, b):
    """Solve the square linear system ``a x = b`` (exact; singular → error)."""
    from ..core.operator import Operator, fresh_uid
    from ..core.tensor import Lazy, expr_of

    if isinstance(a, Operator) or isinstance(b, Operator):
        uid = fresh_uid()
        names: list[str] = []
        for op in (a, b):
            if isinstance(op, Operator):
                for p in op.params:
                    if p not in names:
                        names.append(p)

        def node(x):
            from ..core.tensor import Lazy, Tensor

            if isinstance(x, Operator):
                return ir.remap_params(x.root, uid)
            if isinstance(x, Tensor):
                return x._snapshot_node()
            if isinstance(x, Lazy):
                return x.root
            return ir.Const(x)

        return Operator(tuple(names), ir.Solve(node(a), node(b)), uid)

    def leaf(x) -> ir.Expr:
        if isinstance(x, Lazy):
            return x.root
        if hasattr(x, "_snapshot_node"):
            return x._snapshot_node()
        return expr_of(x, "solve")

    return Lazy(ir.Solve(leaf(a), leaf(b)))


def evaluate_solve(a, b):
    import scipy.linalg as _la

    ma = np.asarray(a, dtype=np.float64)
    mb = np.asarray(b, dtype=np.float64)
    if ma.ndim != 2 or ma.shape[0] != ma.shape[1]:
        raise TLShapeError(
            f"solve needs a square coefficient matrix, got shape {ma.shape}",
            code="TL0506",
            details={"shape": tuple(ma.shape)},
        )
    if mb.ndim not in (1, 2) or mb.shape[0] != ma.shape[0]:
        raise TLShapeError(
            f"solve: right-hand side shape {mb.shape} does not match "
            f"matrix shape {ma.shape}",
            code="TL0506",
            details={"matrix": tuple(ma.shape), "rhs": tuple(mb.shape)},
        )
    try:
        return _la.solve(ma, mb), "dense"
    except _la.LinAlgError as exc:
        raise TLShapeError(f"solve: singular system ({exc})",
                           code="TL0506") from exc


def norm(x, ord=None, axis=None):
    """Matrix/vector norm (NumPy ``ord``/``axis`` conventions) → scalar."""
    from ._lift import lift_unary_operand

    return lift_unary_operand(
        x, lambda root: ir.Norm(root, _norm_ord(ord), _norm_axis(axis)))


def _norm_ord(ord):
    if ord is None or isinstance(ord, (int, float, str)):
        if isinstance(ord, str) and ord not in ("fro", "nuc", "inf", "-inf", "fro"):
            try:
                return float(ord)
            except ValueError:
                raise TLTypeError(
                    f"unknown norm {ord!r}; use a number, 'fro', 'nuc', "
                    f"'inf' or '-inf'"
                )
        return ord
    raise TLTypeError(f"invalid norm {ord!r}")


def _norm_axis(axis):
    if axis is None or isinstance(axis, int):
        return axis
    try:
        return tuple(axis)
    except TypeError:
        raise TLTypeError(f"invalid norm axis {axis!r}") from None


def evaluate_norm(value, kind: str, ord, axis):
    if kind == "series":
        value = np.asarray(value.to_list(), dtype=np.float64)
        kind = "dense"
    if kind != "dense":
        raise TLTypeError(
            "norm needs dense numerical data or a field series; select "
            "numeric fields explicitly for structured data"
        )
    arr = np.asarray(value, dtype=np.float64)
    try:
        return float(np.linalg.norm(arr, ord=ord, axis=axis)), "scalar"
    except ValueError as exc:
        raise TLTypeError(f"cannot compute norm: {exc}") from exc
