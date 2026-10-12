"""TensorLang runtime objects: ``Tensor`` and ``Lazy``.

A ``Tensor`` is an eager, mutable handle to data (dense NumPy arrays or
structured Polars frames/series).  Every non-mutating operation on a
``Tensor`` — arithmetic, comparisons, indexing, reductions — returns a
``Lazy`` computation graph instead of executing immediately.  ``Lazy``
graphs are materialized with ``.collect()`` and committed back to their
source tensor with ``.inplace()``.

Snapshots: ``Lazy`` nodes capture immutable copies of their inputs at
construction time, so a later in-place mutation of the source tensor can
never change an already constructed lazy expression.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import polars as pl

from . import expression as ir
from .errors import TLShapeError, TLTypeError, TLUnsupportedError
from .types import (
    PROVENANCE_DESCRIPTOR,
    PROVENANCE_IO,
    PROVENANCE_LITERAL,
    ScanSpec,
)

_SCALAR_TYPES = (int, float, bool, str, complex, np.generic)


def _readonly_view(arr: np.ndarray) -> np.ndarray:
    """Zero-copy snapshot of internal dense storage.

    Views share memory with the source tensor (no silent gigabyte
    duplication); the read-only flag plus TensorLang's never-mutate-
    snapshots discipline keeps them immutable.  External arrays still go
    through ``literal_of`` (copy) so later caller-side mutation cannot
    leak into a graph.
    """
    try:
        view = np.array(arr, copy=False, subok=True)
    except ValueError:
        view = np.array(arr, copy=True, subok=True)
    try:
        view.setflags(write=False)
    except ValueError:
        view = np.array(arr, copy=True, subok=True)
        view.setflags(write=False)
    return view


def literal_of(value: Any) -> ir.Literal:
    """Wrap a materialized value as an immutable ``Literal`` IR node."""
    if isinstance(value, np.ndarray):
        return ir.Literal(np.array(value, copy=True), "dense")
    if isinstance(value, pl.DataFrame):
        return ir.Literal(value.clone(), "frame")
    if isinstance(value, pl.Series):
        return ir.Literal(value.clone(), "series")
    if isinstance(value, _SCALAR_TYPES) or hasattr(value, "__sympy__") or "sympy" in type(value).__module__:
        return ir.Literal(value, "scalar")
    raise TLTypeError(f"cannot represent {type(value).__name__} as tensor data")


def expr_of(operand: Any, op: str = "operation") -> ir.Expr:
    """Coerce a ``Tensor`` / ``Lazy`` / scalar operand to a closed IR node."""
    if isinstance(operand, Lazy):
        return operand.root
    if isinstance(operand, Tensor):
        return operand._snapshot_node()
    if isinstance(operand, (np.ndarray, pl.DataFrame, pl.Series)):
        return literal_of(operand)
    if isinstance(operand, _SCALAR_TYPES):
        return ir.Const(operand.value if isinstance(operand, np.generic) else operand)
    if "sympy" in type(operand).__module__:
        return ir.Const(operand)
    raise TLTypeError(
        f"unsupported operand of type '{type(operand).__name__}' for '{op}'"
    )


def _index_expr(index: Any) -> ir.Expr:
    if isinstance(index, Lazy):
        return index.root
    if isinstance(index, Tensor):
        return index._snapshot_node()
    if isinstance(index, slice):
        return ir.Const(index)
    if isinstance(index, tuple):
        return ir.Const(tuple(index))
    if isinstance(index, list):
        if all(isinstance(x, str) for x in index):
            return ir.Const(list(index))
        return literal_of(np.asarray(index))
    if isinstance(index, (np.ndarray, pl.Series)):
        return literal_of(index)
    if isinstance(index, (int, str, bool)):
        return ir.Const(index)
    if isinstance(index, np.integer):
        return ir.Const(int(index))
    raise TLTypeError(f"unsupported index of type '{type(index).__name__}'")


class Tensor:
    """The universal TensorLang data object.

    Wraps one of three representations:

    - ``dense`` — an n-dimensional :class:`numpy.ndarray`
    - ``frame`` — a heterogeneous :class:`polars.DataFrame`
    - ``series`` — a named :class:`polars.Series` (a single field)
    """

    def __init__(self, data: Any = None) -> None:
        kind, payload = _coerce_data(data)
        self._kind = kind
        self._data = payload
        self._version = 0
        self._tl_name: str | None = None
        self._provenance = PROVENANCE_LITERAL

    # -- construction helpers -------------------------------------------------
    @staticmethod
    def _from_parts(kind: str, payload: Any,
                    provenance: str = PROVENANCE_LITERAL) -> "Tensor":
        obj = Tensor.__new__(Tensor)
        obj._kind = kind
        obj._data = payload
        obj._version = 0
        obj._tl_name = None
        obj._provenance = provenance
        return obj

    @property
    def provenance(self) -> str:
        """How this tensor came about: literal | io | derived | descriptor."""
        return self._provenance

    def _snapshot_node(self) -> ir.Expr:
        if self._kind == "dense":
            return ir.Literal(_readonly_view(self._data), "dense")
        if self._kind == "scan":
            return ir.Scan(self._data)
        return literal_of(self._material())

    def _material(self) -> Any:
        if self._kind == "dense":
            return np.array(self._data, copy=True)
        if self._kind == "frame":
            return self._data.clone()
        if self._kind == "series":
            return self._data.clone()
        if self._kind == "scan":
            from ..runtime import io as _io

            return _io.materialize_scan(self._data)
        raise TLTypeError(f"unknown tensor kind '{self._kind}'")  # pragma: no cover

    def _replace(self, kind: str, payload: Any) -> None:
        self._kind = kind
        self._data = payload
        self._version += 1

    # -- metadata ---------------------------------------------------------------
    @property
    def kind(self) -> str:
        return self._kind

    @property
    def shape(self) -> tuple[int, ...]:
        if self._kind == "dense":
            return tuple(self._data.shape)
        if self._kind == "frame":
            return (self._data.height, self._data.width)
        if self._kind == "scan":
            raise TLTypeError(
                "a lazy scan has no known shape until it is collected; "
                "use .schema/.fields for metadata or .collect() to read it"
            )
        return (len(self._data),)

    @property
    def rank(self) -> int:
        if self._kind == "dense":
            return int(self._data.ndim)
        if self._kind == "scan":
            raise TLTypeError("a lazy scan has no known rank until collected")
        if self._kind == "frame":
            return 2
        return 1

    @property
    def ndim(self) -> int:
        return self.rank

    @property
    def dtype(self):
        if self._kind == "dense":
            return self._data.dtype
        if self._kind in ("frame", "scan"):
            return dict(self.schema)
        return self._data.dtype

    @property
    def schema(self):
        if self._kind == "frame":
            return dict(self._data.schema)
        if self._kind == "scan":
            from ..runtime import io as _io

            return dict(_io.scan_schema(self._data))
        raise TLTypeError("schema is only defined for structured tensors")

    @property
    def fields(self) -> list[str]:
        if self._kind in ("frame", "scan"):
            return list(self.schema)
        if self._kind == "series":
            return [self._data.name]
        raise TLTypeError("fields are only defined for structured tensors")

    @property
    def columns(self) -> list[str]:
        return self.fields

    @property
    def version(self) -> int:
        return self._version

    def abstract(self) -> dict:
        """Structured, JSON-serializable summary (never reads scan data)."""
        from .abstract import tensor_abstract

        return tensor_abstract(self)

    # -- materialization ----------------------------------------------------------
    def collect(self):
        """Materialize this tensor's current contents (a snapshot)."""
        from ..runtime.executor import collect_value

        return collect_value(self._material(), self._kind)

    def to_numpy(self) -> np.ndarray:
        if self._kind == "dense":
            return np.array(self._data, copy=True)
        if self._kind == "series":
            return self._data.to_numpy()
        raise TLTypeError(
            "to_numpy() is undefined for heterogeneous structured tensors; "
            "select numerical fields first or use to_polars()"
        )

    def to_polars(self):
        if self._kind == "frame":
            return self._data.clone()
        if self._kind == "series":
            return self._data.clone()
        raise TLTypeError("to_polars() is only defined for structured tensors")

    # -- dunder operators (all non-mutating -> Lazy) -------------------------------
    def _binop(self, op: str, other: Any) -> "Lazy":
        return Lazy(ir.BinOp(op, expr_of(self), expr_of(other, op)))

    def _rbinop(self, op: str, other: Any) -> "Lazy":
        return Lazy(ir.BinOp(op, expr_of(other, op), expr_of(self)))

    def __add__(self, other): return self._binop("add", other)
    def __radd__(self, other): return self._rbinop("add", other)
    def __sub__(self, other): return self._binop("sub", other)
    def __rsub__(self, other): return self._rbinop("sub", other)
    def __mul__(self, other): return self._binop("mul", other)
    def __rmul__(self, other): return self._rbinop("mul", other)
    def __truediv__(self, other): return self._binop("div", other)
    def __rtruediv__(self, other): return self._rbinop("div", other)
    def __pow__(self, other): return self._binop("pow", other)
    def __rpow__(self, other): return self._rbinop("pow", other)
    def __mod__(self, other): return self._binop("mod", other)
    def __rmod__(self, other): return self._rbinop("mod", other)
    def __matmul__(self, other): return self._binop("matmul", other)
    def __eq__(self, other): return self._binop("eq", other)  # type: ignore[override]
    def __ne__(self, other): return self._binop("ne", other)  # type: ignore[override]
    def __gt__(self, other): return self._binop("gt", other)
    def __lt__(self, other): return self._binop("lt", other)
    def __ge__(self, other): return self._binop("ge", other)
    def __le__(self, other): return self._binop("le", other)
    def __and__(self, other): return self._binop("and", other)
    def __rand__(self, other): return self._rbinop("and", other)
    def __or__(self, other): return self._binop("or", other)
    def __ror__(self, other): return self._rbinop("or", other)
    def __neg__(self): return Lazy(ir.UnOp("neg", expr_of(self)))
    def __invert__(self): return Lazy(ir.UnOp("not", expr_of(self)))
    def __abs__(self): return Lazy(ir.UnOp("abs", expr_of(self)))

    __hash__ = None  # type: ignore[assignment]

    def __bool__(self) -> bool:
        if self._kind == "dense" and self._data.ndim == 0:
            return bool(self._data.item())
        raise TLTypeError(
            "truth value of a tensor is ambiguous; apply a reduction or "
            "collect() an explicit Boolean result first",
            code="TL0401",
        )

    def __len__(self) -> int:
        if self._kind == "dense":
            if self._data.ndim == 0:
                raise TLTypeError("len() of a rank-zero tensor is undefined")
            return int(self._data.shape[0])
        if self._kind == "frame":
            return self._data.height
        if self._kind == "scan":
            raise TLTypeError(
                "len() of a lazy scan is unknown until collected; "
                "collect() it or bound the pipeline with an explicit sort"
            )
        return len(self._data)

    def __iter__(self):
        raise TLTypeError(
            "iteration over a tensor is not implicit; collect() it first "
            "or index it explicitly",
            code="TL0402",
        )

    def __getitem__(self, index) -> "Lazy":
        if self._kind == "dense" and (
            isinstance(index, str)
            or (isinstance(index, list) and all(isinstance(x, str) for x in index))
        ):
            raise TLTypeError(
                f"cannot select field {index!r} from a dense tensor; "
                f"named fields exist only on structured tensors"
            )
        if self._kind == "series" and isinstance(index, str):
            raise TLTypeError(
                f"cannot select field '{index}' from a single field series"
            )
        return Lazy(ir.GetItem(expr_of(self), _index_expr(index)))

    def __setitem__(self, key, value) -> None:
        from ..operators import indexing as _indexing
        from ..runtime.executor import evaluate

        if self._kind == "scan":
            raise TLTypeError(
                "cannot assign into a lazy scan descriptor; collect() it "
                "into a tensor first"
            )

        if isinstance(key, Lazy):
            key, _ = evaluate(key.root)
        elif isinstance(key, Tensor):
            key = key._material()
        new_value, _ = (
            evaluate(value.root) if isinstance(value, Lazy) else (value, None)
        )
        if isinstance(new_value, Tensor):
            new_value = new_value._material()
        updated = _indexing.set_item(self._material(), self._kind, key, new_value)
        kind, payload = _coerce_raw(updated)
        self._replace(kind, payload)

    def __getattr__(self, name: str) -> "Lazy":
        if name.startswith("_"):
            raise AttributeError(name)
        if self._kind in ("frame", "scan"):
            if name not in self.schema:
                raise AttributeError(
                    f"structured tensor has no field '{name}'; "
                    f"available fields: {self.fields}"
                )
            return Lazy(ir.FieldAccess(self._snapshot_node(), name))
        if self._kind == "series":
            if name == self._data.name:
                return Lazy(self._snapshot_node())
            raise AttributeError(
                f"series tensor has a single field '{self._data.name}', "
                f"not '{name}'"
            )
        raise AttributeError(
            f"dense tensor has no field '{name}'; fields exist only on "
            f"structured tensors"
        )

    # -- named transformations (all non-mutating -> Lazy) ---------------------------
    @property
    def T(self) -> "Lazy":
        if self._kind != "dense":
            raise TLUnsupportedError(
                "transpose is undefined for heterogeneous structured tensors",
                code="TL0901",
                details={"operation": "transpose", "kind": self._kind},
            )
        return Lazy(ir.Transpose(self._snapshot_node()))

    def transpose(self) -> "Lazy":
        return self.T

    def reshape(self, *shape) -> "Lazy":
        if self._kind != "dense":
            raise TLUnsupportedError(
                "reshape is undefined for structured tensors",
                code="TL0901",
                details={"operation": "reshape", "kind": self._kind},
            )
        return Lazy(ir.Reshape(self._snapshot_node(), _norm_shape(shape)))

    def flatten(self) -> "Lazy":
        if self._kind == "frame":
            raise TLUnsupportedError(
                "flatten is undefined for structured tensors",
                code="TL0901",
                details={"operation": "flatten", "kind": self._kind},
            )
        return Lazy(ir.Flatten(self._snapshot_node()))

    def sum(self, by=None) -> "Lazy":
        return Lazy(ir.Reduce("sum", self._snapshot_node(), _by_expr(by)))

    def mean(self, by=None) -> "Lazy":
        return Lazy(ir.Reduce("mean", self._snapshot_node(), _by_expr(by)))

    def min(self, by=None) -> "Lazy":
        return Lazy(ir.Reduce("min", self._snapshot_node(), _by_expr(by)))

    def max(self, by=None) -> "Lazy":
        return Lazy(ir.Reduce("max", self._snapshot_node(), _by_expr(by)))

    def std(self, by=None) -> "Lazy":
        return Lazy(ir.Reduce("std", self._snapshot_node(), _by_expr(by)))

    def var(self, by=None) -> "Lazy":
        return Lazy(ir.Reduce("var", self._snapshot_node(), _by_expr(by)))

    def is_null(self) -> "Lazy":
        return Lazy(ir.FuncCall("is_null", (self._snapshot_node(),)))

    def not_null(self) -> "Lazy":
        return Lazy(ir.FuncCall("not_null", (self._snapshot_node(),)))

    def fill_null(self, value) -> "Lazy":
        return Lazy(ir.FillNull(self._snapshot_node(), expr_of(value, "fill_null")))

    def drop_null(self) -> "Lazy":
        return Lazy(ir.FuncCall("drop_null", (self._snapshot_node(),)))

    def sort(self, by=None, descending: bool = False) -> "Lazy":
        from ..operators.relational import sort as _sort

        return _sort(self, by=by, descending=descending)

    def unique(self) -> "Lazy":
        from ..operators.relational import unique as _unique

        return _unique(self)

    def join(self, other, on, how: str = "inner", suffix: str = "_right") -> "Lazy":
        from ..operators.relational import join as _join

        return _join(self, other, on=on, how=how, suffix=suffix)

    def clip(self, lo=None, hi=None) -> "Lazy":
        from ..operators.transform import clip as _clip

        return _clip(self, lo, hi)

    def over(self, by, op: str = "mean") -> "Lazy":
        from ..operators.transform import over as _over

        return _over(self, by, op)

    def rolling(self, window: int, op: str = "mean",
                min_periods: int | None = None) -> "Lazy":
        from ..operators.transform import rolling as _rolling

        return _rolling(self, window, op, min_periods)

    def write_parquet(self, path: str) -> str:
        """Explicit execution boundary: run the pipeline into Parquet."""
        from ..runtime import polars_plan as _lower
        from ..runtime import io as _io

        if self._kind == "scan":
            node: ir.Expr = ir.Scan(self._data)
            ok, _ = _lower.try_sink(node, path, "parquet")
            if ok:
                return path
        value = self._material()
        kind = self._kind if self._kind != "scan" else "frame"
        return _io.write_value(value, kind, path, "parquet")

    def write_csv(self, path: str) -> str:
        """Explicit execution boundary: run the pipeline into CSV."""
        from ..runtime import polars_plan as _lower
        from ..runtime import io as _io

        if self._kind == "scan":
            node = ir.Scan(self._data)
            ok, _ = _lower.try_sink(node, path, "csv")
            if ok:
                return path
        value = self._material()
        kind = self._kind if self._kind != "scan" else "frame"
        return _io.write_value(value, kind, path, "csv")

    def explain(self) -> str:
        from ..runtime.inspect import explain_lazy

        return explain_lazy(self._snapshot_node())

    def profile(self):
        from ..runtime.inspect import profile_lazy

        return profile_lazy(self._snapshot_node())

    # -- display ----------------------------------------------------------------------
    def __repr__(self) -> str:
        name = f"{self._tl_name} := " if self._tl_name else ""
        if self._kind == "dense":
            return f"{name}T({self._data!r})"
        if self._kind == "frame":
            return f"{name}T(frame{self._data.shape}, fields={self._data.columns})"
        return f"{name}T(series '{self._data.name}', len={len(self._data)})"

    def __str__(self) -> str:
        return str(self._data)


def _by_expr(by) -> ir.Expr | None:
    if by is None:
        return None
    if isinstance(by, str):
        return ir.Const(by)
    return expr_of(by, "group-by")


def _norm_shape(shape) -> tuple[int, ...]:
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    try:
        out = tuple(int(s) for s in shape)
    except (TypeError, ValueError) as exc:
        raise TLShapeError(f"invalid reshape shape {shape!r}") from exc
    return out


def _coerce_data(data: Any) -> tuple[str, Any]:
    if isinstance(data, Tensor):
        return data._kind, data._material()
    if isinstance(data, np.ndarray):
        return "dense", np.array(data, copy=True)
    if isinstance(data, pl.DataFrame):
        return "frame", data.clone()
    if isinstance(data, pl.Series):
        return "series", data.clone()
    if isinstance(data, dict):
        try:
            return "frame", pl.DataFrame(dict(data))
        except Exception as exc:
            raise TLTypeError(f"cannot build a structured tensor: {exc}") from exc
    if isinstance(data, (list, tuple)):
        try:
            arr = np.asarray(data)
        except ValueError as exc:
            raise TLShapeError(f"ragged sequence cannot form a tensor: {exc}") from exc
        return "dense", arr
    if isinstance(data, _SCALAR_TYPES):
        value = data.value if isinstance(data, np.generic) else data
        return "dense", np.asarray(value)
    if data is None:
        raise TLTypeError("T() requires data; use T.zeros/T.ones helpers if needed")
    raise TLTypeError(f"cannot build a tensor from {type(data).__name__}")


def _coerce_raw(value: Any) -> tuple[str, Any]:
    if isinstance(value, np.ndarray):
        return "dense", value
    if isinstance(value, pl.DataFrame):
        return "frame", value
    if isinstance(value, pl.Series):
        return "series", value
    raise TLTypeError(  # pragma: no cover - defensive
        f"cannot store {type(value).__name__} in a tensor"
    )


class Lazy:
    """A pending (not yet executed) computation over snapshotted inputs.

    Created by operator application and by every non-mutating ``Tensor``
    operation.  Materialize with :meth:`collect`; commit back to the source
    tensor with :meth:`inplace` (only for single-input operator
    applications over a mutable :class:`Tensor`).
    """

    def __init__(self, root: ir.Expr, target: Optional[Tensor] = None) -> None:
        self._root = root
        self._target = target
        self._cached: Any = None
        self._has_cached = False

    @property
    def root(self) -> ir.Expr:
        """The (closed) expression graph behind this pending computation."""
        return self._root

    def abstract(self) -> dict:
        """Structured, JSON-serializable summary (never executes)."""
        from .abstract import lazy_abstract

        return lazy_abstract(self)

    # -- execution ----------------------------------------------------------------------
    def collect(self):
        """Execute the graph and return the materialized result."""
        from ..runtime.executor import evaluate, collect_value

        if not self._has_cached:
            value, kind = evaluate(self._root)
            self._cached = collect_value(value, kind)
            self._has_cached = True
        return self._cached

    def inplace(self) -> "Tensor":
        """Execute and atomically commit the result to the source tensor."""
        from ..runtime import mutation as _mutation

        return _mutation.apply_inplace(self)

    def to_numpy(self) -> np.ndarray:
        value = self.collect()
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, pl.Series):
            return value.to_numpy()
        raise TLTypeError("to_numpy() is undefined for this result")

    def to_polars(self):
        value = self.collect()
        if isinstance(value, (pl.DataFrame, pl.Series)):
            return value
        raise TLTypeError("to_polars() is only defined for structured results")

    # -- graph combinators (same surface as Tensor) ---------------------------------------
    def _binop(self, op: str, other: Any) -> "Lazy":
        return Lazy(ir.BinOp(op, self._root, expr_of(other, op)))

    def _rbinop(self, op: str, other: Any) -> "Lazy":
        return Lazy(ir.BinOp(op, expr_of(other, op), self._root))

    def __add__(self, other): return self._binop("add", other)
    def __radd__(self, other): return self._rbinop("add", other)
    def __sub__(self, other): return self._binop("sub", other)
    def __rsub__(self, other): return self._rbinop("sub", other)
    def __mul__(self, other): return self._binop("mul", other)
    def __rmul__(self, other): return self._rbinop("mul", other)
    def __truediv__(self, other): return self._binop("div", other)
    def __rtruediv__(self, other): return self._rbinop("div", other)
    def __pow__(self, other): return self._binop("pow", other)
    def __rpow__(self, other): return self._rbinop("pow", other)
    def __mod__(self, other): return self._binop("mod", other)
    def __rmod__(self, other): return self._rbinop("mod", other)
    def __matmul__(self, other): return self._binop("matmul", other)
    def __eq__(self, other): return self._binop("eq", other)  # type: ignore[override]
    def __ne__(self, other): return self._binop("ne", other)  # type: ignore[override]
    def __gt__(self, other): return self._binop("gt", other)
    def __lt__(self, other): return self._binop("lt", other)
    def __ge__(self, other): return self._binop("ge", other)
    def __le__(self, other): return self._binop("le", other)
    def __and__(self, other): return self._binop("and", other)
    def __rand__(self, other): return self._rbinop("and", other)
    def __or__(self, other): return self._binop("or", other)
    def __ror__(self, other): return self._rbinop("or", other)
    def __neg__(self): return Lazy(ir.UnOp("neg", self._root))
    def __invert__(self): return Lazy(ir.UnOp("not", self._root))
    def __abs__(self): return Lazy(ir.UnOp("abs", self._root))

    __hash__ = None  # type: ignore[assignment]

    def __bool__(self) -> bool:
        raise TLTypeError(
            "truth value of a pending computation is ambiguous; "
            "collect() an explicit Boolean result first",
            code="TL0401",
        )

    def __len__(self) -> int:
        raise TLTypeError("len() of a pending computation is undefined; collect() first")

    def __iter__(self):
        raise TLTypeError("cannot iterate a pending computation; collect() first")

    def __getitem__(self, index) -> "Lazy":
        return Lazy(ir.GetItem(self._root, _index_expr(index)))

    def __setitem__(self, key, value) -> None:
        # Graph-level field assignment (e.g. filtered["net"] = ...): stays
        # lazy as a WithField node, so scan→filter→derive pipelines lower
        # into one backend query instead of materializing here.
        if isinstance(key, str):
            new_root = ir.WithField(
                self._root,
                key,
                value.root if isinstance(value, Lazy) else (
                    value._snapshot_node() if isinstance(value, Tensor)
                    else expr_of(value, "field assignment")),
            )
            self._root = new_root
            self._target = None
            self._cached = None
            self._has_cached = False
            return
        # Row/mask assignment stays eager (as in v0.1): evaluate now and
        # re-root onto the materialized result.
        from ..operators import indexing as _indexing
        from ..runtime.executor import evaluate

        if isinstance(key, Lazy):
            key, _ = evaluate(key.root)
        elif isinstance(key, Tensor):
            key = key._material()
        current, _ = evaluate(self._root)
        new_value, _ = (
            evaluate(value.root) if isinstance(value, Lazy) else (value, None)
        )
        if isinstance(new_value, Tensor):
            new_value = new_value._material()
        updated = _indexing.set_item(current, None, key, new_value)
        self._root = literal_of(updated)
        self._target = None
        self._cached = None
        self._has_cached = False

    def __getattr__(self, name: str) -> "Lazy":
        if name.startswith("_"):
            raise AttributeError(name)
        return Lazy(ir.FieldAccess(self._root, name))

    @property
    def T(self) -> "Lazy":
        return Lazy(ir.Transpose(self._root))

    def transpose(self) -> "Lazy":
        return self.T

    def reshape(self, *shape) -> "Lazy":
        return Lazy(ir.Reshape(self._root, _norm_shape(shape)))

    def flatten(self) -> "Lazy":
        return Lazy(ir.Flatten(self._root))

    def sum(self, by=None) -> "Lazy":
        return Lazy(ir.Reduce("sum", self._root, _by_expr(by)))

    def mean(self, by=None) -> "Lazy":
        return Lazy(ir.Reduce("mean", self._root, _by_expr(by)))

    def min(self, by=None) -> "Lazy":
        return Lazy(ir.Reduce("min", self._root, _by_expr(by)))

    def max(self, by=None) -> "Lazy":
        return Lazy(ir.Reduce("max", self._root, _by_expr(by)))

    def std(self, by=None) -> "Lazy":
        return Lazy(ir.Reduce("std", self._root, _by_expr(by)))

    def var(self, by=None) -> "Lazy":
        return Lazy(ir.Reduce("var", self._root, _by_expr(by)))

    def is_null(self) -> "Lazy":
        return Lazy(ir.FuncCall("is_null", (self._root,)))

    def not_null(self) -> "Lazy":
        return Lazy(ir.FuncCall("not_null", (self._root,)))

    def fill_null(self, value) -> "Lazy":
        return Lazy(ir.FillNull(self._root, expr_of(value, "fill_null")))

    def drop_null(self) -> "Lazy":
        return Lazy(ir.FuncCall("drop_null", (self._root,)))

    def sort(self, by=None, descending: bool = False) -> "Lazy":
        from ..operators.relational import sort as _sort

        return _sort(self, by=by, descending=descending)

    def unique(self) -> "Lazy":
        from ..operators.relational import unique as _unique

        return _unique(self)

    def join(self, other, on, how: str = "inner", suffix: str = "_right") -> "Lazy":
        from ..operators.relational import join as _join

        return _join(self, other, on=on, how=how, suffix=suffix)

    def clip(self, lo=None, hi=None) -> "Lazy":
        from ..operators.transform import clip as _clip

        return _clip(self, lo, hi)

    def over(self, by, op: str = "mean") -> "Lazy":
        from ..operators.transform import over as _over

        return _over(self, by, op)

    def rolling(self, window: int, op: str = "mean",
                min_periods: int | None = None) -> "Lazy":
        from ..operators.transform import rolling as _rolling

        return _rolling(self, window, op, min_periods)

    def write_parquet(self, path: str) -> str:
        """Explicit execution boundary: stream the pipeline into Parquet."""
        from ..runtime import polars_plan as _lower
        from ..runtime import io as _io
        from ..runtime.executor import collect_value, evaluate

        ok, _ = _lower.try_sink(self._root, path, "parquet")
        if ok:
            return path
        value, kind = evaluate(self._root)
        return _io.write_value(collect_value(value, kind), kind, path, "parquet")

    def write_csv(self, path: str) -> str:
        """Explicit execution boundary: stream the pipeline into CSV."""
        from ..runtime import polars_plan as _lower
        from ..runtime import io as _io
        from ..runtime.executor import collect_value, evaluate

        ok, _ = _lower.try_sink(self._root, path, "csv")
        if ok:
            return path
        value, kind = evaluate(self._root)
        return _io.write_value(collect_value(value, kind), kind, path, "csv")

    def explain(self) -> str:
        """Logical IR graph plus physical backend plan (no execution)."""
        from ..runtime.inspect import explain_lazy

        return explain_lazy(self._root)

    def profile(self):
        """Execute and report measured costs (time, memory, backend)."""
        from ..runtime.inspect import profile_lazy

        return profile_lazy(self._root)

    def release(self) -> "Lazy":
        """Drop the cached ``collect()`` result to free memory."""
        self._cached = None
        self._has_cached = False
        return self

    def __repr__(self) -> str:
        return f"Lazy({self._root!r})"


def _flatten_paths(paths) -> list[str]:
    flat: list[str] = []
    for p in paths:
        if isinstance(p, (list, tuple)):
            flat.extend(str(x) for x in p)
        else:
            flat.append(str(p))
    return flat


class T(Tensor):
    """User-facing tensor constructor: ``T([1, 2, 3])``."""

    def __new__(cls, data: Any = None, **fields: Any):
        if data is None and fields:
            return Tensor._from_parts(*_coerce_data(dict(fields)))
        obj = Tensor(data)
        return obj

    @staticmethod
    def linspace(start: float, stop: float, num: int = 50) -> Tensor:
        return Tensor(np.linspace(start, stop, num))

    @staticmethod
    def arange(*args) -> Tensor:
        return Tensor(np.arange(*args))

    @staticmethod
    def zeros(shape) -> Tensor:
        return Tensor(np.zeros(shape))

    @staticmethod
    def ones(shape) -> Tensor:
        return Tensor(np.ones(shape))

    @staticmethod
    def read(path: str, **kwargs) -> Tensor:
        """Read a structured dataset (``.parquet`` / ``.csv`` / ``.json``).

        Effectful: reading executes I/O, so the result has ``io``
        provenance and must be bound with ``=``, not ``:=``.
        """
        lower = str(path).lower()
        try:
            if lower.endswith(".parquet"):
                tensor = Tensor(pl.read_parquet(path, **kwargs))
            elif lower.endswith(".csv"):
                tensor = Tensor(pl.read_csv(path, **kwargs))
            elif lower.endswith(".json"):
                tensor = Tensor(pl.read_json(path, **kwargs))
            else:
                raise TLTypeError(
                    f"unsupported dataset format for '{path}': "
                    f"use .parquet, .csv or .json"
                )
        except TLTypeError:
            raise
        except Exception as exc:
            raise TLTypeError(f"cannot read dataset '{path}': {exc}") from exc
        tensor._provenance = PROVENANCE_IO
        return tensor

    @staticmethod
    def scan_parquet(*paths: str, **options) -> Tensor:
        """Describe a lazily scanned Parquet dataset (reads nothing).

        Declarative: the result is a source descriptor, legal under ``:=``.
        Nothing is read until an explicit execution boundary
        (``collect()`` / ``write_*``).  Extra keywords (e.g. ``columns``)
        are forwarded to the backend scan.
        """
        from .types import ScanSpec

        flat = _flatten_paths(paths)
        if not flat:
            raise TLTypeError("scan_parquet needs at least one path or glob")
        return Tensor._from_parts(
            "scan",
            ScanSpec(tuple(flat), "parquet", tuple(sorted(options.items()))),
            PROVENANCE_DESCRIPTOR,
        )

    @staticmethod
    def scan_csv(*paths: str, **options) -> Tensor:
        """Describe a lazily scanned CSV dataset (reads nothing)."""
        from .types import ScanSpec

        flat = _flatten_paths(paths)
        if not flat:
            raise TLTypeError("scan_csv needs at least one path or glob")
        return Tensor._from_parts(
            "scan",
            ScanSpec(tuple(flat), "csv", tuple(sorted(options.items()))),
            PROVENANCE_DESCRIPTOR,
        )
