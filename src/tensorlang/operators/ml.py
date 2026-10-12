"""First-class machine learning: fittable operators and fitted pipelines.

``Operator -> fit(training) -> FittedOperator -> apply(data) -> Tensor``

A normal operator is stateless; a fitted operator carries immutable learned
state (exclusively from its fitting dataset — application data is never
used for statistics).  Both share the ordinary operator interface:
``f(data)`` application, ``>>`` composition, ``.collect()`` and
``.inplace()``.  Numerical routines use NumPy/SciPy only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from ..core import expression as ir
from ..core.errors import TLShapeError, TLTypeError
from ..core.operator import Operator, fresh_uid

FITTABLE_KINDS = ("standardize", "minmax", "normalize", "impute", "pca")


def _cfg_tuple(config: dict) -> tuple:
    return tuple(sorted(config.items()))


@dataclass(frozen=True)
class FittedState:
    """Immutable learned state. NumPy payloads are read-only views."""

    kind: str
    config: tuple = ()
    arrays: tuple = field(default_factory=tuple, compare=False)
    objects: tuple = field(default_factory=tuple, compare=False)
    feature_names: tuple | None = None
    n_features: int = 0
    input_kind: str = "dense"
    numeric_cols: tuple = ()

    def array(self, name: str) -> np.ndarray:
        for key, value in self.arrays:
            if key == name:
                return value
        raise KeyError(name)  # pragma: no cover - defensive

    def obj(self, name: str):
        for key, value in self.objects:
            if key == name:
                return value
        raise KeyError(name)  # pragma: no cover - defensive

    def as_dict(self) -> dict:
        out = {}
        for key, value in self.arrays:
            copy = np.array(value, copy=True, subok=True)
            copy.setflags(write=False)
            out[key] = copy
        out.update({key: value for key, value in self.objects})
        out["feature_names"] = list(self.feature_names) if self.feature_names else None
        out["n_features_in_"] = self.n_features
        return out


def _ro(arr: np.ndarray) -> np.ndarray:
    arr = np.array(arr, copy=True, subok=True)
    arr.setflags(write=False)
    return arr


# -- feature extraction ---------------------------------------------------------


@dataclass
class _Features:
    matrix: np.ndarray  # float64 (n, d)
    names: list | None
    frame: Any  # original frame or None
    numeric_cols: list
    is_series: bool = False
    series_name: Any = None


def extract_features(value, vkind: str, *, estimator: str,
                     allow_mixed: bool) -> _Features:
    """Extract a numeric feature matrix without unnecessary copies.

    ``allow_mixed`` (shape-preserving ops): transform applicable numeric
    fields, preserve the rest.  Otherwise (e.g. PCA) heterogeneous frames
    are rejected with an explicit selection hint.
    """
    if vkind == "dense":
        arr = np.asarray(value)
        if arr.ndim == 1:
            return _Features(_as_float_matrix(arr.reshape(-1, 1), estimator),
                             None, None, [], False, None)
        if arr.ndim == 2:
            return _Features(_as_float_matrix(arr, estimator),
                             None, None, [], False, None)
        raise TLTypeError(
            f"{estimator} needs 1-D or 2-D dense data, got rank {arr.ndim}"
        )
    if vkind == "series":
        if value.dtype == pl.String or value.dtype == pl.Categorical:
            raise TLTypeError(
                f"{estimator} needs numeric data; encode the "
                f"'{value.name}' series with onehot() first",
                code="TL0410",
                details={"field": value.name, "dtype": str(value.dtype)},
            )
        try:
            matrix = np.asarray(value.to_numpy()).astype(np.float64).reshape(-1, 1)
        except (TypeError, ValueError) as exc:
            raise TLTypeError(f"{estimator} needs numeric data: {exc}",
                              code="TL0410") from exc
        return _Features(matrix, [value.name], None, [], True, value.name)
    if vkind == "frame":
        from ..backends import polars_backend as _pl_be

        numeric = _pl_be.numeric_columns(value)
        non_numeric = [c for c in value.columns if c not in numeric]
        if not numeric:
            raise TLTypeError(
                f"{estimator} has no applicable numerical fields in "
                f"{value.columns}",
                code="TL0410",
                details={"columns": list(value.columns)},
            )
        if non_numeric and not allow_mixed:
            raise TLTypeError(
                f"{estimator} needs an unambiguous numeric feature selection; "
                f"non-numeric fields {non_numeric} present — select explicitly, "
                f"e.g. data[{numeric}]",
                code="TL0410",
                details={"non_numeric": list(non_numeric),
                         "numeric": list(numeric)},
            )
        try:
            matrix = value.select(numeric).to_numpy().astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise TLTypeError(f"{estimator} needs numeric data: {exc}") from exc
        return _Features(matrix, list(numeric), value, list(numeric), False, None)
    raise TLTypeError(f"{estimator} cannot learn from scalar data")


def _as_float_matrix(arr: np.ndarray, estimator: str) -> np.ndarray:
    if arr.dtype.kind in ("i", "u", "f", "b"):
        out = arr.astype(np.float64, copy=False)
        return np.array(out, copy=True) if out is arr else out
    raise TLTypeError(
        f"{estimator} needs numeric data, got dtype {arr.dtype}; encode or "
        f"select numeric fields explicitly",
        code="TL0410",
        details={"dtype": str(arr.dtype)},
    )


def _check_finite(matrix: np.ndarray, estimator: str, *, what: str) -> None:
    bad_nan = bool(np.isnan(matrix).any())
    bad_inf = bool(np.isinf(matrix).any())
    if bad_nan or bad_inf:
        problem = "missing values" if bad_nan else "infinite values"
        raise TLTypeError(
            f"{estimator} {what} contains {problem}; handle them explicitly "
            f"first, e.g. with impute() (fit) or fill_null()",
            code="TL0412",
            details={"problem": problem},
        )


def _check_samples(matrix: np.ndarray, estimator: str, minimum: int = 1) -> None:
    if matrix.shape[0] < minimum:
        raise TLShapeError(
            f"{estimator} needs at least {minimum} sample(s), got "
            f"{matrix.shape[0]}"
        )


# -- fitting --------------------------------------------------------------------


def fit_state(kind: str, config: dict, value, vkind: str) -> FittedState:
    """Learn parameters from materialized fitting data (explicit boundary)."""
    allow_mixed = kind in ("standardize", "minmax", "normalize", "impute")
    feats = extract_features(value, vkind, estimator=kind, allow_mixed=allow_mixed)
    matrix = feats.matrix
    if kind != "impute":
        _check_finite(matrix, kind, what="fitting data")
    _check_samples(matrix, kind, minimum=1)
    cfg = dict(config)
    if kind == "standardize":
        mean = matrix.mean(axis=0)
        var = matrix.var(axis=0)
        scale = np.sqrt(var)
        scale[var == 0.0] = 1.0  # constant columns standardize to exactly 0
        arrays = (("mean_", _ro(mean)), ("scale_", _ro(scale)))
        objects = ()
    elif kind == "minmax":
        lo, hi = float(cfg["lo"]), float(cfg["hi"])
        dmin = matrix.min(axis=0)
        dmax = matrix.max(axis=0)
        rng = dmax - dmin
        rng[rng == 0.0] = 1.0  # constant columns map to lo
        arrays = (("min_", _ro(dmin)), ("range_", _ro(rng)))
        objects = ()
    elif kind == "normalize":
        arrays, objects = (), ()
    elif kind == "impute":
        arrays, objects = _fit_impute(cfg, feats)
    elif kind == "pca":
        arrays, objects = _fit_pca(cfg, matrix)
    else:  # pragma: no cover - guarded at construction
        raise TLTypeError(f"unknown estimator '{kind}'")
    return FittedState(
        kind=kind,
        config=_cfg_tuple(cfg),
        arrays=arrays,
        objects=objects,
        feature_names=tuple(feats.names) if feats.names else None,
        n_features=matrix.shape[1],
        input_kind=vkind,
        numeric_cols=tuple(feats.numeric_cols),
    )


def _fit_impute(cfg: dict, feats: _Features):
    strategy = cfg["strategy"]
    matrix = feats.matrix
    if strategy in ("mean", "median"):
        import warnings

        fn = np.nanmean if strategy == "mean" else np.nanmedian
        with warnings.catch_warnings():
            # All-missing columns are reported below as errors, not warnings.
            warnings.simplefilter("ignore", RuntimeWarning)
            fills = fn(matrix, axis=0)
        bad = [feats.names[i] if feats.names else f"column {i}"
               for i, v in enumerate(np.atleast_1d(fills)) if not np.isfinite(v)]
        if bad:
            raise TLTypeError(
                f"impute('{strategy}') cannot fill all-missing columns {bad}",
                code="TL0413",
                details={"columns": list(bad)},
            )
        return (("fill_", _ro(np.atleast_1d(fills).astype(np.float64))),), ()
    if strategy == "most_frequent":
        num_fills, fill_map = _column_modes(matrix, feats)
        return (("fill_num_", _ro(np.array(num_fills, dtype=np.float64))),), \
            (("fill_map_", tuple(fill_map)),)
    if strategy == "constant":
        return (), (("fill_value_", cfg["fill_value"]),)
    raise TLTypeError(f"unknown impute strategy '{strategy}'")  # pragma: no cover


def _column_modes(matrix: np.ndarray, feats: _Features):
    """Modes of each numeric column (matrix order) plus a full fill map."""
    num_fills = []
    fills = []
    width = matrix.shape[1]
    for i in range(width):
        name = (feats.numeric_cols[i] if feats.frame is not None
                else f"column {i}")
        col = matrix[:, i]
        present = col[~np.isnan(col)]
        if len(present) == 0:
            raise TLTypeError(
                f"impute('most_frequent') cannot fill all-missing column '{name}'",
                code="TL0413",
                details={"columns": [name]},
            )
        vals, counts = np.unique(present, return_counts=True)
        mode = float(vals[int(np.argmax(counts))])
        num_fills.append(mode)
        fills.append((name, mode))
    if feats.frame is not None:
        # String columns are handled at the frame level (see apply path).
        for col in feats.frame.columns:
            if col in feats.numeric_cols:
                continue
            series = feats.frame[col].drop_nulls()
            if len(series) == 0:
                raise TLTypeError(
                    f"impute('most_frequent') cannot fill all-missing column '{col}'",
                    code="TL0413",
                    details={"columns": [col]},
                )
            top = series.value_counts().sort("count", descending=True)[col][0]
            fills.append((col, top))
    return num_fills, fills


def _fit_pca(cfg: dict, matrix: np.ndarray):
    import scipy.linalg as _la

    n, d = matrix.shape
    k = int(cfg["n_components"])
    if k > min(n, d):
        raise TLShapeError(
            f"pca({k}) needs at most min(n_samples, n_features) components, "
            f"got n_samples={n}, n_features={d}",
            code="TL0508",
            details={"n_components": k, "n_samples": n, "n_features": d},
        )
    if n < 2:
        raise TLShapeError(f"pca needs at least 2 samples, got {n}",
                           code="TL0508",
                           details={"n_samples": n})
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    _, singular, vt = _la.svd(centered, full_matrices=False)
    components = vt[:k]
    explained = (singular[:k] ** 2) / (n - 1)
    return (
        (("mean_", _ro(mean)), ("components_", _ro(components)),
         ("explained_variance_", _ro(explained)),
         ("singular_", _ro(singular[:k]))),
        (),
    )


# -- application ------------------------------------------------------------------


def apply_fitted(node: ir.FittedTransform):
    """Evaluate a fitted transformation (or raise if unfitted)."""
    from ..runtime.executor import evaluate

    if node.state is None:
        raise TLTypeError(
            f"operator '{node.kind}' is not fitted: learned parameters must "
            f"come from fitting data — call .fit(training_data) first (with "
            f"'=', fitting executes) and apply the fitted result"
        )
    value, vkind = evaluate(node.arg)
    return _apply_state(node.state, value, vkind)


def _apply_state(state: FittedState, value, vkind: str):
    kind = state.kind
    cfg = dict(state.config)
    if kind == "pca":
        matrix, _ = _align_matrix(state, value, vkind)
        out = _apply_pca(state, cfg, matrix)
        cols = [f"pc{i}" for i in range(out.shape[1])]
        return pl.DataFrame(
            {c: out[:, j].tolist() for j, c in enumerate(cols)}
        ), "frame"
    if vkind == "series":
        if state.n_features != 1:
            raise TLShapeError(
                f"operator '{kind}' was fitted on {state.n_features} features, "
                f"got a single field series"
            )
        matrix = _series_matrix(value, kind)
        out = _apply_matrix(kind, cfg, state, matrix)
        return pl.Series(value.name, out[:, 0].tolist()), "series"
    if vkind == "dense":
        matrix, squeeze = _dense_matrix(value, kind, state.n_features)
        out = _apply_matrix(kind, cfg, state, matrix)
        return (out[:, 0] if squeeze else out), "dense"
    if vkind == "frame":
        return _apply_frame(state, kind, cfg, value)
    raise TLTypeError(f"operator '{kind}' cannot apply to scalar data")


def _series_matrix(value: pl.Series, kind: str) -> np.ndarray:
    try:
        return np.asarray(value.to_numpy()).astype(np.float64).reshape(-1, 1)
    except (TypeError, ValueError) as exc:
        raise TLTypeError(f"operator '{kind}' needs numeric data: {exc}") from exc


def _dense_matrix(value: np.ndarray, kind: str, n_features: int):
    arr = np.asarray(value)
    if arr.ndim == 1:
        matrix = _as_float_matrix(arr.reshape(-1, 1), kind)
        squeeze = True
    elif arr.ndim == 2:
        matrix = _as_float_matrix(arr, kind)
        squeeze = False
    else:
        raise TLTypeError(f"operator '{kind}' needs 1-D or 2-D dense data")
    if matrix.shape[1] != n_features:
        raise TLShapeError(
            f"operator '{kind}' was fitted on {n_features} features, got "
            f"{matrix.shape[1]}",
            code="TL0509",
            details={"n_features": n_features, "got": matrix.shape[1]},
        )
    return matrix, squeeze


def _align_matrix(state: FittedState, value, vkind: str) -> tuple[np.ndarray, Any]:
    if vkind == "dense":
        return _dense_matrix(np.asarray(value), state.kind, state.n_features)[0], None
    if vkind == "series":
        if state.n_features != 1:
            raise TLShapeError(
                f"operator '{state.kind}' was fitted on {state.n_features} "
                f"features, got a single field series"
            )
        return _series_matrix(value, state.kind), None
    if vkind == "frame":
        return _frame_matrix(state, value)
    raise TLTypeError(f"operator '{state.kind}' cannot apply to scalar data")


def _frame_matrix(state: FittedState, frame: pl.DataFrame) -> tuple[np.ndarray, pl.DataFrame]:
    missing = [c for c in state.numeric_cols if c not in frame.columns]
    if missing:
        raise TLTypeError(
            f"operator '{state.kind}' is missing features {missing}; have "
            f"{frame.columns}",
            code="TL0415",
            details={"missing": list(missing), "available": list(frame.columns)},
        )
    try:
        matrix = frame.select(list(state.numeric_cols)).to_numpy().astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise TLTypeError(
            f"operator '{state.kind}' needs numeric data: {exc}") from exc
    return matrix, frame


def _apply_frame(state: FittedState, kind: str, cfg: dict, frame: pl.DataFrame):
    matrix, _ = _frame_matrix(state, frame)
    if kind == "impute" and cfg["strategy"] in ("most_frequent", "constant"):
        return _apply_impute_frame(state, cfg, frame, matrix)
    out = _apply_matrix(kind, cfg, state, matrix)
    result = frame.clone()
    for j, col in enumerate(state.numeric_cols):
        result = result.with_columns(pl.Series(out[:, j].tolist()).alias(col))
    return result, "frame"


def _apply_impute_frame(state, cfg, frame, matrix):
    result = frame.clone()
    if cfg["strategy"] in ("mean", "median"):
        fills = state.array("fill_")
        for j, col in enumerate(state.numeric_cols):
            filled = np.where(np.isnan(matrix[:, j]), fills[j], matrix[:, j])
            result = result.with_columns(pl.Series(filled.tolist()).alias(col))
        return result, "frame"
    if cfg["strategy"] == "most_frequent":
        fills = dict(state.obj("fill_map_"))
        for j, col in enumerate(state.numeric_cols):
            filled = np.where(np.isnan(matrix[:, j]), fills[col], matrix[:, j])
            result = result.with_columns(pl.Series(filled.tolist()).alias(col))
        for col in frame.columns:
            if col in state.numeric_cols:
                continue
            result = result.with_columns(
                frame[col].fill_null(fills[col]).alias(col))
        return result, "frame"
    constant = cfg.get("fill_value")
    for j, col in enumerate(state.numeric_cols):
        filled = np.where(np.isnan(matrix[:, j]), constant, matrix[:, j])
        result = result.with_columns(pl.Series(filled.tolist()).alias(col))
    for col in frame.columns:
        if col in state.numeric_cols:
            continue
        result = result.with_columns(frame[col].fill_null(constant).alias(col))
    return result, "frame"


def _apply_matrix(kind: str, cfg: dict, state: FittedState, matrix: np.ndarray):
    if kind == "standardize":
        return (matrix - state.array("mean_")) / state.array("scale_")
    if kind == "minmax":
        lo, hi = float(cfg["lo"]), float(cfg["hi"])
        return (matrix - state.array("min_")) / state.array("range_") * (hi - lo) + lo
    if kind == "normalize":
        return _normalize_rows(matrix, cfg["norm"])
    if kind == "impute":
        return _apply_impute_matrix(cfg, state, matrix)
    raise TLTypeError(f"unknown estimator '{kind}'")  # pragma: no cover


def _normalize_rows(matrix: np.ndarray, norm: str) -> np.ndarray:
    if norm == "l1":
        norms = np.abs(matrix).sum(axis=1, keepdims=True)
    elif norm == "l2":
        norms = np.sqrt((matrix**2).sum(axis=1, keepdims=True))
    else:
        norms = np.abs(matrix).max(axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0  # zero rows stay zero
    return matrix / norms


def _apply_impute_matrix(cfg: dict, state: FittedState, matrix: np.ndarray):
    strategy = cfg["strategy"]
    if strategy in ("mean", "median"):
        fills = state.array("fill_")
        return np.where(np.isnan(matrix), fills, matrix)
    if strategy == "most_frequent":
        num_fills = state.array("fill_num_")
        out = matrix.copy()
        for j in range(out.shape[1]):
            out[:, j] = np.where(np.isnan(out[:, j]), num_fills[j], out[:, j])
        return out
    fill = cfg["fill_value"]
    return np.where(np.isnan(matrix), fill, matrix)


def _apply_pca(state: FittedState, cfg: dict, matrix: np.ndarray) -> np.ndarray:
    out = (matrix - state.array("mean_")) @ state.array("components_").T
    if cfg.get("whiten"):
        out = out / np.sqrt(state.array("explained_variance_"))
    return out


# -- constructors -----------------------------------------------------------------


class FittableOperator(Operator):
    """An unfitted (or fitted) learnable transformation over ``{X}``."""

    _tl_fittable_ = True

    def __init__(self, kind: str, config: dict, state: FittedState | None = None,
                 uid: str | None = None) -> None:
        if kind not in FITTABLE_KINDS:
            raise TLTypeError(f"unknown estimator '{kind}'")
        uid = uid or fresh_uid()
        self.estimator_kind = kind
        self.config = dict(config)
        root = ir.FittedTransform(kind, _cfg_tuple(config), state,
                                  ir.Param(uid, "X"))
        super().__init__(("X",), root, uid)

    @property
    def is_fitted(self) -> bool:
        return self.root.state is not None

    @property
    def parameters(self) -> dict:
        """Learned parameters (immutable); raises until fitted."""
        if not self.is_fitted:
            raise TLTypeError(
                f"operator '{self.estimator_kind}' is not fitted: call "
                f".fit(training_data) first"
            )
        return self.root.state.as_dict()

    def fit(self, data):
        """Learn from ``data``; return a NEW fitted operator (self unchanged)."""
        value, vkind = _fit_input(data, self.estimator_kind)
        state = fit_state(self.estimator_kind, self.config, value, vkind)
        return FittableOperator(self.estimator_kind, self.config, state,
                                uid=self.uid)

    def __rshift__(self, other) -> Operator:
        return _compose_pipeline(self, other)

    def _tl_rcompose(self, left) -> Operator:
        return _compose_pipeline(left, self)

    def explain(self) -> str:
        from ..runtime.inspect import format_node

        status = "fitted" if self.is_fitted else "unfitted"
        lines = [
            f"{self.estimator_kind}({status}, config={self.config}):",
            format_node(self.root),
        ]
        if self.is_fitted:
            state = self.root.state
            lines.append(
                f"learned: features={list(state.feature_names) if state.feature_names else None}, "
                f"n_features={state.n_features}, params={sorted(dict(state.arrays))}"
            )
        return "\n".join(lines)


class PipelineOperator(Operator):
    """A ``>>`` composition involving fittable stages, fitted sequentially."""

    _tl_fittable_ = True

    def __init__(self, stages: list[Operator], uid: str | None = None) -> None:
        if not stages:
            raise TLTypeError("a pipeline needs at least one stage")
        for stage in stages:
            if not isinstance(stage, Operator) or len(stage.params) != 1:
                raise TLTypeError(
                    "pipeline stages must be unary operators; "
                    f"got {type(stage).__name__}"
                )
        uid = uid or fresh_uid()
        self.stages = list(stages)
        pname = stages[0].params[0]
        root: ir.Expr = ir.Param(uid, pname)
        for stage in stages:
            root = ir.ApplyOp(stage, (root,))
        super().__init__((pname,), root, uid)

    def fit(self, data):
        """Fit unfitted stages in order, threading transformed outputs.

        Already-fitted stages are reused without refitting, so test data
        can never overwrite training statistics.
        """
        from ..core.tensor import Tensor

        value, vkind = _fit_input(data, "pipeline")
        new_stages: list[Operator] = []
        for stage in self.stages:
            if isinstance(stage, FittableOperator) and not stage.is_fitted:
                wrapped = Tensor._from_parts(vkind, value)
                stage = stage.fit(wrapped)
            value, vkind = _apply_stage(stage, value, vkind)
            new_stages.append(stage)
        return PipelineOperator(new_stages, uid=self.uid)

    def __rshift__(self, other) -> Operator:
        if isinstance(other, PipelineOperator):
            return PipelineOperator(self.stages + other.stages)
        if not isinstance(other, Operator) or len(other.params) != 1:
            raise TLTypeError(
                "pipeline stages must be unary operators; "
                f"cannot compose with {type(other).__name__}"
            )
        return PipelineOperator(self.stages + [other])

    def _tl_rcompose(self, left) -> Operator:
        if not isinstance(left, Operator) or len(left.params) != 1:
            raise TLTypeError(
                "pipeline stages must be unary operators; "
                f"cannot compose with {type(left).__name__}"
            )
        return PipelineOperator([left] + self.stages)

    def explain(self) -> str:
        lines = ["pipeline:"]
        for i, stage in enumerate(self.stages):
            if isinstance(stage, FittableOperator):
                status = "fitted" if stage.is_fitted else "unfitted"
                lines.append(f"  [{i}] {stage.estimator_kind}({status})")
            else:
                lines.append(f"  [{i}] operator(params={list(stage.params)})")
        lines.append(super().explain())
        return "\n".join(lines)


def _compose_pipeline(left, right) -> Operator:
    from ..core.operator import Operator as _Operator

    if isinstance(right, PipelineOperator):
        return right._tl_rcompose(left)
    if isinstance(left, PipelineOperator):
        return left.__rshift__(right)
    if not isinstance(left, _Operator) or len(getattr(left, "params", ())) != 1:
        raise TLTypeError(
            "pipeline stages must be unary operators; "
            f"cannot compose with {type(left).__name__}"
        )
    if not isinstance(right, _Operator) or len(right.params) != 1:
        raise TLTypeError(
            "pipeline stages must be unary operators; "
            f"cannot compose with {type(right).__name__}"
        )
    return PipelineOperator([left, right])


def _fit_input(data, estimator: str):
    from ..core.tensor import Lazy, Tensor
    from ..runtime.executor import evaluate

    if isinstance(data, Tensor):
        if data._kind == "scan":
            return data._material(), "frame"
        return data._material(), data._kind
    if isinstance(data, Lazy):
        return evaluate(data.root)
    raise TLTypeError(
        f"{estimator}.fit needs a Tensor or lazy result, got "
        f"{type(data).__name__}",
        code="TL0403",
        details={"got": type(data).__name__},
    )


def _lit(value, vkind: str) -> ir.Expr:
    from ..core.tensor import _readonly_view, literal_of

    if vkind == "dense":
        return ir.Literal(_readonly_view(np.asarray(value)), "dense")
    return literal_of(value)


def _apply_stage(stage: Operator, value, vkind: str):
    from ..runtime.executor import evaluate

    env = {(stage.uid, stage.params[0]): _lit(value, vkind)}
    return evaluate(ir.substitute(stage.root, env))


def standardize() -> FittableOperator:
    """Zero-mean, unit-variance scaling (population std; constants → 0)."""
    return FittableOperator("standardize", {})


def minmax(lo: float = 0.0, hi: float = 1.0) -> FittableOperator:
    """Min-max scaling into ``[lo, hi]`` (constant columns map to ``lo``)."""
    lo, hi = float(lo), float(hi)
    if not lo < hi:
        raise TLTypeError(f"minmax needs lo < hi, got ({lo}, {hi})",
                          code="TL0411",
                          details={"lo": lo, "hi": hi})
    return FittableOperator("minmax", {"lo": lo, "hi": hi})


def normalize(norm: str = "l2") -> FittableOperator:
    """Row-wise vector normalization (``l1`` | ``l2`` | ``max``)."""
    if norm not in ("l1", "l2", "max"):
        raise TLTypeError(
            f"unknown norm '{norm}'; use 'l1', 'l2' or 'max'",
            code="TL0414",
            details={"got": norm},
        )
    return FittableOperator("normalize", {"norm": norm})


def impute(strategy: str = "mean", fill_value=None) -> FittableOperator:
    """Missing-value imputation: mean | median | most_frequent | constant."""
    if strategy not in ("mean", "median", "most_frequent", "constant"):
        raise TLTypeError(f"unknown impute strategy '{strategy}'",
                          code="TL0414",
                          details={"got": strategy})
    if strategy == "constant" and fill_value is None:
        raise TLTypeError("impute('constant') needs an explicit fill_value",
                          code="TL0411")
    return FittableOperator(
        "impute", {"strategy": strategy, "fill_value": fill_value})


def pca(n_components: int, whiten: bool = False) -> FittableOperator:
    """Principal component analysis via SVD (exact, deterministic)."""
    if isinstance(n_components, bool) or not isinstance(n_components, int):
        raise TLTypeError(
            f"pca needs an integer component count, got {n_components!r}",
            code="TL0411",
            details={"got": repr(n_components)})
    if n_components < 1:
        raise TLTypeError(
            f"pca needs at least 1 component, got {n_components}",
            code="TL0411",
            details={"got": n_components})
    return FittableOperator(
        "pca", {"n_components": n_components, "whiten": bool(whiten)})
