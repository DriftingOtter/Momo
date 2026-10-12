"""NumPy backend: dense (n-dimensional array) execution primitives."""
from __future__ import annotations

import numpy as np

from ..core.errors import TLShapeError, TLTypeError

_BOOL_OPS = {"eq", "ne", "gt", "lt", "ge", "le", "and", "or"}


def _as_array(value):
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def _power(a: np.ndarray, b):
    """Power with integer-exponent fast paths.

    Plain ``np.power`` routes every call through the generic (slow) libm
    loop, while expressions users actually write (``a**2``) take NumPy's
    ``square`` fast path.  Matching that keeps backend parity.
    """
    exp = None
    if isinstance(b, np.ndarray) and b.ndim == 0:
        try:
            f = float(b.item())
        except (TypeError, ValueError):
            f = None
        if f is not None and f.is_integer():
            exp = int(f)
    elif isinstance(b, (int, np.integer)) and not isinstance(b, bool):
        exp = int(b)
    if exp == 0:
        return np.ones_like(a)
    if exp == 1:
        return np.array(a, copy=True)
    if exp == 2:
        return np.square(a)
    if exp == 3:
        return np.multiply(np.square(a), a)
    if exp == -1 and a.dtype.kind == "f":
        # Float reciprocal; integer bases fall through so np.power raises
        # exactly as native `ints ** -1` does.
        return np.reciprocal(a)
    return np.power(a, b)


def apply_binop(op: str, left, right):
    """Elementwise binary op with NumPy-compatible broadcasting."""
    a = _as_array(left)
    b = _as_array(right)
    try:
        broadcast_shape = np.broadcast_shapes(a.shape, b.shape)
    except ValueError:
        raise TLShapeError(
            f"cannot broadcast shapes {a.shape} and {b.shape} "
            f"for operator '{op}'"
        ) from None
    try:
        if op == "add":
            return np.add(a, b)
        if op == "sub":
            return np.subtract(a, b)
        if op == "mul":
            return np.multiply(a, b)
        if op == "div":
            return np.true_divide(a, b)
        if op == "pow":
            return _power(a, b)
        if op == "mod":
            return np.mod(a, b)
        if op == "eq":
            return np.equal(a, b)
        if op == "ne":
            return np.not_equal(a, b)
        if op == "gt":
            return np.greater(a, b)
        if op == "lt":
            return np.less(a, b)
        if op == "ge":
            return np.greater_equal(a, b)
        if op == "le":
            return np.less_equal(a, b)
        if op == "and":
            return np.logical_and(a, b)
        if op == "or":
            return np.logical_or(a, b)
    except ValueError as exc:
        # Shapes already broadcast (checked above): a genuine NumPy domain
        # error (e.g. integer ** negative). Report it, don't mislabel it.
        raise TLTypeError(f"cannot apply '{op}': {exc}") from exc
    raise TLTypeError(f"unknown binary operator '{op}'")  # pragma: no cover


def apply_unop(op: str, operand):
    a = _as_array(operand)
    if op == "neg":
        return np.negative(a)
    if op == "abs":
        return np.abs(a)
    if op == "not":
        return np.logical_not(a)
    raise TLTypeError(f"unknown unary operator '{op}'")  # pragma: no cover


def apply_math_func(func: str, arg):
    a = _as_array(arg)
    table = {
        "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
    }
    try:
        return table[func](a)
    except KeyError:
        raise TLTypeError(f"unknown math function '{func}'") from None


def apply_reduce(op: str, arr: np.ndarray):
    table = {
        "sum": np.sum, "mean": np.mean, "min": np.min,
        "max": np.max, "std": np.std, "var": np.var,
    }
    try:
        fn = table[op]
    except KeyError:
        raise TLTypeError(f"unknown reduction '{op}'") from None
    return fn(arr)


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    try:
        return np.matmul(a, b)
    except ValueError as exc:
        raise TLShapeError(
            f"matmul requires compatible numerical operands, got shapes "
            f"{a.shape} and {b.shape}"
        ) from exc


def transpose(a: np.ndarray) -> np.ndarray:
    return np.transpose(a)


def reshape(a: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    try:
        return np.reshape(a, shape)
    except ValueError as exc:
        raise TLShapeError(
            f"cannot reshape array of shape {a.shape} to {shape}"
        ) from exc


def getitem(arr: np.ndarray, index):
    try:
        return arr[index]
    except IndexError as exc:
        raise TLShapeError(f"invalid index for shape {arr.shape}: {exc}") from exc


def concat(parts: list[np.ndarray], axis: int = 0) -> np.ndarray:
    try:
        return np.concatenate(parts, axis=axis)
    except ValueError as exc:
        raise TLShapeError(f"cannot concatenate arrays: {exc}") from exc


def stack(parts: list[np.ndarray], axis: int = 0) -> np.ndarray:
    try:
        return np.stack(parts, axis=axis)
    except ValueError as exc:
        raise TLShapeError(f"cannot stack arrays: {exc}") from exc


def is_null(arr: np.ndarray) -> np.ndarray:
    a = _as_array(arr)
    if a.dtype.kind == "f":
        return np.isnan(a)
    if a.dtype.kind in ("O",):
        return np.array([v is None for v in a.flat]).reshape(a.shape)
    return np.zeros(a.shape, dtype=bool)
