"""Explicit in-place execution: ``f(data).inplace()``.

Semantics:

1. Only unambiguous single-input operator applications over a mutable
   :class:`Tensor` qualify (``Lazy`` objects carrying a ``_target``).
2. The computation is evaluated fully *before* touching the target.
3. The target's logical contents are replaced atomically: on failure the
   input tensor is left unchanged (same kind, data and version).
4. Aliasing is by object identity: every reference to the same ``Tensor``
   observes the update; rebinding a name never mutates.
5. Lazy expressions built before the mutation hold snapshots and are
   therefore unaffected.
"""
from __future__ import annotations

from ..core.errors import TLMutationError
from ..core.tensor import Lazy, Tensor


def apply_inplace(lazy: Lazy) -> Tensor:
    from .executor import evaluate

    target = lazy._target
    if target is None:
        raise TLMutationError(
            "inplace() needs an unambiguous single-input operator applied "
            "to a mutable Tensor, e.g. f(data).inplace(); this computation "
            "has no single mutable target"
        )
    if not isinstance(target, Tensor):
        raise TLMutationError(  # pragma: no cover - defensive
            "inplace() target must be a mutable Tensor"
        )
    if target._kind == "scan":
        raise TLMutationError(
            "cannot inplace() into a lazy scan descriptor: scans are "
            "immutable sources; collect() into a tensor first"
        )
    # Evaluate first; the target is untouched until the commit below.
    value, kind = evaluate(lazy._root)
    _commit(target, value, kind)
    lazy._target = None
    return target


def _commit(target: Tensor, value, kind: str) -> None:
    import numpy as np
    import polars as pl

    from ..core.types import PROVENANCE_DERIVED

    old_kind, old_data, old_version = target._kind, target._data, target._version
    old_prov = target._provenance
    try:
        if kind == "dense":
            payload = np.array(value, copy=True)
        elif kind == "frame":
            if not isinstance(value, pl.DataFrame):
                raise TLMutationError(
                    f"cannot commit a {type(value).__name__} result into a "
                    f"structured tensor"
                )
            payload = value.clone()
        elif kind == "series":
            if not isinstance(value, pl.Series):
                raise TLMutationError(  # pragma: no cover - defensive
                    f"cannot commit a {type(value).__name__} result"
                )
            payload = value.clone()
        elif kind == "scalar":
            payload = np.asarray(value)
        else:  # pragma: no cover - defensive
            raise TLMutationError(f"cannot commit a '{kind}' result inplace")
        target._replace(kind if kind != "scalar" else "dense", payload)
        target._provenance = PROVENANCE_DERIVED
    except TLMutationError:
        target._kind, target._data, target._version = old_kind, old_data, old_version
        target._provenance = old_prov
        raise
    except Exception as exc:
        target._kind, target._data, target._version = old_kind, old_data, old_version
        target._provenance = old_prov
        raise TLMutationError(f"inplace execution failed: {exc}") from exc
