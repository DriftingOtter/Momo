"""Shared vocabulary: representation kinds, expression provenance, scan specs.

v0.2 formalizes three expression classes:

- **declarative** — describes a computation without executing it: tensor
  literals (``T(...)``), operator definitions, compositions, pure symbolic
  constructors, and lazy *descriptors* such as file scans.
- **runtime** — depends on execution: operator applications, ``collect()``,
  conversions, and any pending (``Lazy``) computation.
- **effectful** — performs I/O or mutation: ``T.read``, ``write_*``,
  ``.inplace()``.

Only declarative expressions are legal under ``:=``; ``=`` handles the
rest.  See ``frontend/effects.py`` for the syntactic classifier.
"""
from __future__ import annotations

from dataclasses import dataclass

PROVENANCE_LITERAL = "literal"      # pure construction: T(...), linspace, ...
PROVENANCE_IO = "io"                # effectful read: T.read(...)
PROVENANCE_DERIVED = "derived"      # produced by in-place execution
PROVENANCE_DESCRIPTOR = "descriptor"  # lazy source description: T.scan_*


@dataclass(frozen=True)
class ScanSpec:
    """A declarative description of a lazily scanned dataset."""

    paths: tuple[str, ...]
    format: str  # 'parquet' | 'csv'
    options: tuple[tuple[str, object], ...] = ()

    def describe(self) -> str:
        return f"scan_{self.format}({', '.join(self.paths)})"

    def kwargs(self) -> dict:
        return dict(self.options)


@dataclass(frozen=True)
class ChunkSpec:
    """Logical shape plus physical partitioning of a persistent array.

    Chunking is a physical execution property: operators always see one
    logical tensor regardless of partition count.
    """

    path: str
    shape: tuple[int, ...]
    dtype: str
    chunks: tuple[int, ...]

    def describe(self) -> str:
        return (f"scan_zarr({self.path}, shape={list(self.shape)}, "
                f"chunks={list(self.chunks)})")

    @property
    def npartitions(self) -> int:
        import math

        total = 1
        for dim, chunk in zip(self.shape, self.chunks):
            total *= max(math.ceil(dim / chunk), 1)
        return total

    @property
    def nbytes(self) -> int:
        import numpy as _np

        total = 1
        for dim in self.shape:
            total *= dim
        return total * _np.dtype(self.dtype).itemsize

    @property
    def chunk_nbytes(self) -> int:
        import numpy as _np

        total = 1
        for chunk in self.chunks:
            total *= chunk
        return total * _np.dtype(self.dtype).itemsize


JOIN_HOW = frozenset({
    "inner", "left", "right", "full", "outer", "cross", "semi", "anti",
})

WINDOW_OPS = frozenset({"sum", "mean", "min", "max"})
