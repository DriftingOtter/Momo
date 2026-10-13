"""Syntactic classification of ``:=`` right-hand sides.

Three classes (see ``core/types.py``):

- **declarative** — describes a computation without executing it.  Always
  legal under ``:=``.
- **runtime** — needs execution (operator application, ``collect()``,
  conversions).  Must use ``=``.
- **effectful** — performs I/O or mutation (``T.read``, ``write_*``,
  ``.inplace()``).  Must use ``=``.

The classifier works on the translated AST, so verdicts never depend on
which values happen to be returned at runtime.  Data-dependent calls whose
class cannot be known statically (e.g. ``sin(x)`` — declarative for an
operator argument, runtime for tensor data) are left to the runtime value
check in ``__tl_declare__``, which rejects pending ``Lazy`` computations
and tensors of I/O provenance.  Both layers report source locations.
"""
from __future__ import annotations

import ast

# Attribute calls that are always effectful, whoever the receiver is.
# These names are reserved as execution boundaries / mutation entry points.
EFFECTFUL_ATTRS = frozenset({"write_parquet", "write_csv", "write_zarr",
                             "inplace"})

# Attribute calls that always execute a computation (runtime, not declarative).
RUNTIME_ATTRS = frozenset({"collect", "to_numpy", "to_polars", "fit"})


def find_effects(tree: ast.AST) -> list[tuple[str, int]]:
    """Statically locate effectful calls (writes, inplace, T.read).

    Used by introspection tooling (`graph`/`explain`/`profile`) to warn
    before executing a file whose inspection would otherwise perform I/O
    or mutation as a side effect. Returns ``(dotted-name, lineno)``.
    """
    found: list[tuple[str, int]] = []

    class _Finder(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            func = node.func
            if isinstance(func, ast.Attribute):
                if func.attr in EFFECTFUL_ATTRS:
                    found.append((f".{func.attr}(...)", node.lineno))
                elif (func.attr == "read"
                        and isinstance(func.value, ast.Name)
                        and func.value.id == "T"):
                    found.append(("T.read(...)", node.lineno))
            self.generic_visit(node)

    _Finder().visit(tree)
    return found


def classify(tree: ast.AST) -> tuple[str, str]:
    """Classify an expression AST. Returns ``(class, reason)``."""
    checker = _Classifier()
    checker.visit(tree)
    if checker.effectful:
        return "effectful", checker.effectful
    if checker.runtime:
        return "runtime", checker.runtime
    return "declarative", "pure construction / operator expression"


class _Classifier(ast.NodeVisitor):
    def __init__(self) -> None:
        self.effectful: str | None = None
        self.runtime: str | None = None

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute):
            if func.attr in EFFECTFUL_ATTRS:
                self.effectful = (
                    f".{func.attr}(...) performs I/O or mutation; "
                    f"use '=' for effectful operations"
                )
            elif func.attr in RUNTIME_ATTRS:
                self.runtime = (
                    f".{func.attr}(...) executes a computation; "
                    f"use '=' for runtime-dependent assignment"
                )
            elif (
                func.attr == "read"
                and isinstance(func.value, ast.Name)
                and func.value.id == "T"
            ):
                self.effectful = (
                    "T.read(...) reads a dataset eagerly; use '=' for "
                    "effectful reads (or T.scan_* for a lazy descriptor)"
                )
        self.generic_visit(node)
