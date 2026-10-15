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

Calls are fail-closed: every call (and every attribute/subscript access)
in a ``:=`` right-hand side must resolve to a *known* TensorLang
capability — a name imported from ``tensorlang``, the ``tensorlang``
module itself, a name introduced by ``:=`` in the same file, a name
whose bound value is provably a TensorLang public object (namespace
identity seeding, e.g. across REPL inputs), or the internal
``__tl_param__`` hook.  Anything else (imported Python helpers, bound
methods of foreign objects, ``os.path`` utilities, builtins such as
``print``/``len``) is rejected as *unclassified* (``TL0305``) without
being executed.  This is effect enforcement, not sandboxing: property
getters, operator dunders on foreign objects, and comprehension
iteration can still run user code — see ``docs/EFFECTS.md``.
"""
from __future__ import annotations

import ast

from ..core.errors import TLDeclarationError

# Attribute calls that are always effectful, whoever the receiver is.
# These names are reserved as execution boundaries / mutation entry points.
EFFECTFUL_ATTRS = frozenset({"write_parquet", "write_csv", "write_zarr",
                             "inplace"})

# Attribute calls that always execute a computation (runtime, not declarative).
RUNTIME_ATTRS = frozenset({"collect", "to_numpy", "to_polars", "fit"})

# Internal hooks rewritten by the lexer; never user code, always admissible
# inside a declaration right-hand side.
INTERNAL_FNS = frozenset({"__tl_declare__", "__tl_param__"})


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


def tl_export_names() -> set[str]:
    """Names the ``tensorlang`` package exports (for ``import *`` handling).

    Read lazily from the live package so this module has no import cycle.
    """
    import sys

    tl = sys.modules.get("tensorlang")
    names = set(getattr(tl, "__all__", ()) or ())
    names.update({"T", "P", "tensorlang"})
    return names


def _names_in(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def check_declaration_calls(rhs: ast.AST, *, tl_names: set[str],
                            declared: set[str], filename: str,
                            bind_name: str, lineno: int) -> None:
    """Reject unclassified calls/accesses in a ``:=`` right-hand side.

    ``tl_names`` holds names bound to the ``tensorlang`` package in this
    file (imports, plus identity-seeded namespace names); ``declared``
    holds ``:=``-introduced names.  Known effectful/runtime attribute
    calls are classified by :func:`classify` first — this check only
    fires for calls that survived it.  Raises ``TLDeclarationError``
    (``TL0305``) without executing anything.
    """
    allowed = set(tl_names) | set(declared) | set(INTERNAL_FNS)

    class _Finder(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            func = node.func
            if isinstance(func, ast.Name):
                names = {func.id}
            elif isinstance(func, ast.Attribute):
                names = _names_in(func.value)
                if not names:
                    # Literal receiver: "s".upper(), [1].copy() — builtin
                    # methods on literals run no user code.
                    self.generic_visit(node)
                    return
            else:  # subscripted / computed callee: fail closed
                names = _names_in(func)
            if not names or not names <= allowed:
                _reject(node, func)
                return
            self.generic_visit(node)

        def visit_Attribute(self, node: ast.Attribute) -> None:
            # Property getters and __getattr__ run user code on access, so
            # access roots must resolve just like callees.  Literal
            # receivers (no names) are builtin-typed and safe.
            names = _names_in(node.value)
            if names and not names <= allowed:
                _reject(node, node)
                return
            self.generic_visit(node)

        def visit_Subscript(self, node: ast.Subscript) -> None:
            # __getitem__ runs user code: same resolution rule.  The slice
            # itself is visited for nested calls.
            names = _names_in(node.value)
            if names and not names <= allowed:
                _reject(node, node.value)
                return
            self.generic_visit(node)

    def _reject(node: ast.AST, target: ast.AST) -> None:
        try:
            shown = ast.unparse(target).strip()
        except Exception:  # pragma: no cover - defensive
            shown = "<expression>"
        if len(shown) > 80:
            shown = shown[:77] + "..."
        raise TLDeclarationError(
            f"':=' cannot bind '{bind_name}' to an expression containing "
            f"an unclassified Python access: {shown}. Arbitrary Python "
            f"calls cannot execute under ':='; use a supported "
            f"declarative TensorLang operation, or move the call into an "
            f"explicitly runtime context ('{bind_name} = ...' instead)",
            loc=f"{filename}:{getattr(node, 'lineno', lineno)}",
            code="TL0305",
            details={"call": shown},
        )

    _Finder().visit(rhs)
