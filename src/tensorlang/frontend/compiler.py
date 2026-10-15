"""TensorLang compiler: ``.tl`` source text to executable code.

Pipeline: :mod:`lexer` (token rewrite) -> :mod:`parser` (AST + scope
validation) -> ``compile`` -> execution with a namespace providing the
``__tl_declare__`` / ``__tl_param__`` runtime hooks.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any

from ..core.errors import TLDeclarationError, TLScopeError
from ..core.operator import Operator, P
from ..core.tensor import Lazy, Tensor
from . import lexer as _lexer
from . import modules as _modules
from . import parser as _parser

CURRENT_FILENAME = "<tl>"


def __tl_param__(name: str) -> Operator:
    """Runtime hook for ``{X}``: a fresh symbolic parameter."""
    return P(name)


def __tl_declare__(name: str, value: Any, lineno: int) -> Any:
    """Runtime hook for ``:=``: validate declarative definitions.

    Allowed: literal tensor constructors, symbolic operator expressions,
    compositions of declared operators, pure symbolic constructors, and
    other plain (non-computed) values.  Rejected: runtime operator
    applications such as ``result := f(data)`` — those must use ``=``.
    """
    loc = f"{CURRENT_FILENAME}:{lineno}"
    if isinstance(value, Lazy):
        raise TLDeclarationError(
            f"':=' cannot bind '{name}' to a runtime operator application; "
            f"operator application is a runtime operation, use "
            f"'{name} = ...' instead",
            loc=loc,
            code="TL0301",
        )
    if isinstance(value, Tensor):
        from ..core.types import PROVENANCE_IO

        if value._provenance == PROVENANCE_IO:
            raise TLDeclarationError(
                f"':=' cannot bind '{name}' to an effectful dataset read; "
                f"use '{name} = ...' for I/O (or T.scan_* for a lazy, "
                f"declarative source descriptor)",
                loc=loc,
                code="TL0304",
            )
        value._tl_name = name
    elif isinstance(value, Operator):
        value._bound_name = name
    return value


def translate(source: str) -> str:
    return _lexer.translate(source)


def compile_source(source: str, filename: str = "<tl>",
                   trusted_names: frozenset[str] | None = None,
                   injected_names: frozenset[str] | None = None) -> Any:
    """Translate, validate and compile ``.tl`` source to a code object."""
    code, _ = compile_source_tree(source, filename, trusted_names,
                                  injected_names)
    return code


def compile_source_tree(source: str, filename: str = "<tl>",
                        trusted_names: frozenset[str] | None = None,
                        injected_names: frozenset[str] | None = None):
    """Translate and validate, returning ``(code, ast)``.

    Exposed for tooling (REPL display rewriting) that needs the validated
    tree alongside the executable code.  ``trusted_names`` seeds names
    already bound to TensorLang public objects (identity-checked by the
    caller).  ``injected_names`` holds Python-injected bindings that
    ``:=`` must not silently shadow (``TL0203``).
    """
    rewritten = _lexer.translate(source)
    tree = _parser.parse(rewritten, filename, trusted_names, injected_names)
    return compile(tree, filename, "exec"), tree


def check_source(source: str, filename: str = "<tl>",
                 trusted_names: frozenset[str] | None = None,
                 injected_names: frozenset[str] | None = None) -> ast.Module:
    """Translate and validate without compiling (for ``tensorlang check``)."""
    return _parser.parse(_lexer.translate(source), filename, trusted_names,
                         injected_names)


_TRUSTED_IDS: frozenset[int] | None = None


def _trusted_ids() -> frozenset[int]:
    """Identities of the ``tensorlang`` module and its public members."""
    global _TRUSTED_IDS
    if _TRUSTED_IDS is None:
        ids: set[int] = set()
        tl = sys.modules.get("tensorlang")
        if tl is not None:
            ids.add(id(tl))
            for name in getattr(tl, "__all__", ()) or ():
                try:
                    ids.add(id(getattr(tl, name)))
                except AttributeError:
                    pass
        _TRUSTED_IDS = frozenset(ids)
    return _TRUSTED_IDS


def namespace_trusted_names(namespace: dict[str, Any]) -> frozenset[str]:
    """Names in ``namespace`` bound to TensorLang public objects.

    Identity-based (never name-based): a REPL input may reuse ``T``
    imported by an earlier input, but never ``evil``.
    """
    ids = _trusted_ids()
    if not ids:
        return frozenset()
    return frozenset(k for k, v in namespace.items()
                     if isinstance(k, str) and id(v) in ids)


def base_namespace() -> dict[str, Any]:
    return {
        "__tl_declare__": __tl_declare__,
        "__tl_param__": __tl_param__,
    }


def run_source(source: str, filename: str = "<tl>",
               extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Execute ``.tl`` source; return the resulting namespace.

    The outermost call opens a :class:`LoaderSession` (per-execution
    ``.tl`` module cache); nested calls join the active session.

    ``extra`` injects Python values into the entry namespace only
    (imported ``.tl`` modules never inherit them).  Precedence is
    deterministic: ``=`` rebinding keeps module precedence (assignment
    is explicit override intent); ``:=`` declaring over an injected
    name is a ``TL0203`` conflict; dunder names are rejected.  Injected
    callables are never trusted implicitly — M1 ``TL0305`` rules apply.
    """
    base = _modules.entry_base_dir(filename)
    sess = _modules.active_session()
    if sess is None:
        with _modules.loader_session(base) as fresh:
            return _run_with_session(source, filename, extra, fresh, base,
                                     prescan=True)
    _modules.ensure_owner(sess)
    return _run_with_session(source, filename, extra, sess, base,
                             prescan=False)


def _run_with_session(source: str, filename: str,
                      extra: dict[str, Any] | None,
                      sess, base: str, prescan: bool) -> dict[str, Any]:
    """Compile, cycle pre-scan, and execute under ``sess``."""
    global CURRENT_FILENAME
    previous = CURRENT_FILENAME
    CURRENT_FILENAME = filename
    if extra:
        for key in extra:
            if not isinstance(key, str) or not key.isidentifier() \
                    or key.startswith("__"):
                raise TLScopeError(
                    f"invalid injected binding {key!r}: extra= names must be "
                    f"plain identifiers and must not override interpreter "
                    f"names",
                    loc=filename, code="TL0203",
                )
    try:
        code, tree = compile_source_tree(
            source, filename,
            namespace_trusted_names(extra) if extra else None,
            frozenset(extra) if extra else None)
    finally:
        CURRENT_FILENAME = previous
    if prescan:
        legs = _modules.plan_cycle(tree, base, entry_name=filename)
        if legs:
            raise TLDeclarationError(
                _modules.describe_cycle(legs),
                loc=None, code="TL0312", details={"cycle": [r for r, _, _ in legs]})
    namespace: dict[str, Any] = {"__name__": "__tl_main__", "__file__": filename}
    namespace.update(base_namespace())
    if extra:
        namespace.update(extra)
    sess.exec_stack.append(base)
    had_top = hasattr(sess, "_exec_top")
    previous_top = getattr(sess, "_exec_top", None)
    sess._exec_top = filename  # type: ignore[attr-defined]
    CURRENT_FILENAME = filename
    try:
        exec(code, namespace)
    except ImportError as exc:
        raise _modules.convert_entry_import_error(
            sess, filename, exc) from exc
    finally:
        CURRENT_FILENAME = previous
        sess.exec_stack.pop()
        if had_top:
            sess._exec_top = previous_top  # type: ignore[attr-defined]
        else:
            try:
                delattr(sess, "_exec_top")
            except AttributeError:
                pass
    return namespace


def run_file(path: str | Path) -> dict[str, Any]:
    """Execute a ``.tl`` source file."""
    path = Path(path)
    if path.suffix != ".tl":
        raise TLDeclarationError(f"TensorLang source files use '.tl': {path}")
    source = path.read_text(encoding="utf-8")
    sys.path.insert(0, str(path.resolve().parent))
    try:
        return run_source(source, filename=str(path))
    finally:
        try:
            sys.path.remove(str(path.resolve().parent))
        except ValueError:
            pass
