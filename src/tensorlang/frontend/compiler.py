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

from ..core.errors import TLDeclarationError
from ..core.operator import Operator, P
from ..core.tensor import Lazy, Tensor
from . import lexer as _lexer
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
        )
    if isinstance(value, Tensor):
        from ..core.types import PROVENANCE_IO

        if value._provenance == PROVENANCE_IO:
            raise TLDeclarationError(
                f"':=' cannot bind '{name}' to an effectful dataset read; "
                f"use '{name} = ...' for I/O (or T.scan_* for a lazy, "
                f"declarative source descriptor)",
                loc=loc,
            )
        value._tl_name = name
    elif isinstance(value, Operator):
        value._bound_name = name
    return value


def translate(source: str) -> str:
    return _lexer.translate(source)


def compile_source(source: str, filename: str = "<tl>") -> Any:
    """Translate, validate and compile ``.tl`` source to a code object."""
    rewritten = _lexer.translate(source)
    tree = _parser.parse(rewritten, filename)
    return compile(tree, filename, "exec")


def check_source(source: str, filename: str = "<tl>") -> ast.Module:
    """Translate and validate without compiling (for ``tensorlang check``)."""
    return _parser.parse(_lexer.translate(source), filename)


def base_namespace() -> dict[str, Any]:
    return {
        "__tl_declare__": __tl_declare__,
        "__tl_param__": __tl_param__,
    }


def run_source(source: str, filename: str = "<tl>",
               extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Execute ``.tl`` source; return the resulting namespace."""
    global CURRENT_FILENAME
    previous = CURRENT_FILENAME
    CURRENT_FILENAME = filename
    try:
        code = compile_source(source, filename)
    finally:
        CURRENT_FILENAME = previous
    namespace: dict[str, Any] = {"__name__": "__tl_main__", "__file__": filename}
    namespace.update(base_namespace())
    if extra:
        namespace.update(extra)
    CURRENT_FILENAME = filename
    try:
        exec(code, namespace)
    finally:
        CURRENT_FILENAME = previous
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
