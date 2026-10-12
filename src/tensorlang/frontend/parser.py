"""TensorLang parser: AST validation over translated ``.tl`` source.

The lexer rewrites ``:=`` declarations into ``__tl_declare__`` calls; the
parser then walks the real Python AST to enforce declaration rules with
exact source locations:

- every ``:=`` introduces a name in the current lexical scope,
- duplicate declarations in the same scope are rejected,
- declaration targets must be plain names.
"""
from __future__ import annotations

import ast

from ..core.errors import TLDeclarationError, TLScopeError, TLSyntaxError
from . import effects as _effects
from .lexer import DECLARE_FN
from .scope import ScopeTable


def parse(translated: str, filename: str) -> ast.Module:
    """Parse translated source and validate declarations."""
    try:
        tree = ast.parse(translated, filename=filename)
    except SyntaxError as exc:
        raise TLSyntaxError(
            f"invalid TensorLang syntax: {exc.msg}",
            loc=f"{filename}:{exc.lineno or '?'}",
            code="TL0101",
        ) from exc
    _DeclarationChecker(filename).visit(tree)
    return tree


class _DeclarationChecker(ast.NodeVisitor):
    def __init__(self, filename: str) -> None:
        self.scopes = ScopeTable(filename)

    # -- scopes ------------------------------------------------------------------
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.scopes.push("function", node.name)
        self.generic_visit(node)
        self.scopes.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.scopes.push("function", node.name)
        self.generic_visit(node)
        self.scopes.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scopes.push("class", node.name)
        self.generic_visit(node)
        self.scopes.pop()

    # -- declarations ---------------------------------------------------------------
    def visit_Assign(self, node: ast.Assign) -> None:
        value = node.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == DECLARE_FN
        ):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                raise TLScopeError(
                    "':=' must introduce a plain name in the current scope",
                    loc=f"{self.scopes.filename}:{node.lineno}",
                    code="TL0202",
                )
            name = node.targets[0].id
            args = value.args
            if not args or not isinstance(args[0], ast.Constant) or args[0].value != name:
                raise TLSyntaxError(  # pragma: no cover - defensive
                    "malformed declaration; do not call __tl_declare__ directly",
                    loc=f"{self.scopes.filename}:{node.lineno}",
                    code="TL0102",
                )
            self.scopes.declare(name, node.lineno)
            rhs = args[1]
            self._check_effects(name, value, node.lineno)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        value = node.value
        if (
            value is not None
            and isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == DECLARE_FN
            and isinstance(node.target, ast.Name)
        ):
            self.scopes.declare(node.target.id, node.lineno)
            self._check_effects(node.target.id, value, node.lineno)
        self.generic_visit(node)

    def _check_effects(self, name: str, call: ast.Call, lineno: int) -> None:
        cls, reason = _effects.classify(call.args[1])
        if cls != "declarative":
            code = "TL0302" if cls == "effectful" else "TL0303"
            raise TLDeclarationError(
                f"':=' cannot bind '{name}' to a {cls} expression: {reason}",
                loc=f"{self.scopes.filename}:{lineno}",
                code=code,
            )

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        # A surviving walrus is ordinary Python; declarations were rewritten.
        self.generic_visit(node)
