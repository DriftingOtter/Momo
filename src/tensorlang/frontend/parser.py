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


def parse(translated: str, filename: str,
          trusted_names: frozenset[str] | None = None,
          injected_names: frozenset[str] | None = None) -> ast.Module:
    """Parse translated source and validate declarations.

    ``trusted_names`` seeds names already bound to TensorLang public
    objects in the execution namespace (identity-checked by the caller;
    used for REPL inputs where imports live in earlier pushes).
    ``injected_names`` holds Python-injected (``extra=``) bindings: a
    ``:=`` declaration colliding with one is an explicit ``TL0203``
    conflict (``=`` rebinding keeps module precedence).
    """
    try:
        tree = ast.parse(translated, filename=filename)
    except SyntaxError as exc:
        raise TLSyntaxError(
            f"invalid TensorLang syntax: {exc.msg}",
            loc=f"{filename}:{exc.lineno or '?'}",
            code="TL0101",
        ) from exc
    _DeclarationChecker(filename, trusted_names, injected_names).visit(tree)
    return tree


class _DeclarationChecker(ast.NodeVisitor):
    def __init__(self, filename: str,
                 trusted_names: frozenset[str] | None = None,
                 injected_names: frozenset[str] | None = None) -> None:
        self.scopes = ScopeTable(filename)
        # Names bound to the tensorlang package: same-file imports plus
        # identity-seeded namespace names.  Any other rebinding shadows
        # (fail-closed); see _shadow().
        self.tl_names: set[str] = set(trusted_names or ())
        # All ``:=``-introduced names in visit order (flat approximation:
        # function-local declarations also register; cross-scope leakage
        # fails at runtime with NameError as before).
        self.declared: set[str] = set()
        # Python-injected bindings: declaring over one is TL0203.
        self.injected: set[str] = set(injected_names or ())

    def _shadow(self, name: str) -> None:
        """A rebinding to an unknown value: drop all trust in ``name``."""
        self.tl_names.discard(name)
        self.declared.discard(name)

    def _shadow_target(self, target: ast.AST) -> None:
        for node in ast.walk(target):
            if isinstance(node, ast.Name):
                self._shadow(node.id)

    # -- imports -------------------------------------------------------------------
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            bound = alias.asname or alias.name.split(".")[0]
            if alias.name.split(".")[0] == "tensorlang":
                self.tl_names.add(bound)
                self.declared.discard(bound)
            else:
                self._shadow(bound)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        is_tl = node.level == 0 and (
            module == "tensorlang" or module.startswith("tensorlang."))
        for alias in node.names:
            if alias.name == "*":
                if is_tl:
                    self.tl_names.update(_effects.tl_export_names())
                continue
            bound = alias.asname or alias.name
            if is_tl:
                self.tl_names.add(bound)
                self.declared.discard(bound)
            else:
                self._shadow(bound)
        self.generic_visit(node)

    # -- scopes ------------------------------------------------------------------
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._shadow(node.name)
        self._shadow_args(node.args)
        self.scopes.push("function", node.name)
        self.generic_visit(node)
        self.scopes.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._shadow(node.name)
        self._shadow_args(node.args)
        self.scopes.push("function", node.name)
        self.generic_visit(node)
        self.scopes.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._shadow(node.name)
        self.scopes.push("class", node.name)
        self.generic_visit(node)
        self.scopes.pop()

    def _shadow_args(self, args: ast.arguments) -> None:
        for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
            self._shadow(arg.arg)
        for extra in (args.vararg, args.kwarg):
            if extra is not None:
                self._shadow(extra.arg)

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
            self._check_injected(name, node.lineno)
            self.tl_names.discard(name)
            rhs = args[1]
            self._check_effects(name, value, node.lineno)
            self.declared.add(name)
        else:
            for target in node.targets:
                self._shadow_target(target)
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
            self._check_injected(node.target.id, node.lineno)
            self.tl_names.discard(node.target.id)
            self._check_effects(node.target.id, value, node.lineno)
            self.declared.add(node.target.id)
        elif value is not None:
            self._shadow_target(node.target)
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self._shadow_target(node.target)
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._shadow_target(node.target)
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            if item.optional_vars is not None:
                self._shadow_target(item.optional_vars)
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        for item in node.items:
            if item.optional_vars is not None:
                self._shadow_target(item.optional_vars)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self._shadow(node.name)
        self.generic_visit(node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        # A surviving walrus is ordinary Python; declarations were rewritten.
        # Its target still rebinds, so it shadows trust.
        self._shadow_target(node.target)
        self.generic_visit(node)

    def _check_injected(self, name: str, lineno: int) -> None:
        if name in self.injected:
            raise TLScopeError(
                f"':=' cannot declare '{name}': it collides with a "
                f"Python-injected binding (extra=); rename the declaration "
                f"or stop injecting the name ('=' rebinding keeps module "
                f"precedence)",
                loc=f"{self.scopes.filename}:{lineno}",
                code="TL0203",
            )

    def _check_effects(self, name: str, call: ast.Call, lineno: int) -> None:
        cls, reason = _effects.classify(call.args[1])
        if cls != "declarative":
            code = "TL0302" if cls == "effectful" else "TL0303"
            raise TLDeclarationError(
                f"':=' cannot bind '{name}' to a {cls} expression: {reason}",
                loc=f"{self.scopes.filename}:{lineno}",
                code=code,
            )
        _effects.check_declaration_calls(
            call.args[1], tl_names=self.tl_names, declared=self.declared,
            filename=self.scopes.filename, bind_name=name, lineno=lineno)
