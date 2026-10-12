"""Interactive TensorLang loop (try-it-now playground).

Each input goes through the real frontend (translate → scope-check →
compile), so `.tl` semantics — `:=`, `{X}`, effects — hold exactly. A
persistent namespace carries bindings across inputs. Expression statements
display their values (display-only; nothing is ever mutated to show it).
Colon commands (`:help`) drive introspection without polluting the
language itself.
"""
from __future__ import annotations

from typing import Any

from ..core.errors import TensorLangError

INCOMPLETE_CORES = (
    "unexpected EOF",
    "expected an indented block",
    "was never closed",
    "unterminated",
)

HELP = """TensorLang REPL — every line is real .tl source.
Statements execute silently; a trailing expression displays its value.
Multiline blocks buffer until complete (Ctrl-C cancels the buffer).

commands:
  :help             this text
  :vars             bound names with one-line summaries
  :explain NAME     logical + physical plan for a binding
  :profile NAME     measured costs for a lazy binding
  :graph NAME       logical IR graph for a binding
  :reset            clear all bindings
  :quit             leave (:exit and :q work too)"""


class Repl:
    """Testable core loop: push() lines, get output text back."""

    def __init__(self) -> None:
        from . import compiler as _compiler

        self._compiler = _compiler
        self.namespace: dict = {"__name__": "__tl_main__",
                                "__file__": "<repl>"}
        self.namespace.update(_compiler.base_namespace())
        self._buffer: list[str] = []

    # -- input handling -----------------------------------------------------------
    def push(self, line: str) -> str | None:
        """Feed one line. Returns output text, or None when buffering."""
        if line.strip().startswith(":") and not self._buffer:
            return self._command(line.strip())
        self._buffer.append(line)
        source = "\n".join(self._buffer)
        if self._incomplete(source):
            return None
        self._buffer.clear()
        return self._execute(source)

    def _incomplete(self, source: str) -> bool:
        import tokenize as _tokenize

        from ..core.errors import TLSyntaxError as _TLSyntaxError

        try:
            self._compiler.compile_source_tree(source, filename="<repl>")
        except _TLSyntaxError as exc:
            marker = "invalid TensorLang syntax: "
            message = exc.message
            if message.startswith(marker):
                message = message[len(marker):]
            return any(core in message for core in INCOMPLETE_CORES)
        except (SyntaxError, _tokenize.TokenError) as exc:
            # Raw lexer failures (translate phase): TokenError means more
            # input is coming; anything else surfaces on execution.
            if isinstance(exc, _tokenize.TokenError):
                return True
            message = getattr(exc, "msg", str(exc)) or ""
            return any(core in message for core in INCOMPLETE_CORES)
        except TensorLangError:
            # Scope/effect errors are complete inputs: surface them now.
            return False
        return False

    def _execute(self, source: str) -> str:
        from . import compiler as _compiler

        try:
            code, tree = _compiler.compile_source_tree(source, filename="<repl>")
        except TensorLangError as exc:
            return _format_error(exc)
        import ast as _ast

        if tree.body and isinstance(tree.body[-1], _ast.Expr):
            code = self._with_display_capture(source, tree)
        previous = _compiler.CURRENT_FILENAME
        _compiler.CURRENT_FILENAME = "<repl>"
        try:
            exec(code, self.namespace)
        except TensorLangError as exc:
            return _format_error(exc)
        except Exception as exc:  # noqa: BLE001 - surface honestly
            from . import diagnose as _diagnose

            report = {"ok": False, "error": _diagnose.error_report(exc),
                      "suggestions": _diagnose.suggest_for(exc, self.namespace)}
            return _diagnose.format_human(report)
        finally:
            _compiler.CURRENT_FILENAME = previous
        captured = self.namespace.pop("__tl_repl_display__", None)
        return self._display_captured(tree, captured)

    @staticmethod
    def _with_display_capture(source: str, tree) -> Any:
        """Recompile with the trailing expression captured once.

        Splices ``__tl_repl_display__ = (<expr>)`` over the trailing
        expression's exact span (translated coordinates). Only lines at or
        after the final statement change, so user line numbers — and
        therefore tracebacks — are preserved, and side-effecting
        expressions execute exactly once.
        """
        import ast as _ast

        from . import lexer as _lexer

        node = tree.body[-1]
        translated = _lexer.translate(source)
        lines = translated.splitlines(keepends=True)
        first, last = node.lineno - 1, node.end_lineno - 1
        # End splice first: on a single-line expression the start splice
        # would otherwise shift the end offsets stale.
        lines[last] = (lines[last][:node.end_col_offset] + ")"
                       + lines[last][node.end_col_offset:])
        lines[first] = (lines[first][:node.col_offset]
                        + "__tl_repl_display__ = ("
                        + lines[first][node.col_offset:])
        return compile("".join(lines), "<repl>", "exec")

    def _display_captured(self, tree, captured) -> str:
        import ast as _ast

        if not tree.body or not isinstance(tree.body[-1], _ast.Expr):
            return ""
        if captured is None:
            return ""  # like Python: None results stay silent
        return _display_value(captured)

    # -- commands --------------------------------------------------------------------
    def _command(self, line: str) -> str:
        parts = line.split()
        cmd, args = parts[0], parts[1:]
        if cmd in (":quit", ":exit", ":q"):
            raise SystemExit(0)
        if cmd == ":help":
            return HELP
        if cmd == ":vars":
            return self._vars()
        if cmd == ":reset":
            keep = {k: v for k, v in self.namespace.items()
                    if k in ("__name__", "__file__", "__tl_declare__",
                             "__tl_param__", "__builtins__")}
            self.namespace.clear()
            self.namespace.update(keep)
            return "(bindings cleared)"
        if cmd in (":explain", ":profile", ":graph") and len(args) == 1:
            return self._inspect(cmd[1:], args[0])
        if cmd in (":explain", ":profile", ":graph"):
            return f"usage: {cmd} NAME"
        return f"unknown command '{cmd}'. Type :help."

    def _vars(self) -> str:
        names = sorted(k for k in self.namespace
                       if not k.startswith("_")
                       and k not in ("__name__", "__file__"))
        if not names:
            return "(no bindings)"
        return "\n".join(f"{name} : {_summary(self.namespace[name])}"
                         for name in names)

    def _inspect(self, kind: str, name: str) -> str:
        if name not in self.namespace:
            import difflib as _difflib

            pool = [k for k in self.namespace if not k.startswith("_")]
            close = _difflib.get_close_matches(name, pool, n=1, cutoff=0.6)
            hint = f" Did you mean '{close[0]}'?" if close else ""
            return f"unknown name '{name}'.{hint}"
        value = self.namespace[name]
        if kind == "graph":
            from ..runtime.inspect import format_node

            root = getattr(value, "root", getattr(value, "_root", None))
            if root is None:
                return _summary(value)
            return format_node(root)
        if kind == "explain":
            if hasattr(value, "explain"):
                return value.explain()
            return _summary(value)
        if kind == "profile":
            if hasattr(value, "profile"):
                return str(value.profile())
            return (f"{_summary(value)} has no profile; "
                    f"apply it to data first for a lazy result")
        return _summary(value)  # pragma: no cover - defensive


def _summary(value) -> str:
    from ..core.operator import Operator
    from ..core.tensor import Lazy, Tensor

    if isinstance(value, Tensor):
        try:
            abstract = value.abstract()
        except Exception:  # pragma: no cover - defensive
            return "Tensor"
        if abstract["kind"] == "dense":
            return f"Tensor{abstract.get('shape')} {abstract.get('dtype')}"
        if abstract["kind"] == "scan":
            return f"scan{abstract.get('fields')}"
        return f"Tensor({abstract['kind']}) {abstract.get('fields')}"
    if isinstance(value, Lazy):
        return "Lazy (pending; .collect() to materialize)"
    if isinstance(value, Operator):
        return f"Operator{list(value.params)}"
    text = repr(value)
    return text if len(text) <= 120 else text[:117] + "..."


def _display_value(value) -> str:
    from ..core.operator import Operator
    from ..core.tensor import Lazy, Tensor

    if isinstance(value, Lazy):
        try:
            return _display_value(value.collect())
        except TensorLangError as exc:
            return _format_error(exc)
    if isinstance(value, Tensor):
        return _summary(value)
    if isinstance(value, Operator):
        return _summary(value)
    import numpy as _np
    import polars as _pl

    if isinstance(value, _np.ndarray):
        if value.size > 20:
            return (f"array(shape={list(value.shape)}, dtype={value.dtype}, "
                    f"head={value.flat[:3].tolist()} ...)")
        return repr(value)
    if isinstance(value, _pl.DataFrame):
        if value.height > 12:
            return f"{value.head(8)}\n... ({value.shape[0]} rows total)"
        return str(value)
    if isinstance(value, _pl.Series):
        if len(value) > 20:
            return f"series '{value.name}' len={len(value)} head={value.head(3).to_list()} ..."
        return str(value)
    text = repr(value)
    return text if len(text) <= 500 else text[:497] + "..."


def _format_error(exc: TensorLangError) -> str:
    from . import diagnose as _diagnose

    lines = [f"error[{exc.code}] ({exc.category}): {exc}"]
    for suggestion in _diagnose.suggest_for(exc)[:2]:
        lines.append(f"  hint: {suggestion}")
    return "\n".join(lines)


def run_repl() -> int:
    """Interactive loop (Ctrl-D exits, Ctrl-C cancels the buffer)."""
    import sys

    repl = Repl()
    try:
        import readline as _readline
    except ImportError:  # pragma: no cover - platform dependent
        _readline = None
    if _readline is not None:
        import os as _os

        history = _os.path.expanduser("~/.tensorlang_history")
        try:
            _readline.read_history_file(history)
        except OSError:
            pass
        _readline.set_history_length(500)
        import atexit as _atexit

        _atexit.register(_readline.write_history_file, history)
    print("TensorLang REPL — :help for commands, Ctrl-D to quit.")
    while True:
        prompt = ">>> " if not repl._buffer else "... "
        try:
            line = input(prompt)
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print()
            repl._buffer.clear()
            continue
        try:
            out = repl.push(line)
        except SystemExit as exit:
            return int(exit.code or 0)
        except KeyboardInterrupt:
            print()
            repl._buffer.clear()
            continue
        if out:
            print(out)
    return 0  # pragma: no cover - unreachable
