"""Lexical scopes for ``:=`` declarations.

Each ``Module``, ``FunctionDef``/``AsyncFunctionDef`` and ``ClassDef``
opens a new scope.  Control-flow blocks (``for``/``if``/``while``/``with``)
share their enclosing scope, matching Python name-binding rules.
Declaring the same name twice in one scope is an error; shadowing an
outer scope is allowed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..core.errors import TLScopeError


@dataclass
class Scope:
    kind: str
    name: str
    declared: set[str] = field(default_factory=set)


class ScopeTable:
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.stack: list[Scope] = [Scope("module", "<module>")]

    def push(self, kind: str, name: str) -> None:
        self.stack.append(Scope(kind, name))

    def pop(self) -> None:
        self.stack.pop()

    def declare(self, name: str, lineno: int) -> None:
        current = self.stack[-1]
        if name in current.declared:
            where = "module scope" if len(self.stack) == 1 else (
                f"{current.kind} '{current.name}'"
            )
            raise TLScopeError(
                f"duplicate declaration '{name}': '{name}' is already "
                f"declared in the same {where}; use '=' to reassign",
                loc=f"{self.filename}:{lineno}",
                code="TL0201",
            )
        current.declared.add(name)
