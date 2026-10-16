"""Native file-based ``.tl`` module loading (v0.8 M2).

A narrow extension of the existing frontend: ``import`` / ``from`` statements
inside ``.tl`` programs resolve sibling ``.tl`` files relative to the
*importing file* (never ``sys.path``, never CWD), compile them through the
existing frontend (lexer → parser with M1 effect validation → ``compile``),
execute them in isolated namespaces, and cache them per loader session.

Design rules:

- **Python wins**: this finder sits last on ``sys.meta_path`` and only
  serves names no Python finder resolves.  A sibling ``json.tl`` never
  shadows the stdlib from ``.tl`` code.
- **No packages**: dotted names map to relative paths
  (``operators.calibration`` → ``operators/calibration.tl``); parent
  directories import as inert namespace packages, ``__init__`` files are
  never executed, relative imports are rejected (``TL0314``).
- **Per-session cache**: one program execution uses one
  :class:`LoaderSession`; repeated imports reuse initialized modules and
  first-class operator objects.  Independent executions are isolated;
  numerical lazy-result caching is untouched and separate.
- **Fail closed**: static pre-scan rejects import cycles (``TL0312``)
  before executing anything; module code passes the exact M1 validation;
  failures never leave partially initialized modules in the cache.
"""
from __future__ import annotations

import ast
import importlib.abc
import importlib.util
import sys
import threading
import traceback
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

from ..core.errors import TLDeclarationError

_SESSION_STACK: list["LoaderSession"] = []
_STACK_LOCK = threading.Lock()


def _me() -> int:
    return threading.get_ident()


def _ownership_error() -> TLDeclarationError:
    return TLDeclarationError(
        "loader sessions cannot overlap across threads: an independent "
        ".tl execution is already running on another thread; run .tl "
        "programs sequentially or in separate processes",
        loc=None, code="TL0315", details={})


def active_session() -> "LoaderSession | None":
    """Innermost active loader session, if any ``.tl`` code is executing."""
    return _SESSION_STACK[-1] if _SESSION_STACK else None


class LoaderSession:
    """Per-execution ``.tl`` module cache and import context."""

    def __init__(self, base_dir: str) -> None:
        self.base_dir = base_dir
        # Owning thread: overlapping independent sessions from another
        # thread are rejected fail-fast (TL0315).  Same-thread nesting
        # joins the active session.
        self.owner: int = _me()
        # (Canonical path, module name) -> initialized module.  Same file
        # under different dotted spellings initializes independently,
        # exactly like CPython treats one file imported under two names.
        self.modules: dict[tuple[str, str], ModuleType] = {}
        # sys.modules names owned by this session -> module object.
        self.owned: dict[str, ModuleType] = {}
        # Conflicting bindings to sweep on close.
        self.trash: dict[str, ModuleType] = {}
        # sys.modules snapshot: everything added during the session that
        # is not a regular import (namespace-package parents, fresh
        # third-party imports) is dropped on close so later independent
        # sessions resolve deterministically.
        self._pre_modules: set[str] = set(sys.modules)
        # Canonical paths currently initializing (dynamic backstop).
        self.loading: list[str] = []
        # Import stack for diagnostics: (requester, lineno, requested).
        self.stack: list[tuple[str, int | None, str]] = []
        # Directories of the .tl modules currently executing (innermost last).
        self.exec_stack: list[str] = []

    def cached(self, path: str, name: str) -> ModuleType | None:
        return self.modules.get((path, name))

    def own(self, name: str, module: ModuleType) -> None:
        existing = sys.modules.get(name)
        if existing is not None and existing is not module:
            self.trash[name] = module
            raise TLDeclarationError(
                f"ambiguous module name '{name}': already bound to "
                f"{getattr(existing, '__file__', existing)}; cannot also "
                f"bind {getattr(module, '__file__', module)}",
                loc=None, code="TL0314", details={"name": name},
            )
        sys.modules[name] = module
        self.owned[name] = module

    def close(self) -> None:
        for name, module in list(self.owned.items()) + list(self.trash.items()):
            if sys.modules.get(name) is module:
                del sys.modules[name]
        # Drop namespace-package parents created for dotted `.tl` imports
        # (they point at this session's directories).  Regular third-party
        # imports made during the session are intentionally kept: unloading
        # them can break extension modules on re-import.
        for name in set(sys.modules) - self._pre_modules:
            module = sys.modules.get(name)
            if module is not None and _is_namespace_entry(module):
                try:
                    del sys.modules[name]
                except KeyError:
                    pass
        self.owned.clear()
        self.trash.clear()
        self.modules.clear()
        self.loading.clear()
        self.stack.clear()
        self.exec_stack.clear()


def _child_dir(name: str, anchors: list[str]) -> str | None:
    """Directory `name` under the first anchor holding it (no file check)."""
    for anchor in anchors:
        candidate = Path(anchor) / name
        try:
            if candidate.is_dir():
                return str(candidate)
        except OSError:
            continue
    return None


def _is_namespace_entry(module: ModuleType) -> bool:
    spec = getattr(module, "__spec__", None)
    return (spec is not None and spec.origin is None
            and getattr(spec, "submodule_search_locations", None) is not None)


@contextmanager
def loader_session(base_dir: str):
    """Push a fresh session; close it on exit."""
    with _STACK_LOCK:
        if _SESSION_STACK and _SESSION_STACK[-1].owner != _me():
            raise _ownership_error()
        sess = LoaderSession(base_dir)
        _SESSION_STACK.append(sess)
    try:
        yield sess
    finally:
        with _STACK_LOCK:
            if sess in _SESSION_STACK:
                _SESSION_STACK.remove(sess)
        sess.close()


@contextmanager
def session_for(base_dir: str):
    """Join the active session, or open (and close) a fresh one."""
    with _STACK_LOCK:
        sess = active_session()
        if sess is not None and sess.owner != _me():
            raise _ownership_error()
    if sess is not None:
        yield sess
    else:
        with loader_session(base_dir) as fresh:
            yield fresh


def ensure_owner(sess: "LoaderSession") -> None:
    """Reject joining a session owned by another thread (TL0315)."""
    if sess.owner != _me():
        raise _ownership_error()


def claim_session(sess: "LoaderSession") -> None:
    """Push a long-lived session (REPL) with ownership enforced."""
    with _STACK_LOCK:
        if sess.owner != _me():
            raise _ownership_error()
        if _SESSION_STACK and _SESSION_STACK[-1].owner != _me():
            raise _ownership_error()
        _SESSION_STACK.append(sess)


def release_session(sess: "LoaderSession") -> None:
    """Pop a claimed session (no-op unless it is top)."""
    with _STACK_LOCK:
        if _SESSION_STACK and _SESSION_STACK[-1] is sess:
            _SESSION_STACK.pop()


def entry_base_dir(filename: str) -> str:
    """Directory `.tl` imports in `filename` resolve against."""
    return str(Path(filename).absolute().parent)


def resolve_tl(fullname: str, anchors: list[str]) -> Path | None:
    """Map a dotted name to a sibling `.tl` file under the first anchor holding it."""
    rel = Path(*fullname.split(".")).with_suffix(".tl")
    for anchor in anchors:
        candidate = Path(anchor) / rel
        if candidate.is_file():
            return candidate
    return None


def _frames_below_machinery() -> list[traceback.FrameSummary]:
    """Stack frames excluding importlib internals and this module."""
    import os

    here = os.path.abspath(__file__)
    out = []
    for frame in traceback.extract_stack():
        name = frame.filename.replace("\\", "/")
        if "/importlib/" in name or name.startswith("<frozen "):
            continue
        try:
            if os.path.abspath(frame.filename) == here:
                continue
        except OSError:
            pass
        out.append(frame)
    return out


def _tl_requester(sess: LoaderSession) -> tuple[str, int | None] | None:
    """(requesting .tl file, import-statement lineno), or None.

    Only when the true Python requester — the innermost code below the
    import machinery that ran the import statement — is the innermost
    executing ``.tl`` module.  Imports issued by ``.py`` helpers or
    lazily by third-party backends return None so ordinary
    ``ModuleNotFoundError`` behavior is preserved.
    """
    top = getattr(sess, "_exec_top", None)
    if top is None:
        return None
    frames = _frames_below_machinery()
    if not frames:
        return None
    if _same_file(frames[-1].filename, top):
        return frames[-1].filename, frames[-1].lineno
    return None


def _same_file(a: str, b: str | None) -> bool:
    if b is None:
        return False
    try:
        return Path(a).absolute() == Path(b).absolute()
    except OSError:
        return a == b


def format_chain(stack: list[tuple[str, int | None, str]],
                 leaf: str) -> str:
    """Render an import chain: requester lines plus the leaf diagnostic."""
    lines = []
    for requester, lineno, requested in stack:
        at = f":{lineno}" if lineno is not None else ""
        lines.append(f"{requester}{at}\n  imports {requested}")
    lines.append(leaf)
    return "\n".join(lines)


# -- static import graph (pre-scan + CLI inspection) ---------------------------

def imported_names(tree: ast.AST) -> list[tuple[str, int]]:
    """Absolute import requests: (dotted base, lineno).

    ``import a.b`` → ``a.b``; ``from a import b, c`` → ``a`` plus
    ``a.b``/``a.c`` candidates (a submodule file may satisfy the latter).
    Relative imports (level > 0) yield ``("", lineno)`` markers for the
    caller to reject as ``TL0314`` candidates.
    """
    out: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append((alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                out.append(("", node.lineno))
                continue
            module = node.module or ""
            if module:
                out.append((module, node.lineno))
            for alias in node.names:
                if alias.name != "*" and module:
                    out.append((f"{module}.{alias.name}", node.lineno))
    return out


def _resolve_request(name: str, anchor: str):
    """Resolve one import request like the runtime loader would.

    Returns ``(kind, target)``: ``"python"`` when normal Python resolution
    (anchored, side-effect-free lookup) wins, ``"tl"`` with the `.tl`
    target file, or ``(None, None)`` when neither resolves here (the run
    path reports it when reached).
    """
    from importlib.machinery import PathFinder

    parts = name.split(".")
    leaf_anchor = anchor
    if len(parts) > 1:
        cur = Path(anchor)
        for part in parts[:-1]:
            cur = cur / part
            if not cur.is_dir():
                return None, None
        leaf_anchor = str(cur)
    try:
        if PathFinder.find_spec(parts[-1], [leaf_anchor]) is not None:
            return "python", None
    except Exception:
        pass
    target = resolve_tl(parts[-1], [leaf_anchor])
    if target is not None:
        return "tl", target
    return None, None


def _parse_tl_file(path: Path) -> ast.AST | None:
    from . import lexer as _lexer

    try:
        return ast.parse(_lexer.translate(path.read_text(encoding="utf-8")),
                         filename=str(path))
    except (SyntaxError, ValueError, OSError):
        return None


def plan_cycle(tree: ast.AST, base_dir: str,
               entry_name: str = "<entry>"):
    """Static cycle pre-scan from an entry tree.

    Resolves like the runtime loader (file-relative; Python-first
    precedence via side-effect-free lookup) without executing anything.
    Returns the cycle as requester legs
    ``[(requester, lineno, requested)]``; ``[]`` when acyclic.
    Unresolvable names are ignored here; the run path reports them
    when reached.
    """
    state: dict[str, int] = {}
    order: list[str] = []
    requests: dict[str, list[tuple[str, int, str | None]]] = {}
    legs: list[tuple[str, int | None, str]] = []

    def visit_file(path: Path) -> None:
        key = str(path.resolve())
        anchor = str(path.resolve().parent)
        mark = state.get(key, 0)
        if mark == 1:
            start = order.index(key)
            loop = order[start:] + [key]
            for i in range(len(loop) - 1):
                hit = next((r for r in requests.get(loop[i], [])
                            if r[2] == loop[i + 1]), None)
                if hit is not None:
                    legs.append((loop[i], hit[1], hit[0]))
                else:  # pragma: no cover - defensive
                    legs.append((loop[i], None, loop[i + 1]))
            return
        if mark == 2 or legs:
            return
        state[key] = 1
        order.append(key)
        rows: list[tuple[str, int, str | None]] = []
        dep_tree = _parse_tl_file(path)
        for name, lineno in imported_names(dep_tree) if dep_tree else []:
            target = None
            if name:
                kind, found = _resolve_request(name, anchor)
                target = found if kind == "tl" else None
            rows.append((name, lineno,
                         str(target.resolve()) if target else None))
        requests[key] = rows
        for _name, _lineno, resolved in rows:
            if resolved is not None and not legs:
                visit_file(Path(resolved))
        order.pop()
        state[key] = 2

    for name, lineno in imported_names(tree):
        if name and not legs:
            kind, target = _resolve_request(name, base_dir)
            if kind != "tl" or target is None:
                continue
            visit_file(target)
            if legs and not _same_file(legs[0][0], entry_name):
                legs.insert(0, (entry_name, lineno, name))
    return legs


def describe_cycle(legs: list[tuple[str, int | None, str]]) -> str:
    """Render cycle legs in the standard import-chain format."""
    return format_chain(list(legs), "TL0312: circular TensorLang import")


def transitive_effects(entry: Path, base_dir: str):
    """Static effect preflight over the entry file and its `.tl` graph.

    Returns ``(effects, cycle, unsupported)``: ``effects`` is a list of
    ``(file, dotted, lineno)`` from :func:`effects.find_effects` across
    all reachable files; ``cycle`` as in :func:`plan_cycle`;
    ``unsupported`` lists ``(file, lineno)`` of relative imports.
    Nothing is executed.
    """
    from . import effects as _effects

    effects: list[tuple[str, str, int]] = []
    unsupported: list[tuple[str, int]] = []
    seen: set[str] = set()

    def collect(path: Path) -> None:
        key = str(path.resolve())
        if key in seen:
            return
        seen.add(key)
        anchor = str(path.resolve().parent)
        tree = _parse_tl_file(path)
        if tree is None:
            return
        for dotted, lineno in _effects.find_effects(tree):
            effects.append((key, dotted, lineno))
        for name, lineno in imported_names(tree):
            if not name:
                unsupported.append((key, lineno))
                continue
            kind, target = _resolve_request(name, anchor)
            if kind == "tl" and target is not None:
                collect(target)

    collect(entry)
    entry_tree = _parse_tl_file(entry)
    cycle = plan_cycle(entry_tree, base_dir) if entry_tree is not None else []
    return effects, cycle, unsupported


# -- meta-path finder ----------------------------------------------------------

_PASSTHROUGH_CODES = ("TL0312", "TL0314")


class _TlFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Last-resort finder serving sibling `.tl` files to `.tl` requesters."""

    def find_spec(self, fullname: str, path=None, target=None):
        sess = active_session()
        if sess is None:
            return None
        if sess.owner != _me():
            raise _ownership_error()
        requester = _tl_requester(sess)
        if requester is None:
            return None
        req_file, lineno = requester
        if path:
            # Submodule request: resolve the trailing component under the
            # parent package path (namespace-package parents welcome).
            candidate = resolve_tl(fullname.split(".")[-1],
                                   [str(p) for p in path])
            anchors = [str(p) for p in path]
            pkgdir = _child_dir(fullname.split(".")[-1], anchors)
        else:
            if not sess.exec_stack:
                return None
            anchors = [sess.exec_stack[-1]]
            candidate = resolve_tl(fullname, anchors)
            pkgdir = _child_dir(fullname.split(".")[0], anchors)
        if candidate is None and pkgdir is None:
            sess.stack.append((req_file, lineno, fullname))
            chain = format_chain(
                sess.stack,
                f"TL0310: no TensorLang module '{fullname}' "
                f"(no {fullname.replace('.', '/')}.tl under {anchors[0]}; "
                f"Python imports take precedence and also failed)")
            sess.stack.pop()
            raise TLDeclarationError(chain, loc=None,
                                     code="TL0310",
                                     details={"module": fullname})
        if pkgdir is not None and candidate is None:
            # Bare directory: synthesize an inert package shell so dotted
            # `.tl` imports work without sys.path luck.  `__init__` files
            # are never executed.  (A sibling `.tl` file takes precedence.)
            spec = importlib.util.spec_from_loader(fullname, self)
            assert spec is not None
            spec.submodule_search_locations = [pkgdir]  # type: ignore[attr-defined]
            spec.loader_state = {"pkgdir": pkgdir}  # type: ignore[attr-defined]
            spec.origin = pkgdir  # type: ignore[attr-defined]
            return spec
        canonical = str(candidate.resolve())
        if canonical in sess.loading:
            chain = format_chain(
                sess.stack + [(req_file, lineno, fullname)],
                f"TL0312: circular TensorLang import of '{fullname}' "
                f"({candidate})")
            raise TLDeclarationError(chain, loc=None,
                                     code="TL0312",
                                     details={"module": fullname})
        # Cache identity is (path, name): same file under a different
        # spelling initializes an independent module (CPython semantics).
        cached = sess.cached(canonical, fullname)
        spec = importlib.util.spec_from_loader(fullname, self)
        assert spec is not None
        spec.origin = canonical  # type: ignore[attr-defined]
        if cached is not None:
            sys.modules[fullname] = cached
            spec.loader_state = {"path": canonical,  # type: ignore[attr-defined]
                                 "cached": True}
            return spec
        sess.stack.append((req_file, lineno, fullname))
        try:
            spec.loader_state = {"path": canonical}  # type: ignore[attr-defined]
            return spec
        finally:
            sess.stack.pop()

    def create_module(self, spec):
        state = getattr(spec, "loader_state", None) or {}
        if state.get("cached"):
            sess = active_session()
            if sess is not None:
                mod = sess.cached(state["path"], spec.name)
                if mod is not None:
                    return mod
        return None

    def exec_module(self, module: ModuleType) -> None:
        from . import compiler as _compiler

        sess = active_session()
        assert sess is not None
        if sess.owner != _me():
            raise _ownership_error()
        state = module.__spec__.loader_state or {}  # type: ignore[union-attr]
        if not state.get("path"):
            # Synthesized package shell (no code to execute).
            if state.get("pkgdir"):
                sess.own(module.__name__, module)
            return
        canonical: str = state["path"]
        if (canonical, module.__name__) in sess.modules:
            return
        if canonical in sess.loading:
            raise TLDeclarationError(
                format_chain(sess.stack,
                             f"TL0312: circular TensorLang import of "
                             f"'{module.__name__}' ({canonical})"),
                loc=None, code="TL0312",
                details={"module": module.__name__})
        sess.own(module.__name__, module)
        sess.loading.append(canonical)
        module_dir = str(Path(canonical).parent)
        sess.exec_stack.append(module_dir)
        sys.path.insert(0, module_dir)
        previous_top = sess._exec_top if hasattr(sess, "_exec_top") else None
        sess._exec_top = canonical  # type: ignore[attr-defined]
        previous_file = _compiler.CURRENT_FILENAME
        _compiler.CURRENT_FILENAME = canonical
        try:
            source = Path(canonical).read_text(encoding="utf-8")
            try:
                code = _compiler.compile_source(source, canonical)
            except Exception as exc:
                raise _chain_failure(sess, module.__name__, canonical,
                                     exc) from exc
            namespace = module.__dict__
            namespace["__file__"] = canonical
            namespace.update(_compiler.base_namespace())
            try:
                exec(code, namespace)
            except ImportError as exc:
                raise _convert_import_error(sess, module.__name__,
                                            canonical, exc) from exc
            except Exception as exc:
                raise _chain_failure(sess, module.__name__, canonical,
                                     exc) from exc
        finally:
            _compiler.CURRENT_FILENAME = previous_file
            sess.exec_stack.pop()
            try:
                sys.path.remove(module_dir)
            except ValueError:
                pass
            if previous_top is None:
                try:
                    delattr(sess, "_exec_top")
                except AttributeError:
                    pass
            else:
                sess._exec_top = previous_top  # type: ignore[attr-defined]
            if canonical in sess.loading:
                sess.loading.remove(canonical)
        sess.modules[(canonical, module.__name__)] = module


def _chain_failure(sess: LoaderSession, name: str, canonical: str,
                   exc: BaseException) -> TLDeclarationError:
    from ..core.errors import TensorLangError

    if isinstance(exc, TLDeclarationError) and exc.code in _PASSTHROUGH_CODES:
        return exc
    if isinstance(exc, TensorLangError):
        leaf = f"{canonical}: {exc}"
    else:
        leaf = f"{canonical}: {type(exc).__name__}: {exc}"
    return TLDeclarationError(
        format_chain(sess.stack, leaf), loc=None, code="TL0313",
        details={"module": name,
                 "cause": getattr(exc, "code", type(exc).__name__)})


def _convert_import_error(sess: LoaderSession, name: str, canonical: str,
                          exc: ImportError) -> BaseException:
    import re

    text = str(exc)
    if "attempted relative import" in text:
        return TLDeclarationError(
            format_chain(sess.stack,
                         f"{canonical}: {text} (relative .tl imports are "
                         f"unsupported; use top-level absolute imports)"),
            loc=None, code="TL0314", details={"module": name})
    converted = _missing_symbol(sess, text)
    if converted is not None:
        requester, lineno = _importer_frame(canonical, exc.__traceback__)
        return TLDeclarationError(
            format_chain(sess.stack + ([(requester, lineno, converted[1])]
                                       if requester else []),
                         f"{converted[1]}: TL0311: module '{converted[1]}' "
                         f"has no symbol '{converted[0]}'"),
            loc=None, code="TL0311",
            details={"module": converted[1], "symbol": converted[0]})
    return exc


def _missing_symbol(sess: LoaderSession,
                    text: str) -> tuple[str, str] | None:
    """(missing, owner) when `text` reports a missing symbol of a loaded module."""
    import re

    match = re.search(r"cannot import name '([\w.]+)' from '([\w.]+)'", text)
    if match and match.group(2) in {m.__name__ for m in sess.modules.values()}:
        return match.group(1), match.group(2)
    return None


def _importer_frame(canonical: str, tb) -> tuple[str | None, int | None]:
    """Innermost error-traceback frame inside the failing `.tl` file."""
    for frame in reversed(traceback.extract_tb(tb)):
        if _same_file(frame.filename, canonical):
            return frame.filename, frame.lineno
    return None, None


def convert_entry_import_error(sess: LoaderSession, filename: str,
                               exc: ImportError) -> BaseException:
    """Convert entry-level import failures to coded diagnostics."""
    text = str(exc)
    if "attempted relative import" in text:
        return TLDeclarationError(
            f"{filename}: {text} (relative .tl imports are unsupported; "
            f"use top-level absolute imports)",
            loc=filename, code="TL0314", details={})
    converted = _missing_symbol(sess, text)
    if converted is not None:
        requester, lineno = _importer_frame(filename, exc.__traceback__)
        at = f":{lineno}" if lineno is not None else ""
        return TLDeclarationError(
            f"{requester or filename}{at}\n  imports {converted[1]}\n"
            f"{converted[1]}: TL0311: module '{converted[1]}' has no "
            f"symbol '{converted[0]}'",
            loc=None, code="TL0311",
            details={"module": converted[1], "symbol": converted[0]})
    return exc


_INSTALLED = False


def ensure_installed() -> None:
    """Append the `.tl` finder last on ``sys.meta_path`` (once)."""
    global _INSTALLED
    if _INSTALLED:
        return
    sys.meta_path.append(_TlFinder())
    _INSTALLED = True


ensure_installed()
