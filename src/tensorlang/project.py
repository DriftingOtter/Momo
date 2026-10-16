"""``tl``: Cargo-inspired project manager over uv + TensorLang runtime.

``tl`` orchestrates project scaffolding, dependency management (via uv),
environment synchronization, and execution of the established
TensorLang compiler/runtime.  It is not a second compiler, resolver,
registry, or execution engine:

- project manifest: standard ``pyproject.toml`` (``[project]`` metadata
  and dependencies are the single source of truth) plus a small
  ``[tool.tensorlang]`` section (entry point, source directory).
- dependencies: ``uv add`` / ``uv remove`` / ``uv sync`` invoked as
  subprocess argument arrays; ``uv.lock`` is the reproducibility
  record (never parsed or rewritten by ``tl``).
- execution/testing/checking: the project's own synchronized
  environment running the installed TensorLang CLI/pytest.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

CONFIG_SECTION = "tensorlang"
TOOL_TABLE = "tool"
ENTRY_DEFAULT = "src/main.tl"

_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-]*\Z")
_PKG_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-\.]*(\[[A-Za-z0-9_,\- ]+\])?([<>=!~]=?[^;\s]+)?\Z")


class ProjectError(Exception):
    """Expected project-management failure with an actionable hint."""

    def __init__(self, message: str, hint: str | None = None,
                 code: int = 1) -> None:
        super().__init__(message)
        self.hint = hint
        self.code = code


# -- discovery and manifest -----------------------------------------------------

def find_root(start: str | Path) -> Path:
    """Nearest ancestor directory holding a TensorLang project manifest."""
    current = Path(start).absolute()
    searched: list[str] = []
    while True:
        manifest = current / "pyproject.toml"
        if manifest.is_file():
            try:
                data = tomllib.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, tomllib.TOMLDecodeError):
                data = {}
            tools = data.get(TOOL_TABLE, {})
            if isinstance(tools, dict) and CONFIG_SECTION in tools:
                return current
        searched.append(str(current))
        parent = current.parent
        if parent == current:
            raise ProjectError(
                "no TensorLang project found (no pyproject.toml with a "
                "[tool.tensorlang] section in this directory or parents)",
                hint=f"searched: {searched[0]}"
                     + (f" (+{len(searched) - 1} parents)" if len(searched) > 1 else "")
                     + "; run 'tl new NAME' or 'tl init'",
            )
        current = parent


def load_manifest(root: str | Path) -> dict:
    """Read and validate the project manifest."""
    root = Path(root)
    path = root / "pyproject.toml"
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ProjectError(f"missing manifest: {path}",
                           hint="run 'tl init' in the project directory")
    except tomllib.TOMLDecodeError as exc:
        raise ProjectError(f"invalid manifest {path}: {exc}",
                           hint="fix the TOML syntax and retry")
    except OSError as exc:
        raise ProjectError(f"cannot read manifest {path}: {exc}")
    project = data.get("project")
    if not isinstance(project, dict):
        raise ProjectError(f"manifest {path} has no [project] table",
                           hint="see 'tl doctor' and docs/PROJECTS.md")
    for field in ("name", "version"):
        if not project.get(field):
            raise ProjectError(
                f"manifest {path} is missing [project] {field!r}",
                hint="add the field; see docs/PROJECTS.md")
    tools = data.get(TOOL_TABLE, {})
    section = tools.get(CONFIG_SECTION) if isinstance(tools, dict) else None
    if not isinstance(section, dict):
        raise ProjectError(
            f"manifest {path} has no [tool.{CONFIG_SECTION}] table",
            hint="run 'tl init' or add the table manually")
    entry = section.get("entry", ENTRY_DEFAULT)
    if not isinstance(entry, str) or not entry:
        raise ProjectError(
            f"manifest {path} has an invalid [tool.{CONFIG_SECTION}] entry",
            hint=f"set entry to a .tl path, e.g. entry = {ENTRY_DEFAULT!r}")
    return {"root": root, "path": path, "project": project,
            "tl": section, "entry": entry,
            "source_dir": section.get("source-dir", "src")}


def entry_path(manifest: dict) -> Path:
    return manifest["root"] / manifest["entry"]


# -- uv subprocess ---------------------------------------------------------------

def find_uv() -> str:
    uv = shutil.which("uv")
    if uv is None:
        raise ProjectError(
            "the 'uv' executable was not found on PATH",
            hint="install uv (https://docs.astral.sh/uv/) and retry; "
                 "tl delegates dependency management to uv and never "
                 "falls back to pip or the global Python")
    return uv


def run_uv(args: list[str], root: Path, check: bool = True) -> subprocess.CompletedProcess:
    """Invoke uv with an argument array in the project directory."""
    uv = find_uv()
    try:
        proc = subprocess.run([uv, *args], cwd=str(root),
                              capture_output=True, text=True)
    except OSError as exc:
        raise ProjectError(f"cannot execute uv: {exc}",
                           hint="check the uv installation")
    if check and proc.returncode != 0:
        tail = (proc.stderr.strip() or proc.stdout.strip()).splitlines()
        detail = tail[-1] if tail else f"exit {proc.returncode}"
        raise ProjectError(f"uv {' '.join(args)} failed: {detail}",
                           hint="run with more context or check dependencies",
                           code=proc.returncode or 1)
    return proc


# -- environment --------------------------------------------------------------------

def project_python(root: Path) -> Path | None:
    candidate = root / ".venv" / "bin" / "python"
    if os.name == "nt":  # pragma: no cover - platform dependent
        candidate = root / ".venv" / "Scripts" / "python.exe"
    return candidate if candidate.is_file() else None


def _file_hash(path: Path) -> str | None:
    import hashlib

    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def sync_state_path(root: Path) -> Path:
    return root / ".venv" / "tl-sync-state.json"


def record_sync_state(root: Path) -> None:
    """Remember manifest/lockfile hashes after a successful sync."""
    import json as _json

    state = {
        "pyproject": _file_hash(root / "pyproject.toml"),
        "lockfile": _file_hash(root / "uv.lock"),
    }
    try:
        sync_state_path(root).write_text(
            _json.dumps(state, indent=2), encoding="utf-8")
    except OSError:
        pass  # state tracking is advisory; sync itself succeeded


def sync_freshness(root: Path) -> str:
    """Freshness of the environment without touching the network.

    Returns one of: "no-environment", "no-state" (synced by other
    means), "fresh", or "stale:<what-changed>".
    """
    import json as _json

    if project_python(root) is None:
        return "no-environment"
    try:
        recorded = _json.loads(sync_state_path(root).read_text(
            encoding="utf-8"))
    except (OSError, ValueError):
        return "no-state"
    stale: list[str] = []
    if recorded.get("pyproject") != _file_hash(root / "pyproject.toml"):
        stale.append("pyproject.toml")
    if recorded.get("lockfile") != _file_hash(root / "uv.lock"):
        stale.append("uv.lock")
    if not stale:
        return "fresh"
    return "stale:" + "+".join(stale)


def ensure_env(root: Path) -> None:
    """Create the project environment if absent (explicit `uv sync`)."""
    if project_python(root) is not None:
        return
    print("tl: creating project environment (uv sync) ...", flush=True)
    run_uv(["sync"], root)
    record_sync_state(root)


def warn_if_stale(root: Path) -> None:
    """Advise when the environment may not reflect the manifest."""
    state = sync_freshness(root)
    if state.startswith("stale:"):
        print(f"tl: warning: environment may be stale "
              f"({state[len('stale:'):]} changed since last 'tl sync'); "
              f"run 'tl sync' to refresh (or 'tl sync --locked' to verify)",
              file=sys.stderr, flush=True)


def uv_run(root: Path, *args: str) -> int:
    """Run a command in the project environment without implicit sync."""
    uv = find_uv()
    cmd = [uv, "run", "--no-sync", "--project", str(root),
           "python", *args]
    proc = subprocess.run(cmd, cwd=str(root))
    return proc.returncode


# -- scaffolding -----------------------------------------------------------------------

_TEMPLATE_MAIN = """# {name} — TensorLang application entry point.
from tensorlang import T

from operators.transforms import offset, scale

data := T([1.0, 2.0, 3.0, 4.0])
result = offset(scale(data))
print(result.collect())
"""

_TEMPLATE_TRANSFORMS = """# Reusable numerical operators (pure, declarative).
scale := {X} * 2.0
offset := {X} + 10.0
"""

_TEMPLATE_TEST = '''"""{name} application tests (ordinary pytest)."""
import os

import numpy as np

from tensorlang import T, run_file

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_operators():
    ns = run_file(os.path.join(APP, "src", "operators", "transforms.tl"))
    out = ns["offset"](ns["scale"](T([1.0, 2.0]))).collect()
    assert np.asarray(out).tolist() == [12.0, 14.0]


def test_main_result():
    ns = run_file(os.path.join(APP, "src", "main.tl"))
    assert np.asarray(ns["result"].collect()).tolist() == [12.0, 14.0, 16.0, 18.0]
'''

_TEMPLATE_PYPROJECT = """[project]
name = "{name}"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["tensorlang"]
{uv_sources}
[tool.tensorlang]
entry = "src/main.tl"
source-dir = "src"
"""

_TEMPLATE_README = """# {name}

A TensorLang application project (managed with `tl`).

```bash
tl sync        # create/synchronize the environment
tl run         # execute src/main.tl
tl check       # validate sources without executing
tl test        # run pytest suite
tl doctor      # project diagnostics
```
"""

_TEMPLATE_GITIGNORE = """.venv/
__pycache__/
*.pyc
out/
"""


def validate_name(name: str) -> None:
    if not _NAME_RE.match(name or ""):
        raise ProjectError(
            f"invalid project name {name!r}: use letters, digits, '-' or '_'",
            hint="example: tl new scientific-model")


def _write_new(path: Path, content: str) -> None:
    if path.exists():
        raise ProjectError(f"refusing to overwrite existing file: {path}",
                           hint="remove it or choose another project")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def scaffold(root: Path, name: str, tensorlang_source: str | None = None) -> None:
    """Create a project tree; never overwrites existing files."""
    validate_name(name)
    uv_sources = ""
    if tensorlang_source:
        uv_sources = ("[tool.uv.sources]\n"
                      f'tensorlang = {{ path = "{tensorlang_source}" }}\n')
    _write_new(root / "pyproject.toml",
               _TEMPLATE_PYPROJECT.format(name=name, uv_sources=uv_sources))
    _write_new(root / ".gitignore", _TEMPLATE_GITIGNORE)
    _write_new(root / "README.md", _TEMPLATE_README.format(name=name))
    _write_new(root / "src" / "main.tl",
               _TEMPLATE_MAIN.format(name=name))
    _write_new(root / "src" / "operators" / "transforms.tl",
               _TEMPLATE_TRANSFORMS)
    _write_new(root / "tests" / "test_application.py",
               _TEMPLATE_TEST.format(name=name))


# -- requirement parsing (minimal, no new dependency) --------------------------

def _check_requires_python(spec: str, version: tuple[int, ...]) -> tuple[bool, str]:
    """Evaluate a small subset of PEP 440 specifiers (>=, >, ==, ~=, comma AND)."""
    clauses = [c.strip() for c in spec.split(",") if c.strip()]
    for clause in clauses:
        match = re.match(r"(>=|>|==|~=|<=|<|!=)\s*([\d.]+)", clause)
        if not match:
            return False, f"unsupported requires-python clause {clause!r}"
        op, nums = match.group(1), tuple(int(x) for x in match.group(2).split("."))
        have = version[:len(nums)]
        ok = {"==": have == nums, "!=": have != nums,
              ">=": have >= nums, ">": have > nums,
              "<=": have <= nums, "<": have < nums,
              "~=": have >= nums}[op]
        if not ok:
            return False, f"requires {clause}, have {'.'.join(map(str, version[:3]))}"
    return True, ""


# -- commands --------------------------------------------------------------------------

def cmd_new(args: argparse.Namespace) -> int:
    raw = args.name.replace("\\", "/")
    if "/" in raw or raw in (".", "..") or raw.startswith("."):
        raise ProjectError(
            f"invalid project name {args.name!r}: 'tl new' takes a plain "
            f"name, not a path",
            hint="run it inside the parent directory, or use 'tl init' "
                 "inside an existing directory")
    validate_name(raw)
    target = (Path.cwd() / raw).absolute()
    if target.exists() and any(target.iterdir()):
        raise ProjectError(f"directory is not empty: {target}",
                           hint="run in an empty directory or use 'tl init'")
    target.mkdir(parents=True, exist_ok=True)
    scaffold(target, raw, args.tensorlang)
    print(f"created TensorLang project '{target.name}' in {target}")
    print("next: cd into it, then 'tl sync', 'tl run', 'tl test'")
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    root = Path.cwd()
    manifest = root / "pyproject.toml"
    if manifest.exists():
        raise ProjectError(f"manifest already exists: {manifest}",
                           hint="add a [tool.tensorlang] table manually; "
                                "'tl init' never rewrites existing files")
    validate_name(root.name)
    scaffold(root, root.name, args.tensorlang)
    print(f"initialized TensorLang project '{root.name}' in {root}")
    return 0


def _check_spec(spec: str) -> None:
    if not spec or not _PKG_RE.match(spec) or len(spec) > 200:
        raise ProjectError(
            f"invalid package specifier {spec!r}",
            hint="examples: tl add numpy / tl add 'scipy>=1.14'")


def cmd_add(args: argparse.Namespace) -> int:
    root = find_root(Path.cwd())
    for spec in args.packages:
        _check_spec(spec)
    cmd = ["add"]
    if args.dev:
        cmd.append("--dev")
    cmd.extend(args.packages)
    run_uv(cmd, root)
    record_sync_state(root)
    print(f"added {', '.join(args.packages)}"
          + (" (dev)" if args.dev else ""))
    return 0


def cmd_remove(args: argparse.Namespace) -> int:
    root = find_root(Path.cwd())
    for spec in args.packages:
        _check_spec(spec)
    cmd = ["remove"]
    if args.dev:
        cmd.append("--dev")
    cmd.extend(args.packages)
    run_uv(cmd, root)
    record_sync_state(root)
    print(f"removed {', '.join(args.packages)}")
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    root = find_root(Path.cwd())
    cmd = ["sync"]
    if args.locked:
        cmd.append("--locked")
    run_uv(cmd, root)
    record_sync_state(root)
    print("environment synchronized"
          + (" (locked)" if args.locked else ""))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    root = find_root(Path.cwd())
    manifest = load_manifest(root)
    entry = entry_path(manifest)
    if not entry.is_file():
        raise ProjectError(f"configured entry point does not exist: {entry}",
                           hint="fix [tool.tensorlang] entry or create the file")
    ensure_env(root)
    warn_if_stale(root)
    forwarded = [a for a in (args.args or []) if a != "--"]
    cmd = ["-m", "tensorlang", "run", str(entry), *forwarded]
    return uv_run(root, *cmd)


def cmd_check(args: argparse.Namespace) -> int:
    root = find_root(Path.cwd())
    manifest = load_manifest(root)
    entry = entry_path(manifest)
    if not entry.is_file():
        raise ProjectError(f"configured entry point does not exist: {entry}",
                           hint="fix [tool.tensorlang] entry or create the file")
    ensure_env(root)
    warn_if_stale(root)
    return uv_run(root, "-m", "tensorlang", "check", str(entry))


def cmd_test(args: argparse.Namespace) -> int:
    root = find_root(Path.cwd())
    manifest = load_manifest(root)
    ensure_env(root)
    warn_if_stale(root)
    probe = subprocess.run(
        [find_uv(), "run", "--no-sync", "--project", str(root),
         "python", "-c", "import pytest"],
        cwd=str(root), capture_output=True, text=True)
    if probe.returncode != 0:
        raise ProjectError("pytest is not installed in the project environment",
                           hint="run 'tl add --dev pytest' then 'tl sync'")
    tests = root / "tests"
    if not tests.is_dir():
        raise ProjectError(f"no tests directory: {tests}",
                           hint="create tests/ with pytest test files")
    cmd = ["-m", "pytest", "tests", "-q", *args.args]
    return uv_run(root, *cmd)


def cmd_doctor(args: argparse.Namespace) -> int:
    del args
    problems: list[str] = []
    try:
        root = find_root(Path.cwd())
        print(f"project: {root}")
    except ProjectError as exc:
        print(f"project: MISSING ({exc})")
        if exc.hint:
            print(f"  hint: {exc.hint}")
        return 1
    try:
        manifest = load_manifest(root)
        print(f"manifest: {manifest['path']} ok "
              f"(project {manifest['project']['name']}-"
              f"{manifest['project']['version']})")
    except ProjectError as exc:
        print(f"manifest: INVALID ({exc})")
        if exc.hint:
            print(f"  hint: {exc.hint}")
        return 1
    entry = entry_path(manifest)
    print(f"entry: {entry} {'ok' if entry.is_file() else 'MISSING'}")
    if not entry.is_file():
        problems.append("entry")
    try:
        uv = find_uv()
        proc = subprocess.run([uv, "--version"], capture_output=True, text=True)
        print(f"uv: {uv} ({proc.stdout.strip() or 'version unknown'})")
    except ProjectError as exc:
        print(f"uv: MISSING ({exc})")
        if exc.hint:
            print(f"  hint: {exc.hint}")
        problems.append("uv")
    py = project_python(root)
    if py is None:
        print("environment: missing .venv (run 'tl sync')")
        problems.append("env")
    else:
        print(f"environment: {py}")
        out = subprocess.run([str(py), "--version"],
                             capture_output=True, text=True)
        ver = (out.stdout or out.stderr).strip().split()[-1]
        print(f"python: {ver}")
        req = manifest["project"].get("requires-python")
        if req:
            try:
                nums = tuple(int(x) for x in ver.split(".")[:3])
            except ValueError:
                nums = (0,)
            ok, detail = _check_requires_python(req, nums)
            print(f"requires-python {req}: {'ok' if ok else 'MISMATCH (' + detail + ')'}")
            if not ok:
                problems.append("python")
        tl = subprocess.run(
            [str(py), "-c", "import tensorlang; print(tensorlang.__version__)"],
            capture_output=True, text=True)
        if tl.returncode == 0:
            print(f"tensorlang: {tl.stdout.strip()} (project environment)")
        else:
            print("tensorlang: NOT INSTALLED in project environment")
            problems.append("tensorlang")
    lock = root / "uv.lock"
    print(f"lockfile: {'present' if lock.is_file() else 'absent (run tl sync)'}")
    state = sync_freshness(root)
    if state == "fresh":
        print("freshness: synchronized (manifest and lockfile unchanged "
              "since last sync; installed package contents are not "
              "independently verified)")
    elif state == "no-state":
        print("freshness: unknown (no tl sync record; run 'tl sync')")
    elif state == "no-environment":
        print("freshness: no environment (run 'tl sync')")
    else:
        print(f"freshness: STALE ({state[len('stale:'):]} changed since last "
              f"'tl sync'; run 'tl sync' to refresh)")
        problems.append("stale")
    if problems:
        print(f"doctor: {len(problems)} problem(s): {', '.join(problems)}")
        return 1
    print("doctor: project looks healthy")
    return 0


# -- entry point --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tl", description="TensorLang project manager (uv-backed).")
    sub = parser.add_subparsers(dest="command", required=True)

    new = sub.add_parser("new", help="create a new TensorLang project")
    new.add_argument("name", help="project name (also the directory name)")
    new.add_argument("--tensorlang",
                     help="local TensorLang source checkout or wheel for "
                          "development (recorded via [tool.uv.sources])")

    init = sub.add_parser("init", help="configure the current directory")
    init.add_argument("--tensorlang",
                      help="local TensorLang source checkout or wheel for "
                           "development (recorded via [tool.uv.sources])")

    add = sub.add_parser("add", help="add dependencies via uv")
    add.add_argument("packages", nargs="+", help="e.g. numpy 'scipy>=1.14'")
    add.add_argument("--dev", action="store_true",
                     help="development dependency")

    remove = sub.add_parser("remove", help="remove dependencies via uv")
    remove.add_argument("packages", nargs="+")
    remove.add_argument("--dev", action="store_true",
                        help="development dependency")

    sync = sub.add_parser("sync", help="synchronize the project environment")
    sync.add_argument("--locked", action="store_true",
                      help="fail when the lockfile is missing or stale")

    run = sub.add_parser("run", help="execute the project entry point")
    run.add_argument("args", nargs=argparse.REMAINDER,
                     help="optional arguments forwarded to the application "
                          "after --")

    check = sub.add_parser("check", help="validate sources without executing")

    test = sub.add_parser("test", help="run pytest inside the environment")
    test.add_argument("args", nargs=argparse.REMAINDER,
                      help="optional arguments forwarded to pytest")

    sub.add_parser("doctor", help="project diagnostics")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    commands = {
        "new": cmd_new, "init": cmd_init, "add": cmd_add,
        "remove": cmd_remove, "sync": cmd_sync, "run": cmd_run,
        "check": cmd_check, "test": cmd_test, "doctor": cmd_doctor,
    }
    try:
        return commands[args.command](args)
    except ProjectError as exc:
        print(f"error: {exc}", file=sys.stderr)
        if exc.hint:
            print(f"  hint: {exc.hint}", file=sys.stderr)
        return exc.code
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
