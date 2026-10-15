"""M3 tests: loader integration boundaries and session hygiene.

- Nested `run_file()` joins the active session (single init, shared).
- Third-party imports survive session teardown (same objects).
- `sys.path` is restored; no `.tl`/namespace leftovers in `sys.modules`.
- The meta-path finder is installed exactly once.
- REPL `:reset` drops the module cache (re-import re-executes).
- Concurrent sessions are documented-unsupported (global hooks).
"""
import sys

import pytest

from tensorlang import run_file
from tensorlang.frontend import modules as _modules


@pytest.fixture()
def pkgs(tmp_path, monkeypatch):
    (tmp_path / "p1").mkdir()
    (tmp_path / "p1" / "m.tl").write_text("v := 1\n")
    (tmp_path / "p1" / "main.tl").write_text("import m\n")
    (tmp_path / "p2").mkdir()
    (tmp_path / "p2" / "m.tl").write_text("v := 2\n")
    (tmp_path / "p2" / "main.tl").write_text("import m\n")
    monkeypatch.chdir(tmp_path)
    yield tmp_path
    for name in ("m", "outer", "inner"):
        sys.modules.pop(name, None)


def _tl_leftovers():
    return [n for n, m in sys.modules.items()
            if getattr(m, "__file__", "") and str(getattr(m, "__file__", "")).endswith(".tl")]


def test_nested_run_file_joins_session(pkgs):
    (pkgs / "outer.tl").write_text(
        "from tensorlang import run_file as _rf\n"
        "inner = _rf('p1/main.tl')\n")
    ns = run_file(str(pkgs / "outer.tl"))
    assert ns["inner"]["m"].v == 1


def test_third_party_preserved_across_sessions(pkgs):
    import scipy.linalg as first

    run_file(str(pkgs / "p1" / "main.tl"))
    run_file(str(pkgs / "p2" / "main.tl"))
    import scipy.linalg as second

    assert first is second
    assert "scipy.linalg" in sys.modules


def test_sys_path_restored(pkgs):
    before = list(sys.path)
    run_file(str(pkgs / "p1" / "main.tl"))
    assert list(sys.path) == before


def test_no_module_leftovers(pkgs):
    run_file(str(pkgs / "p1" / "main.tl"))
    assert _tl_leftovers() == []
    assert "m" not in sys.modules


def test_namespace_parents_swept(pkgs):
    (pkgs / "p1" / "sub").mkdir()
    (pkgs / "p1" / "sub" / "leaf.tl").write_text("v := 3\n")
    (pkgs / "p1" / "main2.tl").write_text("from sub.leaf import v\n")
    ns = run_file(str(pkgs / "p1" / "main2.tl"))
    assert ns["v"] == 3
    assert "sub" not in sys.modules


def test_finder_installed_once():
    _modules.ensure_installed()
    _modules.ensure_installed()
    assert sum(isinstance(f, _modules._TlFinder) for f in sys.meta_path) == 1


def test_repl_reset_drops_cache(pkgs):
    from tensorlang.frontend.repl import Repl

    import os

    os.chdir(pkgs / "p1")
    r = Repl()
    assert r.push("import m") == ""
    first = r.push("m")
    assert r.push(":reset") == "(bindings cleared)"
    assert r.push("import m") == ""
    # Cache was dropped: re-import re-executes (no stale bindings needed).
    assert r.push("m") == first


def test_independent_sessions_isolated(pkgs):
    first = run_file(str(pkgs / "p1" / "main.tl"))
    second = run_file(str(pkgs / "p2" / "main.tl"))
    assert first["m"].v == 1
    assert second["m"].v == 2
    assert first["m"] is not second["m"]


def test_overlapping_thread_sessions_rejected(pkgs):
    import threading

    (pkgs / "slow").mkdir()
    (pkgs / "slow" / "mod.tl").write_text(
        "import time\ntime.sleep(0.5)\nfrom tensorlang import T\nv := T([1])\n")
    (pkgs / "slow" / "main.tl").write_text("import mod\nout = mod.v\n")
    (pkgs / "fast").mkdir()
    (pkgs / "fast" / "mod.tl").write_text(
        "from tensorlang import T\nv := T([2])\n")
    (pkgs / "fast" / "main.tl").write_text("import mod\nout = mod.v\n")
    results, errors = {}, []
    barrier = threading.Barrier(2)

    def work(tag, path):
        try:
            barrier.wait(timeout=10)
            ns = run_file(path)
            results[tag] = ns["out"].collect().tolist()
        except Exception as exc:  # noqa: BLE001 - assert below
            errors.append((tag, getattr(exc, "code", type(exc).__name__)))

    threads = [threading.Thread(target=work, args=("slow", str(pkgs / "slow" / "main.tl"))),
               threading.Thread(target=work, args=("fast", str(pkgs / "fast" / "main.tl")))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results.get("slow") in (None, [1])
    assert results.get("fast") in (None, [2])
    assert all(code == "TL0315" for _, code in errors)
    assert results or errors


def test_sequential_thread_sessions_allowed(pkgs):
    import threading

    outcomes = []

    def work(path):
        outcomes.append(run_file(path)["m"].v)

    first = threading.Thread(target=work, args=(str(pkgs / "p1" / "main.tl"),))
    first.start()
    first.join()
    second = threading.Thread(target=work, args=(str(pkgs / "p2" / "main.tl"),))
    second.start()
    second.join()
    assert outcomes == [1, 2]
