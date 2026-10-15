"""M3 tests: `extra=` namespace-injection precedence.

- `:=` over an injected name → structured `TL0203` (no silent shadow).
- `=` over an injected name → module wins (explicit rebinding intent).
- Dunder/interpreter names cannot be injected (`TL0203`).
- Injected modules never inherit `extra` (isolated namespaces).
- `extra` cannot smuggle foreign callables into trusted `:=` positions.
"""
import os

import pytest

from tensorlang import T, TLDeclarationError, TLScopeError, run_file, run_source


def run(src, **kw):
    return run_source(src, filename="inj.tl", **kw)


def test_declare_over_extra_TL0203_without_effect(tmp_path):
    sentinel = tmp_path / "fired.txt"
    src = (f"import os\nos.system('touch {sentinel}')\n"
           "DATA_DIR := 'mod'\n")
    with pytest.raises(TLScopeError) as exc_info:
        run(src, extra={"DATA_DIR": "/inj"})
    assert exc_info.value.code == "TL0203"
    assert "DATA_DIR" in str(exc_info.value)
    assert "inj.tl:" in str(exc_info.value)
    assert not sentinel.exists()


def test_assign_over_extra_module_wins():
    ns = run("DATA_DIR = '/mod'\n", extra={"DATA_DIR": "/inj"})
    assert ns["DATA_DIR"] == "/mod"


def test_extra_visible_without_collision():
    ns = run("from tensorlang import T\nx = DATA_DIR\n",
             extra={"DATA_DIR": "/inj"})
    assert ns["x"] == "/inj"


def test_dunder_injection_rejected():
    with pytest.raises(TLScopeError) as exc_info:
        run("x = 1\n", extra={"__tl_declare__": 1})
    assert exc_info.value.code == "TL0203"
    with pytest.raises(TLScopeError) as exc_info:
        run("x = 1\n", extra={"__file__": "evil"})
    assert exc_info.value.code == "TL0203"


def test_injected_names_do_not_leak_into_imports(tmp_path):
    from tensorlang import run_source as _rs

    (tmp_path / "dep.tl").write_text("v := 1\n")
    main = tmp_path / "main.tl"
    main.write_text("import dep\n"
                    "x = DATA_DIR\n")
    ns = _rs("import dep\nx = DATA_DIR\n", filename=str(main),
             extra={"DATA_DIR": "/inj"})
    assert ns["x"] == "/inj"
    with pytest.raises(KeyError):
        _ = ns["dep"].__dict__["DATA_DIR"]


def test_smuggle_foreign_callable_TL0305():
    def grab():
        return 1

    with pytest.raises(TLDeclarationError) as exc_info:
        run("x := grab()\n", extra={"grab": grab})
    assert exc_info.value.code == "TL0305"


def test_smuggle_operator_valued_extra_TL0305():
    from tensorlang import P

    op = P("X") + 1
    with pytest.raises(TLDeclarationError) as exc_info:
        run("from tensorlang import T\nd := T([1.0])\nr := f(d)\n",
            extra={"f": op})
    assert exc_info.value.code == "TL0305"


def test_smuggle_aliased_extra_TL0305():
    def grab():
        return 1

    with pytest.raises(TLDeclarationError) as exc_info:
        run("h = grab\nx := h()\n", extra={"grab": grab})
    assert exc_info.value.code == "TL0305"


def test_trusted_extra_still_declarative():
    ns = run("x := T([1, 2])\n", extra={"T": T})
    assert ns["x"].collect().tolist() == [1, 2]


def test_runtime_use_of_extra_unaffected():
    def grab():
        return 42

    ns = run("x = grab()\n", extra={"grab": grab})
    assert ns["x"] == 42


def test_diagnose_conflict_hint():
    from tensorlang.frontend import diagnose as _diagnose

    report = _diagnose.diagnose_source(
        "DATA_DIR := 'mod'\n", filename="d.tl")
    # No extra here: plain declaration succeeds; conflict needs injection.
    assert report["ok"] is True
    with pytest.raises(TLScopeError):
        run("DATA_DIR := 'mod'\n", extra={"DATA_DIR": "/inj"})


def test_repl_rebinding_unchanged():
    from tensorlang.frontend.repl import Repl

    r = Repl()
    assert r.push("v := 1") == ""
    assert r.push("v := 2") == ""  # REPL keeps per-push semantics
    assert r.push("v") == "2"
    _ = os  # silence lint
