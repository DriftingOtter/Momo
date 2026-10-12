"""REPL tests: drive Repl.push() directly (no pty/stdin needed)."""
import pytest

from tensorlang.frontend.repl import Repl


def test_declare_apply_display():
    r = Repl()
    assert r.push("from tensorlang import T") == ""
    assert r.push("v := T([1, 2, 3])") == ""
    assert r.push("v * 2") == "array([2, 4, 6])"
    assert r.push("f := {X} ** 2") == ""
    assert r.push("f(v)") == "array([1, 4, 9])"


def test_statements_stay_silent_and_persist():
    r = Repl()
    r.push("from tensorlang import T")
    r.push("a = T([5])")
    assert r.push("a") != ""
    assert "Tensor" in r.push("a")


def test_multiline_blocks_buffer():
    r = Repl()
    assert r.push("from tensorlang import T") == ""
    assert r.push("def g(a):") is None
    assert r.push("    return a + 1") == ""
    assert r.push("g(41)") == "42"
    assert r.push("total = 0") == ""
    assert r.push("for i in [1, 2]:") is None
    assert r.push("    total = total + i") == ""
    assert r.push("total") == "3"


def test_errors_carry_codes_and_hints():
    r = Repl()
    r.push("from tensorlang import T")
    out = r.push("(T([1, 2]) + T([1, 2, 3])).collect()")
    assert "TL0501" in out
    assert "hint:" in out
    out = r.push("nosuchname_xyz")
    assert "PY0000" in out and "NameError" in out
    out = r.push("1 +")
    assert out != "" and "TL0101" in out  # genuine syntax error surfaces


def test_commands():
    r = Repl()
    r.push("from tensorlang import T")
    r.push("v := T([1])")
    assert "v :" in r.push(":vars")
    assert "logical plan" in r.push(":explain v").lower() or \
        "Literal" in r.push(":explain v")
    assert "not defined" not in r.push(":explain v")
    assert "Tensor" in r.push(":graph v") or "Literal" in r.push(":graph v")
    assert r.push(":bogus") != ""
    assert "cleared" in r.push(":reset")
    assert "(no bindings)" in r.push(":vars")
    with pytest.raises(SystemExit):
        r.push(":quit")


def test_unknown_name_suggests():
    r = Repl()
    r.push("from tensorlang import T")
    r.push("vector := T([1])")
    out = r.push(":explain vectro")
    assert "vector" in out


def test_side_effects_execute_once():
    r = Repl()
    r.push("from tensorlang import T")
    r.push("calls = []")
    out = r.push("calls.append(1)")
    assert out == ""  # append returns None → silent, executed once
    assert r.push("calls") == "[1]"


def test_display_truncates_big_arrays():
    r = Repl()
    r.push("from tensorlang import T")
    r.push("big := T.arange(100)")
    out = r.push("big")
    assert "100" in out and len(out) < 200
