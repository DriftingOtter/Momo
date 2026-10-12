"""Phase 1 tests: the `.tl` language frontend.

Covers `:=` declarations, `{X}` parameters, lexical scopes, ordinary
Python control flow, and source-level diagnostics.
"""
import numpy as np
import pytest

from tensorlang import (
    P,
    T,
    TLDeclarationError,
    TLScopeError,
    TLSyntaxError,
    run_source,
)
from tensorlang.frontend import compiler as _compiler


def run(src):
    return _compiler.run_source(src, filename="test.tl")


def test_declare_tensor_literal():
    ns = run("from tensorlang import T\nscalar := T(5)\n")
    assert ns["scalar"].collect() == 5


def test_declare_operator():
    ns = run("from tensorlang import T\nf := {X}**2 + 1\n")
    assert ns["f"](T(4)).collect() == 17


def test_runtime_application_rejected_under_declare():
    with pytest.raises(TLDeclarationError):
        run("from tensorlang import T\nf := {X} + 1\ndata := T([1])\nresult := f(data)\n")


def test_runtime_assignment_allows_application():
    ns = run("from tensorlang import T\nf := {X} + 1\ndata := T([1])\nresult = f(data)\n")
    assert list(ns["result"].collect()) == [2]


def test_duplicate_declaration_same_scope():
    with pytest.raises(TLScopeError):
        run("from tensorlang import T\na := T(1)\na := T(2)\n")


def test_reassignment_with_equals_is_fine():
    ns = run("from tensorlang import T\na := T(1)\na = T(2)\n")
    assert ns["a"].collect() == 2


def test_shadowing_in_function_scope_is_allowed():
    ns = run(
        "from tensorlang import T\n"
        "a := T(1)\n"
        "def f():\n"
        "    a := T(2)\n"
        "    return a.collect()\n"
        "out = f()\n"
    )
    assert ns["out"] == 2
    assert ns["a"].collect() == 1


def test_duplicate_inside_function_rejected():
    with pytest.raises(TLScopeError):
        run("from tensorlang import T\ndef f():\n    a := T(1)\n    a := T(2)\n")


def test_param_scopes_are_independent():
    ns = run(
        "from tensorlang import T\n"
        "f := {X}**2\n"
        "g := {X} + 1\n"
        "h := f >> g\n"
    )
    assert ns["h"](T(3)).collect() == 10
    assert ns["f"].uid != ns["g"].uid
    assert ns["f"].params == ("X",)
    assert ns["g"].params == ("X",)


def test_repeated_param_in_one_definition_unifies():
    ns = run("from tensorlang import T\nmask := ({X} > 10) & ({X} < 100)\n")
    op = ns["mask"]
    assert op.params == ("X",)
    assert list(op(T([5, 50, 500])).collect()) == [False, True, False]


def test_assignment_does_not_apply_operator():
    # `x = f` binds the operator itself; it must NOT mean `x = f(x)`.
    ns = run("from tensorlang import T\nf := {X} + 1\nx = f\n")
    assert ns["x"] is ns["f"]
    assert ns["x"](T(10)).collect() == 11


def test_control_flow_for_if_def_import():
    ns = run(
        "from tensorlang import T\n"
        "import math\n"
        "def pick(v):\n"
        "    if v.collect() > 1:\n"
        "        return T([v.collect() * 10])\n"
        "    return T([0])\n"
        "outs = []\n"
        "for i in [T(0), T(3)]:\n"
        "    outs = outs + [pick(i).collect()]\n"
    )
    assert ns["outs"] == [0, 30]


def test_comments_strings_multiline():
    ns = run(
        'from tensorlang import T\n'
        '# a comment with := and {X} in it\n'
        's = "{X} is not a param := here"\n'
        'm := T([\n'
        '    [1, 2],  # trailing comment\n'
        '    [3, 4],\n'
        '])\n'
        'total = (m + m).collect()\n'
    )
    assert ns["s"] == "{X} is not a param := here"
    assert ns["total"].tolist() == [[2, 4], [6, 8]]


def test_dict_and_set_literals_untouched():
    ns = run(
        "from tensorlang import T\n"
        "d = {'amount': [1, 2]}\n"
        "t = T(d)\n"
        "s = {1, 2, 3}\n"
    )
    assert ns["t"].fields == ["amount"]
    assert ns["s"] == {1, 2, 3}


def test_walrus_still_python():
    ns = run("from tensorlang import T\ny = (x := 5)\n")
    assert ns["y"] == 5 and ns["x"] == 5


def test_error_carries_source_location():
    # `ok + ok` is already a runtime (lazy) computation, so `:=` rejects it.
    with pytest.raises(TLDeclarationError) as exc_info:
        run("from tensorlang import T\nok := T(1)\nbad := ok + ok\n")
    assert "test.tl:3" in str(exc_info.value)


def test_syntax_error_carries_location():
    with pytest.raises(TLSyntaxError) as exc_info:
        _compiler.compile_source("from tensorlang import T\nf := {X} + \n", "demo.tl")
    assert "demo.tl:2" in str(exc_info.value)


def test_param_helper_matches_brace_syntax():
    ns = run("from tensorlang import T\nf := {ABC} * 2\n")
    g = P("ABC") * 2
    assert ns["f"](T(21)).collect() == g(T(21)).collect() == 42


def test_multiline_operator_definition():
    ns = run(
        "from tensorlang import T\n"
        "eligible := (\n"
        "    ({D}.amount > 100) &\n"
        "    ({D}.status == 'completed')\n"
        ")\n"
    )
    assert ns["eligible"].params == ("D",)
