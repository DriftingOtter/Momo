"""Error-code foundations: stable codes, registry, catalogue freshness."""
import ast
import json
import pathlib
import re

import pytest

from tensorlang import (
    P,
    T,
    TLDeclarationError,
    TLDiffError,
    TLMutationError,
    TLScopeError,
    TLShapeError,
    TLSyntaxError,
    TensorLangError,
    TLTypeError,
)
from tensorlang.core.errors import ERROR_REGISTRY, TensorLangError
from tensorlang.frontend import diagnose as _diagnose

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
DOCS = pathlib.Path(__file__).resolve().parent.parent / "docs" / "ERROR_CODES.md"


def test_registry_shape():
    assert len(ERROR_REGISTRY) >= 40
    for code, entry in ERROR_REGISTRY.items():
        assert re.fullmatch(r"TL\d{4}", code), code
        assert entry["category"].startswith("TL"), code
        assert entry["title"].strip(), code
        assert entry["hint"].strip(), code
    assert len(set(ERROR_REGISTRY)) == len(ERROR_REGISTRY)


def test_every_error_renders_its_code():
    err = TLShapeError("bad shape", code="TL0501")
    assert err.code == "TL0501"
    assert "[TL0501]" in str(err)
    assert "bad shape" in str(err)
    defaulted = TLTypeError("plain")
    assert defaulted.code == "TL0499"
    assert "[TL0499]" in str(defaulted)
    assert defaulted.details == {}
    assert defaulted.category == "TLTypeError"


def test_all_literal_codes_are_registered():
    used: set[str] = set()
    for f in SRC.rglob("*.py"):
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
                for kw in node.exc.keywords:
                    if kw.arg == "code" and isinstance(kw.value, ast.Constant):
                        used.add(kw.value.value)
    assert used, "expected curated code= sites"
    orphans = used - set(ERROR_REGISTRY)
    assert not orphans, f"unregistered codes used: {orphans}"


def test_catalogue_matches_registry():
    assert DOCS.exists()
    assert _diagnose.render_catalogue() == DOCS.read_text(encoding="utf-8")
    text = DOCS.read_text(encoding="utf-8")
    for code in ERROR_REGISTRY:
        assert code in text


def test_frontend_codes():
    from tensorlang.frontend import compiler as _compiler

    with pytest.raises(TLScopeError) as e:
        _compiler.run_source("from tensorlang import T\na := T(1)\na := T(2)\n",
                             filename="t.tl")
    assert e.value.code == "TL0201"
    with pytest.raises(TLSyntaxError) as e:
        _compiler.compile_source("from tensorlang import T\nf := {X} + \n", "t.tl")
    assert e.value.code == "TL0101"
    with pytest.raises(TLDeclarationError) as e:
        _compiler.run_source("from tensorlang import T\nf := {X}+1\n"
                             "d := T([1])\nr := f(d)\n", filename="t.tl")
    assert e.value.code == "TL0301"
    with pytest.raises(TLDeclarationError) as e:
        _compiler.run_source("from tensorlang import T\n"
                             "t := T([1])\nc := t.collect()\n", filename="t.tl")
    assert e.value.code == "TL0303"


def test_runtime_codes():
    v = T([1.0, 2.0])
    with pytest.raises(TLShapeError) as e:
        (T([1.0, 2.0, 3.0]) + T([1.0, 2.0])).collect()
    assert e.value.code == "TL0501"
    assert e.value.details["shapes"] == [(3,), (2,)]
    with pytest.raises(TLShapeError) as e:
        P("X")(T(1), T(2))
    assert e.value.code == "TL0503"
    assert e.value.details == {"expected": ["X"], "got": 2}
    with pytest.raises(TLMutationError) as e:
        (v * 2).inplace()
    assert e.value.code == "TL0601"
    with pytest.raises(TLTypeError):
        bool(v)
    try:
        bool(v)
    except TLTypeError as exc:
        assert exc.code == "TL0401"


def test_autodiff_codes():
    with pytest.raises(TLDiffError) as e:
        grad_unknown = __import__("tensorlang").grad(P("X") + 1, wrt="Q")
    assert e.value.code == "TL0801"
    assert e.value.details["params"] == ["X"]


def test_error_report_is_json_serializable():
    report = _diagnose.error_report(TLShapeError("x", code="TL0501",
                                                 details={"shapes": [(1,), (2,)]}))
    assert json.loads(json.dumps(report))["code"] == "TL0501"
