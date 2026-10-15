"""M1 tests: fail-closed declarative effect enforcement (``TL0305``).

``:=`` admits only known TensorLang capabilities (names imported from
``tensorlang``, identity-seeded namespace names, ``:=``-declared names,
``__tl_param__``).  Arbitrary Python calls — direct, nested, aliased,
bound, or through attribute chains — are rejected *without executing*.
Runtime (``=``) interoperability is fully preserved.
"""
import subprocess
import sys

import pytest

from tensorlang import P, T, TLDeclarationError, run_file, run_source
from tensorlang.frontend import compiler as _compiler

EVIL_SRC = (
    "SENTINEL = None\n"
    "def grab():\n"
    "    global SENTINEL\n"
    "    SENTINEL = 'grabbed'\n"
    "    return 42\n"
    "class Box:\n"
    "    def __init__(self):\n"
    "        self.fired = False\n"
    "    def method(self):\n"
    "        self.fired = True\n"
    "        return 7\n"
    "    def __call__(self):\n"
    "        self.fired = True\n"
    "        return 8\n"
    "    @property\n"
    "    def prop(self):\n"
    "        self.fired = True\n"
    "        return 9\n"
    "box = Box()\n"
)


@pytest.fixture()
def evil(tmp_path, monkeypatch):
    mod = tmp_path / "evil.py"
    mod.write_text(EVIL_SRC)
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("evil", None)
    yield tmp_path
    sys.modules.pop("evil", None)


def run(src):
    return _compiler.run_source(src, filename="decl.tl")


def assert_blocked(src, needle="TL0305"):
    with pytest.raises(TLDeclarationError) as exc_info:
        run(src)
    assert exc_info.value.code == "TL0305", exc_info.value.code
    assert "decl.tl:" in str(exc_info.value)
    return exc_info.value


# -- adversarial: direct, nested, indirect -------------------------------------

def test_original_reproducer_blocked_without_effect(evil):
    import evil as _evil

    assert_blocked("import evil\nx := evil.grab()\n", )
    assert _evil.SENTINEL is None


def test_original_reproducer_in_subprocess(tmp_path):
    import os
    import pathlib

    repo = pathlib.Path(__file__).resolve().parent.parent
    sub = tmp_path / "m1sub"
    sub.mkdir()
    (sub / "evil.py").write_text(EVIL_SRC)
    code = (
        "import sys; sys.path.insert(0, 'src'); "
        f"sys.path.insert(0, {str(sub)!r});\n"
        "import evil;\n"
        "from tensorlang import run_source;\n"
        "run_source('import evil\\nx := evil.grab()\\n', filename='s.tl');\n"
        "print('SENTINEL=' + str(evil.SENTINEL))\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, cwd=str(repo))
    assert proc.returncode != 0
    assert "TL0305" in proc.stderr or "TL0305" in proc.stdout
    assert "SENTINEL=grabbed" not in (proc.stdout + proc.stderr)


def test_nested_unknown_call_blocked(evil):
    import evil as _evil

    assert_blocked("import evil\nfrom tensorlang import T\n"
                   "x := T(1 + evil.grab())\n")
    assert_blocked("import evil\nfrom tensorlang import T\n"
                   "x := T(evil.grab())\n")
    assert _evil.SENTINEL is None


def test_aliased_import_blocked(evil):
    import evil as _evil

    assert_blocked("import evil as e\nx := e.grab()\n")
    assert_blocked("from evil import grab\nx := grab()\n")
    assert _evil.SENTINEL is None


def test_runtime_alias_blocked(evil):
    import evil as _evil

    assert_blocked("import evil\nf = evil.grab\nx := f()\n")
    assert _evil.SENTINEL is None


def test_bound_method_and_callable_blocked(evil):
    import evil as _evil

    assert_blocked("import evil\nx := evil.box.method()\n")
    assert_blocked("import evil\nx := evil.box()\n")
    assert_blocked("import evil\nx := evil.box.prop\n")
    assert _evil.box.fired is False


def test_unknown_attribute_chain_blocked(evil):
    assert_blocked("import os\nx := os.path.join('a', 'b')\n")
    assert_blocked("x := print('hi')\n")
    assert_blocked("x := len([1])\n")


def test_shadowed_tensorlang_name_blocked():
    assert_blocked("import evil as T\nx := T([1])\n")
    assert_blocked("from tensorlang import T\nT = 5\nx := T([1])\n")


def test_repeated_declaration_stays_blocked(evil):
    import evil as _evil

    src = "import evil\nx := evil.grab()\n"
    for _ in range(2):
        with pytest.raises(TLDeclarationError) as exc_info:
            run(src)
        assert exc_info.value.code == "TL0305"
    assert _evil.SENTINEL is None


def test_run_file_blocked_without_effect(evil):
    src = evil / "app.tl"
    src.write_text("import evil\nx := evil.grab()\n")
    with pytest.raises(TLDeclarationError) as exc_info:
        run_file(str(src))
    assert exc_info.value.code == "TL0305"
    assert str(src) in str(exc_info.value)
    import evil as _evil

    assert _evil.SENTINEL is None


# -- known effectful/runtime codes preserved -----------------------------------

def test_known_codes_unchanged():
    with pytest.raises(TLDeclarationError) as exc_info:
        run("from tensorlang import T\nt := T([1])\nc := t.collect()\n")
    assert exc_info.value.code == "TL0303"
    with pytest.raises(TLDeclarationError) as exc_info:
        run("from tensorlang import T\nt := T({'a': [1]})\n"
            "p := t.write_parquet('/tmp/x.parquet')\n")
    assert exc_info.value.code == "TL0302"
    with pytest.raises(TLDeclarationError) as exc_info:
        run("from tensorlang import T\nf := {X} + 1\nd := T([1])\nr := f(d)\n")
    assert exc_info.value.code == "TL0301"


# -- positive compatibility ----------------------------------------------------

def test_legitimate_declarations():
    ns = run("from tensorlang import T, P, sin, diff, integrate, grad, "
             "standardize, join\n"
             "data := T([1, 2, 3])\n"
             "square := {X} ** 2\n"
             "f := sin({X}) * {X}\n"
             "df := diff(f)\n"
             "area := integrate({X}**2, 0, 5)\n"
             "scaler := standardize()\n"
             "loss := ((P('P') - P('Y'))**2).mean()\n"
             "d_loss := grad(loss, wrt='P')\n"
             "s := T.scan_parquet('/tmp/nope.parquet')\n"
             "big := T.arange(4)\n"
             "u := data\n"
             "result = square(data)\n")
    assert ns["result"].collect().tolist() == [1, 4, 9]
    assert ns["u"] is ns["data"]
    assert ns["s"].kind == "scan"


def test_import_alias_of_tensorlang_allowed():
    ns = run("from tensorlang import sin as s\nf := s({X}) + {X}\n")
    assert ns["f"](T([0.0])).collect() == 0.0


def test_identity_seeded_extra_allowed():
    ns = run_source("x := T([1, 2])\n", filename="e.tl", extra={"T": T})
    assert ns["x"].collect().tolist() == [1, 2]


def test_identity_seeding_does_not_trust_foreign():
    def grab():
        return 1

    with pytest.raises(TLDeclarationError) as exc_info:
        run_source("x := grab()\n", filename="e.tl", extra={"grab": grab})
    assert exc_info.value.code == "TL0305"


def test_declared_operator_composition_allowed():
    ns = run("from tensorlang import T\nf := {X} + 1\ng := {X} * 2\n"
             "h := f >> g\n")
    assert ns["h"](T([3])).collect() == 8


# -- runtime interop preserved --------------------------------------------------

def test_runtime_python_calls_preserved(evil):
    import evil as _evil

    ns = run("import evil\nx = evil.grab()\n")
    assert ns["x"] == 42
    assert _evil.SENTINEL == "grabbed"


def test_runtime_path_join_preserved():
    ns = run("import os\nhere = os.path.dirname('/a/b')\n")
    assert ns["here"] == "/a"


# -- tooling --------------------------------------------------------------------

def test_diagnose_reports_TL0305(evil):
    from tensorlang.frontend import diagnose as _diagnose

    report = _diagnose.diagnose_source("import evil\nx := evil.grab()\n",
                                       filename="d.tl")
    assert report["ok"] is False
    assert report["error"]["code"] == "TL0305"
    assert any("'='" in s for s in report["suggestions"])


def test_repl_blocks_without_effect(evil):
    from tensorlang.frontend.repl import Repl

    r = Repl()
    assert r.push("import evil") == ""
    out = r.push("x := evil.grab()")
    assert "TL0305" in out
    import evil as _evil

    assert _evil.SENTINEL is None
    assert r.push("from tensorlang import T") == ""
    assert r.push("v := T([1, 2])") == ""
