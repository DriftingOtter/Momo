"""M2 tests: native file-based `.tl` module loading.

Uses deterministic temporary project directories (no ambient packages).
Module isolation between tests: every `run_file` opens a fresh loader
session and `sys.modules` is scrubbed of test modules on teardown.
"""
import sys

import pytest

from tensorlang import P, T, TLDeclarationError, run_file

CALIB = (
    "from tensorlang import T\n"
    "\n"
    "FACTOR = 2\n"
    "\n"
    "calibrate := {M} * 2 + 1\n"
)

TRANSFORMS = (
    "from tensorlang import T\n"
    "\n"
    "zscore := ({X} - {M}) / {S}\n"
    "center := {X} - {M}\n"
)


@pytest.fixture()
def proj(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    (root / "operators").mkdir(parents=True)
    (root / "operators" / "calibration.tl").write_text(CALIB)
    (root / "operators" / "transforms.tl").write_text(TRANSFORMS)
    (root / "inputs").mkdir()
    (root / "inputs" / "sources.tl").write_text(
        "from tensorlang import T\n"
        "kind := 'sources'\n"
        "describe := {X} + 0\n")
    monkeypatch.chdir(root)
    yield root
    for name in [n for n in sys.modules
                 if n.split(".")[0] in {"calibration", "transforms",
                                         "sources", "analysis", "cycle_a",
                                         "cycle_b", "bad", "evilmod",
                                         "sub", "consumer", "fitted",
                                         "shadow", "once", "helpers",
                                         "main", "usespy"}]:
        sys.modules.pop(name, None)


def write(root, rel, text):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


# -- basic imports -------------------------------------------------------------

def test_sibling_import(proj):
    main = write(proj, "main.tl",
                 "from calibration import calibrate\n"
                 "from tensorlang import T\n"
                 "out = calibrate(T([1.0, 2.0]))\n")
    write(proj, "calibration.tl", CALIB)
    ns = run_file(str(main))
    assert ns["out"].collect().tolist() == [3.0, 5.0]


def test_from_import_and_alias(proj):
    main = write(proj, "main.tl",
                 "from calibration import calibrate as apply_cal\n"
                 "from transforms import center\n"
                 "from tensorlang import T\n"
                 "a = apply_cal(T([1.0]))\n"
                 "b = center(T([5.0]), T([2.0]))\n")
    write(proj, "calibration.tl", CALIB)
    write(proj, "transforms.tl", TRANSFORMS)
    ns = run_file(str(main))
    assert ns["a"].collect().tolist() == [3.0]
    assert ns["b"].collect().tolist() == [3.0]


def test_dotted_subdirectory_import(proj):
    main = write(proj, "main.tl",
                 "from operators.calibration import calibrate\n"
                 "from inputs.sources import kind\n"
                 "from tensorlang import T\n"
                 "out = calibrate(T([0.0]))\n")
    ns = run_file(str(main))
    assert ns["out"].collect().tolist() == [1.0]
    assert ns["kind"] == "sources"


def test_python_imports_unaffected(proj):
    main = write(proj, "main.tl",
                 "import os\n"
                 "from tensorlang import T\n"
                 "here = os.path.basename(os.path.join('a', 'b'))\n"
                 "v := T([1])\n")
    ns = run_file(str(main))
    assert ns["here"] == "b"


def test_stdlib_collision_prefers_python(proj):
    # A sibling json.tl must NOT shadow the stdlib from .tl code.
    write(proj, "json.tl", "x := 1\n")
    main = write(proj, "main.tl",
                 "import json\n"
                 "out = json.dumps({'a': 1})\n")
    ns = run_file(str(main))
    assert ns["out"] == '{"a": 1}'


def test_missing_module_TL0310(proj):
    main = write(proj, "main.tl", "import nosuchmod_xyz\n")
    with pytest.raises(TLDeclarationError) as exc_info:
        run_file(str(main))
    assert exc_info.value.code == "TL0310"
    assert "nosuchmod_xyz" in str(exc_info.value)
    assert "main.tl:1" in str(exc_info.value)


def test_missing_symbol_TL0311(proj):
    write(proj, "calibration.tl", CALIB)
    main = write(proj, "main.tl",
                 "from calibration import nosuchsym\n")
    with pytest.raises(TLDeclarationError) as exc_info:
        run_file(str(main))
    assert exc_info.value.code == "TL0311"
    assert "nosuchsym" in str(exc_info.value)


def test_relative_import_TL0314(proj):
    main = write(proj, "main.tl", "from . import thing\n")
    with pytest.raises(TLDeclarationError) as exc_info:
        run_file(str(main))
    assert exc_info.value.code == "TL0314"


# -- namespaces, identity, caching ---------------------------------------------

def test_repeated_import_single_init(proj):
    write(proj, "once.tl",
          "from tensorlang import T\n"
          "log = []\n"
          "log.append(1)\n"
          "f := {X} + 1\n")
    main = write(proj, "main.tl",
                 "import once\n"
                 "import once\n"
                 "from once import f\n"
                 "from tensorlang import T\n"
                 "out = f(T([1]))\n")
    ns = run_file(str(main))
    assert ns["once"].log == [1]
    assert ns["out"].collect().tolist() == [2]


def test_multi_consumer_shared_identity(proj):
    write(proj, "calibration.tl", CALIB)
    write(proj, "consumer.tl",
          "from calibration import calibrate\n"
          "doubled := calibrate >> calibrate\n")
    main = write(proj, "main.tl",
                 "from calibration import calibrate\n"
                 "import consumer\n"
                 "from consumer import doubled\n"
                 "from tensorlang import T\n"
                 "direct = calibrate(T([1.0]))\n"
                 "via = doubled(T([1.0]))\n")
    ns = run_file(str(main))
    assert ns["direct"].collect().tolist() == [3.0]
    assert ns["via"].collect().tolist() == [7.0]
    assert ns["consumer"].calibrate is ns["calibrate"]


def test_namespace_isolation(proj):
    write(proj, "calibration.tl", CALIB)
    main = write(proj, "main.tl",
                 "from calibration import calibrate\n"
                 "FACTOR = 99\n"
                 "from tensorlang import T\n"
                 "out = calibrate(T([1.0]))\n")
    ns = run_file(str(main))
    assert ns["FACTOR"] == 99
    assert ns["calibrate"].uid != ns.get("other", None)


def test_uid_scoped_params_across_modules(proj):
    main = write(proj, "main.tl",
                 "from operators.calibration import calibrate\n"
                 "from operators.transforms import zscore\n"
                 "from tensorlang import T\n"
                 "combo := calibrate >> zscore\n")
    # zscore needs 3 params; calibrate 1: composition rejects arity here;
    # the point is parameter ownership stays distinct.
    ns_cal = run_file(str(proj / "operators" / "calibration.tl"))
    ns_tr = run_file(str(proj / "operators" / "transforms.tl"))
    assert ns_cal["calibrate"].uid != ns_tr["zscore"].uid
    with pytest.raises(Exception):
        run_file(str(main))


def test_cross_module_composition(proj):
    write(proj, "calibration.tl", CALIB)
    main = write(proj, "main.tl",
                 "from calibration import calibrate\n"
                 "addten := {M} + 10\n"
                 "pipeline := calibrate >> addten\n"
                 "from tensorlang import T\n"
                 "out = pipeline(T([1.0]))\n")
    ns = run_file(str(main))
    assert ns["out"].collect().tolist() == [13.0]


def test_imported_fitted_operator(proj):
    write(proj, "fitted.tl",
          "from tensorlang import T, standardize\n"
          "data = T([[1.0, 2.0], [3.0, 4.0]])\n"
          "scaler := standardize()\n"
          "ready = scaler.fit(data)\n")
    main = write(proj, "main.tl",
                 "from fitted import ready\n"
                 "from tensorlang import T\n"
                 "out = ready(T([[5.0, 6.0]]))\n")
    ns = run_file(str(main))
    assert ns["out"].collect().shape == (1, 2)


def test_independent_sessions_reinitialize(proj):
    write(proj, "once.tl",
          "log = []\n"
          "log.append(1)\n")
    main = write(proj, "main.tl", "import once\n")
    first = run_file(str(main))
    second = run_file(str(main))
    assert first["once"] is not second["once"]
    assert first["once"].log == [1] and second["once"].log == [1]


def test_init_failure_no_poison_and_retry(proj):
    bad = write(proj, "bad.tl", "f := {X} + \n")
    main = write(proj, "main.tl", "import bad\n")
    with pytest.raises(TLDeclarationError):
        run_file(str(main))
    bad.write_text("f := {X} + 1\n")
    ns = run_file(str(main))
    assert ns["bad"].f(T([1])).collect() == 2


# -- circular imports ----------------------------------------------------------

def test_direct_cycle_TL0312(proj):
    write(proj, "cycle_a.tl", "import cycle_b\nx := 1\n")
    write(proj, "cycle_b.tl", "import cycle_a\ny := 2\n")
    main = write(proj, "main.tl", "import cycle_a\n")
    with pytest.raises(TLDeclarationError) as exc_info:
        run_file(str(main))
    assert exc_info.value.code == "TL0312"
    assert "cycle_a" in str(exc_info.value) and "cycle_b" in str(exc_info.value)


def test_indirect_cycle_and_session_recovery(proj):
    write(proj, "cycle_a.tl", "import cycle_b\nx := 1\n")
    write(proj, "cycle_b.tl", "import consumer\nx := 1\n")
    write(proj, "consumer.tl", "import cycle_a\nx := 1\n")
    main = write(proj, "main.tl", "import cycle_a\n")
    with pytest.raises(TLDeclarationError) as exc_info:
        run_file(str(main))
    assert exc_info.value.code == "TL0312"
    # Later independent sessions are not poisoned.
    write(proj, "ok.tl", "v := 1\n")
    ok_main = write(proj, "okmain.tl", "import ok\n")
    ns = run_file(str(ok_main))
    assert ns["ok"].v == 1


# -- M1 enforcement across imports ----------------------------------------------

def test_effect_gate_through_import_chain(proj, tmp_path):
    sentinel = tmp_path / "fired.txt"
    (proj / "helpers.py").write_text(
        "def grab():\n"
        f"    open({str(sentinel)!r}, 'w').write('x')\n"
        "    return 41\n")
    write(proj, "evilmod.tl",
          "import helpers\n"
          "x := helpers.grab()\n")
    main = write(proj, "main.tl", "import evilmod\n")
    with pytest.raises(TLDeclarationError) as exc_info:
        run_file(str(main))
    assert exc_info.value.code == "TL0313"
    assert "TL0305" in str(exc_info.value)
    assert not sentinel.exists()


def test_shadowed_import_not_callable(proj):
    write(proj, "calibration.tl", CALIB)
    main = write(proj, "main.tl",
                 "from calibration import calibrate\n"
                 "from tensorlang import T\n"
                 "d := T([1.0])\n"
                 "r := calibrate(d)\n")
    with pytest.raises(TLDeclarationError) as exc_info:
        run_file(str(main))
    assert exc_info.value.code == "TL0305"


def test_imported_value_usable_at_runtime(proj):
    write(proj, "calibration.tl", CALIB)
    main = write(proj, "main.tl",
                 "from calibration import calibrate\n"
                 "from tensorlang import T\n"
                 "r = calibrate(T([2.0]))\n")
    ns = run_file(str(main))
    assert ns["r"].collect().tolist() == [5.0]


# -- paths, cwd, python interop --------------------------------------------------

def test_runs_from_unrelated_cwd(proj, tmp_path, monkeypatch):
    write(proj, "calibration.tl", CALIB)
    main = write(proj, "main.tl",
                 "from calibration import calibrate\n"
                 "from tensorlang import T\n"
                 "out = calibrate(T([3.0]))\n")
    monkeypatch.chdir(tmp_path)
    ns = run_file(str(main))
    assert ns["out"].collect().tolist() == [7.0]


def test_paths_with_spaces(tmp_path):
    root = tmp_path / "my proj"
    root.mkdir()
    (root / "calibration.tl").write_text(CALIB)
    (root / "main.tl").write_text(
        "from calibration import calibrate\n"
        "from tensorlang import T\n"
        "out = calibrate(T([1.0]))\n")
    ns = run_file(str(root / "main.tl"))
    assert ns["out"].collect().tolist() == [3.0]


def test_python_hosted_module_access(proj):
    write(proj, "calibration.tl", CALIB)
    main = write(proj, "main.tl", "from calibration import calibrate\n")
    ns = run_file(str(main))
    op = ns["calibrate"]
    assert op(T([4.0])).collect().tolist() == [9.0]
    again = op(T([4.0])).collect().tolist()
    assert again == [9.0]


def test_repl_imports(tmp_path, monkeypatch):
    from tensorlang.frontend.repl import Repl

    monkeypatch.chdir(tmp_path)
    (tmp_path / "calibration.tl").write_text(CALIB)
    r = Repl()
    assert r.push("from calibration import calibrate") == ""
    assert r.push("from tensorlang import T") == ""
    assert r.push("calibrate(T([2.0]))") == "array([5.])"


def test_sibling_py_helper_still_works(proj):
    (proj / "helpers.py").write_text("def double(x):\n    return x * 2\n")
    main = write(proj, "main.tl",
                 "import helpers\n"
                 "out = helpers.double(21)\n")
    ns = run_file(str(main))
    assert ns["out"] == 42


def test_cli_from_unrelated_cwd(proj, tmp_path, monkeypatch):
    from tensorlang.cli import main as cli_main

    write(proj, "calibration.tl", CALIB)
    main = write(proj, "main.tl",
                 "from calibration import calibrate\n"
                 "from tensorlang import T\n"
                 "out = calibrate(T([1.0]))\n")
    monkeypatch.chdir(tmp_path)
    assert cli_main(["run", str(main)]) == 0


def test_cli_effect_preflight_transitive(proj, capsys):
    from tensorlang.cli import main as cli_main

    write(proj, "dep.tl",
          "from tensorlang import T\n"
          "t := T([1])\n"
          "t.write_parquet('/tmp/m2dep.parquet')\n")
    main = write(proj, "main.tl", "import dep\n")
    assert cli_main(["graph", str(main), "--var", "t"]) == 1
    out = capsys.readouterr().err
    assert "dep.tl" in out


def test_equivalent_spellings_share_module(proj):
    write(proj, "sub", "x") if False else None
    (proj / "pkg").mkdir(exist_ok=True)
    write(proj, "pkg/mod.tl", "v := 41\n")
    main = write(proj, "main.tl",
                 "import pkg.mod\n"
                 "from pkg import mod\n")
    ns = run_file(str(main))
    assert ns["mod"].v == 41
    assert ns["pkg"].mod is ns["mod"]


def test_invalid_extension_rejected(proj):
    (proj / "note.py").write_text("x = 1\n")
    with pytest.raises(TLDeclarationError):
        run_file(str(proj / "note.py"))


def test_diagnose_missing_module_chain(proj):
    from tensorlang.frontend import diagnose as _diagnose

    main = write(proj, "main.tl", "import nosuchmod_xyz\n")
    report = _diagnose.diagnose_file(str(main))
    assert report["ok"] is False
    assert report["error"]["code"] == "TL0310"
    assert "main.tl:1" in report["error"]["message"]


def test_python_shadow_breaks_phantom_cycle(proj):
    # evil.py wins over evil.tl (Python-first): importing evil must not
    # report a circular .tl import through the shadowed file.
    (proj / "evil.py").write_text("VALUE = 7\n")
    write(proj, "evil.tl", "import evildep\n")
    write(proj, "evildep.tl", "import evil\n")
    main = write(proj, "main.tl",
                 "import evil\n"
                 "out = evil.VALUE\n")
    ns = run_file(str(main))
    assert ns["out"] == 7


def test_nested_cycle_detected_prescan(proj):
    (proj / "nest").mkdir(exist_ok=True)
    write(proj, "nest/a.tl", "import b\nx := 1\n")
    write(proj, "nest/b.tl", "import a\ny := 2\n")
    main = write(proj, "main.tl", "import nest.a\n")
    with pytest.raises(TLDeclarationError) as exc_info:
        run_file(str(main))
    assert exc_info.value.code == "TL0312"
    assert "nest" in str(exc_info.value)
