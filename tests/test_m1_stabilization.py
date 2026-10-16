"""v1.0 M1 tests: contract stabilization and release-blocker closure.

Hermetic tests (manifest discovery, uv-failure summaries, wheel-spec
resolution, missing-file diagnostics, check_source status, to_dense
and join-order contracts, language-ambiguity conformance, stable-API
presence) run anywhere. The local-wheel end-to-end test needs uv plus
package-index access and skips explicitly otherwise. Every project
lives under tmp_path; no developer paths, no ambient packages.
"""
import os
import pathlib
import shutil
import subprocess
import sys
import zipfile

import pytest

import tensorlang as tl
from tensorlang import run_source
from tensorlang.cli import main as tl_main
from tensorlang.frontend import compiler as _compiler
from tensorlang.project import (
    ProjectError,
    entry_path,
    find_root,
    load_manifest,
    main,
    summarize_uv_failure,
)
from tensorlang.project import _resolve_local_wheel

ROOT = pathlib.Path(__file__).resolve().parent.parent
UV = shutil.which("uv")


def _write_manifest(root, body):
    (root / "pyproject.toml").write_text(body, encoding="utf-8")


VALID_MANIFEST = (
    '[project]\nname = "demo"\nversion = "0.1.0"\n'
    '[tool.tensorlang]\nentry = "src/main.tl"\n'
)


def _chdir(monkeypatch, path):
    monkeypatch.chdir(path)


# -- P0: invalid-manifest discovery ---------------------------------------------

def test_discovery_three_states(tmp_path, monkeypatch):
    outer = tmp_path / "outer"
    outer.mkdir()
    _write_manifest(outer, VALID_MANIFEST)
    inner = outer / "inner"
    inner.mkdir()
    _write_manifest(inner, "this is [ not valid toml {{{\n")
    sub = inner / "sub"
    sub.mkdir()
    # Nearest invalid manifest wins over the valid outer project.
    _chdir(monkeypatch, sub)
    with pytest.raises(ProjectError) as exc_info:
        find_root(sub)
    assert "invalid manifest" in str(exc_info.value)
    assert "no TensorLang project found" not in str(exc_info.value)
    # The valid outer project still resolves from itself.
    _chdir(monkeypatch, outer)
    assert find_root(outer) == outer


def test_discovery_no_manifest(tmp_path, monkeypatch):
    plain = tmp_path / "plain" / "nested"
    plain.mkdir(parents=True)
    _chdir(monkeypatch, plain)
    with pytest.raises(ProjectError) as exc_info:
        find_root(plain)
    assert "no TensorLang project found" in str(exc_info.value)


def test_discovery_malformed_section(tmp_path, monkeypatch):
    root = tmp_path / "bad"
    root.mkdir()
    _write_manifest(root, '[project]\nname="b"\nversion="0.1"\n'
                           '[tool]\ntensorlang = 5\n')
    _chdir(monkeypatch, root)
    with pytest.raises(ProjectError) as exc_info:
        find_root(root)
    assert "invalid manifest" in str(exc_info.value)


def test_discovery_plain_pyproject_skipped(tmp_path, monkeypatch):
    outer = tmp_path / "outer"
    outer.mkdir()
    _write_manifest(outer, VALID_MANIFEST)
    plain = outer / "plainpkg"
    plain.mkdir()
    _write_manifest(plain, '[project]\nname="p"\nversion="0.1.0"\n')
    sub = plain / "sub"
    sub.mkdir()
    _chdir(monkeypatch, sub)
    assert find_root(sub) == outer


def test_discovery_spaces_unicode(tmp_path, monkeypatch):
    root = tmp_path / "my proj ünï"
    root.mkdir()
    _write_manifest(root, VALID_MANIFEST)
    sub = root / "nested dir"
    sub.mkdir()
    _chdir(monkeypatch, sub)
    assert find_root(sub) == root
    assert load_manifest(root)["project"]["name"] == "demo"


def test_manifest_wrong_field_types(tmp_path, monkeypatch):
    root = tmp_path / "badtype"
    root.mkdir()
    _write_manifest(root, '[project]\nname="b"\nversion=123\n'
                           '[tool.tensorlang]\nentry="src/main.tl"\n')
    _chdir(monkeypatch, root)
    assert find_root(root) == root
    with pytest.raises(ProjectError) as exc_info:
        load_manifest(root)
    assert "version" in str(exc_info.value)
    assert "string" in str(exc_info.value)


def test_manifest_unknown_fields_tolerated(tmp_path, monkeypatch):
    root = tmp_path / "extra"
    root.mkdir()
    _write_manifest(root, VALID_MANIFEST + '[tool.other]\nx = 1\n'
                           '[project.optional-dependencies]\ntest = ["pytest"]\n')
    _chdir(monkeypatch, root)
    assert load_manifest(root)["project"]["name"] == "demo"


def test_doctor_reports_invalid_without_touching(tmp_path, monkeypatch):
    from tensorlang.project import cmd_doctor
    import argparse

    root = tmp_path / "proj"
    root.mkdir()
    _write_manifest(root, "broken [[ toml\n")
    _chdir(monkeypatch, root)
    rc = cmd_doctor(argparse.Namespace())
    assert rc == 1
    assert not (root / ".venv").exists()


# -- P1: uv failure context ------------------------------------------------------

class _Proc:
    def __init__(self, rc, err, out=""):
        self.returncode = rc
        self.stderr = err
        self.stdout = out


def test_uv_failure_keeps_context():
    err = "\n".join(f"line{i}: detail {i}" for i in range(12))
    msg = summarize_uv_failure(["sync"], _Proc(1, err))
    assert "uv sync failed (exit 1)" in msg
    for i in range(4, 12):
        assert f"line{i}" in msg
    assert "line0" not in msg  # bounded tail, newest kept


def test_uv_failure_stdout_fallback_and_empty():
    assert "fallback" in summarize_uv_failure(["sync"], _Proc(1, "", "fallback"))
    assert "no output captured" in summarize_uv_failure(["sync"], _Proc(1, ""))


def test_uv_failure_redacts_credentials():
    msg = summarize_uv_failure(
        ["sync"], _Proc(1, "error fetching https://user:s3cret@example.com/x"))
    assert "s3cret" not in msg
    assert "https://***@example.com/x" in msg


# -- P1: local-wheel spec resolution ----------------------------------------------

def _make_wheel(directory, stem="mylib_tl", version="0.1.0"):
    dist = f"{stem}-{version}.dist-info"
    name = f"{stem}-{version}-py3-none-any.whl"
    path = pathlib.Path(directory) / name
    record = "".join(f"{f},,\n" for f in
                     [f"{stem}/__init__.py", f"{stem}/calibration.tl",
                      f"{dist}/METADATA", f"{dist}/WHEEL"])
    record += f"{dist}/RECORD,,\n"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{stem}/__init__.py", '"""mini"""\n')
        z.writestr(f"{stem}/calibration.tl", "calibrate := {X} + 1.0\n")
        z.writestr(f"{dist}/METADATA",
                   f"Metadata-Version: 2.1\nName: {stem}\nVersion: {version}\n")
        z.writestr(f"{dist}/WHEEL", "Wheel-Version: 1.0\nGenerator: m1\n"
                                    "Root-Is-Purelib: true\nTag: py3-none-any\n")
        z.writestr(f"{dist}/RECORD", record)
    return path


def test_wheel_index_spec_untouched(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    root.mkdir()
    _write_manifest(root, VALID_MANIFEST)
    _chdir(monkeypatch, root)
    assert _resolve_local_wheel("numpy", root) is None
    assert _resolve_local_wheel("scipy>=1.14", root) is None


def test_wheel_missing_and_non_wheel_rejected(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    root.mkdir()
    _write_manifest(root, VALID_MANIFEST)
    _chdir(monkeypatch, root)
    with pytest.raises(ProjectError) as exc_info:
        _resolve_local_wheel("./dist/absent-0.1-py3-none-any.whl", root)
    assert "local artifact not found" in str(exc_info.value)
    other = root / "sub"
    other.mkdir()
    (other / "notes.txt").write_text("hi")
    with pytest.raises(ProjectError) as exc_info:
        _resolve_local_wheel("sub/notes.txt", root)
    assert ".whl" in str(exc_info.value)


def test_wheel_inside_root_is_relative(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    (root / "dist").mkdir(parents=True)
    _write_manifest(root, VALID_MANIFEST)
    wheel = _make_wheel(root / "dist")
    _chdir(monkeypatch, root)
    got = _resolve_local_wheel("dist/" + wheel.name, root)
    assert got == "dist/" + wheel.name
    assert not os.path.isabs(got)


def test_wheel_outside_root_is_absolute(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    root.mkdir()
    _write_manifest(root, VALID_MANIFEST)
    wheel = _make_wheel(tmp_path)
    _chdir(monkeypatch, root)
    got = _resolve_local_wheel(str(wheel), root)
    assert os.path.isabs(got)


def test_wheel_spaces_in_path(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    spaced = root / "my wheels"
    spaced.mkdir(parents=True)
    _write_manifest(root, VALID_MANIFEST)
    wheel = _make_wheel(spaced)
    _chdir(monkeypatch, root)
    got = _resolve_local_wheel("my wheels/" + wheel.name, root)
    assert " " in got and got.endswith(".whl")


# -- P1: missing-file diagnostics -------------------------------------------------

def test_cli_missing_file_structured():
    for cmd in (["run"], ["check"], ["diagnose"], ["graph"]):
        rc = tl_main(cmd + ["/definitely/not/here.tl"])
        assert rc == 1


def test_cli_missing_file_names_path(capsys):
    rc = tl_main(["run", "/definitely/not/here.tl"])
    assert rc == 1
    assert "TL0103" in capsys.readouterr().err


def test_run_file_and_diagnose_coded(tmp_path):
    missing = str(tmp_path / "absent.tl")
    with pytest.raises(tl.TLSyntaxError) as exc_info:
        tl.run_file(missing)
    assert exc_info.value.code == "TL0103"
    assert missing in str(exc_info.value)
    report = tl.diagnose_file(missing)
    assert report["ok"] is False
    assert report["error"]["code"] == "TL0103"
    assert report["path"] == missing
    sub = tmp_path / "adir"
    sub.mkdir()
    with pytest.raises(tl.TLSyntaxError) as exc_info:
        _compiler.read_source_file(sub)
    assert exc_info.value.code == "TL0103"


# -- P1: check_source public status ------------------------------------------------

def test_check_source_stable_signature():
    tree = tl.check_source("x := 1\n", "s.tl")
    import ast as _ast

    assert isinstance(tree, _ast.Module)


def test_check_source_plumbing_warns_but_works():
    with pytest.warns(DeprecationWarning):
        tree = tl.check_source("x := 1\n", "s.tl",
                               trusted_names=frozenset())
    import ast as _ast

    assert isinstance(tree, _ast.Module)


# -- P1: to_dense admission contract -------------------------------------------------

def test_bridge_describe_states_scope():
    import polars as pl

    lazy = tl.T(pl.DataFrame({"a": [1.0], "b": [2]})).to_dense(
        columns=["a", "b"])
    text = lazy.explain()
    assert "AFTER the upstream frame materializes" in text
    assert "not bounded by this policy" in text


def test_bridge_unknown_cardinality_not_zero(tmp_path):
    import polars as pl

    path = str(tmp_path / "f.parquet")
    pl.DataFrame({"a": [1.0, 2.0], "b": [3, 4]}).write_parquet(path)
    lazy = tl.T.scan_parquet(path).to_dense(columns=["a", "b"])
    summary = lazy.abstract()
    assert summary["bridge"]["row_known"] is False
    assert "unknown" in lazy.explain()


def test_bridge_rejection_before_allocation():
    import polars as pl

    df = tl.T(pl.DataFrame({"a": [1.0, 2.0], "b": [3, 4]}))
    lazy = df.to_dense(columns=["a", "b"])
    old = tl.current_policy()
    tl.set_chunk_policy(memory_budget_bytes=8)
    try:
        with pytest.raises(Exception) as exc_info:
            lazy.collect()
        assert getattr(exc_info.value, "code", None) == "TL1020"
    finally:
        if old.get("memory_budget_bytes") is None:
            tl.clear_chunk_policy()
        else:
            tl.set_chunk_policy(
                memory_budget_bytes=old["memory_budget_bytes"])
    import numpy as np

    assert np.allclose(lazy.collect(), [[1.0, 3.0], [2.0, 4.0]])


def test_bridge_matches_explicit_conversion():
    import numpy as np
    import polars as pl

    df = tl.T(pl.DataFrame({"a": [1.0, None], "b": [3, 4]}))
    got = df.to_dense(columns=["a", "b"]).collect()
    assert np.allclose(np.asarray(got), [[1.0, 3.0], [np.nan, 4.0]],
                       equal_nan=True)


# -- P1: join-order contract ------------------------------------------------------------

def _people_dept():
    left = tl.T({"id": [3, 1, 2], "dept": ["x", "y", "x"], "v": [30, 10, 20]})
    right = tl.T({"dept": ["x", "y"], "budget": [300, 100]})
    return left, right


def test_join_content_order_independent():
    left, right = _people_dept()
    first = tl.join(left, right, on="dept").collect()
    second = tl.join(left, right, on="dept").collect()
    assert first.sort("id").equals(second.sort("id"))
    assert first.sort("id")["v"].to_list() == [10, 20, 30]


def test_join_sorted_by_key_deterministic():
    left, right = _people_dept()
    again = [tl.join(left, right, on="dept").collect().sort("id")["id"].to_list()
             for _ in range(3)]
    assert again[0] == again[1] == again[2] == [1, 2, 3]


def test_join_mask_identity_not_position():
    # Fused-mask identity holds regardless of backend row order: the
    # selected record is k=2 however the join is ordered.
    src = ("from tensorlang import T\n"
           "import polars as pl\n"
           "left = T(pl.DataFrame({'k': [3, 1, 2], "
           "'a': [30.0, 10.0, 20.0]}))\n"
           "right = T(pl.DataFrame({'k': [1, 2, 3], "
           "'b': [1.0, 20.0, 3.0]}))\n"
           "j = left.join(right, on='k')\n"
           "both := (({D}.a > 15.0) & ({D}.b > 15.0))\n")
    ns = run_source(src, filename="m.tl")
    picked = ns["j"][ns["both"](ns["j"])].collect().sort("k")
    assert picked["k"].to_list() == [2]
    assert picked["a"].to_list() == [20.0]


# -- language-ambiguity conformance (M1 documented rules) ---------------------------------

def test_loop_declaration_is_static():
    ns = run_source("from tensorlang import T\n"
                    "for i in range(2):\n    a := T([1])\n", "loop.tl")
    import numpy as np

    assert np.asarray(ns["a"].collect()).tolist() == [1]


def test_flat_trust_static_then_name_error():
    src = ("from tensorlang import T\n"
           "def f():\n    q := T([1])\n\n"
           "def g():\n    z := q\n")
    tl.check_source(src, "flat.tl")  # static check passes (file-flat)
    with pytest.raises(NameError):
        run_source(src + "g()\n", "flat.tl")


def test_trusted_root_attr_subscript_rules():
    import numpy as np

    ns = run_source("from tensorlang import T\nx := T([1, 2])\n"
                    "s := x.shape\nal := x\n", "roots.tl")
    assert ns["s"] == (2,)
    assert np.asarray(ns["al"].collect()).tolist() == [1, 2]
    with pytest.raises(Exception) as exc_info:
        run_source("from tensorlang import T\nx := T([1, 2])\ny := x[0]\n",
                   "roots.tl")
    assert getattr(exc_info.value, "code", None) == "TL0301"
    with pytest.raises(Exception) as exc_info:
        run_source("from tensorlang import T\nx = T([1, 2])\ny := x.shape\n",
                   "untrusted.tl")
    assert getattr(exc_info.value, "code", None) == "TL0305"


def test_literal_receiver_admitted():
    ns = run_source("x := 'hi'.upper()\n", "lit.tl")
    assert ns["x"] == "HI"


def test_open_declarative_set_grad():
    ns = run_source("from tensorlang import grad\n"
                    "f := {X} ** 2\ng := grad(f)\n", "open.tl")
    assert ns["g"] is not None


def test_operator_alias_identity():
    ns = run_source("f := {X} * 2\nx = f\n", "alias.tl")
    assert ns["x"] is ns["f"]


def test_dunder_limitation_pinned():
    seen = []

    class Probed:
        def __add__(self, other):
            seen.append(1)
            return 5

    ns = run_source("y := obj + 1\n", "dunder.tl", extra={"obj": Probed()})
    assert seen == [1]  # disclosed: dunders execute; not a sandbox
    assert ns["y"] == 5


# -- stable API presence ----------------------------------------------------------------------

_STABLE = (
    "T Tensor Lazy Operator P sin cos tan exp log sqrt diff integrate "
    "simplify concat stack reshape flatten transpose logical_and "
    "logical_or logical_not join sort where clip standardize minmax "
    "normalize impute pca FittableOperator PipelineOperator grad "
    "value_and_grad vjp svd eigh solve norm cov corr quantile is_null "
    "not_null fill_null drop_null run_source run_file check_source "
    "diagnose_source diagnose_file set_chunk_policy clear_chunk_policy "
    "current_policy TensorLangError TLSyntaxError TLScopeError "
    "TLDeclarationError TLTypeError TLShapeError TLMutationError "
    "TLSymbolicError TLDiffError TLUnsupportedError".split()
)

_PROVISIONAL = (
    "equal not_equal unique over rolling einsum tensordot moveaxis "
    "onehot polynomial GradOperator ValueGradOperator VjpOperator "
    "ValueGradResult SVDResult EighResult".split()
)


# -- P1: local-wheel end-to-end (uv + index; skips explicitly) --------------------------

def _index_reachable():
    if UV is None:
        return False
    probe = subprocess.run(
        [UV, "pip", "install", "--dry-run", "--python",
         sys.executable, "packaging"],
        capture_output=True, text=True)
    return probe.returncode == 0


needs_network = pytest.mark.skipif(
    not _index_reachable(), reason="uv or package index unavailable")


def _make_two_op_wheel(directory):
    stem, version = "mylib_tl", "0.1.0"
    dist = f"{stem}-{version}.dist-info"
    name = f"{stem}-{version}-py3-none-any.whl"
    path = pathlib.Path(directory) / name
    files = {
        f"{stem}/__init__.py": '"""mini"""\n',
        f"{stem}/calibration.tl":
            "calibrate := {X} + 1.0\ngain := {X} * 2.0\n",
        f"{dist}/METADATA":
            f"Metadata-Version: 2.1\nName: {stem}\nVersion: {version}\n",
        f"{dist}/WHEEL": "Wheel-Version: 1.0\nGenerator: m1\n"
                         "Root-Is-Purelib: true\nTag: py3-none-any\n",
    }
    record = "".join(f"{f},,\n" for f in files) + f"{dist}/RECORD,,\n"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for arc, text in files.items():
            z.writestr(arc, text)
        z.writestr(f"{dist}/RECORD", record)
    return path


@needs_network
def test_local_wheel_end_to_end(tmp_path, monkeypatch, capfd):
    repo = str(ROOT)
    monkeypatch.chdir(tmp_path)
    assert main(["new", "libapp", "--tensorlang", repo]) == 0
    root = tmp_path / "libapp"
    monkeypatch.chdir(root)
    assert main(["sync"]) == 0
    wheel = _make_two_op_wheel(root)
    assert main(["add", wheel.name]) == 0
    assert main(["sync"]) == 0
    (root / "src" / "main.tl").write_text(
        "from tensorlang import T\n"
        "from mylib_tl.calibration import calibrate, gain\n"
        "data := T([1.0, 2.0])\n"
        "result = gain(calibrate(data))\n"
        "print(result.collect())\n",
        encoding="utf-8")
    assert main(["check"]) == 0
    assert main(["run"]) == 0
    assert "[4. 6.]" in capfd.readouterr().out
    # Locked reproduction: wheel copied to the same relative location.
    clone = tmp_path / "clone"
    shutil.copytree(root, clone,
                    ignore=shutil.ignore_patterns(".venv", "__pycache__"))
    monkeypatch.chdir(clone)
    assert main(["sync", "--locked"]) == 0
    assert main(["run"]) == 0
    assert "[4. 6.]" in capfd.readouterr().out
    # Missing artifact fails actionably on a clean reproduction: without
    # the wheel file (and an empty uv cache, as on a fresh machine), the
    # dependency cannot resolve.
    for candidate in clone.rglob("*.whl"):
        candidate.unlink()
    shutil.rmtree(clone / ".venv", ignore_errors=True)
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "emptycache"))
    assert main(["sync"]) != 0
    captured = capfd.readouterr()
    assert "mylib" in (captured.out + captured.err).lower()


def test_stable_and_provisional_cover_all_exports():
    assert len(tl.__all__) == 82
    assert sorted(_STABLE + _PROVISIONAL + ["__version__"]) == sorted(tl.__all__)
    for name in _STABLE:
        assert callable(getattr(tl, name)), name
