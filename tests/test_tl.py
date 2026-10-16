"""M2 tests: `tl` project manager (Cargo-inspired, uv-backed).

Hermetic unit tests (manifest, discovery, validation, templates) run
anywhere. Integration tests need `uv` plus package-index access and
skip explicitly otherwise. Nothing touches real user environments:
every project lives under tmp_path.
"""
import os
import pathlib
import shutil
import sys

import pytest

from tensorlang.project import (
    ProjectError,
    entry_path,
    find_root,
    load_manifest,
    main,
)

UV = shutil.which("uv")

needs_uv = pytest.mark.skipif(UV is None, reason="uv not available")


def run(*argv, cwd=None):
    if cwd is not None:
        os.chdir(cwd)
    return main(list(argv))


def make_project(monkeypatch, tmp_path, name="demo", tensorlang=None):
    monkeypatch.chdir(tmp_path)
    args = ["new", name]
    if tensorlang is not None:
        args += ["--tensorlang", tensorlang]
    assert run(*args) == 0
    return tmp_path / name


# -- hermetic: scaffolding -------------------------------------------------------

def test_new_creates_template(tmp_path, monkeypatch):
    root = make_project(monkeypatch, tmp_path)
    assert (root / "pyproject.toml").is_file()
    assert (root / "src" / "main.tl").is_file()
    assert (root / "src" / "operators" / "transforms.tl").is_file()
    assert (root / "tests" / "test_application.py").is_file()
    text = (root / "src" / "main.tl").read_text()
    assert "operators.transforms" in text
    blob = "\n".join(p.read_text() for p in root.rglob("*") if p.is_file())
    assert "/Users/" not in blob and str(tmp_path) not in blob


def test_new_invalid_names(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for bad in ["", "has space", "semi;colon", "../x"]:
        assert main(["new", bad]) != 0


def test_new_refuses_nonempty(tmp_path, monkeypatch):
    target = tmp_path / "demo"
    target.mkdir()
    (target / "existing.txt").write_text("keep")
    monkeypatch.chdir(tmp_path)
    assert main(["new", "demo"]) != 0
    assert (target / "existing.txt").is_file()


def test_init_fresh_and_existing_manifest(tmp_path, monkeypatch):
    work = tmp_path / "proj"
    work.mkdir()
    (work / "notes.txt").write_text("unrelated")
    monkeypatch.chdir(work)
    assert main(["init"]) == 0
    assert (work / "src" / "main.tl").is_file()
    assert (work / "notes.txt").is_file()
    assert main(["init"]) != 0


def test_init_rejects_bad_dirname(tmp_path, monkeypatch):
    for bad in ["has space", "semi;colon"]:
        work = tmp_path / bad
        work.mkdir()
        monkeypatch.chdir(work)
        assert main(["init"]) != 0
    ok = tmp_path / "my projects" / "fine_name"
    ok.mkdir(parents=True)
    monkeypatch.chdir(ok)
    assert main(["init"]) == 0


# -- hermetic: discovery and manifest ---------------------------------------------

def test_discovery_nested(tmp_path, monkeypatch):
    root = make_project(monkeypatch, tmp_path)
    nested = root / "src" / "operators"
    monkeypatch.chdir(nested)
    assert find_root(nested) == root


def test_discovery_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ProjectError) as exc_info:
        find_root(tmp_path)
    assert "no TensorLang project found" in str(exc_info.value)
    assert main(["run"]) != 0


def test_manifest_validation(tmp_path, monkeypatch):
    root = make_project(monkeypatch, tmp_path)
    manifest = load_manifest(root)
    assert manifest["project"]["name"] == "demo"
    assert entry_path(manifest) == root / "src" / "main.tl"
    (root / "pyproject.toml").write_text("not = [valid")
    with pytest.raises(ProjectError):
        load_manifest(root)


def test_manifest_missing_fields(tmp_path, monkeypatch):
    root = make_project(monkeypatch, tmp_path)
    (root / "pyproject.toml").write_text('[project]\nname = "x"\n')
    with pytest.raises(ProjectError) as exc_info:
        load_manifest(root)
    assert "version" in str(exc_info.value) or "tensorlang" in str(exc_info.value)


def test_run_missing_entry(tmp_path, monkeypatch):
    root = make_project(monkeypatch, tmp_path)
    (root / "src" / "main.tl").unlink()
    monkeypatch.chdir(root)
    assert main(["run"]) != 0


def test_run_outside_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["run"]) != 0


def test_requires_python_parser():
    from tensorlang.project import _check_requires_python

    assert _check_requires_python(">=3.12", (3, 14, 7))[0] is True
    assert _check_requires_python(">=3.12, <3.15", (3, 14, 0))[0] is True
    assert _check_requires_python(">=3.13", (3, 12, 0))[0] is False
    assert _check_requires_python("==3.14.7", (3, 14, 7))[0] is True
    assert _check_requires_python("nonsense", (3, 14, 0))[0] is False


def test_doctor_outside_project(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["doctor"]) == 1
    assert "tl new" in capsys.readouterr().out


def test_doctor_missing_uv(tmp_path, monkeypatch, capsys):
    root = make_project(monkeypatch, tmp_path)
    monkeypatch.chdir(root)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert main(["doctor"]) == 1
    out = capsys.readouterr().out
    assert "uv" in out


def test_add_rejects_bad_specs(tmp_path, monkeypatch):
    root = make_project(monkeypatch, tmp_path)
    monkeypatch.chdir(root)
    assert main(["add", ""]) != 0
    assert main(["add", "a;b"]) != 0


def test_help_and_unknown():
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    with pytest.raises(SystemExit):
        main(["nosuchcommand"])


# -- integration (needs uv + index) -------------------------------------------------

def _networked():
    if UV is None:
        pytest.skip("uv not available")
    import subprocess

    probe = subprocess.run(
        [UV, "pip", "install", "--dry-run", "--python",
         sys.executable, "packaging"],
        capture_output=True, text=True)
    if probe.returncode != 0:
        pytest.skip("package index unreachable")


@pytest.fixture()
def dev_repo():
    import tensorlang

    here = pathlib.Path(tensorlang.__file__).resolve()
    for parent in here.parents:
        manifest = parent / "pyproject.toml"
        if manifest.is_file() and (parent / "src").is_dir():
            return str(parent)
    pytest.skip("TensorLang source checkout not found")


def test_sync_run_check(tmp_path, monkeypatch, dev_repo):
    _networked()
    root = make_project(monkeypatch, tmp_path, tensorlang=dev_repo)
    monkeypatch.chdir(root)
    assert main(["sync"]) == 0
    assert (root / "uv.lock").is_file()
    assert (root / ".venv").is_dir()
    assert main(["check"]) == 0
    assert main(["run"]) == 0


def test_test_command(tmp_path, monkeypatch, dev_repo, capfd):
    _networked()
    root = make_project(monkeypatch, tmp_path, tensorlang=dev_repo)
    monkeypatch.chdir(root)
    assert main(["add", "--dev", "pytest"]) == 0
    assert main(["sync"]) == 0
    assert main(["test"]) == 0


def test_add_remove_tiny_dep(tmp_path, monkeypatch, dev_repo):
    _networked()
    root = make_project(monkeypatch, tmp_path, tensorlang=dev_repo)
    monkeypatch.chdir(root)
    assert main(["sync"]) == 0
    assert main(["add", "packaging"]) == 0
    assert "packaging" in (root / "pyproject.toml").read_text()
    assert main(["remove", "packaging"]) == 0
    assert "packaging" not in (root / "pyproject.toml").read_text()


def test_locked_rejects_stale(tmp_path, monkeypatch, dev_repo):
    _networked()
    root = make_project(monkeypatch, tmp_path, tensorlang=dev_repo)
    monkeypatch.chdir(root)
    assert main(["sync"]) == 0
    text = (root / "pyproject.toml").read_text()
    assert 'dependencies = ["tensorlang"]' in text
    (root / "pyproject.toml").write_text(
        text.replace('dependencies = ["tensorlang"]',
                     'dependencies = ["tensorlang", "packaging"]'))
    assert main(["sync", "--locked"]) != 0


def test_offline_sync_fails_cleanly(tmp_path, monkeypatch):
    _networked()
    root = make_project(monkeypatch, tmp_path)
    monkeypatch.chdir(root)
    monkeypatch.setenv("UV_OFFLINE", "1")
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "empty-cache"))
    assert main(["sync"]) != 0


def test_run_auto_syncs_fresh(tmp_path, monkeypatch, dev_repo, capfd):
    _networked()
    root = make_project(monkeypatch, tmp_path, tensorlang=dev_repo)
    monkeypatch.chdir(root)
    assert not (root / ".venv").exists()
    assert main(["run"]) == 0
    assert (root / ".venv").is_dir()
    assert "[12." in capfd.readouterr().out


def test_args_forwarding(tmp_path, monkeypatch, dev_repo, capfd):
    _networked()
    root = make_project(monkeypatch, tmp_path, tensorlang=dev_repo)
    (root / "src" / "main.tl").write_text(
        "import sys\nprint('got:', sys.argv[1:])\n")
    monkeypatch.chdir(root)
    assert main(["sync"]) == 0
    assert main(["run", "--", "hello", "world"]) == 0
    assert "hello" in capfd.readouterr().out


def test_nested_cwd_and_spaces(tmp_path, monkeypatch, dev_repo, capfd):
    _networked()
    base = tmp_path / "my projects"
    base.mkdir()
    monkeypatch.chdir(base)
    root = make_project(monkeypatch, base, name="spaced",
                        tensorlang=dev_repo)
    monkeypatch.chdir(root / "src" / "operators")
    assert main(["run"]) == 0
    assert "[12." in capfd.readouterr().out


def test_two_projects_isolated(tmp_path, monkeypatch, dev_repo, capfd):
    _networked()
    monkeypatch.chdir(tmp_path)
    for name in ("proj-a", "proj-b"):
        root = make_project(monkeypatch, tmp_path, name=name,
                            tensorlang=dev_repo)
        monkeypatch.chdir(root)
        assert main(["sync"]) == 0
        monkeypatch.chdir(tmp_path)
    for name, want in (("proj-a", "[12."), ("proj-b", "[12.")):
        monkeypatch.chdir(tmp_path / name)
        capfd.readouterr()
        assert main(["run"]) == 0
        assert want in capfd.readouterr().out


def test_copy_reproduce(tmp_path, monkeypatch, dev_repo, capfd):
    _networked()
    root = make_project(monkeypatch, tmp_path, tensorlang=dev_repo)
    monkeypatch.chdir(root)
    assert main(["sync"]) == 0
    assert main(["run"]) == 0
    capfd.readouterr()
    assert main(["run"]) == 0
    first = capfd.readouterr().out
    import shutil as _shutil

    clone = tmp_path / "clone"
    _shutil.copytree(root, clone,
                     ignore=_shutil.ignore_patterns(".venv", "__pycache__"))
    monkeypatch.chdir(clone)
    assert main(["sync"]) == 0
    capfd.readouterr()
    assert main(["run"]) == 0
    assert capfd.readouterr().out == first


def test_doctor_healthy(tmp_path, monkeypatch, dev_repo, capfd):
    _networked()
    root = make_project(monkeypatch, tmp_path, tensorlang=dev_repo)
    monkeypatch.chdir(root)
    assert main(["sync"]) == 0
    assert main(["doctor"]) == 0
    assert "healthy" in capfd.readouterr().out


def test_sync_records_freshness(tmp_path, monkeypatch, dev_repo):
    _networked()
    root = make_project(monkeypatch, tmp_path, tensorlang=dev_repo)
    monkeypatch.chdir(root)
    assert main(["sync"]) == 0
    from tensorlang.project import sync_freshness

    assert sync_freshness(root) == "fresh"


def test_manifest_change_marks_stale(tmp_path, monkeypatch, dev_repo, capfd):
    _networked()
    root = make_project(monkeypatch, tmp_path, tensorlang=dev_repo)
    monkeypatch.chdir(root)
    assert main(["sync"]) == 0
    with open(root / "pyproject.toml", "a") as fh:
        fh.write("\n# touch\n")
    from tensorlang.project import sync_freshness

    assert sync_freshness(root).startswith("stale:")
    assert main(["doctor"]) == 1
    assert "STALE" in capfd.readouterr().out
    assert main(["run"]) == 0
    assert "may be stale" in capfd.readouterr().err


def test_run_without_lockfile_warns_nothing(tmp_path, monkeypatch, dev_repo):
    _networked()
    root = make_project(monkeypatch, tmp_path, tensorlang=dev_repo)
    monkeypatch.chdir(root)
    (root / "uv.lock").unlink(missing_ok=True)
    assert main(["sync"]) == 0
    assert (root / "uv.lock").is_file()


def test_global_package_does_not_leak(tmp_path, monkeypatch, dev_repo):
    _networked()
    root = make_project(monkeypatch, tmp_path, tensorlang=dev_repo)
    (root / "src" / "main.tl").write_text("import pytest\n")
    monkeypatch.chdir(root)
    assert main(["sync"]) == 0
    assert main(["run"]) != 0


def test_app_exit_status_and_stderr(tmp_path, monkeypatch, dev_repo):
    _networked()
    root = make_project(monkeypatch, tmp_path, tensorlang=dev_repo)
    (root / "src" / "main.tl").write_text("import sys\nsys.exit(7)\n")
    monkeypatch.chdir(root)
    assert main(["sync"]) == 0
    assert main(["run"]) == 7


def test_unicode_args(tmp_path, monkeypatch, dev_repo, capfd):
    _networked()
    root = make_project(monkeypatch, tmp_path, tensorlang=dev_repo)
    (root / "src" / "main.tl").write_text(
        "import sys\nprint('got:', sys.argv[3:])\n")
    monkeypatch.chdir(root)
    assert main(["sync"]) == 0
    assert main(["run", "--", "héllo", "a b"]) == 0
    assert "héllo" in capfd.readouterr().out
