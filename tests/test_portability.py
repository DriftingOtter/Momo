"""Portability gate: no developer-path assumptions (M4).

Runs scripts/check_portability.py logic in-process: rule fixtures,
allowlist discipline, and a self-scan of the repository tree.
"""
import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "check_portability.py"


def _load():
    spec = importlib.util.spec_from_file_location(
        "check_portability", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def checker():
    return _load()


def test_rules_fire_on_fixtures(checker, tmp_path):
    _U = "/" + "Users" + "/"
    _H = "/" + "home" + "/"
    _D = "/" + "tmp" + "/"
    _P = "/" + "private" + "/"
    cases = {
        _U + "dev/pkg": "macos-home",
        _H + "dev/pkg": "linux-home",
        "x = $HO" + "ME/y": "home-env",
        "p = ~" + "/data": "tilde-home",
        "C:\\" + "Users" + "\\x": "windows-home",
        _P + "tmp/z": "macos-private",
        "p = '" + _D + "x.parquet'": "hardcoded-tmp",
        "PYTHON" + "PATH=/x": "pythonpath",
    }
    for text, rule in cases.items():
        target = tmp_path / "case.py"
        target.write_text(text + "\n", encoding="utf-8")
        hits = checker.check_file(target, tmp_path)
        assert any(f"[{rule}]" in hit for hit in hits), (rule, text)


def test_allowlist_entries_are_live(checker):
    from pathlib import Path as _Path

    for sub, rule, reason in checker.ALLOWLIST:
        assert reason.strip(), sub
        matches = [p for p in ROOT.rglob("*")
                   if sub in str(p.relative_to(ROOT).as_posix())
                   and p.is_file()]
        assert matches, f"stale allowlist entry (no such path): {sub}"
        _, pattern, _ = next(r for r in checker.RULES if r[0] == rule)
        assert any(pattern.search(p.read_text(encoding="utf-8",
                                              errors="ignore"))
                   for p in matches if p.suffix in checker.SCAN_SUFFIXES), \
            f"dead allowlist entry (rule never fires): {sub} [{rule}]"


def test_repository_tree_is_portable(checker):
    assert checker.check_tree(ROOT) == []
