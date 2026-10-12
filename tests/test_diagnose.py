"""Analyser tests: suggestions, diagnose flows, CLI surface."""
import json

import pytest

from tensorlang import T
from tensorlang.cli import main as cli_main
from tensorlang.frontend import diagnose as _diagnose


def test_diagnose_success(tmp_path):
    report = _diagnose.diagnose_source(
        "from tensorlang import T\na := T([1, 2])\n", filename="ok.tl")
    assert report["ok"] is True
    assert "a" in report["names"]


def test_diagnose_declare_application():
    report = _diagnose.diagnose_source(
        "from tensorlang import T\nf := {X} + 1\nd := T([1])\nr := f(d)\n",
        filename="bad.tl")
    assert report["ok"] is False
    assert report["error"]["code"] == "TL0301"
    assert any("'='" in s for s in report["suggestions"])
    assert "bad.tl:4" in (report["error"]["loc"] or "")


def test_diagnose_typo_field():
    report = _diagnose.diagnose_source(
        "from tensorlang import T\n"
        "sales := T({'amount': [1, 2]})\n"
        "x = sales.amout.collect()\n",
        filename="typo.tl")
    assert report["ok"] is False
    assert any("amount" in s for s in report["suggestions"]), report


def test_diagnose_missing_import():
    report = _diagnose.diagnose_source("x = sin(1.0)\n", filename="imp.tl")
    assert report["ok"] is False
    assert any("from tensorlang import sin" in s
               for s in report["suggestions"]), report


def test_diagnose_unknown_wrt_param():
    report = _diagnose.diagnose_source(
        "from tensorlang import T, P, grad\n"
        "f := {X} + 1\n"
        "d := grad(f, wrt='Q')\n",
        filename="w.tl")
    assert report["error"]["code"] == "TL0801"
    assert any("X" in s for s in report["suggestions"]), report


def test_diagnose_missing_file(tmp_path):
    report = _diagnose.diagnose_file(tmp_path / "nope.tl")
    assert report["ok"] is False
    assert report["error"]["category"] == "FileNotFoundError"


def test_cli_diagnose_human_and_json(tmp_path, capsys):
    bad = tmp_path / "bad.tl"
    bad.write_text("from tensorlang import T\nf := {X} + 1\nd := T([1])\n"
                   "r := f(d)\n")
    assert cli_main(["diagnose", str(bad)]) == 1
    out = capsys.readouterr().out
    assert "TL0301" in out
    assert cli_main(["diagnose", str(bad), "--format", "json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "TL0301"
    assert isinstance(payload["suggestions"], list)
    ok = tmp_path / "ok.tl"
    ok.write_text("from tensorlang import T\na := T(1)\n")
    assert cli_main(["diagnose", str(ok)]) == 0


def test_cli_errors_command(capsys):
    assert cli_main(["errors", "TL0301"]) == 0
    assert "TL0301" in capsys.readouterr().out
    assert cli_main(["errors", "TL9999"]) == 1
    assert cli_main(["errors"]) == 0
    assert "TL0301" in capsys.readouterr().out


def test_error_report_plain_python():
    report = _diagnose.error_report(NameError("name 'foo' is not defined"))
    assert report["code"] == "PY0000"
    suggestions = _diagnose.suggest_for(
        NameError("name 'foo' is not defined"), {"food": 1})
    assert any("food" in s for s in suggestions)
    suggestions = _diagnose.suggest_for(
        NameError("name 'sin' is not defined"), {})
    assert any("from tensorlang import sin" in s for s in suggestions)
