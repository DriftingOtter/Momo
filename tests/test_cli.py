"""CLI tests: graph/explain/profile/doctor/bench wiring via main(argv)."""
from pathlib import Path

from tensorlang.cli import main as cli_main

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_graph_var_and_dot(capsys):
    assert cli_main(["graph", str(EXAMPLES / "core_demo.tl"),
                     "--var", "pipeline"]) == 0
    out = capsys.readouterr().out
    assert "Param" in out
    assert cli_main(["graph", str(EXAMPLES / "core_demo.tl"),
                     "--var", "pipeline", "--dot"]) == 0
    assert "digraph" in capsys.readouterr().out


def test_graph_all_bindings(capsys):
    assert cli_main(["graph", str(EXAMPLES / "core_demo.tl")]) == 0
    assert "pipeline" in capsys.readouterr().out


def test_graph_unknown_var(capsys):
    assert cli_main(["graph", str(EXAMPLES / "core_demo.tl"),
                     "--var", "nosuchvar"]) == 1


def test_explain_and_profile(capsys):
    assert cli_main(["explain", str(EXAMPLES / "core_demo.tl"), "result"]) == 0
    assert "logical plan" in capsys.readouterr().out
    assert cli_main(["profile", str(EXAMPLES / "core_demo.tl"), "result"]) == 0
    assert "profile:" in capsys.readouterr().out
    assert cli_main(["explain", str(EXAMPLES / "core_demo.tl"), "nope"]) == 1
    assert cli_main(["profile", str(EXAMPLES / "core_demo.tl"), "datasets"]) == 1


def test_doctor(capsys):
    assert cli_main(["doctor"]) == 0
    out = capsys.readouterr().out
    for token in ("tensorlang", "python", "numpy", "polars"):
        assert token in out


def test_bench_help_wired():
    import pytest

    with pytest.raises(SystemExit) as exc:
        cli_main(["bench", "--help"])
    assert exc.value.code == 0
