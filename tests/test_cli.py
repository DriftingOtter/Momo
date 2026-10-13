"""CLI tests: graph/explain/profile/doctor/bench wiring via main(argv)."""
from pathlib import Path

from tensorlang.cli import main as cli_main

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_graph_var_and_dot(capsys):
    assert cli_main(["graph", str(EXAMPLES / "core_demo.tl"),
                     "--var", "pipeline", "--allow-effects"]) == 0
    out = capsys.readouterr().out
    assert "Param" in out
    assert cli_main(["graph", str(EXAMPLES / "core_demo.tl"),
                     "--var", "pipeline", "--dot", "--allow-effects"]) == 0
    assert "digraph" in capsys.readouterr().out


def test_graph_all_bindings(capsys):
    assert cli_main(["graph", str(EXAMPLES / "core_demo.tl"),
                     "--allow-effects"]) == 0
    assert "pipeline" in capsys.readouterr().out


def test_graph_unknown_var(capsys):
    assert cli_main(["graph", str(EXAMPLES / "core_demo.tl"),
                     "--var", "nosuchvar"]) == 1


def test_explain_and_profile(capsys):
    assert cli_main(["explain", str(EXAMPLES / "core_demo.tl"), "result",
                     "--allow-effects"]) == 0
    assert "logical plan" in capsys.readouterr().out
    assert cli_main(["profile", str(EXAMPLES / "core_demo.tl"), "result",
                     "--allow-effects"]) == 0
    assert "profile:" in capsys.readouterr().out
    assert cli_main(["explain", str(EXAMPLES / "core_demo.tl"), "nope"]) == 1
    assert cli_main(["profile", str(EXAMPLES / "core_demo.tl"), "datasets"]) == 1


def test_doctor(capsys):
    assert cli_main(["doctor"]) == 0
    out = capsys.readouterr().out
    for token in ("tensorlang", "python", "numpy", "polars", "dask", "zarr"):
        assert token in out


def test_bench_help_wired():
    import pytest

    with pytest.raises(SystemExit) as exc:
        cli_main(["bench", "--help"])
    assert exc.value.code == 0


def test_introspect_effect_gate(tmp_path, capsys):
    src = tmp_path / "eff.tl"
    src.write_text("from tensorlang import T\n"
                   "t = T([1.0])\n"
                   "t.write_parquet('x.parquet')\n")
    assert cli_main(["graph", str(src), "--var", "t"]) == 1
    err = capsys.readouterr().err
    assert "--allow-effects" in err
    assert ".write_parquet" in err
    clean = tmp_path / "clean.tl"
    clean.write_text("from tensorlang import T\na := T([1, 2])\n")
    assert cli_main(["graph", str(clean), "--var", "a"]) == 0


def test_bench_all_skipped_touches_nothing(capsys):
    assert cli_main(["bench", "--sizes", "small",
                     "--skip-dense", "--skip-structured", "--skip-pipeline",
                     "--skip-memory", "--skip-autodiff",
                     "--skip-chunked"]) == 0
    assert "no workloads selected" in capsys.readouterr().out
