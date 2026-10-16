"""Phase 7 tests: complete example programs end to end."""
import sys
from pathlib import Path

import numpy as np
import pytest

from tensorlang import run_file, run_source

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_core_demo_runs_with_expected_values(capsys):
    ns = run_file(EXAMPLES / "core_demo.tl")
    out = capsys.readouterr().out
    assert "[ 2 10 24 40 60]" in out
    assert "[149 405]" in out
    assert "[ 1  5 12 20 30]" in out
    assert "[  6  30 149 405 905]" in out
    assert "[149 405 905]" in out
    assert "East" in out and "170" in out
    assert "West" in out and "325" in out
    assert "125/3" in out
    # bindings hold the final program state
    assert ns["vector"].collect().tolist() == [149, 405, 905]
    assert ns["area"].collect() == 125 / 3 or str(ns["area"].collect()) == "125/3"


def test_definitions_masks_mutation_structured_symbolic_combined():
    ns = run_source(
        "from tensorlang import T, sin, diff\n"
        "data := T([1.0, 2.0, 3.0, 4.0])\n"
        "big := {X} > 2.5\n"
        "sq := {X} ** 2\n"
        "picked = data[big(data)]\n"
        "total = sq(picked).sum()\n"
        "assert total.collect() == 25.0\n"
        "sq(data).inplace()\n"
        "assert data.collect().tolist() == [1.0, 4.0, 9.0, 16.0]\n"
        "df := diff(sin({X}))\n"
        "assert abs(df(T(0.0)).collect() - 1.0) < 1e-12\n",
        filename="combo.tl",
    )
    assert ns["data"].collect().tolist() == [1.0, 4.0, 9.0, 16.0]


def test_structured_pipeline_in_tl():
    ns = run_source(
        "from tensorlang import T\n"
        "sales := T({'a': [1, 2, 3, 4], 'g': ['x', 'x', 'y', 'y']})\n"
        "pos := {D}.a > 1\n"
        "keep = sales[pos(sales)]\n"
        "keep['b'] = keep.a * 10\n"
        "totals = keep.b.sum(by=keep.g)\n"
        "rows = totals.collect().sort('g')\n"
        "assert rows['b'].to_list() == [20, 70]\n",
        filename="struct.tl",
    )
    assert ns["rows"]["b"].to_list() == [20, 70]


def test_run_file_missing_and_wrong_suffix(tmp_path):
    with pytest.raises(Exception):
        run_file(tmp_path / "nope.tl")
    py = tmp_path / "x.py"
    py.write_text("x = 1")
    with pytest.raises(Exception):
        run_file(py)


def test_v02_sales_pipeline_example(tmp_path):
    import shutil

    import polars as pl

    repo_examples = Path(__file__).resolve().parent.parent / "examples"
    workdir = tmp_path / "demo"
    (workdir / "data").mkdir(parents=True)
    shutil.copy(repo_examples / "sales_pipeline.tl", workdir / "sales_pipeline.tl")
    # Same generator/seed as examples/make_demo_data.py, fewer rows.
    sys.path.insert(0, str(repo_examples.parent / "benchmarks"))
    try:
        from datasets import make_sales

        make_sales(workdir / "data" / "sales-demo.parquet", 20_000, seed=20260201)
    finally:
        sys.path.remove(str(repo_examples.parent / "benchmarks"))
    ns = run_file(workdir / "sales_pipeline.tl")
    got = ns["summary"].collect().sort("region")
    expected = (
        pl.scan_parquet(workdir / "data" / "sales-demo.parquet")
        .filter((pl.col("amount") > 100) & (pl.col("status") == "completed"))
        .with_columns((pl.col("amount") - pl.col("fee")).alias("net"))
        .group_by("region")
        .agg(pl.col("net").sum())
        .sort("region")
        .collect()
    )
    assert got.equals(expected)
    # File round-trip: grouped floating-point sums may differ in the
    # last ulp by summation order (documented, not frozen —
    # docs/COMPATIBILITY.md §4), so compare the persisted copy with
    # tolerance here while the in-memory check above stays exact.
    import numpy as _np

    _back = pl.read_parquet(workdir / "data" / "revenue-demo.parquet")
    assert _back.columns == got.columns
    assert _back["region"].to_list() == got["region"].to_list()
    assert _np.allclose(_back["net"].to_numpy(), got["net"].to_numpy(),
                        rtol=1e-9, atol=1e-9)
    assert "polars-lazy" in ns["summary"].explain()
    assert ns["summary"].profile().lowered is True


def test_v03_ml_pipeline_example():
    import shutil
    import tempfile
    from pathlib import Path as _Path

    import numpy as np
    import polars as pl

    repo_examples = _Path(__file__).resolve().parent.parent / "examples"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = _Path(tmp)
        shutil.copy(repo_examples / "ml_pipeline.tl", tmp_path / "ml_pipeline.tl")
        ns = run_file(tmp_path / "ml_pipeline.tl")
    # Shapes from the reference program.
    assert ns["train_result"].collect().shape == (4, 2)
    assert ns["test_result"].collect().shape == (2, 2)
    assert ns["test"].shape == (2, 2)  # explicitly mutated in place
    assert list(ns["test"].fields) == ["pc0", "pc1"]
    # Matches a NumPy/scikit-learn reference computed from TRAIN only.
    train = np.array([
        [25.0, 45000.0, 6.0], [45.0, 105000.0, np.nan],
        [35.0, 75000.0, 8.0], [55.0, 95000.0, 9.0],
    ])
    test = np.array([[25.0, 45000.0, 6.0], [45.0, 105000.0, 9.0]])
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        pytest.skip("scikit-learn cross-check needs sklearn")
    imp = SimpleImputer(strategy="mean").fit(train)
    std = StandardScaler().fit(imp.transform(train))
    ref = PCA(n_components=2).fit(std.transform(imp.transform(train)))
    got = ns["test_result"].collect().to_numpy()
    assert np.allclose(
        np.abs(got), np.abs(ref.transform(std.transform(imp.transform(test)))),
        atol=1e-8)
    # SVD singular values match SciPy exactly.
    assert np.allclose(ns["S"].collect(), [9.52551809, 0.51430058])
    # Execution plan is displayed and names the fitted stages.
    assert "pca(fitted)" in ns["train_result"].explain()


def test_v04_autodiff_example():
    from pathlib import Path as _Path

    repo_examples = _Path(__file__).resolve().parent.parent / "examples"
    ns = run_file(repo_examples / "autodiff.tl")
    assert ns["gradient"].collect().tolist() == pytest.approx([2 / 3, 2 / 3, 2.0])
    assert ns["value"].collect() == pytest.approx(11 / 3)
    assert "grad(wrt=['P'])" in ns["d_loss"].explain()


def test_v04_gradient_descent_example():
    from pathlib import Path as _Path

    repo_examples = _Path(__file__).resolve().parent.parent / "examples"
    ns = run_file(repo_examples / "gradient_descent.tl")
    assert np.allclose(ns["w"].collect(), [3.0, -2.0], atol=1e-4)


def test_v05_chunked_demo_example(tmp_path):
    import shutil
    from pathlib import Path as _Path

    import zarr

    repo_examples = _Path(__file__).resolve().parent.parent / "examples"
    workdir = tmp_path / "chunked"
    (workdir / "data").mkdir(parents=True)
    for name in ("chunked_demo.tl", "make_chunked_data.py"):
        shutil.copy(repo_examples / name, workdir / name)
    import subprocess as _sp
    import sys as _sys

    _sp.run([_sys.executable, str(workdir / "make_chunked_data.py"),
             "--rows", "2000", "--cols", "4",
             "--out", str(workdir / "data" / "large-demo.zarr")],
            check=True, capture_output=True)
    ns = run_file(workdir / "chunked_demo.tl")
    ref = zarr.open_array(workdir / "data" / "large-demo.zarr", mode="r")[:]
    assert ns["average"].collect() == pytest.approx(
        float((np.sin(ref) + ref**2).mean()))
    out = zarr.open_array(workdir / "data" / "processed-demo.zarr", mode="r")
    assert out.shape == (2000, 4)
    assert np.allclose(out[:], np.sin(ref) + ref**2)
    assert "dask-chunked" in ns["result"].explain()


def test_v05_chunk_planning_demo(tmp_path):
    import shutil
    from pathlib import Path as _Path

    import zarr

    repo_examples = _Path(__file__).resolve().parent.parent / "examples"
    workdir = tmp_path / "planning"
    (workdir / "data").mkdir(parents=True)
    for name in ("chunk_planning.tl", "make_chunked_data.py"):
        shutil.copy(repo_examples / name, workdir / name)
    import subprocess as _sp
    import sys as _sys

    _sp.run([_sys.executable, str(workdir / "make_chunked_data.py"),
             "--rows", "4000", "--cols", "4", "--chunk-rows", "1000",
             "--chunk-rows-b", "2000",
             "--out", str(workdir / "data" / "large-demo.zarr"),
             "--out-b", str(workdir / "data" / "other-demo.zarr")],
            check=True, capture_output=True)
    ns = run_file(workdir / "chunk_planning.tl")
    assert "rechunk" in ns["m"].explain()
    out = zarr.open_array(workdir / "data" / "planned-demo.zarr", mode="r")
    assert out.shape == (4000, 4)


def test_v05_memory_budget_demo(tmp_path):
    import shutil
    from pathlib import Path as _Path

    import zarr

    repo_examples = _Path(__file__).resolve().parent.parent / "examples"
    workdir = tmp_path / "budget"
    (workdir / "data").mkdir(parents=True)
    for name in ("memory_budget.tl", "make_chunked_data.py"):
        shutil.copy(repo_examples / name, workdir / name)
    import subprocess as _sp
    import sys as _sys

    _sp.run([_sys.executable, str(workdir / "make_chunked_data.py"),
             "--rows", "4000", "--cols", "4", "--chunk-rows", "1000",
             "--out", str(workdir / "data" / "large-demo.zarr")],
            check=True, capture_output=True)
    ns = run_file(workdir / "memory_budget.tl")
    ref = zarr.open_array(workdir / "data" / "large-demo.zarr", mode="r")[:]
    assert ns["average"].collect() == pytest.approx(
        float((ref**2 + ref).mean()))
    out = zarr.open_array(workdir / "data" / "planned-budget.zarr", mode="r")
    assert out.shape == (4000, 4)
    assert np.allclose(out[:], ref**2 + ref)
    assert "resources" in ns["m"].explain()
    from tensorlang import clear_chunk_policy as _clear
    from tensorlang import set_chunk_policy as _set

    _set(memory_budget_bytes=2800000, max_workers=2)
    try:
        assert ns["average"].profile().budget_mb == pytest.approx(2.8)
    finally:
        _clear()


def test_v05_reliable_execution_demo(tmp_path):
    import shutil
    from pathlib import Path as _Path

    import zarr

    repo_examples = _Path(__file__).resolve().parent.parent / "examples"
    workdir = tmp_path / "reliable"
    (workdir / "data").mkdir(parents=True)
    for name in ("reliable_execution.tl", "make_chunked_data.py"):
        shutil.copy(repo_examples / name, workdir / name)
    import subprocess as _sp
    import sys as _sys

    _sp.run([_sys.executable, str(workdir / "make_chunked_data.py"),
             "--rows", "4000", "--cols", "4", "--chunk-rows", "1000",
             "--out", str(workdir / "data" / "large-demo.zarr")],
            check=True, capture_output=True)
    ns = run_file(workdir / "reliable_execution.tl")
    ref = zarr.open_array(workdir / "data" / "large-demo.zarr", mode="r")[:]
    out = zarr.open_array(workdir / "data" / "reliable-demo.zarr", mode="r")
    assert out.shape == (4000, 4)
    assert np.allclose(out[:], ref**2 + ref)
    assert "staged" in ns["m"].describe_write(
        str(workdir / "data" / "reliable-demo.zarr"), overwrite=True)


def test_v05_release_check_workflow(tmp_path):
    import shutil
    from pathlib import Path as _Path

    import zarr

    repo_examples = _Path(__file__).resolve().parent.parent / "examples"
    workdir = tmp_path / "release"
    (workdir / "data").mkdir(parents=True)
    for name in ("release_check.tl", "make_chunked_data.py"):
        shutil.copy(repo_examples / name, workdir / name)
    import subprocess as _sp
    import sys as _sys

    _sp.run([_sys.executable, str(workdir / "make_chunked_data.py"),
             "--rows", "4000", "--cols", "4", "--chunk-rows", "1000",
             "--out", str(workdir / "data" / "large-demo.zarr")],
            check=True, capture_output=True)
    ns = run_file(workdir / "release_check.tl")
    ref = zarr.open_array(workdir / "data" / "large-demo.zarr", mode="r")[:]
    assert ns["average"] == pytest.approx(float((ref**2 + ref).mean()))
    out = zarr.open_array(workdir / "data" / "release-check.zarr", mode="r")
    assert out.shape == (4000, 4)
    assert np.allclose(out[:], ref**2 + ref)
