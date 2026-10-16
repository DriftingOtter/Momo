"""App C tests: persistence, statistics, repeatability (public API only)."""
import os

import numpy as np
import zarr

from tensorlang import T, run_file

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(APP, "data")


def run_main():
    return run_file(os.path.join(APP, "main.tl"))


def test_output_persisted_with_shape():
    run_main()
    out = zarr.open_array(os.path.join(DATA, "processed.zarr"), mode="r")
    assert out.shape == (8000, 8000)
    assert str(out.dtype) == "float64"


def test_global_mean():
    run_main()
    out = zarr.open_array(os.path.join(DATA, "processed.zarr"), mode="r")
    assert abs(float(out[:100, :100].mean()) - 1.0) < 0.05
    ref = zarr.open_array(os.path.join(DATA, "processed_ref.zarr"), mode="r")
    assert np.allclose(out[:500, :500], ref[:500, :500], rtol=1e-12)


def test_repeat_execution():
    run_main()
    first = zarr.open_array(os.path.join(DATA, "processed.zarr"), mode="r")[:200, :200].copy()
    run_main()
    second = zarr.open_array(os.path.join(DATA, "processed.zarr"), mode="r")[:200, :200]
    assert np.array_equal(first, second)


def test_unsupported_op_rejected():
    from tensorlang import TLUnsupportedError

    src = T.scan_zarr(os.path.join(DATA, "source.zarr"))
    with __import__("pytest").raises(TLUnsupportedError):
        src.sort().collect()
