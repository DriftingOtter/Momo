"""Application-level tests for the experiment pipeline.

Exercises reusable operators, composition, structured transforms, dense
outputs, fitting, laziness, errors, imports, repetition, and
determinism using ordinary pytest (no custom framework).
"""
import os

import numpy as np
import pytest

from tensorlang import P, T, grad, not_null, run_file, solve, standardize

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = APP + "/data"


@pytest.fixture(scope="module")
def stages():
    return {
        "cal": run_file(f"{APP}/operators/calibration.tl")["calibrate"],
        "tr": run_file(f"{APP}/operators/transforms.tl"),
    }


@pytest.fixture(scope="module")
def small():
    import polars as pl

    return T(pl.DataFrame({
        "experiment_id": [0, 0, 1, 1],
        "meas": [100.0, 102.0, 200.0, 204.0],
        "temp": [20.0, 21.0, 22.0, 23.0],
    }))


def test_reusable_operator(stages):
    out = stages["cal"](T([100.0]), T(1.02), T(0.5)).collect()
    assert out.tolist() == [102.5]


def test_composed_operators(stages):
    zscore, cal = stages["tr"]["zscore"], stages["cal"]
    out = zscore(cal(T([100.0, 200.0]), T(2.0), T(1.0)),
                 T(150.0), T(50.0)).collect()
    assert np.allclose(out, (np.array([200.0, 400.0]) + 1.0 - 150.0) / 50.0)


def test_overlapping_params_stay_scoped(stages):
    # calibrate uses {M},{G},{O}; zscore uses {X},{M},{S}: shared names
    # across files must not collide (uid-scoped).
    assert stages["cal"].uid != stages["tr"]["zscore"].uid
    out = stages["tr"]["zscore"](
        stages["cal"](T([10.0]), T(1.0), T(0.0)), T(5.0), T(5.0)).collect()
    assert out.tolist() == [1.0]


def test_structured_transform(small):
    assert small[small.meas > 150.0].collect().height == 2
    grouped = small.meas.mean(by=small.experiment_id).collect()
    assert grouped.sort("experiment_id")["meas"].to_list() == [101.0, 202.0]


def test_dense_output(small):
    arr = small.to_dense(columns=["meas", "temp"]).collect()
    assert arr.shape == (4, 2) and str(arr.dtype) == "float64"


def test_fitted_transform(small):
    arr = small.to_dense(columns=["meas", "temp"]).collect()
    fitted = standardize().fit(T(arr))
    out = np.asarray(fitted(T(arr)).collect())
    assert abs(out.mean()) < 1e-9
    # Frozen reuse on new data.
    out2 = np.asarray(fitted(T(arr + 1.0)).collect())
    assert out2.shape == out.shape


def test_lazy_execution(small):
    lazy = small.meas.mean(by=small.experiment_id)
    assert not lazy._has_cached
    lazy.collect()
    assert lazy._has_cached


def test_error_handling():
    A, B, X = P("A"), P("B"), P("X")
    with pytest.raises(Exception) as exc:
        ((A @ B) + 1)(T(np.ones((2, 3))), T(np.ones((4, 5)))).collect()
    assert getattr(exc.value, "code", "") == "TL0502"
    with pytest.raises(Exception) as exc:
        grad(X.sum(axis=9))(T(np.ones(3))).collect()
    assert getattr(exc.value, "code", "") == "TL0511"


def test_imported_modules(stages):
    # Same file loaded twice yields independent namespaces.
    a = run_file(f"{APP}/operators/calibration.tl")
    b = run_file(f"{APP}/operators/calibration.tl")
    assert a["calibrate"] is not b["calibrate"]
    assert a["calibrate"].uid != b["calibrate"].uid


def test_repeated_application(stages):
    cal = stages["cal"]
    for args in [([1.0], [1.0], [0.0]), ([2.0], [3.0], [1.0])]:
        got = cal(*[T(np.array(a)) for a in args]).collect()
        assert np.allclose(got, args[0][0] * args[1][0] + args[2][0])


def test_deterministic_outputs(small):
    first = small.meas.mean(by=small.experiment_id).collect()["meas"].to_list()
    second = small.meas.mean(by=small.experiment_id).collect()["meas"].to_list()
    assert first == second == [101.0, 202.0]
