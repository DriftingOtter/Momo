"""abstract() tests: structured summaries for operators, tensors, lazies."""
import json

import polars as pl
import pytest

from tensorlang import P, T, grad, pca, standardize


def _jsonable(value):
    return json.loads(json.dumps(value))


def test_stateless_operator_abstract():
    op = (P("X") ** 2) >> (P("Y") + 1)
    summary = _jsonable(op.abstract())
    assert summary["kind"] == "stateless"
    assert summary["params"] == ["X"]
    assert summary["nodes"] > 0
    assert isinstance(summary["backends"], dict)


def test_fittable_and_fitted_abstract():
    scaler = standardize()
    assert scaler.abstract()["kind"] == "fittable"
    assert scaler.abstract()["estimator"] == "standardize"
    fitted = scaler.fit(T([[1.0, 2.0], [3.0, 4.0]]))
    summary = _jsonable(fitted.abstract())
    assert summary["kind"] == "fitted"
    assert summary["n_features"] == 2
    assert "mean_" in summary["learned"]


def test_pipeline_abstract():
    pipe = standardize() >> pca(2)
    summary = _jsonable(pipe.abstract())
    assert summary["kind"] == "pipeline"
    assert [s["kind"] for s in summary["stages"]] == ["fittable", "fittable"]
    fitted = pipe.fit(T([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    assert [s["kind"] for s in fitted.abstract()["stages"]] == ["fitted", "fitted"]


def test_gradient_abstract():
    d = grad((P("X") ** 2).mean())
    summary = _jsonable(d.abstract())
    assert summary["kind"] == "gradient"
    assert summary["wrt"] == ["X"]


def test_tensor_abstract():
    assert _jsonable(T([1, 2, 3]).abstract())["shape"] == [3]
    frame = T({"a": [1.0], "b": ["x"]})
    summary = _jsonable(frame.abstract())
    assert summary["kind"] == "frame"
    assert summary["fields"] == ["a", "b"]
    assert summary["schema"]["a"] == "Float64"
    assert summary["provenance"] == "literal"
    assert T([1]).abstract()["version"] == 0


def test_scan_abstract_lazy(tmp_path):
    src = tmp_path / "s.parquet"
    pl.DataFrame({"a": [1]}).write_parquet(src)
    s = T.scan_parquet(str(src))
    summary = _jsonable(s.abstract())
    assert summary["kind"] == "scan"
    assert summary["shape"] is None
    assert summary["fields"] == ["a"]


def test_lazy_abstract():
    v = T([1.0, 2.0])
    plain = (v * 2)
    summary = _jsonable(plain.abstract())
    assert summary["target"] is None
    assert summary["cached"] is False
    applied = (P("X") + 1)(v)
    assert applied.abstract()["target"]["kind"] == "dense"
    applied.collect()
    assert applied.abstract()["cached"] is True
