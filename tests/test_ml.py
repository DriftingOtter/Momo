"""v0.3 tests: fittable operators, fitted pipelines, train/test discipline."""
import numpy as np
import polars as pl
import pytest

from tensorlang import (
    P,
    T,
    TLDeclarationError,
    TLShapeError,
    TLTypeError,
    FittableOperator,
    PipelineOperator,
    impute,
    minmax,
    normalize,
    pca,
    run_source,
    standardize,
)

sklearn = pytest.importorskip("sklearn")


@pytest.fixture
def train():
    return T(np.array([
        [1.0, 10.0, 0.5],
        [2.0, 20.0, 1.5],
        [3.0, 30.0, 2.5],
        [4.0, 40.0, 3.5],
    ]))


@pytest.fixture
def test():
    return T(np.array([[5.0, 50.0, 4.5], [0.0, 0.0, -0.5]]))


def test_unfitted_application_raises(train):
    with pytest.raises(TLTypeError):
        standardize()(train).collect()
    with pytest.raises(TLTypeError):
        pca(2)(train).collect()
    with pytest.raises(TLTypeError):
        (standardize() >> pca(2))(train).collect()
    with pytest.raises(TLTypeError):
        standardize().parameters


def test_fit_returns_new_object_original_unchanged(train):
    scaler = standardize()
    fitted = scaler.fit(train)
    assert isinstance(fitted, FittableOperator)
    assert fitted.is_fitted and not scaler.is_fitted
    assert fitted is not scaler
    with pytest.raises(TLTypeError):
        scaler(train).collect()


def test_train_stats_reused_on_test(train, test):
    fitted = standardize().fit(train)
    mean = np.array([2.5, 25.0, 2.0])
    assert np.allclose(fitted.parameters["mean_"], mean)
    assert np.allclose(fitted(test).collect(), (test.collect() - mean) /
                       fitted.parameters["scale_"])
    # No leakage: parameters equal training statistics exactly.
    assert np.allclose(fitted.parameters["mean_"], train.collect().mean(axis=0))


def test_standardize_matches_sklearn(train, test):
    from sklearn.preprocessing import StandardScaler

    ref = StandardScaler().fit(train.collect())
    got = standardize().fit(train)
    assert np.allclose(got.parameters["mean_"], ref.mean_)
    assert np.allclose(got.parameters["scale_"], ref.scale_)
    assert np.allclose(got(test).collect(), ref.transform(test.collect()))


def test_minmax_matches_sklearn(train, test):
    from sklearn.preprocessing import MinMaxScaler

    ref = MinMaxScaler().fit(train.collect())
    got = minmax().fit(train)
    assert np.allclose(got(test).collect(), ref.transform(test.collect()))
    got2 = minmax(-1, 1).fit(train)
    from sklearn.preprocessing import MinMaxScaler as _MMS

    ref2 = _MMS(feature_range=(-1, 1)).fit(train.collect())
    assert np.allclose(got2(test).collect(), ref2.transform(test.collect()))
    with pytest.raises(TLTypeError):
        minmax(2, 1)


def test_normalize_matches_sklearn(train):
    from sklearn.preprocessing import Normalizer

    for norm in ("l1", "l2", "max"):
        ref = Normalizer(norm=norm).fit(train.collect())
        got = normalize(norm).fit(train)
        assert np.allclose(got(train).collect(),
                           ref.transform(train.collect())), norm
    with pytest.raises(TLTypeError):
        normalize("l0")


def test_impute_matches_sklearn():
    from sklearn.impute import SimpleImputer

    data = T(np.array([[1.0, np.nan], [np.nan, 2.0], [3.0, 4.0]]))
    for strategy in ("mean", "median", "most_frequent", "constant"):
        kwargs = {"fill_value": -1.0} if strategy == "constant" else {}
        ref = SimpleImputer(strategy=strategy, **kwargs).fit_transform(
            data.collect())
        got = impute(strategy, **kwargs).fit(data)(data).collect()
        assert np.allclose(got, ref), strategy
    with pytest.raises(TLTypeError):
        impute("nope")
    with pytest.raises(TLTypeError):
        impute("constant")


def test_impute_all_missing_column_errors():
    with pytest.raises(TLTypeError):
        impute("mean").fit(T(np.array([[np.nan], [np.nan]])))


def test_impute_string_frame():
    t = T({"g": ["a", None, "a", "b"], "x": [1.0, 2.0, None, 4.0]})
    out = impute("most_frequent").fit(t)(t).collect()
    assert out["g"].to_list() == ["a", "a", "a", "b"]
    assert out["x"].to_list() == [1.0, 2.0, 1.0, 4.0]


def test_pipeline_sequential_fit_with_nulls():
    train = T({"a": [1.0, None, 3.0, 4.0], "b": [2.0, 4.0, 6.0, 8.0]})
    test = T({"a": [None, 10.0], "b": [1.0, 2.0]})
    pipe = impute("mean") >> standardize() >> pca(2)
    assert isinstance(pipe, PipelineOperator)
    fitted = pipe.fit(train)
    assert all(s.is_fitted if isinstance(s, FittableOperator) else True
               for s in fitted.stages)
    assert not any(s.is_fitted for s in pipe.stages
                   if isinstance(s, FittableOperator))
    train_out = fitted(train).collect()
    test_out = fitted(test).collect()
    assert train_out.shape == (4, 2) and test_out.shape == (2, 2)
    assert not np.isnan(test_out["pc0"].to_numpy()).any()
    # Imputation used training means (a: (1+3+4)/3, b: 5.0).
    assert np.allclose(pipe.stages[0].fit(train).parameters["fill_"],
                       [(1 + 3 + 4) / 3, 5.0])


def test_no_silent_refit(train):
    pipe = standardize() >> pca(2)
    first = pipe.fit(train)
    other = T(np.array([[100.0, 100.0, 100.0]] * 4))
    second = pipe.fit(other)
    # pipe itself was never mutated; fitting the unfitted original on new
    # data learns new statistics, while `first` keeps training statistics.
    assert np.allclose(first.stages[0].parameters["mean_"], [2.5, 25.0, 2.0])
    assert np.allclose(second.stages[0].parameters["mean_"], [100.0] * 3)
    refit = first.fit(other)
    assert np.allclose(refit.stages[0].parameters["mean_"], [2.5, 25.0, 2.0])


def test_dense_frame_equivalence_and_names():
    frame = T({"age": [25.0, 45.0, 35.0, 55.0],
               "income": [45.0, 105.0, 75.0, 95.0]})
    matrix = T(frame.collect().select(["age", "income"]).to_numpy())
    f_frame = standardize().fit(frame)
    f_dense = standardize().fit(matrix)
    assert np.allclose(f_frame(matrix).collect(), f_dense(matrix).collect())
    out = f_frame(frame).collect()
    assert out.columns == ["age", "income"]
    assert f_frame.parameters["feature_names"] == ["age", "income"]


def test_feature_reorder_aligns_by_name():
    train = T({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
    fitted = standardize().fit(train)
    shuffled = T({"b": [20.0], "a": [2.0]})
    assert np.allclose(fitted(shuffled).collect()["a"].to_list(), [0.0])
    with pytest.raises(TLTypeError):
        fitted(T({"a": [1.0]})).collect()
    with pytest.raises(TLShapeError):
        fitted(T(np.array([[1.0]]))).collect()


def test_mixed_frame_passthrough_and_pca_rejection():
    mixed = T({"num": [1.0, 2.0, 3.0], "lab": ["x", "y", "z"]})
    out = standardize().fit(mixed)(mixed).collect()
    assert out.columns == ["num", "lab"]
    assert out["lab"].to_list() == ["x", "y", "z"]
    with pytest.raises(TLTypeError):
        pca(1).fit(mixed)


def test_constant_columns_standardize_to_zero():
    t = T(np.array([[1.0, 5.0], [1.0, 7.0], [1.0, 9.0]]))
    out = standardize().fit(t)(t).collect()
    assert np.allclose(out[:, 0], [0.0, 0.0, 0.0])
    assert np.allclose(out[:, 1], [-1.22474487, 0.0, 1.22474487])


def test_null_and_inf_fit_errors():
    with pytest.raises(TLTypeError):
        standardize().fit(T(np.array([[1.0, np.nan]])))
    with pytest.raises(TLTypeError):
        pca(1).fit(T(np.array([[1.0, np.inf], [2.0, 3.0]])))


def test_pca_correctness_and_shapes(train, test):
    from sklearn.decomposition import PCA

    ref = PCA(n_components=2).fit(train.collect())
    got = pca(2).fit(train)
    assert np.allclose(got.parameters["explained_variance_"],
                       ref.explained_variance_)
    assert got(train).collect().shape == (4, 2)
    assert got(test).collect().shape == (2, 2)
    assert list(got(test).collect().columns) == ["pc0", "pc1"]
    # Sign-agnostic: reconstruction from components matches.
    rec = (got(train).collect().to_numpy()
           @ got.parameters["components_"] + got.parameters["mean_"])
    assert np.allclose(rec, train.collect(), atol=1e-8)
    assert np.allclose(
        np.abs(got(train).collect().to_numpy()),
        np.abs(ref.transform(train.collect())), atol=1e-8)


def test_pca_validation():
    with pytest.raises(TLShapeError):
        pca(5).fit(T(np.eye(3)))
    with pytest.raises(TLShapeError):
        pca(1).fit(T(np.array([[1.0, 2.0]])))
    with pytest.raises(TLTypeError):
        pca(0)
    with pytest.raises(TLTypeError):
        pca(1.5)


def test_mixed_stateless_and_fittable_composition(train):
    pipe = (P("X") * 2) >> standardize()
    assert isinstance(pipe, PipelineOperator)
    fitted = pipe.fit(train)
    assert np.allclose(fitted(train).collect(),
                       (train.collect() * 2 - (train.collect() * 2).mean(0)) /
                       (train.collect() * 2).std(0))
    pipe2 = standardize() >> (P("Y") + 1)
    assert np.allclose(pipe2.fit(train)(train).collect(),
                       standardize().fit(train)(train).collect() + 1)


def test_partial_pipeline_reuses_fitted(train, test):
    first = standardize().fit(train)
    pipe = first >> pca(2)
    fitted = pipe.fit(train)
    assert fitted.stages[0] is first  # reused, never refit
    assert fitted(test).collect().shape == (2, 2)


def test_inplace_after_fitting(train):
    alias_target = T(train.collect().copy())
    alias = alias_target
    fitted = standardize().fit(train)
    fitted(alias_target).inplace()
    assert alias_target.collect().shape == (4, 3)
    assert alias.collect().shape == (4, 3)  # aliasing by identity
    assert np.allclose(alias_target.collect(),
                       (train.collect() - train.collect().mean(0)) /
                       train.collect().std(0))


def test_pca_inplace_changes_representation():
    from tensorlang import TLMutationError  # noqa: F401

    t = T(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    pca(1).fit(t)(t).inplace()
    assert t.kind == "frame"
    assert t.fields == ["pc0"]


def test_snapshot_preserved_across_fit(train):
    X = P("X")
    pending = (X + 1)(train)
    standardize().fit(train)
    assert pending.collect().tolist() == (train.collect() + 1).tolist()


def test_failed_fit_preserves_pipeline():
    pipe = standardize() >> pca(5)
    bad = T(np.eye(3))
    with pytest.raises(TLShapeError):
        pipe.fit(bad)
    assert not any(s.is_fitted for s in pipe.stages
                   if isinstance(s, FittableOperator))


def test_fit_effect_class_and_tl_program():
    with pytest.raises(TLDeclarationError):
        run_source("from tensorlang import T, standardize\n"
                   "t = T([1.0])\n"
                   "f := standardize().fit(t)\n", filename="m.tl")
    ns = run_source(
        "from tensorlang import T, standardize\n"
        "t = T([[1.0, 2.0], [3.0, 4.0]])\n"
        "scaler := standardize()\n"
        "fitted = scaler.fit(t)\n"
        "out = fitted(t)\n", filename="m.tl")
    assert ns["out"].collect().shape == (2, 2)
    assert isinstance(ns["scaler"], FittableOperator)


def test_base_operator_fit_helpful_error():
    with pytest.raises(TLTypeError):
        (P("X") + 1).fit(T([1.0]))


def test_series_and_vector_inputs():
    s = T({"x": [1.0, 2.0, 3.0, 4.0]})
    fitted = standardize().fit(s.x)
    out = fitted(s.x).collect()
    assert out.name == "x"
    assert np.allclose(out.to_list(), [-1.34164079, -0.4472136, 0.4472136, 1.34164079])
    v = T([1.0, 2.0, 3.0, 4.0])
    assert np.allclose(standardize().fit(v)(v).collect(), out.to_list())
