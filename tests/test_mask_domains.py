"""M1 tests: structured mask composition row-domain correctness (F-B4).

`and`/`or` over one logical row domain fuse into a single relational
query (deterministic, correct). Masks from provably different row
domains are rejected with `TL0512` — equal length is not proof of
equal row identity. Unanalyzable shapes keep the historical path.
"""
import pytest

from tensorlang import P, T, TLShapeError, run_source

JOIN_SRC = (
    "from tensorlang import T\n"
    "import polars as pl\n"
    "left = T(pl.DataFrame({'k': [0, 1, 2, 3, 4, 5], "
    "'a': [10.0, 1.0, 10.0, 1.0, 10.0, 1.0]}))\n"
    "right = T(pl.DataFrame({'k': [0, 1, 2, 3, 4, 5], "
    "'b': [1.0, 10.0, 10.0, 1.0, 1.0, 10.0]}))\n"
    "j = left.join(right, on='k')\n"
)


def run(src):
    return run_source(src, filename="m.tl")


def test_six_row_combined_mask_exact_records():
    ns = run(JOIN_SRC + "both := (({D}.a > 5.0) & ({D}.b > 5.0))\n")
    for _ in range(5):
        got = ns["both"](ns["j"]).collect()
        assert got.to_list().count(True) == 1
    rows = ns["j"][ns["both"](ns["j"])].collect().sort("k")
    assert rows["k"].to_list() == [2]
    assert rows["a"].to_list() == [10.0]
    assert rows["b"].to_list() == [10.0]


def test_or_and_not_family():
    ns = run(JOIN_SRC + "either := (({D}.a > 5.0) | ({D}.b > 5.0))\n"
             "neg := ~(({D}.a > 5.0) & ({D}.b > 5.0))\n")
    for _ in range(3):
        assert int(ns["either"](ns["j"]).collect().sum()) == 5
        assert int(ns["neg"](ns["j"]).collect().sum()) == 5


def test_reversed_operands_and_nesting():
    ns = run(JOIN_SRC + "rev := (({D}.b > 5.0) & ({D}.a > 5.0))\n"
             "nest := ((({D}.a > 5.0) & ({D}.b > 5.0)) | ({D}.k > 4))\n")
    for _ in range(3):
        assert int(ns["rev"](ns["j"]).collect().sum()) == 1
        assert int(ns["nest"](ns["j"]).collect().sum()) == 2


def test_reordered_input_rows():
    ns = run(JOIN_SRC + "both := (({D}.a > 5.0) & ({D}.b > 5.0))\n")
    rev = ns["j"].collect().sort("k", descending=True)
    got = ns["both"](T(rev)).collect()
    assert int(got.sum()) == 1


def test_incompatible_domains_rejected():
    import polars as pl

    left = T(pl.DataFrame({"a": [10.0] * 6}))
    right = T(pl.DataFrame({"b": [10.0] * 6}))
    ns = run("cross := (({D}.a > 5.0) & ({E}.b > 5.0))\n")
    with pytest.raises(TLShapeError) as exc_info:
        ns["cross"](left, right).collect()
    assert exc_info.value.code == "TL0512"


def test_equal_length_proves_nothing():
    # Same length, different sources: rejected, not zipped.
    import polars as pl

    ns = run(JOIN_SRC)
    other = T(pl.DataFrame({"k": [0, 1, 2, 3, 4, 5],
                            "c": [10.0, 1.0, 1.0, 1.0, 1.0, 1.0]}))
    D, E = P("D"), P("E")
    cross = ((D.a > 5.0) & (E.c > 5.0))
    with pytest.raises(TLShapeError) as exc_info:
        cross(ns["j"], other).collect()
    assert exc_info.value.code == "TL0512"


def test_null_boolean_combination():
    # Kleene logic preserved through fusion: F&~=F, None&T=None, F&None=F.
    import polars as pl

    df = T(pl.DataFrame({"a": [10.0, None, 1.0], "b": [1.0, 10.0, None]}))
    D = P("D")
    out = ((D.a > 5.0) & (D.b > 5.0))(df).collect().to_list()
    assert out == [False, None, False]


def test_filter_after_join_exact():
    ns = run(JOIN_SRC + "both := (({D}.a > 5.0) & ({D}.b > 5.0))\n"
             "sel = j[both(j)]\n")
    for _ in range(3):
        assert ns["sel"].collect().height == 1


def test_unanalyzable_structured_combination_rejected():
    # Where-built masks defeat row-domain analysis: rejection, never a
    # silent positional zip (review gate).
    import polars as pl

    from tensorlang import where as _where

    ns = run("from tensorlang import T\n"
             "import polars as pl\n"
             "a = T(pl.DataFrame({'v': [10.0, 1.0, 10.0, 1.0]}))\n"
             "b = T(pl.DataFrame({'v': [1.0, 10.0, 1.0, 10.0]}))\n")
    D = P("D")
    m1 = _where(D.v > 5.0, T([True] * 4), T([False] * 4))(ns["a"])
    m2 = _where(D.v > 5.0, T([True] * 4), T([False] * 4))(ns["b"])
    with pytest.raises(TLShapeError) as exc_info:
        (m1 & m2).collect()
    assert exc_info.value.code == "TL0512"


def test_dense_positional_preserved():
    import numpy as np

    assert (T(np.array([True, False])) & T(np.array([True, True]))
            ).collect().tolist() == [True, False]
    assert (T(np.array([True, False])) & True).collect().tolist() == [
        True, False]


def test_predicate_after_grouping():
    ns = run("from tensorlang import T\n"
             "import polars as pl\n"
             "df = T(pl.DataFrame({'k': [0,0,1,1,2,2], "
             "'v': [10.0, 1.0, 10.0, 1.0, 10.0, 1.0]}))\n"
             "g = df.v.sum(by=df.k)\n"
             "m := (({D}.v > 10.0) & ({D}.v < 12.0))\n")
    out = ns["m"](ns["g"])
    from tensorlang import Lazy

    assert isinstance(out, Lazy)
    assert sorted(out.collect().to_list()) == [True, True, True]
    assert "materialized fallback" not in out.explain()


def test_scalar_bool_and_reuse():
    ns = run(JOIN_SRC + "both := (({D}.a > 5.0) & ({D}.b > 5.0))\n")
    applied = ns["both"](ns["j"])
    from tensorlang import Lazy

    assert isinstance(applied, Lazy)
    first = applied.collect().to_list()
    assert applied.collect().to_list() == first
    assert first.count(True) == 1
