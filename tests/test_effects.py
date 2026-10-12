"""Priority 1 tests: formal declarative / runtime / effectful classification.

``:=`` accepts declarative expressions only (literals, operators,
compositions, symbolic constructors, lazy scan descriptors).  Runtime
applications and effectful I/O must use ``=``.  Two layers enforce this:
the syntactic classifier (no execution, exact locations) and the runtime
value check (pending computations, I/O provenance).
"""
import pytest

from tensorlang import P, T, TLDeclarationError, diff, join, run_source, sin
from tensorlang.frontend import compiler as _compiler


def run(src):
    return _compiler.run_source(src, filename="sem.tl")


def test_scan_descriptor_is_declarative(tmp_path):
    ns = run(
        "from tensorlang import T\n"
        f"s := T.scan_parquet({str(tmp_path / 'nope.parquet')!r})\n"
    )
    assert ns["s"].kind == "scan"
    assert ns["s"].provenance == "descriptor"
    # Declaring a scan reads nothing: the file need not even exist.


def test_scan_csv_descriptor(tmp_path):
    ns = run(
        "from tensorlang import T\n"
        f"s := T.scan_csv({str(tmp_path / 'd.csv')!r})\n"
    )
    assert ns["s"].kind == "scan"


def test_read_under_declare_rejected(tmp_path):
    with pytest.raises(TLDeclarationError) as exc_info:
        run(
            "from tensorlang import T\n"
            f"d := T.read({str(tmp_path / 'd.parquet')!r})\n"
        )
    assert "sem.tl:2" in str(exc_info.value)
    assert "effectful" in str(exc_info.value)


def test_write_under_declare_rejected():
    with pytest.raises(TLDeclarationError) as exc_info:
        run(
            "from tensorlang import T\n"
            "t := T({'a': [1]})\n"
            "p := t.write_parquet('/tmp/x.parquet')\n"
        )
    assert "effectful" in str(exc_info.value)


def test_collect_under_declare_rejected():
    with pytest.raises(TLDeclarationError) as exc_info:
        run("from tensorlang import T\nt := T([1, 2])\nc := t.collect()\n")
    assert "runtime" in str(exc_info.value)


def test_inplace_under_declare_rejected():
    with pytest.raises(TLDeclarationError) as exc_info:
        run(
            "from tensorlang import T\n"
            "t := T([1])\n"
            "f := {X} + 1\n"
            "r := f(t).inplace()\n"
        )
    assert "effectful" in str(exc_info.value)


def test_application_under_declare_rejected_with_location():
    with pytest.raises(TLDeclarationError) as exc_info:
        run("from tensorlang import T\nf := {X} + 1\nd := T([1])\nr := f(d)\n")
    assert "sem.tl:4" in str(exc_info.value)


def test_effectful_paths_work_with_equals(tmp_path):
    import polars as pl

    src = tmp_path / "d.parquet"
    pl.DataFrame({"a": [1, 2]}).write_parquet(src)
    out = tmp_path / "o.parquet"
    ns = run(
        "from tensorlang import T\n"
        f"d = T.read({str(src)!r})\n"
        f"p = d.write_parquet({str(out)!r})\n"
    )
    assert ns["d"].provenance == "io"
    assert ns["p"] == str(out)
    assert pl.read_parquet(out).equals(pl.DataFrame({"a": [1, 2]}))


def test_symbolic_constructors_are_declarative():
    ns = run(
        "from tensorlang import T, diff, integrate\n"
        "f := {X}**2\n"
        "df := diff(f)\n"
        "area := integrate({X}**2, 0, 5)\n"
    )
    assert ns["df"](T(3)).collect() == 6
    assert str(ns["area"].collect()) == "125/3"


def test_relational_application_is_runtime(tmp_path):
    import polars as pl

    a = tmp_path / "a.parquet"
    b = tmp_path / "b.parquet"
    pl.DataFrame({"k": [1, 2], "x": [1, 2]}).write_parquet(a)
    pl.DataFrame({"k": [2, 3], "y": [1, 2]}).write_parquet(b)
    with pytest.raises(TLDeclarationError):
        run(
            "from tensorlang import T, join\n"
            f"l = T.read({str(a)!r})\n"
            f"r = T.read({str(b)!r})\n"
            "j := join(l, r, on='k')\n"
        )
    ns = run(
        "from tensorlang import T, join\n"
        f"l = T.read({str(a)!r})\n"
        f"r = T.read({str(b)!r})\n"
        "j = join(l, r, on='k')\n"
    )
    assert ns["j"].collect().height == 1


def test_provenance_transitions():
    t = T([1, 2, 3])
    assert t.provenance == "literal"
    assert T.linspace(0, 1, 3).provenance == "literal"
    (P("X") + 1)(t).inplace()
    assert t.provenance == "derived"
    # Aliasing an existing tensor with := is a pure binding, not execution.
    ns = run("from tensorlang import T\nt = T([1])\nu := t\n")
    assert ns["u"] is ns["t"]


def test_data_dependent_call_rejected_by_value_check():
    # sin(data) is runtime; the classifier cannot know statically, so the
    # runtime value check (Lazy) rejects it with a source location.
    with pytest.raises(TLDeclarationError) as exc_info:
        run("from tensorlang import T, sin\nd = T([1.0])\ns := sin(d)\n")
    assert "sem.tl:3" in str(exc_info.value)


def test_aliases_observe_structured_inplace():
    ns = run(
        "from tensorlang import T\n"
        "s = T({'a': [1, 2, 3]})\n"
        "alias = s\n"
        "f := {D}[{D}.a > 1]\n"
        "f(s).inplace()\n"
    )
    assert ns["alias"].collect()["a"].to_list() == [2, 3]
    assert ns["s"].provenance == "derived"


def test_scan_cannot_be_inplaced(tmp_path):
    import polars as pl

    from tensorlang import TLMutationError

    src = tmp_path / "d.parquet"
    pl.DataFrame({"a": [1]}).write_parquet(src)
    s = T.scan_parquet(str(src))
    with pytest.raises(TLMutationError):
        (P("X") + 1)(s).inplace()


def test_missing_data_errors():
    _orders = T({"a": [1, None, 3]})
    assert _orders_is_null(_orders) == [False, True, False]


def _orders_is_null(t):
    from tensorlang import is_null

    return is_null(t).collect()["a"].to_list()


def test_scan_missing_file_errors_on_boundary(tmp_path):
    s = T.scan_parquet(str(tmp_path / "missing.parquet"))
    with pytest.raises(Exception):
        s.collect()
