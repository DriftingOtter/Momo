"""Public API inventory regression gate (M4).

Pins the 82-export surface: exact names, exact call signatures, and
the stable/provisional partition from docs/API_STABILITY.md. Any
accidental change to an exported name or signature fails here and
forces a conscious contract decision (see docs/COMPATIBILITY.md).
"""
import inspect

import pytest

import tensorlang as tl

EXPECTED_SIGNATURES = {
    "T": "T(data: 'Any' = None, **fields: 'Any')",
    "Tensor": "Tensor(data: 'Any' = None) -> 'None'",
    "Lazy": "Lazy(root: 'ir.Expr', target: 'Optional[Tensor]' = None) -> 'None'",
    "Operator": "Operator(params: 'tuple[str, ...]', root: 'ir.Expr', uid: 'str') -> 'None'",
    "P": 'P(name: \'str\') -> "\'Operator\'"',
    "sin": "sin(x)",
    "cos": "cos(x)",
    "tan": "tan(x)",
    "exp": "exp(x)",
    "log": "log(x)",
    "sqrt": "sqrt(x)",
    "diff": "diff(op)",
    "integrate": "integrate(op, lo=None, hi=None)",
    "simplify": "simplify(op)",
    "concat": "concat(parts, axis: 'int' = 0)",
    "stack": "stack(parts, axis: 'int' = 0)",
    "reshape": "reshape(x, *shape)",
    "flatten": "flatten(x)",
    "transpose": "transpose(x)",
    "logical_and": "logical_and(a, b)",
    "logical_or": "logical_or(a, b)",
    "logical_not": "logical_not(a)",
    "equal": "equal(a, b)",
    "not_equal": "not_equal(a, b)",
    "join": "join(left, right, on=None, how: 'str' = 'inner', suffix: 'str' = '_right')",
    "sort": "sort(x, by=None, descending: 'bool' = False)",
    "unique": "unique(x)",
    "where": "where(cond, a, b)",
    "clip": "clip(x, lo=None, hi=None)",
    "over": "over(x, by, op: 'str' = 'mean')",
    "rolling": "rolling(x, window: 'int', op: 'str' = 'mean', min_periods: 'int | None' = None)",
    "standardize": "standardize() -> 'FittableOperator'",
    "minmax": "minmax(lo: 'float' = 0.0, hi: 'float' = 1.0) -> 'FittableOperator'",
    "normalize": "normalize(norm: 'str' = 'l2') -> 'FittableOperator'",
    "impute": "impute(strategy: 'str' = 'mean', fill_value=None) -> 'FittableOperator'",
    "pca": "pca(n_components: 'int', whiten: 'bool' = False) -> 'FittableOperator'",
    "FittableOperator": "FittableOperator(kind: 'str', config: 'dict', state: 'FittedState | None' = None, uid: 'str | None' = None) -> 'None'",
    "PipelineOperator": "PipelineOperator(stages: 'list[Operator]', uid: 'str | None' = None) -> 'None'",
    "grad": "grad(loss, wrt=None) -> 'GradOperator'",
    "value_and_grad": "value_and_grad(loss, wrt=None)",
    "vjp": "vjp(loss, wrt=None)",
    "GradOperator": "GradOperator(source: 'Operator', wrt: 'list[str]') -> 'None'",
    "ValueGradOperator": "ValueGradOperator(source: 'Operator', wrt: 'list[str]') -> 'None'",
    "VjpOperator": "VjpOperator(source: 'Operator', wrt: 'list[str]') -> 'None'",
    "ValueGradResult": "ValueGradResult(iterable=(), /)",
    "svd": "svd(x, full_matrices: 'bool' = True) -> 'SVDResult'",
    "eigh": "eigh(x) -> 'EighResult'",
    "solve": "solve(a, b)",
    "norm": "norm(x, ord=None, axis=None)",
    "SVDResult": "SVDResult(iterable=(), /)",
    "EighResult": "EighResult(iterable=(), /)",
    "cov": "cov(x, rowvar: 'bool | None' = None)",
    "corr": "corr(x, rowvar: 'bool | None' = None)",
    "quantile": "quantile(x, q=0.5)",
    "einsum": "einsum(spec: 'str', *operands)",
    "tensordot": "tensordot(a, b, axes=2)",
    "moveaxis": "moveaxis(x, source, destination)",
    "onehot": "onehot(x, categories=None)",
    "polynomial": "polynomial(x, degree: 'int' = 2, include_bias: 'bool' = True)",
    "is_null": "is_null(x)",
    "not_null": "not_null(x)",
    "fill_null": "fill_null(x, value)",
    "drop_null": "drop_null(x)",
    "run_source": "run_source(source: 'str', filename: 'str' = '<tl>', extra: 'dict[str, Any] | None' = None) -> 'dict[str, Any]'",
    "run_file": "run_file(path: 'str | Path') -> 'dict[str, Any]'",
    "check_source": "check_source(source: 'str', filename: 'str' = '<tl>', trusted_names: 'frozenset[str] | None' = None, injected_names: 'frozenset[str] | None' = None) -> 'ast.Module'",
    "diagnose_source": "diagnose_source(source: 'str', filename: 'str' = '<tl>') -> 'dict'",
    "diagnose_file": "diagnose_file(path: 'str | Path') -> 'dict'",
    "set_chunk_policy": "set_chunk_policy(*, max_tasks=None, max_block_bytes=None, memory_budget_bytes=None, max_workers=None, union_growth=None) -> 'dict'",
    "clear_chunk_policy": "clear_chunk_policy() -> 'None'",
    "current_policy": "current_policy() -> 'dict'",
    "TensorLangError": "TensorLangError(message: 'str', loc: 'str | None' = None, code: 'str | None' = None, details: 'dict | None' = None) -> 'None'",
    "TLSyntaxError": "TLSyntaxError(message: 'str', loc: 'str | None' = None, code: 'str | None' = None, details: 'dict | None' = None) -> 'None'",
    "TLScopeError": "TLScopeError(message: 'str', loc: 'str | None' = None, code: 'str | None' = None, details: 'dict | None' = None) -> 'None'",
    "TLDeclarationError": "TLDeclarationError(message: 'str', loc: 'str | None' = None, code: 'str | None' = None, details: 'dict | None' = None) -> 'None'",
    "TLTypeError": "TLTypeError(message: 'str', loc: 'str | None' = None, code: 'str | None' = None, details: 'dict | None' = None) -> 'None'",
    "TLShapeError": "TLShapeError(message: 'str', loc: 'str | None' = None, code: 'str | None' = None, details: 'dict | None' = None) -> 'None'",
    "TLMutationError": "TLMutationError(message: 'str', loc: 'str | None' = None, code: 'str | None' = None, details: 'dict | None' = None) -> 'None'",
    "TLSymbolicError": "TLSymbolicError(message: 'str', loc: 'str | None' = None, code: 'str | None' = None, details: 'dict | None' = None) -> 'None'",
    "TLDiffError": "TLDiffError(message: 'str', loc: 'str | None' = None, code: 'str | None' = None, details: 'dict | None' = None) -> 'None'",
    "TLUnsupportedError": "TLUnsupportedError(message: 'str', loc: 'str | None' = None, code: 'str | None' = None, details: 'dict | None' = None) -> 'None'",
}

# 65 stable + __version__; 16 provisional (docs/API_STABILITY.md).
STABLE = {n for n in EXPECTED_SIGNATURES
          if n not in {"equal", "not_equal", "unique", "over", "rolling",
                       "einsum", "tensordot", "moveaxis", "onehot",
                       "polynomial", "GradOperator", "ValueGradOperator",
                       "VjpOperator", "ValueGradResult", "SVDResult",
                       "EighResult"}}
PROVISIONAL = set(EXPECTED_SIGNATURES) - STABLE


def test_export_set_exact():
    assert set(tl.__all__) == set(EXPECTED_SIGNATURES) | {"__version__"}
    assert len(tl.__all__) == 82
    assert isinstance(tl.__version__, str)


def test_signatures_pinned():
    mismatches = []
    for name, expected in EXPECTED_SIGNATURES.items():
        actual = f"{name}{inspect.signature(getattr(tl, name))}"
        if actual != expected:
            mismatches.append(f"{name}:\n  pinned:   {expected}\n  actual:   {actual}")
    assert not mismatches, "\n".join(mismatches)


def test_partition_stable_provisional():
    assert len(STABLE) == 65
    assert len(PROVISIONAL) == 16
    assert STABLE | PROVISIONAL == set(EXPECTED_SIGNATURES)
    for name in STABLE:
        assert callable(getattr(tl, name)), name
