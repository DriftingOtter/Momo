"""TensorLang error hierarchy.

Every error carries a stable machine-readable ``code`` (``TL0000``–``TL9999``,
grouped by category — see ``ERROR_REGISTRY`` and ``docs/ERROR_CODES.md``),
an optional source location, and optional structured ``details`` for
tooling (available names, expected/got shapes, ...). Rendering is
``ClassName[CODE]: loc: message`` so existing substring matches keep
working while tools can rely on codes.
"""
from __future__ import annotations


class TensorLangError(Exception):
    """Base class for all TensorLang errors."""

    default_code = "TL0000"

    def __init__(self, message: str, loc: str | None = None,
                 code: str | None = None, details: dict | None = None) -> None:
        self.loc = loc
        self.code = code or self.default_code
        self.details = dict(details) if details else {}
        self.message = message
        super().__init__(self._render())

    def _render(self) -> str:
        prefix = f"{self.loc}: " if self.loc else ""
        return f"{self.__class__.__name__}[{self.code}]: {prefix}{self.message}"

    @property
    def category(self) -> str:
        return self.__class__.__name__

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self._render()


class TLSyntaxError(TensorLangError):
    """Raised for malformed ``.tl`` source (lexing / parsing failures)."""

    default_code = "TL0199"


class TLScopeError(TensorLangError):
    """Raised for lexical-scope violations (e.g. duplicate ``:=``)."""

    default_code = "TL0299"


class TLDeclarationError(TensorLangError):
    """Raised when ``:=`` binds something that is not declarative."""

    default_code = "TL0399"


class TLTypeError(TensorLangError):
    """Raised for type / representation mismatches."""

    default_code = "TL0499"


class TLShapeError(TensorLangError):
    """Raised for incompatible shapes / schemas."""

    default_code = "TL0599"


class TLMutationError(TensorLangError):
    """Raised for invalid or failed in-place (``.inplace()``) execution."""

    default_code = "TL0699"


class TLSymbolicError(TensorLangError):
    """Raised for symbolic-math failures (non-symbolic expr, bad integral)."""

    default_code = "TL0799"


class TLDiffError(TensorLangError):
    """Raised for automatic-differentiation failures: nondifferentiable
    operations, non-scalar gradient targets, invalid wrt parameters,
    unsupported input representations, and higher-order attempts."""

    default_code = "TL0899"


class TLUnsupportedError(TensorLangError):
    """Raised for operations that are genuinely undefined for a representation."""

    default_code = "TL0999"


# -- stable code registry ----------------------------------------------------------
# Maps every explicitly assigned code to its category, title, and static
# guidance. The catalogue in docs/ERROR_CODES.md is generated from this
# table (see frontend.diagnose.render_catalogue); tests enforce that used
# codes are registered and the docs are fresh. Ranges: TL01 syntax,
# TL02 scope, TL03 declaration/effects, TL04 types, TL05 shapes, TL06
# mutation, TL07 symbolic, TL08 autodiff, TL09 unsupported, TL10
# chunked/out-of-core; x99 in each range is the generic class fallback.

def _reg(code: str, category: str, title: str, hint: str) -> str:
    ERROR_REGISTRY[code] = {"category": category, "title": title, "hint": hint}
    return code


ERROR_REGISTRY: dict[str, dict[str, str]] = {}

_reg("TL0101", "TLSyntaxError", "invalid syntax",
     "Check the reported line for TensorLang/Python syntax mistakes.")
_reg("TL0102", "TLSyntaxError", "malformed declaration",
     "Do not call __tl_declare__ directly; write NAME := expr.")
_reg("TL0201", "TLScopeError", "duplicate declaration",
     "A name may be declared once per scope; use = to reassign.")
_reg("TL0202", "TLScopeError", "declaration needs a plain name",
     ":= must introduce a plain name in the current scope.")
_reg("TL0301", "TLDeclarationError", "runtime application under :=",
     "Operator application executes; bind its result with = instead.")
_reg("TL0302", "TLDeclarationError", "effectful expression under :=",
     "Reads, writes, and inplace() are effectful; use = (or T.scan_* for a lazy descriptor).")
_reg("TL0303", "TLDeclarationError", "runtime expression under :=",
     "collect()/fit()/conversions execute; bind their result with =.")
_reg("TL0304", "TLDeclarationError", "effectful tensor under :=",
     "Tensors from T.read() carry I/O provenance; use = to bind them.")
_reg("TL0401", "TLTypeError", "ambiguous truth value",
     "Tensors have no truth value; collect() an explicit Boolean or reduce first.")
_reg("TL0402", "TLTypeError", "cannot iterate a tensor",
     "Iteration is never implicit; collect() or index explicitly first.")
_reg("TL0403", "TLTypeError", "fit needs tensor data",
     "Pass a Tensor or lazy result to .fit(), not scalars or operators.")
_reg("TL0404", "TLTypeError", "cannot read or describe source",
     "Check the path, format, and scan options.")
_reg("TL0405", "TLTypeError", "cannot write output",
     "Write structured frames/series to .parquet/.csv; select fields first for dense data.")
_reg("TL0410", "TLTypeError", "non-numeric estimator data",
     "Select or encode numeric feature columns explicitly.")
_reg("TL0411", "TLTypeError", "invalid estimator configuration",
     "Check constructor arguments (ranges, norms, strategies, counts).")
_reg("TL0412", "TLTypeError", "missing or infinite fitting data",
     "Handle nulls/infinities explicitly first, e.g. with impute().")
_reg("TL0413", "TLTypeError", "cannot impute all-missing column",
     "Drop the column or supply values; no statistic exists.")
_reg("TL0414", "TLTypeError", "unknown estimator or option",
     "Use a documented estimator kind, strategy, or norm name.")
_reg("TL0415", "TLTypeError", "missing features on apply",
     "Application data must contain every fitted feature column.")
_reg("TL0416", "TLTypeError", "operator is not fittable",
     "Only standardize()/minmax()/normalize()/impute()/pca() (and >> pipelines) support .fit().")
_reg("TL0417", "TLTypeError", "invalid operator composition",
     "Both sides of >> must be operators; the right side must be unary.")
_reg("TL0418", "TLTypeError", "unsupported structured-to-dense bridge",
     "Select existing numeric columns with to_dense(columns=[...]); "
     "use nulls='error' to forbid missing values, or fit() numerical "
     "data directly.")
_reg("TL0420", "TLTypeError", "gradient constructor needs an operator",
     "Pass an Operator to grad()/value_and_grad()/vjp(), not data.")
_reg("TL0430", "TLTypeError", "invalid join configuration",
     "Use a documented join type; non-cross joins need explicit on= keys.")
_reg("TL0431", "TLTypeError", "join key or shape problem",
     "Keys must exist on both sides with compatible lengths.")
_reg("TL0501", "TLShapeError", "incompatible broadcast shapes",
     "Align shapes explicitly; broadcasting follows NumPy rules.")
_reg("TL0502", "TLShapeError", "incompatible matrix multiplication",
     "matmul needs (…, n, k) @ (…, k, m) numerical operands.")
_reg("TL0503", "TLShapeError", "operator arity mismatch",
     "Apply with exactly one argument per parameter, positionally.")
_reg("TL0504", "TLShapeError", "mask length mismatch",
     "Boolean masks need one entry per element/row.")
_reg("TL0505", "TLShapeError", "assignment shape mismatch",
     "Assigned values must match the target's rows/positions.")
_reg("TL0506", "TLShapeError", "invalid linear system",
     "solve() needs a square non-singular matrix with matching right-hand side.")
_reg("TL0507", "TLShapeError", "invalid decomposition input",
     "svd() needs rank-2 matrices; eigh() needs square matrices.")
_reg("TL0508", "TLShapeError", "invalid PCA request",
     "Need n_components <= min(n_samples, n_features) and >= 2 samples.")
_reg("TL0509", "TLShapeError", "feature width mismatch on apply",
     "Application data must have the fitted feature count (frames align by name).")
_reg("TL0510", "TLShapeError", "join execution failed",
     "Check key dtypes and join compatibility.")
_reg("TL0511", "TLShapeError", "invalid reduction axis",
     "Use axis=None, an in-range axis, or a tuple of in-range axes; "
     "keepdims must be bool.")
_reg("TL0601", "TLMutationError", "inplace without a mutable target",
     "Use f(tensor).inplace() with a single-input operator over a Tensor.")
_reg("TL0602", "TLMutationError", "inplace into a lazy scan",
     "Scans are immutable descriptors; collect() into a tensor first.")
_reg("TL0603", "TLMutationError", "inplace result incompatible",
     "The computed result cannot replace the target's contents.")
_reg("TL0604", "TLMutationError", "inplace execution failed",
     "The input was left unchanged; inspect the cause.")
_reg("TL0701", "TLSymbolicError", "non-symbolic expression",
     "Differentiate/integrate pure mathematical operators only.")
_reg("TL0702", "TLSymbolicError", "constant without exact form",
     "The constant has no exact symbolic representation.")
_reg("TL0703", "TLSymbolicError", "symbolic arity mismatch",
     "diff()/integrate() need single-parameter operators.")
_reg("TL0704", "TLSymbolicError", "no exact closed form",
     "No exact integral exists; numerical quadrature is never silent.")
_reg("TL0801", "TLDiffError", "invalid differentiation target",
     "Specify wrt= with an existing parameter name (default: the sole parameter).")
_reg("TL0802", "TLDiffError", "second-order differentiation",
     "Higher-order derivatives are deferred; grad() of a gradient operator is rejected.")
_reg("TL0803", "TLDiffError", "non-scalar gradient target",
     "grad() needs scalar-valued operators; use vjp() with a cotangent otherwise.")
_reg("TL0804", "TLDiffError", "non-floating gradient input",
     "Differentiable inputs must be floating-point tensors; pass float data.")
_reg("TL0805", "TLDiffError", "unfitted estimator in graph",
     "Differentiation through fitting is deferred; call .fit() first (only fitted application differentiates).")
_reg("TL0806", "TLDiffError", "nondifferentiable operation",
     "See LANGUAGE_SPEC.md for the supported rule set; restructure the graph.")
_reg("TL0807", "TLDiffError", "gradient shape mismatch",
     "Cotangent shapes must be consistent through the graph.")
_reg("TL0808", "TLDiffError", "vjp cotangent mismatch",
     "The seed must match the output shape and kind exactly.")
_reg("TL0901", "TLUnsupportedError", "undefined for representation",
     "The operation has no meaning for this representation; select/convert explicitly.")
_reg("TL0809", "TLDiffError", "differentiation over chunked data deferred",
     "Out-of-core differentiation is deferred; collect() explicitly to materialize first.")
_reg("TL1001", "TLUnsupportedError", "unsupported chunked operation",
     "Not chunk-lowerable in v0.5; restructure with elementwise/shape/reduction ops.")
_reg("TL1002", "TLTypeError", "chunked write target exists",
     "Pass overwrite=True to replace an existing store, or choose a fresh path.")
_reg("TL1003", "TLTypeError", "chunked source unreadable",
     "Check the store path, format, and permissions.")
_reg("TL1010", "TLUnsupportedError", "incompatible chunk layouts",
     "Layouts cannot align safely (unknown dimensions, unmergeable reshapes, "
     "partitioned contraction axes); rechunk the store or restructure.")
_reg("TL1011", "TLUnsupportedError", "unsafe rechunking",
     "The plan exceeds conservative block-byte limits; use smaller chunks "
     "or fewer partitions.")
_reg("TL1012", "TLUnsupportedError", "excessive graph expansion",
     "The plan exceeds the task-count limit; simplify or coarsen chunks.")
_reg("TL1013", "TLUnsupportedError", "unsupported global operation",
     "Sorting, joins, grouping, and overlap exchange need global data "
     "movement; restructure outside chunked graphs.")
_reg("TL1020", "TLUnsupportedError", "memory-budget admission failure",
     "The modeled working set exceeds the configured budget; raise the "
     "budget (see the suggested value) or use smaller chunks.")
_reg("TL1021", "TLUnsupportedError", "unbounded resource estimate",
     "Resource needs cannot be bounded reliably; unknown requirements "
     "are never treated as zero.")
_reg("TL1022", "TLUnsupportedError", "execution-time resource pressure",
     "Observed memory growth far exceeds budget and model; "
     "already-running native operations can overshoot modeled limits.")
_reg("TL1023", "TLUnsupportedError", "incompatible chunk/concurrency",
     "A single task's working set exceeds the budget, so reducing "
     "concurrency cannot help; rechunk the store to smaller chunks.")
_reg("TL1024", "TLUnsupportedError", "materialization exceeds budget",
     "collect() would materialize more than the budget allows; stream "
     "with write_zarr() instead.")
_reg("TL1030", "TLUnsupportedError", "persistent source changed",
     "A zarr source changed after planning; the admitted plan no longer "
     "describes it. Detection is metadata-only and best-effort.")
_reg("TL1031", "TLUnsupportedError", "staging or commit failure",
     "Execution or publication failed; no partial output was published. "
     "See details for stage and backend error.")
_reg("TL1032", "TLUnsupportedError", "interrupted execution",
     "The run was cancelled; no new work is scheduled. Running native "
     "tasks may have completed; staging is cleaned after workers stop.")
_reg("TL1033", "TLUnsupportedError", "stale staging artifacts",
     "A previous interrupted run left staging behind; use "
     "recover_stale_staging() to inspect or remove it.")
_reg("TL1034", "TLUnsupportedError", "output validation failure",
     "Chunk census or metadata mismatch before publication; the "
     "destination was left untouched.")
_reg("TL1035", "TLUnsupportedError", "resource cleanup failure",
     "Staging removal failed; the report names the remaining path and a "
     "manual recovery instruction.")
_reg("TL0199", "TLSyntaxError", "unspecified syntax error",
     "Generic syntax failure; see the message.")
_reg("TL0299", "TLScopeError", "unspecified scope error",
     "Generic scope failure; see the message.")
_reg("TL0399", "TLDeclarationError", "unspecified declaration error",
     "Generic declaration failure; see the message.")
_reg("TL0499", "TLTypeError", "unspecified type error",
     "Generic type failure; see the message.")
_reg("TL0599", "TLShapeError", "unspecified shape error",
     "Generic shape failure; see the message.")
_reg("TL0699", "TLMutationError", "unspecified mutation error",
     "Generic mutation failure; see the message.")
_reg("TL0799", "TLSymbolicError", "unspecified symbolic error",
     "Generic symbolic failure; see the message.")
_reg("TL0899", "TLDiffError", "unspecified differentiation error",
     "Generic differentiation failure; see the message.")
_reg("TL0999", "TLUnsupportedError", "unsupported operation",
     "Generic unsupported-operation failure; see the message.")
