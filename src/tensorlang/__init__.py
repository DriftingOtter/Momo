"""TensorLang: a tensor-oriented language for numerical computing,
structured data, and symbolic mathematics."""

from .core.errors import (
    TensorLangError,
    TLDiffError,
    TLDeclarationError,
    TLMutationError,
    TLScopeError,
    TLShapeError,
    TLSymbolicError,
    TLSyntaxError,
    TLTypeError,
    TLUnsupportedError,
)
from .core.operator import Operator, P
from .core.tensor import Lazy, T, Tensor
from .frontend.compiler import check_source, run_file, run_source
from .frontend.diagnose import diagnose_file, diagnose_source
from .operators.calculus import diff, integrate, simplify
from .operators.data_science import drop_null, fill_null, is_null, not_null
from .operators.elementary import cos, exp, log, sin, sqrt, tan
from .operators.linear_algebra import concat, flatten, reshape, stack, transpose
from .operators.logical import (
    equal,
    logical_and,
    logical_not,
    logical_or,
    not_equal,
)
from .operators.ml import (
    FittableOperator,
    PipelineOperator,
    impute,
    minmax,
    normalize,
    pca,
    standardize,
)
from .operators.autodiff import (
    GradOperator,
    ValueGradOperator,
    ValueGradResult,
    VjpOperator,
    grad,
    value_and_grad,
    vjp,
)
from .operators.linalg import EighResult, SVDResult, eigh, norm, solve, svd
from .operators.statistics import corr, cov, quantile
from .operators.tensor_algebra import einsum, moveaxis, tensordot
from .operators.features import onehot, polynomial
from .operators.relational import join, sort, unique
from .operators.transform import clip, over, rolling, where
from .runtime.chunk_plan import (
    clear_chunk_policy,
    current_policy,
    set_chunk_policy,
)

__version__ = "0.6.0"

__all__ = [
    "T", "Tensor", "Lazy", "Operator", "P",
    "sin", "cos", "tan", "exp", "log", "sqrt",
    "diff", "integrate", "simplify",
    "concat", "stack", "reshape", "flatten", "transpose",
    "logical_and", "logical_or", "logical_not", "equal", "not_equal",
    "join", "sort", "unique",
    "where", "clip", "over", "rolling",
    "standardize", "minmax", "normalize", "impute", "pca",
    "FittableOperator", "PipelineOperator",
    "grad", "value_and_grad", "vjp",
    "GradOperator", "ValueGradOperator", "VjpOperator", "ValueGradResult",
    "svd", "eigh", "solve", "norm", "SVDResult", "EighResult",
    "cov", "corr", "quantile",
    "einsum", "tensordot", "moveaxis",
    "onehot", "polynomial",
    "is_null", "not_null", "fill_null", "drop_null",
    "run_source", "run_file", "check_source",
    "diagnose_source", "diagnose_file",
    "set_chunk_policy", "clear_chunk_policy", "current_policy",
    "TensorLangError", "TLSyntaxError", "TLScopeError", "TLDeclarationError",
    "TLTypeError", "TLShapeError", "TLMutationError", "TLSymbolicError",
    "TLDiffError", "TLUnsupportedError",
    "__version__",
]
