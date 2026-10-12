"""TensorLang error hierarchy.

Every error may carry an optional source location (``filename:lineno``) so
that diagnostics from the ``.tl`` frontend point at user code.
"""
from __future__ import annotations


class TensorLangError(Exception):
    """Base class for all TensorLang errors."""

    def __init__(self, message: str, loc: str | None = None) -> None:
        self.loc = loc
        prefix = f"{loc}: " if loc else ""
        super().__init__(f"{prefix}{message}")
        self.message = message

    def __str__(self) -> str:  # pragma: no cover - trivial
        prefix = f"{self.loc}: " if self.loc else ""
        return f"{self.__class__.__name__}: {prefix}{self.message}"


class TLSyntaxError(TensorLangError):
    """Raised for malformed ``.tl`` source (lexing / parsing failures)."""


class TLScopeError(TensorLangError):
    """Raised for lexical-scope violations (e.g. duplicate ``:=``)."""


class TLDeclarationError(TensorLangError):
    """Raised when ``:=`` binds something that is not declarative."""


class TLTypeError(TensorLangError):
    """Raised for type / representation mismatches."""


class TLShapeError(TensorLangError):
    """Raised for incompatible shapes / schemas."""


class TLMutationError(TensorLangError):
    """Raised for invalid or failed in-place (``.inplace()``) execution."""


class TLSymbolicError(TensorLangError):
    """Raised for symbolic-math failures (non-symbolic expr, bad integral)."""


class TLDiffError(TensorLangError):
    """Raised for automatic-differentiation failures: nondifferentiable
    operations, non-scalar gradient targets, invalid wrt parameters,
    unsupported input representations, and higher-order attempts."""


class TLUnsupportedError(TensorLangError):
    """Raised for operations that are genuinely undefined for a representation."""
