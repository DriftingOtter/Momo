"""Comparison and logical operations (named-function spellings)."""
from __future__ import annotations

from ._lift import lift_binary, lift_unary_operand
from ..core import expression as ir


def logical_and(a, b):
    """Elementwise AND / record-mask conjunction (``&`` as a function)."""
    return lift_binary(a, b, "and")


def logical_or(a, b):
    """Elementwise OR / record-mask disjunction (``|`` as a function)."""
    return lift_binary(a, b, "or")


def logical_not(a):
    """Elementwise NOT (``~`` as a function)."""
    return lift_unary_operand(a, lambda root: ir.UnOp("not", root),
                              const_build=lambda v: not v)


def equal(a, b):
    return lift_binary(a, b, "eq")


def not_equal(a, b):
    return lift_binary(a, b, "ne")
