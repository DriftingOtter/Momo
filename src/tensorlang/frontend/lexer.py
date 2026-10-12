"""TensorLang lexer: token-level translation of ``.tl`` source to Python.

``.tl`` files are Python-compatible syntax plus two extensions:

- ``{X}`` — a lexically scoped symbolic parameter, rewritten to
  ``__tl_param__('X')``.  Only braces holding exactly one bare identifier
  are rewritten, so dict/set displays, annotations and f-strings pass
  through untouched.
- ``NAME := expr`` — a declarative definition at statement level, rewritten
  to ``NAME = __tl_declare__('NAME', expr, lineno)``.

No regular-expression substitution is used: everything operates on the
CPython token stream, so comments, strings, and multiline statements keep
working and source locations stay exact.
"""
from __future__ import annotations

import io
import tokenize
from dataclasses import dataclass


DECLARE_FN = "__tl_declare__"
PARAM_FN = "__tl_param__"

_SKIP = {tokenize.COMMENT, tokenize.NL, tokenize.ENCODING}


@dataclass
class Edit:
    start: tuple[int, int]
    end: tuple[int, int]
    text: str


def _offset(line_starts: list[int], pos: tuple[int, int]) -> int:
    row, col = pos
    return line_starts[row - 1] + col


def translate(source: str) -> str:
    """Rewrite ``.tl`` source into executable Python source."""
    toks = list(tokenize.generate_tokens(io.StringIO(source).readline))
    line_starts = [0]
    for line in source.splitlines(keepends=True):
        line_starts.append(line_starts[-1] + len(line))
    edits: list[Edit] = []

    # Pass 1: {NAME} -> __tl_param__('NAME')  (outside f-strings)
    fstring = 0
    for i, tok in enumerate(toks):
        if tok.type == tokenize.FSTRING_START:
            fstring += 1
        elif tok.type == tokenize.FSTRING_END:
            fstring -= 1
        if fstring:
            continue
        if tok.type == tokenize.OP and tok.string == "{":
            j = _next_significant(toks, i + 1)
            k = _next_significant(toks, j + 1) if j is not None else None
            if (
                j is not None and k is not None
                and toks[j].type == tokenize.NAME
                and toks[k].type == tokenize.OP and toks[k].string == "}"
            ):
                edits.append(Edit(toks[i].start, toks[k].end,
                                  f"{PARAM_FN}({toks[j].string!r})"))

    # Pass 2: NAME := ... -> NAME = __tl_declare__('NAME', ..., lineno)
    depth = 0
    fstring = 0
    for i, tok in enumerate(toks):
        ttype = tok.type
        if ttype == tokenize.FSTRING_START:
            fstring += 1
            continue
        if ttype == tokenize.FSTRING_END:
            fstring -= 1
            continue
        if fstring:
            continue
        if ttype == tokenize.OP and tok.string in "([{":
            depth += 1
            continue
        if ttype == tokenize.OP and tok.string in ")]}":
            depth -= 1
            continue
        if ttype == tokenize.OP and tok.string == ":=" and depth == 0:
            prev = _prev_significant(toks, i - 1)
            if prev is None or toks[prev].type != tokenize.NAME:
                continue
            before = _prev_significant(toks, prev - 1)
            if before is None:
                ok = True
            else:
                btok = toks[before]
                ok = (
                    (btok.type == tokenize.NEWLINE)
                    or (btok.type == tokenize.INDENT)
                    or (btok.type == tokenize.DEDENT)
                    or (btok.type == tokenize.OP and btok.string in (";", ":"))
                )
            if not ok:
                continue  # ordinary Python walrus usage
            name = toks[prev].string
            lineno = tok.start[0]
            edits.append(Edit(tok.start, tok.end,
                              f"= {DECLARE_FN}({name!r}, "))
            end = _statement_end(toks, i + 1)
            edits.append(Edit(end, end, f", {lineno})"))

    if not edits:
        return source
    # Apply from the end so offsets stay valid.
    keyed = sorted(
        ((_offset(line_starts, e.start), _offset(line_starts, e.end), e.text)
         for e in edits),
        reverse=True,
    )
    out = source
    for s, e, text in keyed:
        out = out[:s] + text + out[e:]
    return out


def _next_significant(toks, i: int):
    while i < len(toks):
        if toks[i].type not in _SKIP:
            return i
        i += 1
    return None


def _prev_significant(toks, i: int):
    while i >= 0:
        if toks[i].type not in _SKIP:
            return i
        i -= 1
    return None


def _statement_end(toks, i: int):
    """Position just before the COMMENT/NEWLINE/';' ending the statement."""
    j = _next_significant(toks, i)
    while j is not None:
        tok = toks[j]
        if tok.type == tokenize.COMMENT:
            return tok.start
        if tok.type == tokenize.NEWLINE:
            return tok.start
        if tok.type == tokenize.OP and tok.string == ";":
            return tok.start
        if tok.type == tokenize.ENDMARKER:
            return tok.end
        j = _next_significant(toks, j + 1)
    return toks[-1].end
