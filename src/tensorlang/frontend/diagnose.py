"""Error analysis: stable codes, suggestions, and catalogue rendering.

``diagnose_source`` / ``diagnose_file`` execute TensorLang code and, on
failure, return a structured report: the error (code, category, message,
location, details), the static catalogue hint, and dynamic suggestions
(did-you-mean names, shape restatements, :=/= fixes, missing imports).
Reports are JSON-serializable for editor integration.
"""
from __future__ import annotations

import ast
import difflib
import re
import tokenize
from pathlib import Path

from ..core.errors import ERROR_REGISTRY, TensorLangError, TLSyntaxError

_AVAILABLE_RE = re.compile(r"available fields?: (\[.*?\])")
_PARAMS_RE = re.compile(r"params (are )?(\[.*?\])")


def _json_safe(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def error_report(exc: BaseException) -> dict:
    """Structured report for any failure (TensorLang or plain Python)."""
    if isinstance(exc, TensorLangError):
        entry = ERROR_REGISTRY.get(exc.code, {})
        return {
            "code": exc.code,
            "category": exc.category,
            "title": entry.get("title", "error"),
            "message": exc.message,
            "loc": exc.loc,
            "details": _json_safe(exc.details),
            "hint": entry.get("hint", ""),
        }
    if isinstance(exc, SyntaxError):
        return {
            "code": "TL0101",
            "category": "TLSyntaxError",
            "title": ERROR_REGISTRY["TL0101"]["title"],
            "message": f"invalid TensorLang syntax: {exc.msg}",
            "loc": None,
            "details": {},
            "hint": ERROR_REGISTRY["TL0101"]["hint"],
        }
    return {
        "code": "PY0000",
        "category": type(exc).__name__,
        "title": "python runtime error",
        "message": str(exc),
        "loc": None,
        "details": {},
        "hint": "",
    }


def suggest_for(exc: BaseException, namespace: dict | None = None) -> list[str]:
    """Ordered, human-readable fix suggestions for a failure."""
    suggestions: list[str] = []
    report = error_report(exc)
    if report["hint"]:
        suggestions.append(report["hint"])
    details = report["details"] if isinstance(report.get("details"), dict) else {}

    if isinstance(exc, TensorLangError):
        suggestions.extend(_from_details(report["code"], details, str(exc)))
    if isinstance(exc, NameError):
        suggestions.extend(_suggest_name(str(exc), namespace))
    if isinstance(exc, AttributeError):
        suggestions.extend(_suggest_attribute(str(exc), namespace))
    # Deduplicate while preserving order.
    seen, ordered = set(), []
    for s in suggestions:
        if s not in seen:
            seen.add(s)
            ordered.append(s)
    return ordered


def _names_from(details: dict) -> tuple[list[str], str]:
    for avail_key in ("available", "params", "columns", "valid"):
        if isinstance(details.get(avail_key), list):
            for got_key in ("got", "key", "field", "unknown", "missing", "name"):
                got = details.get(got_key)
                if isinstance(got, str):
                    return details[avail_key], got
                if isinstance(got, list) and got and isinstance(got[0], str):
                    return details[avail_key], got[0]
    return [], ""


def _from_details(code: str, details: dict, text: str) -> list[str]:
    out: list[str] = []
    available, got = _names_from(details)
    if available and got:
        close = difflib.get_close_matches(got, [str(a) for a in available],
                                          n=3, cutoff=0.6)
        if close:
            out.append(f"did you mean {close[0]!r}? "
                       f"(available: {', '.join(str(a) for a in available[:8])})")
        elif available:
            out.append(f"available: {', '.join(str(a) for a in available[:8])}")
    if code == "TL0301":
        out.append("Change ':=' to '=': the right-hand side executes an "
                   "operator application at runtime.")
    if code == "TL0305":
        out.append("Change ':=' to '=': arbitrary Python calls run only in "
                   "an explicitly runtime context; ':=' admits supported "
                   "declarative TensorLang operations.")
    if code == "TL0310":
        out.append("Import a sibling .tl file by name (resolved relative to "
                   "the importing file); Python imports take precedence.")
    if code == "TL0312":
        out.append("Restructure so .tl dependencies flow one way; circular "
                   "imports are rejected.")
    if code == "TL0304":
        out.append("Bind reads with '=', or describe the source lazily with "
                   "T.scan_parquet()/T.scan_csv() under ':='.")
    if "shapes" in details:
        out.append(f"incompatible shapes: {details['shapes']}")
    if "expected" in details and "got" in details:
        out.append(f"expected {details['expected']}, got {details['got']}")
    if "missing" in details and "available" in details:
        out.append(f"missing: {details['missing']}; have: {details['available']}")
    return out


def _suggest_name(text: str, namespace: dict | None) -> list[str]:
    out: list[str] = []
    match = re.search(r"name '(\w+)' is not defined", text)
    if not match:
        return out
    name = match.group(1)
    try:
        from .. import __all__ as _exports
    except Exception:  # pragma: no cover - defensive
        _exports = []
    if name in _exports:
        out.append(f"add 'from tensorlang import {name}' at the top of the file")
    pool = [k for k in (namespace or {}) if not k.startswith("__")]
    close = difflib.get_close_matches(name, pool + list(_exports), n=3, cutoff=0.6)
    if close and close[0] != name:
        out.append(f"did you mean {close[0]!r}?")
    return out


def _suggest_attribute(text: str, namespace: dict | None) -> list[str]:
    del namespace  # attribute hints come from the message itself
    out: list[str] = []
    match = re.search(r"has no field '(\w+)'", text)
    available = _AVAILABLE_RE.search(text)
    if match and available:
        try:
            fields = ast.literal_eval(available.group(1))
        except (SyntaxError, ValueError):
            return out
        close = difflib.get_close_matches(match.group(1),
                                          [str(f) for f in fields],
                                          n=3, cutoff=0.6)
        if close:
            out.append(f"did you mean field {close[0]!r}?")
    return out


def diagnose_source(source: str, filename: str = "<tl>") -> dict:
    """Compile and execute source; return a structured report dict."""
    from . import compiler as _compiler

    try:
        namespace = _compiler.run_source(source, filename=filename)
    except TensorLangError as exc:
        return {"ok": False, "error": error_report(exc),
                "suggestions": suggest_for(exc)}
    except (SyntaxError, tokenize.TokenError) as exc:
        wrapped = TLSyntaxError(f"invalid TensorLang syntax: {exc}",
                                code="TL0101")
        return {"ok": False, "error": error_report(wrapped),
                "suggestions": suggest_for(wrapped)}
    except Exception as exc:  # noqa: BLE001 - diagnose everything honestly
        return {"ok": False, "error": error_report(exc),
                "suggestions": suggest_for(exc)}
    names = sorted(k for k in namespace
                   if not (k.startswith("__") or k.startswith("_tl")))
    return {"ok": True, "names": names}


def diagnose_file(path: str | Path) -> dict:
    """Diagnose a `.tl` file; adds the file path to the report."""
    path = Path(path)
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        return {"ok": False, "path": str(path),
                "error": error_report(exc), "suggestions": []}
    report = diagnose_source(source, filename=str(path))
    report["path"] = str(path)
    return report


def render_catalogue() -> str:
    """Render docs/ERROR_CODES.md from ERROR_REGISTRY (single source)."""
    lines = [
        "# TensorLang error codes",
        "",
        "Stable, machine-readable codes attached to every `TensorLangError`",
        "(`ClassName[CODE]: loc: message`). Tools and programmers can rely on",
        "codes across message rewording. Ranges: `TL01` syntax, `TL02` scope,",
        "`TL03` declaration/effects, `TL04` types, `TL05` shapes, `TL06`",
        "mutation, `TL07` symbolic, `TL08` autodiff, `TL09` unsupported,",
        "`TL10` chunked/out-of-core (M1 planning, M2 layout, M3 budgets,",
        "M4 reliability); `x99` in each range is the generic class fallback.",
        "",
        "| code | category | title | guidance |",
        "|---|---|---|---|",
    ]
    for code in sorted(ERROR_REGISTRY):
        entry = ERROR_REGISTRY[code]
        lines.append(f"| `{code}` | {entry['category']} | {entry['title']} | "
                     f"{entry['hint']} |")
    return "\n".join(lines) + "\n"


def format_human(report: dict) -> str:
    """One-screen human rendering of a diagnose report."""
    if report.get("ok"):
        names = ", ".join(report.get("names", [])) or "(no bindings)"
        return f"{report.get('path', '<input>')}: OK ({names})"
    err = report["error"]
    loc = f" @ {err['loc']}" if err.get("loc") else ""
    lines = [f"error[{err['code']}] ({err['category']}){loc}: {err['message']}"]
    for suggestion in report.get("suggestions", []):
        lines.append(f"  hint: {suggestion}")
    return "\n".join(lines)
