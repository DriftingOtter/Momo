"""Workload-evaluation harness (v0.6 phase, investigative only).

Methodology: each measured kernel runs in a FRESH SUBPROCESS so peak RSS
is attributable (``os.wait4`` child rusage — a true per-workload process
peak, unlike in-process high-water marks that never decrease). Timing is
measured inside the child (excludes spawn overhead): median of N repeats
after 1 warmup, with min/max reported as variability. Equivalence is
asserted on every run; only measured numbers are reported, never claims.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_CHILD_MAX_MB = 0.0


def _children_peak_mb():
    """High-water max RSS of reaped children (macOS: bytes, Linux: KB).

    Cumulative-max semantics: attributable exactly when the just-finished
    child dominates all previous children (true for these workloads;
    otherwise reported as missing, never fabricated)."""
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        return peak / 1e6 if sys.platform == "darwin" else peak / 1e3
    except Exception:
        return None


def run_isolated(driver: str, repeats: int = 5, timeout: int = 1200) -> dict:
    """Run ``driver`` (a ``def main() -> dict`` source string) in a fresh
    Python subprocess. The driver must import workload code itself and
    return JSON-serializable measurements. Returns the driver dict plus
    ``peak_rss_mb`` (child process peak) and ``returncode``."""
    with tempfile.TemporaryDirectory(prefix="tl-eval-") as tmp:
        script = os.path.join(tmp, "driver.py")
        out_path = os.path.join(tmp, "result.json")
        with open(script, "w") as fh:
            fh.write(textwrap.dedent(driver))
        env = dict(os.environ, TL_EVAL_OUT=out_path,
                   TL_EVAL_REPEATS=str(repeats),
                   PYTHONPATH=os.path.join(REPO, "src") +
                   os.pathsep + os.environ.get("PYTHONPATH", ""))
        # Per-command peak RSS via /usr/bin/time -l (macOS) or -v (Linux);
        # exact per-child attribution, immune to high-water accumulation.
        time_bin = "/usr/bin/time"
        argv = [sys.executable, script]
        use_time = (sys.platform == "darwin"
                    and os.path.exists(time_bin))
        if use_time:
            argv = [time_bin, "-l", sys.executable, script]
        before = _children_peak_mb()
        proc = subprocess.Popen(argv, env=env, cwd=REPO,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate(timeout=timeout)
        after = _children_peak_mb()
        result = {"returncode": proc.returncode,
                  "stdout_tail": stdout.decode()[-500:] if stdout else "",
                  "stderr_tail": stderr.decode()[-2000:] if stderr else ""}
        if proc.returncode != 0:
            return result
        with open(out_path) as fh:
            result.update(json.load(fh))
        peak = _parse_time_peak(stderr.decode() if stderr else "")
        if peak is not None:
            result["peak_rss_mb"] = peak
            result["peak_rss_source"] = "time -l"
        else:
            global _CHILD_MAX_MB
            if after is not None and (before is None or after > before):
                result["peak_rss_mb"] = after
                result["peak_rss_source"] = "children high-water"
                _CHILD_MAX_MB = max(_CHILD_MAX_MB, after)
        return result


def _parse_time_peak(stderr_text: str) -> float | None:
    """Parse per-command max RSS. macOS `time -l` prints the number BEFORE
    the label (bytes); GNU `time -v` prints label-then-number (kilobytes)."""
    import re as _re

    match = _re.search(r"(\d+)\s+maximum resident set size", stderr_text)
    if match:
        try:
            return int(match.group(1)) / 1e6
        except ValueError:
            return None
    match = _re.search(r"Maximum resident set size[^0-9]*(\d+)", stderr_text)
    if match:
        try:
            return int(match.group(1)) / 1e3
        except ValueError:
            return None
    return None


def driver_template(body: str) -> str:
    """Wrap ``body`` into a driver program. Body may use ``time_phase(name,
    fn)`` and must define ``main() -> dict`` (JSON-serializable); the
    wrapper dumps it to ``$TL_EVAL_OUT``. ``body`` must set
    ``result["phases"] = _PHASES`` and any ``checks`` entries itself."""
    return """
        import json, os, sys, time
        sys.path.insert(0, %r)
        sys.path.insert(0, os.path.join(%r, "evaluation"))
        _PHASES = {}
        def time_phase(name, fn, repeats=None):
            n = int(repeats or os.environ.get("TL_EVAL_REPEATS", "5"))
            fn()  # warmup (imports, caches, first-touch allocation)
            times = []
            out = None
            for _ in range(n):
                start = time.perf_counter()
                out = fn()
                times.append(time.perf_counter() - start)
            times.sort()
            _PHASES[name] = {"median_s": times[n // 2],
                             "min_s": times[0], "max_s": times[-1],
                             "repeats": n}
            return out
    """ % (os.path.join(REPO, "src"), REPO) + "\n" + body + """
        if __name__ == "__main__":
            import json as _json, os as _os
            _result = main()
            with open(_os.environ["TL_EVAL_OUT"], "w") as _fh:
                _json.dump(_result, _fh)
    """


def user_loc(source: str) -> int:
    """Non-blank, non-comment source lines (expressiveness metric)."""
    n = 0
    for line in source.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            n += 1
    return n


def fmt_table(headers: list, rows: list) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    lines = [" | ".join(str(h).ljust(w) for h, w in zip(headers, widths)),
             "-|-".join("-" * w for w in widths)]
    for row in rows:
        lines.append(" | ".join(str(c).ljust(w)
                                for c, w in zip(row, widths)))
    return "\n".join(lines)
