"""TensorLang command-line interface: run, check, diagnose, repl, and friends."""
from __future__ import annotations

import argparse
import sys
import traceback

from .core.errors import TensorLangError
from .frontend import compiler as _compiler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tensorlang", description="Run TensorLang (.tl) programs."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="execute a .tl source file")
    run.add_argument("file", help="path to a .tl source file")

    check = sub.add_parser("check", help="parse and validate a .tl file")
    check.add_argument("file", help="path to a .tl source file")

    diag = sub.add_parser("diagnose", help="run a .tl file with enriched diagnostics")
    diag.add_argument("file", help="path to a .tl source file")
    diag.add_argument("--format", choices=["human", "json"], default="human")

    errors = sub.add_parser("errors", help="browse the error-code catalogue")
    errors.add_argument("code", nargs="?", help="e.g. TL0301 (omit to list all)")

    repl = sub.add_parser("repl", help="interactive TensorLang playground")

    graph = sub.add_parser("graph", help="dump an expression graph (runs the file)")
    graph.add_argument("file", help="path to a .tl source file")
    graph.add_argument("--var", default=None, help="binding to show (default: all)")
    graph.add_argument("--dot", action="store_true", help="Graphviz DOT output")
    graph.add_argument("--allow-effects", action="store_true",
                       help="permit I/O or mutation while introspecting")

    explain = sub.add_parser("explain", help="print explain() for a binding (runs the file)")
    explain.add_argument("file", help="path to a .tl source file")
    explain.add_argument("var", help="binding to explain")
    explain.add_argument("--allow-effects", action="store_true",
                         help="permit I/O or mutation while introspecting")

    profile = sub.add_parser("profile", help="print profile() for a binding (runs the file)")
    profile.add_argument("file", help="path to a .tl source file")
    profile.add_argument("var", help="binding to profile")
    profile.add_argument("--allow-effects", action="store_true",
                         help="permit I/O or mutation while introspecting")

    doctor = sub.add_parser("doctor", help="environment and backend report")

    bench = sub.add_parser("bench", help="run the benchmark suite")
    bench.add_argument("--sizes", nargs="+", default=["small", "medium"],
                       choices=["small", "medium", "large"])
    bench.add_argument("--skip-dense", action="store_true")
    bench.add_argument("--skip-structured", action="store_true")
    bench.add_argument("--skip-pipeline", action="store_true")
    bench.add_argument("--skip-memory", action="store_true")
    bench.add_argument("--skip-autodiff", action="store_true")
    bench.add_argument("--regen-data", action="store_true",
                       help="regenerate benchmark datasets even when present")

    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            _compiler.run_file(args.file)
        elif args.command == "check":
            from pathlib import Path

            source = Path(args.file).read_text(encoding="utf-8")
            _compiler.check_source(source, filename=args.file)
            print(f"{args.file}: OK")
        elif args.command == "diagnose":
            from .frontend import diagnose as _diagnose
            import json as _json

            report = _diagnose.diagnose_file(args.file)
            if args.format == "json":
                print(_json.dumps(report, indent=2))
            else:
                print(_diagnose.format_human(report))
            return 0 if report.get("ok") else 1
        elif args.command == "errors":
            from .core.errors import ERROR_REGISTRY
            from .frontend import diagnose as _diagnose

            if args.code:
                entry = ERROR_REGISTRY.get(args.code.upper())
                if entry is None:
                    print(f"unknown error code '{args.code}'", file=sys.stderr)
                    return 1
                print(f"{args.code.upper()} ({entry['category']}): "
                      f"{entry['title']}\n  {entry['hint']}")
            else:
                print(_diagnose.render_catalogue())
        elif args.command == "repl":
            from .frontend.repl import run_repl

            return run_repl()
        elif args.command in ("graph", "explain", "profile"):
            return _introspect(args)
        elif args.command == "doctor":
            _doctor()
        elif args.command == "bench":
            return _bench(args)
        return 0
    except TensorLangError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception:  # noqa: BLE001 - surface unexpected failures honestly
        traceback.print_exc()
        return 2


def _binding(namespace: dict, var: str):
    if var in namespace:
        return namespace[var]
    import difflib as _difflib

    pool = [k for k in namespace
            if not k.startswith("_") and k not in ("__name__", "__file__")]
    close = _difflib.get_close_matches(var, pool, n=1, cutoff=0.6)
    hint = f" Did you mean '{close[0]}'?" if close else ""
    print(f"unknown binding '{var}'.{hint} Available: {', '.join(sorted(pool))}",
          file=sys.stderr)
    return None


def _introspect(args) -> int:
    from pathlib import Path as _Path

    from .frontend import effects as _effects
    from .frontend import lexer as _lexer

    import ast as _ast

    try:
        translated = _lexer.translate(_Path(args.file).read_text(encoding="utf-8"))
        effects = _effects.find_effects(_ast.parse(translated))
    except Exception:  # pragma: no cover - unreadable source surfaces below
        effects = []
    print(f"note: {args.file} executes to build its graphs; "
          f"introspection observes the resulting bindings", file=sys.stderr)
    if effects and not args.allow_effects:
        shown = ", ".join(f"{name} at line {lineno}"
                          for name, lineno in effects[:3])
        print(f"refusing to introspect {args.file}: it may perform I/O or "
              f"mutation ({shown}). Re-run with --allow-effects to proceed.",
              file=sys.stderr)
        return 1
    namespace = _compiler.run_file(args.file)
    if args.command == "graph":
        from .runtime.inspect import format_node, to_dot

        if args.var is None:
            for name in sorted(k for k in namespace
                               if not k.startswith("_")
                               and k not in ("__name__", "__file__")):
                print(f"{name} : {_summary(namespace[name])}")
            return 0
        value = _binding(namespace, args.var)
        if value is None:
            return 1
        root = getattr(value, "root", getattr(value, "_root", None))
        if root is None:
            print(_summary(value))
            return 0
        print(to_dot(root) if args.dot else format_node(root))
        return 0
    value = _binding(namespace, args.var)
    if value is None:
        return 1
    if args.command == "explain":
        if not hasattr(value, "explain"):
            print(f"{args.var} ({_summary(value)}) has no execution plan; "
                  f"apply it to data first", file=sys.stderr)
            return 1
        print(value.explain())
        return 0
    if not hasattr(value, "profile"):
        print(f"{args.var} ({_summary(value)}) has no profile; apply it to "
              f"data first for a lazy result", file=sys.stderr)
        return 1
    print(value.profile())
    return 0


def _summary(value) -> str:
    if hasattr(value, "abstract"):
        try:
            abstract = value.abstract()
            if abstract.get("kind") == "scan":
                return f"scan{abstract.get('fields')}"
            if "params" in abstract:
                return f"{abstract['kind']} operator{abstract['params']}"
            if "shape" in abstract and abstract["shape"] is not None:
                return f"{abstract['kind']}{abstract['shape']}"
            return str(abstract.get("kind", type(value).__name__))
        except Exception:  # pragma: no cover - defensive
            pass
    return type(value).__name__


def _doctor() -> None:
    import platform

    print(f"tensorlang {__import__('tensorlang').__version__}")
    print(f"python {platform.python_version()} ({platform.platform()})")
    print(f"cpus: {__import__('os').cpu_count()}")
    for package in ("numpy", "polars", "sympy", "scipy", "sklearn",
                    "autograd", "pytest"):
        try:
            module = __import__(package)
            version = getattr(module, "__version__", "present")
        except ImportError:
            version = "not installed"
        print(f"{package}: {version}")


def _bench(args) -> int:
    import sys as _sys
    from pathlib import Path as _Path

    bench_dir = _Path(__file__).resolve().parent.parent.parent / "benchmarks"
    _sys.path.insert(0, str(bench_dir))
    try:
        import run_all as _run_all
    except ImportError:
        print("benchmark suite not found next to the installed package; "
              "run benchmarks/run_all.py from a source checkout",
              file=sys.stderr)
        return 2
    argv = ["run_all.py", "--sizes", *args.sizes]
    for flag in ("skip_dense", "skip_structured", "skip_pipeline",
                 "skip_memory", "skip_autodiff", "regen_data"):
        if getattr(args, flag):
            argv.append("--" + flag.replace("_", "-"))
    old_argv, _sys.argv = _sys.argv, argv
    try:
        return _run_all.main()
    finally:
        _sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())
