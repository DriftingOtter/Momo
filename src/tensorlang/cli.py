"""TensorLang command-line interface: ``tensorlang run|check file.tl``."""
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

    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            _compiler.run_file(args.file)
        elif args.command == "check":
            from pathlib import Path

            source = Path(args.file).read_text(encoding="utf-8")
            _compiler.check_source(source, filename=args.file)
            print(f"{args.file}: OK")
        return 0
    except TensorLangError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception:  # noqa: BLE001 - surface unexpected failures honestly
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
