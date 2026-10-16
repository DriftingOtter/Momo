"""Portability lint: fail on developer-specific path assumptions.

Scans tests, examples, packaged scaffolding, CI scripts, and
documentation for hardcoded checkout paths, home directories,
absolute temporary locations, and ambient-environment requirements.
Intentional fixtures use the documented allowlist below — never add a
broad exemption to hide a real assumption.

Usage: python scripts/check_portability.py [--root DIR]
Exit 1 when a non-allowlisted finding exists. No network access.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

RULES = [
    ("macos-home", re.compile(r"/Users/"),
     "hardcoded macOS home directory"),
    ("linux-home", re.compile(r"/home/"),
     "hardcoded Linux home directory"),
    ("home-env", re.compile(r"\$HOME|\$\{HOME\}"),
     "assumed $HOME location"),
    ("tilde-home", re.compile(r"(?<![\w/])~/"),
     "~/ path assumption (use tmp_path or derived roots in tests)"),
    ("windows-home", re.compile(r"C:\\Users|C:/Users", re.IGNORECASE),
     "hardcoded Windows profile path"),
    ("macos-private", re.compile(r"/private/"),
     "macOS /private symlink target (leaked tmp_path?)"),
    ("hardcoded-tmp", re.compile(r"/tmp/"),
     "absolute /tmp location (use tmp_path in tests)"),
    ("pythonpath", re.compile(r"PYTHONPATH"),
     "ambient PYTHONPATH requirement"),
]

# (path substring, rule name, reason) — precise, reviewed exemptions.
ALLOWLIST = [
    ("tests/test_tl.py", "macos-home",
     "asserts the project template contains no /Users/ paths"),
    (".github/workflows/ci.yml", "hardcoded-tmp",
     "CI-only scratch locations (/tmp/smoke, /tmp/tlci)"),
    ("docs/GETTING_STARTED.md", "hardcoded-tmp",
     "instructional user-local copy target (/tmp/repro)"),
    # Intentional fixtures: .tl source strings whose paths are rejected
    # statically (TL0302/TL0305) and never touch the filesystem.
    ("tests/test_declaration_effects.py", "hardcoded-tmp",
     ".tl placeholder path, rejected before execution"),
    ("tests/test_effects.py", "hardcoded-tmp",
     ".tl placeholder path, rejected before execution"),
    ("tests/test_modules.py", "hardcoded-tmp",
     ".tl placeholder path, rejected before execution"),
    ("tests/test_reliability.py", "hardcoded-tmp",
     "declarative describe_write/write_zarr strings, never executed"),
    # Legitimate product behavior: REPL history lives in the user's home
    # by design (not a test assumption or checkout path).
    ("src/tensorlang/frontend/repl.py", "tilde-home",
     "REPL history file location, user-scoped by design"),
]

SCAN_DIRS = ["tests", "examples", "src/tensorlang", ".github", "docs"]
SCAN_SUFFIXES = {".py", ".tl", ".toml", ".yml", ".md", ".cfg", ".ini"}


def iter_files(root: Path):
    for base in SCAN_DIRS:
        here = root / base
        if not here.is_dir():
            continue
        for path in sorted(here.rglob("*")):
            if path.is_file() and path.suffix in SCAN_SUFFIXES \
                    and "__pycache__" not in path.parts:
                yield path


def check_file(path: Path, root: Path) -> list[str]:
    rel = str(path.relative_to(root))
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    findings = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for name, pattern, what in RULES:
            if not pattern.search(line):
                continue
            if any(sub in rel and rule == name
                   for sub, rule, _ in ALLOWLIST):
                continue
            findings.append(f"{rel}:{lineno} [{name}] {what}: "
                            f"{line.strip()[:100]}")
    return findings


def check_tree(root: Path) -> list[str]:
    findings = []
    for path in iter_files(root):
        findings.extend(check_file(path, root))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".",
                        help="repository root to scan")
    args = parser.parse_args(argv)
    findings = check_tree(Path(args.root).resolve())
    for finding in findings:
        print(finding)
    if findings:
        print(f"{len(findings)} portability finding(s); fix or add a "
              f"reviewed allowlist entry with a reason.")
        return 1
    print("portability: no developer-path assumptions found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
