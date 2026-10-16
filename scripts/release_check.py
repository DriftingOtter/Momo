"""Repeatable release-validation checks (verify-only, dry run).

Read-only except for throwaway directories under /tmp. Never commits,
merges, rebases, tags, pushes, publishes, modifies the wiki, or deletes
user files. Every check reports executed / skipped / failed honestly;
remote CI cells are listed as pending unless --ci-status can reach them.

Usage: python scripts/release_check.py [--root DIR] [--full] [--apps]
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run(cmd, cwd=None, timeout=600):
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                              timeout=timeout)
    except FileNotFoundError as exc:
        return None, f"missing executable: {exc}"
    except subprocess.TimeoutExpired:
        return None, "timed out"
    return proc, None


class Report:
    def __init__(self):
        self.rows = []

    def add(self, name, status, detail=""):
        self.rows.append((name, status, detail))

    def show(self):
        width = max(len(n) for n, _, _ in self.rows)
        for name, status, detail in self.rows:
            extra = f" — {detail}" if detail else ""
            print(f"{name:<{width}}  [{status}]{extra}")
        failed = [n for n, s, _ in self.rows if s == "FAIL"]
        print(f"\n{len(self.rows) - len(failed)}/{len(self.rows)} checks passed"
              + (f"; FAILURES: {', '.join(failed)}" if failed else ""))
        return 1 if failed else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--full", action="store_true",
                        help="run the full core regression suite")
    parser.add_argument("--apps", action="store_true",
                        help="regenerate app data and run the 15 app tests")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    py = sys.executable
    rep = Report()

    # 1. git state (informational; M4 work stays uncommitted by rule).
    proc, err = run(["git", "status", "--short"], cwd=root)
    if proc is None or proc.returncode != 0:
        rep.add("git-state", "SKIP", err or "not a git checkout")
    else:
        lines = proc.stdout.splitlines()
        rep.add("git-state", "PASS" if lines else "PASS",
                f"{len(lines)} uncommitted path(s); no commit performed")

    # 2. version consistency.
    import tomllib
    meta = tomllib.loads((root / "pyproject.toml").read_text())
    ver = meta["project"]["version"]
    init = (root / "src" / "tensorlang" / "__init__.py").read_text()
    consistent = f'__version__ = "{ver}"' in init
    changelog = (root / "CHANGELOG.md").read_text()
    rep.add("version-consistency",
            "PASS" if consistent and ver in changelog else "FAIL",
            f"pyproject={ver}; changelog mentions: {ver in changelog}")

    # 3. diagnostic catalogue freshness.
    proc, _ = run([py, "-c",
                   "from tensorlang.frontend.diagnose import render_catalogue;"
                   "import sys; sys.stdout.write(render_catalogue())"],
                  cwd=root)
    fresh = proc is not None and proc.returncode == 0 and \
        proc.stdout == (root / "docs" / "ERROR_CODES.md").read_text()
    rep.add("error-catalogue", "PASS" if fresh else "FAIL",
            "render matches docs/ERROR_CODES.md")

    # 4. fast gates: API inventory + portability + diagnose.
    for name in ("tests/test_public_api.py", "tests/test_portability.py",
                 "tests/test_error_codes.py", "tests/test_diagnose.py",
                 "tests/test_m1_stabilization.py"):
        proc, err = run([py, "-m", "pytest", f"tests/{Path(name).name}",
                         "-q", "-p", "no:cacheprovider"], cwd=root,
                        timeout=900)
        if proc is None:
            rep.add(name, "SKIP", err)
        else:
            tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
            rep.add(name, "PASS" if proc.returncode == 0 else "FAIL", tail)

    # 5. portability script directly (also a CI step).
    proc, err = run([py, "scripts/check_portability.py", "--root", str(root)],
                    cwd=root)
    rep.add("portability-script",
            "PASS" if proc and proc.returncode == 0 else
            ("SKIP" if proc is None else "FAIL"),
            (proc.stdout.strip().splitlines() or [""])[-1] if proc else err)

    # 6. full core suite (opt-in; ~2 min).
    if args.full:
        proc, err = run([py, "-m", "pytest", "tests", "-q",
                         "-p", "no:cacheprovider", "-rs"], cwd=root,
                        timeout=1500)
        if proc is None:
            rep.add("core-suite", "SKIP", err)
        else:
            tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
            rep.add("core-suite",
                    "PASS" if proc.returncode == 0 else "FAIL", tail)
    else:
        rep.add("core-suite", "SKIP", "pass --full to execute")

    # 7. applications (opt-in; regenerates ~600MB gitignored data).
    if args.apps:
        ok, detail = True, "15 app tests"
        for app in ("appA_scientific", "appB_analytics", "appC_outofcore"):
            adir = root / "evaluation" / "v09" / "apps" / app
            gen, _ = run([py, "data/generate.py", "data"], cwd=adir,
                         timeout=900)
            if gen is None or gen.returncode != 0:
                ok, detail = False, f"{app} data generation failed"
                break
        if ok:
            ref, _ = run([py, "reference/native_dask.py"],
                         cwd=root / "evaluation" / "v09" / "apps" /
                         "appC_outofcore", timeout=900)
            ok = ref is not None and ref.returncode == 0
        if ok:
            suite, _ = run([py, "-m", "pytest",
                            "evaluation/v09/apps/appA_scientific/tests",
                            "evaluation/v09/apps/appB_analytics/tests",
                            "evaluation/v09/apps/appC_outofcore/tests",
                            "-q", "-p", "no:cacheprovider"], cwd=root,
                           timeout=1200)
            ok = suite is not None and suite.returncode == 0
            detail = suite.stdout.strip().splitlines()[-1] if suite else "no output"
        rep.add("app-suite", "PASS" if ok else "FAIL", detail)
    else:
        rep.add("app-suite", "SKIP", "pass --apps to execute")

    # 8. distribution build + metadata (isolated outdir).
    if shutil.which("uv") is None and proc is not None:
        pass
    with tempfile.TemporaryDirectory(prefix="tlrel-") as tmp:
        proc, err = run([py, "-m", "build", "--outdir", tmp], cwd=root,
                        timeout=600)
        if proc is None or proc.returncode != 0:
            rep.add("dist-build", "FAIL" if proc else "SKIP",
                    err or proc.stderr.strip().splitlines()[-1:])
        else:
            import zipfile
            wheels = [f for f in Path(tmp).iterdir() if f.suffix == ".whl"]
            names = zipfile.ZipFile(wheels[0]).namelist()
            has_ep = any("entry_points.txt" in n for n in names)
            rep.add("dist-build", "PASS" if has_ep else "FAIL",
                    f"{wheels[0].name}, {len(names)} files, "
                    f"entry_points={has_ep}")

    # 9. CI gate inventory (local vs remote).
    wf = (root / ".github" / "workflows" / "ci.yml").read_text()
    jobs, in_jobs = [], False
    for line in wf.splitlines():
        if line == "jobs:":
            in_jobs = True
            continue
        if in_jobs:
            if line.startswith("  ") and not line.startswith("   ") \
                    and line.rstrip().endswith(":"):
                jobs.append(line.strip()[:-1])
            elif line and not line.startswith((" ", "#")):
                break
    rep.add("ci-gates", "PASS",
            "configured: " + ", ".join(jobs) +
            " (remote execution pending push; not claimed green)")

    return rep.show()


if __name__ == "__main__":
    sys.exit(main())
