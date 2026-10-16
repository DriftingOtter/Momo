# M1 — Public contract stabilization and release-blocker closure

Branch `evaluation/v1-readiness`, baseline v0.9.0 (`bbbef00`). All
changes uncommitted; no merge/tag/push. Before/after matrix first,
then per-finding records, then contract, regressions, and residuals.

## 1. Before/after status (8 blockers)

| # | Finding | Before (M0) | After (M1) | Acceptance |
|---|---|---|---|---|
| P0 | corrupt manifest ≈ missing project | `find_root` swallowed `TOMLDecodeError`, fell through to outer project or "no project found" | three states distinguished; nearest invalid manifest raises `invalid manifest <path>: <toml loc>`; `tl doctor` prints `project: INVALID` | 9 discovery/validation tests green |
| P1 | `tl add` rejects local wheels | `_PKG_RE` index-only | existing `.whl` resolved deterministically (project-relative inside root, absolute outside), forwarded via arg array to `uv add`; missing/non-wheel rejected actionably | unit matrix + guarded e2e (add→sync→import→apply→compose→run→locked-repro→missing-fails) green |
| P1 | truncated uv errors | last stderr line only | up to 8 trailing lines + exit status; stdout fallback; `user:pass@` redaction; no shell | summarizer/redaction unit tests green |
| P1 | nonexistent `.tl` raw traceback, exit 2 | `FileNotFoundError` escaped `run`/`check`/`diagnose`/graph paths | `TLSyntaxError[TL0103]` naming the path (exit 1) on all CLI paths; `run_file`/`diagnose_file` coded; never substitutes another file | CLI + API tests green; `test_diagnose` updated to the new contract |
| P1 | `to_dense` memory overclaim | spec read as whole-pipeline bound | contract states post-upstream admission (budget gates conversion working memory AFTER upstream materializes; not a hard RSS cap); `bridge.describe` already honest | describe-scope, unknown-stays-unknown, TL1020-before-alloc, numerical-equivalence tests green |
| P1 | staged-publication atomicity overclaim | "atomic rename" unqualified | per-rename POSIX atomic; backup-overwrite = recoverable multi-step, briefly absent destination, no power-loss durability; historical M-line qualified | `docs/RECOVERY.md` already accurate; existing recovery tests green |
| P1 | `check_source()` plumbing status | `trusted_names=`/`injected_names=` public, undecided | **Option A**: stable `(source, filename)`; plumbing deprecated (warns, still honored); policy records the window | stable + deprecation tests green |
| P1 | join-order semantics | "backend-defined" only | no-order-guarantee formalized (identity ≠ ordering ≠ logical equivalence); safe patterns documented (single-query derive, materialize together, sort-by-key, stable-id align) | order-independent + sort-deterministic + fused-mask-identity tests green; TL0512 preserved |

No open P0/P1 remains from the M0 list.

## 2. Root causes and corrections

- **P0**: discovery conflated "unparseable" with "absent" (`data = {}` fallback). Fixed by failing fast on `TOMLDecodeError`/`OSError` at the nearest manifest and treating non-dict `[tool.tensorlang]` as malformed; a manifest without the table is still "not a project". `load_manifest` additionally rejects non-string `name`/`version`.
- **Wheel**: no resolver existed. Fixed without touching uv internals: normalize-then-delegate (`uv add` records project-relative path sources, verified empirically; wheel copied with the project reproduces; absent file fails at resolve).
- **uv errors**: lossy single-line tail. Fixed with a pure summarizer (tested without uv) + redaction; exit codes unchanged.
- **Missing file**: five loading paths read files without a coded error. Fixed with one reader (`read_source_file`) used by `run_file`, `tensorlang check`, introspection pre-scan, and `diagnose_file`; new `TL0103` registered and catalogue regenerated (`docs/ERROR_CODES.md` byte-identical to renderer, enforced by test).
- **to_dense/persistence**: code and recovery docs were already honest; the normative spec overstated. Fixed by wording (no engine changes).
- **check_source**: the only in-repo caller never used the plumbing, so deprecation is behavior-preserving.
- **Join order**: one unsafe assumption (positional correspondence across collects) now explicitly forbidden with supported alternatives.

## 3. Language ambiguities (8, resolved by documentation + tests)

Loop-`:=` is one static declaration (re-executes per iteration);
trust set is file-flat (later cross-scope uses fail at runtime with
`NameError`); attribute/subscript rule is root-based (trusted roots
admitted statically, value check decides: `x.shape` tuple ok,
`x[0]` → `TL0301`; untrusted roots → `TL0305`); literal-receiver
methods admitted; plain values admissible (`Lazy`/I/O-provenance
only count as runtime); declarative set open by identity
(`grad` included); `x = f` aliases (immutable ⇒ indistinguishable);
dunders/`__format__` execute (disclosed non-sandbox). Each rule has a
conformance test in `tests/test_m1_stabilization.py`. One M0
classification was corrected in passing: `x[0]` on a Tensor fails
`TL0301` only with a trusted root; with a runtime-bound root it is
`TL0305` (spec example now states the precondition).

## 4. Stable API subset

`docs/API_STABILITY.md` now enumerates **66 Stable** (with verified
signatures and per-export behavior/errors), **16 Provisional**
(NaN-divergent `equal`/`not_equal`, order-sensitive `unique`,
windowed `over`/`rolling`, advanced algebra, expansion conventions,
carrier types), and Experimental/Internal (nonexistent `build`/
`publish`, chunked-AD deferrals, harness). A presence test pins the
82-export coverage (`66 + 16 + __version__`).

## 5. Compatibility/deprecation policy

`docs/COMPATIBILITY.md` (proposed, pending release approval):
semver; promised contracts (§2: syntax, semantics, stable
signatures minus deprecated plumbing, code identities not wording,
manifest/CLI/import/persistence promises); internal freedoms;
numerical tolerances (exact integer paths, `rtol=1e-4`/`atol=1e-6`
float/AD references, no bitwise promise, ~1e-12 group-noise
documented); correctness-driven change rule with v0.9 precedents;
three-step deprecation (announce + `DeprecationWarning` + removal ≥
next major); platform/dependency rules (CI matrix 3.12–3.14,
Windows Tier-2, `uv` required, URL redaction).

## 6. Regressions

- Core: **633 passed** (596 baseline + 37 new M1), zero skips.
- Apps: **15/15** (A/B/C incl. 1774-record + 16-digit parities).
- Benchmarks: small suite ran clean; outputs reverted; no numerical
  changes (M1 adds CLI/discovery/docs overhead only, never in the
  numerical path).
- Examples: 10/10 covered in M0; language contract unchanged.

## 7. Remaining P2 issues (deferred, not blockers)

`new` vs `init` unicode-name inconsistency; venv-without-pip
undocumented (now documented; template unchanged); `diagnose` JSON
schema undiscoverable; `needs_uv` dead marker; `[tool.uv.sources]`
URI escaping untested; manual release steps (script proposed for
M4); `to_dense` default null policy; aggregate column names;
grouped-float last-ulp; REPL retention; `extra=` override;
check_source removal target (≥ next major); Windows Tier-2.

## 8. Recommendation

**M1 complete against its acceptance criteria; no P0/P1 remains.**
Propose: adopt `docs/COMPATIBILITY.md` as the intended v1.0 policy
at release time, freeze the 66-name stable subset pending M4
validation, and scope M2 to the conditional thin distribution check
(`tl package --check` + path-spec `add` already delivered the
mechanism) only if real-library evidence demands more.
