# M0 — Language contract (core semantics + effects)

Baseline: v0.9.0 (`bbbef00`, tag `v0.9.0`), branch `evaluation/v1-readiness`.
Method: public `.tl` + `run_source`/`run_file` only; no code changes.
Spec: `LANGUAGE_SPEC.md` §§3–9; boundary `docs/EFFECTS.md`; codes `src/tensorlang/core/errors.py:113-132`.

## 1. `:=` vs `=` (verified)

- `x := T(...)`, `f := {X}*2`, `h := f >> g` bind declarative Tensor/Operator — pass.
- `r := f(t)`, `r := f(t).collect()`, `y := x+1` (Tensor) rejected (`TL0301`/`TL0303`) — pass; runtime form is `r = f(t)` + `collect()`.
- `x = f` binds the operator itself (`x is f`), no application — pass (§3:65).
- `a := T` twice same scope → `TL0201`; `if`-block double → `TL0201`; function-scope double → `TL0201` — pass.
- `for`-loop body `a := T` passes static check (one AST node) and re-executes per iteration — implemented as static property (see Ambiguity A).
- `a := T` then `a = T` rebinds fine; outer-shadow allowed; `extra=` injection: `=` keeps module precedence, `:=` over injected name → `TL0203`, dunder injected names → `TL0203` — pass.

## 2. Effect enforcement before execution (TL0305)

Static classifier `frontend/effects.py:92-204` + parser wiring `frontend/parser.py:220-231`; runtime value check `frontend/compiler.py:29-60`.
Side-effect sentinels prove no execution before diagnosis:

- `x := print('hi')`, `x := os.system('touch sentinel')`, alias `p.join`, nested `T.scan_parquet(os.path.join(...))`, `len`, local `def grab`, `math.pi`, foreign subscript — all `TL0305`, sentinel files absent — pass.
- `x := T.read(...)` → `TL0302`; `t.write_parquet`, `f(t).inplace()`, `t.collect()`, `s.fit(t)` → `TL0302`/`TL0303` — pass.
- `s := T.scan_parquet(path)`, `f := sin({X})`, `g := diff(f)`, `g := grad(f)` — pass (declarative).
- Runtime `=` interop allowed: `p = os.path.join('a','b')` → `a/b` — pass (effectful by design, not pure).
- Module shadowing fail-closes: `import os as T`, `T = 5`, `def sin`, walrus/for-arg shadowing → later `:= T(...)` → `TL0305` — pass.
- Trust: identity-seeded `tensorlang` imports pass; `from tensorlang import *`, `import tensorlang [as tl]`, aliased `T as TT` pass — pass.
- Modules: `from calib import calibrate` then `r := calibrate(data)` → `TL0305` (imported name untrusted); `pipe := calibrate >> calibrate` passes — pass.
- Residual (documented, `docs/EFFECTS.md:27-34`): custom `__add__` / `__format__` run under `:=` (dunder/format paths execute). Calls/attr/subscript are covered; dunder/format are disclosed non-guarantees, not sandboxing.

## 3. Tensor/Operator model

- `{X}` scoping: spelling-independent identity (`{A}+{A}` ≡ `{X}+{X}`); per-definition UID (`f is g` False); factory functions give distinct objects, same numerics — pass.
- Explicit application: `h := f >> g` left-to-right (`h(x)=g(f(x))`); op-to-op `f(g)` → `TL0499`; arity `f(t,t)` → `TL0503`; `h := f >> t` → `TL0417` — pass.
- No implicit mutation: `r=f(t)` leaves `t`; `f(t).inplace()` mutates target + aliases observe; pre-mutation graph snapshot-preserved; `u=t; t=T(new)` never mutates `u` — pass.
- Lazy boundary: `r = f(t)` is `Lazy`; `.collect()` materializes; per-object cache + `.release()` — pass.
- Structured: frame assign eager; multi-input `.inplace()` → `TL0601`; scan-descriptor inplace → `TL0602` — pass.

## 4. Ambiguities (classified, no fix in M0)

A. Loop-body `:=` (static vs per-iteration redeclare) — code is static-property reading.
B. `declared` trust set file-flat, not scope-local (`parser.py:55-57` admits approximation).
C. Attribute/subscript admission: any trusted-root access admitted, value check decides (vs closed whitelist).
D. Literal-receiver method exemption (`'hi'.upper()` passes) undocumented in spec.
E. Plain-value derivations (`y := x`, `y := x.shape`) admitted; only `Lazy`/IO-provenance rejected.
F. Declarative set open (any `tensorlang`-imported name) vs closed example list.
G. `x = f` aliases (not copies) — harmless given operator immutability, unspecified.
H. "Without executing, including nested" reads absolute in §3; EFFECTS.md discloses dunder/format limits.

## 5. M0 verdict

Core contract + TL0305 fail-closed enforcement verified; no silent-execution gaps in the covered surface. Decision for M1: narrow wording fixes for A–H (no effect-system redesign); explicit literal-receiver + dunder policy; loop/`:=` static semantics documented.
