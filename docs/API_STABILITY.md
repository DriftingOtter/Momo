# TensorLang public API and compatibility status (v0.9 M3)

Provisional contract inventory. No v1.0 guarantee is established;
classifications reflect v0.9 M0–M3 evidence only.

Conventions used below — **A** candidate stable, **B** needs
clarification, **C** needs correction, **D** experimental/internal.

## Language core (A)

- `:=` declarative vs `=` runtime assignment; `{X}` uid-scoped
  parameters; operator definition, application (`f(data)`), and `>>`
  composition; `.inplace()` explicit mutation with aliasing-by-identity
  and rebinding-never-mutates; TL0305 fail-closed validation.
- `T(...)` construction (scalars/sequences/mappings/arrays/frames);
  rank-0 scalars collect to scalars.
- Dense arithmetic/broadcasting (NumPy rules); `collect()` boundaries;
  snapshot semantics; ownership (NumPy views, single copy at graph
  boundary).

## Data and math (A, with noted B items)

- Scans, joins (explicit keys; order backend-defined), grouping,
  `count` row/value semantics, `to_dense` policies
  (`nulls="nan"|"error"`, `TL0418`), `is_null`/`not_null`,
  `fill_null`/`drop_null`.
- Same-domain `&`/`|` fusion; `TL0512` for unproven domains; `~`
  order-preserving; dense positional logic (B: `==` of masks unfused
  by design).
- `sin/cos/tan/exp/log/sqrt`, reductions with `axis=`/`keepdims`
  (`TL0511`), `solve`/`svd`/`eigh` (square/symmetric contracts),
  `grad`/`value_and_grad`/`vjp` (first-order, fitted-affine rules).
- `standardize`/`minmax`/`normalize`/`impute`/`pca` + `fit` returns-new
  semantics; `TL0419` unfitted-application rejection; pipelines fit
  in order without refitting.

## Modules and sessions (A, B details)

- Sibling/dotted/alias imports; file-relative resolution;
  Python-first precedence; inert namespace parents; isolated
  namespaces; per-session `(path, name)` cache identity;
  `TL0310`–`TL0315` diagnostics with chains; `TL0315` thread guard.
- B: REPL session retention across `:reset` boundaries is by design
  but subtle; `extra=` `=`-override is silent-but-documented.

## Project workflow (A-minus: new in v0.9, stable by test)

- `tl new/init/add/remove/sync/run/check/test/doctor`; manifest
  discovery (nearest `pyproject.toml` with `[tool.tensorlang]`);
  `[project]` single dependency truth; `uv.lock` reproducibility
  record; freshness states (`fresh`/`stale:*`/`no-state`/
  `no-environment`); `tl sync --locked` strict path; argv forwarding
  (`sys.argv[3:]` are app args); exit-status propagation.

## C resolved in v0.9

F-B4 (mask fusion/`TL0512`), F-A2 (canonical identity + cleanup),
unfitted timing (`TL0419`).

## D experimental/internal

`trusted_names`/`injected_names` plumbing, chunked-AD/diff deferrals,
REPL history, benchmark harness internals, `tl build`/`publish`
(non-existent by design).

## Pre-1.0 policy direction

Additive changes free; correctness fixes ship with codes + migration
notes; diagnostic codes stable once documented (no renumbering without
cause); Python support follows the CI matrix (3.12–3.14); dependency
floors pinned per release; manifest evolution additive-only;
persistent-data formats validated by round-trip tests before any
change. v1.0 contract requires an independent-application evidence
round (v0.9 M0-style) against the frozen API.
