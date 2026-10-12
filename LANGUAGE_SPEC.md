# TensorLang Language Specification — v0.3

This document defines TensorLang independently of its backing libraries
(NumPy, Polars, SymPy, SciPy). Implementations must honor the observable
semantics below; backend choice, plan shape, and performance may vary.

Normative keywords: **must**, **must not**, **should** (recommended),
**may** (permitted).

## 1. Program structure

A TensorLang program is UTF-8 source in a `.tl` file using Python-compatible
syntax plus the constructs in §3–§5. The frontend translates `.tl` source to
an executable form via token-stream rewriting (no regular-expression
substitution), then validates declarations on the abstract syntax tree with
exact source locations. Ordinary Python control flow (`for`, `if`, `while`,
`def`, `import`, `with`) retains Python semantics.

`{X}` braces containing exactly one bare identifier denote a symbolic
parameter (§5). All other brace uses (dict/set displays, comprehensions,
f-string fields) retain Python meaning.

## 2. Values

- **Tensor**: an eager, mutable handle to data of kind `dense` (N-D array),
  `frame` (heterogeneous named fields), `series` (single named field), or
  `scan` (lazy source descriptor, §9).
- **Operator**: a reusable transformation with named parameters and an
  immutable expression graph. Subtypes: stateless operators,
  **fittable** operators (unfitted estimators), **fitted** operators
  (immutable learned state), and **pipelines** (ordered stage lists, §10).
- **Lazy**: a pending computation over snapshotted inputs (§7).
- Plain Python values (numbers, strings, lists, …) behave as in Python.

## 3. Declarative definition (`:=`)

`NAME := expr` introduces `NAME` in the current lexical scope
(module/function/class; control-flow blocks share the enclosing scope).
Redeclaring in the same scope is an error; shadowing an outer scope is
allowed; `=` rebinding afterwards is allowed.

`expr` is classified syntactically **and** by value:

| Class | Meaning | Examples |
|---|---|---|
| declarative | describes without executing | `T(...)`, `{X}**2`, `f >> g`, `diff(f)`, `T.scan_*` |
| runtime | needs execution | `f(data)`, `t.collect()`, `op.fit(data)` |
| effectful | I/O or mutation | `T.read(...)`, `t.write_*`, `o(x).inplace()` |

`:=` **must** reject runtime and effectful expressions; `=` **must** be used
for them. Data-dependent calls whose class is statically unknowable (e.g.
`sin(x)`: declarative for operator input, runtime for tensor input) are
decided by the value check: pending `Lazy` results and I/O-provenance
tensors are rejected. A lazy scan descriptor **must** be accepted (it reads
nothing). `x = f` binds the operator itself and **must not** apply it.

## 4. Runtime assignment (`=`)

`=` binds or rebinds names to runtime values. Assignment itself **must
never** imply operator application. `t["field"] = v` on a materialized
tensor assigns eagerly; on a pending computation it extends the graph
lazily (§7).

## 5. Parameters (`{X}`)

Brace parameters are lexically scoped to their operator definition.
Identity is the pair `(owner, name)` where `owner` is unique per
definition occurrence: repeated `{X}` in one definition is one parameter;
identical names in different definitions never collide. Diagnostics render
`f_x`-style names; identity **must not** rely on rendered names. `{X}`
**must not** be a global variable or global symbolic name. Single-parameter
operators are primary; multi-parameter operators bind positionally.

## 6. Application and composition

- Operators apply only via call notation `f(data)`. Arity mismatches and
  operator-to-operator application are errors.
- `f >> g` composes left-to-right (`h(x) = g(f(x))`) for unary-compatible
  operators, preserving independent parameter scopes in a new operator.
  Composing with a non-operator, or a non-unary right side, is an error.
- `f(data)` **must not** mutate its input and **may** return a pending
  computation.

## 7. Laziness (`.collect()`)

Every non-mutating operation **should** return a pending graph over
immutable snapshots of its inputs; `.collect()` executes it. Results:
dense arrays (rank-zero becomes a scalar), structured frames/series,
scalars, and exact symbolic values as-is. Graphs built before a mutation
**must not** change when the source is later mutated (snapshot semantics).
`Lazy.collect()` **may** cache; `.release()` drops the cache.

## 8. In-place execution (`.inplace()`)

`f(data).inplace()` is allowed only for an unambiguous single-input
operator application over a mutable `Tensor` (scan descriptors excluded).
The implementation **must** evaluate fully before touching the target,
then replace the target's logical contents atomically (shape, dtype, and
schema **may** change); on failure the input **must** be unchanged.
Aliasing is by object identity: all references observe the update;
rebinding never mutates. Multi-input or non-application graphs **must**
reject `.inplace()` with a descriptive error.

## 9. Tensors

- `T(...)` constructs from scalars, sequences, mappings (→ structured),
  backend arrays/frames, or keyword fields. Rank-zero scalars, vectors,
  matrices, N-D arrays, and heterogeneous frames are supported.
- Helpers: `linspace`, `arange`, `zeros`, `ones`; `read` (effectful);
  `scan_parquet`/`scan_csv` (declarative descriptors; globs and backend
  options allowed).
- Metadata: `kind`, `shape`, `rank`, `dtype`/`schema`, `fields`.
  Scans expose schema/fields from metadata only; `shape`/`len` are unknown
  until an explicit execution boundary (`collect`, `write_*`, `profile`).
- Indexing: integers, slices, Boolean masks (length-checked), field names,
  field lists. Heterogeneous frames **must not** be implicitly converted
  to homogeneous arrays. Conversions (`to_numpy`, `to_polars`) are explicit
  and **must** reject genuinely undefined cases.
- Dense snapshots are zero-copy read-only views; external arrays are
  copied once at the graph boundary. Frames use cheap clones.

## 10. Fittable operators and pipelines

- `standardize()`, `minmax(lo, hi)`, `normalize(norm)`,
  `impute(strategy[, fill_value])`, `pca(n_components[, whiten])` construct
  **unfitted** unary operators (invalid configuration fails at
  construction).
- `.fit(data)` is an explicit runtime boundary: it materializes `data`,
  learns parameters, and returns a **new** operator; the original is
  unchanged. Already-fitted stages are reused without refitting, so test
  data can never overwrite training statistics.
- Applying an unfitted operator is an error naming `.fit`. Learned state
  is immutable and exposed via `.parameters`; application input is never a
  statistics source.
- `>>` involving fittables builds a **pipeline** (ordered unary stages,
  plain or fittable). `pipeline.fit(data)` fits unfitted stages in order,
  threading each stage's transformed output into the next. Applying a
  fitted pipeline reuses learned state without refitting.
- Estimators consume numeric feature matrices: dense 1-D/2-D, single
  series, or all-numeric frames/selections. Shape-preserving transforms on
  mixed frames transform applicable numeric fields and preserve the rest
  (names, order); dimensionality-changing transforms (PCA) require
  all-numeric input. Missing fitted features are errors; extra columns
  pass through untransformed. Dense width mismatches are errors.
- PCA returns a rank-two frame with `pc0…` coordinates. Fitted application
  is non-mutating and lazy-compatible; `.inplace()` follows §8 (output
  kind may differ from input kind).
- Fit data with missing/infinite values is rejected (except `impute`
  itself); constant columns have defined outputs (standardize → 0,
  min-max → range lower bound); zero-norm rows stay zero; PCA requires
  `n_components ≤ min(n_samples, n_features)` and `n_samples ≥ 2`.
  Statistics use population conventions (variance `ddof=0`); PCA variance
  uses `n − 1`.

## 11. Structured operator semantics

The same operator has the same logical meaning across representations:

- Arithmetic/broadcasting on frames transforms applicable numeric fields,
  preserving names, order, and unrelated fields.
- Field comparisons yield per-record Boolean masks; `t[mask]` filters rows.
  Whole-frame comparisons yield Boolean frames, which **must not** be
  silently merged into record conditions.
- Joins require explicit keys (`cross` excepted); row order without an
  explicit `sort` is backend-defined.
- Reductions preserve labels (one-row frames); `by=` groups explicitly.
  `over()` broadcasts per-group aggregates back to row shape (unlike
  reductions).
- `@`, transpose, reshape on heterogeneous frames are rejected as
  genuinely undefined. Ambiguous operations raise descriptive errors;
  nothing is silently discarded, reordered, or invented.

## 12. Mathematics

- Elementwise: `sin cos tan exp log sqrt`; reductions; `norm`, `cov`,
  `corr`, `quantile`; tensor algebra `einsum`, `tensordot`, `moveaxis`;
  features `onehot`, `polynomial`; linear algebra `svd`, `eigh`, `solve`.
- Symbolic calculus (SymPy): `diff`, `simplify`, exact `integrate`.
  Definite integrals with no exact closed form raise; numerical fallback
  is never silent. SymPy handles mathematical subexpressions only.
- Numerical linear algebra uses stable LAPACK-backed routines. `svd`
  returns a `(U, S, Vh)` triple sharing one computation; `eigh` requires a
  square symmetric matrix (ascending values) and rejects general matrices
  toward `svd`; `solve` requires square non-singular systems. Sign
  ambiguity of singular/eigenvectors is inherent: compare reconstructions,
  variances, or subspaces — never raw signs.
- `cov`/`corr` default to NumPy `rowvar` for dense data; frames always use
  columns-as-variables with labeled output.

## 13. Execution model

Compatible structured subgraphs lower into a single backend lazy query
(predicate/projection pushdown delegated); anything else evaluates
node-by-node with identical results. `explain()` shows the logical graph
plus the physical plan or fallback reason without executing; `profile()`
reports measured time, memory, backend, and shape — never estimates.
For gradient graphs, `profile()` additionally reports measured forward
and backward phase times. Fitting and I/O are explicit execution
boundaries. Multi-output nodes share one computation. Dense evaluation is
node-by-node (one temporary per operator); no loop fusion is assumed.

## 14. Automatic differentiation (v0.4)

`grad(loss, wrt)` and `value_and_grad(loss, wrt)` construct first-class
operators; `vjp(loss, wrt)` constructs a vector-Jacobian operator applied
as `df(x, seed)`. Construction is declarative; execution is runtime.
`wrt` names an existing parameter (default: the sole parameter);
multi-parameter `wrt` returns gradients in order.

- The loss must be scalar-valued (dense rank-zero); vector outputs are
  rejected (use `vjp` with an explicit cotangent). Seeds must match the
  output shape/kind exactly.
- Differentiable inputs must be floating-point (dense float, float
  series, all-float frames); integer, Boolean, complex, string, and
  categorical inputs are rejected, never silently converted.
- Supported rules: elementwise arithmetic with broadcast reduction of
  adjoints, negation, abs (0 at 0), sin/cos/tan/exp/log/sqrt,
  sum/mean/min/max/std/var (ties split evenly; zero-variance std gives 0),
  matmul (1-D/2-D/batched), transpose/reshape/flatten, concat/stack/
  moveaxis, indexing with scatter (slices, integers, masks), where/clip/
  fill_null branch masks, linear solves (implicit), the vector 2-norm,
  and fitted affine transforms (standardize/minmax/l2-normalize/impute/
  PCA) with learned parameters held constant.
- Sorting, comparisons, logic, grouping, joins, row filtering of frames,
  SVD/eigendecomposition, tensor contractions, and unfitted estimators
  are rejected with precise diagnostics. Differentiation through `.fit()`
  is deferred; `grad()` of a gradient operator is rejected (no
  second-order); gradient values reused as data are constants.
- Application binds `DiffVar` input leaves holding snapshots, so later
  `.inplace()` mutations cannot alter constructed gradient computations.
  Unused parameters yield zeros. `value_and_grad` shares one forward
  evaluation between value and gradient. Integer intermediates compute in
  float64; results are cast to input dtypes.
- Structured gradients preserve labels: all-numeric frames yield frames
  with identical columns; series yield series with identical names.
  Heterogeneous frames on differentiable paths are rejected (select
  numeric fields explicitly).

## 14. Errors

Typed errors with source locations where applicable: syntax, scope,
declaration/effect, type, shape, mutation, symbolic, differentiation
(`TLDiffError`), unsupported. Messages **must** name the offending
operation, the representations involved, and the corrective action
(explicit selection, `.fit` first, `=` vs `:=`).

## 15. Version history

- **v0.1**: frontend, scoped operators, dense/structured/symbolic core,
  lazy execution, explicit mutation.
- **v0.2**: effect classification with provenance; scans; joins/sort/
  unique/conditional/window operators; single-query lowering with fallback;
  `explain`/`profile`; streaming writes; CoW snapshots; benchmarks.
- **v0.3**: fittable operators and sequential pipeline fitting with
  no-refit guarantees; ML preprocessing (standardize/minmax/normalize/
  impute/PCA); numerical linear algebra and tensor algebra; this
  specification.
- **v0.4**: reverse-mode automatic differentiation (`grad`,
  `value_and_grad`, `vjp`) over existing operator graphs with
  vector-Jacobian rules, fitted-affine differentiation, structured
  gradients with label preservation, and phase-split profiling.
