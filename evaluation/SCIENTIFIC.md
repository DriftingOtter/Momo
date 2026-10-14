# Workload 1 — Lennard-Jones cluster (v2, natural syntax)

v0.6 M2 rewrite (`evaluation/workload_scientific.py`; v1 frozen as
`evaluation/workload_scientific_v1.py`): axis reductions replace
matmul-with-ones, `.T` replaces explicit transpose calls, no manual
reshape scaffolding. Same model, data, native reference, thresholds.
32 → 27 user LOC (target was ~24; readability kept over number-golfing);
all workarounds gone.

`evaluation/workload_scientific.py` (v2), driver `evaluation/run_scientific.py`,
results `evaluation/results/scientific_v2.json` (v1 frozen as
`workload_scientific_v1.py` / `scientific_v1.json`). N=2000 particles
(4M-pair interactions); medians of 5 repeats after warmup; per-process
peak RSS via `/usr/bin/time -l`; equivalence asserted inside every run.

## What was built

Total pair energy, analytic-force gradient (reverse-mode `grad` vs a
hand-derived NumPy reference), inertia tensor + `eigh` principal
moments, center of mass, and a symbolic `diff` of the pair law
cross-checked against finite differences.

## Correctness — all equivalent

| Check | Result |
|---|---|
| Energy TL vs NumPy | 14-digit agreement |
| Gradient TL vs analytic native | rtol 1e-6 (after fixing a sign error in the *reference*) |
| Inertia TL vs NumPy | exact (`allclose`) |
| `eigh` values vs `np.linalg.eigvalsh` | exact |
| Center of mass vs `mean(axis=0)` | exact |
| Symbolic dV/dr vs finite differences | rtol 1e-5 |

## Performance (N=2000, medians; v2 rewritten workload)

| Phase | TensorLang | NumPy | Ratio |
|---|---|---|---|
| build (planning, one-time) | 0.0093 s | — | — |
| energy | 0.10 s | 0.15 s | **0.68x** |
| gradient | 0.87 s | 0.21 s | **4.15x** (see attribution below) |
| inertia+eigh | 0.0003 s | 0.0000 s | µs-noise |
| center of mass | 0.0001 s | 0.0000 s | µs-noise |
| force law (symbolic, one-time) | 0.0002 s | — | — |

v1-vs-v2 on the same engine: identical except energy (v2 1.44x v1).
That delta is NumPy kernel behavior, not framework overhead: the
natural `(D*D).sum(axis=2)` costs 41 ms (strided) where the
matmul-with-ones workaround costs 6 ms (GEMM) — native code pays the
same 41 ms inside its own 0.15 s. Framework overhead is eliminated in
both versions; v2 trades 35 ms of kernel time for the natural spelling.

Gradient attribution (warmed medians; timing boundaries traced in
`_GradShared.reverse`: the backward timer starts *before* the forward
call, so `backward` *includes* forward — total ≈ backward ⊇ forward,
not a sum). Measured: total 0.87 s, forward 0.08 s, backward-inclusive
0.87 s (backward-exclusive ≈ 0.79 s). 52 cotangents released per pass.
Peak RSS: **1059–1067 MB vs 433 MB (≈2.45x)** — target ≤2.5x held.

A well-conditioned supplement (N=1000, jittered lattice, min
separation enforced; max|g|=529) agrees to abs_err 2.4e-12 at
rtol=1e-9 — separating AD correctness from large-magnitude effects
(the raw workload's 8.4M absolute discrepancy is magnitude-driven:
relative agreement holds throughout).

## Memory analysis (v0.5.3 baseline → v0.6 M1)

Baseline backward-pass deltas: N=500 → 261MB, N=1000 → 1035MB, N=2000 →
2234MB (~0.5–1KB per pairwise element). A control gradient (`sum(x²)`,
no broadcast/where/matmul) showed ~0MB delta at all sizes — general AD
is lean; retention concentrated in the LJ backward cotangent dict plus
the pinned forward tape (the latter required by the `forward_runs == 1`
repeat-evaluation guarantee).

M1 changes: reverse loop pops each cotangent after propagation (52
released per LJ pass; DiffVar readout entries survive; accumulation
order unchanged — shared-branch correctness preserved by
reverse-topological processing). Forward tape stays pinned across
applies (tested guarantee); forward *evaluation* dedupes via
structural aliasing. Result: 4026 → ~1060 MB peak (≈2.45x native).

## Expressiveness (v2: 27 LOC TL vs 20 native; v1 baseline: 32)

v1 workarounds, all removed in v2: axis sums via matmul-with-ones (now
`.sum(axis=)`), newaxis via reshape (still reshape-based construction —
`None`-indexing remains unsupported, documented), diagonal handling via
`where` (unchanged — correct tool for the job), data enters operators
only through parameters (unchanged — ontology). The reusable-operator
form (`tl_build` once, apply many times) is genuinely ergonomic;
planning cost (9ms) amortizes to zero. v1 retained frozen for
attribution; LOC target (~24) held approximately with readability
preserved over number-golfing.

Ergonomics defect found during evaluation and fixed in M2: `X.T` on an
operator used to build successfully but fail at application with
`TL0499: cannot access field 'T' on a dense value`. `.T` now builds a
transpose operator; field-confusion errors suggest the operator
function.

## Verdict

Correctness and symbolic/AD cross-checking are excellent. M1 execution
changes (below) brought forward execution under native parity and the
gradient from 11.9x to 4.2x with RSS at ≈2.45x. Remaining gradient cost
is backward cotangent arithmetic (`_rev_binop`/`_unbroadcast`
reductions — profiled post-rewrite at 0.37 s + 0.34 s of 0.88 s; no
separate optimizer recommended). F1 (executor temporaries) and F2
(buffer release) are done within the repeat-evaluation guarantee; F3
axis reductions and F4 `.T` ergonomics completed in M2 (workload v2).

### M1 root causes (profiled, not assumed)

1. **Graph bloat, not dispatch**: 42 BinOps for ~12 user ops —
   per-composition whole-tree copies plus per-reference doubling
   (measured growth 17→34 nodes across one `where`). Per-call wrapper
   overhead is only ~6%.
2. **Duplicate evaluation**: 32/42 binop evaluations value-identical.
3. **AD transient**: full cotangent dict + rule temporaries (~3.5GB);
   pinned tape (~0.4GB) is required guarantees, not waste.

### M1 mechanisms

- `remap_params`: sharing-preserving rebuild (identical output, zero
  copies on no-op paths; uid policy untouched).
- Executor: scoped memo with use-count eviction + structural-key
  dedupe (post-substitution duplicates share leaf objects, so id-keyed
  leaves unify copies soundly); taint rule (unkeyable children poison
  parents — proven by gradient-descent divergence when disabled);
  AD/scan bypasses preserved.
- AD `reverse()`: pop-after-propagate cotangent release (DiffVars
  survive for readout); `last_released_cotangents` exposed.
- Forward tape aliasing: skipped subtrees lockstep-alias canonical
  values so per-node tape stays complete.
