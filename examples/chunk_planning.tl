# ==================================================
# TensorLang v0.5 M2 — Chunk-aware planning demonstration
# ==================================================
# 1. Generate data:  python examples/make_chunked_data.py
# 2. Run:            tensorlang run examples/chunk_planning.tl
#
# Two stores, different chunkings, one logical computation. The planner
# aligns layouts explicitly (visible below), reduces without loading,
# streams output, and rejects the unsafe contraction with a diagnostic.

import os

from tensorlang import T

here = os.path.dirname(os.path.abspath(__file__))

# 1. Two lazily loaded chunked tensors (different chunkings).
# (Path joins are runtime calls, so they are precomputed with '='.)
a_path = os.path.join(here, "data", "large-demo.zarr")
b_path = os.path.join(here, "data", "other-demo.zarr")
a := T.scan_zarr(a_path)
b := T.scan_zarr(b_path)

# 2. A reusable mathematical operator (chunk-layout independent).
f := {X} * {Y}

# 3. Compatible chunk alignment (identical layouts: no movement).
s = a + a

print(s.mean().collect())

# 4. Supported safe rechunking (subdivision recorded in the plan).
m = f(a, b)

print(m.explain())

# 5. A reduction and incremental output.
total = m.sum()

print(total.collect())

m.write_zarr(os.path.join(here, "data", "planned-demo.zarr"))

# 6. Introspection of the resulting physical plan.
print(m.profile())

# 7. A deliberately unsafe operation with a meaningful diagnostic.
try:
    bad = (m.T @ m).collect()
    print("unexpected success:", bad)
except Exception as e:
    print("expected diagnostic:", e)
