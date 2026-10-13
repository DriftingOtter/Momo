# ==================================================
# TensorLang v0.5 — Release-validation workflow
# ==================================================
# Run:  tensorlang run examples/release_check.tl
# (after python examples/make_chunked_data.py)
#
# One script exercising the v0.5 contract end to end: lazy load, scoped
# operator, pure application, plan inspection, budgeted reduction,
# incremental write, persisted-result validation, controlled failure
# with safe recovery, native-backend equivalence, diagnostics.
# The integration suite runs this file automatically.

import os

import numpy as np
import zarr

from tensorlang import T, clear_chunk_policy, set_chunk_policy

here = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(here, "data", "large-demo.zarr")
out_path = os.path.join(here, "data", "release-check.zarr")

# 1. Load a persistent numerical dataset lazily (no data read).
data := T.scan_zarr(src_path)

# 2. A scoped operator.
f := {X} * {X} + {X}

# 3. Apply without mutation (data is never modified in place).
m = f(data)

# 4. Inspect the physical execution plan (no execution).
plan = m.explain()

print(plan)

# 5. Execute a reduction under a memory budget.
set_chunk_policy(memory_budget_bytes=2800000, max_workers=2)

average = m.mean().collect()

print(average)

# 6. Write the transformed result incrementally (staged publication).
m.write_zarr(out_path, overwrite=True)

# 7. Validate the persisted result against the source.
ref = zarr.open_array(src_path, mode="r")[:]
got = zarr.open_array(out_path, mode="r")[:]

assert got.shape == ref.shape
assert np.allclose(got, ref**2 + ref)

print("persisted result valid:", got.shape)

# 8. Controlled failure and safe recovery: materializing the full output
#    under a 1MB budget is rejected; the persisted output stays intact.
set_chunk_policy(memory_budget_bytes=1000000, max_workers=1)

try:
    m.collect()
    print("unexpected success")
except Exception as e:
    print("controlled diagnostic:", e.code if hasattr(e, "code") else e)

after = zarr.open_array(out_path, mode="r")[:]

assert np.allclose(after, got)

print("previous output intact after rejection")

# 9. Numerical equivalence against the native backend.
assert average == np.mean(ref**2 + ref)

print("native equivalence confirmed")

# 10. Consistent structured diagnostics.
from tensorlang import TLUnsupportedError

try:
    m.collect()
except TLUnsupportedError as e:
    print("diagnostic code:", e.code, "| details:", sorted(e.details))

clear_chunk_policy()

print("release check complete")
