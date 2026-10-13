# ==================================================
# TensorLang v0.5 M4 — Reliable execution demonstration
# ==================================================
# 1. Generate data:  python examples/make_chunked_data.py
# 2. Run:            tensorlang run examples/reliable_execution.tl
#
# A memory-aware engine that writes persistent data must behave
# predictably when a job fails halfway. Local destinations use staged
# publication: complete and validate in a staging directory, then rename
# atomically. Failed runs never publish partial outputs and never destroy
# preexisting valid destinations (overwrite needs a completed replacement).

import os

from tensorlang import T, clear_chunk_policy, set_chunk_policy
from tensorlang.runtime.chunk_plan import (
    list_stale_staging,
    recover_stale_staging,
)

here = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(here, "data", "large-demo.zarr")
out_path = os.path.join(here, "data", "reliable-demo.zarr")

# 1. A lazy persistent source (reads no data).
data := T.scan_zarr(data_path)

# 2. An ordinary reusable operator.
f := {X} * {X} + {X}

# 3. Memory-budget admission.
set_chunk_policy(memory_budget_bytes=2800000, max_workers=2)

m = f(data)

# 4. A normal streamed write (staged publication under the hood).
m.write_zarr(out_path, overwrite=True)

# 5. A safely staged local output: describe first (read-only, no writes).
print(m.describe_write(out_path, overwrite=True))

# 6. Output inspection and validation.
print(m.explain())

# 7. A deliberately injected execution failure (test hook driving the
#    same code path as a real mid-write failure: partial staging is
#    cleaned, the destination is untouched).
import zarr

from tensorlang.runtime import chunk_plan as hooks

before = zarr.open_array(out_path, mode="r")[:]

hooks._TEST_HOOKS["fail_after_partitions"] = 1

try:
    m.write_zarr(out_path, overwrite=True)
    print("unexpected success")
except Exception as e:
    print("expected diagnostic:", e)

hooks._TEST_HOOKS["fail_after_partitions"] = None

# 8. The failed operation did not publish an incomplete output: the
#    previous valid destination is byte-identical.
after = zarr.open_array(out_path, mode="r")[:]

print("destination intact:", bool((before == after).all()))

# 9. Recovery of staging artifacts (none leak from the cleaned failure;
#    the helper reports whatever remains).
print(recover_stale_staging(out_path))
print("stale now:", list_stale_staging(out_path))

# 10. Clear execution diagnostics: lifecycle stage of the last run.
from tensorlang.runtime.execution import last_record

print(last_record()["stage"], last_record()["error"].get("code"))

clear_chunk_policy()
