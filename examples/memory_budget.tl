# ==================================================
# TensorLang v0.5 M3 — Memory-budget demonstration
# ==================================================
# 1. Generate data:  python examples/make_chunked_data.py
# 2. Run:            tensorlang run examples/memory_budget.tl
#
# The same working-memory budget admits a dataset far larger than the
# budget itself: peak memory tracks chunk size, not data size. Streaming
# writes carry outputs larger than the budget; full materialization of
# such outputs is rejected with a structured diagnostic (never silently
# executed). Estimates are planning heuristics — only the operating
# system can enforce hard process containment (see LANGUAGE_SPEC §16).

import os

from tensorlang import T, clear_chunk_policy, set_chunk_policy

here = os.path.dirname(os.path.abspath(__file__))

# 1. Open a persistent chunked dataset lazily (reads no data).
# (Path joins are runtime calls, so they are precomputed with '='.)
data_path = os.path.join(here, "data", "large-demo.zarr")
data := T.scan_zarr(data_path)

# 2. Configure a working-memory policy: ~2.8MB budget, at most 2 workers.
#    The demo dataset (default 200K x 8 = 12.8MB) is several times larger.
set_chunk_policy(memory_budget_bytes=2800000, max_workers=2)

# 3. A reusable mathematical operator (chunk-layout independent).
f := {X} * {X} + {X}

m = f(data)

# 4. Inspect the physical plan. Full materialization (3.2MB > 2.8MB
#    budget) is rejected with a reason; the reduction and the streaming
#    write below still admit — one graph, purpose-dependent decisions.
print(m.explain())

# 5. Execute a reduction under the policy (scalar output, small estimate).
average = m.mean()

print(average.collect())

# 6. Write a result larger than the budget incrementally (streaming is
#    the documented alternative to materialization).
m.write_zarr(os.path.join(here, "data", "planned-budget.zarr"))

# 7. Display observed versus estimated memory use (labeled separately).
print(average.profile())

# 8. Reject a deliberately impossible memory budget with a diagnostic.
set_chunk_policy(memory_budget_bytes=100000, max_workers=1)

try:
    bad = m.collect()
    print("unexpected success:", bad)
except Exception as e:
    print("expected diagnostic:", e)

clear_chunk_policy()
