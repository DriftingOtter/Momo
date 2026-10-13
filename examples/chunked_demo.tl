# ==================================================
# TensorLang v0.5 M1 — Out-of-core numerical pipeline
# ==================================================
# 1. Generate data once:  python examples/make_chunked_data.py
# 2. Run the pipeline:    tensorlang run examples/chunked_demo.tl
#
# One logical tensor, thousands of physical chunks: the operator below
# never knows the partitioning. Everything executes chunk-by-chunk;
# only explicit boundaries (collect/write) touch data.

import os

from tensorlang import T, sin

here = os.path.dirname(os.path.abspath(__file__))

# 1. A lazy persistent numerical tensor (reads no data).
data := T.scan_zarr(os.path.join(here, "data", "large-demo.zarr"))

# 2. An ordinary operator — same meaning on one array or many chunks.
transform := sin({X}) + {X}**2

result = transform(data)

# Inspect without materializing.
print(result.abstract())
print(result.explain())

# 3. Reduce without loading the full input.
average = result.mean()

print(average.collect())

# 4. Persist incrementally without full materialization.
result.write_zarr(os.path.join(here, "data", "processed-demo.zarr"))

print(result.profile())
