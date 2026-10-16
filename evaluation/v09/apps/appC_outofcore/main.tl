# Out-of-core pipeline: scan -> transform -> stats -> streaming write.
import os

from tensorlang import T, clear_chunk_policy, set_chunk_policy

from processing.statistics import tx
from storage.outputs import DEST

here = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(here, "data", "source.zarr")

set_chunk_policy(memory_budget_bytes=64000000, max_workers=2)
data := T.scan_zarr(src)
result = tx(data)
print(result.mean().collect())
print(result.explain())
result.write_zarr(DEST, overwrite=True)
clear_chunk_policy()
