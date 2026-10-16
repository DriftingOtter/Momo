"""Independent dask reference: same transform, mean, and write."""
import os
import dask.array as da
import zarr

here = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(here, "..", "data")
x = da.from_zarr(f"{DATA}/source.zarr")
y = da.sin(x) + x ** 2
print("mean:", float(y.mean().compute()))
y.to_zarr(f"{DATA}/processed_ref.zarr", overwrite=True)
