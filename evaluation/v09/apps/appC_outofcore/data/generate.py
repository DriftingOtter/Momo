"""512MB float64 zarr source (seeded chunks, no network)."""
import numpy as np
import zarr

def main(outdir, rows=8000, cols=8000, chunk=1000, seed=5):
    import os
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)
    path = os.path.join(outdir, "source.zarr")
    z = zarr.open_array(path, mode="w", shape=(rows, cols),
                        chunks=(chunk, chunk), dtype="float64")
    for i in range(0, rows, chunk):
        z[i:i + chunk, :] = rng.normal(0, 1, (min(chunk, rows - i), cols))
    return {"shape": [rows, cols], "mb": rows * cols * 8 / 1e6}
if __name__ == "__main__":
    import sys
    print(main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/v09c-data"))
