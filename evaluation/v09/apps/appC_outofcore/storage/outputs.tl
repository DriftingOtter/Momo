# Output path descriptors (runtime precompute; declarative names).
import os

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DEST = os.path.join(OUT_DIR, "processed.zarr")
