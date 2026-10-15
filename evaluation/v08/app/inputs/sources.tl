# Lazy source descriptors, located relative to this file so the
# application runs from any working directory.
# (Path joins are runtime calls under M1 rules, so they are
# precomputed with '=' before the declarative scan bindings.)
import os

from tensorlang import T

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OBS_PATH = os.path.join(DATA_DIR, "observations.parquet")
PARAMS_PATH = os.path.join(DATA_DIR, "params.parquet")

obs := T.scan_parquet(OBS_PATH)
params := T.scan_parquet(PARAMS_PATH)
