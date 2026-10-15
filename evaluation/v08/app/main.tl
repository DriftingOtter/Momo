# Single-file version of the experiment pipeline (baseline for the
# multi-file comparison).
import os

from tensorlang import T, not_null

here = os.path.dirname(os.path.abspath(__file__))
obs_path = os.path.join(here, "data", "observations.parquet")

obs := T.scan_parquet(obs_path)

valid = obs[not_null(obs.meas)]
print("valid rows:", valid.collect().height)
print(valid.meas.mean(by=valid.experiment_id).collect().height)
