# Native multi-file experiment pipeline (M2: no Python orchestration).
#
# Composition root: fans in operators/, inputs/, and pipelines/ with
# ordinary TensorLang imports. All orchestration is `.tl`; Python
# remains only for data generation and the independent reference.
import os

import polars as pl

from tensorlang import T, not_null, solve, standardize

from inputs.sources import obs, params
from operators.calibration import calibrate
from operators.transforms import zscore
from pipelines.analysis import method

here = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(here, "out", "coefficients_tl.parquet")

base = obs[not_null(obs.meas)]
joined = base.join(params, on="experiment_id")

# Row-level calibration with the cross-file reusable operator,
# derived inside the lazy graph (join order is backend-defined).
joined["cal"] = calibrate(joined.meas, joined.gain, joined.offset)
enriched = joined.collect()
print("rows:", enriched.height)

agg_base = T(enriched)
means = agg_base.cal.mean(by=agg_base.experiment_id).collect()
counts = agg_base.count(by="experiment_id").collect()
print("experiments:", means.height)

dense = agg_base.to_dense(columns=["cal", "temp"]).collect()

fitted = standardize().fit(T(dense))
Z = fitted(T(dense)).collect()

# Model cal = a + b*temp on standardized features (ridge, lambda=0.1).
y = Z[:, 0]
X = Z[:, 1]
n = y.shape[0]
XtX = T([[n, float(X.sum())], [float(X.sum()), float((X ** 2).sum())]])
Xty = T([float(y.sum()), float((X * y).sum())])
w = solve(XtX + 0.1 * T([[1.0, 0.0], [0.0, 1.0]]), Xty).collect()
print("temp coefficient (standardized):", w[1])

# Reusable cross-module operator applied at runtime (re-exported
# through pipelines/analysis.tl).
tres = method(T([20.0, 21.0]), T(20.5), T(0.5)).collect()
print("rescaled temp:", tres.tolist())

out = pl.DataFrame({"term": ["intercept", "temp_coef"], "w": w.tolist()})
out.write_parquet(out_path)

print(base.meas.mean(by=base.experiment_id).explain())
