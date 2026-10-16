"""Independent Polars/sklearn reference for the spend model."""
import os

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

here = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(here, "..", "data")
txn = pl.read_parquet(f"{DATA}/transactions.parquet")
cust = pl.read_parquet(f"{DATA}/customers.parquet")
valid = txn.filter(pl.col("amount").is_not_null())
filt = valid.filter((pl.col("amount") > 200) | (pl.col("category") == "grocery"))
agg = (filt.group_by("cust_id")
       .agg(pl.col("amount").sum().alias("total"),
            pl.len().alias("count")))
feat = agg.join(cust, on="cust_id").filter(pl.col("credit_limit").is_not_null())
print("complete rows:", feat.height)
train = feat.filter(pl.col("cust_id") < 1500)
valid = feat.filter(pl.col("cust_id") >= 1500)
cols = ["tenure", "credit_limit", "count"]
Xtr, ytr = train.select(cols).to_numpy(), train["total"].to_numpy()
Xva, yva = valid.select(cols).to_numpy(), valid["total"].to_numpy()
scaler = StandardScaler()
Ztr, Zva = scaler.fit_transform(Xtr), scaler.transform(Xva)
model = Ridge(alpha=1.0, fit_intercept=True, solver="cholesky")
model.fit(Ztr, ytr)
pred = model.predict(Zva)
print("valid mse:", float(((yva - pred) ** 2).mean()))
print("weights:", [float(model.intercept_)] + [float(c) for c in model.coef_])
