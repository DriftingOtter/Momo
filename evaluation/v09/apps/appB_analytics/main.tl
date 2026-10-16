# Customer-spend prediction: scans -> features -> dense -> fit -> ridge.
import os

import polars as pl

from tensorlang import T, not_null, solve, standardize

from features.transforms import big_ticket, is_grocery, is_high, log_amt
from models.prediction import score
from models.preprocessing import prep
from sources.transactions import customers, transactions, valid_txn

here = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(here, "out_metrics.parquet")

# Per-record reusable operators applied inside the lazy graph.
enriched = valid_txn[is_high(valid_txn) | is_grocery(valid_txn)]
enriched["log_amount"] = log_amt(valid_txn)
print("rows after filter:", enriched.collect().height)

# Per-customer aggregates.
agg = enriched.amount.sum(by=enriched.cust_id)
n = enriched.count(by="cust_id")
lg = enriched.log_amount.mean(by=enriched.cust_id)
feat = agg.join(n, on="cust_id").join(lg, on="cust_id")
feat = feat.join(customers, on="cust_id")
print("customers:", feat.collect().height)

# Explicit missing-value handling: model only complete rows.
clean = feat[not_null(feat.credit_limit)]
print("complete rows:", clean.collect().height)

# Flag big-ticket customers. NOTE (v0.9 M0 finding F-B4): `&`-combined
# masks evaluate sides in backend-defined order and combine
# positionally, so sequential single-predicate filters are used here;
# each is lowered atomically and the count is exact.
# M1: same-domain `&` fuses into one query (exact, deterministic).
print("big ticket:", clean[big_ticket(clean)].collect().height)

# Deterministic train/validation split by customer id.
train = clean[clean.cust_id < 1500]
valid = clean[clean.cust_id >= 1500]
# NOTE (v0.9 M0 finding F-B3): separate collects of one lazy frame are
# NOT row-aligned (order is backend-defined). Materialize features and
# target together, once per split, then divide in NumPy.
alltr = train.to_dense(columns=["tenure", "credit_limit", "count", "amount"], nulls="error").collect()
allva = valid.to_dense(columns=["tenure", "credit_limit", "count", "amount"], nulls="error").collect()
xtr = alltr[:, :3]
ytr = alltr[:, 3:4]
xva = allva[:, :3]
yva = allva[:, 3:4]

fitted = prep.fit(T(xtr))
Ztr = fitted(T(xtr)).collect()
Zva = fitted(T(xva)).collect()

# Ridge via normal equations with intercept.
import numpy as _np

Xtr = _np.column_stack([_np.ones(Ztr.shape[0]), _np.asarray(Ztr)])
# NOTE: intercept excluded from the ridge penalty (sklearn convention).
pen = _np.diag([0.0, 1.0, 1.0, 1.0])
w = solve(T(Xtr.T @ Xtr + pen), T(Xtr.T @ _np.asarray(ytr).ravel())).collect()
w = _np.asarray(w)
pred = _np.asarray(score(T(_np.asarray(Zva)), T(w[1:].reshape(-1, 1)), T(w[:1])).collect()).ravel()
mse = float(((_np.asarray(yva).ravel() - pred) ** 2).mean())
print("valid mse:", mse)
print("weights:", w.tolist())

out = pl.DataFrame({"metric": ["mse", "w0", "w1", "w2", "w3"],
                    "value": [mse] + w.tolist()})
out.write_parquet(out_path)
