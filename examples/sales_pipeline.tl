# ==================================================
# TensorLang v0.2 — Lazy sales pipeline (scan → write)
# ==================================================
# 1. Generate data once:  python examples/make_demo_data.py
# 2. Run the pipeline:    tensorlang run examples/sales_pipeline.tl
#
# The whole pipeline lowers into ONE Polars lazy query (predicate and
# projection pushdown included) and streams to Parquet without
# materializing intermediate frames in Python.

import os

from tensorlang import T

here = os.path.dirname(os.path.abspath(__file__))

# 1. A lazy scan is a declarative descriptor: nothing is read here.
sales := T.scan_parquet(os.path.join(here, "data", "sales-demo.parquet"))

# 2. Reusable scoped predicate and transformation operators.
valid := (
    ({D}.amount > 100) &
    ({D}.status == "completed")
)

net := {D}.amount - {D}.fee

# 3-4. Filter records, then derive a numerical field (still lazy).
filtered = sales[valid(sales)]

filtered["net"] = net(filtered)

# 5. Grouped aggregation (still lazy: one optimized backend query).
summary = filtered.net.sum(by=filtered.region)

# 6. Materialize or write the result (explicit execution boundaries).
print(summary.collect().sort("region"))

summary.write_parquet(os.path.join(here, "data", "revenue-demo.parquet"))

# 7. Inspect the optimized execution plan and measured costs.
print(summary.explain())
print(summary.profile())
