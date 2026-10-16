# Feature derivations + re-exports (sibling import).
#
# NOTE (v0.9 M0 finding F-A2): every file is imported under exactly one
# spelling app-wide (bare here via this sibling import; dotted nowhere
# else), because v0.8.0 cannot serve one file under two spellings.
from aggregations import grocery
from aggregations import high_value
from aggregations import log_amount

# Runtime re-exports for the application root.
is_high = high_value
is_grocery = grocery
log_amt = log_amount

# NOTE: grouped sum/count columns keep backend names.
# NOTE (v0.9 M0 finding F-B1): `.count` hits a method, not the
# field; explicit string indexing is required for method-named fields.
big_ticket := ({D}.amount > 1000) & ({D}["count"] > 5)
