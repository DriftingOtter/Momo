# Lazy source descriptors (paths computed at runtime, scans declarative).
import os

from tensorlang import T, not_null

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CUST_PATH = os.path.join(DATA_DIR, "customers.parquet")
TXN_PATH = os.path.join(DATA_DIR, "transactions.parquet")

customers := T.scan_parquet(CUST_PATH)
transactions := T.scan_parquet(TXN_PATH)

valid_txn = transactions[not_null(transactions.amount)]
