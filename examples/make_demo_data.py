"""Generate the demo dataset for examples/sales_pipeline.tl.

Usage: python examples/make_demo_data.py [--rows N] [--out PATH]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))

from datasets import make_sales


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument("--out", type=str,
                        default=str(Path(__file__).resolve().parent
                                    / "data" / "sales-demo.parquet"))
    args = parser.parse_args()
    make_sales(Path(args.out), args.rows, seed=20260201)
    print(f"wrote {args.out} ({args.rows} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
