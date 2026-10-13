"""Generate the demo dataset for examples/chunked_demo.tl.

Usage: python examples/make_chunked_data.py [--rows N] [--cols M] [--out PATH]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--chunk-rows", type=int, default=10_000)
    parser.add_argument("--out", type=str,
                        default=str(Path(__file__).resolve().parent
                                    / "data" / "large-demo.zarr"))
    parser.add_argument("--out-b", type=str,
                        default=str(Path(__file__).resolve().parent
                                    / "data" / "other-demo.zarr"))
    parser.add_argument("--chunk-rows-b", type=int, default=20_000)
    args = parser.parse_args()
    import zarr

    def _write(path, chunk_rows, seed):
        rng = np.random.default_rng(seed)
        out = Path(path)
        arr = zarr.open_array(str(out), mode="w",
                              shape=(args.rows, args.cols),
                              chunks=(chunk_rows, args.cols),
                              dtype="float64")
        for start in range(0, args.rows, chunk_rows):
            stop = min(start + chunk_rows, args.rows)
            arr[start:stop] = rng.random((stop - start, args.cols)) + 0.5
        print(f"wrote {out} ({args.rows}x{args.cols}, "
              f"chunks {(chunk_rows, args.cols)})")

    _write(args.out, args.chunk_rows, 20260501)
    _write(args.out_b, args.chunk_rows_b, 20260502)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
