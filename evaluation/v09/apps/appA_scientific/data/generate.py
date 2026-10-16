"""Synthetic exponential-decay observations (seeded, no network)."""
import numpy as np

def main(outdir, n=400, seed=11):
    import os
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 5, n))
    y = 2.5 * np.exp(-0.8 * t) + 0.3 + rng.normal(0, 0.08, n)
    np.savez(os.path.join(outdir, "obs.npz"), t=t, y=y,
             truth=np.array([2.5, 0.8, 0.3]))
    return {"n": n, "truth": [2.5, 0.8, 0.3]}
if __name__ == "__main__":
    import sys
    print(main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/v09a-data"))
