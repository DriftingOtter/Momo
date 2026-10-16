"""Independent SciPy reference: curve_fit on the same data."""
import os
import numpy as np
from scipy.optimize import curve_fit

here = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(here, "..", "data", "obs.npz"))
t, y = z["t"], z["y"]
model = lambda t, A, K, C: A * np.exp(-K * t) + C
popt, _ = curve_fit(model, t, y, p0=[1.0, 0.3, 0.0], maxfev=20000)
print("params:", popt.tolist())
resid = y - model(t, *popt)
print("mse:", float((resid ** 2).mean()))
