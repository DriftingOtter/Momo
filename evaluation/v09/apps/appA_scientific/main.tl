# Exponential-decay fit: data + loss fan-in at the application root.
import os

import numpy as np

from tensorlang import T, value_and_grad

from analysis.estimation import gd_step
from models.equations import predict
from models.loss import mse

here = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(here, "data", "obs.npz"))
t = z["t"]
y = z["y"]

vg = value_and_grad(mse, wrt=["A", "K", "C"])
fa = 1.0
fk = 0.3
fc = 0.0
lr = 0.2
for i in range(1500):
    v, g = vg(T(fa), T(fk), T(t), T(fc), T(y))
    fa = float(gd_step(T(fa), T(lr), g[0]).collect())
    fk = float(gd_step(T(fk), T(lr), g[1]).collect())
    fc = float(gd_step(T(fc), T(lr), g[2]).collect())
print("loss:", float(v.collect()))
print("params:", [fa, fk, fc])
pred = predict(T(fa), T(fk), T(t), T(fc)).collect()
print("pred mse:", float((((np.asarray(pred) - y) ** 2).mean())))
np.savez(os.path.join(here, "out_params.npz"), params=np.array([fa, fk, fc]))
