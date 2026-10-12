# ==================================================
# TensorLang v0.4 — Automatic differentiation
# ==================================================
# Reverse-mode gradients over ordinary operator graphs: define a
# scalar-valued loss, build a first-class gradient operator, apply it
# to dense tensors, and verify against analytic derivatives.

from tensorlang import T, grad, value_and_grad

# ----------------------------------
# 1. A scalar-valued loss operator
# ----------------------------------
loss := (({P} - {Y})**2).mean()

# ----------------------------------
# 2. A first-class gradient operator
# ----------------------------------
d_loss := grad(loss, wrt="P")

# ----------------------------------
# 3-4. Data and explicit application
# ----------------------------------
pred := T([1.0, 2.0, 4.0])
target := T([0.0, 1.0, 1.0])

gradient = d_loss(pred, target)

print(gradient.collect())
# [0.666667, 0.666667, 2.0]

# ----------------------------------
# 5. Value and gradient together
# ----------------------------------
both := value_and_grad(loss, wrt="P")

value, gradient2 = both(pred, target)

print(value.collect())
# 3.6666666666666665

print(gradient2.collect())
# [0.666667, 0.666667, 2.0]

# ----------------------------------
# 6. Analytic verification
# ----------------------------------
# d/dP mean((P - Y)^2) = 2(P - Y)/n
n = 3.0
expected = (pred - target) * (2.0 / n)

print(expected.collect())
# [0.666667, 0.666667, 2.0]

# ----------------------------------
# 7. Nondifferentiable operations err
# ----------------------------------
from tensorlang import sort

step := sort({S}).sum()

try:
    grad(step)(T([3.0, 1.0, 2.0])).collect()
except Exception as e:
    print("expected error:", e)

# ----------------------------------
# 8. Source-level graph introspection
# ----------------------------------
print(d_loss.explain())
print(both(pred, target)[1].profile())
