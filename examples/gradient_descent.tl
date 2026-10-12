# ==================================================
# TensorLang v0.4 — Numerical optimization with explicit gradients
# ==================================================
# Gradient descent on a quadratic bowl using ordinary TensorLang
# runtime assignment: no optimizer framework, just grad() applied
# in a loop.

from tensorlang import T, grad

# A quadratic loss with a known minimum at [3.0, -2.0].
loss := (({W} - {T})**2).mean()

d_loss := grad(loss, wrt="W")

# Fixed target and starting point.
target = T([3.0, -2.0])
w = T([0.0, 0.0])

learning_rate = 0.5

for step in range(20):
    g = d_loss(w, target)
    w = w - learning_rate * g
    if step % 5 == 0:
        print(step, w.collect(), loss(w, target).collect())

print("final:", w.collect())
# final: [3. -2.] (up to numerical tolerance)
