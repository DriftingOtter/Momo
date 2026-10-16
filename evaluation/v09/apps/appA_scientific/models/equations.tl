# Exponential-decay measurement model (pure operator definitions).
#
# NOTE (v0.9 M0 finding F-A1): operators cannot be applied to symbolic
# arguments (`predict({A}, ...)` is TL0499, same-file or cross-file), so
# composites are written flat or composed with `>>`, not nested.
from tensorlang import exp

predict := {A} * exp(-{K} * {T}) + {C}
residual := ({A} * exp(-{K} * {T}) + {C}) - {Y}
