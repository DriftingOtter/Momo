# Mean-squared-error over model residuals, stated flat.
#
# NOTE (v0.9 M0 finding F-A1): operators cannot be applied to symbolic
# arguments (TL0499), so `mse` cannot nest the `predict` operator
# textually. NOTE (finding F-A2): this module deliberately does NOT
# import `equations` — importing the same file under two spellings
# (bare here, dotted at the root) fails on v0.8.0.
from tensorlang import exp

mse := (({A} * exp(-{K} * {T}) + {C} - {Y}) ** 2).mean()
