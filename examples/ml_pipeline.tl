# ==================================================
# TensorLang v0.3 — Fittable ML pipeline (learn, apply, mutate)
# ==================================================
# A single reusable operator pipeline learns a transformation from a
# heterogeneous structured dataset, applies it to separate data, and
# produces correctly shaped numerical tensors — learn imputation means,
# standardization statistics, and PCA components on TRAIN only.

from tensorlang import T, impute, pca, standardize, svd

# ----------------------------------
# Training and test data
# ----------------------------------
train := T({
    "age": [25, 45, 35, 55],
    "income": [45000, 105000, 75000, 95000],
    "score": [6.0, None, 8.0, 9.0],
})

test := T({
    "age": [25, 45],
    "income": [45000, 105000],
    "score": [6.0, 9.0],
})

# ----------------------------------
# Operator definitions
# ----------------------------------
preprocess := (
    impute("mean")
    >> standardize()
    >> pca(2)
)

# ----------------------------------
# Runtime fitting
# ----------------------------------
fitted = preprocess.fit(train)

# Explicit operator application
train_result = fitted(train)
test_result = fitted(test)

# Expected shapes
# train_result: (4, 2)
# test_result:  (2, 2)
print(train_result.collect())
print(test_result.collect())

# ----------------------------------
# Reuse and mutation
# ----------------------------------
copy_result = fitted(test)

# Mutate test explicitly to hold
# the reduced numerical representation
fitted(test).inplace()

print(test.shape)
# (2, 2)

# ----------------------------------
# Numerical linear algebra
# ----------------------------------
A := T([
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0],
])

U, S, Vh = svd(A)

print(S.collect())

# ----------------------------------
# Execution introspection
# ----------------------------------
print(train_result.explain())
