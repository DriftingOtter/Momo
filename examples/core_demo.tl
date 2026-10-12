# ==================================================
# TensorLang v0.1 — Core Functionality Demonstration
# ==================================================

from tensorlang import T, sin, exp, diff, integrate


# --------------------------------------------------
# 1. TENSOR DEFINITIONS
# --------------------------------------------------

scalar := T(5)
vector := T([1, 5, 12, 20, 30])

matrix := T([
    [1, 2, 3],
    [4, 5, 6]
])

# Ordinary numerical operations
doubled = vector * 2
transposed = matrix.T
matrix_product = matrix @ matrix.T

print(doubled.collect())
# [2, 10, 24, 40, 60]


# --------------------------------------------------
# 2. LOCALLY SCOPED OPERATOR DEFINITIONS
# --------------------------------------------------

square := {X}**2

shift := {X} + 5

mask := ({X} > 10) & ({X} < 25)

# Each operator has an independent local X
pipeline := square >> shift


# --------------------------------------------------
# 3. EXPLICIT OPERATOR APPLICATION
# --------------------------------------------------

selected = vector[mask(vector)]

result = pipeline(selected)

print(result.collect())
# [149, 405]

# The original vector is unchanged
print(vector.collect())
# [1, 5, 12, 20, 30]


# --------------------------------------------------
# 4. EXPLICIT IN-PLACE EXECUTION
# --------------------------------------------------

# Update the original tensor itself
pipeline(vector).inplace()

print(vector.collect())
# [6, 30, 149, 405, 905]

# In-place operations may change shape
filter_large := {X}[{X} > 100]

filter_large(vector).inplace()

print(vector.collect())
# [149, 405, 905]


# --------------------------------------------------
# 5. STRUCTURED TENSORS
# --------------------------------------------------

sales := T({
    "amount": [80, 180, 240, 110],
    "fee": [2, 10, 20, 5],
    "status": [
        "pending", "completed",
        "completed", "completed"
    ],
    "region": [
        "East", "East",
        "West", "West"
    ]
})

# A reusable predicate operating on fields
eligible := (
    ({D}.amount > 100) &
    ({D}.status == "completed")
)

# Filter rows
filtered = sales[eligible(sales)]

# Define an operator that calculates net revenue
net := {D}.amount - {D}.fee

# Assign a computed field
filtered["net"] = net(filtered)

# Grouped aggregation
revenue = filtered.net.sum(by=filtered.region)

print(revenue.collect())
# East: 170
# West: 325


# --------------------------------------------------
# 6. SYMBOLIC MATHEMATICS
# --------------------------------------------------

f := sin({X}) * exp(-{X}**2)

# Symbolic differentiation
df := diff(f)

# Generate input values at runtime
points = T.linspace(0, 5, 1000)

# Evaluate the derivative numerically
values = df(points)

print(values.collect())

# Exact symbolic integration
area := integrate({X}**2, 0, 5)

print(area.collect())
# 125/3


# --------------------------------------------------
# 7. PYTHON CONTROL FLOW
# --------------------------------------------------

datasets := [
    T([1, 20, 30]),
    T([5, 15, 40]),
    T([2, 80, 100])
]

for data in datasets:
    selected = data[mask(data)]
    print(selected.collect())
