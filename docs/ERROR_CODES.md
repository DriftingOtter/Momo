# TensorLang error codes

Stable, machine-readable codes attached to every `TensorLangError`
(`ClassName[CODE]: loc: message`). Tools and programmers can rely on
codes across message rewording. Ranges: `TL01` syntax, `TL02` scope,
`TL03` declaration/effects, `TL04` types, `TL05` shapes, `TL06`
mutation, `TL07` symbolic, `TL08` autodiff, `TL09` unsupported;
`x99` in each range is the generic class fallback.

| code | category | title | guidance |
|---|---|---|---|
| `TL0101` | TLSyntaxError | invalid syntax | Check the reported line for TensorLang/Python syntax mistakes. |
| `TL0102` | TLSyntaxError | malformed declaration | Do not call __tl_declare__ directly; write NAME := expr. |
| `TL0201` | TLScopeError | duplicate declaration | A name may be declared once per scope; use = to reassign. |
| `TL0202` | TLScopeError | declaration needs a plain name | := must introduce a plain name in the current scope. |
| `TL0301` | TLDeclarationError | runtime application under := | Operator application executes; bind its result with = instead. |
| `TL0302` | TLDeclarationError | effectful expression under := | Reads, writes, and inplace() are effectful; use = (or T.scan_* for a lazy descriptor). |
| `TL0303` | TLDeclarationError | runtime expression under := | collect()/fit()/conversions execute; bind their result with =. |
| `TL0304` | TLDeclarationError | effectful tensor under := | Tensors from T.read() carry I/O provenance; use = to bind them. |
| `TL0401` | TLTypeError | ambiguous truth value | Tensors have no truth value; collect() an explicit Boolean or reduce first. |
| `TL0402` | TLTypeError | cannot iterate a tensor | Iteration is never implicit; collect() or index explicitly first. |
| `TL0403` | TLTypeError | fit needs tensor data | Pass a Tensor or lazy result to .fit(), not scalars or operators. |
| `TL0404` | TLTypeError | cannot read or describe source | Check the path, format, and scan options. |
| `TL0405` | TLTypeError | cannot write output | Write structured frames/series to .parquet/.csv; select fields first for dense data. |
| `TL0410` | TLTypeError | non-numeric estimator data | Select or encode numeric feature columns explicitly. |
| `TL0411` | TLTypeError | invalid estimator configuration | Check constructor arguments (ranges, norms, strategies, counts). |
| `TL0412` | TLTypeError | missing or infinite fitting data | Handle nulls/infinities explicitly first, e.g. with impute(). |
| `TL0413` | TLTypeError | cannot impute all-missing column | Drop the column or supply values; no statistic exists. |
| `TL0414` | TLTypeError | unknown estimator or option | Use a documented estimator kind, strategy, or norm name. |
| `TL0415` | TLTypeError | missing features on apply | Application data must contain every fitted feature column. |
| `TL0416` | TLTypeError | operator is not fittable | Only standardize()/minmax()/normalize()/impute()/pca() (and >> pipelines) support .fit(). |
| `TL0417` | TLTypeError | invalid operator composition | Both sides of >> must be operators; the right side must be unary. |
| `TL0420` | TLTypeError | gradient constructor needs an operator | Pass an Operator to grad()/value_and_grad()/vjp(), not data. |
| `TL0430` | TLTypeError | invalid join configuration | Use a documented join type; non-cross joins need explicit on= keys. |
| `TL0431` | TLTypeError | join key or shape problem | Keys must exist on both sides with compatible lengths. |
| `TL0501` | TLShapeError | incompatible broadcast shapes | Align shapes explicitly; broadcasting follows NumPy rules. |
| `TL0502` | TLShapeError | incompatible matrix multiplication | matmul needs (…, n, k) @ (…, k, m) numerical operands. |
| `TL0503` | TLShapeError | operator arity mismatch | Apply with exactly one argument per parameter, positionally. |
| `TL0504` | TLShapeError | mask length mismatch | Boolean masks need one entry per element/row. |
| `TL0505` | TLShapeError | assignment shape mismatch | Assigned values must match the target's rows/positions. |
| `TL0506` | TLShapeError | invalid linear system | solve() needs a square non-singular matrix with matching right-hand side. |
| `TL0507` | TLShapeError | invalid decomposition input | svd() needs rank-2 matrices; eigh() needs square matrices. |
| `TL0508` | TLShapeError | invalid PCA request | Need n_components <= min(n_samples, n_features) and >= 2 samples. |
| `TL0509` | TLShapeError | feature width mismatch on apply | Application data must have the fitted feature count (frames align by name). |
| `TL0510` | TLShapeError | join execution failed | Check key dtypes and join compatibility. |
| `TL0601` | TLMutationError | inplace without a mutable target | Use f(tensor).inplace() with a single-input operator over a Tensor. |
| `TL0602` | TLMutationError | inplace into a lazy scan | Scans are immutable descriptors; collect() into a tensor first. |
| `TL0603` | TLMutationError | inplace result incompatible | The computed result cannot replace the target's contents. |
| `TL0604` | TLMutationError | inplace execution failed | The input was left unchanged; inspect the cause. |
| `TL0701` | TLSymbolicError | non-symbolic expression | Differentiate/integrate pure mathematical operators only. |
| `TL0702` | TLSymbolicError | constant without exact form | The constant has no exact symbolic representation. |
| `TL0703` | TLSymbolicError | symbolic arity mismatch | diff()/integrate() need single-parameter operators. |
| `TL0704` | TLSymbolicError | no exact closed form | No exact integral exists; numerical quadrature is never silent. |
| `TL0801` | TLDiffError | invalid differentiation target | Specify wrt= with an existing parameter name (default: the sole parameter). |
| `TL0802` | TLDiffError | second-order differentiation | Higher-order derivatives are deferred; grad() of a gradient operator is rejected. |
| `TL0803` | TLDiffError | non-scalar gradient target | grad() needs scalar-valued operators; use vjp() with a cotangent otherwise. |
| `TL0804` | TLDiffError | non-floating gradient input | Differentiable inputs must be floating-point tensors; pass float data. |
| `TL0805` | TLDiffError | unfitted estimator in graph | Differentiation through fitting is deferred; call .fit() first (only fitted application differentiates). |
| `TL0806` | TLDiffError | nondifferentiable operation | See LANGUAGE_SPEC.md for the supported rule set; restructure the graph. |
| `TL0807` | TLDiffError | gradient shape mismatch | Cotangent shapes must be consistent through the graph. |
| `TL0808` | TLDiffError | vjp cotangent mismatch | The seed must match the output shape and kind exactly. |
| `TL0901` | TLUnsupportedError | undefined for representation | The operation has no meaning for this representation; select/convert explicitly. |
