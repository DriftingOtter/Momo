# v0.9 M0 — compatibility outlook

No v1.0 contract is established. Classification per M0 evidence:

- **A candidate stable**: `T` construction, operator def/apply/`>>`,
  `collect`, scans, joins/groups/counts, `to_dense` policies,
  `standardize`+`fit`, `grad`/`value_and_grad`, `run_file`/
  `run_source`, CLI run/check/diagnose, `TL03xx` declaration codes,
  UID parameter scoping, per-session module caching.
- **B needs clarification**: unfitted-apply timing/code (`TL0499`
  at collect); aggregated column naming; `check_source` internal
  params; grouped-sum float-order noise; `extra=` `=`-override
  documentation; REPL session retention.
- **C needs correction**: F-B4 mask alignment; F-A2 dual-spelling.
- **D experimental/internal**: `trusted/injected_names` params,
  chunked-AD/diff deferrals, REPL history internals.

Proposed policy direction (not enacted): additive changes freely;
behavioral corrections with coded diagnostics + migration notes;
diagnostic codes stable once documented; minimum versions pinned per
release; Python support follows CI matrix (3.12–3.14).
