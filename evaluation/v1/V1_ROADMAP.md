# V1 roadmap (evidence-ranked, post-M0)

M0 verdict: **NO-GO for v1.0 freeze; GO for stabilization.** One P0 + seven P1s; no silent-math, mutation, or architectural risks. All milestones below are fixes/freezes of demonstrated gaps — no new language features.

## M1 — Contract stabilization (required)

1. Fix P0: corrupt-manifest diagnosis (surface `TOMLDecodeError` via CLI; regression test through CLI path, not just `load_manifest`).
2. Fix P1s: `tensorlang run` nonexistent-file coded error; `tl add` path-spec decision (accept or explicitly reject with documented `uv pip` path); uv-error context (multi-line tail or `--verbose`); spec §9 `to_dense` scope sentence; spec §16 M4 atomic qualifier; `check_source` plumbing decision (deprecate or freeze-as-D); join-order non-promise in docs.
3. Freeze: narrow-A stable list (M0_PUBLIC_API §2); diagnostic code identities (wording not frozen); tolerance policy (exact vs allclose vs last-ulp); `fit`-returns-new; TL0419/TL0512 verdicts; `(path,name)` identity; `tl run/check` exit contract.
4. Adopt: semver/deprecation policy (M0_COMPATIBILITY §2); publish v0.8→v0.9 migration guide.
5. Wording: 8 language ambiguities (loop/`:=`, trust-flat, attr rule, literal-exempt, plain-values, open-set, aliasing, dunder limits); collect-aliasing; describe-cache; stale `M0_INTEROP` note refresh.
6. Lock-in tests: TL0512 identities, TL0419 timing, 1774-record appB, `(path,name)` dual-spelling, no-poisoning, TL0315, locked-repro, argv/exit passthrough.

## M2 — Distribution readiness (conditional, thin only)

Implement ONLY if M1 confirms the gap still blocks real libs: `tl package --check` (wheel contains `.tl` + fresh-venv import probe) + `tl add` local-path acceptance. No custom packager, no registry, no `tl publish` (stay `uv`/twine). If the `uv pip` workaround + docs prove sufficient in practice, skip M2 code entirely.

## M3 — Developer experience & docs (required, small)

Document 7 onboarding assumptions (index-vs-`--tensorlang`, pytest-dev step, `sys.argv[3:]`, no-pip/`uv pip --project`, name rules, locked-vs-stale, `tensorlang doctor` vs `tl doctor`); `diagnose` JSON schema; run the M0 study protocol with real unfamiliar developers (SIMULATED label until then).

## M4 — Release-candidate hardening (required, narrow)

Portability lint (hardcoded paths bit twice — standing CI check, per v0.9 retro); dependency-floor installs; `KeyboardInterrupt→130`/edge exits; release script (bump/changelog/build/install/CI/merge/tag checklist); Windows stays Tier-2 (no parity claims); security-boundary statement (effect contract ≠ sandbox).

## Release validation → v1.0.0

Full matrix (6 cells + packaging + project gate with 33 executed, 0 silently skipped), clean-install both CLIs, managed-app reproduction outside checkout, compat re-run (10/10 + 15/15), zero open P0/P1, docs freeze, annotated tag. Publication to an index is a separate distribution action.

## Explicitly deferred (not v1.0)

New grammar/compiler/ontology; GPU/distributed; chunked/second-order AD; general contractions; dedicated registry; custom resolver; interpreter replacement; import-system rewrite; global auto-cache; sandboxing; new operator families; `tl build` packager; `tl publish`.

## Gates

M1 done when: P0+7 P1 closed, narrow-A frozen, policy adopted, lock-in tests green. M2 done when: lib install→import→apply→test→reproduce works via documented `tl` path (or skipped with evidence). M3 done when: onboarding walk unaided + real-study results analyzed. M4 done when: portability lint + floor installs + release script green.
