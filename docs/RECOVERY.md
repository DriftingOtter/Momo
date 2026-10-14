# TensorLang v0.6 persistent-storage and recovery guide

## Guarantee table

| Operation | Guarantee |
|---|---|
| New local destination | Staged publication through a final same-filesystem rename (atomic on POSIX) |
| Overwrite existing dataset | Recoverable staged replacement, **not** continuously atomic visibility: between the backup rename and the publish rename the destination is briefly absent |
| SIGINT during execution | Cooperative cancellation; staging cleaned after workers stop (`TL1032`) |
| SIGKILL during publication | Cleanup cannot run; recovery may be necessary (see table) |
| Remote store | Direct, explicitly non-transactional writes |
| External source consistency | Best-effort metadata change detection (`TL1030`, report-once) |
| Durability after power loss | **Not claimed**: rename atomicity is a namespace guarantee, not an fsync/durability guarantee |

## Recovery state table (`tensorlang.runtime.chunk_plan.recover(dest)`)

| # | Filesystem state | Safest action (what `recover` does) |
|---|---|---|
| 1 | No staging, no backup | Report clean; no action |
| 2 | Incomplete staging | Remove staging; destination untouched |
| 3 | Validated staging, destination absent | Retain + report; publish only with `publish_validated=True` |
| 4 | Backup present, destination absent | Restore newest backup → destination (regain previous valid data) |
| 5 | Backup + valid destination | Drop superseded backup |
| 6 | Publish done, backup removal failed | Same as 5 |
| 7 | Ambiguous (multiples, unreadable, invalid dest) | Per-item safe actions only; validated copies are never deleted |

Invariants: recovery never automatically deletes the only verified
valid dataset; never publishes over an existing destination without
`overwrite=True`; every run is idempotent (a second `recover` reports
the settled state).

## Helpers (`tensorlang.runtime.chunk_plan`, unless noted)

- `list_stale_staging(dest)` — identify leftovers (no reads).
- `recover_stale_staging(dest, remove=True)` — explicit removal.
- `recover(dest, publish_validated=False, drop_validated=False,
  overwrite=False)` — full state-table recovery.
- `Lazy.describe_write(path, ...)` — read-only write plan.

## Failure injection (tests only)

`tensorlang.runtime.chunk_plan.injected_failures(...)` arms
deterministic write-path faults inside a `with` block. It is not
exported at package top level and no public API accepts hook
arguments; production execution never sets these flags.
