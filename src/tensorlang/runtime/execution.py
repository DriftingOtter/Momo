"""Internal execution lifecycle for chunked runs.

Stages: planned -> admitted -> running -> (committing ->) completed, or
failed / cancelled at any point. Records are per-run (a ``ContextVar``
carries the in-flight execution, so concurrent runs never share state)
and summarized into ``last_record()`` for :func:`profile` introspection.
Ordinary computations never touch this API directly.
"""
from __future__ import annotations

import contextvars
import time
import uuid
from dataclasses import dataclass, field

STAGES = ("planned", "admitted", "running", "committing",
          "completed", "failed", "cancelled")

TERMINAL = ("completed", "failed", "cancelled")


@dataclass
class ExecutionRecord:
    run_id: str
    kind: str  # "collect" | "write"
    stage: str = "planned"
    stages: dict = field(default_factory=dict)  # stage -> perf_counter
    decision: dict = field(default_factory=dict)
    sources: list = field(default_factory=list)
    destination: str | None = None
    strategy: str | None = None  # "staged" | "direct" (writes)
    timings_ms: dict = field(default_factory=dict)
    error: dict = field(default_factory=dict)
    _token: object = field(default=None, repr=False, compare=False)

    def transition(self, stage: str) -> None:
        assert stage in STAGES, stage
        self.stage = stage
        self.stages[stage] = time.perf_counter()

    def split_ms(self, *names: str) -> None:
        """Record named durations between consecutive stage timestamps."""
        order = [s for s in STAGES if s in self.stages]
        stamps = [self.stages[s] for s in order]
        for name, (before, after) in zip(names, zip(stamps, stamps[1:])):
            self.timings_ms[name] = (after - before) * 1000.0


_current: contextvars.ContextVar = contextvars.ContextVar(
    "tensorlang_execution", default=None)

_LAST: dict = {}


def begin(kind: str, **info) -> ExecutionRecord:
    """Start a run; sets the in-flight record for this context."""
    record = ExecutionRecord(run_id=uuid.uuid4().hex[:12], kind=kind,
                             **info)
    record.transition("planned")
    record._token = _current.set(record)
    return record


def current() -> ExecutionRecord | None:
    return _current.get()


def finish(record: ExecutionRecord, stage: str,
           error: dict | None = None) -> ExecutionRecord:
    assert stage in TERMINAL, stage
    record.transition(stage)
    if error:
        record.error = dict(error)
    if record._token is not None:
        _current.reset(record._token)
        record._token = None
    global _LAST
    _LAST = {"run_id": record.run_id, "kind": record.kind,
             "stage": record.stage, "stages": dict(record.stages),
             "decision": dict(record.decision),
             "destination": record.destination, "strategy": record.strategy,
             "timings_ms": dict(record.timings_ms),
             "sources": list(record.sources), "error": dict(record.error)}
    return record


def last_record() -> dict:
    """Summary of the most recent finished chunked run (for tooling)."""
    return dict(_LAST)
