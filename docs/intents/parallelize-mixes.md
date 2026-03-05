---
title: "Parallelize Mix Creation"
author: "human:aaron"
version: 1
created: 2026-03-04
---

# Parallelize Mix Creation

## WANT

Enable the backend to process multiple remix jobs in parallel instead of single-flight locking.

Target behavior:
- Allow up to `max_concurrent_mixes` concurrent jobs across both `/api/remix` and `/api/remix/youtube` (default `1`, rollout target `2`).
- Keep current fail-fast API behavior when capacity is full (return `409`).
- Preserve existing session/SSE flow per mix job.
- Keep rollout safe by defaulting concurrency to `1` unless explicitly configured higher.

Implementation direction:
- Add `max_concurrent_mixes` setting with explicit bounds `1..8`.
- Replace global single lock with one app-wide bounded capacity gate (`BoundedSemaphore`) shared by both create endpoints.
- Scale executor worker count from config instead of hardcoded `1`.
- Use explicit permit ownership handoff: request handler acquires; background wrapper releases after successful submit; handler releases only on pre-submit failures.
- Harden thread safety around module-level mutable state touched across pipelines.
- Shared mutable state scope for Phase 1 is explicit: `taste_stage._consecutive_fallbacks`, `taste_stage._circuit_open_since`, and `processor._rubberband_version`.
- Each shared mutable state item must be classified as guarded-by-lock, immutable-after-init, or request-scoped before rollout.
- Startup guard contract is explicit: treat `UVICORN_WORKERS` or `WEB_CONCURRENCY` values `>1` as multi-worker mode; this requires `distributed_limiter_enabled=true`.

## DON'T

- Don't switch to an unbounded queue in this phase.
- Don't remove fail-fast behavior yet (no silent backlog growth).
- Don't assume process-local semaphore limits apply across multiple server workers/processes.
- Don't redesign the full pipeline architecture or storage model.
- Don't weaken existing file validation, SSRF checks, or error handling.
- Don't ship with default concurrency above `1` until validation passes.

## LIKE

- Current per-session directory isolation (`uploads/stems/remixes/{session_id}`) is good and should stay.
- Current immediate `session_id` response + SSE progress model is good and should stay.
- Current endpoint contract (`/api/remix`, `/api/remix/youtube`) is good and should stay.

## FOR

- Backend FastAPI service in `backend/src/musicmixer/`.
- Primary files: `main.py`, `api/remix.py`, and concurrency-sensitive services (`taste_stage.py`, `processor.py`).
- Use case: two separate clients can create mixes concurrently without one being blocked by single-flight lock.
- Runtime environment: local dev first, then controlled rollout.
- Phase 1 deployment model: single backend process/worker; process-local semaphore is authoritative only within that process.
- Phase 2 deployment model (if multi-worker): replace/augment process-local gate with a distributed capacity limiter (e.g., Redis-backed permit counter).

## ENSURE

- With `max_concurrent_mixes=2`, first two create requests are accepted; a third concurrent request gets `409`.
- With mixed traffic (`/api/remix` + `/api/remix/youtube`), capacity is still globally enforced by the same gate.
- Capacity semantics are explicit: in Phase 1 this is global within a single backend process; multi-worker rollout requires distributed admission control.
- Once one running job finishes, a new request is accepted without restart.
- Existing single-concurrency behavior remains intact when `max_concurrent_mixes=1`.
- `max_concurrent_mixes` bounds are explicit and enforced as `1..8`.
- No permit leak across explicit failure matrix: upload failure, validation failure, executor submit failure, and wrapper exception.
- `max_concurrent_mixes` config is validated at startup (integer, min `1`, bounded max) and effective runtime value is logged.
- All backend tests pass after updating concurrency expectations.
- New tests exist for bounded parallel behavior on both `/api/remix` and `/api/remix/youtube`.
- Executor sizing is documented and aligned with configured capacity so accepted jobs can actually run in parallel.
- Thread-safety tests explicitly cover concurrent updates of `taste_stage` circuit-breaker counters and concurrent access to `processor` rubberband version cache.
- Phase 1 startup guard is concrete: if `UVICORN_WORKERS>1` or `WEB_CONCURRENCY>1` and `distributed_limiter_enabled=false`, startup fails fast with a clear configuration error.

## TRUST

- [autonomous] Implement config-driven concurrency and replace single lock with bounded slots.
- [autonomous] Refactor acquire/release paths to guarantee exactly-once release semantics.
- [autonomous] Add thread-safety guards for mutable module-level state where needed.
- [autonomous] Update and add backend tests for bounded parallelism.
- [autonomous] Run backend test suite and iterate on failures.
- [autonomous] Implement the explicit startup guard contract for multi-worker detection (`UVICORN_WORKERS`/`WEB_CONCURRENCY`) and distributed limiter flag enforcement.
- [ask] Decide if Phase 2 should move from fail-fast `409` to explicit queued mode with position/status reporting.
