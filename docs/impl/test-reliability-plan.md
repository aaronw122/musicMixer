# Test Reliability Plan: Fix Subagent Test Timeout/Stall Loop

**Status:** Implemented (Fixes 1-4 shipped, Fix 5 deferred)
**Revision:** 7
**Date:** 2026-02-28

## Problem Statement

Subagents deployed for implementation tasks in the backend repo frequently get stuck running tests, consuming 100k+ tokens and timing out. This is a recurring issue across multiple sessions and agents. The problem compounds from five independent root causes that interact to create a failure loop.

## Root Cause Analysis

### RC1: No pytest timeout — tests can hang forever
The backend has no `pytest-timeout` dependency and no timeout configuration anywhere (`pyproject.toml`, `pytest.ini`, `conftest.py`). Threading fixtures in `test_pipeline_sse.py` use `threading.Event().wait(timeout=30)` — if the event never signals, each test silently waits 30 seconds. The full suite has ~735 tests across 25 files; a handful of hanging tests can stall the suite for minutes.

### RC2: Sandbox blocks pytest on first attempt, every time
Claude Code's filesystem sandbox blocks write access to the `uv` cache directory. Every session that runs pytest follows the same 2-step pattern: (1) `uv run pytest` fails with `Operation not permitted`, (2) retry with `dangerouslyDisableSandbox: true` succeeds. This wastes 1-2 tool calls and ~30 seconds per test run, observed in at least 6 separate sessions.

### RC3: Skills hardcode `npm test` — backend uses `uv run pytest`
The `ship`, `fix-the-things`, `wrap-it-up`, `code-review`, `code-review-critical`, and `team-three-review` skills all hardcode `npm test` as the test command. The musicMixer backend is Python with `uv` and `pytest`. When a subagent lands in the backend, the skill instructions actively mislead it into running the wrong command.

### RC4: No fast vs full test distinction
No pytest markers exist (`@pytest.mark.slow`). Every test run is the full suite (~1-2 minutes best case, 7-12 minutes worst case with hangs). There's no way for an agent to quickly check "do tests pass?" vs running a full diagnostic.

### RC5: Backend CLAUDE.md is stale
The backend `CLAUDE.md` says "No test suite yet (coming Day 4)" but there are 25 test files with ~735 test functions. Agents reading this think there's nothing to test, leading to confusion and wasted turns discovering the test infrastructure.

## Fix Plan

### Fix 1: Add pytest-timeout to backend

**What:** Install `pytest-timeout` and configure a 10-second default per-test timeout.

**Changes:**
- Run `cd backend && uv add --dev pytest-timeout`
- Add to `backend/pyproject.toml` under `[tool.pytest.ini_options]`:
  ```toml
  [tool.pytest.ini_options]
  timeout = 10
  timeout_method = "thread"
  asyncio_mode = "auto"
  addopts = "-v --tb=short"
  ```

- Keep `timeout_method = "thread"` as the global default (correct for threading-heavy tests).
- Add `asyncio_mode = "auto"` to the same config block so async test functions are collected without manual `@pytest.mark.asyncio` on every test. (This is a QoL improvement bundled with the timeout config, not a timeout fix itself.)
- Create `backend/tests/conftest.py` with a `pytest_collection_modifyitems` hook that switches async tests to `signal` timeout method:
  ```python
  import asyncio
  import pytest

  def pytest_collection_modifyitems(items):
      """Use signal-based timeout for async tests (thread method can't interrupt event loops)."""
      for item in items:
          if asyncio.iscoroutinefunction(item.obj):
              item.add_marker(pytest.mark.timeout(method="signal"))
  ```
  **Why the hook approach instead of a fixture:** With `asyncio_mode = "auto"`, async tests don't receive the `@pytest.mark.asyncio` marker at collection time — the marker is added later by the plugin. A fixture checking `get_closest_marker("asyncio")` would never match, silently leaving all async tests on the `thread` timeout method. The `pytest_collection_modifyitems` hook inspects `item.obj` directly with `asyncio.iscoroutinefunction()`, which is reliable regardless of how the test was registered as async.

  **Why `signal`:** The `thread` timeout method fires `KeyboardInterrupt` from a separate thread, which asyncio's event loop exception handler may swallow — meaning async tests can hang past the timeout. The `signal` method uses `SIGALRM`, which reliably interrupts the event loop.

  **Platform note:** `SIGALRM` is Unix-only. This hook has no effect on Windows. Not a concern for this project but should be documented for future contributors.

  **Limitation:** The hook only detects `async def` test functions via `iscoroutinefunction()`. Sync tests that use async fixtures indirectly are not switched. In this codebase all async tests are explicitly `async def` (12 functions in `test_youtube.py`), so this is a non-issue currently.

**Why 10 seconds:** The slowest legitimate unit tests (audio processing in `test_processor.py`) should complete in 2-3 seconds. Threading fixtures have 30-second waits that are the primary hang risk. A 10-second ceiling is generous enough for unit tests but catches hangs before they burn agent time. Known-slow integration tests get explicit `@pytest.mark.timeout(30)` overrides (see Fix 2).

**Impact:** Prevents indefinite hangs. Any test that exceeds 10 seconds fails fast instead of silently blocking.

### Fix 2: Add pytest markers for test categories

**What:** Define the `slow` marker and tag existing tests to enable selective execution.

**Changes:**
- Add marker to `backend/pyproject.toml`:
  ```toml
  markers = [
      "slow: tests that take >2 seconds (audio processing, subprocess calls)",
  ]
  ```

- Tag known slow tests with `@pytest.mark.slow`:
  - **File-level** (all tests in the file are slow):
    - `test_pipeline_sse.py` — threading fixtures with event waits
    - `test_youtube_endpoint.py` — threading + event waits
  - **Class-level** within `test_pipeline_wiring.py`:
    - `TestFullPipelineWithMocks` — runs full pipeline
    - `TestPipelineEmitsProgressEvents` — runs full pipeline
    - `TestPipelineSetsSessionState` — runs full pipeline
    - `TestPipelineHandlesSeparationError` — runs full pipeline
    - `TestPipelineWithSoundQualityFlags` — runs full pipeline with flag combos
    - `TestLossySourceWiring` — runs full pipeline with lossy source flags
    - `TestAutoLeveler4State` — runs full pipeline
    - **Leave `TestLoudnessFixPipeline` UNMARKED** — 7 fast numpy mastering chain tests (true peak limit, soft clip, LUFS normalize). These are the most important "is the mastering chain broken?" fast-feedback tests and must stay in the quick tier.
  - **Class-level** within `test_processor.py`:
    - `TestRubberband` — subprocess calls to rubberband
    - `TestExportMp3` — only `test_export` actually calls ffmpeg; the other two tests mock `subprocess.run`. Use method-level `@pytest.mark.slow` on `test_export` only, keeping the mocked tests in the fast tier
    - Leave fast unit-test classes unmarked: `TestValidateStem`, `TestTrimAudio`, `TestBandpassFilter`, `TestComputeTempoPlan`, `TestCrossSongLevelMatch`, `TestLufsNormalize`, `TestSoftClip`, `TestTruePeak`, `TestTruePeakLimit`, `TestApplyFades`
  - **Class-level** within `test_taste_stage.py`:
    - `TestTimeoutFallback` — deliberately calls `time.sleep(2.0)`, currently under 10s but fragile
    - `TestFlagGating` — runs the full pipeline with real librosa/rubberband/ffmpeg; needs both `@pytest.mark.slow` and `@pytest.mark.timeout(30)`
  - **File-level** for `test_mixer.py` — all 6 tests call ffmpeg via `overlay_and_export()`

- **Add explicit `@pytest.mark.timeout(30)` on all slow-marked test files and classes.** The slow marker and the timeout override serve different purposes: the marker controls *when* tests run (`-m "not slow"`), the timeout controls *how long they're allowed to run*. Without this, running `pytest` without `-m "not slow"` would cause mass false failures on slow tests, and agents would enter debugging loops trying to figure out why tests "broke." Specifically, apply `@pytest.mark.timeout(30)` to:
  - All slow pipeline classes in `test_pipeline_wiring.py` (listed above)
  - `test_pipeline_sse.py` (file-level) — threading fixtures with 30-second event waits
  - `test_youtube_endpoint.py` (file-level) — threading + event waits
  - `test_mixer.py` (file-level) — ffmpeg subprocess calls
  - `TestFlagGating` in `test_taste_stage.py` (already noted above)

**Why class-level for `test_pipeline_wiring.py`:** This file contains both integration tests (full pipeline, genuinely slow) and unit tests (`TestLoudnessFixPipeline` — pure numpy DSP, sub-second). File-level `@pytest.mark.slow` would exclude the most valuable mastering chain tests from quick runs.

**Why only `slow` marker:** The `integration` and `unit` markers add no value unless tests are explicitly tagged with them. Only `slow` is actionable for the `-m "not slow"` use case. Add finer markers later if needed.

**Impact:** Agents can run `pytest -m "not slow"` for a quick pass/fail check (~10 seconds) vs the full suite (~1-2 minutes).

### Fix 3: Update backend CLAUDE.md with correct test information

**What:** Replace the stale "No test suite yet" section with accurate test documentation. Absorbs the sandbox documentation (formerly Fix 6).

**Changes to `backend/CLAUDE.md`:**
- Remove "No test suite yet (coming Day 4)"
- Add a Testing section with:
  - How to run tests: `uv run pytest tests/ -v`
  - Quick check (fast tests only): `uv run pytest tests/ -m "not slow" -v`
  - Full suite: `uv run pytest tests/ -v`
  - Test file categories (which files are slow, which are fast)
  - **Sandbox note:** Always use `dangerouslyDisableSandbox: true` when running pytest via Bash tool (uv cache access requires it)
  - **Timeout override:** Agents can use `--timeout=30` or `--timeout=0` (disable) on a per-run basis if needed
  - Number of test files (~25) and approximate run times (quick: ~10s, full: ~1-2min)

### Fix 4: Update workspace CLAUDE.md with cross-repo testing section

**What:** Add a "Testing" section to the workspace-level `CLAUDE.md` that covers both repos.

**Changes to `CLAUDE.md` (workspace root):**
- The workspace `CLAUDE.md` already has a Testing section (around line 128) that discusses test philosophy. UPDATE/extend that existing section with the command documentation below — do not create a duplicate section.
- Document the correct test commands for each repo:
  - Frontend: No test suite configured yet — `frontend/package.json` has no `test` script. Skip frontend tests until a test runner is added.
  - Backend: `cd backend && uv run pytest tests/ -v`
  - Backend (quick): `cd backend && uv run pytest tests/ -m "not slow" -v`
- Note that all backend test commands require `dangerouslyDisableSandbox: true`
- Guidance for subagents: "When implementing backend changes, run quick tests first (`-m 'not slow'`). Only run the full suite before committing."

### Fix 5: Update skills to detect Python repos and use correct test/lint/type-check commands

**Status: DEFERRED to separate PR.** This fix touches 6 global skill files shared across all projects. Fixes 1-4 solve the immediate velocity blocker for musicMixer by updating CLAUDE.md (which agents read before skills). Fix 5 is a global tooling improvement that deserves its own review cycle.

**Skills to update (when this is picked up):**
- `~/.claude/skills/ship/SKILL.md`
- `~/.claude/skills/fix-the-things/SKILL.md`
- `~/.claude/skills/wrap-it-up/SKILL.md`
- `~/.claude/skills/code-review/SKILL.md`
- `~/.claude/skills/code-review-critical/SKILL.md`
- `~/.claude/skills/team-three-review/SKILL.md`

**Note:** `execute-plan` does NOT hardcode `npm test`. `code-review-critical` and `team-three-review` DO. When scanning skill files, search for both `npm run type-check` and `npm run typecheck` (no hyphen) — `fix-the-things` uses the no-hyphen variant.

**Approach:** In each skill's test/lint/type-check execution sections, replace the hardcoded commands with detection logic:
```
Test/lint/type-check command detection:
- If pyproject.toml with [tool.pytest.ini_options] exists, or a Python project signal is present (e.g., `requirements.txt`, `uv.lock`, `poetry.lock`, or `[dependency-groups]` in pyproject.toml):
  - Test: `uv run pytest tests/ -v` (with dangerouslyDisableSandbox)
  - Lint: `uv run ruff check .` (if ruff configured) or skip
  - Type-check: `uv run pyright` (if configured) or skip
- If package.json exists:
  - Test: check that a `"test"` key exists in the `"scripts"` section of package.json
    before running `bun run test`. If no `test` script is defined, skip with
    a note: "No test script configured in package.json — skipping tests."
  - Lint: check for `"lint"` key in scripts; run `bun run lint` if present, skip otherwise
  - Type-check: check for `"type-check"` or `"typecheck"` key in scripts; run if present, skip otherwise
- If neither: skip with warning
```

**cwd guidance:**
- Detection is cwd-relative. Skill subagents must `cd` into the correct child repo before running tests.
- If neither config file found in cwd, check immediate subdirectories for `pyproject.toml` or `package.json`.
- If both exist in the same directory, check for `[tool.pytest.ini_options]` in pyproject.toml — if present, treat as Python project.

## Implementation Order

### Step 0: Validate timing assumption (5 min)
```bash
cd backend && uv run pytest tests/test_pipeline_wiring.py -v --durations=0
```
If all tests complete under 10 seconds: Fix 1 and Fix 2 can ship independently (the coupling constraint disappears). If some exceed 10 seconds: they must ship together as originally planned.

### Step 1: Add pytest-timeout + markers (Fix 1 + Fix 2)
- Install `pytest-timeout`, add config to `pyproject.toml`
- Create `conftest.py` with async timeout hook
- Apply `@pytest.mark.slow` at class/file level per Fix 2
- Apply `@pytest.mark.timeout(30)` on slow pipeline wiring classes

### Step 2: Verify (5 min)
```bash
uv run pytest tests/ --timeout=10 -v
```
All tests should pass. If any legitimate tests fail on timeout, add `@pytest.mark.timeout(30)` to those specific tests/classes. Also verify:
- Threading tests in `test_pipeline_sse.py` are properly interrupted (not hanging)
- Total suite runtime is under 2 minutes
- `pytest -m "not slow"` completes in ~10 seconds

### Step 3: Update backend CLAUDE.md (Fix 3, 5 min)
- Replace stale "No test suite yet" with accurate docs + sandbox note

### Step 4: Update workspace CLAUDE.md (Fix 4, 3 min)
- Add backend test commands to Testing section

### Step 5: Commit to integration branch
- Commit all changes before spawning any worktree agents
- **This is critical:** worktrees snapshot from the branch point, so uncommitted CLAUDE.md changes are invisible to worktree agents
- **Dual-repo boundary:** Backend changes (Fix 1, Fix 2, Fix 3) live in the `backend/` child repo and must be committed there before creating worktrees that branch from it. Workspace changes (Fix 4) live in the parent workspace repo, which worktrees do not snapshot — it is always visible at its original path. An implementing agent must NOT accidentally commit workspace-level changes (e.g., workspace `CLAUDE.md`) to the backend repo, or vice versa.

### Step 6 (separate PR, later): Skill updates (Fix 5)
- Scope, test, and review as a global tooling change

**Total time for Steps 0-5: ~25 minutes.**

## Success Criteria

After implementing these fixes:
- No subagent should spend more than 2 minutes running backend tests
- No subagent should hit a sandbox permission error when running pytest
- No subagent should use `npm test` for the backend (once Fix 5 ships)
- Agents should be able to run a quick "do tests pass?" check in ~10 seconds
- Backend CLAUDE.md should accurately reflect the test infrastructure
- `TestLoudnessFixPipeline` (mastering chain tests) runs in the fast tier

## Risks

- **Marker coverage may be incomplete initially.** Not all 735 tests will be tagged in the first pass. Untagged tests run by default, which is the safe behavior.
- **10-second timeout may be too aggressive** for some legitimate tests. Mitigated by: (a) explicit `@pytest.mark.timeout(30)` on known-slow classes, (b) agents can override with `--timeout=30` or `--timeout=0` per-run, (c) Step 2 validation catches false failures before shipping.
- **Non-deterministic BPM flakiness (post-demo follow-up):** `test_pipeline_wiring.py` runs real `librosa` BPM detection on synthetic sine waves. BPM estimates on synthetic signals are fragile and can change between librosa versions. Timeouts don't fix this. Long-term fix: mock the analysis stage to return fixed BPM/key values.
- **Temp file cleanup on timeout kills:** When pytest-timeout kills a test via SIGALRM, fixture teardown may be skipped, leaving temp files. Minor operational concern, not a correctness issue.

---

## Review History

| Reviewer | Role | Key Findings |
|----------|------|-------------|
| Expert 1 | Senior Engineer | Timing assumption for Fix 1+2 coupling is unverified (run Step 0 first); no validation step (added Step 2); Fix 5 is scope creep (deferred); Fix 6 is redundant (merged into Fix 3/4); minimum viable fix is 3 items in ~25 min |
| Expert 2 | ML Scientist | Pipeline wiring tests need explicit `@pytest.mark.timeout(30)` as safety net (not just slow marker); `test_taste_stage.py::TestTimeoutFallback` missing from slow list; three-tier markers over-scoped (only `slow` needed); non-deterministic BPM detection is a flakiness source timeouts can't fix |
| Expert 3 | Sound Engineer | `test_pipeline_wiring.py` must use class-level markers (not file-level) to keep `TestLoudnessFixPipeline` in fast tier; `test_mixer.py` uses real ffmpeg (add to slow list); missing rubberband slowdown direction test (post-demo); missing reverse gain direction test for `cross_song_level_match` (post-demo) |

Full reviews: `notes/2026-02-28-test-reliability-review-{engineer,ml-scientist,sound-engineer}.md`
