# Plan: A/B Test Suite — Reuse LLM Plan Across Variants

## Context

The A/B test suite burns through Anthropic API credits because every variant of every pair makes its own LLM call, even though all variants of the same pair produce the **identical** remix plan (same songs, same prompt). Each variant runs as a separate subprocess, and stem separation (6-10 min) runs before the LLM call, so Anthropic's 5-minute cache TTL always expires. Result: every call is a cache miss at ~$0.035 each.

**Fix:** Call the LLM once per pair, save the plan to a JSON file, and have subsequent variants load it — skipping the LLM entirely.

**Savings:** Compare mode drops from 10 → 5 LLM calls (50%), sweep mode from 25 → 5 (80%).

## Changes

### 0. Prerequisite crash fix — `backend/scripts/run_pipeline_phase.py`

The A/B suite already passes `--prompt` to `run_pipeline_phase.py`, but the script doesn't define that argument. This is not a latent bug — it's a **hard crash** that fails every run immediately. Must be fixed first.

- Add `--prompt` argument to `run_pipeline_phase.py`, pass it through to `run_pipeline()`
- **Side effect:** after this fix, the LLM will receive actual user prompts instead of empty strings. Cached plans generated after this fix will differ from any prior behavior, which is expected and correct.

### 1. Add serialization helpers — `backend/src/musicmixer/models.py`

Add `remix_plan_to_dict(plan)` using `dataclasses.asdict()` after the `RemixPlan` class (~line 191).

For deserialization, reuse the existing `_parse_remix_plan()` in `interpreter.py` (line ~905) — it already reconstructs a `RemixPlan` from a raw dict, handling nested `Section` objects. Rename it to `parse_remix_plan` (or add a thin public wrapper) and add an optional `used_fallback: bool = False` param so the cache-load path can propagate that flag. Do **not** add a separate `remix_plan_from_dict()` — that would duplicate logic.

### 2. Add `plan_file` param — `backend/src/musicmixer/services/pipeline.py`

- Add `plan_file: str | None = None` to `run_pipeline()` signature (line 43)
- At the LLM call site (line 366), replace direct `interpret_prompt()` call with:
  - If `plan_file` exists on disk → load JSON dict and reconstruct via `parse_remix_plan()`, skip LLM
  - Otherwise → call `interpret_prompt()` as normal, then save plan to `plan_file` if provided
- Wrap both load and save in try/except so a corrupt file gracefully falls back to LLM
- **Validation note:** The plan returned by `interpret_prompt()` is already post-`_validate_remix_plan()`. Serialization captures the validated plan as-is. On load, do **not** re-run validation — re-validation could mutate the plan (e.g. bar-boundary snapping) or reject a borderline plan that was already accepted.
- **Fingerprinting:** Include a `fingerprint` field in the saved JSON — a hash of `(song_a_path, song_b_path, prompt)`. On load, recompute the fingerprint from the current inputs and compare. If mismatch, log a warning and regenerate (call LLM fresh). This prevents stale plan reuse if the output directory isn't cleaned between runs.

### 3. Add CLI arg — `backend/scripts/run_pipeline_phase.py`

- Add `--plan-file` argument, pass through to `run_pipeline()`
- (`--prompt` is already handled by Change 0)

### 4. Pass plan file per pair — `scripts/run_ab_test_suite.py`

- Create `plan_file = mashup_dir / f"{pair_name}-plan.json"` once per pair
- Pass `--plan-file` to every `run_pipeline_variant()` call
- First variant writes the file, subsequent variants reuse it
- Files auto-clean because `mashup_dir` is wiped at suite start

### 5. Same for shell script — `scripts/run_modal_ab_phases.sh`

- Add `PLAN_FILE="$OUT/remix-plan.json"` and pass `--plan-file "$PLAN_FILE"` to each `run_phase` call
- Add `rm -f "$PLAN_FILE"` before the first `run_phase` call as a belt-and-suspenders guard against stale plan files from prior runs

## Files Modified

| File | Change |
|------|--------|
| `backend/scripts/run_pipeline_phase.py` | **(Change 0)** Add `--prompt` arg (crash fix) |
| `backend/src/musicmixer/models.py` | Add `remix_plan_to_dict()` |
| `backend/src/musicmixer/services/interpreter.py` | Make `_parse_remix_plan` public, add `used_fallback` param |
| `backend/src/musicmixer/services/pipeline.py` | Add `plan_file` param, load/save + fingerprint logic at step 4 |
| `backend/scripts/run_pipeline_phase.py` | Add `--plan-file` CLI arg |
| `scripts/run_ab_test_suite.py` | Create plan path per pair, pass to variants |
| `scripts/run_modal_ab_phases.sh` | Pass `--plan-file` to each phase, `rm -f` stale plan before first run |

## Verification

1. Run A/B suite: `cd backend && uv run python ../scripts/run_ab_test_suite.py --pairs 1`
2. Check logs: should see 1x `"LLM request:"` and 1x `"loaded cached plan from"` for the pair
3. Verify `mashupTests/1-biggie-althea-plan.json` exists and contains valid plan JSON
4. Both variants should produce valid remix MP3s
