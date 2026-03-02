# Plan: Prompt Caching + 4-Stem Code Removal

## Context

The LLM interpreter (`interpreter.py`) costs ~$0.028 per remix request. The system prompt mixes static rules with dynamic song data in a single string, defeating Anthropic's prompt caching. Additionally, the codebase maintains dead 4-stem code paths (~160 lines) that are no longer needed — the app is 6-stem only. This plan removes the dead code and restructures the prompt for caching.

**Expected savings:** ~80-85% on total input cost (~90% on cached input tokens specifically). ~160 lines of dead code removed.

**Model note:** The 1024-token minimum cache threshold applies to `claude-sonnet-4-20250514`. Upgrading to Sonnet 4.6 (min 2048 tokens) or Opus 4.x (min 4096 tokens) could silently disable caching if the static block is below the new threshold. The `cache_creation_input_tokens` check in manual validation will catch this.

## Files Modified

| File | Changes |
|------|---------|
| `backend/src/musicmixer/services/interpreter.py` | All prompt/caching changes + 4-stem removal |
| `backend/tests/test_interpreter.py` | New tests |

---

## Phase 1: Remove 4-Stem Dead Code

**All deletions in this phase are prerequisites for Phase 2.** The static/dynamic section classification is only valid after 4-stem code is removed. Do not start Phase 2 until Phase 1 is complete and tests pass.

### Explicit deletion list (exhaustive)

1. **`FOUR_STEMS` constant (~L37)** — dead code, no consumers. Delete.

2. **`EXTRA_STEMS` constant (~L40)** — used only in `_adapt_schema_for_4stem()`. Delete after step 3.

3. **`_adapt_schema_for_4stem()` function (~L185-195)** and the `# 4-stem adaptation` comment block above it (~L182). Delete both.

4. **`_build_few_shot_messages_4stem()` function (~L923-1080, ~160 lines)** — Delete.

5. **Simplify `_build_few_shot_messages()` (~L724-727)**
   - Remove the `is_4stem` branch. Function unconditionally returns 6-stem examples.

6. **Remove 4-stem logic in `_build_system_prompt()` (~L215-236)**
   - Delete: `is_4stem = settings.stem_backend == "local"` (L215)
   - Replace ternaries with hardcoded literals:
     ```python
     stem_list = "vocals, drums, bass, guitar, piano, other"
     stem_count = 6
     ```
   - Delete: `stem_gains_note = ""` and the entire `if is_4stem:` block (~L230-236)
   - Delete: the `{stem_gains_note}` reference in the Section 1 f-string (~L248). Leaving the variable deleted but the reference present causes a `NameError`.
   - The `stem_gains_required = stem_list` variable (~L305) now always resolves to the same literal. Inline it directly: replace `{stem_gains_required}` with `{stem_list}` or just hardcode the string.

7. **Remove `settings.stem_backend == "local"` branches in `interpret_prompt()` (~L1391-1394)**
   - Tool schema is always `REMIX_PLAN_TOOL`. No adaptation.

8. **Add startup guard in `interpret_prompt()`**
   After removing 4-stem support from the interpreter, `stem_backend=local` produces a mismatch: separation returns 4 stems, LLM plans for 6, mixer silently drops guitar/piano. Add a guard:
   ```python
   if settings.stem_backend != "modal":
       raise ValueError(
           f"stem_backend={settings.stem_backend!r} is not supported. "
           "The interpreter requires 6-stem separation (stem_backend='modal')."
       )
   ```
   Place this at the top of `interpret_prompt()`, before any LLM call.

---

## Phase 2: System Prompt Caching

**Requires Phase 1 complete.** The caching strategy depends on `stem_list`, `stem_count`, and `stem_gains_note` being true constants.

### Prefix-based caching with section reordering

Anthropic's prompt caching is **prefix-based**: the cached region is everything from the start up to the `cache_control` breakpoint. The current code interleaves static and dynamic sections, which limits the cacheable prefix to only sections 1-3 (~30% of the prompt).

**Solution: reorder sections to group all static sections first, then all dynamic sections.** This is safe because the sections are independent instruction blocks with no cross-references ("as mentioned above", etc. — verified via grep). The LLM processes them as a set of rules, not a sequential narrative.

**Reordered layout:**

| Block | # | Section | Static/Dynamic |
|-------|---|---------|----------------|
| **Cached** | 1 | Role + Constraints | Static |
| | 2 | Failure Mode Guards | Static |
| | 3 | Mixing Advisory | Static |
| | 4 | Section Rules | Static |
| | 5 | Genre Guidance | Static |
| | 6 | Stem Artifacts | Static |
| | 7 | Explanation/Warnings | Static |
| **Uncached** | 8 | Transitions | Dynamic (`{stretch_advisory}`) |
| | 9 | Arrangement Templates | Dynamic (`{total_available_beats}`) |
| | 10 | Tempo/Key Guidance | Dynamic (`{key_matching_detail}`) |
| | 11 | Ambiguity Handling | Dynamic (`{total_available_beats}`) |
| | 12 | Song Data (5 layers) | Dynamic (per-request metadata) |
| | 13 | Beat Reference | Dynamic (`{target_bpm}`) |

**Result: 7 static sections cached (~55-60% of prompt tokens) vs. 3 previously.** Nearly double the cache coverage.

### 9. Refactor `_build_system_prompt()` → `_build_system_prompt_blocks()`

Create the new function **alongside** the old one (do not rename yet). The old function remains intact and callable so the code is never broken between phases.

```python
def _build_system_prompt_blocks(
    song_a_meta: AudioMetadata,
    song_b_meta: AudioMetadata,
    key_matching_available: bool,
    key_matching_detail: str,
    total_available_beats: int,
    stretch_pct: float | None = None,
    lyrics_a: LyricsData | None = None,
    lyrics_b: LyricsData | None = None,
) -> list[dict]:
    """Construct system prompt as two content blocks for Anthropic prompt caching.

    Block 1 (cached): all 7 static sections grouped first.
    Block 2 (uncached): all 6 dynamic sections grouped after.

    Sections are reordered from original layout to maximize cache prefix.
    This is safe — sections are independent instruction blocks with no
    cross-references. Verified by grep for "above/below/previous/following".
    """
    # --- Static block: all static sections ---
    static_sections = [
        section_role_constraints,       # was section 1
        section_failure_mode_guards,    # was section 2
        section_mixing_advisory,        # was section 3
        section_rules,                  # was section 6
        section_genre_guidance,         # was section 7
        section_stem_artifacts,         # was section 10
        section_explanation_warnings,   # was section 11
    ]
    static_block = {
        "type": "text",
        "text": "\n\n".join(static_sections),
        "cache_control": {"type": "ephemeral"},
    }

    # --- Dynamic block: all dynamic sections ---
    dynamic_sections = [
        build_transitions(stretch_pct),             # was section 4
        build_arrangement_templates(total_beats),   # was section 5
        build_tempo_key(key_matching_detail),       # was section 8
        build_ambiguity(total_beats),               # was section 9
        build_song_data(song_a_meta, song_b_meta, lyrics_a, lyrics_b),  # was section 12
        build_beat_reference(target_bpm),           # was section 13
    ]
    dynamic_block = {
        "type": "text",
        "text": "\n\n".join(dynamic_sections),
    }

    return [static_block, dynamic_block]
```

**Note:** The equivalence test changes from byte-identical comparison to a **content-equivalent** comparison: both old and new prompts contain the same sections with the same text, just in different order. The test should verify that the set of section texts is identical, not the joined string.

Note: the helper functions `_build_section_1_role_constraints()`, etc. are optional internal refactors for readability — the same result is achieved by extracting the relevant sections list entries from the existing `_build_system_prompt()` body. Preserve the exact text content; the only structural change is splitting into two blocks.

### 10. Add `cache_control` to last few-shot message

Add the `cache_control` key **inline** in the dict literal inside `_build_few_shot_messages()`, on the last `tool_result` block (the final message in the returned list). Do not add it as a post-hoc mutation in the caller.

```python
# In _build_few_shot_messages(), last message:
{
    "role": "user",
    "content": [{
        "type": "tool_result",
        "tool_use_id": "example_3",
        "content": "Plan accepted.",
        "cache_control": {"type": "ephemeral"},  # <-- add here
    }]
}
```

`tool_result` blocks support `cache_control` per the Anthropic docs. No fallback to `tool_use` block is needed.

**Cache breakpoint hierarchy** (Anthropic ordering: tools → system → messages):
- `tools`: `REMIX_PLAN_TOOL` is always static (6-stem only). It sits at the top of the prefix and is implicitly cached. **Any change to the tool schema (including whitespace or description edits) invalidates all downstream caches (system + messages).**
- `system`: `cache_control` on the last static block (sections 1-3) = first breakpoint
- `messages`: `cache_control` on the last few-shot `tool_result` = second breakpoint

---

## Phase 3: API Call Update

**Atomic transition with Phase 2.** Once `_build_system_prompt_blocks()` exists and is tested, update the caller in `interpret_prompt()` in a single commit.

### 11. Update `interpret_prompt()` API calls

Replace `system_prompt = _build_system_prompt(...)` with `system_blocks = _build_system_prompt_blocks(...)`.

Update **both** API call sites — the primary call (~L1411) and the duration-retry call (~L1424):
```python
# Before
system=system_prompt,  # str

# After
system=system_blocks,  # list[dict]
```

The Anthropic SDK accepts `list[TextBlockParam]` for the `system` parameter. Both calls must be updated in the same commit.

**Duration retry path:** The retry loop appends messages after the `cache_control` breakpoint. The cached prefix (tools + system blocks + few-shot messages) is unchanged across attempts, so the second attempt gets a cache hit. `messages = _build_few_shot_messages() + [...]` uses `+` (creates a new list), so retry appends do not mutate the few-shot source.

### 12. Add cache stats logging

```python
cache_read = getattr(response.usage, "cache_read_input_tokens", 0)
cache_created = getattr(response.usage, "cache_creation_input_tokens", 0)
logger.info(f"Cache stats: read={cache_read}, created={cache_created}")
```

Use `getattr` with defaults to handle older SDK versions that lack cache usage fields.

---

## Phase 4: Cleanup

13. Delete the old `_build_system_prompt()` function (the string-returning version). All callers have been updated in Phase 3.
14. Run full test suite. All existing tests must pass unchanged.

---

## Testing

**New unit tests:**

- `test_build_system_prompt_blocks_structure` — returns exactly 2 blocks; first has `cache_control: {type: ephemeral}`, second does not
- `test_static_block_is_constant` — different songs/BPMs produce identical `block[0]["text"]` (the cached block content never changes)
- `test_dynamic_block_contains_song_metadata` — song names appear in `block[1]["text"]` only, not `block[0]["text"]`
- `test_blocks_text_equals_original` — **parameterized**: `block[0]["text"] + "\n\n" + block[1]["text"]` is byte-identical to old `_build_system_prompt()` output for all variants:
  - (a) no lyrics, no stretch
  - (b) with lyrics
  - (c) with `stretch_pct > 12`
  - (d) with key matching detail
- `test_few_shot_messages_always_6stem` — all 6 stems present in every section's `stem_gains` in the returned few-shot list
- `test_few_shot_last_message_has_cache_control` — `_build_few_shot_messages()[-1]["content"][-1]["cache_control"]` is `{"type": "ephemeral"}`
- `test_cache_stats_logging_no_crash` — mock `response.usage` without cache fields, verify no exception
- `test_interpret_prompt_passes_blocks_to_api` — mock `anthropic.Anthropic.messages.create`, assert the `system` kwarg is `list[dict]` with 2 entries, first has `cache_control`
- `test_stem_backend_local_raises` — assert `interpret_prompt()` raises `ValueError` when `settings.stem_backend == "local"`

**Existing tests:** All pass unchanged (no behavior change in the 6-stem path).

**Manual validation:** Make 2 requests within 5 minutes, check logs for `cache_read_input_tokens > 0` on the second request.

---

## Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Static block below 1024 token minimum (Claude Sonnet 4) | Low | ~2500+ tokens for sections 1-3; verify via `cache_creation_input_tokens`. Minimum threshold increases on model upgrade — recheck if model changes. |
| 5-minute TTL → cache misses in dev/staging | Medium | Expected during development; not a correctness issue. Production traffic frequency is sufficient to keep cache warm. |
| SDK lacks cache usage fields | Very low | `getattr` with defaults handles gracefully |
| Tool schema change invalidates all caches | Low | Document in code: any change to `REMIX_PLAN_TOOL` busts system + message caches. Review tool schema changes carefully. |
| Phase 2 started before Phase 1 complete | Low | Phase 1 must be committed and tests passing before Phase 2 begins |
