# Plan: Enforce Song A = Vocals, Song B = Instrumentals

## Context

Currently, the LLM decides which song provides vocals based on audio analysis (vocal prominence, separation quality, etc.). This means the user has to specify in their prompt things like "vocals from Song A, instrumentals from Song B" — or hope the LLM picks correctly.

The user wants to enforce a fixed convention: **Song A always provides vocals, Song B always provides instrumentals.** This simplifies the UX — users know which slot is which — and removes a decision point from the LLM, making results more predictable.

## Changes

### 1. Frontend: Update labels and hints

**`frontend/src/components/RemixForm.tsx`** (lines 77, 87)
- Change `label="Song A"` → `label="Song A (Vocals)"`
- Change `label="Song B"` → `label="Song B (Instrumentals)"`

**`frontend/src/components/PromptInput.tsx`** (lines 7-11)
- Update `EXAMPLES` array to remove role-specifying language. New examples:
  - `"Layer the vocals over a chill beat, boost the bass"`
  - `"Make it sound like a club remix with heavy drums"`
  - `"Slow it down and keep it mellow"`
- These focus on **vibe/style** instead of which-song-does-what, since roles are now fixed.

### 2. Backend: Remove `vocal_source` from LLM tool schema

**`backend/src/musicmixer/services/interpreter.py`**

**Tool schema** (lines 60-76):
- Remove `vocal_source` from `required` array
- Remove `vocal_source` property entirely from the schema
- The LLM no longer gets to choose; it's hardcoded downstream

**System prompt — Section 1** (lines 206-220):
- Remove the capability line: `"Choose which song provides vocals (vocal_source)"`
- Add explicit constraint: `"Song A ALWAYS provides vocals. Song B ALWAYS provides all instrumentals. This is fixed — do not reference vocal_source in your output."`
- Update the CONSTRAINTS block (line 209): change `"Vocals ALWAYS come from one song"` → `"Vocals ALWAYS come from Song A. Instrumentals ALWAYS come from Song B."` to eliminate ambiguity about which song provides which role.

**Cross-song analysis — Layer 4** (`_build_cross_song_layer`, lines 597-610):
- Remove the vocal source recommendation block (the `vp_a > vp_b` comparison). Since vocal source is fixed, telling the LLM which song has "cleaner" vocals is misleading noise.

**Few-shot examples** (lines 682-873):
- Example 1 (line 744): Remove `"vocal_source": "song_a"` from tool output
- Example 2 (line 803): Change `"vocal_source": "song_b"` → remove field. Update the explanation text to not mention choosing Song B for vocals.
- Example 3 (line 850): Same — remove `"vocal_source": "song_b"`, update explanation.
- For examples 2 & 3 (which currently use song_b vocals), rewrite the example scenarios so Song A is the vocal source. The examples should demonstrate arrangement variety, not vocal source selection.

**Parser** (`_parse_remix_plan`, line 898):
- Hardcode `vocal_source="song_a"` instead of reading from `raw["vocal_source"]`

**Validator** (`_validate_remix_plan`, lines 925-926, 1017-1023):
- Remove conditional logic that checks `plan.vocal_source`. Always use `song_a_meta` for vocal and `song_b_meta` for instrumental.

**`interpret_prompt`** (lines 1167-1172):
- Remove the comment "Pre-LLM: assume song_a=vocal as default" — it's no longer an assumption, it's the rule.

### 3. Backend: Simplify pipeline routing

**`backend/src/musicmixer/services/pipeline.py`** (lines 385-416):
- Remove the `if plan.vocal_source == "song_a" ... else ...` conditional
- Hardcode: `vocal_stems_paths = song_a_stems`, `inst_stems_paths = song_b_stems`, `vocal_meta = meta_a`, `inst_meta = meta_b`
- Similarly simplify the lossy source flag mapping (lines 414-415)
- Simplify BPM logging (lines 385-388)

### 4. Backend: Simplify fallback plan

**`backend/src/musicmixer/services/interpreter.py`** (`generate_fallback_plan`, lines 1345-1393):
- Already defaults to `song_a` — no code change needed, but update the comment to say "Fixed convention" instead of "Defaults to song_a"

### 5. Backend: Hardcode vocal source in analysis

**`backend/src/musicmixer/services/analysis.py`** (`compute_relationships()`, lines 1498-1548)

Currently, `compute_relationships()` dynamically determines `vocal_source` by comparing vocal prominence between Song A and Song B (lines 1498-1502). This value is stored on `CrossSongRelationships.vocal_source` (models.py line 105) and also drives the BPM stretch calculation (lines 1536-1542). After enforcing the fixed convention, the dynamic computation must be replaced:

- **Hardcode vocal source** (lines 1498-1502): Replace the `prom_a_db >= prom_b_db` comparison with `vocal_source = "song_a"`. The prominence values are still computed and stored (they're useful context), but they no longer determine vocal source.
- **Simplify stretch calculation** (lines 1536-1542): Remove the `if vocal_source == "song_b"` branch. Always use `_vocal_bpm = meta_a.bpm` and `_inst_bpm = meta_b.bpm`.
- **Simplify instrumental source** (lines 1504-1510): Remove the `if vocal_source == "song_a" else` conditional — `instrumental_source_meta` is always `meta_b`.

**`backend/src/musicmixer/models.py`** (`CrossSongRelationships.vocal_source`, line 105):
- Keep the field but add a comment: `# Always "song_a" — fixed convention, Song A provides vocals`
- This field now always matches `RemixPlan.vocal_source`. Both are always `"song_a"`.

### 6. Model: Keep `vocal_source` field (low-risk)

**`backend/src/musicmixer/models.py`** (line 180):
- Keep the `vocal_source` field on `RemixPlan` — it's used in logging and downstream code. It's just always `"song_a"` now. Add a comment: `# Always "song_a" — Song A is the fixed vocal source`

## Files Modified

| File | Change |
|------|--------|
| `frontend/src/components/RemixForm.tsx` | Update Song A/B labels |
| `frontend/src/components/PromptInput.tsx` | Update example prompts |
| `backend/src/musicmixer/services/interpreter.py` | Remove vocal_source from schema, update system prompt + CONSTRAINTS block, fix examples, hardcode in parser/validator |
| `backend/src/musicmixer/services/pipeline.py` | Remove vocal source conditional routing |
| `backend/src/musicmixer/services/analysis.py` | Hardcode `vocal_source = "song_a"` in `compute_relationships()`, simplify stretch/instrumental logic |
| `backend/src/musicmixer/models.py` | Add comments to `vocal_source` on both `RemixPlan` and `CrossSongRelationships` |

## Verification

1. **Unit check**: Run backend tests (`cd backend && uv run pytest`) to catch any regressions
2. **Manual check**: Start dev servers, upload two test songs, submit a prompt — verify:
   - UI shows "Song A (Vocals)" and "Song B (Instrumentals)"
   - Prompt examples don't mention song roles
   - Remix uses Song A vocals regardless of prompt wording
3. **LLM check**: Verify the LLM's tool output no longer contains `vocal_source` and the system prompt clearly states the fixed convention

## Subagent Deployment

Two parallel subagents:
1. **Frontend agent**: Changes to `RemixForm.tsx` and `PromptInput.tsx`
2. **Backend agent**: Changes to `interpreter.py`, `pipeline.py`, `analysis.py`, and `models.py`
