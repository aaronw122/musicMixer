# LLM Integration Reference

> Implementation reference extracted from the full plan. Canonical source: `docs/plans/2026-02-23-feat-mvp-prompt-based-remix-plan.md`

## Service Interface

```python
# services/interpreter.py

def interpret_prompt(prompt: str, song_a_meta: AudioMetadata, song_b_meta: AudioMetadata) -> RemixPlan:
    """Convert user prompt + song metadata into a structured remix plan.
    Uses Claude with tool_use for schema enforcement.
    Default: Sonnet (configurable via settings.llm_model).
    Falls back to deterministic plan on total failure.
    """
```

See `audio-pipeline.md` for `AudioMetadata`, `RemixPlan`, and `Section` dataclass definitions.

---

## System Prompt Structure

Target: ~2,500 tokens total (system prompt + few-shot + condensed context).

### Required Sections

1. **MVP constraint and capability boundaries:**
   - Vocals from one song, all instrumentals from the other. No cross-song stem mixing.
   - "You can combine stems, adjust volumes, choose sections, and control arrangement structure. You CANNOT add effects, generate new sounds, isolate individual instruments within 'other', or use vocals from both songs. Acknowledge limitations in `warnings`."

2. **Available stems:** vocals, drums, bass, other (guitar/synths/keys — be honest about what "other" contains)

3. **Song metadata format:**
   - BPM, key, duration, `total_beats` (eliminates LLM arithmetic), `vocal_energy_ratio`

4. **Energy regions (condensed format):**
   Do NOT pass raw `EnergyRegion` objects. Pre-process into condensed text (~10x fewer tokens):
   ```
   Song A (120 BPM, C major, 240s, 480 beats, vocal_energy: 0.82):
     Rhythmic (chorus/drop): 64s-80s, 128s-144s
     Sustained (breakdown): 48s-64s
     Sparse (verse): 16s-48s
     Moderate (intro/outro): 0s-16s, 200s-240s
   ```
   Group by `character` label. Include conversion note: "1 beat = 0.5s at 120 BPM."

5. **Arrangement templates** (constrains creative space to proven structures):
   ```
   Template A (Standard Mashup): intro(8 bars, inst) -> verse(16 bars, vocals) -> breakdown(8 bars, drums drop) -> drop(16 bars, full) -> outro(8 bars)
   Template B (DJ Set): build(16 bars) -> vocals in(16 bars) -> peak(16 bars) -> vocals out(8 bars) -> outro(8 bars)
   Template C (Quick Hit): intro(4 bars) -> vocal drop(16 bars) -> outro(4 bars)
   Template D (Chill): intro(16 bars, sparse) -> vocals(32 bars, gentle) -> outro(16 bars, fade)
   ```

6. **Tempo guidance** (separate from key):
   - `"average"` only when BPMs differ by <15%
   - 15-30%: prefer vocal source tempo
   - >30%: skip, explain in `explanation`

7. **Key guidance** (separate from tempo):
   - Skip if keys >4 semitones apart or `key_confidence` < 0.3

8. **Stem artifact awareness:**
   "Stem separation is imperfect. Vocal stem may contain instrument traces. Instrumental stems may contain ghost vocals. Bleed is less noticeable during high-energy sections."

9. **Section guidance:**
   - Sections should be 4, 8, or 16 beats long (max 16 per section)
   - Default: start with instrumental only (establishes beat before vocals enter)
   - Always end with instrumental only or a fade
   - Genre-appropriate patterns:
     - **Hip-hop/rap** (80-100 BPM): Keep drums consistent. Build energy through vocal intensity.
     - **EDM/dance** (120-130 BPM): breakdown -> build -> drop patterns.
     - **Pop/rock** (100-130 BPM): verse-chorus dynamics.
     - **R&B/soul** (60-90 BPM): Smooth transitions, gradual layering.
   - Infer genre from BPM range and energy profile
   - Total duration should approximate target (from source time ranges + tempo matching)

10. **Groove awareness** (known limitation):
    "Even when BPMs are matched, different groove feels can cause rhythmic clashing. When a half-time track is matched with a straight-time track, prefer breakdown sections from the instrumental."

11. **Alternative techniques for extreme tempo gaps (>50%):**
    - A cappella over breakdowns (no tempo matching needed)
    - Loop-based arrangement (4-8 bar instrumental phrase)
    - Focus on non-rhythmic sections

12. **Explanation quality:** Non-technical, 2-3 sentences, key creative decisions, no internal jargon.

13. **`warnings` field:** Populate for vague/contradictory/unverifiable prompts.

---

## Prompt Category Handling

| Category | Strategy |
|----------|----------|
| **Vague** ("make it cool") | Use energy profiles, pick vocals from higher `vocal_energy_ratio`, use standard template |
| **Contradictory** ("vocals from both") | Acknowledge in `warnings`, produce best plan within limits |
| **Genre jargon** ("trap", "lo-fi") | Translate to actions — tempo, gains, structure |
| **Inaudible references** ("guitar solo at 2:30") | Use the time range, add warning |

---

## Section Schema (tool_use format)

Use Claude `tool_use` for schema enforcement. The Section output format:

```json
{
  "sections": [
    {
      "label": "intro",
      "start_beat": 0,
      "end_beat": 16,
      "stem_gains": {"vocals": 0.0, "drums": 0.9, "bass": 0.8, "other": 1.0},
      "transition_in": "fade",
      "transition_beats": 4
    },
    {
      "label": "verse",
      "start_beat": 16,
      "end_beat": 48,
      "stem_gains": {"vocals": 1.0, "drums": 0.7, "bass": 0.8, "other": 0.5},
      "transition_in": "crossfade",
      "transition_beats": 4
    }
  ]
}
```

Full RemixPlan tool_use output includes: `vocal_source`, `start_time_vocal`, `end_time_vocal`, `start_time_instrumental`, `end_time_instrumental`, `sections`, `tempo_source`, `key_source`, `explanation`, `warnings`.

**Schema design notes:**
- Use explicit per-stem gain fields (not generic dict), enum constraints, min/max bounds, rich descriptions
- `label` enum: `"intro" | "verse" | "breakdown" | "drop" | "outro"`
- `transition_in` enum: `"fade" | "crossfade" | "cut"`
- `stem_gains` values: 0.0-2.0 range
- `transition_beats`: 0-16 range

---

## Few-Shot Examples

Include 3 diverse examples in the system prompt (~300 tokens each, ~900 total):

1. **Clear directive:** "Put Song A vocals over Song B beat" — straightforward vocal/instrumental split
2. **Vague prompt:** "mix them" / "make it cool" — demonstrates energy-profile-guided decisions
3. **Contradictory prompt:** "vocals from both songs" — demonstrates graceful warning + best-effort plan

Each example includes full context (condensed metadata) + complete tool_use output.

---

## Post-LLM Validation

### Time Range Validation

- `0 <= start_time_X < end_time_X <= duration_X`, min 5.0s
- Clamp out-of-bounds values
- If clamping by >5s, append to `warnings`. Log original vs clamped.

### Section Validation (10-Point Checklist)

Applied in order:

1. Sections sorted by `start_beat` ascending — re-sort if not
2. No overlaps: `sections[i].end_beat <= sections[i+1].start_beat` — truncate earlier section
3. No gaps > 1 beat — extend earlier section's `end_beat`
4. Minimum section length: `end_beat - start_beat >= 4` — merge tiny sections into adjacent
5. `transition_beats <= (end_beat - start_beat) / 2` — clamp
6. All `stem_gains` keys in `{"vocals", "drums", "bass", "other"}` — add missing (default 0.0), remove unknown
7. All `stem_gains` values in `[0.0, 2.0]` — clamp
8. Total beat range within available audio duration — truncate last section
9. At least 2 sections — if only 1, split into intro (instrumental) + main
10. `end_beat` of last section is multiple of 4 (bar boundary) — extend or truncate by up to 2 beats

**On failure:** Fix automatically (clamp, merge, extend). Log corrections with severity. Error only if 0 sections after merging.

### Semantic Validation

- Verify enum values (`vocal_source`, `tempo_source`, etc.)
- Check `response.stop_reason` — if `max_tokens`, fall back immediately

---

## Deterministic Fallback Algorithm

Fires on every LLM failure. Data-driven, no LLM required.

```python
def generate_fallback_plan(meta_a: AudioMetadata, meta_b: AudioMetadata) -> RemixPlan:
    """Deterministic fallback using analysis data. No LLM required."""
    vocal_src = "song_a" if meta_a.vocal_energy_ratio >= meta_b.vocal_energy_ratio else "song_b"
    vocal_meta = meta_a if vocal_src == "song_a" else meta_b
    inst_meta = meta_b if vocal_src == "song_a" else meta_a

    best_vocal = find_best_region(vocal_meta.energy_regions, prefer_character="rhythmic")
    best_inst = find_best_region(inst_meta.energy_regions, prefer_character="rhythmic")

    def expand_range(region, duration, min_len=90.0):
        """Expand a region to at least min_len seconds, centered on the original."""
        center = (region.start_sec + region.end_sec) / 2
        half = max(min_len, region.end_sec - region.start_sec) / 2
        return max(0, center - half), min(duration, center + half)

    v_start, v_end = expand_range(best_vocal, vocal_meta.duration_seconds)
    i_start, i_end = expand_range(best_inst, inst_meta.duration_seconds)

    tempo_src = "song_a" if meta_a.bpm_confidence >= meta_b.bpm_confidence else "song_b"
    total_beats = int(inst_meta.bpm * 90 / 60)
    sixth, two_thirds = total_beats // 6, total_beats * 2 // 3

    return RemixPlan(
        vocal_source=vocal_src,
        start_time_vocal=v_start, end_time_vocal=v_end,
        start_time_instrumental=i_start, end_time_instrumental=i_end,
        sections=[
            Section("intro", 0, sixth, {"vocals":0,"drums":0.8,"bass":0.8,"other":1}, "fade", 4),
            Section("verse", sixth, two_thirds, {"vocals":1,"drums":0.7,"bass":0.8,"other":0.5}, "crossfade", 4),
            Section("outro", two_thirds, total_beats, {"vocals":0,"drums":0.6,"bass":0.5,"other":0.8}, "crossfade", 8),
        ],
        tempo_source=tempo_src, key_source="none",
        explanation="We created a remix using the strongest sections of each song. "
                    "(Your prompt couldn't be interpreted — try rephrasing.)",
        warnings=["Prompt couldn't be interpreted. Default remix uses most energetic sections."],
        used_fallback=True,
    )
```

---

## Default Arrangement Template

Used when LLM output is completely unrecoverable (0 valid sections after validation).

```python
def default_arrangement(total_beats: int) -> list[Section]:
    """Generate a safe 3-section arrangement when LLM output is unusable."""
    intro_len = min(8, total_beats // 4)
    outro_len = min(8, total_beats // 4)
    main_len = total_beats - intro_len - outro_len
    return [
        Section(label="intro", start_beat=0, end_beat=intro_len,
                stem_gains={"vocals": 0.0, "drums": 0.8, "bass": 0.7, "other": 1.0},
                transition_in="fade", transition_beats=4),
        Section(label="main", start_beat=intro_len, end_beat=intro_len + main_len,
                stem_gains={"vocals": 1.0, "drums": 0.7, "bass": 0.8, "other": 0.5},
                transition_in="crossfade", transition_beats=4),
        Section(label="outro", start_beat=intro_len + main_len, end_beat=total_beats,
                stem_gains={"vocals": 0.0, "drums": 0.6, "bass": 0.5, "other": 0.8},
                transition_in="crossfade", transition_beats=4),
    ]
```

---

## LLM Observability Logging

```python
# Request logging
logger.info("llm_request", session_id=session_id, prompt=prompt,
            song_a_bpm=meta_a.bpm, song_b_bpm=meta_b.bpm,
            song_a_vocal_energy=meta_a.vocal_energy_ratio)

# Response logging
logger.info("llm_response", session_id=session_id, raw_response=raw_json,
            latency_ms=latency, model=response.model,
            used_fallback=False, clamped_fields=[], warnings=plan.warnings)
```

Log `model` field from every response to correlate quality with model version changes.

---

## Error Handling

### Tiered Timeouts

- First attempt: 15s
- Retry: 30s
- Most calls complete in 2-5s

### Retry Policy

- 1-2 retries with backoff on transient errors (429, 500, 529)
- Schema violation: surgical code fix -> re-request -> fallback
- `max_tokens` stop reason: fall back immediately

### Escalation Chain

1. LLM returns valid plan -> use it
2. LLM returns plan with fixable issues -> surgical code fix (clamp, merge, extend sections)
3. Fixed plan still invalid -> re-prompt (1 retry)
4. Re-prompt fails -> deterministic fallback (`generate_fallback_plan`)

---

## Test Prompt Suite

30 prompts across 9 categories. Save as `tests/fixtures/llm_test_prompts.json`.

| Category | Count | Examples |
|----------|-------|---------|
| Clear directives | 5 | "Put Song A vocals over Song B beat" |
| Vague / open-ended | 5 | "mix them", "make it cool" |
| Genre-specific jargon | 4 | "trap beat", "lo-fi chill", "EDM drop" |
| Contradictory / impossible | 3 | "vocals from both", "drums from both" |
| References to inaudible content | 3 | "guitar solo at 2:30", "the quiet part" |
| Extreme tempo/key mismatch | 3 | Prompts paired with 85 BPM + 170 BPM metadata |
| Long / complex prompts | 2 | Multi-sentence detailed instructions |
| Adversarial / nonsense | 2 | "asdfjkl", "make it taste purple" |
| Edge cases | 3 | 5-char prompt, 1000-char prompt, emoji-heavy |

Each test case includes: input prompt, mock song metadata (BPM/key/energy/vocal_energy_ratio), expected `vocal_source`, and qualitative expectations.

### Evaluation Rubric (Sonnet vs Haiku)

| Criterion | Metric | Weight |
|-----------|--------|--------|
| Schema validity | % valid without retries | High |
| Prompt fidelity | 1-5: does plan reflect user intent? | High |
| Musical coherence | 1-5: does arrangement tell a sensible story? | High |
| Metadata utilization | 1-3: uses energy data, respects tempo/key limits | Medium |
| Explanation quality | 1-3: clear, accurate, useful | Medium |
| Latency | Measured (Haiku ~1-2s, Sonnet ~3-5s) | Low |

---

## Cross-References

- `AudioMetadata`, `RemixPlan`, `Section` dataclasses: see `audio-pipeline.md`
- API endpoint that triggers this service: see `api-contract.md` (POST /api/remix)
- Frontend display of explanation/warnings: see `frontend.md` (RemixPlayer)
- Pipeline that calls `interpret_prompt`: see `audio-pipeline.md` (pipeline orchestrator)
