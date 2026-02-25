# Day 3: "Intelligence" -- LLM + React Frontend

> Daily implementation plan for 2026-02-24.
> Canonical source: `docs/plans/2026-02-23-feat-mvp-prompt-based-remix-plan.md`
> Reference docs: `docs/impl/llm-integration.md`, `docs/impl/audio-pipeline.md`, `docs/impl/frontend.md`, `docs/impl/api-contract.md`

## Progress

- [ ] Step 1: Install Backend Dependency
- [ ] Step 2: Build the Interpreter Service (B5)
- [ ] Step 3: Wire LLM into Pipeline
- [ ] Step 4: Test LLM with Real Songs
- [ ] Step 5: Frontend Scaffolding (F1)
- [ ] Step 6: Upload UI (F2)
- [ ] Step 7: Prompt Input + Form Submission (F3)
- [ ] Step 8: SSE Progress Hook + Display (F4)
- [ ] Step 9: Audio Player (F5)
- [ ] Step 10: End-to-End Integration Test

---

## Assumptions (Day 1+2 Complete)

By end of Day 2, the following is working:

- Backend FastAPI server with `POST /api/remix`, SSE progress, status, and audio endpoints
- BS-RoFormer 6-stem separation via Modal (with `htdemucs_ft` local fallback)
- BPM detection + `reconcile_bpm()` with expanded interpretation matrix
- Rubberband tempo matching (single-pass CLI wrapper, tiered limits, instrumental-source-tempo default)
- Sample rate + channel standardization (44.1kHz stereo float32)
- Cross-song level matching with `pyloudnorm` (fixed +3 dB vocal offset, silence guard)
- LUFS normalization on final mix + soft-knee peak limiter
- Fade-in/fade-out + MP3 export via ffmpeg
- **Deterministic fallback plan** (`generate_fallback_plan()`) producing a 5-section arrangement
- **Section-based arrangement renderer** (QR-1 mixer logic): beat-to-sample conversion via beat grid, per-stem gains, transition envelopes (fade, crossfade, cut with micro-crossfade)
- ThreadPoolExecutor + Queue for async processing, SSE streaming
- Basic HTML page for testing (upload, progress, audio playback)

---

## Exit Criteria

Type a creative prompt like "put the rap vocals over the pop beat, drop the drums in the middle" in a real React UI, see LLM-driven remix with custom arrangement play back, with progress updates throughout.

Specifically:
1. LLM interprets the user's prompt and produces a section-based RemixPlan
2. RemixPlan drives the arrangement renderer (sections, per-stem gains, transitions)
3. Deterministic fallback fires gracefully on LLM failure
4. React frontend has: upload zones, prompt input, SSE progress display, audio player with explanation
5. `warnings` and `used_fallback` displayed distinctly in the player

---

## Implementation Order

Day 3 is split into two parallel tracks after a brief sequential setup:

```
Morning (sequential):
  Step 1: Install backend dependency (anthropic SDK)
  Step 2: Build interpreter service (B5)
  Step 3: Wire LLM into pipeline
  Step 4: Test with 5-10 prompts

Afternoon (can overlap with morning LLM testing):
  Step 5: Frontend scaffolding (F1)
  Step 6: Upload UI (F2)
  Step 7: Prompt + submission (F3)

Evening (sequential, needs backend + frontend):
  Step 8: SSE progress hook + display (F4)
  Step 9: Audio player (F5)
  Step 10: End-to-end integration test
```

---

## Step 1: Install Backend Dependency

**Time estimate:** 5 minutes

```bash
cd backend
uv add anthropic
```

Verify the anthropic SDK is importable:
```bash
uv run python -c "import anthropic; print(anthropic.__version__)"
```

Also ensure `.env` has `ANTHROPIC_API_KEY` set (required by `config.py`).

**Files modified:**
- `backend/pyproject.toml` (new dependency)
- `backend/.env` (add `ANTHROPIC_API_KEY=sk-ant-...` if not already present)

---

## Step 2: Build the Interpreter Service (B5)

**Time estimate:** 2-3 hours (system prompt is a first-class deliverable)

### 2a. Create `backend/src/musicmixer/services/interpreter.py`

This is the core of Day 3. The file has four main parts:

1. **System prompt construction** (`_build_system_prompt`)
2. **tool_use schema definition** (`_remix_plan_tool`)
3. **LLM call + response parsing** (`interpret_prompt`)
4. **Deterministic fallback** (`generate_fallback_plan` -- already exists from Day 2, move/import here)

#### System Prompt (~400 tokens of instructions + ~200 templates + ~250 genre + ~200 ambiguity + ~150 mixing/transitions)

Build as a function that accepts song metadata and returns the complete system prompt string:

```python
def _build_system_prompt(
    song_a_meta: AudioMetadata,
    song_b_meta: AudioMetadata,
    key_matching_available: bool,
    key_matching_detail: str,
    total_available_beats: int,
) -> str:
```

The system prompt must contain these sections in order:

**Section 1: Role and MVP Constraints**
```
You are a music remix planner. You decide how to combine two songs into a mashup remix.

CONSTRAINTS:
- Vocals ALWAYS come from one song. All instrumentals (drums, bass, guitar, piano, other) ALWAYS come from the other song.
- You CANNOT mix stems across songs (e.g., no "drums from Song A with bass from Song B").
- You CANNOT add effects, generate new sounds, or use vocals from both songs.
- "other" contains synths, strings, wind instruments, and anything not captured by the 5 named stems.
- If the user asks for something impossible, acknowledge it in the `warnings` field and produce the best plan within these limits.

CAPABILITIES:
- Choose which song provides vocals (vocal_source)
- Select source regions from each song (start/end times in seconds)
- Design a section-based arrangement with per-stem volume control (0.0-1.0 for each of: vocals, drums, bass, guitar, piano, other)
- Choose transitions between sections (fade, crossfade, cut)
- Set tempo and key matching strategy
```

**Section 2: Mixing Philosophy**
```
MIXING PRINCIPLES:
- Contrast creates energy: if a section has drums at 0.0, the next section's drums at 1.0 will feel powerful
- When vocals are active, reduce competing stems (guitar, piano, other) to 0.3-0.5 unless the user asks for a "full" sound
- Every remix should have an energy arc: build, peak, resolve
- Muted stems (0.0) are a tool, not a failure -- silence in the right place is more powerful than sound
- Use the full 0.0-1.0 range. Avoid keeping all stems at 0.5-0.8 throughout -- that produces a flat, unengaging mix
```

**Section 3: Transition Guidance**
```
TRANSITIONS:
- "cut": Use between sections at similar energy levels for a punchy feel. Good for drop-to-verse or chorus-to-chorus.
- "crossfade": Use when energy changes significantly between sections. Default choice for most transitions.
- "fade": Use for the first section (intro) and last section (outro). Also good for bringing vocals in from silence.
```

**Section 4: Arrangement Templates (proportional, injected total_beats)**
```
Your sections must sum to approximately {total_available_beats} beats.

Template A (Standard Mashup): intro(~15%) -> verse(~30%) -> breakdown(~15%) -> drop(~30%) -> outro(~10%)
Template B (DJ Set): build(~25%) -> vocals in(~25%) -> peak(~25%) -> vocals out(~12%) -> outro(~13%)
Template C (Quick Hit): intro(~15%) -> vocal drop(~70%) -> outro(~15%)
Template D (Chill): intro(~25%) -> vocals(~50%) -> outro(~25%)

If total beats < 48, use Template C (Quick Hit). If 48-96, use Standard Mashup. If > 96, you may use DJ Set or add a second verse.
```

**Section 5: Section Rules**
```
SECTION RULES:
- Sections should be 4, 8, or 16 beats long (max 16 beats per section)
- Default: start with instrumental only (establishes the beat before vocals enter), unless the prompt suggests otherwise
- Always end with instrumental only or a fade
- section labels: "intro", "verse", "breakdown", "drop", "outro"
- stem_gains values must be between 0.0 and 1.0 (never exceed 1.0 -- it causes distortion)
- transition_in: "fade", "crossfade", or "cut"
- transition_beats: how many beats the transition lasts (0-8, must be less than half the section length)
```

**Section 6: Genre-Aware Arrangement**
```
GENRE GUIDANCE (infer from BPM + energy profile):
- Hip-hop/rap (80-100 BPM): Keep drums consistent throughout. Build energy through vocal intensity and layering, not drum drops.
- EDM/dance (120-130 BPM): Use breakdown -> build -> drop patterns.
- Pop/rock (100-130 BPM): Use verse-chorus dynamics -- stripped for verses, full for choruses.
- R&B/soul (60-90 BPM): Smooth transitions, no abrupt changes. Layer elements gradually.
```

**Section 7: Tempo and Key Guidance**
```
TEMPO MATCHING:
- tempo_source "average" only when BPMs differ by <15%.
- 15-30% gap: prefer vocal source tempo (the song providing vocals gets stretched less).
- >30% gap: system will stretch vocals only. Note this in your explanation.

KEY MATCHING:
{key_matching_detail}
```

**Section 8: Ambiguity Handling**
```
HANDLING AMBIGUOUS PROMPTS:
- Vague ("make it cool"): Use energy profiles. Pick vocals from the song with more prominent vocals (higher vocal_prominence_db, closer to 0 dB). Use Standard Mashup template.
- Contradictory ("vocals from both"): Acknowledge in warnings. Pick the better vocal source and explain why.
- Genre jargon ("trap", "lo-fi"): Translate to volume/structure decisions. "Trap" = heavy bass, sparse hi-hats. "Lo-fi" = reduce other, gentle, Template D.
- Time references ("guitar solo at 2:30"): Use the time range in source region selection. Add a warning that you can't verify what's there.

DURATION: Target remix duration 60-120 seconds. Minimum 30s, maximum 180s.
```

**Section 9: Stem Artifact Awareness**
```
STEM SEPARATION ARTIFACTS:
Stem separation is imperfect. Vocal stem may contain instrument traces. Instrumental stems may contain ghost vocals.
Bleed is less noticeable during high-energy sections. When the instrumental source song has prominent vocals
(vocal_prominence_db > -6), avoid purely-instrumental sections longer than 8 beats -- ghost vocals bleed through.
```

**Section 10: Explanation and Warnings**
```
EXPLANATION: Write 2-3 non-technical sentences explaining what you did and why. No internal jargon. This is shown directly to the user.

WARNINGS: Populate this array when:
- The prompt is vague and you had to make assumptions
- The prompt asks for something impossible (cross-song stem mixing, effects)
- You're uncertain about a time reference or genre interpretation
- Tempo/key gap is large and the remix may sound noticeably different from the originals
```

**Section 11: Song Metadata (injected at call time)**
```
SONG DATA:

Song A ({song_a_meta.bpm:.1f} BPM, {song_a_meta.key} {song_a_meta.scale}, {song_a_meta.duration_seconds:.0f}s, {total_beats_a} beats, vocal_prominence: {song_a_meta.vocal_prominence_db:.0f} dB):
  [condensed energy profile -- see below]

Song B ({song_b_meta.bpm:.1f} BPM, {song_b_meta.key} {song_b_meta.scale}, {song_b_meta.duration_seconds:.0f}s, {total_beats_b} beats, vocal_prominence: {song_b_meta.vocal_prominence_db:.0f} dB):
  [condensed energy profile -- see below]

1 beat = {60/target_bpm:.2f}s at {target_bpm:.0f} BPM.
```

**Condensed Energy Profile Format** (pre-processed from `AudioMetadata.energy_regions`):

```python
def _condense_energy_profile(meta: AudioMetadata) -> str:
    """Convert energy_regions into compact text for LLM context."""
    # Group regions by character label
    groups: dict[str, list[EnergyRegion]] = {}
    for r in meta.energy_regions:
        groups.setdefault(r.character, []).append(r)

    lines = []
    for character in ["rhythmic", "sustained", "sparse", "moderate"]:
        regions = groups.get(character, [])
        if not regions:
            continue
        label_map = {
            "rhythmic": "Rhythmic (chorus/drop)",
            "sustained": "Sustained (breakdown)",
            "sparse": "Sparse (verse)",
            "moderate": "Moderate (intro/outro)",
        }
        entries = ", ".join(
            f"{r.start_sec:.0f}s-{r.end_sec:.0f}s [{r.relative_energy:.2f}]"
            for r in regions
        )
        lines.append(f"  {label_map[character]}: {entries}")

    # Add temporal structure summary
    structure = " -> ".join(
        f"{r.character}({r.start_sec:.0f}-{r.end_sec:.0f}s)"
        for r in sorted(meta.energy_regions, key=lambda r: r.start_sec)
    )
    lines.append(f"  Structure: {structure}")

    return "\n".join(lines)
```

**Gotcha:** If Day 2 did not implement energy profiles or vocal_prominence_db, the system prompt should degrade gracefully -- omit those sections and use simpler metadata (BPM, key, duration, total_beats only). The few-shot examples should still work without energy data.

#### Few-Shot Examples (~350 tokens each, ~1,050 total)

Include 3 examples directly in the system prompt as `user`/`assistant` message pairs (not embedded in the system string). Each shows a prompt + metadata + complete tool_use output.

**Example 1: Clear directive, matched tempos**
```
User prompt: "Put Song A's vocals over Song B's beat, boost the bass"
Song A: 120 BPM, C major, 240s, 480 beats, vocal_prominence: -3 dB
Song B: 118 BPM, C major, 210s, 413 beats, vocal_prominence: -12 dB
```

Output:
```json
{
  "vocal_source": "song_a",
  "start_time_vocal": 16.0,
  "end_time_vocal": 136.0,
  "start_time_instrumental": 8.0,
  "end_time_instrumental": 128.0,
  "sections": [
    {"label": "intro", "start_beat": 0, "end_beat": 16, "stem_gains": {"vocals": 0.0, "drums": 0.8, "bass": 1.0, "guitar": 0.6, "piano": 0.5, "other": 0.8}, "transition_in": "fade", "transition_beats": 4},
    {"label": "verse", "start_beat": 16, "end_beat": 48, "stem_gains": {"vocals": 1.0, "drums": 0.7, "bass": 1.0, "guitar": 0.4, "piano": 0.3, "other": 0.4}, "transition_in": "crossfade", "transition_beats": 4},
    {"label": "breakdown", "start_beat": 48, "end_beat": 64, "stem_gains": {"vocals": 0.8, "drums": 0.0, "bass": 0.6, "guitar": 0.7, "piano": 0.8, "other": 0.6}, "transition_in": "crossfade", "transition_beats": 4},
    {"label": "drop", "start_beat": 64, "end_beat": 96, "stem_gains": {"vocals": 1.0, "drums": 0.9, "bass": 1.0, "guitar": 0.5, "piano": 0.4, "other": 0.5}, "transition_in": "cut", "transition_beats": 0},
    {"label": "outro", "start_beat": 96, "end_beat": 112, "stem_gains": {"vocals": 0.0, "drums": 0.6, "bass": 0.8, "guitar": 0.5, "piano": 0.5, "other": 0.7}, "transition_in": "crossfade", "transition_beats": 8}
  ],
  "tempo_source": "song_b",
  "key_source": "none",
  "explanation": "I put Song A's vocals over Song B's instrumental with the bass boosted to full. The remix starts with an instrumental intro, builds into the vocal verse, has a drum-free breakdown for contrast, then hits a full drop before fading out.",
  "warnings": []
}
```

**Example 2: Vague prompt, tempo mismatch**
```
User prompt: "mix them together, make it sound good"
Song A: 88 BPM, G minor, 195s, 286 beats, vocal_prominence: -15 dB
Song B: 125 BPM, A minor, 210s, 437 beats, vocal_prominence: -4 dB
```

Output:
```json
{
  "vocal_source": "song_b",
  "start_time_vocal": 30.0,
  "end_time_vocal": 120.0,
  "start_time_instrumental": 10.0,
  "end_time_instrumental": 100.0,
  "sections": [
    {"label": "intro", "start_beat": 0, "end_beat": 16, "stem_gains": {"vocals": 0.0, "drums": 0.7, "bass": 0.8, "guitar": 0.6, "piano": 0.5, "other": 1.0}, "transition_in": "fade", "transition_beats": 4},
    {"label": "verse", "start_beat": 16, "end_beat": 48, "stem_gains": {"vocals": 0.6, "drums": 0.7, "bass": 0.8, "guitar": 0.5, "piano": 0.4, "other": 0.5}, "transition_in": "crossfade", "transition_beats": 4},
    {"label": "drop", "start_beat": 48, "end_beat": 80, "stem_gains": {"vocals": 1.0, "drums": 0.8, "bass": 0.9, "guitar": 0.5, "piano": 0.3, "other": 0.4}, "transition_in": "crossfade", "transition_beats": 2},
    {"label": "outro", "start_beat": 80, "end_beat": 96, "stem_gains": {"vocals": 0.0, "drums": 0.5, "bass": 0.6, "guitar": 0.6, "piano": 0.7, "other": 0.8}, "transition_in": "crossfade", "transition_beats": 8}
  ],
  "tempo_source": "song_a",
  "key_source": "none",
  "explanation": "I chose Song B's vocals since they're more prominent. The vocals are stretched to match Song A's slower tempo, which gives the remix a laid-back feel. The arrangement builds from an instrumental intro into the vocal sections.",
  "warnings": ["These songs have very different tempos, so the vocals have been slowed down to fit the beat. They may sound a bit different from the original."]
}
```

**Example 3: Contradictory prompt, genre clash (demonstrates dramatic gain dynamics)**
```
User prompt: "I want the drums from Song A with the vocals from both songs"
Song A: 140 BPM, F# minor, 180s, 420 beats, vocal_prominence: -10 dB
Song B: 72 BPM, Bb major, 240s, 288 beats, vocal_prominence: -2 dB
```

Output:
```json
{
  "vocal_source": "song_b",
  "start_time_vocal": 20.0,
  "end_time_vocal": 100.0,
  "start_time_instrumental": 0.0,
  "end_time_instrumental": 60.0,
  "sections": [
    {"label": "intro", "start_beat": 0, "end_beat": 8, "stem_gains": {"vocals": 0.0, "drums": 0.9, "bass": 0.7, "guitar": 0.0, "piano": 0.0, "other": 0.5}, "transition_in": "fade", "transition_beats": 4},
    {"label": "verse", "start_beat": 8, "end_beat": 24, "stem_gains": {"vocals": 1.0, "drums": 0.0, "bass": 0.6, "guitar": 0.3, "piano": 0.5, "other": 0.4}, "transition_in": "crossfade", "transition_beats": 4},
    {"label": "drop", "start_beat": 24, "end_beat": 40, "stem_gains": {"vocals": 1.0, "drums": 0.9, "bass": 0.8, "guitar": 0.4, "piano": 0.3, "other": 0.5}, "transition_in": "cut", "transition_beats": 0},
    {"label": "outro", "start_beat": 40, "end_beat": 48, "stem_gains": {"vocals": 0.0, "drums": 0.5, "bass": 0.4, "guitar": 0.3, "piano": 0.4, "other": 0.6}, "transition_in": "crossfade", "transition_beats": 4}
  ],
  "tempo_source": "song_a",
  "key_source": "none",
  "explanation": "I used Song B's vocals since they're more prominent. The vocals are stretched significantly to match Song A's fast tempo, giving a unique energy. I used a Quick Hit template since the tempo difference is extreme.",
  "warnings": ["I can only use vocals from one song at a time -- I chose Song B since its vocals are clearer.", "These songs have extremely different tempos. The vocals have been sped up significantly and may sound different from the original."]
}
```

Note: Example 3 demonstrates drums going from 0.9 (intro) to 0.0 (verse) to 0.9 (drop) -- the dramatic gain dynamics the plan requires to prevent monotonous "all stems at 0.5-0.8" output.

#### tool_use Schema Definition

The Anthropic SDK's `tool_use` feature enforces structured output. Define the tool schema:

```python
REMIX_PLAN_TOOL = {
    "name": "create_remix_plan",
    "description": "Create a structured remix plan based on the user's prompt and song analysis data.",
    "input_schema": {
        "type": "object",
        "required": [
            "vocal_source", "start_time_vocal", "end_time_vocal",
            "start_time_instrumental", "end_time_instrumental",
            "sections", "tempo_source", "key_source",
            "explanation", "warnings"
        ],
        "properties": {
            "vocal_source": {
                "type": "string",
                "enum": ["song_a", "song_b"],
                "description": "Which song provides the vocals. The other song provides ALL instrumentals."
            },
            "start_time_vocal": {
                "type": "number",
                "minimum": 0,
                "description": "Start time (seconds) in the vocal source song. Choose the most interesting/energetic region."
            },
            "end_time_vocal": {
                "type": "number",
                "minimum": 5,
                "description": "End time (seconds) in the vocal source song. Must be > start_time_vocal."
            },
            "start_time_instrumental": {
                "type": "number",
                "minimum": 0,
                "description": "Start time (seconds) in the instrumental source song."
            },
            "end_time_instrumental": {
                "type": "number",
                "minimum": 5,
                "description": "End time (seconds) in the instrumental source song. Must be > start_time_instrumental."
            },
            "sections": {
                "type": "array",
                "minItems": 2,
                "maxItems": 12,
                "description": "Ordered list of remix sections. Must be contiguous (end_beat of one = start_beat of next). First section starts at beat 0.",
                "items": {
                    "type": "object",
                    "required": ["label", "start_beat", "end_beat", "stem_gains", "transition_in", "transition_beats"],
                    "properties": {
                        "label": {
                            "type": "string",
                            "enum": ["intro", "verse", "breakdown", "drop", "outro"],
                            "description": "Section type. Determines the energy character."
                        },
                        "start_beat": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "Starting beat (inclusive). Must be a multiple of 4 for bar alignment."
                        },
                        "end_beat": {
                            "type": "integer",
                            "minimum": 4,
                            "description": "Ending beat (exclusive). Section length = end_beat - start_beat. Should be 4, 8, or 16."
                        },
                        "stem_gains": {
                            "type": "object",
                            "required": ["vocals", "drums", "bass", "guitar", "piano", "other"],
                            "description": "Volume level for each stem in this section. 0.0 = silent, 1.0 = full volume.",
                            "properties": {
                                "vocals": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "drums": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "bass": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "guitar": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "piano": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "other": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                            },
                            "additionalProperties": false
                        },
                        "transition_in": {
                            "type": "string",
                            "enum": ["fade", "crossfade", "cut"],
                            "description": "How this section transitions from the previous one."
                        },
                        "transition_beats": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 8,
                            "description": "Length of transition in beats. Must be < half the section length. Use 0 for 'cut'."
                        }
                    },
                    "additionalProperties": false
                }
            },
            "tempo_source": {
                "type": "string",
                "enum": ["song_a", "song_b", "average"],
                "description": "Which song's tempo to use as target. 'average' only when BPMs differ by <15%."
            },
            "key_source": {
                "type": "string",
                "enum": ["song_a", "song_b", "none"],
                "description": "Which song's key to match to. 'none' to skip key matching."
            },
            "explanation": {
                "type": "string",
                "maxLength": 500,
                "description": "2-3 non-technical sentences explaining what you did and why. Shown directly to the user."
            },
            "warnings": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Caveats: what you couldn't fulfill, prompt ambiguities, quality concerns. Empty array if none."
            }
        },
        "additionalProperties": false
    }
}
```

**Key schema decisions:**
- `stem_gains` has explicit per-stem fields with `additionalProperties: false` (not a generic dict)
- Gains capped at 1.0 (not 2.0) -- the plan explicitly says gains > 1.0 cause limiter distortion
- `sections` has `minItems: 2` and `maxItems: 12` to bound output
- Rich descriptions guide the LLM's output quality
- `additionalProperties: false` at every level prevents extra fields

**4-stem fallback adaptation:** When `settings.stem_backend == 'local'` and 4-stem mode is active, dynamically modify the tool schema to remove `guitar` and `piano` from `stem_gains.required` and `stem_gains.properties`, and update the system prompt to list only 4 stems. Add a helper function `_adapt_schema_for_4stem(tool_schema)`.

#### LLM Call + Response Parsing

```python
import anthropic
import time
import logging

logger = logging.getLogger(__name__)

async def interpret_prompt(
    prompt: str,
    song_a_meta: AudioMetadata,
    song_b_meta: AudioMetadata,
) -> RemixPlan:
    """Convert user prompt + song metadata into a structured remix plan."""
    settings = get_settings()
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    # Pre-compute key matching decision
    key_matching_available, key_matching_detail = _compute_key_guidance(
        song_a_meta, song_b_meta
    )

    # Compute total available beats (from instrumental source, approximated)
    # This is a rough estimate -- the pipeline will validate/clamp later
    target_bpm = max(song_a_meta.bpm, song_b_meta.bpm)  # rough approximation
    total_available_beats = int(target_bpm * 90 / 60)  # ~90 seconds worth

    system_prompt = _build_system_prompt(
        song_a_meta, song_b_meta,
        key_matching_available, key_matching_detail,
        total_available_beats,
    )

    # Build messages: system + few-shot examples + user prompt
    messages = _build_few_shot_messages() + [
        {"role": "user", "content": f"Create a remix plan for this prompt: \"{prompt}\""}
    ]

    # Log request
    logger.info("llm_request", extra={
        "prompt": prompt,
        "song_a_bpm": song_a_meta.bpm,
        "song_b_bpm": song_b_meta.bpm,
    })

    start = time.monotonic()

    try:
        response = client.messages.create(
            model=settings.llm_model,
            max_tokens=2048,
            system=system_prompt,
            messages=messages,
            tools=[REMIX_PLAN_TOOL],
            tool_choice={"type": "tool", "name": "create_remix_plan"},
            timeout=settings.llm_timeout_seconds,
        )
    except anthropic.APIStatusError as e:
        if e.status_code in (429, 500, 529) and settings.llm_max_retries > 0:
            # Retry once with longer timeout
            logger.warning("LLM transient error %d, retrying", e.status_code)
            time.sleep(2)
            try:
                response = client.messages.create(
                    model=settings.llm_model,
                    max_tokens=2048,
                    system=system_prompt,
                    messages=messages,
                    tools=[REMIX_PLAN_TOOL],
                    tool_choice={"type": "tool", "name": "create_remix_plan"},
                    timeout=30,
                )
            except Exception:
                logger.exception("LLM retry failed, using fallback")
                return generate_fallback_plan(song_a_meta, song_b_meta)
        else:
            logger.exception("LLM error, using fallback")
            return generate_fallback_plan(song_a_meta, song_b_meta)
    except Exception:
        logger.exception("LLM error, using fallback")
        return generate_fallback_plan(song_a_meta, song_b_meta)

    latency_ms = (time.monotonic() - start) * 1000

    # Check stop reason
    if response.stop_reason == "max_tokens":
        logger.warning("LLM hit max_tokens, using fallback")
        return generate_fallback_plan(song_a_meta, song_b_meta)

    # Extract tool_use result
    tool_use_block = next(
        (b for b in response.content if b.type == "tool_use"), None
    )
    if tool_use_block is None:
        logger.warning("LLM returned no tool_use block, using fallback")
        return generate_fallback_plan(song_a_meta, song_b_meta)

    raw_plan = tool_use_block.input

    # Log response
    logger.info("llm_response", extra={
        "raw_response": raw_plan,
        "latency_ms": latency_ms,
        "model": response.model,
    })

    # Parse and validate
    try:
        plan = _parse_remix_plan(raw_plan, song_a_meta, song_b_meta)
        plan = _validate_remix_plan(plan, song_a_meta, song_b_meta)
        return plan
    except Exception:
        logger.exception("LLM plan validation failed, using fallback")
        return generate_fallback_plan(song_a_meta, song_b_meta)
```

**Important:** The function uses the synchronous `anthropic.Anthropic` client (not the async one) because it runs inside the pipeline thread, not the async event loop. Do NOT use `await` here.

#### Post-LLM Validation (`_validate_remix_plan`)

Implement the 10-point section validation checklist from the plan:

```python
def _validate_remix_plan(
    plan: RemixPlan,
    song_a_meta: AudioMetadata,
    song_b_meta: AudioMetadata,
) -> RemixPlan:
    """Validate and fix the LLM's remix plan. Fixes issues in-place where possible."""
    clamped_fields = []

    # Time range validation
    vocal_meta = song_a_meta if plan.vocal_source == "song_a" else song_b_meta
    inst_meta = song_b_meta if plan.vocal_source == "song_a" else song_a_meta

    plan.start_time_vocal = max(0, plan.start_time_vocal)
    plan.end_time_vocal = min(vocal_meta.duration_seconds, plan.end_time_vocal)
    if plan.end_time_vocal - plan.start_time_vocal < 5.0:
        plan.end_time_vocal = min(plan.start_time_vocal + 30.0, vocal_meta.duration_seconds)
        clamped_fields.append("vocal_time_range")

    plan.start_time_instrumental = max(0, plan.start_time_instrumental)
    plan.end_time_instrumental = min(inst_meta.duration_seconds, plan.end_time_instrumental)
    if plan.end_time_instrumental - plan.start_time_instrumental < 5.0:
        plan.end_time_instrumental = min(plan.start_time_instrumental + 30.0, inst_meta.duration_seconds)
        clamped_fields.append("instrumental_time_range")

    # Section validation (10-point checklist)
    sections = plan.sections

    # 1. Sort by start_beat
    sections.sort(key=lambda s: s.start_beat)

    # 2. No overlaps
    for i in range(len(sections) - 1):
        if sections[i].end_beat > sections[i + 1].start_beat:
            sections[i].end_beat = sections[i + 1].start_beat
            clamped_fields.append(f"overlap_section_{i}")

    # 3. No gaps > 1 beat
    for i in range(len(sections) - 1):
        gap = sections[i + 1].start_beat - sections[i].end_beat
        if gap > 1:
            sections[i].end_beat = sections[i + 1].start_beat
            clamped_fields.append(f"gap_section_{i}")

    # 4. Minimum section length (4 beats)
    sections = [s for s in sections if s.end_beat - s.start_beat >= 4]
    if not sections:
        # Completely unrecoverable -- use default arrangement
        total_beats = int(inst_meta.bpm * 90 / 60)
        plan.sections = default_arrangement(total_beats)
        plan.used_fallback = True
        plan.warnings.append("Section arrangement was regenerated automatically.")
        return plan

    # 5. transition_beats <= (end_beat - start_beat) / 2
    for s in sections:
        max_transition = (s.end_beat - s.start_beat) // 2
        if s.transition_beats > max_transition:
            s.transition_beats = max_transition
            clamped_fields.append(f"transition_beats_{s.label}")

    # 6. stem_gains keys (add missing, remove unknown)
    valid_stems = {"vocals", "drums", "bass", "guitar", "piano", "other"}
    for s in sections:
        for stem in valid_stems:
            if stem not in s.stem_gains:
                s.stem_gains[stem] = 0.0
        s.stem_gains = {k: v for k, v in s.stem_gains.items() if k in valid_stems}

    # 7. stem_gains values in [0.0, 1.0]
    for s in sections:
        for stem, gain in s.stem_gains.items():
            if gain < 0.0 or gain > 1.0:
                s.stem_gains[stem] = max(0.0, min(1.0, gain))
                clamped_fields.append(f"gain_{s.label}_{stem}")

    # 8. Total beat range within available audio (deferred to pipeline -- needs beat grid)

    # 9. At least 2 sections
    if len(sections) < 2:
        total_beats = sections[0].end_beat
        half = total_beats // 2
        intro = Section("intro", 0, half, {**sections[0].stem_gains, "vocals": 0.0}, "fade", 4)
        main = sections[0]
        main.start_beat = half
        sections = [intro, main]

    # 10. Last section end_beat on bar boundary (multiple of 4)
    last = sections[-1]
    remainder = last.end_beat % 4
    if remainder != 0:
        last.end_beat += (4 - remainder)

    plan.sections = sections

    if clamped_fields:
        logger.info("Section validation clamped fields: %s", clamped_fields)

    return plan
```

#### Duration Validation

After section validation, check total remix duration:

```python
def _validate_duration(plan: RemixPlan, target_bpm: float) -> RemixPlan:
    """Clamp total duration to 30-180 seconds."""
    total_beats = plan.sections[-1].end_beat
    total_seconds = total_beats * 60 / target_bpm

    if total_seconds < 30:
        # Extend last section
        needed_beats = int(30 * target_bpm / 60) - total_beats
        plan.sections[-1].end_beat += max(needed_beats, 8)
        plan.warnings.append("Remix was extended to meet minimum duration.")
    elif total_seconds > 180:
        # Truncate
        max_beats = int(180 * target_bpm / 60)
        plan.sections[-1].end_beat = min(plan.sections[-1].end_beat, max_beats)
        plan.sections = [s for s in plan.sections if s.start_beat < max_beats]
        plan.warnings.append("Remix was shortened to fit maximum duration.")

    return plan
```

**Files created:**
- `backend/src/musicmixer/services/interpreter.py`

**Files modified:**
- `backend/src/musicmixer/models.py` (if `RemixPlan` or `Section` need updates -- likely already defined from Day 2)

### 2b. Key Matching Pre-Computation

The system prompt needs key guidance injected at call time. Build a helper:

```python
def _compute_key_guidance(
    meta_a: AudioMetadata, meta_b: AudioMetadata
) -> tuple[bool, str]:
    """Pre-compute key matching decision for the LLM context.
    Returns (available, detail_string)."""
    min_confidence = min(meta_a.key_confidence, meta_b.key_confidence)

    if meta_a.has_modulation or meta_b.has_modulation:
        return False, "Key matching: unavailable (one or both songs modulate key mid-song)"

    if min_confidence < 0.40:
        return False, f"Key matching: unavailable (low confidence: {min_confidence:.2f})"
    elif min_confidence < 0.55:
        return True, f"Key matching: available with half shift (moderate confidence: {min_confidence:.2f})"
    else:
        return True, f"Key matching: available (confidence: {min_confidence:.2f})"
```

**Gotcha:** If Day 2 did not implement key detection, `meta.key_confidence` may be 0.0 and `meta.key`/`meta.scale` may be placeholder values. The key guidance should degrade to "unavailable" and `key_source` should be `"none"`.

---

## Step 3: Wire LLM into Pipeline

**Time estimate:** 30 minutes

Modify the pipeline orchestrator to call `interpret_prompt` instead of the hardcoded `generate_fallback_plan`.

**File modified:** `backend/src/musicmixer/services/pipeline.py`

The change is surgical -- replace:
```python
# Day 2: hardcoded fallback
remix_plan = generate_fallback_plan(song_a_meta, song_b_meta)
```

With:
```python
# Day 3: LLM interpretation with fallback
from musicmixer.services.interpreter import interpret_prompt

emit_progress(event_queue, {
    "step": "interpreting",
    "detail": "Planning your remix...",
    "progress": 0.58
})

remix_plan = interpret_prompt(prompt, song_a_meta, song_b_meta)

if remix_plan.used_fallback:
    logger.warning("LLM failed, using deterministic fallback", extra={"session_id": session_id})
```

**Important:** The `interpret_prompt` function is synchronous (runs in the pipeline thread). Do NOT wrap it in `await`.

Also update the `complete` event to include `warnings` and `usedFallback`:
```python
emit_progress(event_queue, {
    "step": "complete",
    "detail": "Remix ready!",
    "progress": 1.0,
    "explanation": remix_plan.explanation,
    "warnings": remix_plan.warnings,
    "usedFallback": remix_plan.used_fallback,
})
```

---

## Step 4: Test LLM with Real Songs

**Time estimate:** 30-60 minutes

Before building the frontend, test the LLM integration end-to-end with the existing HTML test page (from Day 1/2).

1. Start the backend: `cd backend && uv run uvicorn musicmixer.main:app --reload`
2. Upload two real songs with different prompts:
   - "Put Song A's vocals over Song B's beat"
   - "Mix them together, make it cool"
   - "Boost the bass, drop the drums in the middle"
   - "Song B vocals over Song A, chill vibe"
   - "Make it sound like a trap remix"
3. Listen to every output. Check:
   - Does the LLM pick a reasonable vocal source?
   - Are the sections musically coherent?
   - Do per-stem gains create dynamics (not all 0.5-0.8)?
   - Does the explanation make sense?
   - Are warnings generated for vague/impossible prompts?

**If the LLM produces bad output:** Tune the system prompt. The system prompt is the most important piece -- iterate on it until the output is consistently good across 5+ prompts. Common issues:
- All stems at similar gains -> Strengthen the mixing philosophy section
- Sections too short/long -> Adjust template proportions
- Wrong vocal source -> Check that vocal_prominence_db is being passed correctly
- Nonsensical explanation -> Add more specific explanation guidance

---

## Step 5: Frontend Scaffolding (F1)

**Time estimate:** 45-60 minutes

### 5a. Initialize the Project

```bash
cd /Users/aaron/Projects/musicMixer
bun create vite frontend --template react-ts
cd frontend
bun install
bun add -d tailwindcss @tailwindcss/vite
```

### 5b. Configure Vite

**File:** `frontend/vite.config.ts`
```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
```

### 5c. Configure Tailwind

**File:** `frontend/src/index.css`
```css
@import "tailwindcss";
```

### 5d. TypeScript Types

**File:** `frontend/src/types/index.ts`

```typescript
// === API Contract Types ===

export type ProgressStep =
  | 'separating'
  | 'analyzing'
  | 'interpreting'
  | 'processing'
  | 'rendering'
  | 'complete'
  | 'error'
  | 'keepalive';

export type ProgressEvent = {
  step: ProgressStep;
  detail: string;
  progress: number;
  explanation?: string;
  warnings?: string[];
  usedFallback?: boolean;
};

export type CreateRemixResponse = {
  session_id: string;
};

export type SessionStatus = {
  status: 'processing' | 'complete' | 'error';
  progress?: number;
  detail?: string;
  explanation?: string;
  warnings?: string[];
  usedFallback?: boolean;
};

// === App State (discriminated union) ===

export type AppState =
  | {
      phase: 'idle';
      songA: File | null;
      songB: File | null;
      prompt: string;
    }
  | {
      phase: 'uploading';
      songA: File;
      songB: File;
      prompt: string;
      uploadProgress: number;
    }
  | {
      phase: 'processing';
      sessionId: string;
      progress: ProgressEvent;
    }
  | {
      phase: 'ready';
      sessionId: string;
      explanation: string;
      warnings: string[];
      usedFallback: boolean;
    }
  | {
      phase: 'error';
      message: string;
      songA: File | null;
      songB: File | null;
      prompt: string;
    };

// === Reducer Actions (discriminated union) ===

export type AppAction =
  | { type: 'SET_SONG_A'; file: File | null }
  | { type: 'SET_SONG_B'; file: File | null }
  | { type: 'SET_PROMPT'; prompt: string }
  | { type: 'START_UPLOAD' }
  | { type: 'UPLOAD_PROGRESS'; percent: number }
  | { type: 'UPLOAD_SUCCESS'; sessionId: string }
  | { type: 'PROGRESS_EVENT'; event: ProgressEvent }
  | { type: 'REMIX_READY'; explanation: string; warnings: string[]; usedFallback: boolean }
  | { type: 'ERROR'; message: string }
  | { type: 'RESET' };

// === API Error Types ===

export type CreateRemixError =
  | { type: 'network' }
  | { type: 'timeout' }
  | { type: 'http'; status: number; body: { detail: string } };
```

### 5e. API Client

**File:** `frontend/src/api/client.ts`

```typescript
import type { CreateRemixResponse, SessionStatus, CreateRemixError } from '../types';

const API_BASE = '/api';

/**
 * Upload two songs and a prompt to create a remix.
 * Uses XMLHttpRequest for upload progress support.
 */
export function createRemix(
  songA: File,
  songB: File,
  prompt: string,
  onUploadProgress?: (pct: number) => void,
): Promise<CreateRemixResponse> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const formData = new FormData();
    formData.append('song_a', songA);
    formData.append('song_b', songB);
    formData.append('prompt', prompt);

    xhr.open('POST', `${API_BASE}/remix`);

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onUploadProgress) {
        onUploadProgress(Math.round((e.loaded / e.total) * 100));
      }
    };

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(JSON.parse(xhr.responseText));
      } else {
        let body = { detail: 'Unknown error' };
        try {
          body = JSON.parse(xhr.responseText);
        } catch { /* use default */ }
        reject({ type: 'http', status: xhr.status, body } as CreateRemixError);
      }
    };

    xhr.onerror = () => reject({ type: 'network' } as CreateRemixError);
    xhr.ontimeout = () => reject({ type: 'timeout' } as CreateRemixError);

    xhr.send(formData);
  });
}

/**
 * Connect to SSE progress stream for a remix session.
 */
export function connectProgress(sessionId: string): EventSource {
  return new EventSource(`${API_BASE}/remix/${sessionId}/progress`);
}

/**
 * Get the current status of a remix session (for reconnection).
 */
export async function getSessionStatus(sessionId: string): Promise<SessionStatus> {
  const res = await fetch(`${API_BASE}/remix/${sessionId}/status`);
  if (res.status === 404) {
    throw new Error('Session not found');
  }
  return res.json();
}

/**
 * Get the audio URL for a completed remix.
 */
export function getAudioUrl(sessionId: string): string {
  return `${API_BASE}/remix/${sessionId}/audio`;
}
```

### 5f. Reducer

**File:** `frontend/src/hooks/useRemixReducer.ts`

```typescript
import type { AppState, AppAction } from '../types';

export const initialState: AppState = {
  phase: 'idle',
  songA: null,
  songB: null,
  prompt: '',
};

export function remixReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case 'SET_SONG_A':
      if (state.phase !== 'idle') return state;
      return { ...state, songA: action.file };

    case 'SET_SONG_B':
      if (state.phase !== 'idle') return state;
      return { ...state, songB: action.file };

    case 'SET_PROMPT':
      if (state.phase !== 'idle') return state;
      return { ...state, prompt: action.prompt };

    case 'START_UPLOAD':
      if (state.phase !== 'idle' || !state.songA || !state.songB) return state;
      return {
        phase: 'uploading',
        songA: state.songA,
        songB: state.songB,
        prompt: state.prompt,
        uploadProgress: 0,
      };

    case 'UPLOAD_PROGRESS':
      if (state.phase !== 'uploading') return state;
      return { ...state, uploadProgress: action.percent };

    case 'UPLOAD_SUCCESS':
      if (state.phase !== 'uploading') return state;
      return {
        phase: 'processing',
        sessionId: action.sessionId,
        progress: { step: 'separating', detail: 'Starting...', progress: 0 },
      };

    case 'PROGRESS_EVENT':
      if (state.phase !== 'processing') return state;
      return { ...state, progress: action.event };

    case 'REMIX_READY':
      if (state.phase !== 'processing') return state;
      return {
        phase: 'ready',
        sessionId: state.sessionId,
        explanation: action.explanation,
        warnings: action.warnings,
        usedFallback: action.usedFallback,
      };

    case 'ERROR':
      if (state.phase === 'idle') {
        return { ...state, phase: 'error' as never }; // shouldn't happen
      }
      return {
        phase: 'error',
        message: action.message,
        songA: state.phase === 'uploading' ? state.songA : null,
        songB: state.phase === 'uploading' ? state.songB : null,
        prompt: state.phase === 'uploading' ? state.prompt : '',
      };

    case 'RESET':
      return initialState;

    default:
      return state;
  }
}
```

### 5g. App Shell

**File:** `frontend/src/App.tsx`
```typescript
import { RemixSession } from './components/RemixSession';

function App() {
  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      <div className="mx-auto max-w-2xl px-4 py-12">
        <header className="mb-10 text-center">
          <h1 className="text-4xl font-bold tracking-tight text-white">musicMixer</h1>
          <p className="mt-3 text-lg text-gray-400">
            Upload two songs. Describe your mashup. AI does the rest.
          </p>
          <p className="mt-1 text-sm text-gray-500">
            musicMixer takes the vocals from one song and layers them over the other song's beat.
          </p>
        </header>
        <RemixSession />
      </div>
    </div>
  );
}

export default App;
```

**File:** `frontend/src/components/RemixSession.tsx` (skeleton -- fleshed out in steps 6-9)
```typescript
import { useReducer } from 'react';
import { remixReducer, initialState } from '../hooks/useRemixReducer';

export function RemixSession() {
  const [state, dispatch] = useReducer(remixReducer, initialState);

  // Render based on phase -- filled in by Steps 6-9
  switch (state.phase) {
    case 'idle':
      return <div>Upload form goes here (Step 6-7)</div>;
    case 'uploading':
      return <div>Uploading... {state.uploadProgress}%</div>;
    case 'processing':
      return <div>Processing... (Step 8)</div>;
    case 'ready':
      return <div>Player goes here (Step 9)</div>;
    case 'error':
      return (
        <div>
          <p>Error: {state.message}</p>
          <button onClick={() => dispatch({ type: 'RESET' })}>Try Again</button>
        </div>
      );
  }
}
```

**Files created:**
- `frontend/vite.config.ts` (modified from scaffold)
- `frontend/src/index.css` (modified)
- `frontend/src/types/index.ts`
- `frontend/src/api/client.ts`
- `frontend/src/hooks/useRemixReducer.ts`
- `frontend/src/App.tsx` (modified)
- `frontend/src/components/RemixSession.tsx`

---

## Step 6: Upload UI (F2)

**Time estimate:** 30-45 minutes

### `frontend/src/components/SongUpload.tsx`

```typescript
import { useRef, useCallback, useState } from 'react';

type Props = {
  label: string;           // "Song A" or "Song B"
  file: File | null;
  onFileChange: (file: File | null) => void;
  disabled?: boolean;
};

const MAX_SIZE_MB = 50;
const ACCEPTED_TYPES = ['.mp3', '.wav'];
const ACCEPTED_MIME = ['audio/mpeg', 'audio/wav', 'audio/x-wav'];

export function SongUpload({ label, file, onFileChange, disabled }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const validateAndSet = useCallback((f: File) => {
    setError(null);
    const ext = f.name.substring(f.name.lastIndexOf('.')).toLowerCase();
    if (!ACCEPTED_TYPES.includes(ext)) {
      setError('Only MP3 and WAV files are supported.');
      return;
    }
    if (f.size > MAX_SIZE_MB * 1024 * 1024) {
      setError(`File too large. Maximum ${MAX_SIZE_MB}MB.`);
      return;
    }
    onFileChange(f);
  }, [onFileChange]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (disabled) return;
    const f = e.dataTransfer.files[0];
    if (f) validateAndSet(f);
  }, [disabled, validateAndSet]);

  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) validateAndSet(f);
  }, [validateAndSet]);

  return (
    <div
      className={`relative rounded-xl border-2 border-dashed p-6 text-center transition-colors ${
        dragOver ? 'border-blue-400 bg-blue-950/30' :
        file ? 'border-green-500/50 bg-green-950/20' :
        'border-gray-700 bg-gray-900/50 hover:border-gray-500'
      } ${disabled ? 'opacity-50 pointer-events-none' : 'cursor-pointer'}`}
      onClick={() => !disabled && inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".mp3,.wav,audio/mpeg,audio/wav"
        className="sr-only"
        onChange={handleChange}
        disabled={disabled}
      />
      <p className="text-sm font-medium text-gray-400 mb-1">{label}</p>
      {file ? (
        <div>
          <p className="text-white font-medium truncate">{file.name}</p>
          <p className="text-xs text-gray-500 mt-1">
            {(file.size / (1024 * 1024)).toFixed(1)} MB
          </p>
          {!disabled && (
            <button
              className="mt-2 text-xs text-gray-500 hover:text-gray-300 underline"
              onClick={(e) => {
                e.stopPropagation();
                onFileChange(null);
                setError(null);
              }}
            >
              Remove
            </button>
          )}
        </div>
      ) : (
        <div>
          <p className="text-gray-500">
            Drop an audio file here or <span className="text-blue-400 underline">browse</span>
          </p>
          <p className="text-xs text-gray-600 mt-1">MP3 or WAV, max 50MB</p>
        </div>
      )}
      {error && (
        <p className="mt-2 text-xs text-red-400">{error}</p>
      )}
    </div>
  );
}
```

**Files created:**
- `frontend/src/components/SongUpload.tsx`

---

## Step 7: Prompt Input + Form Submission (F3)

**Time estimate:** 30-45 minutes

### `frontend/src/components/PromptInput.tsx`

```typescript
type Props = {
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
};

const EXAMPLES = [
  "Put the vocals from Song A over Song B's beat",
  "Song B vocals over Song A instrumentals, boost the bass",
  "Slow it down and layer the singing over the other track",
];

export function PromptInput({ value, onChange, disabled }: Props) {
  return (
    <div>
      <label htmlFor="prompt" className="block text-sm font-medium text-gray-400 mb-2">
        Describe your mashup
      </label>
      <div className="mb-3 space-y-1">
        {EXAMPLES.map((ex) => (
          <p key={ex} className="text-xs text-gray-600 italic">"{ex}"</p>
        ))}
      </div>
      <textarea
        id="prompt"
        rows={3}
        className="w-full rounded-lg border border-gray-700 bg-gray-900 px-4 py-3 text-white placeholder-gray-600 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 resize-none"
        placeholder="What kind of remix do you want?"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        maxLength={1000}
      />
      <p className="mt-1 text-xs text-gray-600 text-right">{value.length}/1000</p>
    </div>
  );
}
```

### `frontend/src/components/RemixForm.tsx`

```typescript
import { SongUpload } from './SongUpload';
import { PromptInput } from './PromptInput';
import type { AppAction } from '../types';

type Props = {
  songA: File | null;
  songB: File | null;
  prompt: string;
  dispatch: React.Dispatch<AppAction>;
  submitting?: boolean;
  uploadProgress?: number;
};

export function RemixForm({ songA, songB, prompt, dispatch, submitting, uploadProgress }: Props) {
  const canSubmit = songA !== null && songB !== null && prompt.trim().length >= 5 && !submitting;

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <SongUpload
          label="Song A"
          file={songA}
          onFileChange={(f) => dispatch({ type: 'SET_SONG_A', file: f })}
          disabled={submitting}
        />
        <SongUpload
          label="Song B"
          file={songB}
          onFileChange={(f) => dispatch({ type: 'SET_SONG_B', file: f })}
          disabled={submitting}
        />
      </div>

      <PromptInput
        value={prompt}
        onChange={(p) => dispatch({ type: 'SET_PROMPT', prompt: p })}
        disabled={submitting}
      />

      {submitting && uploadProgress !== undefined && (
        <div className="space-y-1">
          <div className="h-2 rounded-full bg-gray-800 overflow-hidden">
            <div
              className="h-full rounded-full bg-blue-500 transition-all duration-300"
              style={{ width: `${uploadProgress}%` }}
            />
          </div>
          <p className="text-xs text-gray-500 text-center">Uploading... {uploadProgress}%</p>
        </div>
      )}

      <button
        className={`w-full rounded-lg py-3 px-6 font-semibold text-white transition-colors ${
          canSubmit
            ? 'bg-blue-600 hover:bg-blue-500 cursor-pointer'
            : 'bg-gray-700 cursor-not-allowed text-gray-500'
        }`}
        disabled={!canSubmit}
        onClick={() => dispatch({ type: 'START_UPLOAD' })}
      >
        {submitting ? 'Uploading...' : 'Create Remix'}
      </button>

      {!canSubmit && songA && songB && prompt.trim().length < 5 && (
        <p className="text-xs text-gray-500 text-center">Prompt must be at least 5 characters</p>
      )}
    </div>
  );
}
```

Now update `RemixSession.tsx` to wire the form and handle submission:

### Updated `frontend/src/components/RemixSession.tsx`

```typescript
import { useReducer, useCallback, useEffect } from 'react';
import { remixReducer, initialState } from '../hooks/useRemixReducer';
import { RemixForm } from './RemixForm';
import { ProgressDisplay } from './ProgressDisplay';
import { RemixPlayer } from './RemixPlayer';
import { createRemix } from '../api/client';
import { useRemixProgress } from '../hooks/useRemixProgress';
import type { CreateRemixError } from '../types';

export function RemixSession() {
  const [state, dispatch] = useReducer(remixReducer, initialState);

  // Handle submission
  const handleSubmit = useCallback(async () => {
    if (state.phase !== 'uploading') return;

    try {
      const response = await createRemix(
        state.songA,
        state.songB,
        state.prompt,
        (pct) => dispatch({ type: 'UPLOAD_PROGRESS', percent: pct }),
      );
      dispatch({ type: 'UPLOAD_SUCCESS', sessionId: response.session_id });
    } catch (err) {
      const error = err as CreateRemixError;
      let message: string;
      switch (error.type) {
        case 'network':
          message = 'Upload failed. Check your connection and try again.';
          break;
        case 'timeout':
          message = 'Upload timed out. Please try again.';
          break;
        case 'http':
          if (error.status === 429) {
            message = 'Another remix is being created. Please wait and try again.';
          } else if (error.status === 413) {
            message = 'File too large. Maximum 50MB per song.';
          } else if (error.status === 422) {
            message = error.body.detail || 'Invalid file. Please check your uploads.';
          } else {
            message = 'Something went wrong. Please try again.';
          }
          break;
        default:
          message = 'Something went wrong. Please try again.';
      }
      dispatch({ type: 'ERROR', message });
    }
  }, [state]);

  // Trigger upload when entering uploading phase
  useEffect(() => {
    if (state.phase === 'uploading') {
      handleSubmit();
    }
  }, [state.phase]); // eslint-disable-line react-hooks/exhaustive-deps

  // SSE progress connection (Step 8)
  useRemixProgress(
    state.phase === 'processing' ? state.sessionId : null,
    dispatch,
  );

  switch (state.phase) {
    case 'idle':
      return (
        <RemixForm
          songA={state.songA}
          songB={state.songB}
          prompt={state.prompt}
          dispatch={dispatch}
        />
      );

    case 'uploading':
      return (
        <RemixForm
          songA={state.songA}
          songB={state.songB}
          prompt={state.prompt}
          dispatch={dispatch}
          submitting={true}
          uploadProgress={state.uploadProgress}
        />
      );

    case 'processing':
      return (
        <ProgressDisplay
          progress={state.progress}
          onCancel={() => dispatch({ type: 'RESET' })}
        />
      );

    case 'ready':
      return (
        <RemixPlayer
          sessionId={state.sessionId}
          explanation={state.explanation}
          warnings={state.warnings}
          usedFallback={state.usedFallback}
          onNewRemix={() => dispatch({ type: 'RESET' })}
        />
      );

    case 'error':
      return (
        <div className="text-center space-y-4">
          <div className="rounded-lg border border-red-800/50 bg-red-950/30 p-6">
            <p className="text-red-300">{state.message}</p>
          </div>
          <button
            className="rounded-lg bg-gray-700 px-6 py-2 text-sm text-gray-300 hover:bg-gray-600"
            onClick={() => dispatch({ type: 'RESET' })}
          >
            Try Again
          </button>
        </div>
      );
  }
}
```

**Files created:**
- `frontend/src/components/PromptInput.tsx`
- `frontend/src/components/RemixForm.tsx`

**Files modified:**
- `frontend/src/components/RemixSession.tsx`

---

## Step 8: SSE Progress Hook + Display (F4)

**Time estimate:** 45-60 minutes

### `frontend/src/hooks/useRemixProgress.ts`

```typescript
import { useEffect, useRef } from 'react';
import { connectProgress, getSessionStatus } from '../api/client';
import type { AppAction, ProgressEvent } from '../types';

const TIMEOUT_BY_STEP: Record<string, number> = {
  separating: 180_000,  // 3 minutes (cold start + separation)
  analyzing: 60_000,
  interpreting: 60_000,
  processing: 120_000,
  rendering: 120_000,
};
const DEFAULT_TIMEOUT = 120_000;

export function useRemixProgress(
  sessionId: string | null,
  dispatch: React.Dispatch<AppAction>,
) {
  const eventSourceRef = useRef<EventSource | null>(null);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const errorCountRef = useRef(0);
  const currentStepRef = useRef<string>('separating');

  useEffect(() => {
    if (!sessionId) return;

    const resetTimeout = () => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
      const ms = TIMEOUT_BY_STEP[currentStepRef.current] ?? DEFAULT_TIMEOUT;
      timeoutRef.current = setTimeout(() => {
        dispatch({ type: 'ERROR', message: 'Processing is taking longer than expected. Please try again.' });
        eventSourceRef.current?.close();
      }, ms);
    };

    const es = connectProgress(sessionId);
    eventSourceRef.current = es;
    errorCountRef.current = 0;

    es.onmessage = (event) => {
      errorCountRef.current = 0; // Reset on any successful message
      try {
        const data: ProgressEvent = JSON.parse(event.data);

        // Keepalive -- reset timeout but don't update UI
        if (data.step === 'keepalive') {
          resetTimeout();
          return;
        }

        currentStepRef.current = data.step;
        resetTimeout();

        if (data.step === 'complete') {
          dispatch({
            type: 'REMIX_READY',
            explanation: data.explanation ?? '',
            warnings: data.warnings ?? [],
            usedFallback: data.usedFallback ?? false,
          });
          es.close();
          return;
        }

        if (data.step === 'error') {
          dispatch({ type: 'ERROR', message: data.detail || 'Something went wrong.' });
          es.close();
          return;
        }

        dispatch({ type: 'PROGRESS_EVENT', event: data });
      } catch {
        // Malformed event -- log and ignore
        console.warn('Malformed SSE event:', event.data);
      }
    };

    es.onerror = () => {
      errorCountRef.current++;
      if (errorCountRef.current >= 5) {
        dispatch({
          type: 'ERROR',
          message: 'Lost connection to the server. Please try again.',
        });
        es.close();
      }
      // Otherwise let EventSource auto-reconnect
    };

    resetTimeout();

    return () => {
      es.close();
      eventSourceRef.current = null;
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
    };
  }, [sessionId, dispatch]);
}
```

### `frontend/src/components/ProgressDisplay.tsx`

```typescript
import type { ProgressEvent } from '../types';

type Props = {
  progress: ProgressEvent;
  onCancel: () => void;
};

const STEP_LABELS: Record<string, string> = {
  separating: 'Extracting stems',
  analyzing: 'Analyzing audio',
  interpreting: 'Planning your remix',
  processing: 'Processing audio',
  rendering: 'Building your remix',
};

export function ProgressDisplay({ progress, onCancel }: Props) {
  const pct = Math.max(0, Math.min(100, Math.round(progress.progress * 100)));
  const stepLabel = STEP_LABELS[progress.step] ?? progress.step;

  return (
    <div className="space-y-6 text-center">
      <div>
        <p className="text-lg font-medium text-white">{stepLabel}</p>
        <p className="mt-1 text-sm text-gray-400">{progress.detail}</p>
      </div>

      <div className="space-y-2">
        <div className="h-3 rounded-full bg-gray-800 overflow-hidden">
          <div
            className="h-full rounded-full bg-blue-500 transition-all duration-500 ease-out"
            style={{ width: `${pct}%` }}
          />
        </div>
        <p className="text-xs text-gray-500">{pct}%</p>
      </div>

      <button
        className="text-sm text-gray-500 hover:text-gray-300 underline"
        onClick={onCancel}
      >
        Cancel
      </button>
    </div>
  );
}
```

**Files created:**
- `frontend/src/hooks/useRemixProgress.ts`
- `frontend/src/components/ProgressDisplay.tsx`

---

## Step 9: Audio Player (F5)

**Time estimate:** 30 minutes

### `frontend/src/components/RemixPlayer.tsx`

```typescript
import { useState } from 'react';
import { getAudioUrl } from '../api/client';

type Props = {
  sessionId: string;
  explanation: string;
  warnings: string[];
  usedFallback: boolean;
  onNewRemix: () => void;
};

export function RemixPlayer({ sessionId, explanation, warnings, usedFallback, onNewRemix }: Props) {
  const [confirmNew, setConfirmNew] = useState(false);
  const audioUrl = getAudioUrl(sessionId);

  return (
    <div className="space-y-6">
      {/* Audio player */}
      <div className="rounded-xl border border-gray-700 bg-gray-900/50 p-6">
        <h2 className="text-lg font-semibold text-white mb-4">Your Remix</h2>
        <audio
          controls
          autoPlay
          className="w-full"
          src={audioUrl}
          onError={() => {
            // Remix expired or not found -- handled by a future enhancement
          }}
        >
          Your browser does not support the audio element.
        </audio>
      </div>

      {/* Explanation */}
      <div
        className={`rounded-lg border p-4 ${
          usedFallback
            ? 'border-amber-700/50 bg-amber-950/20'
            : 'border-gray-700 bg-gray-900/30'
        }`}
      >
        {usedFallback && (
          <p className="text-xs text-amber-400 mb-2 font-medium">
            Generated automatically (your prompt could not be fully interpreted)
          </p>
        )}
        <p className="text-sm text-gray-300">{explanation}</p>
      </div>

      {/* Warnings */}
      {warnings.length > 0 && (
        <div className="space-y-2">
          {warnings.map((w, i) => (
            <div
              key={i}
              className="rounded-lg border border-amber-800/40 bg-amber-950/20 px-4 py-3"
            >
              <p className="text-xs text-amber-300">{w}</p>
            </div>
          ))}
        </div>
      )}

      {/* Expiration notice */}
      <p className="text-xs text-gray-600 text-center">
        This remix will expire in approximately 3 hours.
      </p>

      {/* New remix button */}
      {confirmNew ? (
        <div className="text-center space-y-2">
          <p className="text-sm text-gray-400">
            Creating a new remix will replace this one. Continue?
          </p>
          <div className="flex justify-center gap-3">
            <button
              className="rounded-lg bg-blue-600 px-4 py-2 text-sm text-white hover:bg-blue-500"
              onClick={onNewRemix}
            >
              Yes, create new
            </button>
            <button
              className="rounded-lg bg-gray-700 px-4 py-2 text-sm text-gray-300 hover:bg-gray-600"
              onClick={() => setConfirmNew(false)}
            >
              Keep listening
            </button>
          </div>
        </div>
      ) : (
        <button
          className="w-full rounded-lg border border-gray-700 py-3 text-sm text-gray-400 hover:border-gray-500 hover:text-gray-300"
          onClick={() => setConfirmNew(true)}
        >
          Create New Remix
        </button>
      )}
    </div>
  );
}
```

**Files created:**
- `frontend/src/components/RemixPlayer.tsx`

---

## Step 10: End-to-End Integration Test

**Time estimate:** 30-60 minutes

### Manual Test Checklist

1. Start the backend: `cd backend && uv run uvicorn musicmixer.main:app --reload --port 8000`
2. Start the frontend: `cd frontend && bun run dev`
3. Open `http://localhost:5173` in a browser

**Test scenarios:**

| # | Test | Expected |
|---|------|----------|
| 1 | Upload two MP3s, type "put Song A vocals over Song B beat", click Create Remix | Upload progress bar, then SSE progress through separating -> analyzing -> interpreting -> processing -> rendering -> complete. Audio plays. Explanation shown. |
| 2 | Try a vague prompt: "make it sound cool" | Remix completes. Explanation acknowledges creative choice. May include warnings. |
| 3 | Try a contradictory prompt: "vocals from both songs" | Remix completes. Warnings displayed about the limitation. |
| 4 | Cancel mid-processing | Resets to idle. Can upload again. |
| 5 | Upload an invalid file (e.g., a .txt renamed to .mp3) | Client-side rejection (wrong extension) or server 422 error displayed. |
| 6 | Try to submit with empty prompt | Button disabled. "Prompt must be at least 5 characters" shown. |
| 7 | Listen to the audio output | Does the arrangement have dynamics? Are stems at different levels across sections? Does it sound like a remix (not two songs playing at once)? |

---

## B4 Completion: Key Detection, Energy Profiles, Groove Detection

These features feed into the LLM context and improve prompt interpretation quality. They are listed as Day 3 tasks in the plan but are **deferred to Day 4 per the "What's Deferred to Post-Demo" table** in the recommended implementation order. The Day 3 implementation should degrade gracefully without them:

| Feature | Status for Day 3 | Where it matters |
|---------|------------------|------------------|
| Key detection | If not yet implemented, `key_confidence` = 0.0, `key`/`scale` = "unknown". Key guidance says "unavailable". `key_source` = `"none"`. | System prompt, key matching in processor |
| Energy profiles | If not yet implemented, `energy_regions` = `[]`. System prompt omits energy data. LLM uses BPM/key/duration only. | System prompt condensed format |
| Groove detection | If not yet implemented, `groove_type` = `"unknown"`. System prompt omits groove data. | System prompt, compatibility |
| Vocal prominence | If not yet implemented, `vocal_prominence_db` = `0.0`. LLM uses default heuristic. | System prompt, fallback vocal_source decision |

**Recommendation:** If time permits on Day 3 evening, add these in priority order:
1. **Vocal prominence** (highest impact -- drives vocal_source decisions): ~20 lines in analysis.py after stem separation
2. **Energy profiles** (second highest -- enables better section selection): `compute_energy_profile()` function is ~30 lines
3. **Key detection** (third -- enables key matching on Day 4): Essentia `KeyExtractor` with librosa fallback
4. **Groove detection** (lowest priority): Can wait for Day 4 or post-demo

---

## B4b: Compatibility Analysis Endpoint

Per the "What's Deferred to Post-Demo" table, the `POST /api/analyze` endpoint is **deferred**. It is a nice-to-have UX feature (pre-submission compatibility signal) that is not needed to produce a remix. Skip for Day 3.

---

## Complete File Inventory

### Files Created (Backend)

| File | Description |
|------|-------------|
| `backend/src/musicmixer/services/interpreter.py` | LLM prompt interpretation: system prompt, tool_use schema, validation, fallback |

### Files Modified (Backend)

| File | Change |
|------|--------|
| `backend/pyproject.toml` | Add `anthropic` dependency |
| `backend/src/musicmixer/services/pipeline.py` | Replace hardcoded fallback with `interpret_prompt()` call |
| `backend/src/musicmixer/models.py` | Update `RemixPlan` if needed (add `warnings`, `used_fallback` fields if not present) |

### Files Created (Frontend)

| File | Description |
|------|-------------|
| `frontend/src/types/index.ts` | TypeScript types: AppState, AppAction, ProgressEvent, etc. |
| `frontend/src/api/client.ts` | API functions: createRemix, connectProgress, getSessionStatus, getAudioUrl |
| `frontend/src/hooks/useRemixReducer.ts` | State machine reducer |
| `frontend/src/hooks/useRemixProgress.ts` | SSE progress hook |
| `frontend/src/components/RemixSession.tsx` | State machine orchestrator |
| `frontend/src/components/RemixForm.tsx` | Composed upload + prompt + submit |
| `frontend/src/components/SongUpload.tsx` | Drag-and-drop file upload zone |
| `frontend/src/components/PromptInput.tsx` | Textarea + example prompts |
| `frontend/src/components/ProgressDisplay.tsx` | Progress bar + step description + cancel |
| `frontend/src/components/RemixPlayer.tsx` | Audio player + explanation + warnings + new remix |

### Files Modified (Frontend)

| File | Change |
|------|--------|
| `frontend/vite.config.ts` | Add Tailwind plugin + API proxy |
| `frontend/src/index.css` | Tailwind import |
| `frontend/src/App.tsx` | App shell with header + RemixSession |

---

## Dependencies to Install

### Backend
```bash
cd backend
uv add anthropic
```

### Frontend
```bash
cd frontend
bun create vite . --template react-ts  # (if not already scaffolded)
bun install
bun add -d tailwindcss @tailwindcss/vite
```

No additional frontend runtime dependencies. The app uses:
- Native `EventSource` API (no library)
- Native `XMLHttpRequest` (no axios)
- Native `<audio>` element (no Tone.js)
- `useReducer` (no Redux)

---

## Frontend State Machine Reference

```
     +---------+
     |  IDLE   |<----------------------------+
     +----+----+                              |
          |                                   |
   START_UPLOAD                              RESET
          |                                   |
     +----v------+                            |
     | UPLOADING |--ERROR---> +-------+       |
     +----+------+            | ERROR |-------+
          |                   +---+---+
   UPLOAD_SUCCESS                 ^
          |                       |
     +----v-------+               |
     | PROCESSING |--ERROR--------+
     +----+-------+
          |
    REMIX_READY
          |
     +----v----+
     |  READY  |------RESET------> IDLE
     +---------+
```

**State transitions:**
- `IDLE` -> `UPLOADING`: `START_UPLOAD` (when both songs + prompt are present)
- `UPLOADING` -> `PROCESSING`: `UPLOAD_SUCCESS` (server returns session_id)
- `UPLOADING` -> `ERROR`: `ERROR` (network, 413, 422, 429, 500)
- `PROCESSING` -> `READY`: `REMIX_READY` (SSE complete event)
- `PROCESSING` -> `ERROR`: `ERROR` (SSE error event, timeout, connection loss)
- `ERROR` -> `IDLE`: `RESET`
- `READY` -> `IDLE`: `RESET` (user clicks "Create New Remix")

**Data preservation:**
- `ERROR` preserves `songA`, `songB`, `prompt` from the `UPLOADING` phase (user can retry without re-selecting files)
- `PROCESSING` stores `sessionId` for SSE connection
- `READY` stores `sessionId` (for audio URL), `explanation`, `warnings`, `usedFallback`

---

## Risk Items

### 1. LLM Response Quality

**Risk:** Claude Sonnet may produce musically incoherent arrangements or schema violations.

**Mitigations:**
- tool_use guarantees schema compliance at the JSON level
- 10-point section validation catches structural issues (gaps, overlaps, out-of-range values)
- Deterministic fallback fires on any unrecoverable failure
- 3 few-shot examples demonstrate the expected output format
- System prompt constrains creative space with arrangement templates

**Monitoring:** Log every raw LLM response. Compare `used_fallback` rate across sessions. If fallback fires >20% of the time, the system prompt needs tuning.

### 2. Token Budget

**Risk:** System prompt + few-shot + song metadata exceeds the ~3,500-4,000 token budget, increasing latency and cost.

**Mitigations:**
- Condensed energy profile format (~10x fewer tokens than raw regions)
- Proportional arrangement templates (not fixed beat counts)
- Tool schema descriptions are concise but specific
- If budget is tight: reduce few-shot examples from 3 to 2, or shorten the mixing philosophy section

**Verification:** After building the system prompt, measure actual token count with `anthropic.count_tokens()` or estimate at ~4 chars/token. Adjust if over budget.

### 3. LLM Latency

**Risk:** Sonnet takes 3-5 seconds per call, adding to total processing time.

**Mitigations:**
- LLM call is ~5% of total processing time (separation is 50-60%)
- Progress UI shows "Planning your remix..." during the LLM call
- If latency is a concern, `settings.llm_model` can be changed to Haiku (1-2 seconds) without code changes

### 4. Fallback Behavior

**Risk:** The deterministic fallback produces a "correct but boring" remix with no prompt awareness.

**Mitigations:**
- Fallback uses energy profiles to select the best source regions
- 5-section arrangement with dynamics (drums at 0.0 in breakdown, full in drop)
- `used_fallback = True` + honest explanation tells the user what happened
- Amber styling in the player distinguishes fallback from LLM-driven output

### 5. Frontend-Backend Integration

**Risk:** SSE events don't match TypeScript types, causing runtime errors.

**Mitigations:**
- `ProgressEvent` type defined in both backend (`models.py`) and frontend (`types/index.ts`)
- Malformed SSE events are caught and logged (not crash)
- `keepalive` events have `progress: -1` sentinel value
- Error count threshold (5 consecutive errors) prevents infinite reconnection

### 6. Energy Profile / Vocal Prominence Unavailable

**Risk:** If Day 2 did not implement these analysis features, the LLM has less context for decisions.

**Mitigations:**
- System prompt degrades gracefully: omits energy/vocal sections, uses BPM/key/duration only
- Fallback plan uses simpler heuristics (BPM confidence for tempo source, arbitrary vocal source)
- The LLM is explicitly instructed to handle missing data ("If vocal_prominence is not available, default to Song A as vocal source")

---

## Day 3 Schedule Summary

| Block | Time | Steps | Output |
|-------|------|-------|--------|
| Morning | 2-3h | Steps 1-4 | LLM interpreter built, wired into pipeline, tested with real songs |
| Afternoon | 1.5-2h | Steps 5-7 | React app scaffolded with upload UI + prompt input + submission flow |
| Evening | 1.5-2h | Steps 8-10 | SSE progress display + audio player + end-to-end integration working |

**Total estimated time:** 5-7 hours of focused implementation.

**Critical path:** Step 2 (interpreter service) is the most complex and highest-risk. Allocate extra time if the system prompt needs iteration. The frontend steps (5-9) are straightforward React/TypeScript work that can proceed quickly once the backend is stable.
