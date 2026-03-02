# Plan: Lyrics Lookup for LLM Arrangement Intelligence

## Context

The LLM interpreter (Claude Sonnet) currently receives a 4-layer system prompt with song analysis data (BPM, sections, stem energy, cross-song relationships) but has **no knowledge of what the lyrics say**. This means it can't avoid cutting mid-sentence, identify hooks/choruses by content, or match lyrics thematically to the user's prompt.

**Solution:** Fetch known-correct lyrics from free online databases (LRCLIB, Musixmatch), map them to bar numbers, and inject as Layer 5 in the system prompt. No Whisper, no ML models, no GPU, no frontend work.

**Why lookup over transcription:** Whisper has 30-50% WER on singing. Fetching known lyrics gives ~100% accuracy for popular songs at a fraction of the implementation cost. Graceful degradation: if no lyrics found, the LLM works exactly as it does today.

**Purpose:** Feed lyrics to the LLM so it can make smarter arrangement decisions — avoid cutting mid-phrase, identify hooks, match themes to user prompt. NOT for user display.

## Files to Modify

| File | Change |
|------|--------|
| `backend/src/musicmixer/models.py` | Add `LyricLine`, `LyricsData` dataclasses |
| `backend/src/musicmixer/services/lyrics.py` | **New file.** Filename parsing, ID3 reading, syncedlyrics lookup, LRC parsing, bar mapping |
| `backend/src/musicmixer/config.py` | Add `lyrics_lookup_enabled: bool = True` |
| `backend/src/musicmixer/api/remix.py` | Capture original filenames from `UploadFile.filename`, pass through `_pipeline_wrapper` to `run_pipeline` |
| `backend/src/musicmixer/services/pipeline.py` | Accept original filenames, launch parallel lyrics fetch in Step 1, map to bars after Step 3.5, pass to `interpret_prompt` |
| `backend/src/musicmixer/services/interpreter.py` | Accept `lyrics_a`/`lyrics_b` params, build Layer 5 text, add lyric-aware mixing rule (#8), update few-shot examples |
| `backend/pyproject.toml` | Add `syncedlyrics`, `mutagen` |
| `backend/tests/test_lyrics.py` | **New file.** Unit tests for parsing, bar mapping, Layer 5 format |

## Implementation Steps

### Step 1: Add dependencies (~5 min)
```bash
cd backend && uv add syncedlyrics mutagen
```

### Step 2: Add data models to `models.py` (~10 min)

```python
@dataclass
class LyricLine:
    text: str
    timestamp_seconds: float | None = None
    bar_number: int | None = None

@dataclass
class LyricsData:
    artist: str
    title: str
    source: str          # "lrclib" | "musixmatch" | "filename" | "id3"
    is_synced: bool
    lines: list[LyricLine]
    raw_text: str
    lookup_duration_ms: float = 0.0
```

### Step 3: Create `services/lyrics.py` (~1.5 hrs)

All lyrics logic in one new file:

**3a. Filename parsing** — regex for `"Artist - Title (Official Audio).mp3"` pattern. Strip common suffixes `(Official Audio)`, `[Remaster]`, `[HD]`, etc. Split on ` - ` for artist/title.

**3b. ID3 tag reading** — `mutagen.easyid3.EasyID3` to extract artist/title. Fallback if tags missing.

**3c. Identity resolver** — try filename first (more reliable for YouTube downloads), then ID3, then use cleaned filename stem as search query.

**3d. Lyrics fetch** — single call: `syncedlyrics.search(query)`. The library handles provider ordering and sync-vs-plain fallback internally. Detect sync vs plain by checking the returned string for `[mm:ss` timestamp patterns. Handle `None` return (no lyrics found). Handle import errors, timeouts, exceptions gracefully.

**3e. LRC parser** — use permissive regex `\[(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?\]` to handle: no fractional seconds, 1-3 digit fractional parts, single-digit minutes. Filter metadata lines (`[ar:]`, `[ti:]`, `[al:]`, `[offset:]`). Apply `[offset:]` value (milliseconds) to all timestamps before bar mapping. Handle multiple timestamps per line. Parse into `LyricLine` objects with `timestamp_seconds`.

**3f. Bar mapping** — derive bar boundary times directly from `AudioMetadata.beat_frames` in the lyrics service: use `beat_frames[::4]` (every 4th beat = bar start) converted to seconds via `librosa.frames_to_time()`. This avoids model changes since `beat_frames` is already stored on `AudioMetadata`. Use `np.searchsorted()` against these derived bar boundary timestamps, ensuring lyrics bar numbers match the section map exactly. Fallback to `bar_number = floor(timestamp_seconds / ((60/bpm) * 4))` if `beat_frames` aren't available. Function signature: `map_lyrics_to_bars(lyrics, beat_frames: np.ndarray | None, bpm: float, sr: int = 22050)`. For plain lyrics without timestamps: distribute lines across vocal-active bars proportionally using `vocal_active` array.

**3g. Top-level function** — `lookup_lyrics_for_song(audio_path, original_filename) -> LyricsData | None`. Parameter roles: `audio_path` (Path) = actual file on disk (e.g., `song_a.mp3`), used for mutagen ID3 tag extraction. `original_filename` (str) = the original upload filename string (e.g., `"The Notorious B.I.G. - Hypnotize (Official Audio).mp3"`), used for regex-based artist/title parsing. The saved file on disk is named `song_a.mp3` which is useless for parsing.

### Step 4: Config flag in `config.py` (~2 min)

```python
lyrics_lookup_enabled: bool = True
```

### Step 5: Preserve original filenames in `remix.py` (~15 min)

The upload endpoint saves files as `song_a.mp3`/`song_b.mp3`, losing the original name. Fix:

- Capture `song_a.filename` and `song_b.filename` before saving
- Pass as new params to `_pipeline_wrapper` → `run_pipeline`

Key lines to change:
- `remix.py:34-41` — add params to `_pipeline_wrapper`
- `remix.py:136-144` — pass original filenames in `executor.submit` call

### Step 6: Pipeline integration in `pipeline.py` (~30 min)

**6a. Add params** to `run_pipeline()` signature: `song_a_original_filename: str = ""`, `song_b_original_filename: str = ""`

**6b. Parallel lyrics fetch** — use a SEPARATE `ThreadPoolExecutor(max_workers=2)` for lyrics, submitted BEFORE the stem separation `with` block. Lyrics only need the filename, not stems, so they start immediately. Collect results AFTER the stem separation `with` block exits. This ensures a hung lyrics lookup (15s timeout) never blocks stem separation completion. 15-second timeout per song; any failure logged and skipped.

**6c. Bar mapping** — after Step 3.5 (song structure analysis), call `map_lyrics_to_bars(lyrics, beat_frames=meta_a.beat_frames, bpm=meta_a.bpm)` and `map_plain_lyrics_to_sections(lyrics, vocal_active, total_bars)` for each song. `meta_a.beat_frames` and `meta_a.duration_seconds` are already available from the analysis step; no model changes needed.

**6d. Pass to interpreter** — change `interpret_prompt(prompt, meta_a, meta_b)` call (line 251) to also pass `lyrics_a=lyrics_a_data, lyrics_b=lyrics_b_data`.

### Step 7: Interpreter Layer 5 in `interpreter.py` (~45 min)

**Note:** Layer 5 follows the same dual-injection pattern as Layers 1-4: injected into `_build_system_prompt()` for real song data (Step 7c), and hardcoded into few-shot user message strings (Step 7e). This ensures the LLM sees lyrics in both training examples and real prompts.

**7a. Update signatures** — `interpret_prompt()` and `_build_system_prompt()` accept optional `lyrics_a: LyricsData | None = None`, `lyrics_b: LyricsData | None = None`. Also update the call site: in `interpret_prompt()` (line ~1234), forward `lyrics_a=lyrics_a, lyrics_b=lyrics_b` in the `_build_system_prompt(...)` call. Full call chain: `remix.py (create_remix) -> _pipeline_wrapper -> run_pipeline -> interpret_prompt -> _build_system_prompt -> _build_lyrics_layer`.

**7b. Build Layer 5 text** — new `_build_lyrics_layer()` function. Format for synced lyrics:
```
=== LAYER 5: LYRICS ===
Use these lyrics to avoid cutting mid-phrase, identify hooks, and match themes to the prompt.

Song A lyrics (synced, 47 lines):
  bar   2: Biggie Biggie Biggie, can't you see
  bar   3: Sometimes your words just hypnotize me
  bar   5: And I just love your flashy ways
  ...
Song B lyrics: no lyrics found.
```

**7c. Inject after Layer 4** — at line ~381, after `song_data_parts.append(cross_song)`, append Layer 5 to `song_data_parts` (NOT to `sections`) if lyrics exist.

**7d. Add mixing rule** — add rule #8 to the "CRITICAL MIXING RULES" section:
```
8. LYRIC-AWARE CUTS: When lyrics are available, prefer placing section boundaries at natural lyric breaks (end of line/verse). Cross-reference Layer 5 bar numbers with Layer 2 section boundaries. If lyrics show a hook or repeated phrase, that's a prime candidate for the "drop" section.
```

**7e. Update few-shot examples** — update at least ONE 6-stem and ONE 4-stem few-shot example in `_build_few_shot_messages()` / `_build_few_shot_messages_4stem()` to include Layer 5 lyrics data. The assistant response in those examples MUST reference lyrics in the `explanation` field (e.g., `"Starting with the hook from Track One for immediate impact"`). Additionally, at least one few-shot example's `sections` array must demonstrate a lyric-informed boundary: a section break aligned with a lyric line ending, where `start_time_vocal`/`end_time_vocal` encompass the bar range shown in Layer 5. Use fictional lyrics following the existing "Track One"/"Slow Jam" naming pattern (never real copyrighted lyrics). One example per mode can omit Layer 5 to demonstrate graceful degradation (showing the LLM that lyrics aren't always present).

### Step 8: Tests in `tests/test_lyrics.py` (~45 min)

- Filename parsing: `"Artist - Title (Official Audio).mp3"` → `("Artist", "Title")`
- Filename edge cases: no dash, multiple dashes, unicode, empty string
- LRC parsing: standard `[mm:ss.cc]` format, millisecond variant, empty lines filtered, metadata lines filtered, offset applied
- Bar mapping: known beat_frames + timestamp → expected bar number; BPM fallback
- Plain lyrics section distribution
- Layer 5 format output for synced/plain/no lyrics
- Mocked `syncedlyrics.search` integration test

## Pipeline Timing Impact

| Operation | Time | Parallel? |
|-----------|------|-----------|
| Lyrics lookup (network) | 2-8 sec | Yes, parallel with stem separation |
| Bar mapping | <1 ms | Sequential, after BPM analysis |
| Layer 5 formatting | <1 ms | Part of prompt building |
| **Net latency added** | **~0 sec** | Fully hidden behind stem separation |

## Token Budget Impact

Layer 5 adds ~200-2500 tokens depending on song length and lyric density (rap songs are at the high end). Current 4-layer prompt is ~3-4K tokens. Well within Sonnet's 200K context. Cap at ~60 lines per song; sample if longer.

## Graceful Degradation

Every failure mode falls through silently:
1. Feature flag off → no lookup
2. `syncedlyrics` not installed → warning log, skip
3. Filename unrecognizable + no ID3 → skip
4. No lyrics in databases → `LyricsData` is None, Layer 5 omitted
5. Only plain text (no sync) → approximate bar positions via vocal_active
6. Lookup times out (15s) → warning log, skip
7. Any exception → caught, logged, pipeline continues

## Review Notes (from 2-round multi-agent review)

Medium issues to keep in mind during implementation:
- Wrap lyrics executor in `with` block or `try/finally` for cleanup
- Rule #8 should cross-reference Layer 5 bar numbers with Layer 2 section boundaries
- Filename parser should also handle track-number prefixes, YouTube video IDs, underscore separators
- `syncedlyrics` is a hard dependency (in pyproject.toml); don't also try to handle ImportError — pick one approach
- Update module docstring (line 10) and `_build_system_prompt` docstring (line 194) from "4-layer" to "5-layer"
- Use `if not result:` for syncedlyrics return check (can return empty string, not just None)

## Verification

1. **Unit tests:** `cd backend && uv run pytest tests/test_lyrics.py -v`
2. **Integration test with example songs:**
   - Start backend: `cd backend && uv run uvicorn musicmixer.main:app --port 8000`
   - Upload test songs (Biggie + Althea) via curl or frontend
   - Check logs for `"Lyrics found: X lines, synced=True"` messages
   - Check logs for the full system prompt to verify Layer 5 appears
3. **Verify LLM uses lyrics:** Compare remix plans with and without lyrics enabled — the `explanation` field should reference lyrical content when lyrics are available

## Estimated Total Effort: ~4 hours
