# YouTube Link Input — Implementation Plan

## Sound Engineering Rationale

### The Problem with YTMP3.sc (and Every Clone)

These services perform **double lossy compression**:

```
YouTube (Opus ~128kbps VBR) → decode → re-encode → MP3 128kbps
                                        ↑
                                   generation loss
```

Each lossy codec (Opus, AAC, MP3) uses a different psychoacoustic model. When you decode Opus and re-encode to MP3, the MP3 encoder treats *compression artifacts from Opus* as real audio and wastes bits preserving them, while simultaneously introducing its own artifacts. Even "320kbps MP3" output from these services is just a 320kbps encoding of a ~128kbps source — the extra bitrate is wasted on encoding artifacts, not recovering lost detail.

**Spectral result:** muddy transients, smeared stereo image, pre-echo artifacts compounding, high-frequency content degraded twice through independent lowpass filters.

### Our Approach: Single-Decode, Zero Generation Loss

```
YouTube (Opus ~128kbps VBR)
    ↓
    ↓  yt-dlp: download native stream + FFmpegExtractAudio postprocessor
    ↓
PCM WAV (int16) @ native rate (48kHz Opus / 44.1kHz AAC) stereo
    ↓
    ↓  (existing pipeline — no changes needed)
    ↓
Stem separation → mixing → export
```

**One decode. Zero re-encodes before processing.** The PCM WAV contains exactly what YouTube served — no more, no less. This is the theoretical maximum quality extractable from YouTube.

### Honest Quality Ceiling

YouTube's free tier caps at **Opus 251 (~128-160kbps VBR, 48kHz)** or **AAC 140 (128kbps CBR, 44.1kHz)**. No tool or technique can exceed this ceiling — the information simply isn't there. But we can *preserve* 100% of it, which is meaningfully better than the ~70-80% typical converters preserve after double compression.

For stem separation specifically, cleaner input → cleaner stem isolation → better remix. The artifacts from double compression don't just degrade the final output; they confuse the separation model, creating bleed and ghost artifacts in stems that wouldn't exist with cleaner input.

---

## What YouTube Actually Serves (Format Reference)

### Free Tier

| itag | Codec | Bitrate | Sample Rate | Container |
|------|-------|---------|-------------|-----------|
| 251  | Opus VBR | ~128-160 kbps | 48 kHz | WebM |
| 250  | Opus VBR | ~70 kbps | 48 kHz | WebM |
| 140  | AAC LC | 128 kbps | 44.1 kHz | M4A |
| 139  | AAC HE v1 | 48 kbps | 22 kHz | M4A |

### Premium (requires cookie auth — not planned for v1)

| itag | Codec | Bitrate | Sample Rate | Container |
|------|-------|---------|-------------|-----------|
| 774  | Opus VBR | ~256 kbps | 48 kHz | WebM |
| 141  | AAC LC | 256 kbps | 44.1 kHz | M4A |

**`bestaudio` selection order:** yt-dlp picks the highest-quality audio stream automatically. For free tier this is typically Opus 251.

---

## Implementation Plan

### Phase 1: Backend — YouTube Download Service

**New file:** `backend/src/musicmixer/services/youtube.py`

Core responsibilities:
- Accept a YouTube URL, validate it, extract audio as WAV
- Report download progress via callback (for SSE integration)
- Handle errors gracefully (invalid URL, unavailable video, age-restricted, etc.)

```python
# Conceptual API (not final code)

@dataclass
class YouTubeAudioResult:
    wav_path: Path           # int16 PCM WAV file ready for pipeline
    title: str               # Video title (for display)
    duration_seconds: float  # Duration (for validation)
    source_codec: str        # What YouTube served (opus/aac)
    source_bitrate: int      # kbps of source stream

async def download_youtube_audio(
    url: str,
    output_dir: Path,
    progress_callback: Callable[[float, str], None] | None = None,
) -> YouTubeAudioResult:
    ...
```

**Key implementation details:**

1. **Use yt-dlp as Python library** (not subprocess) — cleaner error handling, progress hooks, no shell escaping issues
2. **Format selection:** `bestaudio/best` — let yt-dlp pick the highest quality stream
3. **Post-processing:** Use yt-dlp's `FFmpegExtractAudio` postprocessor to produce int16 PCM WAV. Rationale: audio-separator converts to float32 internally regardless of input format; int16 halves file size for Modal upload (~20MB vs ~40MB per song); the source material is ~128kbps lossy so float32 adds no information.
4. **Sample rate:** Preserve the native sample rate from YouTube (48kHz for Opus, 44.1kHz for AAC). Do NOT force 44.1kHz during download/conversion. The existing `validate_stem()` in `processor.py` already resamples to 44.1kHz via librosa `soxr_hq` after stem separation — this is the correct place for resampling.
5. **Filename/storage:** Downloads go to `data/uploads/{session_id}/` (same directory as file uploads — no separate download directory). Use UUID-based naming to avoid wild filenames from `%(title)s`
6. **Playlist rejection:** `noplaylist: True` — only single videos accepted
7. **Duration cap:** Reject videos over 15 minutes (configurable) to prevent abuse and keep processing times reasonable
8. **progress_hooks:** Map yt-dlp's download progress to the SSE progress stream
9. **Quality metadata propagation:** `YouTubeAudioResult.source_codec` and `.source_bitrate` must flow through to `AudioMetadata` in `models.py`. Add a `source_quality: str | None` field to `AudioMetadata` (e.g., `"youtube-opus-128kbps"` or `None` for file uploads). This metadata is passed to `interpret_prompt()` so the LLM can make quality-aware mixing decisions (e.g., prefer the higher-quality source for lead elements, apply less aggressive processing to lossy sources).
10. **Title as original_filename:** Pass `YouTubeAudioResult.title` as `original_filename` to `run_pipeline()` for lyrics lookup integration.

**Dependency:** `yt-dlp` (add to `pyproject.toml`). ffmpeg is already a system dependency.

### Phase 2: Backend — API Changes

**Modified file:** `backend/src/musicmixer/api/remix.py`

The current endpoint accepts two file uploads. We need to support a new input mode: YouTube URLs.

**Option A (Recommended): New endpoint**
```
POST /api/remix/youtube
{
  "url_a": "https://youtube.com/watch?v=...",
  "url_b": "https://youtube.com/watch?v=...",
  "prompt": "Hendrix guitar with MF Doom rapping"
}
```

Rationale: Clean separation. File uploads use `multipart/form-data`; URLs use `application/json`. Different validation, different progress phases (download vs upload), different error modes. Mixing them in one endpoint adds complexity for no benefit.

**v1 scope:** Both songs must be YouTube URLs. Mixed input (one file + one YouTube URL) is deferred to v2 alongside the hybrid endpoint.

**Option B: Hybrid endpoint**
Accept a mix of file uploads and YouTube URLs in a single request. More flexible but messier — defer to v2 if needed.

No changes needed to `pipeline.py` — `run_pipeline()` already accepts arbitrary file paths. The YouTube endpoint constructs WAV paths and passes them directly.

#### Execution Model

- Endpoint receives JSON, validates URLs (SSRF check + basic pattern), creates session
- Returns `{ session_id }` immediately (same pattern as file upload endpoint)
- Downloads happen inside the background executor thread (same `ThreadPoolExecutor` as existing pipeline)
- Download progress flows through the SSE event queue

**Concurrency with `processing_lock`:**
- On request: check `processing_lock` non-blocking; if held, return 409 immediately (fail-fast, don't waste time downloading)
- Acquire lock, then download, then run pipeline, then release lock
- This prevents a user from waiting through a full download only to get a 409

**New progress phases:**
```
[0-5%]   Validating & starting download
[5-25%]  Downloading song A (proportional to bytes received)
[25-45%] Downloading song B (proportional to bytes received)
[45-70%] Separating stems
[70-90%] Processing & mixing
[90-100%] Export
```

Note: yt-dlp progress callbacks are throttled to max 1/second or 5% increments (whichever is less frequent). Handle `total_bytes=None` by falling back to `total_bytes_estimate`. Add `postprocessor_hooks` to track the ffmpeg WAV conversion phase. New `ProgressEvent` step: `'downloading'`.

The existing SSE progress stream (`/api/remix/{id}/progress`) needs a new `"downloading"` step with per-song detail.

**Required changes for `'downloading'` step:**
1. **`backend/src/musicmixer/models.py`:** Update the `ProgressEvent.step` docstring to include `'downloading'` as a valid step value (step is a `str` field, not an enum — the valid values are documented in the docstring).
2. **`frontend/src/types/index.ts`:** Add `'downloading'` to the `ProgressStep` string literal union type.
3. **`frontend/src/ProgressDisplay.tsx`:** Add a `'downloading'` case to the step display logic (label: "Downloading from YouTube...", appropriate progress bar styling).

### Phase 3: Frontend — URL Input Mode

**Modified files:**
- `frontend/src/SongUpload.tsx` — Add URL input alongside file upload
- `frontend/src/RemixForm.tsx` — Support new submission mode
- `frontend/src/api/client.ts` — New API call for YouTube URLs

**UX concept:**

Each song slot gets a toggle: **Upload File** | **YouTube Link**

When "YouTube Link" is selected:
- Text input for URL (with paste support)
- Client-side URL validation (basic YouTube URL pattern match)
- Thumbnail preview fetched via YouTube oEmbed API (no API key needed)
- Video title displayed after paste

When submitted:
- Progress bar shows download phase before separation phase
- Download progress per song reported via SSE

#### Type System Changes

- New discriminated union: `type SongInput = { type: 'file'; file: File } | { type: 'youtube'; url: string; title?: string; thumbnailUrl?: string }`
- `AppState` changes: song slots hold `SongInput | null` instead of `File | null`
- New `AppAction` variants: `SET_YOUTUBE_URL_A`, `SET_YOUTUBE_URL_B`
- New `ProgressStep`: `'downloading'` (before `'separating'`)
- New API client function: `submitYouTubeRemix(urlA, urlB, prompt)` returning `{ session_id }`
- `canSubmit` logic: valid when both slots are non-null (either type) and prompt >= 5 chars

#### Quality Indicators

- When a song is loaded via YouTube URL, show a subtle badge: "YouTube source (~128kbps)"
- When both songs are YouTube-sourced, show a gentle info message: "For best remix quality, upload high-quality audio files (WAV or 320kbps MP3) when available"
- After remix completes, the `source_codec` and `source_bitrate` from `YouTubeAudioResult` should be available in the session metadata for display

### Phase 4: Validation & Error Handling

#### Security: SSRF Prevention

Before passing any URL to yt-dlp, apply strict server-side validation **in this exact order** (order matters — step 3 must happen before step 5 to prevent userinfo bypass attacks):

1. Parse URL with `urllib.parse.urlparse()`
2. Reject non-https schemes
3. Reject URLs containing `@` in the netloc (prevents userinfo bypass like `youtube.com@evil.com`)
4. Reject IP literals and non-standard ports
5. Validate `.hostname` against allowlist: `youtube.com`, `www.youtube.com`, `m.youtube.com`, `youtu.be`, `music.youtube.com`

This validation **must happen BEFORE** the URL is passed to yt-dlp.

**URL validation (backend):**
- Must match YouTube URL patterns (`youtube.com/watch`, `youtu.be/`, `music.youtube.com/watch`)
- yt-dlp's own validation as fallback (it handles edge cases like playlists, shorts, etc.)
- Reject: playlists, channels, non-video URLs

**Error cases to handle:**
| Error | User-Facing Message |
|-------|-------------------|
| Video unavailable | "This video is unavailable or has been removed" |
| Age-restricted | "Age-restricted videos are not supported" |
| Region-blocked | "This video is not available in the server's region" |
| Live stream | "Live streams cannot be used — please use a completed video" |
| Too long (>15min) | "Videos must be under 15 minutes" |
| No audio stream | "No audio track found in this video" |
| Rate limited | "YouTube is temporarily limiting downloads. Try again in a minute" |
| Network error | "Failed to download from YouTube. Check the URL and try again" |
| SSRF via crafted URLs | "Invalid URL — only YouTube links are accepted" |

### Phase 5: Config & Dependencies

**`config.py` additions:**
```python
youtube_enabled: bool = True
youtube_max_duration_seconds: int = 900  # 15 minutes
```

**`pyproject.toml` addition:**
```toml
dependencies = [
    # ... existing ...
    "yt-dlp>=2025.1.0",
]
```

**No new system dependencies** — ffmpeg is already required.

---

## Source Quality Metadata Flow into Enhancement Pipeline

When audio is downloaded from YouTube, `YouTubeAudioResult` captures `source_codec` and `source_bitrate`. These flow into the enhancement pipeline as follows:

1. **`youtube.py`** sets `source_codec` and `source_bitrate` on `YouTubeAudioResult`
2. **`remix.py`** (YouTube endpoint) populates `AudioMetadata.source_quality` with a string like `"youtube-opus-128kbps"` (or `None` for file uploads)
3. **`pipeline.py`** reads `source_quality` from `AudioMetadata` at the top of the enhancement chain, determines `is_lossy_source = source_quality is not None and source_quality.startswith("youtube")`, and conditionally adjusts kwargs for each enhancement function BEFORE calling them
4. **Enhancement functions** receive source-quality-aware parameters via kwargs: `apply_corrective_eq(..., skip_hf_boosts=True)`, `multiband_compress(..., high_band_makeup_db=0.0)`, `master_static(..., lossy_lpf_hz=16000)`. See `sound-quality-enhancement-plan.md`, "Pipeline wiring for source_quality" for the full pseudocode.

> **Note:** Source-quality-aware adjustments are always-on for lossy sources (not gated by a feature flag). They are protective — preventing degradation from amplifying compression artifacts — not enhancements. See the design decision note in `sound-quality-enhancement-plan.md`.

This ensures that the sound quality enhancement chain does not amplify compression artifacts from lossy YouTube sources. For file uploads, `source_quality` is `None` and no adjustments are made.

---

## File Change Summary

| File | Change |
|------|--------|
| `backend/src/musicmixer/services/youtube.py` | **New** — YouTube download service |
| `backend/src/musicmixer/api/remix.py` | **Modified** — New `/api/remix/youtube` endpoint |
| `backend/src/musicmixer/config.py` | **Modified** — YouTube config settings |
| `backend/pyproject.toml` | **Modified** — Add yt-dlp dependency |
| `frontend/src/SongUpload.tsx` | **Modified** — Add URL input mode toggle |
| `frontend/src/RemixForm.tsx` | **Modified** — Support URL submission flow |
| `backend/src/musicmixer/models.py` | **Modified** — Add `source_quality` to `AudioMetadata`, add `'downloading'` to `ProgressEvent.step` docstring |
| `frontend/src/api/client.ts` | **Modified** — New API client method |
| `frontend/src/types/index.ts` | **Modified** — Add `SongInput` discriminated union type, add `'downloading'` to `ProgressStep` literal union |
| `frontend/src/hooks/useRemixReducer.ts` | **Modified** — Add `SET_YOUTUBE_URL_A` and `SET_YOUTUBE_URL_B` reducer actions for YouTube URL state management |
| `frontend/src/ProgressDisplay.tsx` | **Modified** — Add `'downloading'` case to step display logic |

---

## Quality Comparison

| Approach | Source | Intermediate | Pipeline Input | Gen. Loss Steps |
|----------|--------|-------------|----------------|-----------------|
| **YTMP3.sc** | Opus 128k | MP3 128k | MP3 128k (if user uploads) | **2** (decode+re-encode to MP3, decode MP3 to PCM) |
| **YTMP3.sc "320k"** | Opus 128k | MP3 320k | MP3 320k (if user uploads) | **2** (same — higher bitrate doesn't help) |
| **Ours** | Opus 128k | *none* | PCM WAV int16 (direct decode) | **0** (single decode, lossless) |

Our pipeline preserves 100% of the information YouTube provides. Converters lose 20-30% through double compression.

---

## What This Does NOT Solve

- **YouTube's quality ceiling:** Free tier maxes at ~128-160kbps. This is a hard limit set by YouTube, not by us. Local file uploads of high-bitrate MP3/WAV will always produce better remixes than YouTube sources.
- **Copyright/TOS:** Downloading YouTube audio violates YouTube's TOS. This is a product/legal decision. The app should position this as the user's responsibility (similar to how yt-dlp itself operates).
- **Age-restricted/Premium content:** Accessing these requires cookie-based auth, which adds significant complexity. Defer to v2.

---

## Implementation Order

1. **Backend: YouTube service** (`youtube.py`) — can be developed and tested independently
2. **Backend: API endpoint** — wire service into new endpoint, add SSE progress phases
3. **Frontend: URL input UI** — add toggle and URL input to song upload components
4. **Integration testing** — end-to-end flow with real YouTube URLs
5. **Error handling polish** — cover all edge cases from the error table above

Estimated scope: ~2-3 focused implementation sessions. No architectural changes to the core mixing pipeline — the YouTube download is purely a new input adapter that produces the same WAV files the pipeline already expects.
