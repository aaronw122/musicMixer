---
title: "feat: MVP Prompt-Based Remix"
type: feat
status: active
date: 2026-02-23
revision: 17
brainstorm: docs/brainstorms/2026-02-23-prompt-based-remix-brainstorm.md
prd: docs/PRD.md
reviews: notes/2026-02-23-audio-pipeline-review.md
---

# MVP: Prompt-Based Music Remix

## Overview

Build the end-to-end musicMixer MVP: upload two songs, describe a mashup in plain English, get an AI-generated remix. Server renders the final audio file; browser plays it back. Remix expires after 3 hours.

### MVP Scope Constraint

The MVP uses a **vocals + instrumentals** model: vocals always come from one song, instrumentals (drums + bass + other recombined) always come from the other. There is no cross-song stem mixing (e.g., drums from Song A with bass from Song B). However, the system still performs full 4-stem separation so the LLM can apply per-stem volume adjustments (e.g., "boost the bass," "quiet the drums") on the instrumental side. This constraint keeps the mixing simple while still delivering a compelling product, and the architecture is designed so cross-song stem mixing can be added later without structural changes.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                          FRONTEND                                 │
│                React + TypeScript + Vite (Bun)                    │
│                                                                   │
│  useReducer state machine: IDLE → UPLOADING → PROCESSING → READY │
│  sessionStorage persistence for crash recovery                    │
│                                                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │  Upload   │→ │  Prompt  │→ │ Progress │→ │     Player       │ │
│  │  2 songs  │  │  input   │  │  (SSE)   │  │ (<audio> + info) │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘ │
└──────────────────────┬───────────────────────────────────────────┘
                       │ REST API + SSE
┌──────────────────────┴───────────────────────────────────────────┐
│                          BACKEND                                  │
│                     Python + FastAPI                               │
│              Single worker, processing lock                       │
│     Pipeline runs in ThreadPoolExecutor, events via Queue         │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                  Pipeline Orchestrator                      │  │
│  │                                                             │  │
│  │  1. File Upload & Validation (audio parse check)            │  │
│  │  2. Stem Separation (audio-separator + htdemucs_ft)         │  │
│  │  3. Audio Analysis (librosa BPM + essentia key)             │  │
│  │  4. LLM Prompt Interpretation (Claude Haiku)                │  │
│  │  5. Audio Processing                                        │  │
│  │     a. Tempo matching (pyrubberband R3 + tiered limits)     │  │
│  │     b. Key matching (pyrubberband + formant preservation)   │  │
│  │     c. LUFS loudness normalization (pyloudnorm)             │  │
│  │     d. Per-stem volume adjustments (from LLM plan)          │  │
│  │     e. Mixing & fade-in/fade-out (pydub)                    │  │
│  │  6. Export MP3 + serve with 3-hour TTL                      │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### Key Architectural Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Backend language | Python | Entire audio ML ecosystem lives in Python. No point wrapping it. |
| Backend framework | FastAPI | Async, SSE via StreamingResponse, automatic OpenAPI docs, simple. |
| Package manager | `uv` | Faster than poetry, handles PyTorch's platform-specific wheels better. |
| Async model | `ThreadPoolExecutor` + `threading.Queue` | Pipeline is CPU-bound (blocks event loop). Thread worker keeps FastAPI responsive for SSE. Queue bridges pipeline→SSE. See "Async Processing Model" section. |
| Concurrency | Single-worker uvicorn, global processing lock | One remix at a time. Second request gets 429. Fine for personal/friends use. |
| Stem separation | `audio-separator` + `htdemucs_ft` | Single package wrapping best models. 4-stem output (vocals/drums/bass/other). ~10s on GPU. |
| BPM detection | `librosa` + half/double-time sanity check | Simple, well-documented. Sanity check catches the ~29% failure rate for tempo octave errors. |
| Key detection | `essentia` (fallback: librosa chromagram) | Best accuracy. Pluggable interface so essentia install issues don't block. |
| Tempo/key matching | `pyrubberband` with R3 engine | Industry-standard quality. Tiered stretch limits. Formant preservation for vocals. |
| Loudness | `pyloudnorm` (LUFS normalization on final mix) | LUFS normalization applied to the final summed mix (not per-stem). Per-stem normalization causes the sum to exceed target loudness, forcing the peak limiter to crush the audio. |
| Frequency management | Spectral ducking in MVP; full cross-song filtering post-MVP | Spectral ducking (QR-1) uses `scipy.signal.sosfiltfilt` to carve a mid-range pocket for vocals in the instrumental. Full cross-song stem filtering (e.g., competing bass lines) deferred to post-MVP when cross-song mixing is added. |
| Audio mixing | `pydub` + `ffmpeg` + fade-in/fade-out | High-level API for overlay, volume, fades, export. |
| Remix delivery | Server-rendered single MP3 file | Massively simplifies frontend — just an `<audio>` tag. No Tone.js needed. |
| Browser playback | HTML5 `<audio>` element | Zero JS dependencies. Native controls. Streams large files. |
| LLM for prompts | Claude Sonnet (structured outputs, configurable) | Section-based arrangement requires structured reasoning where Sonnet outperforms Haiku. ~$0.003/call, schema-guaranteed JSON via tool_use. Downgrade to Haiku if latency is an issue. |
| Progress updates | SSE via `StreamingResponse` | Simpler than WebSockets for one-way server→client. Manual SSE formatting (no sse-starlette needed). |
| Frontend framework | React + Vite + TypeScript | Standard, fast, good DX with Bun. |
| Styling | Tailwind CSS | Rapid prototyping, no component library overhead. |
| State management | `useReducer` with discriminated union state | App has a finite state machine. Typed union prevents impossible states. |
| Session persistence | `sessionStorage` | Survives page refresh during processing. |

### Design Principles

- **Orthogonality**: Each backend service (separation, analysis, LLM, mixing) is a standalone module with a clean interface. Swap any piece without touching others.
- **Reversibility**: Every technology choice has a clear migration path. Switch stem models by changing one config line. Swap LLM providers by changing one module. Swap essentia for librosa key detection if install fails.
- **Server-side rendering**: The single biggest simplification. The frontend never touches raw audio data — it uploads files and plays back a URL.
- **Minimal API surface**: One endpoint to create a remix, one SSE endpoint for progress, one endpoint to serve the audio file, one status endpoint for reconnection.
- **Build locally, buy as fallback**: Run all ML models locally ($0.003/remix on GPU). If GPU setup is painful, Fadr API ($0.05/min) can replace stem separation + analysis in one call.

---

## Async Processing Model

This is the backbone of the backend. The pipeline is CPU-bound (PyTorch stem separation, audio processing), so it **cannot** run in the FastAPI async event loop without blocking all request handling.

### Architecture

```python
# In-memory session state
sessions: dict[str, SessionState] = {}
sessions_lock = threading.Lock()  # Guards sessions dict mutations (add/delete)
processing_lock = threading.Lock()  # One remix at a time
executor = ThreadPoolExecutor(max_workers=1)

@dataclass
class SessionState:
    status: str                      # "queued" | "processing" | "complete" | "error"
    events: queue.Queue              # Pipeline pushes, SSE endpoint reads. maxsize=100 prevents unbounded growth if no consumer is connected.
    created_at: datetime
    remix_path: str | None = None
    explanation: str | None = None
    last_event: dict | None = None   # Most recent ProgressEvent (for reconnecting SSE clients)

# Thread safety contract:
# - sessions dict mutations (add AND delete) are guarded by sessions_lock.
#   This includes session creation in the POST handler (sessions[sid] = SessionState(...)).
# - SessionState attribute assignments (status, remix_path, explanation, last_event) are
#   atomic under CPython's GIL for simple assignments. The pipeline MUST always assign new
#   objects, NEVER mutate existing dicts/lists in place.
# - The events queue is inherently thread-safe (threading.Queue).

# POST /api/remix → saves files, creates session, submits to executor
# Pipeline runs in thread, pushes events to session.events queue
# GET /api/remix/{id}/progress → async wrapper reads from session.events queue
# GET /api/remix/{id}/status → returns current status (for reconnection)
```

### Why ThreadPoolExecutor (not ProcessPoolExecutor)

- `threading.Queue` works across threads natively. `multiprocessing.Queue` requires pickling, which fails with many audio objects.
- GIL is not a problem: PyTorch releases the GIL during GPU computation. Audio I/O releases the GIL during file operations. The thread worker and FastAPI event loop can run concurrently.
- Single worker thread + processing lock = one remix at a time. Simple, correct, sufficient for MVP.

### SSE-to-Pipeline Communication

The pipeline function accepts a `queue.Queue` and pushes `ProgressEvent` dicts to it at each step. The SSE endpoint wraps the queue read in an async generator:

```python
# Dedicated executor for SSE queue reads — isolates from the default executor
# so client disconnects (which leave threads blocked until timeout) don't exhaust
# threads used by other async operations.
# IMPORTANT: When a client disconnects, the blocking queue.get() thread can't be
# interrupted — it sits for up to 5 seconds (the timeout). Rapid reconnections
# (user refreshing repeatedly) can temporarily exhaust all 4 threads. Mitigation:
# 1. Short timeout (5s) means threads recycle quickly.
# 2. The SSE endpoint checks the sessions dict BEFORE acquiring a thread.
# 3. If all SSE threads are busy, the new connection queues (not rejected) and
#    resolves within 5s — acceptable for MVP (single user).
sse_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="sse-reader")

async def event_stream(session: SessionState):
    loop = asyncio.get_running_loop()  # not get_event_loop() (deprecated 3.10+)
    start = time.monotonic()
    while True:
        # Safety cap: if no complete/error arrives within max_sse_duration, emit error and close.
        # Prevents permanent resource leak if the pipeline hangs.
        if time.monotonic() - start > settings.max_sse_duration_seconds:  # default: 1200 (20 min)
            yield 'data: {"step":"error","detail":"Processing timed out","progress":0}\n\n'
            break
        try:
            # Read with 5s timeout (not 30s) — threads return to pool faster on client disconnect.
            # More frequent keepalives are fine for SSE.
            event = await loop.run_in_executor(
                sse_executor, functools.partial(session.events.get, timeout=5)
            )
        except queue.Empty:
            # MUST use data: event, not comment (: keepalive). EventSource.onmessage
            # does not fire for SSE comments, so the frontend's no-event timeout
            # would misfire during long processing steps (stem separation: 3-6 min).
            # Include all ProgressEvent fields so it passes TypeScript type-checking.
            yield 'data: {"step":"keepalive","detail":"","progress":-1}\n\n'
            continue
        except asyncio.CancelledError:
            break  # Client disconnected — exit cleanly
        # Store on session before yielding so reconnecting clients get the latest event
        # even if this consumer is cancelled between dequeue and yield.
        session.last_event = event
        yield f"data: {json.dumps(event)}\n\n"
        if event["step"] in ("complete", "error"):
            break
```

### Reconnection

On SSE connect, immediately send the current session status. If the remix is already complete, send the final event and close. If processing, replay the latest event and continue streaming. This handles page refresh and tab backgrounding without needing `Last-Event-Id`.

A separate `GET /api/remix/{session_id}/status` endpoint returns the current state as JSON (not SSE) for quick reconnection checks.

### Concurrent SSE Consumers

A single `queue.Queue` delivers each event to exactly one consumer. If two SSE clients connect to the same session (e.g., two tabs), they see interleaved, incomplete streams. **MVP mitigation:** Store the latest event on the session object (`session.last_event`). On SSE connect, always send `last_event` first. This ensures reconnecting clients get the current state even if another consumer drained the queue. For true broadcast, post-MVP can switch to `asyncio.Condition` + shared state.

---

## Backend Execution Plan

> All backend work happens in `backend/`. Python project using FastAPI + uv.

### Phase B1: Project Scaffolding

**Goal:** Runnable FastAPI server with health check, CORS, config, logging.

- [ ] Initialize Python project with `uv` and `pyproject.toml`
- [ ] Pin Python version (3.11 or 3.12 — test essentia compatibility)
- [ ] Create project structure:
  ```
  backend/
  ├── pyproject.toml
  ├── CLAUDE.md
  ├── src/
  │   └── musicmixer/
  │       ├── __init__.py
  │       ├── main.py              # FastAPI app, lifespan, CORS
  │       ├── config.py            # Pydantic BaseSettings
  │       ├── models.py            # Shared dataclasses/Pydantic models
  │       ├── exceptions.py        # Custom exception types
  │       ├── api/
  │       │   ├── __init__.py
  │       │   ├── health.py        # GET /health
  │       │   └── remix.py         # POST /api/remix, GET progress, GET audio, GET status
  │       └── services/
  │           └── __init__.py
  ├── data/                        # Runtime data (gitignored)
  │   ├── uploads/
  │   ├── stems/
  │   └── remixes/
  └── tests/
      ├── __init__.py
      └── fixtures/                # Short test audio files (public domain)
  ```
- [ ] Install core dependencies: `fastapi`, `uvicorn[standard]`, `python-multipart`, `pydantic-settings`
- [ ] Configure CORS middleware:
  ```python
  app.add_middleware(
      CORSMiddleware,
      allow_origins=settings.cors_origins,  # ["http://localhost:5173"]
      allow_methods=["*"],
      allow_headers=["*"],
  )
  ```
- [ ] Configure structured logging with `structlog` or Python `logging`:
  - Include session_id in all log entries
  - INFO for request handling, DEBUG for pipeline steps
- [ ] Implement `config.py` with Pydantic `BaseSettings`:
  ```python
  class Settings(BaseSettings):
      # Server
      host: str = "0.0.0.0"
      port: int = 8000
      cors_origins: list[str] = ["http://localhost:5173"]

      # File limits
      max_file_size_mb: int = 50
      max_duration_seconds: int = 600  # 10 minutes
      allowed_extensions: set[str] = {".mp3", ".wav"}

      # Storage
      data_dir: Path = Path("data")
      remix_ttl_seconds: int = 10800  # 3 hours
      cleanup_interval_seconds: int = 300  # 5 minutes
      max_sse_duration_seconds: int = 1200  # 20 minutes — safety cap for hung pipelines

      # Processing
      stem_model: str = "htdemucs_ft"
      max_tempo_adjustment_pct: float = 0.30
      max_pitch_shift_semitones: int = 5
      output_format: str = "mp3"
      output_bitrate: str = "320k"
      target_lufs: float = -14.0

      # LLM
      anthropic_api_key: str  # Required, no default
      llm_model: str = "claude-sonnet-4-20250514"  # Default Sonnet for section-based arrangement quality; configurable to Haiku if latency is an issue
      llm_max_retries: int = 2
      llm_timeout_seconds: int = 30

      model_config = SettingsConfigDict(env_file=".env")
  ```
- [ ] Implement `models.py` with shared types (see Phase B5 for RemixPlan)
- [ ] Implement `exceptions.py`:
  ```python
  class MusicMixerError(Exception): ...
  class ValidationError(MusicMixerError): ...
  class SeparationError(MusicMixerError): ...
  class AnalysisError(MusicMixerError): ...
  class ProcessingError(MusicMixerError): ...
  class PipelineError(MusicMixerError): ...
  ```
- [ ] Implement health check endpoint (`GET /health`)
- [ ] Set up FastAPI `lifespan` for startup/shutdown (used by TTL cleanup later)
- [ ] Add `CLAUDE.md` for backend repo with conventions
- [ ] Verify server starts and responds

**Files created:** `pyproject.toml`, `src/musicmixer/main.py`, `src/musicmixer/config.py`, `src/musicmixer/models.py`, `src/musicmixer/exceptions.py`, `src/musicmixer/api/health.py`, `src/musicmixer/api/remix.py`, `CLAUDE.md`

---

### Phase B2: File Upload & Validation

**Goal:** Accept two audio files via multipart upload. Validate thoroughly and store them.

- [ ] Create `POST /api/remix` endpoint accepting:
  - Two audio files (`song_a`, `song_b`) as multipart form data
  - A `prompt` text field
- [ ] **Check processing lock FIRST** — before accepting the upload. Use `processing_lock.locked()` (a non-mutating, read-only check) to fail-fast with 429 before wasting time on upload. The authoritative lock acquisition happens only once, after validation, before executor submission:
  ```python
  # In POST /api/remix handler:
  # 1. Fail-fast check (non-mutating — never acquires the lock)
  if processing_lock.locked():
      raise HTTPException(429, "Server is busy processing another remix")
  # 2. Accept upload, validate files, save to disk
  # 3. Acquire lock once (authoritative gate) and submit:
  if not processing_lock.acquire(blocking=False):
      raise HTTPException(429, "Server is busy")  # Narrow race — another request won
  # 4. Submit to executor via wrapper that guarantees lock release:
  def pipeline_wrapper():
      try:
          run_pipeline(session_id, song_a_path, song_b_path, prompt, session.events)
      finally:
          processing_lock.release()
  try:
      future = executor.submit(pipeline_wrapper)
      future.add_done_callback(lambda f: _log_unhandled(f, session_id))
  except Exception:
      processing_lock.release()
      raise
  ```
  Note: `Lock.locked()` is still technically a TOCTOU check (the lock state can change between the check and the upload), but it never acquires or releases the lock, so it cannot leak. The single `acquire(blocking=False)` at step 3 is the authoritative gate. The wrapper pattern keeps `run_pipeline` pure (no lock parameter) while guaranteeing release via `finally`.
  **Remaining UX gap:** A user could start uploading when the server is free, but by the time the upload completes (80+ seconds for 100MB on slow connections), another remix has started. They'll see a 429 after waiting. This is acceptable for the single-user MVP — the frontend error message should suggest waiting and trying again.
- [ ] Streaming upload handling — FastAPI's `UploadFile` uses `SpooledTemporaryFile` (doesn't buffer entire file in memory)
- [ ] **Server-level request size limit**: Configure uvicorn with `--limit-max-body-size 104857600` (100MB — room for two 50MB files + form fields). Rejects oversized requests before FastAPI even sees them.
- [ ] **Prompt validation**: Minimum 5 characters, maximum 1000 characters. Return 422 if out of bounds. Prevents empty prompts (LLM has nothing to work with) and excessively long ones (token cost, potential rejection).
- [ ] Multi-layer file validation in `services/upload.py`:
  1. Check file extension (`.mp3`, `.wav`)
  2. Check MIME type / magic bytes
  3. **Parse the file with pydub/ffprobe** — if it can't be decoded as audio, reject it. This is the real validation.
  4. Check file size (max 50MB, enforced during streaming read — abort early if exceeded)
  5. Check duration (max 10 minutes, checked after audio parse)
- [ ] Generate session ID with `uuid.uuid4()` (not uuid1 — uuid4 is cryptographically random)
- [ ] Save uploaded files to `data/uploads/{session_id}/song_a.{ext}`, `data/uploads/{session_id}/song_b.{ext}`
- [ ] Validate resolved file paths are within `data/` directory (path traversal defense via `pathlib.Path.resolve()`)
- [ ] Return session ID immediately; submit pipeline to ThreadPoolExecutor
- [ ] Pydantic response model:
  ```python
  class RemixResponse(BaseModel):
      session_id: str
  ```

**Orthogonality:** Upload handling is isolated in its own service. The rest of the pipeline receives file paths, not upload objects.

**Files created:** `src/musicmixer/services/upload.py`
**Files modified:** `src/musicmixer/api/remix.py`, `src/musicmixer/models.py`

---

### Phase B3: Stem Separation Service

**Goal:** Split each uploaded song into 4 stems (vocals, drums, bass, other).

- [ ] Install `audio-separator`
- [ ] Pin PyTorch version and variant in `pyproject.toml`:
  - For Mac development: CPU or MPS (Metal)
  - For production GPU: CUDA 12.1 wheels
  - Use `--extra-index-url` for PyTorch custom index
  - Document exact installation steps in CLAUDE.md
- [ ] Create `services/separation.py`:
  - `separate_stems(audio_path: Path, output_dir: Path, progress_callback: Callable | None = None) -> dict[str, Path]`
  - Uses `htdemucs_ft` model (configurable via Settings)
  - Returns mapping: `{"vocals": Path, "drums": Path, "bass": Path, "other": Path}`
  - Output format: WAV (uncompressed, needed for quality processing downstream)
  - Model auto-downloads on first run
- [ ] Handle errors: corrupt files, unsupported formats, out-of-memory → raise `SeparationError`
- [ ] Progress callback called at start/end of separation for SSE updates
- [ ] Write unit test with a short (5-second) test audio fixture

**Reversibility:** Swap `htdemucs_ft` for Mel-Band RoFormer (better vocals) or any other model by changing `settings.stem_model`. The service interface doesn't change.

**Disk note:** Each song produces ~4 stems as WAV. A 4-minute song ≈ 40MB per stem = ~160MB per song, ~320MB for two songs. Acknowledged in TTL cleanup design.

**GPU note:** ~10s/song on GPU, ~3-6min/song on CPU. MVP works on CPU but UX will be slow.

**Files created:** `src/musicmixer/services/separation.py`, `tests/test_separation.py`, `tests/fixtures/test_short.wav`

---

### Phase B4: Audio Analysis

**Goal:** Detect BPM, musical key, and energy profile for each uploaded song.

- [ ] Install `librosa`, `essentia`
  - For essentia: pre-built wheels on PyPI for macOS ARM64/x86_64 (Python 3.10-3.13)
  - If essentia install fails, fall back to librosa chromagram — **design this fallback into the interface from the start**
- [ ] Create `services/analysis.py`:
  - `analyze_audio(audio_path: Path) -> AudioMetadata`
  - `AudioMetadata` and `EnergyRegion` in `models.py`:
    ```python
    @dataclass
    class EnergyRegion:
        start_sec: float
        end_sec: float
        relative_energy: float   # 0.0–1.0 normalized (1.0 = loudest region)
        onset_density: float     # 0.0–1.0 normalized — rhythmic activity level
        label: str               # "high" | "medium" | "low" — composite energy label
        character: str           # "rhythmic" | "sustained" | "sparse" | "moderate"
        # Character derived from energy × onset density:
        #   high energy + high onsets = "rhythmic" (chorus, drop)
        #   high energy + low onsets  = "sustained" (breakdown, build, pad section)
        #   low energy + high onsets  = "sparse" (verse with sparse percussion)
        #   low energy + low onsets   = "moderate" (intro, outro, quiet passage)
        # This distinction is critical for the LLM's arrangement decisions —
        # without it, breakdowns and choruses are indistinguishable.

    @dataclass
    class AudioMetadata:
        bpm: float
        bpm_confidence: float     # for half/double-time decision
        beat_frames: np.ndarray   # Beat frame positions from librosa.beat.beat_track — the beat GRID.
                                  # Essential for QR-1's section-based arrangement: the mixer converts
                                  # Section.start_beat / end_beat to sample positions via this grid,
                                  # NOT via constant-BPM math (which drifts from actual beats in songs
                                  # with tempo variation). librosa already computes this as a byproduct
                                  # of BPM detection — just store it instead of discarding.
                                  # After tempo matching via rubberband, scale positions proportionally
                                  # by the tempo ratio: beat_frames_stretched = beat_frames * (original_bpm / target_bpm).
        key: str                  # e.g., "C"
        scale: str                # "major" or "minor"
        key_confidence: float
        duration_seconds: float
        total_beats: int          # total beats in the song (rounded to nearest bar boundary)
        vocal_energy_ratio: float = 0.0  # 0.0–1.0 — vocal stem LUFS relative to full mix.
                                  # Default 0.0, filled by pipeline AFTER stem separation (B3) and BEFORE LLM call (B5).
                                  # Pipeline step: measure LUFS of vocal stem vs full mix using pyloudnorm.
                                  # ~0.8+ = clear prominent vocals. ~0.1 = effectively instrumental.
                                  # Critical signal for LLM's vocal_source decision when prompt is vague.
        energy_regions: list[EnergyRegion]  # sorted by time (temporal order), ~8s windows
    ```
  - BPM via `librosa.beat.beat_track` — **returns both BPM and beat frame positions; store BOTH**
  - **BPM reconciliation (expanded)**: This is a **cross-song reconciliation step** that runs AFTER both songs have been individually analyzed. It cannot happen inside the single-song `analyze_audio` function because it needs both BPMs to compare.
    - **Implementation:** Add a separate `reconcile_bpm(meta_a: AudioMetadata, meta_b: AudioMetadata) -> tuple[AudioMetadata, AudioMetadata]` function called by the pipeline after analyzing both songs.
    - **Expanded interpretation matrix**: For each song, generate all plausible BPM interpretations: {original, halved, doubled, 3/2, 2/3}, then filter to the **70-180 BPM** range (tighter than the naive 60-200 -- 60 BPM is barely a pulse, 200+ is speed metal edge). The 3/2 and 2/3 ratios capture triplet-feel tempos common in hip-hop and jazz (a track at 90 BPM with triplet hi-hats may be better interpreted as 135 BPM).
    - Evaluate all valid pairs across the two songs, select the pair with the minimum percentage difference. This catches the ~29% of songs where librosa reports the wrong octave AND some additional cases the 2:1-only check misses (e.g., R&B at 72 doubled to 144 brings it within 13% of EDM at 128).
    - **What this doesn't rescue:** Hip-hop (85) + pop (120) = 41% gap has no half/double/triplet relationship that brings them into range. This is the most-attempted mashup pairing. See QR-3 for the alternative strategy (vocals-only stretching).
    - Return updated copies of the metadata (don't mutate originals).
  - Key via `essentia.standard.KeyExtractor` (or librosa chromagram fallback)
  - Duration via `librosa.get_duration`
- [ ] **Energy profile analysis** (enables informed LLM section selection):
  - Compute RMS energy in ~8-second windows using `librosa.feature.rms` (already loaded for BPM)
  - Normalize energy values to 0.0–1.0 relative scale
  - Label windows: top 25% as "high", bottom 25% as "low", rest as "medium"
  - Return all windows sorted by energy descending as `energy_regions`
  - **Combine with onset density** (`librosa.onset.onset_strength`) to discriminate between sustained pads (high energy, low onsets = breakdown) vs. busy sections (high energy, high onsets = chorus/drop). This is not optional — RMS energy alone cannot distinguish breakdowns from choruses, which leads to the LLM selecting atmospheric sections instead of the energetic ones users expect. Multiply normalized RMS energy by normalized onset density for the final score.
  - **Cost:** Near-zero — librosa already loads the audio for BPM detection. Adds milliseconds, no new dependencies.
  ```python
  def compute_energy_profile(y: np.ndarray, sr: int, window_sec: float = 8.0) -> list[EnergyRegion]:
      """Compute RMS energy in fixed windows. Returns regions sorted by energy descending."""
      rms = librosa.feature.rms(y=y, hop_length=512)[0]
      frames_per_window = int(window_sec * sr / 512)
      regions = []
      for i in range(0, len(rms), frames_per_window):
          chunk = rms[i:i + frames_per_window]
          if len(chunk) == 0:
              continue
          start = i * 512 / sr
          end = min((i + frames_per_window) * 512 / sr, len(y) / sr)
          regions.append(EnergyRegion(start_sec=start, end_sec=end, relative_energy=float(chunk.mean()), label=""))
      # Normalize to 0.0–1.0
      max_energy = max(r.relative_energy for r in regions) if regions else 1.0
      for r in regions:
          r.relative_energy = r.relative_energy / max_energy if max_energy > 0 else 0.0
          r.label = "high" if r.relative_energy >= 0.75 else "low" if r.relative_energy <= 0.25 else "medium"
      # Return in temporal order (not sorted by energy) — temporal context is critical
      # for the LLM to understand song structure (verse at 0:30 vs chorus at 1:00).
      # The condensed format for the LLM groups by character label anyway.
      regions.sort(key=lambda r: r.start_sec)
      return regions
  ```
- [ ] Key detection interface is pluggable:
  ```python
  def detect_key(audio_path: Path) -> tuple[str, str, float]:
      """Returns (key, scale, confidence). Uses essentia if available, else librosa."""
  ```
- [ ] Write unit test with test fixtures

**Orthogonality:** Analysis is pure — takes a file path, returns metadata. No side effects.

**Note:** Analysis runs on the original full songs, not the stems. BPM and key are properties of the whole track.

**Files created:** `src/musicmixer/services/analysis.py`, `tests/test_analysis.py`

---

### Phase B5: LLM Prompt Interpretation

**Goal:** Convert user's natural language prompt into a structured stem-selection and mixing plan.

- [ ] Install `anthropic` SDK
- [ ] Create `services/interpreter.py`:
  - `interpret_prompt(prompt: str, song_a_meta: AudioMetadata, song_b_meta: AudioMetadata) -> RemixPlan`
  - `RemixPlan` in `models.py` — see QR-1 for the canonical `Section`-based schema. Key fields:
    ```python
    @dataclass
    class RemixPlan:
        vocal_source: str                    # "song_a" | "song_b"
        start_time_vocal: float              # Where to start in the vocal source song (seconds)
        end_time_vocal: float                # Where to end in the vocal source song
        start_time_instrumental: float       # Where to start in the instrumental source song
        end_time_instrumental: float         # Where to end in the instrumental source song
        sections: list[Section]              # Beat-aligned arrangement (see QR-1)
        tempo_source: str                    # "song_a" | "song_b" | "average"
        key_source: str                      # "song_a" | "song_b" | "none"
        explanation: str                     # LLM's reasoning (shown to user)
        warnings: list[str]                  # Caveats about what couldn't be fulfilled
        used_fallback: bool = False          # True when deterministic fallback was used
    ```
    **MVP constraint:** `vocal_source` determines the split. Vocals come from that song; drums, bass, and other all come from the other song. No cross-song stem mixing. Source time ranges (`_vocal`/`_instrumental`) select which region of each song to extract; `sections` arrange the extracted material over time.
  - **System prompt design** — the "product brain"; write and test as a first-class deliverable:
    - **MVP constraint and capability boundaries**: Vocals from one song, all instrumentals from the other. No cross-song stem mixing. Explicit capability section: "You can combine stems, adjust volumes, choose sections, and control arrangement structure. You CANNOT add effects, generate new sounds, isolate individual instruments within 'other', or use vocals from both songs. Acknowledge limitations in `warnings`."
    - Available stems and what "other" contains (guitar/synths/keys — be honest with users)
    - Song metadata: BPM, key, duration, `total_beats` (eliminates LLM arithmetic), `vocal_energy_ratio` (critical signal for vocal_source decision when prompt is vague)
    - **Energy regions (condensed format)**: Do NOT pass raw `EnergyRegion` objects. Pre-process into condensed text summary (~10x fewer tokens):
      ```
      Song A (120 BPM, C major, 240s, 480 beats, vocal_energy: 0.82):
        Rhythmic (chorus/drop): 64s-80s, 128s-144s
        Sustained (breakdown): 48s-64s
        Sparse (verse): 16s-48s
        Moderate (intro/outro): 0s-16s, 200s-240s
      ```
      Groups by `character` label. Includes conversion note: "1 beat = 0.5s at 120 BPM."
    - **Arrangement templates** for the LLM to select from and adapt (constrains creative space to proven structures):
      ```
      Template A (Standard Mashup): intro(8 bars, inst) -> verse(16 bars, vocals) -> breakdown(8 bars, drums drop) -> drop(16 bars, full) -> outro(8 bars)
      Template B (DJ Set): build(16 bars) -> vocals in(16 bars) -> peak(16 bars) -> vocals out(8 bars) -> outro(8 bars)
      Template C (Quick Hit): intro(4 bars) -> vocal drop(16 bars) -> outro(4 bars)
      Template D (Chill): intro(16 bars, sparse) -> vocals(32 bars, gentle) -> outro(16 bars, fade)
      ```
    - **Tempo guidance** (separate from key): `"average"` only when BPMs differ by <15%. 15-30%: prefer vocal source tempo. >30%: skip, explain in `explanation`.
    - **Key guidance** (separate from tempo): Skip if keys >4 semitones apart or `key_confidence` < 0.3.
    - **Stem artifact awareness**: "Stem separation is imperfect. Vocal stem may contain instrument traces. Instrumental stems may contain ghost vocals. Bleed is less noticeable during high-energy sections."
    - **Handling ambiguous/contradictory/impossible prompts** (majority case):
      1. **Vague** ("make it cool"): Use energy profiles, pick vocals from higher `vocal_energy_ratio`, use standard template.
      2. **Contradictory** ("vocals from both"): Acknowledge in `warnings`, produce best plan within limits.
      3. **Genre jargon** ("trap", "lo-fi"): Translate to possible actions — tempo, gains, structure.
      4. **Inaudible references** ("guitar solo at 2:30"): Use time range, add warning.
    - **Explanation quality**: Non-technical, 2-3 sentences, key creative decisions, no internal jargon.
    - **`warnings` field**: Populate for vague/contradictory/unverifiable prompts. Frontend displays distinctly.
    - **Few-shot examples**: 3 diverse examples: (a) clear directive, (b) vague, (c) contradictory. Full context + tool_use output. ~300 tokens each (~900 total).
  - Uses Claude with tool_use for schema enforcement. **Default to Sonnet** (configurable via `settings.llm_model`). Section-based arrangement requires structured reasoning where Sonnet outperforms Haiku. At $0.003/call, cost is negligible for single-user.
  - **Token budget**: System prompt + few-shot + condensed context under ~2,500 tokens.
- [ ] **Post-LLM validation** — two layers:
  - **Time range validation**: `0 <= start_time_X < end_time_X <= duration_X`, min 5.0s. Clamp out-of-bounds. **If clamping by >5s, append to `warnings`**. Log original vs. clamped.
  - **Section list validation**: Sections contiguous (`end_beat[N]` = `start_beat[N+1]`), starts at beat 0, last at/before `total_beats`, `transition_beats` < section length, gains in `[0.0, 1.0]`. On failure: **surgical code fix** (fix gaps, clamp, normalize), then re-prompt if still broken, then deterministic fallback. Log every violation.
- [ ] **Semantic validation**: Verify enum values (`vocal_source`, `tempo_source`, etc.). Check `response.stop_reason` — if `max_tokens`, fall back immediately.
- [ ] **LLM observability** — structured logging:
  ```python
  logger.info("llm_request", session_id=session_id, prompt=prompt,
              song_a_bpm=meta_a.bpm, song_b_bpm=meta_b.bpm,
              song_a_vocal_energy=meta_a.vocal_energy_ratio)
  logger.info("llm_response", session_id=session_id, raw_response=raw_json,
              latency_ms=latency, model=response.model,
              used_fallback=False, clamped_fields=[], warnings=plan.warnings)
  ```
- [ ] Error handling:
  - **Tiered timeouts**: First attempt 15s, retry 30s (most calls 2-5s)
  - Retry 1-2x with backoff on transient errors (429, 500, 529)
  - Schema violation: surgical code fix -> re-request -> fallback
  - Total failure: **deterministic fallback** (below)
- [ ] **Deterministic fallback plan** — fires on every LLM failure; data-driven:
  ```python
  def generate_fallback_plan(meta_a: AudioMetadata, meta_b: AudioMetadata) -> RemixPlan:
      """Deterministic fallback using analysis data. No LLM required."""
      vocal_src = "song_a" if meta_a.vocal_energy_ratio >= meta_b.vocal_energy_ratio else "song_b"
      vocal_meta = meta_a if vocal_src == "song_a" else meta_b
      inst_meta = meta_b if vocal_src == "song_a" else meta_a
      # find_best_region returns the highest-energy region matching the preferred character.
      # Expand to at least 90s centered on that region (energy windows are ~8s — too short for a remix).
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
- [ ] Write unit test with mocked LLM response
- [ ] **Write and test the system prompt** against categorized test suite (see QR-2) before building the pipeline

**Reversibility:** Interface is `(prompt, metadata, metadata) → RemixPlan`. Swap Claude for OpenAI by changing one module. Swap Sonnet for Haiku via `settings.llm_model` if latency is a concern. Deterministic fallback works without any LLM.

**Files created:** `src/musicmixer/services/interpreter.py`, `tests/test_interpreter.py`

---

### Phase B6: Audio Processing (Tempo/Key Matching + Mixing)

**Goal:** Time-stretch, pitch-shift, normalize loudness, filter frequencies, mix stems, and render the final remix.

- [ ] Install `pyrubberband`, `pydub`, `soundfile`, `pyloudnorm`
- [ ] System dependencies: `ffmpeg`, `rubberband` (brew install / apt install)
  - Service should fail with a clear error if `rubberband` CLI is not found
- [ ] Create `services/processor.py`:
  - `process_remix(remix_plan: RemixPlan, song_a_stems: dict[str, Path], song_b_stems: dict[str, Path], song_a_meta: AudioMetadata, song_b_meta: AudioMetadata, output_path: Path, progress_callback: Callable | None = None) -> Path`
  - Steps:
    1. **Sample rate + channel standardization** (MUST be first): Ensure all stems are at 44.1kHz stereo 32-bit float before any processing. htdemucs_ft outputs 44.1kHz but verify and resample with `librosa.resample` if needed. **Mono-to-stereo**: htdemucs preserves input channel count, so a mono Song A and stereo Song B produce mismatched stems. Convert mono stems to stereo by duplicating the channel: `stereo = np.column_stack([mono, mono])`. This prevents shape mismatches during mixing and pitch/timing bugs from feeding mismatched sample rates to rubberband.
    2. **Trim stems** to the time ranges specified in RemixPlan (`start_time_a`/`end_time_a`, etc.). **Important:** Trimming happens BEFORE tempo stretch. The LLM's time ranges are in the original tempo domain. After stretching, the trimmed segment's duration will change proportionally to the tempo ratio (e.g., a 60s segment at 2x stretch = 120s). The processor must account for this when computing final output duration — trim first, then let rubberband change the length.
    3. **Tempo + key matching in a single Rubber Band pass**:
       Tempo stretch and pitch shift MUST be applied in a single `rubberband` CLI invocation per stem. Two separate passes compound artifacts (1-2 dB noise floor increase, phase smearing), especially on vocals.
       - **Tempo matching** with tiered limits:
         - < 10% BPM difference: stretch either/both silently
         - 10-25%: stretch vocals only (preserve instrumental tempo — drums are the most stretch-sensitive element, sounding unnatural beyond 8-10% stretch; vocals tolerate 15-20% stretch with formant preservation)
         - 25-30%: stretch vocals only, strongly warn in explanation
         - 30-50%: **vocals-only stretch** — stretch the vocal stem to match the instrumental's tempo. Do NOT skip tempo matching entirely. Vocals stretched by 30-50% sound noticeably different but remain intelligible; an instrumental stretched by the same amount sounds broken (drums especially). This rescues the critical hip-hop (85) + pop (120) pairing (41% gap) that the BPM sanity check cannot solve. Note the stretch in the explanation.
         - > 50%: skip tempo matching entirely — stretching vocals beyond 50% is unintelligible. Use alternative techniques from the LLM's arrangement (a cappella over breakdowns, loop-based phrases). Let LLM explain why.
         - Use Rubber Band R3 engine via `--engine finer` flag (requires Rubber Band v3.x; fall back to `--fine` for v2.x)
         - **Default tempo target: instrumental source song's tempo** (not average). When `tempo_source` is `"average"`, stretch both songs toward the midpoint only if the gap is <15% (where splitting the stretch produces less artifact than putting it all on one side). For gaps >15%, always stretch vocals toward the instrumental's tempo. Rationale: drums sound unnatural when stretched; vocals are more tolerant; the instrumental is the rhythmic foundation.
       - **Key matching** (if `key_source` != `"none"`):
         - **Key distance algorithm** (concrete implementation):
           1. Normalize both keys to relative major (e.g., A minor → C major, F# minor → A major)
           2. Compute chromatic semitone interval: `interval = (target_root - source_root) % 12`
           3. Choose minimal shift: `shift = interval if interval <= 6 else interval - 12`
           4. **Harmonic compatibility gate**: Before applying, compute circle-of-fifths distance between the normalized keys. If distance is 0 or 1 (same key, relative major/minor, or a perfect fifth apart), skip key matching entirely — these keys are already harmonically compatible without shifting.
           5. Handle enharmonic equivalence (F# = Gb) by mapping to integer pitch classes before comparison.
         - Cap at +/- 4 semitones for vocal stems, +/- 5 for instruments
         - **Enable formant preservation for vocal stems**: `--formant` flag — prevents chipmunk/demonic artifacts. Only include when pitch shift is non-zero.
         - If shift would exceed cap, skip key matching for that stem
         - **Key confidence gate**: If `key_confidence` < 0.45 for either song, skip key matching entirely. An unreliable key detection + pitch shift makes things worse than no shift at all. The threshold is 0.45 (not lower) because essentia reports 0.4-0.6 confidence for wrong keys ~20-30% of the time — at 0.3, the system pitch-shifts based on detections that are correct only ~50% of the time, producing a *negative* expected value. At 0.45, accuracy is ~65-70%, making the expected value of shifting slightly positive. Consider a two-tier system post-MVP: 0.45+ = shift normally, 0.3-0.45 = shift by half the computed distance (hedging), < 0.3 = skip.
       - **Call the `rubberband` CLI directly via subprocess** (not pyrubberband's Python API, which constructs CLI commands internally in undocumented ways that make injecting `--pitch` via rbargs fragile):
         ```python
         import subprocess, soundfile as sf, numpy as np

         def rubberband_process(audio: np.ndarray, sr: int, tempo_ratio: float,
                                semitones: float = 0, is_vocal: bool = False) -> np.ndarray:
             """Single-pass tempo + pitch via rubberband CLI. Guaranteed single invocation."""
             in_path = tmp_dir / "rb_in.wav"
             out_path = tmp_dir / "rb_out.wav"
             sf.write(in_path, audio, sr)

             cmd = ["rubberband", "-t", str(tempo_ratio)]
             if semitones != 0:
                 cmd += ["-p", str(semitones)]
             cmd += ["--engine", "finer"]  # R3 engine (v3.x); use --fine for v2.x
             if is_vocal and semitones != 0:
                 cmd += ["--formant"]
             cmd += [str(in_path), str(out_path)]

             subprocess.run(cmd, check=True, capture_output=True, timeout=120)
             result, _ = sf.read(out_path)
             return result
         ```
         This calls the `rubberband` CLI once per stem with both `-t` (tempo) and `-p` (pitch) flags in a single pass. Add a startup check that verifies `rubberband --version` returns v3.x; log a warning and fall back to `--fine` if only v2.x is available.
    4. **Cross-song level matching**:
       - Stems from different songs can have wildly different absolute loudness out of the separator. Before any mixing, measure the LUFS of the vocal stem and the summed instrumental stems using `pyloudnorm`.
       - **Silence/low-energy guard**: `pyloudnorm` returns `-inf` for silent or near-silent audio, which would corrupt gain calculations with `inf`/NaN values. Guard before computing gain offset:
         ```python
         LUFS_FLOOR = -40.0  # Below this, the stem is effectively silence/noise
         vocal_lufs = meter.integrated_loudness(vocal_audio)
         instrumental_lufs = meter.integrated_loudness(instrumental_sum)

         if vocal_lufs < LUFS_FLOOR or instrumental_lufs < LUFS_FLOOR:
             # One stem is essentially silence — skip cross-song level matching.
             # Log a warning: likely means separation failed or the song has no vocals.
             logger.warning(f"Skipping level matching: vocal={vocal_lufs:.1f}, instrumental={instrumental_lufs:.1f}")
         else:
             vocal_offset_db = compute_vocal_offset(vocal_audio, instrumental_sum, sr)
             target_vocal_lufs = instrumental_lufs + vocal_offset_db
             gain_db = target_vocal_lufs - vocal_lufs
             gain_db = np.clip(gain_db, -12.0, 12.0)  # Safety cap prevents extreme amplification
             vocal_audio = vocal_audio * (10 ** (gain_db / 20.0))
         ```
       - **Spectral-density-adaptive vocal offset** (replaces fixed +3 dB): The correct vocal-to-instrumental balance varies by 6-10 dB across genre contexts (+1-2 dB for sparse acoustic, +5-8 dB for dense EDM/rock). A fixed offset is wrong for most pairings. Instead, measure the instrumental's spectral density in the vocal frequency band (200 Hz - 5 kHz) and adapt:
         ```python
         from scipy.signal import welch

         def compute_vocal_offset(vocal_audio: np.ndarray, instrumental_audio: np.ndarray,
                                  sr: int) -> float:
             """Compute dB offset for vocals based on instrumental spectral density in vocal range."""
             freqs, psd_inst = welch(instrumental_audio.mean(axis=-1) if instrumental_audio.ndim == 2
                                     else instrumental_audio, fs=sr, nperseg=4096)
             vocal_band = (freqs >= 200) & (freqs <= 5000)
             inst_vocal_band_energy = np.mean(psd_inst[vocal_band])
             inst_full_energy = np.mean(psd_inst)
             # How much of the instrumental's energy competes with the vocal
             density_ratio = inst_vocal_band_energy / (inst_full_energy + 1e-10)
             # Map: sparse instrumental (low ratio) = +1-2 dB, dense = +5-8 dB
             offset_db = np.interp(density_ratio, [0.1, 0.3, 0.5, 0.7], [1.0, 3.0, 5.0, 7.0])
             return float(np.clip(offset_db, 1.0, 8.0))
         ```
         This is ~15 lines, uses only scipy (already a transitive dependency via librosa), and directly measures what determines the correct balance: how much the instrumental competes with the vocal in its frequency range. No genre metadata needed. The LLM's `stem_adjustments` remain as a user-directed override on top of this automatic baseline.
    5. **LLM volume adjustments**:
       - Apply the LLM's `volume_db` adjustments from `stem_adjustments` on top of the cross-song balance
       - These are relative tweaks (e.g., +3dB on bass) that the user requested via prompt
    6. **Post-stretch duration alignment**: After tempo stretching, the vocal layer and instrumental layer will be different lengths (the LLM's time ranges are in the original tempo domain, and each song is stretched by a different ratio). Truncate both layers to the length of the shorter one, then apply a fade-out to the last 3 seconds of the truncated point. This prevents vocals playing over silence or instrumentals continuing after vocals end.
       ```python
       min_length = min(len(vocal_audio), len(instrumental_audio))
       vocal_audio = vocal_audio[:min_length]
       instrumental_audio = instrumental_audio[:min_length]
       # Fade-out is applied later (step 10) to the final mix after LUFS + peak limiting
       ```
    7. **Mix**: Vocals (from vocal_source song) + drums/bass/other (from the other song). Sum all processed stems in numpy (32-bit float) rather than pydub overlay — avoids pydub's internal 16-bit clipping during summation.
    **Note:** Frequency management (cross-song high/low-pass filtering) is deferred to post-MVP. In the vocals+instrumentals model, all instrumental stems come from the same song, so they're already mixed well together. Only the vocal-over-instrumental balance matters, which LUFS normalization + volume adjustments handle.
    8. **LUFS loudness normalization (on the final mix)**:
       - Measure LUFS of the summed mix with `pyloudnorm`
       - Normalize to target LUFS (default -14.0, configurable)
       - **Important:** Normalize the final mix, NOT individual stems before summing. Per-stem normalization to -14 LUFS causes the sum to land around -11 LUFS, forcing the peak limiter to crush the audio and produce audible pumping/distortion.
    9. **Peak limiter**: After LUFS normalization, apply soft-clip limiting at -1.0 dBTP ceiling to prevent clipping during MP3 encoding while preserving average loudness:
       ```python
       ceiling = 10 ** (-1.0 / 20.0)  # -1.0 dBTP ≈ 0.891
       peak = np.max(np.abs(mixed))
       if peak > ceiling:
           # Soft-clip via tanh saturation — tames peaks without pulling down the entire mix.
           # Global gain reduction would cause 3-6 dB loudness drift on typical remix content
           # (stems from different songs have different dynamic ranges = transient spikes).
           mixed = np.tanh(mixed / ceiling) * ceiling
       ```
       **Why soft-clip, not global gain reduction:** Global gain reduction (`mixed * (ceiling / peak)`) pulls the *entire* mix down to tame a single transient peak. When remix stems have peaks 6-10 dB above average (common), this produces a quiet, thin mix that drifts far below the -14 LUFS target. `np.tanh` saturation gently rounds peaks above the ceiling while leaving the rest of the signal untouched — average loudness is preserved. Post-MVP: a true lookahead limiter would be even cleaner, but `np.tanh` is a massive improvement over global gain reduction for ~3 lines of code.
    10. **Fade-in/fade-out**: Apply 2-second fade-in and 3-second fade-out in numpy (not pydub) to avoid precision loss from pydub's 16-bit internal representation.
    11. **Export**: Write 32-bit float WAV via `soundfile`, then encode to MP3 (320kbps) via `ffmpeg` subprocess. **Do NOT use pydub for export** — pydub's `AudioSegment` quantizes to 16-bit integers internally, destroying the 32-bit float headroom from LUFS normalization and soft-clip limiting (can cause hard clipping of intersample peaks):
       ```python
       # Write float WAV (preserves full precision)
       sf.write(tmp_wav_path, mixed, sr, subtype="FLOAT")
       # Encode to MP3 via ffmpeg (reads float WAV, encodes lossily)
       subprocess.run(["ffmpeg", "-y", "-i", str(tmp_wav_path), "-codec:a", "libmp3lame",
                        "-b:a", "320k", str(output_path)], check=True, capture_output=True, timeout=120)
       ```
  - Returns path to rendered remix file
- [ ] Write unit test with pre-separated test stems

**Orthogonality:** Processing takes a plan + stems + metadata and produces an audio file. Doesn't know about uploads, LLMs, or the web layer.

**Files created:** `src/musicmixer/services/processor.py`, `tests/test_processor.py`

---

### Phase B7: Pipeline Orchestrator + SSE Progress + TTL

**Goal:** Wire all services into an end-to-end pipeline with real-time progress updates and session lifecycle management.

- [ ] Create `services/pipeline.py`:
  - `run_pipeline(session_id: str, song_a_path: Path, song_b_path: Path, prompt: str, event_queue: queue.Queue) -> None`
  - **Sets session attributes directly** in addition to pushing queue events. The queue may have no consumer (SSE client disconnected), so the pipeline must always update `session.status`, `session.remix_path`, and `session.explanation` on the `SessionState` object at key transitions (processing start, completion, error). The status endpoint reads these attributes, not the queue.
  - Pushes `ProgressEvent` dicts to the queue at each step:
    ```python
    class ProgressEvent(BaseModel):
        step: str      # "separating" | "analyzing" | "interpreting" | "processing" | "complete" | "error"
        detail: str    # Human-readable description
        progress: float  # 0.0 to 1.0
        explanation: str | None = None  # Present when step == "complete"
    ```
  - Progress sequence:
    ```
    {"step": "separating", "detail": "Extracting stems from Song A...", "progress": 0.10}
    {"step": "separating", "detail": "Extracting stems from Song B...", "progress": 0.30}
    {"step": "analyzing", "detail": "Detecting tempo and key...", "progress": 0.45}
    {"step": "interpreting", "detail": "Planning your remix...", "progress": 0.55}
    {"step": "processing", "detail": "Matching tempo...", "progress": 0.65}
    {"step": "processing", "detail": "Normalizing loudness...", "progress": 0.75}
    {"step": "processing", "detail": "Mixing stems...", "progress": 0.85}
    {"step": "processing", "detail": "Rendering final mix...", "progress": 0.95}
    {"step": "complete", "detail": "Remix ready!", "progress": 1.0, "explanation": "..."}
    ```
  - On any error: push `{"step": "error", "detail": "user-friendly message", "progress": 0}` and clean up session files
  - **Processing lock is acquired by the POST handler** (not the pipeline) and released via a `pipeline_wrapper` function's `try/finally` block (see Phase B2). The pipeline itself (`run_pipeline`) does not receive or manage the lock — the wrapper keeps it pure. `finally` executes even on `BaseException` subclasses (`KeyboardInterrupt`, `SystemExit`), preventing orphaned locks on thread death.
    ```python
    def pipeline_wrapper(session_id: str, ...):
        try:
            session.status = "processing"
            run_pipeline(session_id, ...)
            # Pipeline sets session.status, session.remix_path, session.explanation
            # directly before pushing the complete event to the queue.
        except BaseException as e:
            # Push error event so SSE client learns about failure
            try:
                session.status = "error"
                session.events.put_nowait({"step": "error", "detail": str(e), "progress": 0})
            except Exception:
                pass  # Best effort — don't mask the original exception
            raise
        finally:
            # Guard against double-release: release() on an unlocked Lock raises RuntimeError.
            if processing_lock.locked():
                processing_lock.release()
    ```
- [ ] Create SSE endpoint `GET /api/remix/{session_id}/progress`:
  - Returns `StreamingResponse` with `text/event-stream` content type
  - On connect: check session status. If already complete, send final event immediately. If processing, send latest event then continue streaming.
  - Async generator reads from `session.events` queue via `run_in_executor`
  - No-event timeout: if no event arrives within 30 seconds, send a keepalive `data:` event (`{"step":"keepalive"}`). Must be a `data:` event, not an SSE comment — `EventSource.onmessage` does not fire for comments, so the frontend timeout would misfire during long steps.
- [ ] Create status endpoint `GET /api/remix/{session_id}/status`:
  - Returns current session state as JSON (not SSE)
  - Used by frontend for quick reconnection check after page refresh
  - **Note:** `queued` is an internal status. The status endpoint maps it to `processing` with `progress: 0` before returning — the frontend never sees `queued`.
  - **Explanation data flow:** Pipeline sets `session.explanation = remix_plan.explanation` before pushing the `complete` event. Status endpoint reads `session.explanation` directly (not from the queue). This guarantees the explanation is available even if the SSE `complete` event was consumed by a different client or never consumed.
  - Response: `{"status": "processing", "progress": 0, "detail": "Starting..."}` or `{"status": "processing", "progress": 0.45, "detail": "..."}` or `{"status": "complete", "remix_url": "...", "explanation": "..."}` or `{"status": "error", "detail": "..."}`
- [ ] Create audio endpoint `GET /api/remix/{session_id}/audio`:
  - Serves the rendered remix MP3 file via `FileResponse`
  - Returns 404 if remix doesn't exist or has expired
- [ ] Implement TTL cleanup via FastAPI lifespan:
  ```python
  @asynccontextmanager
  async def lifespan(app: FastAPI):
      cleanup_task = asyncio.create_task(cleanup_loop())
      yield
      cleanup_task.cancel()
      executor.shutdown(wait=False, cancel_futures=True)  # Don't hang on SIGTERM

  async def cleanup_loop():
      while True:
          await asyncio.sleep(settings.cleanup_interval_seconds)
          try:
              now = datetime.now()
              with sessions_lock:
                  expired = [
                      (sid, s) for sid, s in sessions.items()
                      if s.status != "processing"
                      and (now - s.created_at).total_seconds() > settings.remix_ttl_seconds
                  ]
                  for sid, _ in expired:
                      del sessions[sid]
              # Delete files OUTSIDE the lock, in a thread to avoid blocking the event loop.
              # shutil.rmtree can take 100s of ms for large session dirs (~500MB).
              for sid, _ in expired:
                  # Delete ALL session directories: uploads, stems, AND remixes (~500MB/session)
                  for subdir in ("uploads", "stems", "remixes"):
                      session_dir = settings.data_dir / subdir / sid
                      await asyncio.to_thread(shutil.rmtree, session_dir, ignore_errors=True)
              logger.info(f"Cleanup: removed {len(expired)} sessions, "
                          f"{len(sessions)} active, lock_held={processing_lock.locked()}")
          except Exception:
              logger.exception("Cleanup cycle failed, will retry next interval")
  ```
  - Cleanup uses `sessions_lock` when mutating the sessions dict (thread-safe with pipeline)
  - File deletion runs via `asyncio.to_thread` to avoid blocking the event loop
  - Cleanup skips sessions that are currently in "processing" status
  - Log session count and lock state each cycle for operational visibility
- [ ] Error handling: any service raising a `MusicMixerError` is caught by the pipeline, logged, and emitted as an SSE error event

**Files created:** `src/musicmixer/services/pipeline.py`
**Files modified:** `src/musicmixer/api/remix.py`, `src/musicmixer/main.py`

---

### Backend Phase Summary

| Phase | Depends On | Can Parallelize With | Estimated Complexity |
|-------|-----------|---------------------|---------------------|
| B1: Scaffolding | — | F1 (frontend scaffolding) | Low |
| B2: Upload | B1 | F2 (upload UI) | Low-Medium (validation) |
| B3: Separation | B1 | B4, F2, F3 | Medium (ML dependencies) |
| B4: Analysis | B1 | B3, B5, F3 | Low-Medium (BPM sanity check) |
| B5: Interpreter | B1 | B3, B4 | Medium (prompt engineering) |
| B6: Processing | B3, B4, B5 | F4 | High (audio engineering) |
| B7: Orchestrator | B2-B6 | F4, F5 | Medium (integration) |

### Backend Dependencies

**Python packages:**
```
fastapi
uvicorn[standard]
python-multipart
pydantic-settings
structlog               # logging
audio-separator         # stem separation (brings PyTorch)
librosa                 # BPM detection, audio loading
essentia                # key detection
pyrubberband            # tempo/key matching
pydub                   # audio mixing, fades, export
soundfile               # WAV I/O
pyloudnorm              # LUFS loudness normalization
numpy                   # float mixing, peak limiter (also a transitive dep of librosa)
scipy                   # spectral ducking bandpass filter (also a transitive dep of librosa, but explicit is safer)
anthropic               # LLM calls
```

**System dependencies (document in CLAUDE.md and README):**
```
ffmpeg                  # required by pydub
rubberband              # required by pyrubberband
libsndfile              # required by soundfile
```

**PyTorch installation (pin in pyproject.toml):**
```
# macOS development (CPU/MPS):
torch --extra-index-url https://download.pytorch.org/whl/cpu

# Production GPU (CUDA 12.1):
torch --extra-index-url https://download.pytorch.org/whl/cu121
```

---

## Frontend Execution Plan

> All frontend work happens in `frontend/`. React + TypeScript + Vite with Bun.

### Phase F1: Project Scaffolding + App Shell

**Goal:** Runnable React app with layout, types, API client, and state machine.

- [ ] Initialize project: `bun create vite frontend --template react-ts`
- [ ] Install Tailwind CSS: `bun add -d tailwindcss @tailwindcss/vite`
  - Verify Preflight (CSS reset) is active for consistent cross-browser rendering
- [ ] Create project structure:
  ```
  frontend/
  ├── package.json
  ├── CLAUDE.md
  ├── vite.config.ts
  ├── tsconfig.json
  ├── index.html
  ├── public/
  └── src/
      ├── main.tsx
      ├── App.tsx                 # Thin shell — renders RemixSession
      ├── index.css
      ├── components/
      │   └── RemixSession.tsx    # State machine orchestrator
      ├── hooks/
      ├── api/
      │   └── client.ts           # API functions
      └── types/
          └── index.ts            # Shared TypeScript types
  ```
- [ ] Set up Vite proxy to backend (`/api` → `http://localhost:8000`)
- [ ] Define TypeScript types in `types/index.ts`:
  ```typescript
  // API contract types
  export type ProgressStep = 'separating' | 'analyzing' | 'interpreting' | 'processing' | 'complete' | 'error' | 'keepalive';

  export type ProgressEvent = {
    step: ProgressStep;
    detail: string;
    progress: number;
    explanation?: string;
  };

  export type CreateRemixResponse = {
    session_id: string;
  };

  export type SessionStatus = {
    status: 'processing' | 'complete' | 'error';  // Backend maps 'queued' → 'processing' before returning
    progress?: number;
    detail?: string;
    remix_url?: string;
    explanation?: string;
  };

  // Compatibility analysis types (QR-3)
  export type CompatibilityLevel = 'great' | 'good' | 'challenging' | 'tough';

  export type CompatibilityResult = {
    level: CompatibilityLevel;
    message: string;        // Plain-language, user-facing (no raw numbers)
    detail?: string;        // Optional technical detail for power users
  };

  export type SongAnalysis = {
    bpm: number;
    key: string;
    scale: string;
  };

  export type AnalyzeResponse = {
    song_a: SongAnalysis;
    song_b: SongAnalysis;
    compatibility: CompatibilityResult;
  };

  // App state (discriminated union)
  export type AppState =
    | { phase: 'idle'; songA: File | null; songB: File | null; prompt: string; analyzing: boolean; compatibility: CompatibilityResult | null; analyzeError: string | null }
    | { phase: 'uploading'; songA: File; songB: File; prompt: string; uploadProgress: number }
    | { phase: 'processing'; sessionId: string; progress: ProgressEvent }
    | { phase: 'ready'; sessionId: string; explanation: string }
    | { phase: 'error'; message: string; songA: File | null; songB: File | null; prompt: string };

  // Reducer actions (discriminated union — prevents illegal state transitions at compile time)
  export type AppAction =
    | { type: 'SET_SONG_A'; file: File | null }
    | { type: 'SET_SONG_B'; file: File | null }
    | { type: 'SET_PROMPT'; prompt: string }
    | { type: 'ANALYZE_STARTED' }
    | { type: 'ANALYZE_COMPLETE'; result: CompatibilityResult }
    | { type: 'ANALYZE_FAILED'; reason: 'transient' }       // Network/timeout — silently ignored
    | { type: 'ANALYZE_FAILED'; reason: 'invalid_file'; detail: string }  // 422 — show inline warning near upload zone
    | { type: 'START_UPLOAD' }
    | { type: 'UPLOAD_PROGRESS'; percent: number }
    | { type: 'UPLOAD_SUCCESS'; sessionId: string }
    | { type: 'PROGRESS_EVENT'; event: ProgressEvent }
    | { type: 'REMIX_READY'; explanation: string }
    | { type: 'ERROR'; message: string }
    | { type: 'RESET' };

  // API error types (discriminated union for createRemix rejection)
  export type CreateRemixError =
    | { type: 'network' }
    | { type: 'timeout' }
    | { type: 'http'; status: number; body: { detail: string } };
  ```
- [ ] Define API client in `api/client.ts`:
  ```typescript
  export async function createRemix(
    songA: File, songB: File, prompt: string,
    onUploadProgress?: (pct: number) => void
  ): Promise<CreateRemixResponse>
  // Uses XMLHttpRequest for upload progress support (fetch doesn't support it reliably)
  // XHR error handling — rejects with CreateRemixError:
  //   - onerror (network failure): reject with { type: 'network' }
  //   - ontimeout: reject with { type: 'timeout' }
  //   - onload with status >= 400: reject with { type: 'http', status, body: JSON.parse(xhr.responseText) }
  //   - onload with status 200: resolve with parsed JSON response

  export async function analyzeCompatibility(songA: File, songB: File): Promise<AnalyzeResponse>
  // POST /api/analyze — lightweight compatibility check (~2-3s)
  // Called automatically after both songs are uploaded (non-blocking)
  // Uses fetch (no upload progress needed — files already uploaded for this endpoint)

  export function connectProgress(sessionId: string): EventSource
  // Returns native EventSource connected to /api/remix/{sessionId}/progress

  export async function getSessionStatus(sessionId: string): Promise<SessionStatus>
  // GET /api/remix/{sessionId}/status — for reconnection after refresh

  export function getAudioUrl(sessionId: string): string
  // Returns /api/remix/{sessionId}/audio URL string
  ```
- [ ] Design mobile-first single-column layout:
  - Upload zones stack vertically on screens < 768px
  - Centered max-width container on desktop
- [ ] Create `RemixSession.tsx` — the state machine orchestrator:
  - Owns the `useReducer` with `AppState` and action types
  - Renders the correct child component based on `state.phase`
  - Handles `sessionStorage` persistence of `sessionId` (write on entering 'processing', clear on entering 'idle')
  - On mount: check `sessionStorage` for active session → call `getSessionStatus()` → map to state:
    - `status: "processing"` → enter `processing` phase, reopen EventSource (note: backend maps internal `queued` status to `processing` so the frontend never sees `queued`)
    - `status: "complete"` → enter `ready` phase with `explanation` from response (skip SSE entirely)
    - `status: "error"` → enter `error` phase (note: file references are lost after refresh since `File` objects are not serializable to sessionStorage — `songA`/`songB` will be `null`, prompt will be empty. User must re-select files.)
    - 404 → session expired or unknown, reset to `idle`
- [ ] Add tagline/heading above the form: "Upload two songs. Describe your mashup. AI does the rest."
- [ ] Add `CLAUDE.md` for frontend repo

**Files created:** Standard Vite scaffold + `CLAUDE.md`, `src/api/client.ts`, `src/types/index.ts`, `src/components/RemixSession.tsx`

---

### Phase F2: Song Upload UI

**Goal:** Two upload areas where users can add their songs.

- [ ] Create `components/SongUpload.tsx`:
  - Drag-and-drop zone with native `<input type="file">` as the accessible fallback
  - `accept=".mp3,.wav,audio/mpeg,audio/wav"` attribute to filter file picker
  - Visually-hidden input (sr-only), not `display: none` (preserves keyboard accessibility)
  - Shows file name and size after selection
  - Visual indicator for "Song A" and "Song B" slots
  - On mobile: stacks vertically; on desktop: side by side
- [ ] Compose two `SongUpload` instances directly inside `RemixForm` (no separate `UploadSection` wrapper — not needed at this scale)
- [ ] Client-side validation: file type, max 50MB. Show inline error for rejected files.
- [ ] Disabled state when processing is in progress
- [ ] **Compatibility analysis trigger** (QR-3): When both songs are selected (both `songA` and `songB` are non-null), automatically dispatch `ANALYZE_STARTED` and call `analyzeCompatibility(songA, songB)`. On success, dispatch `ANALYZE_COMPLETE`. On failure, dispatch `ANALYZE_FAILED` (silently — analysis is best-effort). If the user replaces a song, clear existing compatibility data and re-trigger analysis.
- [ ] Create `components/CompatibilitySignal.tsx`:
  - Renders the `CompatibilityResult` as an inline element between the upload zones and the prompt input
  - Uses color accent: green for "great", neutral/amber for "good", orange for "challenging"/"tough"
  - Shows the plain-language `message` (never raw BPM/key values)
  - Optional collapsed "Details" disclosure for power users (shows `detail` field with raw data)
  - While analyzing: shows subtle "Checking compatibility..." text
  - If analysis failed or not yet run: renders nothing (no empty state)

**Files created:** `src/components/SongUpload.tsx`, `src/components/CompatibilitySignal.tsx`

---

### Phase F3: Prompt Input + Form Submission

**Goal:** Text field for the remix description. Wire up the full submission flow.

- [ ] Create `components/PromptInput.tsx`:
  - Textarea for longer prompts
  - Example prompts shown as **static text above the input** (not as placeholder — placeholders vanish on focus):
    - "Put the vocals from Song A over Song B's beat"
    - "Song B vocals over Song A instrumentals, boost the bass"
    - "Slow it down and layer the singing over the other track"
  - Submit button ("Create Remix")
  - Button disabled when: no songs uploaded, empty prompt, or currently processing
  - **Button disables immediately on click** (before server response) to prevent double submission
- [ ] Create `components/RemixForm.tsx` composing uploads + prompt + submit
- [ ] Wire submission flow:
  1. Transition to `uploading` state
  2. Call `createRemix()` with upload progress callback
  3. On success: transition to `processing` state with session ID
  4. On error (413, 422, 500, network): transition to `error` state with message
- [ ] Upload progress: show percentage during the upload phase (can be 80+ seconds for 100MB on slow connections)
  - Use `XMLHttpRequest.upload.onprogress` in `api/client.ts` (more reliable than fetch for upload progress)

**Error taxonomy (client-side):**

| Error | Response | User Sees |
|-------|----------|-----------|
| Network failure | No response | "Upload failed. Check your connection and try again." |
| 413 File too large | Server rejects | "File too large. Maximum 50MB per song." |
| 422 Validation error | Server message | Display server's error message |
| 429 Server busy | Processing lock held | "Another remix is being created. Please wait and try again." |
| 500 Server error | Generic | "Something went wrong. Please try again." |

**Files created:** `src/components/PromptInput.tsx`, `src/components/RemixForm.tsx`

---

### Phase F4: Progress Display

**Goal:** Real-time progress updates while the remix is being created.

- [ ] Create `hooks/useRemixProgress.ts`:
  - Uses native `EventSource` API (sufficient for MVP — no auth headers needed)
  - **Data flow pattern**: The hook accepts a `dispatch` function from the parent `useReducer` and dispatches actions directly (`PROGRESS_EVENT`, `REMIX_READY`, `ERROR`). All progress state flows through the reducer — the hook does NOT maintain its own progress state. The hook's return is limited to `{ isConnected: boolean }`. `RemixSession` reads `state.progress` from the reducer to render `ProgressDisplay`.
  - **Cleanup on unmount**: `.close()` the EventSource in the hook's cleanup function
  - **No-event timeout**: If no event arrives within 60 seconds, dispatch `ERROR` action ("Processing is taking longer than expected")
  - **Tab backgrounding**: Listen to `document.visibilitychange`. On tab refocus, check connection state. If dead, reconnect by calling `getSessionStatus()` first then reopening EventSource.
  - **Reconnection strategy (MVP)**: On disconnect, call `getSessionStatus()` to get current state. If complete, dispatch `REMIX_READY`. If processing, reopen EventSource (will receive current event on connect thanks to backend design). Accept possible duplicate events — harmless for a progress bar.
  - **Max-retry/give-up policy**: After 5 consecutive `onerror` events without any successful `onmessage`, close the EventSource and dispatch `ERROR` with message "Lost connection to the server. Please try again." This prevents frozen progress UI when the server is permanently down.
  - **Keepalive events**: When `step === "keepalive"`, reset the no-event timeout but do NOT dispatch `PROGRESS_EVENT` or update the progress bar. Keepalives have `progress: -1` — use this as a sentinel to distinguish them from real events.
  - **Error events**: Parse `step === "error"` events, dispatch `ERROR` action with the detail message
  - **Malformed events**: Log and ignore. Don't crash.
- [ ] Create `components/ProgressDisplay.tsx`:
  - Progress bar showing percentage
  - Step description with human-readable text
  - Animated/loading indicator
  - **Cancel button**: Closes SSE connection, resets to idle state. Backend continues processing (cleaned up by TTL). Simple client-side reset only.
  - Error state: shows error message + "Try Again" button (resets to idle)
- [ ] Loading micro-state: Show progress bar at 0% with "Starting..." immediately on entering processing phase (before first SSE event arrives, typically 1-2 seconds)

**Files created:** `src/hooks/useRemixProgress.ts`, `src/components/ProgressDisplay.tsx`

---

### Phase F5: Audio Player

**Goal:** Play back the rendered remix.

- [ ] Create `components/RemixPlayer.tsx`:
  - HTML5 `<audio>` element with native controls
  - `src` points to `getAudioUrl(sessionId)`
  - Shows the LLM's explanation of what it did
  - "Create New Remix" button → resets to idle (with brief confirmation since current remix will be replaced)
  - Expiration notice ("This remix will expire in ~3 hours")
- [ ] Handle expired remix: if `<audio>` fires error event or status endpoint returns 404, show friendly message and reset to idle

**Files created:** `src/components/RemixPlayer.tsx`
**Files modified:** `src/components/RemixSession.tsx`

---

### Frontend Phase Summary

| Phase | Depends On | Can Parallelize With | Estimated Complexity |
|-------|-----------|---------------------|---------------------|
| F1: Scaffolding + Shell | — | B1 | Medium (state machine, types, API client) |
| F2: Upload UI | F1 | B2, B3, B4, F3 | Low |
| F3: Prompt + Submission | F1 | B3, B4, B5, F2 | Low-Medium (upload progress, error handling) |
| F4: Progress | F1 | B7 | Medium (SSE, reconnection, tab handling) |
| F5: Player | F1 | B6, B7 | Low |

---

## Integration Plan

### API Contract

```
POST /api/remix
  Body: multipart/form-data
    - song_a: File (MP3/WAV, max 50MB)
    - song_b: File (MP3/WAV, max 50MB)
    - prompt: string
  Response 200: { "session_id": "uuid" }
  Response 413: File too large
  Response 422: { "detail": "Validation error message" }
  Response 429: { "detail": "Server is busy processing another remix" }

GET /api/remix/{session_id}/progress
  Response: SSE stream (text/event-stream)
    data: {"step":"separating","detail":"Extracting stems from Song A...","progress":0.10}
    ...
    data: {"step":"complete","detail":"Remix ready!","progress":1.0,"explanation":"..."}
    OR
    data: {"step":"error","detail":"Could not match these songs...","progress":0}
  On connect when already complete: sends final event immediately

GET /api/remix/{session_id}/status
  Note: internal "queued" status is mapped to "processing" before returning
  Response 200: {"status":"processing","progress":0,"detail":"Starting..."}
           OR: {"status":"processing","progress":0.45,"detail":"Matching tempo..."}
           OR: {"status":"complete","remix_url":"/api/remix/{id}/audio","explanation":"..."}
           OR: {"status":"error","detail":"..."}
  Response 404: Session not found or expired

GET /api/remix/{session_id}/audio
  Response 200: audio/mpeg (MP3 file)
  Response 404: Remix not found or expired

POST /api/analyze
  Body: multipart/form-data
    - song_a: File (MP3/WAV, max 50MB)
    - song_b: File (MP3/WAV, max 50MB)
  Response 200: {
    "song_a": { "bpm": 95.0, "key": "G", "scale": "minor" },
    "song_b": { "bpm": 92.0, "key": "A", "scale": "minor" },
    "compatibility": {
      "level": "great" | "good" | "challenging" | "tough",
      "message": "These songs should blend really well together.",
      "detail": "Similar tempo, compatible keys."
    }
  }
  Response 422: { "detail": "Invalid audio file: song_a" }
    Returned when a file is missing, not a valid audio file, or cannot be decoded.
    The `detail` field identifies which song failed (song_a, song_b, or both).
  Response 408: { "detail": "Analysis timed out" }
    Returned if analysis exceeds 10 seconds (e.g., very large files on CPU).
  Note: If BPM or key detection fails for one song (e.g., very short file, pure noise),
    return 200 with best-effort partial data — set failed fields to null and degrade
    compatibility level to "challenging" with message explaining limited analysis.
  Note: Does NOT acquire the processing_lock — runs independently of remix processing.
    Uses a separate lightweight thread pool (or async event loop) for parallel per-song analysis.
    If a new /api/analyze request arrives while one is in-flight, the frontend should
    abort the previous request (AbortController) to avoid stale results.
  Note: Runs lightweight analysis (~2-3 seconds). Does NOT trigger the full pipeline.
  Triggers automatically after both songs are uploaded (frontend calls this).
  The prompt input is NOT blocked during analysis — it runs in the background.

GET /health
  Response 200: {"status":"ok"}
```

### Pydantic Response Models

```python
class RemixResponse(BaseModel):
    session_id: str

class ProgressEvent(BaseModel):
    step: str
    detail: str
    progress: float
    explanation: str | None = None

class SessionStatusResponse(BaseModel):
    status: str          # "processing" | "complete" | "error" (internal "queued" mapped to "processing")
    progress: float | None = None
    detail: str | None = None
    remix_url: str | None = None
    explanation: str | None = None

class SongAnalysis(BaseModel):
    bpm: float | None = None       # None if detection failed for this song
    key: str | None = None         # None if detection failed
    scale: str | None = None       # None if detection failed

class CompatibilityResult(BaseModel):
    level: str               # "great" | "good" | "challenging" | "tough"
    message: str             # Plain-language, user-facing (no raw numbers)
    detail: str | None = None  # Optional technical detail for power users

class AnalyzeResponse(BaseModel):
    song_a: SongAnalysis
    song_b: SongAnalysis
    compatibility: CompatibilityResult

class HealthResponse(BaseModel):
    status: str

class ErrorResponse(BaseModel):
    detail: str
```

### Integration Testing Plan

**Test fixtures:** Ship 2-3 short (5-second) public domain audio files in `backend/tests/fixtures/`.

**Automated integration tests (in backend):**
1. Health check returns 200
2. Upload with invalid file type returns 422
3. Upload with oversized file returns 413
4. Upload with valid files returns session_id
5. SSE endpoint streams progress events in correct order
6. SSE endpoint sends current state on reconnect
7. Status endpoint reflects pipeline state
8. Audio endpoint serves MP3 after processing completes
9. Audio endpoint returns 404 for expired/nonexistent session
10. Second concurrent remix request returns 429
11. TTL cleanup removes expired sessions

**Manual integration tests:**
12. Frontend uploads → receives session ID → shows progress → plays audio
13. Page refresh during processing reconnects successfully
14. Cancel button resets frontend cleanly

---

## Open Questions Resolved

| Question | Resolution |
|----------|------------|
| Stem granularity | 4-stem separation (vocals/drums/bass/other) with htdemucs_ft, but MVP uses a **vocals + instrumentals model**: vocals from one song, all instrumentals from the other. No cross-song stem mixing. 4-stem is still run so LLM can apply per-stem volume adjustments (e.g., "boost bass"). Upgrade path: cross-song stem mixing post-MVP, BS-RoFormer models for better guitar/piano isolation. |
| File format/size limits | MP3 and WAV. Max 50MB per file, max 10 minutes. Validated by parsing the audio, not just MIME type. |
| Processing architecture | Server-side everything. Pipeline runs in ThreadPoolExecutor with threading.Queue for progress events. Single-worker, processing lock, one remix at a time. |
| Processing time | GPU: ~30s total. CPU: ~12-15min total. Progress UI keeps users informed either way. |
| First-time experience | Tagline + static example prompts above the input field + self-explanatory 2-upload + 1-text-field interface. |
| Error handling | Multi-layer: upload validation (413/422), pipeline errors via SSE, LLM fallback to default plan, audio 404 for expired remixes. Full error taxonomy defined. |
| Async model | ThreadPoolExecutor with threading.Queue. Pipeline pushes events, SSE endpoint reads them via async wrapper. |
| Concurrency | Single-worker uvicorn, global processing lock. Second request gets 429. |
| Output length | LLM decides via `duration_seconds` field (default 60-120s, max 180s). LLM also picks start/end times for each song. |
| Section selection | Two levels: (1) LLM picks source regions via `start_time_vocal`/`end_time_vocal`/`start_time_instrumental`/`end_time_instrumental` fields, guided by energy profile data. (2) LLM designs section-based arrangement (QR-1) over the extracted material — `sections: list[Section]` with per-stem gains and transitions. Energy profiles (RMS + onset density in ~8s windows) guide both levels. Post-MVP: add `allin1` package for labeled structural analysis. |
| Loudness matching | Per-stem gains now embedded in `Section.stem_gains` (QR-1) — applied per-section during arrangement rendering. Cross-song level matching (LUFS) applied before arrangement. Final LUFS normalization with `pyloudnorm` to -14 LUFS on the **final summed mix** (not per-stem). Spectral ducking (QR-1) carves frequency space for vocals. |
| Frequency clashing | Deferred to post-MVP. Not needed in vocals+instrumentals model — all instrumental stems come from the same song (already balanced). When cross-song stem mixing is added, will use scipy role-based filtering (high-pass on non-bass, low-pass on competing bass). |
| Build vs buy | Build locally for MVP ($0.003/remix). Fadr API ($0.05/min, includes BPM+key+chords) is the fallback if GPU setup takes >1 day. No API exists for the full prompt-based pipeline — this is genuinely novel. |

---

## Risk Analysis

| Risk | Impact | Mitigation |
|------|--------|------------|
| CPU-only processing is too slow (12+ min) | Users abandon | Deploy with GPU, or set expectations with progress UI. Consider Fadr API as a faster alternative for separation. |
| PyTorch installation issues | Blocks development | Pin exact version + variant in pyproject.toml. Document setup steps. Test on a clean machine early. Consider Docker. |
| Essentia installation fails | Blocks key detection | Pluggable interface: auto-fallback to librosa chromagram. Designed in from the start. |
| BPM half/double-time error (~29% of songs) | Wrong tempo = terrible remix | Expanded BPM reconciliation: 5 interpretations per song (original, half, double, 3/2, 2/3) in 70-180 BPM range. Reduces failure from ~73% to ~53% for cross-genre pairings. |
| Tempo stretch >30% sounds bad | Artifact-ridden output | Vocals-only stretching for 30-50% gaps (vocals tolerate more stretch than instruments). Default to instrumental source tempo. Skip entirely only above 50%. Pre-upload compatibility signal sets expectations. |
| Pitch-shifted vocals sound chipmunk/demonic | Unnatural output | Formant preservation enabled by default for vocal stems. Cap at +/- 4 semitones for vocals. |
| Two tracks layered sound like two tracks playing | Unmusical output | Section-based arrangement (QR-1): LLM-generated section timeline with per-stem gains, beat-aligned transitions, spectral ducking. Vocals+instrumentals model keeps it simple. See QR-1 for full solution. |
| LLM produces bad remix plan | Bad remixes | Default to Sonnet for section-based arrangement quality. Condensed energy data + arrangement templates constrain LLM creative space. Few-shot examples for Haiku/Sonnet reliability. Surgical code-level validation fixes common schema errors. `warnings` field surfaces uncertainty. Deterministic data-driven fallback with `used_fallback` flag and honest user messaging. Explanation shown to user. Structured LLM observability logging for debugging. |
| Disk fills up (~500MB per session) | Server crashes | TTL cleanup every 5 minutes. Log disk usage. Reject new remixes if disk is critically low. |
| Server restart loses all sessions | Active remixes lost, orphaned files on disk | Acceptable for MVP (single user). TTL cleanup will eventually remove orphaned files. Post-MVP: persist session state to SQLite or Redis. |
| Page refresh loses session | User frustration | sessionStorage persistence. Status endpoint for quick reconnection. SSE replays current state on connect. |
| File upload is slow (100MB on slow connection) | Users abandon | Upload progress bar via XMLHttpRequest. Client-side file size validation before upload. |

---

## Quality Risks

Output quality red-flags identified during review. Track status here — update as each is addressed.

### QR-1: "Two songs playing at once" — static overlay lacks arrangement

**Status:** OPEN
**Risk level:** High — users will notice immediately
**Reviews:** `notes/2026-02-23-quality-review-audio.md`, `notes/2026-02-23-quality-review-ux.md`

The current plan renders a flat mix: all stems at full volume from start to finish. Real remixes introduce elements over time (instrumental intro → vocals enter → breakdown → build → outro). Without arrangement, the output sounds like two songs colliding, not a remix.

**Contributing factors:**
- `RemixPlan` is a single snapshot (one set of stem gains for the whole duration) — no concept of sections or timeline
- No beat-aligned transitions — vocals enter at arbitrary timestamps, not on downbeats
- No spectral ducking — vocals and instrumentals compete for the same frequency range (300Hz–3kHz)
- Vocal bleed from stem separation creates "ghost vocals" (Song B's vocals leak through the instrumental stems)
- The vocals+instrumentals constraint is never communicated to the user in the UI

**Solutions — what moves the needle:**

#### 1. Arrangement over time (~100-150 lines of mixer code)

The single highest-impact change. Replace the flat mix with an LLM-generated section timeline. The LLM already decides start/end times — this extends it to decide *arrangement* over time.

**New data structures** (replace the current single-snapshot `RemixPlan`):

```python
@dataclass
class Section:
    label: str              # "intro" | "verse" | "breakdown" | "drop" | "outro"
    start_beat: int         # beat-aligned (not seconds — snapped to grid)
    end_beat: int
    stem_gains: dict[str, float]  # {"vocals": 1.0, "drums": 0.7, "bass": 0.8, "other": 0.6}
    transition_in: str      # "fade" | "crossfade" | "cut"
    transition_beats: int   # length of transition envelope

@dataclass
class RemixPlan:
    vocal_source: str       # "song_a" | "song_b"
    start_time_vocal: float       # Where to start in the vocal source song (seconds, original tempo)
    end_time_vocal: float         # Where to end in the vocal source song
    start_time_instrumental: float  # Where to start in the instrumental source song
    end_time_instrumental: float    # Where to end in the instrumental source song
    # Source time ranges select which region of each song to extract.
    # Named _vocal/_instrumental (not _a/_b) so the schema is self-documenting
    # and the LLM doesn't confuse which song is which.
    sections: list[Section] # Arrangement over the extracted material — replaces static stem_adjustments
    tempo_source: str       # "song_a" | "song_b" | "average"
    key_source: str         # "song_a" | "song_b" | "none"
    explanation: str        # LLM's reasoning (shown to user)
    warnings: list[str]     # LLM-generated caveats: what it couldn't fulfill, prompt ambiguities, etc.
                            # E.g., "Both songs have similar energy — choosing vocals from Song A arbitrarily"
                            # Displayed with distinct styling in the frontend.
    used_fallback: bool = False  # Set True when deterministic fallback was used instead of LLM.
                                 # Frontend displays explanation with warning styling, not success styling.
```

**Example LLM output** for a hip-hop vocals over pop beat remix:

```json
{
  "sections": [
    {"label": "intro",     "start_beat": 0,  "end_beat": 16, "stem_gains": {"vocals": 0.0, "drums": 0.9, "bass": 0.8, "other": 1.0}, "transition_in": "fade", "transition_beats": 4},
    {"label": "verse",     "start_beat": 16, "end_beat": 48, "stem_gains": {"vocals": 1.0, "drums": 0.7, "bass": 0.8, "other": 0.5}, "transition_in": "crossfade", "transition_beats": 4},
    {"label": "breakdown", "start_beat": 48, "end_beat": 64, "stem_gains": {"vocals": 0.8, "drums": 0.0, "bass": 0.6, "other": 0.7}, "transition_in": "crossfade", "transition_beats": 2},
    {"label": "drop",      "start_beat": 64, "end_beat": 96, "stem_gains": {"vocals": 1.0, "drums": 1.0, "bass": 1.0, "other": 0.8}, "transition_in": "cut", "transition_beats": 0},
    {"label": "outro",     "start_beat": 96, "end_beat": 112, "stem_gains": {"vocals": 0.0, "drums": 0.6, "bass": 0.5, "other": 0.8}, "transition_in": "crossfade", "transition_beats": 8}
  ]
}
```

**Mixer logic** — deterministic renderer of the LLM's plan:
1. Convert beat positions to sample positions using the **beat grid** from `AudioMetadata.beat_frames` (NOT constant-BPM math — real songs have tempo drift; the beat grid captures actual beat positions). After tempo matching, scale beat frame positions proportionally by the tempo ratio.
2. For each section, extract the sample range from each stem
3. Apply per-stem gain from `stem_gains`
4. Apply transition envelopes at section boundaries:
   - **"fade"**: Cosine fade-in over `transition_beats` at section start. Each stem's gain ramps from 0 to the section's `stem_gains` value.
   - **"crossfade"**: **Per-stem gain interpolation** (NOT summed-audio crossfade). During the overlap region (`transition_beats`), each stem's gain is linearly interpolated from the previous section's value to the current section's value using a cosine curve. This avoids the volume dip that occurs when crossfading two summed signals at 50%.
   - **"cut"**: Immediate transition — BUT apply a **micro-crossfade of ~2ms (88 samples at 44.1kHz)** at the boundary to eliminate clicks from waveform discontinuities. This is too short to be audible as a transition but prevents the hard click that occurs when adjacent samples don't match.
5. Sum all stems per section into the output buffer

**LLM system prompt guidance** for sections:
- Sections should be 4, 8, or 16 beats long (shorter sections create more movement and avoid the "static overlay" problem within sections; max 16 beats per section for MVP)
- Default to starting with instrumental only (establishes the beat before vocals enter), unless the user's prompt suggests otherwise or an a cappella intro would be more effective
- Always end with instrumental only or a fade
- Choose arrangement patterns appropriate to the genres of the input songs:
  - **Hip-hop/rap** (typically 80-100 BPM): Keep drums consistent throughout. Build energy through vocal intensity and layering, not drum drops.
  - **EDM/dance** (typically 120-130 BPM): Use breakdown -> build -> drop patterns.
  - **Pop/rock** (typically 100-130 BPM): Use verse-chorus dynamics — stripped for verses, full for choruses.
  - **R&B/soul** (typically 60-90 BPM): Smooth transitions, no abrupt changes. Layer elements gradually.
  - Infer likely genre from BPM range and energy profile characteristics.
- Total duration: should approximate the target duration (derived from the source time ranges and tempo matching)

#### 2. Beat-aligned transitions (~10 lines)

Snap every section boundary to bar lines. When vocals enter on a downbeat it sounds intentional; 200ms off-grid sounds broken.

```python
def snap_to_bar(sample_position: int, beat_positions: np.ndarray, beats_per_bar: int = 4) -> int:
    """Snap a sample position to the nearest bar (downbeat)."""
    bar_positions = beat_positions[::beats_per_bar]
    idx = np.argmin(np.abs(bar_positions - sample_position))
    return int(bar_positions[idx])
```

Since `Section.start_beat` / `end_beat` are expressed in beats (not seconds), the LLM outputs beat-aligned values by construction. The mixer converts beats -> samples using the beat grid. This is effectively free if the arrangement uses beat-based coordinates. Note: `snap_to_bar` is a validation utility for edge cases where the LLM outputs off-grid values; it is not part of the main processing path.

#### 2b. Section validation (~40-60 lines)

The LLM's section output must be validated before the mixer renders it. This replaces the time-range validation from the old `RemixPlan` (which checked `start_time_X`, `end_time_X`).

**Validation checks (applied in order):**
1. Sections must be sorted by `start_beat` ascending — re-sort if not
2. No overlaps: `sections[i].end_beat <= sections[i+1].start_beat` — truncate the earlier section if overlapping
3. No gaps > 1 beat: `sections[i+1].start_beat - sections[i].end_beat <= 1` — extend the earlier section's `end_beat` to close small gaps
4. Minimum section length: `end_beat - start_beat >= 4` — merge tiny sections into the adjacent one
5. `transition_beats <= (end_beat - start_beat) / 2` — clamp if exceeded
6. All `stem_gains` keys must be in `{"vocals", "drums", "bass", "other"}` — add missing keys with default 0.0, remove unknown keys
7. All `stem_gains` values in range `[0.0, 2.0]` — clamp if out of range
8. Total beat range must be within available audio duration (after tempo matching, converted using `beat_frames`) — truncate last section if exceeded
9. At least 2 sections — if only 1, split it into intro (instrumental only) + main section
10. `end_beat` of last section should be a multiple of 4 (bar boundary) — extend or truncate by up to 2 beats

**On validation failure:** Fix automatically where possible (clamp, merge, extend). Log all corrections with severity for LLM prompt quality debugging. Only raise an error if the sections are completely unrecoverable (e.g., 0 sections after merging).

**Default arrangement template** (used when LLM fails entirely or produces unrecoverable output):
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

This fallback template applies conservative defaults (instrumental intro, full vocals + reduced instrumentals for the main body, instrumental outro). It is vastly better than the old "sensible default plan" which had no arrangement concept. When the fallback is used, set `RemixPlan.used_fallback = True` so the frontend can display the explanation with warning styling.

#### 3. Spectral ducking (~30-40 lines, scipy)

Reduce instrumental energy in the 300Hz–3kHz range when vocals are active. Carves a frequency pocket so the vocal sits *inside* the instrumental rather than on top of it.

```python
from scipy.signal import butter, sosfiltfilt

def spectral_duck(instrumental: np.ndarray, vocal: np.ndarray, sr: int,
                  cut_db: float = -3.5, lo: float = 300, hi: float = 3000) -> np.ndarray:
    """Duck instrumental mid-range energy where vocals are active.

    CRITICAL: Uses sosfiltfilt (zero-phase filtering), NOT sosfilt. Causal sosfilt
    introduces frequency-dependent phase delay. Subtracting a phase-shifted mid-band
    from the original creates comb-filter artifacts (metallic/hollow coloration).
    sosfiltfilt applies the filter forward+backward, eliminating phase shift so the
    subtraction is phase-aligned. Non-causal (processes entire buffer) — fine for
    offline rendering.

    Expects stereo (2D) arrays — pipeline step 1 standardizes all stems to stereo.
    """
    assert instrumental.ndim == 2, "Expected stereo (pipeline step 1 standardizes to stereo)"

    # 1. Detect vocal activity via RMS envelope (50ms frames)
    #    Convert stereo vocal to mono for energy detection
    vocal_mono = vocal.mean(axis=1) if vocal.ndim == 2 else vocal
    frame_len = int(0.05 * sr)
    vocal_energy = np.array([
        np.sqrt(np.mean(vocal_mono[i:i+frame_len]**2))
        for i in range(0, len(vocal_mono) - frame_len, frame_len)
    ])
    # 40th percentile threshold (not 25th) — 25th is too aggressive, classifies
    # instrumental bleed as vocal activity, causing over-ducking during pauses
    threshold = np.percentile(vocal_energy[vocal_energy > 0], 40) if np.any(vocal_energy > 0) else 0
    vocal_active = (vocal_energy > threshold).astype(float)

    # 2. Upsample mask to sample rate, smooth with asymmetric attack/release
    #    30ms time constant for attack, 150ms for release (~90ms/450ms to 95%)
    #    Prevents pumping artifacts from rapid on/off between words
    #
    #    PERFORMANCE NOTE: The naive Python loop below is O(n) and takes 5-15 seconds
    #    for a 2-minute track. For implementation, use scipy.signal.lfilter for vectorized
    #    one-pole IIR smoothing — apply with attack coefficient on rising edges and release
    #    on falling edges (two-pass approach). The Python loop is shown here for clarity.
    mask = np.repeat(vocal_active, frame_len)[:len(instrumental)]
    attack_alpha = 1.0 / int(0.03 * sr)
    release_alpha = 1.0 / int(0.15 * sr)
    # Vectorized implementation: use scipy.signal.lfilter with [1, -coeff] denominator
    # for each direction, then take element-wise max. Or use numba.jit on this loop.
    smoothed = np.copy(mask)
    for i in range(1, len(smoothed)):
        alpha = attack_alpha if smoothed[i] > smoothed[i-1] else release_alpha
        smoothed[i] = smoothed[i-1] + alpha * (smoothed[i] - smoothed[i-1])
    mask = smoothed

    # 3. Extract mid-band from instrumental (ZERO-PHASE — critical)
    sos = butter(4, [lo, hi], btype='band', fs=sr, output='sos')
    mid_band = sosfiltfilt(sos, instrumental, axis=0)

    # 4. Reduce mid-band energy where vocals are active (stereo-safe)
    gain = 10 ** (cut_db / 20)
    reduction = mid_band * (1 - gain) * mask[:, np.newaxis]
    return instrumental - reduction
```

Applied to the rendered vocal and instrumental tracks *after* arrangement rendering (step 7 in the revised pipeline) — so the vocal activity mask reflects the actual arrangement gains, not the raw stem content. Sections where vocals are muted will not trigger ducking. The -3.5 dB default is conservative — enough to create space without making the instrumental sound hollow.

#### 4. UI copy — one sentence

Add near the prompt input: **"musicMixer takes the vocals from one song and layers them over the other song's beat."** Eliminates the expectation gap at zero cost.

#### Pipeline order change

The revised processing pipeline (Phase B6) with these additions:

```
1.  Separate stems (htdemucs_ft)
2.  Trim stems to source time ranges (start_time_vocal/instrumental from RemixPlan)
3a. Tempo match all stems to common BPM
3b. Scale beat_frames: beat_frames_stretched = beat_frames * (original_bpm / target_bpm)
    This MUST happen after rubberband and BEFORE the arrangement renderer uses the grid.
    Alternative: re-run librosa.beat.beat_track on the stretched audio for highest accuracy.
4.  Key match if needed
5.  Cross-song level matching (LUFS with silence guard)
6.  Validate LLM sections (section validation — see 2b above).
    Available beats = len(beat_frames_stretched). Truncate sections exceeding this.
7.  Render arrangement into TWO BUSES: vocal bus + instrumental bus ← NEW
    - Place stems per LLM section timeline using beat grid
    - Apply per-stem gains from Section.stem_gains
    - Apply transition envelopes (per-stem interpolation for crossfade, micro-crossfade for cut)
    - Vocal bus = rendered vocal stem with arrangement gains applied
    - Instrumental bus = sum of rendered drums + bass + other stems with arrangement gains
8.  Spectral ducking on instrumental bus (using vocal bus for activity detection) ← NEW
9.  Sum vocal bus + instrumental bus into final mix
10. LUFS normalize final mix
11. Peak limiter
12. Fade-in/fade-out (on the overall output, beyond the arrangement's own transitions)
13. Export MP3
```

Notes:
- Step 3b is critical: the beat grid must be scaled to match the tempo-matched audio, otherwise all section boundaries are at wrong positions.
- Step 7 outputs TWO separate buses (not a single summed buffer) so spectral ducking in step 8 can operate on the instrumental independently, using the vocal bus for activity detection. The buses are summed in step 9.
- Spectral ducking at step 8 (after arrangement rendering) ensures the vocal activity mask reflects the actual arrangement — sections where `stem_gains["vocals"] == 0.0` will not trigger ducking.

#### Plan updates required when QR-1 is implemented

The new section-based `RemixPlan` requires targeted updates to Phases B5, B6, and B7. These are the specific changes needed (not a rewrite — each is a surgical edit):

**Phase B5 (LLM Prompt Interpretation):**
- Replace time-range validation with **section validation** (see 2b above). The old checks (`0 <= start_time_X < end_time_X <= duration_X`, `>= 5.0s minimum`) are replaced by the 10-point section validation checklist.
- Keep `start_time_vocal` / `end_time_vocal` / `start_time_instrumental` / `end_time_instrumental` validation — these source offsets still exist in the new `RemixPlan`.
- Replace system prompt guidance about start/end times with section guidance (already defined above in "LLM system prompt guidance for sections").
- Add **3+ few-shot examples** of correct section output for different genre pairings to the system prompt. Haiku performs significantly better with examples than instructions alone, and the section schema is complex enough that examples are essential for reliable output.
- **Evaluate Haiku's ability** to produce the section schema during Phase B5 implementation. If test results show unreliable section output, escalate to Sonnet for this one call (cost difference is negligible: $0.001 vs $0.003/call).

**Phase B6 (Audio Processing):**
- Step 2 (trim): Trim using `start_time_vocal` / `end_time_vocal` / `start_time_instrumental` / `end_time_instrumental` (these replaced the old `start_time_a/b`).
- Step 5 (volume adjustments): Remove — per-stem volume adjustments are now embedded in `Section.stem_gains`, not a separate `stem_adjustments` list. The arrangement renderer applies gains per-section.
- Step 6 (post-stretch duration alignment): Remove — replaced by section-based duration management. The arrangement's total beat count defines the output length.
- Steps 7-8 (new): Arrangement rendering and section summing as described above.

**Phase B7 (Pipeline Orchestrator):**
- Progress events: Add a `"rendering"` step between `"processing"` and `"complete"` for the arrangement rendering phase.
- No structural changes needed — the orchestrator calls service functions in sequence, and the service interface (`process_remix(...)`) doesn't change.

**API Contract (Integration Plan):**
- Add `warnings: list[str] | None = None` and `used_fallback: bool = False` to `SessionStatusResponse`.
- Add `warnings: list[str] | None` to the `complete` `ProgressEvent`.

**Frontend (Phase F1 types + Phase F5 player):**
- Add `warnings?: string[]` and `usedFallback?: boolean` to `SessionStatus` and `ProgressEvent` TypeScript types.
- Phase F5 (RemixPlayer): Display `warnings` as amber info boxes below the explanation. When `usedFallback: true`, render the explanation with a "generated automatically" indicator (e.g., different border color or prefix text).

### QR-2: LLM is the weakest link — prompt interpretation quality

**Status:** ADDRESSED (revision 13)
**Risk level:** High — directly determines remix quality
**Reviews:** `notes/2026-02-23-quality-review-pipeline.md`, `notes/2026-02-23-plan-review-r1-agent1.md`, `notes/2026-02-23-plan-review-r1-agent2.md`, `notes/2026-02-23-plan-review-r1-agent3.md`

The LLM system prompt is the "product brain" — it decides which song provides vocals, what sections to use, how to handle tempo/key conflicts, and what to explain to the user. It is the most important piece of the pipeline. The QR-1 section-based arrangement significantly increases the complexity of LLM output.

**Contributing factors (original):**
- ~~No system prompt draft or test suite~~ → Phase B5 now includes detailed system prompt specification with condensed energy formatting, arrangement templates, ambiguity handling, few-shot examples, and token budget
- ~~Vague prompts, genre jargon, contradictory requests~~ → Phase B5 now includes explicit handling strategies for all four prompt categories
- ~~Haiku may lack musical reasoning~~ → Default changed to Sonnet; section-based arrangement requires structured reasoning
- ~~Fallback loses all intelligence~~ → Phase B5 now includes a concrete deterministic fallback algorithm using `vocal_energy_ratio`, energy profiles, and a standard 3-section arrangement
- ~~No feedback loop when LLM can't fulfill request~~ → `warnings` field added to `RemixPlan`; `used_fallback` flag enables distinct frontend treatment
- QR-1's section-based schema triples LLM output complexity → Addressed via arrangement templates, condensed input format, surgical code-level validation, and Sonnet default

**Proposed solutions (updated status):**
1. **Write and test the system prompt first** against a categorized test suite of 30 prompts before building the pipeline. Include 3 diverse few-shot examples. **Test prompt categories defined:**

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
   | **Total** | **30** | |

   Each test case includes: input prompt, mock song metadata (BPM/key/energy/vocal_energy_ratio), expected `vocal_source`, and qualitative expectations. Save as `tests/fixtures/llm_test_prompts.json`.

2. **Deterministic data-driven fallback** — now specified concretely in Phase B5 with `vocal_energy_ratio`-based vocal source selection, energy-profile-guided section selection, BPM-confidence-based tempo decision, and honest user-facing explanation with `used_fallback=True`.

3. **`warnings` field on RemixPlan** — LLM populates with specific caveats (what it couldn't fulfill, prompt ambiguities). Frontend displays with distinct styling. `used_fallback` flag triggers warning treatment even when fallback fires.

4. **Default to Sonnet** (configurable). Section-based arrangement is structured reasoning where Sonnet substantially outperforms Haiku. Evaluate both on the test prompt suite using this scoring rubric:

   | Criterion | Metric | Weight |
   |-----------|--------|--------|
   | Schema validity | % valid without retries | High |
   | Prompt fidelity | 1-5: does plan reflect user intent? | High |
   | Musical coherence | 1-5: does arrangement tell a sensible story? | High |
   | Metadata utilization | 1-3: uses energy data, respects tempo/key limits | Medium |
   | Explanation quality | 1-3: clear, accurate, useful | Medium |
   | Latency | Measured (Haiku ~1-2s, Sonnet ~3-5s) | Low |

   Run each model against same 30 prompts, score blindly. If Sonnet scores >20% higher on quality, use Sonnet. If within 20%, Haiku wins on latency.

**Remaining medium/low items** (address during implementation):
- tool_use schema definition: use explicit per-stem gain fields (not dict), enum constraints, min/max bounds, rich descriptions
- Ongoing monitoring: save test suite as fixture, write runnable regression test against real LLM, log `model` field from every response to correlate quality with model version changes
- Prompt injection: low risk for MVP (tool_use constrains output), add defensive note to system prompt

### QR-3: Common song combinations fail silently

**Status:** ADDRESSED (revision 17)
**Risk level:** High — the most obvious combos users will try
**Reviews:** `notes/2026-02-23-quality-review-audio.md`, `notes/2026-02-23-quality-review-ux.md`, `notes/2026-02-23-qr3-review-r1-agent1.md`, `notes/2026-02-23-qr3-review-r1-agent2.md`, `notes/2026-02-23-qr3-review-r1-agent3.md`, `notes/2026-02-23-qr3-review-r2-agent1.md`, `notes/2026-02-23-qr3-review-r2-agent2.md`, `notes/2026-02-23-qr3-review-r2-agent3.md`, `notes/2026-02-23-qr3-review-r3-agent1.md`, `notes/2026-02-23-qr3-review-r3-agent2.md`, `notes/2026-02-23-qr3-review-r3-agent3.md`

~70% of cross-genre pairings hit the tempo wall (>30% BPM difference). Weighted by user intent (hip-hop + pop is the most-attempted mashup), the effective failure rate is ~75-80%. Users discover the bad result only after waiting the full processing time.

**Contributing factors:**
- The old 30% tempo stretch cap with "skip entirely" above it left the most common mashup pairing (hip-hop + pop, 41% gap) completely unserved — now addressed with vocals-only stretching up to 50% (see Phase B6)
- Fixed +3 dB vocal boost was wrong for most genres — now replaced with spectral-density-adaptive offset (+1 to +8 dB based on instrumental density in the vocal frequency band; see Phase B6 step 4)
- No pre-processing compatibility check — now addressed with `POST /api/analyze` endpoint (~2-3 seconds, runs after upload before prompt submission)
- Confident-but-wrong key detection (~20-30% of songs) — now mitigated by raising key confidence gate from 0.3 to 0.45 (see Phase B6)
- Groove feel mismatch (half-time vs straight-time) not detected — partially addressed (see below)
- No user-facing explanation of WHY songs don't work together — now addressed via compatibility signal and LLM transparency

**Solutions implemented in this revision:**

#### 1. Pre-upload compatibility analysis (new `POST /api/analyze` endpoint)

Runs lightweight analysis (~2-3s on CPU) immediately after both songs are uploaded, BEFORE the user submits their prompt. Returns a plain-language compatibility signal. See API Contract for the full spec.

**Concrete pipeline (fits in ~2-3 seconds):**
1. Load audio (`librosa.load` with `duration=60, sr=22050`) — ~0.3s/song
2. BPM detection (`librosa.beat.beat_track`) — ~0.4s/song
3. Key detection (`essentia.KeyExtractor`) — ~0.3s/song
4. Run both songs in parallel (`ThreadPoolExecutor`) = ~1.2s total
5. BPM reconciliation (expanded interpretation matrix, see Phase B4) — ~0.01s
6. Key compatibility (circle-of-fifths distance) — ~0.01s
7. Composite score → user-facing message

**Compatibility levels (based on reconciled BPM distance + key distance):**
- **Great** (BPM <10%, CoF 0-1): "These songs should blend really well together."
- **Good** (BPM 10-20%, or key shift <= 3 semitones): "These songs have some differences, but we can work with it."
- **Challenging** (BPM 20-35%): "These songs have a different energy. The remix will have more of a mashup feel."
- **Tough** (BPM >35%, or key shift > 5 semitones): "These songs are very different — the AI will do its best, but they may not sync up perfectly."

**Critical UX design decisions:**
- **Never show raw BPM/key values to the user.** "95 BPM, G minor" means nothing to the target audience. Synthesize into the plain-language message above. Reserve raw values for an optional collapsed "Details" disclosure for power users.
- **Never block or gate the submission flow.** The compatibility signal is *informational*, not a gate. No modal, no confirmation, no decision point. The user reads it and proceeds if they want. Friction at the moment of creative commitment kills engagement.
- **Reframe from judgment to description.** Never use "warning," "bad," or "rough." Instead of "Good match!" / "Very different tempos — remix may sound rough," use descriptions that frame both outcomes positively: similar feel = "natural and smooth"; different feel = "mashup feel — the more different, the more creative."
- **Celebrate good matches.** The positive tier ("Great — these songs should blend really well") builds excitement and teaches users what "good" looks like through positive reinforcement.

**Interaction design:**
- Analysis triggers automatically when *both* songs are uploaded
- Show a subtle "Checking compatibility..." indicator between upload zones and prompt input (~1-2s)
- Prompt input is available immediately — do NOT block on analysis (most users spend >2s thinking about their prompt)
- When analysis completes, compatibility signal appears as an inline element
- If user replaces a song, clear signal and re-run
- If user submits before analysis completes, proceed without showing compatibility (it still runs as part of the pipeline)

#### 2. Spectral-density-adaptive vocal balance (replaces fixed +3 dB)

See Phase B6 step 4 for the full implementation. Uses `scipy.signal.welch` to measure instrumental spectral density in the vocal frequency band (200 Hz - 5 kHz) and maps to an offset: sparse instrumental = +1-2 dB, dense = +5-8 dB. ~15 lines, no new dependencies. The LLM's `stem_adjustments` remain as user-directed overrides.

#### 3. Expanded BPM reconciliation (70-180 range, 3/2 and 2/3 ratios)

See Phase B4 for the expanded algorithm. Generates {original, halved, doubled, 3/2, 2/3} interpretations per song, filtered to 70-180 BPM, selects the pair with minimum gap. Reduces the cross-genre failure rate from ~73% to ~53% by rescuing R&B (72 doubled to 144) + pop/rock/EDM pairings.

**What it doesn't rescue:** Hip-hop (85) + pop (120) = 41% gap has no half/double/triplet relationship. This is solved by solution #4 below.

#### 4. Vocals-only stretching for 30-50% tempo gaps

See Phase B6 tempo matching tiered limits. For gaps between 30-50%, the system now stretches ONLY the vocal stem to match the instrumental's tempo, instead of skipping tempo matching entirely. Vocals tolerate 30-50% stretch (noticeably different but intelligible); instrumentals do not (drums sound broken beyond 10%). This rescues the hip-hop + pop pairing.

**Default tempo target changed to instrumental source song's tempo** (not "average"). Rationale: drums are the most stretch-sensitive element; preserving the instrumental's tempo produces a more natural rhythmic foundation.

#### 5. Key confidence gate raised from 0.3 to 0.45

See Phase B6 key matching. At 0.3, essentia is correct only ~50% of the time — wrong pitch shifts sound worse than no shift. At 0.45, accuracy is ~65-70%, making the expected value of shifting slightly positive.

#### 6. Transparency framework (inform → transparent → honest)

Instead of hard-blocking, soft-warning, or silently proceeding, the system follows a three-stage transparency approach:

**Before submission** (compatibility signal): Informational, non-blocking. Describes what the remix will be like, not what's wrong with the songs.

**During processing** (progress events): Transparent about what's happening. When tempo or key matching is skipped or limited, the progress detail says so:
- "Mixing stems... (adjusting vocal speed to match the beat — these songs have different tempos)"
- "Keeping original keys (key detection wasn't confident enough to shift safely)"
- "Matching tempo... (stretching vocals to match the instrumental's rhythm)"

**After processing** (LLM explanation): Honest about what was done and why. The LLM explains any compatibility challenges and what the system did about them: "These songs had very different tempos (Song A is much slower than Song B), so I stretched the vocals to match Song B's beat. The vocal may sound slightly different from the original."

**Hard-block conditions (return 422):** None in MVP. The system always attempts a remix. Even a rough remix is better than "we can't do this."

#### 7. Groove feel (partial — LLM prompt guidance + post-MVP detection)

**The problem:** Two songs can have matching BPMs but incompatible grooves (half-time vs straight-time, swung vs quantized, 6/8 vs 4/4). The plan treats tempo as a scalar number. A half-time hip-hop beat at 170 BPM has its snare on beat 3, while a straight-time pop beat at 128 BPM has snares on beats 2 and 4 — even after tempo matching, the rhythmic feel clashes.

**MVP solution:** Add groove awareness to the LLM system prompt as a known limitation: "Even when BPMs are matched, different groove feels (half-time vs straight-time, swung vs quantized) can cause rhythmic clashing. When a half-time track is matched with a straight-time track, prefer using breakdown sections from the instrumental (where drums drop out) to avoid rhythmic conflicts."

**Post-MVP solution:** Detect groove classification from drum stem onset patterns (kick-snare spacing analysis). Classify as half-time, straight-time, or double-time. Include in `AudioMetadata` and in the LLM's context for better arrangement decisions.

#### 8. Alternative techniques for extreme tempo gaps (LLM system prompt)

For gaps >50% where even vocals-only stretching fails, the LLM system prompt should instruct it to use alternative arrangement techniques rather than giving up:
- **A cappella over breakdowns:** Place vocal phrases over low-onset-density sections where drums drop out — no tempo matching needed
- **Loop-based arrangement:** Loop a 4-8 bar instrumental phrase instead of a continuous section — short loops are more forgiving of tempo mismatches
- **Focus on non-rhythmic sections:** Select breakdown/ambient sections from the instrumental where there is no rhythmic grid to clash with

These require only system prompt changes, zero additional code. The arrangement system from QR-1 (section-based timeline) is the natural foundation.

**Remaining open items (Medium/Low severity, deferred):**
- Drum-stem BPM detection after separation for higher accuracy (~0.5s, reduces octave errors from ~29% to ~10-15%)
- Multi-seed BPM estimation (`start_bpm` in {60, 90, 120, 150}) for reduced octave errors
- Evaluation of `madmom` library as librosa replacement for BPM detection (~95% accuracy vs librosa's ~71%)
- Vocal stem quality validation (SNR check) after separation to warn about poor separation
- Swing/straight detection from drum stem inter-onset-interval analysis
- Recovery flow: "Create New Remix" should pre-fill previously uploaded songs (keep Song A, replace only Song B)
- Post-remix compatibility insight with suggestion: "For a smoother blend, try a song with a similar rhythm"

---

## Recommended Implementation Order

```
Week 1: Foundation + Services
  Backend:  B1 (scaffold + config + CORS + logging)
            B2 (upload + validation)
            B3 (separation) ←→ B4 (analysis)  [parallel]
  Frontend: F1 (scaffold + types + API client + state machine)
            F2 (upload UI) ←→ F3 (prompt + submission)  [parallel]

Week 2: Intelligence + Processing
  Backend:  B5 (LLM interpreter)
            B6 (processing — tempo/key/loudness/filtering/mixing)
            B7 (orchestrator + SSE + TTL)
  Frontend: F4 (progress display + SSE hook)
            F5 (player)

Week 3: Integration + Polish
  End-to-end integration testing
  Error handling edge cases
  SSE reconnection testing
  Audio quality iteration (LLM prompt tuning, filter parameters)
```

---

## What's NOT in This Plan (Deferred)

Per the brainstorm and PRD, these are explicitly out of scope:

- Cross-song stem mixing (e.g., drums from Song A + bass from Song B) — post-MVP
- Iterative refinement ("make the drums louder")
- Export/download/sharing
- Multi-song mixing (3+ songs)
- Voice cloning
- Mobile app (native — the web app IS mobile-responsive)
- User accounts / authentication
- Social features
- Search-based song input
- Streaming integration
- Waveform visualization (consider wavesurfer.js post-MVP)
- Volume/EQ controls in the player
- Advanced section detection (allin1 package — post-MVP)
- Filter sweep transitions (high-pass/low-pass sweeps during transitions — post-MVP, uses same scipy already in deps)
- Stereo panning
- Multiple concurrent users (requires queue system)
