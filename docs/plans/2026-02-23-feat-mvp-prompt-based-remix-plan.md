---
title: "feat: MVP Prompt-Based Remix"
type: feat
status: active
date: 2026-02-23
revision: 4
brainstorm: docs/brainstorms/2026-02-23-prompt-based-remix-brainstorm.md
prd: docs/PRD.md
reviews: notes/2026-02-23-audio-pipeline-review.md
---

# MVP: Prompt-Based Music Remix

## Overview

Build the end-to-end musicMixer MVP: upload two songs, describe a mashup in plain English, get an AI-generated remix. Server renders the final audio file; browser plays it back. Remix expires after 3 hours.

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
│  │     d. Frequency management (scipy high/low-pass filters)   │  │
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
| Loudness | `pyloudnorm` (LUFS normalization) | Stems from different songs have wildly different loudness. LUFS normalization is critical for listenable output. |
| Frequency management | `scipy.signal` high/low-pass filters | Prevents muddy bass stacking and vocal clashing. 80Hz high-pass on non-bass stems, 400Hz low-pass on competing bass. Zero additional dependencies (scipy is a librosa transitive dep). |
| Audio mixing | `pydub` + `ffmpeg` + fade-in/fade-out | High-level API for overlay, volume, fades, export. |
| Remix delivery | Server-rendered single MP3 file | Massively simplifies frontend — just an `<audio>` tag. No Tone.js needed. |
| Browser playback | HTML5 `<audio>` element | Zero JS dependencies. Native controls. Streams large files. |
| LLM for prompts | Claude Haiku (structured outputs) | Fast, cheap (~$0.001/call), schema-guaranteed JSON. |
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
processing_lock = threading.Lock()  # One remix at a time
executor = ThreadPoolExecutor(max_workers=1)

@dataclass
class SessionState:
    status: str                      # "queued" | "processing" | "complete" | "error"
    events: queue.Queue              # Pipeline pushes, SSE endpoint reads
    created_at: datetime
    remix_path: str | None = None
    explanation: str | None = None

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
async def event_stream(session: SessionState):
    loop = asyncio.get_running_loop()  # not get_event_loop() (deprecated 3.10+)
    while True:
        try:
            # Read with 30s timeout — prevents thread leak on client disconnect.
            # When the async generator is cancelled (client disconnects), the
            # executor thread wakes up on the next timeout and returns to the pool.
            event = await loop.run_in_executor(
                None, functools.partial(session.events.get, timeout=30)
            )
        except queue.Empty:
            yield ": keepalive\n\n"
            continue
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

      # Processing
      stem_model: str = "htdemucs_ft"
      max_tempo_adjustment_pct: float = 0.30
      max_pitch_shift_semitones: int = 5
      output_format: str = "mp3"
      output_bitrate: str = "320k"
      target_lufs: float = -14.0

      # LLM
      anthropic_api_key: str  # Required, no default
      llm_model: str = "claude-haiku-4-5-20251001"
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

**Goal:** Detect BPM and musical key for each uploaded song.

- [ ] Install `librosa`, `essentia`
  - For essentia: pre-built wheels on PyPI for macOS ARM64/x86_64 (Python 3.10-3.13)
  - If essentia install fails, fall back to librosa chromagram — **design this fallback into the interface from the start**
- [ ] Create `services/analysis.py`:
  - `analyze_audio(audio_path: Path) -> AudioMetadata`
  - `AudioMetadata` in `models.py`:
    ```python
    @dataclass
    class AudioMetadata:
        bpm: float
        bpm_confidence: float     # for half/double-time decision
        key: str                  # e.g., "C"
        scale: str                # "major" or "minor"
        key_confidence: float
        duration_seconds: float
    ```
  - BPM via `librosa.beat.beat_track`
  - **BPM half/double-time sanity check**: If the two songs' detected BPMs differ by roughly 2:1 (ratio between 1.8-2.2), try the alternative interpretation (half or double) and pick the one that brings them closer together. This catches the ~29% of songs where librosa reports the wrong octave.
  - Key via `essentia.standard.KeyExtractor` (or librosa chromagram fallback)
  - Duration via `librosa.get_duration`
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
  - Expanded `RemixPlan` in `models.py`:
    ```python
    @dataclass
    class StemSelection:
        stem: str            # "vocals" | "drums" | "bass" | "other"
        include: bool
        volume_db: float     # relative volume adjustment (applied AFTER LUFS normalization)

    @dataclass
    class RemixPlan:
        song_a_stems: list[StemSelection]
        song_b_stems: list[StemSelection]
        tempo_source: str    # "song_a" | "song_b" | "average"
        key_source: str      # "song_a" | "song_b" | "none" (skip key matching)
        duration_seconds: int           # Target output length (60-180s)
        start_time_a: float             # Where to start in Song A (seconds)
        end_time_a: float               # Where to end in Song A (seconds)
        start_time_b: float             # Where to start in Song B (seconds)
        end_time_b: float               # Where to end in Song B (seconds)
        explanation: str                # LLM's reasoning (shown to user)
    ```
  - System prompt instructs the LLM about:
    - Available stems and what "other" contains (guitar/synths/keys — be honest with users)
    - Song metadata (BPM, key, duration)
    - Stem compatibility guidelines: avoid selecting both vocals unless user explicitly asks; if both bass stems are selected, note frequency clash risk
    - Tempo/key matching limits: if BPMs differ by >30%, recommend `key_source: "none"` or explain tradeoff. If keys are >4 semitones apart, recommend `key_source: "none"`.
    - Section selection: pick the most interesting section of each song (avoid intros/outros which are usually sparse). Default to 60-120 second output length.
    - Output a clear explanation the user will read
  - Uses Claude Haiku with tool_use for schema enforcement
- [ ] **Validate RemixPlan time ranges** against actual song durations after LLM returns:
  - `0 <= start_time_X < end_time_X <= duration_X` for both songs
  - `end_time_X - start_time_X >= 5.0` seconds (minimum usable segment)
  - Clamp out-of-bounds values to valid ranges (don't re-prompt — just fix silently)
  - Log any clamped values for debugging LLM prompt quality
- [ ] Error handling for LLM calls:
  - Retry 1-2x with backoff on transient errors (429, 500, 529)
  - Timeout after 30 seconds
  - On schema violation: re-request with stricter instructions
  - On total failure: fall back to a sensible default plan (vocals from A + instrumentals from B)
- [ ] Write unit test with mocked LLM response

**Reversibility:** Interface is `(prompt, metadata, metadata) → RemixPlan`. Swap Claude for OpenAI by changing one module. Swap Haiku for Sonnet if interpretation quality needs improvement.

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
    1. **Trim stems** to the time ranges specified in RemixPlan (`start_time_a`/`end_time_a`, etc.)
    2. **Tempo matching** with tiered limits:
       - < 10% BPM difference: stretch silently
       - 10-25%: stretch, note in explanation
       - 25-30%: stretch, strongly warn
       - > 30%: skip tempo matching entirely, let LLM explain why
       - Use pyrubberband with R3 engine: `rbargs={"--fine": ""}`
       - If `tempo_source` is `"average"`, stretch both songs toward the midpoint (sounds better than stretching one song by a large amount)
    3. **Key matching** (if `key_source` != `"none"`):
       - Calculate semitone difference between detected keys
       - Cap at +/- 4 semitones for vocal stems, +/- 5 for instruments
       - **Enable formant preservation for vocal stems**: `rbargs={"--formant": ""}` — prevents chipmunk/demonic artifacts
       - If shift would exceed cap, skip key matching for that stem
    4. **LUFS loudness normalization**:
       - Measure LUFS of each included stem with `pyloudnorm`
       - Normalize all stems to target LUFS (default -14.0, configurable)
       - Then apply the LLM's relative `volume_db` adjustments on top
    5. **Frequency management** (role-based filtering with `scipy.signal`):
       - High-pass filter at ~80Hz on vocal and "other" stems (removes low-end rumble that clashes with bass/drums)
       - If both songs contribute bass stems: low-pass one at ~400Hz to prevent bass mud (250Hz is too aggressive — cuts the harmonics that give bass its character and articulation)
       - Optional: gentle 2-4kHz dip on instrumentals when vocals are present (make room for voice)
    6. **Sample rate standardization**: Ensure all stems are at 44.1kHz stereo 32-bit float before processing. htdemucs_ft outputs 44.1kHz, but verify and resample with `librosa.resample` if needed. Prevents subtle pitch/timing bugs from mismatched rates.
    7. **Mix**: Sum all processed stems in numpy (32-bit float) rather than pydub overlay — avoids pydub's internal 16-bit clipping during summation.
    8. **Peak limiter**: After summing, apply a peak limiter at -1.0 dBTP ceiling to prevent clipping during MP3 encoding. LUFS normalization targets average loudness, not peaks — multiple stems summed together will exceed 0 dBFS without limiting:
       ```python
       ceiling = 10 ** (-1.0 / 20.0)  # -1.0 dBTP
       peak = np.max(np.abs(mixed))
       if peak > ceiling:
           mixed = mixed * (ceiling / peak)
       ```
    9. **Fade-in/fade-out**: Apply 2-second fade-in and 3-second fade-out
    10. **Export**: Convert to pydub AudioSegment, render as MP3 (320kbps) to output path
  - Returns path to rendered remix file
- [ ] Write unit test with pre-separated test stems

**Orthogonality:** Processing takes a plan + stems + metadata and produces an audio file. Doesn't know about uploads, LLMs, or the web layer.

**Files created:** `src/musicmixer/services/processor.py`, `tests/test_processor.py`

---

### Phase B7: Pipeline Orchestrator + SSE Progress + TTL

**Goal:** Wire all services into an end-to-end pipeline with real-time progress updates and session lifecycle management.

- [ ] Create `services/pipeline.py`:
  - `run_pipeline(session_id: str, song_a_path: Path, song_b_path: Path, prompt: str, event_queue: queue.Queue) -> None`
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
  - Acquires `processing_lock` before starting. If lock is held, push immediate error event.
  - **Lock crash safety**: Always release the lock in a `try/finally` block — `finally` executes even on `BaseException` subclasses (`KeyboardInterrupt`, `SystemExit`), preventing orphaned locks on thread death.
- [ ] Create SSE endpoint `GET /api/remix/{session_id}/progress`:
  - Returns `StreamingResponse` with `text/event-stream` content type
  - On connect: check session status. If already complete, send final event immediately. If processing, send latest event then continue streaming.
  - Async generator reads from `session.events` queue via `run_in_executor`
  - No-event timeout: if no event arrives within 120 seconds, send a keepalive comment (`:\n\n`)
- [ ] Create status endpoint `GET /api/remix/{session_id}/status`:
  - Returns current session state as JSON (not SSE)
  - Used by frontend for quick reconnection check after page refresh
  - Response: `{"status": "processing", "progress": 0.45, "detail": "..."}` or `{"status": "complete", "remix_url": "..."}` or `{"status": "error", "detail": "..."}`
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

  async def cleanup_loop():
      while True:
          await asyncio.sleep(settings.cleanup_interval_seconds)
          try:
              # Delete sessions older than remix_ttl_seconds
              # Remove session from sessions dict FIRST (new requests get 404)
              # THEN delete associated files: uploads/, stems/, remixes/
          except Exception:
              logger.exception("Cleanup cycle failed, will retry next interval")
  ```
  - Cleanup skips sessions that are currently in "processing" status
  - Log all cleanup actions with session IDs
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
  export type ProgressStep = 'separating' | 'analyzing' | 'interpreting' | 'processing' | 'complete' | 'error';

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
    status: 'processing' | 'complete' | 'error';
    progress?: number;
    detail?: string;
    remix_url?: string;
    explanation?: string;
  };

  // App state (discriminated union)
  export type AppState =
    | { phase: 'idle'; songA: File | null; songB: File | null; prompt: string }
    | { phase: 'uploading'; songA: File; songB: File; prompt: string; uploadProgress: number }
    | { phase: 'processing'; sessionId: string; progress: ProgressEvent }
    | { phase: 'ready'; sessionId: string; explanation: string }
    | { phase: 'error'; message: string; songA: File | null; songB: File | null; prompt: string };

  // Reducer actions (discriminated union — prevents illegal state transitions at compile time)
  export type AppAction =
    | { type: 'SET_SONG_A'; file: File | null }
    | { type: 'SET_SONG_B'; file: File | null }
    | { type: 'SET_PROMPT'; prompt: string }
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
    - `status: "processing"` → enter `processing` phase, reopen EventSource
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

**Files created:** `src/components/SongUpload.tsx`

---

### Phase F3: Prompt Input + Form Submission

**Goal:** Text field for the remix description. Wire up the full submission flow.

- [ ] Create `components/PromptInput.tsx`:
  - Textarea for longer prompts
  - Example prompts shown as **static text above the input** (not as placeholder — placeholders vanish on focus):
    - "Hendrix guitar with MF Doom rapping over it"
    - "Just the drums from Song A with everything from Song B"
    - "Slow down Song A and layer the vocals over Song B's beat"
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
  - Returns: `{ currentEvent: ProgressEvent | null, error: string | null, isConnected: boolean }`
  - **Cleanup on unmount**: `.close()` the EventSource in the hook's cleanup function
  - **No-event timeout**: If no event arrives within 60 seconds, set error state ("Processing is taking longer than expected")
  - **Tab backgrounding**: Listen to `document.visibilitychange`. On tab refocus, check connection state. If dead, reconnect by calling `getSessionStatus()` first then reopening EventSource.
  - **Reconnection strategy (MVP)**: On disconnect, call `getSessionStatus()` to get current state. If complete, skip SSE and go to ready. If processing, reopen EventSource (will receive current event on connect thanks to backend design). Accept possible duplicate events — harmless for a progress bar.
  - **Error events**: Parse `step === "error"` events and surface the detail message
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
  Response 200: {"status":"processing","progress":0.45,"detail":"Matching tempo..."}
           OR: {"status":"complete","remix_url":"/api/remix/{id}/audio","explanation":"..."}
           OR: {"status":"error","detail":"..."}
  Response 404: Session not found or expired

GET /api/remix/{session_id}/audio
  Response 200: audio/mpeg (MP3 file)
  Response 404: Remix not found or expired

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
    status: str          # "processing" | "complete" | "error"
    progress: float | None = None
    detail: str | None = None
    remix_url: str | None = None
    explanation: str | None = None

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
| Stem granularity | 4-stem (vocals/drums/bass/other) with htdemucs_ft. "Guitar" is in "other" — LLM is honest about this in its explanation. Upgrade path: BS-RoFormer models via audio-separator for better guitar/piano isolation. |
| File format/size limits | MP3 and WAV. Max 50MB per file, max 10 minutes. Validated by parsing the audio, not just MIME type. |
| Processing architecture | Server-side everything. Pipeline runs in ThreadPoolExecutor with threading.Queue for progress events. Single-worker, processing lock, one remix at a time. |
| Processing time | GPU: ~30s total. CPU: ~12-15min total. Progress UI keeps users informed either way. |
| First-time experience | Tagline + static example prompts above the input field + self-explanatory 2-upload + 1-text-field interface. |
| Error handling | Multi-layer: upload validation (413/422), pipeline errors via SSE, LLM fallback to default plan, audio 404 for expired remixes. Full error taxonomy defined. |
| Async model | ThreadPoolExecutor with threading.Queue. Pipeline pushes events, SSE endpoint reads them via async wrapper. |
| Concurrency | Single-worker uvicorn, global processing lock. Second request gets 429. |
| Output length | LLM decides via `duration_seconds` field (default 60-120s, max 180s). LLM also picks start/end times for each song. |
| Section selection | LLM picks via `start_time_a`/`end_time_a` fields. System prompt instructs it to avoid intros/outros. Post-MVP: add `allin1` package for structural analysis (verse/chorus detection). |
| Loudness matching | LUFS normalization with `pyloudnorm` to -14 LUFS target, then LLM's relative volume adjustments applied on top. |
| Frequency clashing | Role-based filtering: high-pass at 80Hz on non-bass stems, optional low-pass on competing bass. Uses scipy (already a transitive dependency). |
| Build vs buy | Build locally for MVP ($0.003/remix). Fadr API ($0.05/min, includes BPM+key+chords) is the fallback if GPU setup takes >1 day. No API exists for the full prompt-based pipeline — this is genuinely novel. |

---

## Risk Analysis

| Risk | Impact | Mitigation |
|------|--------|------------|
| CPU-only processing is too slow (12+ min) | Users abandon | Deploy with GPU, or set expectations with progress UI. Consider Fadr API as a faster alternative for separation. |
| PyTorch installation issues | Blocks development | Pin exact version + variant in pyproject.toml. Document setup steps. Test on a clean machine early. Consider Docker. |
| Essentia installation fails | Blocks key detection | Pluggable interface: auto-fallback to librosa chromagram. Designed in from the start. |
| BPM half/double-time error (~29% of songs) | Wrong tempo = terrible remix | Sanity check: if BPMs differ by ~2:1, try alternative interpretation. |
| Tempo stretch >30% sounds bad | Artifact-ridden output | Tiered limits with user-facing explanations. Average-tempo strategy when possible. |
| Pitch-shifted vocals sound chipmunk/demonic | Unnatural output | Formant preservation enabled by default for vocal stems. Cap at +/- 4 semitones for vocals. |
| Two tracks layered sound like two tracks playing | Unmusical output | LUFS normalization + frequency filtering + fades + smart LLM stem selection. |
| LLM produces bad stem selections | Bad remixes | Detailed system prompt with compatibility guidelines. Explanation shown to user. Fallback default plan on failure. |
| Disk fills up (~500MB per session) | Server crashes | TTL cleanup every 5 minutes. Log disk usage. Reject new remixes if disk is critically low. |
| Server restart loses all sessions | Active remixes lost, orphaned files on disk | Acceptable for MVP (single user). TTL cleanup will eventually remove orphaned files. Post-MVP: persist session state to SQLite or Redis. |
| Page refresh loses session | User frustration | sessionStorage persistence. Status endpoint for quick reconnection. SSE replays current state on connect. |
| File upload is slow (100MB on slow connection) | Users abandon | Upload progress bar via XMLHttpRequest. Client-side file size validation before upload. |

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
- Beat-aligned mixing / dynamic stem introduction
- Stereo panning
- Multiple concurrent users (requires queue system)
