---
title: "feat: MVP Prompt-Based Remix"
type: feat
status: active
date: 2026-02-23
revision: 22
brainstorm: docs/brainstorms/2026-02-23-prompt-based-remix-brainstorm.md
prd: docs/PRD.md
reviews: notes/2026-02-23-audio-pipeline-review.md
---

# MVP: Prompt-Based Music Remix

## Overview

Build the end-to-end musicMixer MVP: upload two songs, describe a mashup in plain English, get an AI-generated remix. Server renders the final audio file; browser plays it back. Remix expires after 3 hours.

### MVP Scope Constraint

The MVP uses a **vocals + instrumentals** model: vocals always come from one song, instrumentals (drums + bass + guitar + piano + other recombined) always come from the other. There is no cross-song stem mixing (e.g., drums from Song A with bass from Song B). However, the system still performs full 6-stem separation (vocals, drums, bass, guitar, piano, other) so the LLM can apply per-stem volume adjustments (e.g., "boost the bass," "mute the piano," "quiet the drums") on the instrumental side. This constraint keeps the mixing simple while still delivering a compelling product, and the architecture is designed so cross-song stem mixing can be added later without structural changes.

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
│  │  2. Stem Separation (cloud GPU via Modal + BS-RoFormer)     │  │
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
| Stem separation | BS-RoFormer SW via Modal (cloud GPU) | 6-stem output (vocals/drums/bass/guitar/piano/other). ~30-60s/song on cloud GPU. Best-in-class quality (SDX'23 winner). `htdemucs_ft` (4-stem, local) available as fast fallback. |
| BPM detection | `librosa` + half/double-time sanity check | Simple, well-documented. Sanity check catches the ~29% failure rate for tempo octave errors. |
| Key detection | `essentia` (fallback: librosa chromagram) | Best accuracy. Pluggable interface so essentia install issues don't block. |
| Tempo/key matching | `pyrubberband` with R3 engine | Industry-standard quality. Tiered stretch limits. Formant preservation for vocals. |
| Loudness | `pyloudnorm` (LUFS normalization on final mix) | LUFS normalization applied to the final summed mix (not per-stem). Per-stem normalization causes the sum to exceed target loudness, forcing the peak limiter to crush the audio. |
| Frequency management | Spectral ducking in MVP; full cross-song filtering post-MVP | Spectral ducking (QR-1) uses `scipy.signal.sosfiltfilt` to carve a mid-range pocket for vocals in the instrumental. Full cross-song stem filtering (e.g., competing bass lines) deferred to post-MVP when cross-song mixing is added. 6-stem separation reduces the need for frequency filtering since guitar/piano are isolated from "other". |
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
- **Cloud GPU for separation, local for everything else**: Stem separation runs on Modal (serverless cloud GPU) using BS-RoFormer SW for best quality at ~$0.001-0.005/remix. All other processing (analysis, tempo/key matching, mixing) runs locally. `htdemucs_ft` (local, 4-stem) available as fallback if Modal is down or for fast iteration during development.

---

## Async Processing Model

This is the backbone of the backend. The pipeline includes network I/O (Modal cloud GPU for stem separation) and CPU-bound work (audio processing), so it **cannot** run in the FastAPI async event loop without blocking all request handling.

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
    events: queue.Queue              # Pipeline pushes via emit_progress(), SSE endpoint reads. maxsize=100.
                                     # IMPORTANT: Pipeline MUST use put_nowait() (via emit_progress helper)
                                     # to avoid blocking when the queue is full (no SSE consumer connected).
                                     # Non-terminal events are dropped on full queue. Terminal events
                                     # (complete/error) drain one old event first to guarantee delivery.
    created_at_mono: float           # time.monotonic() at creation — immune to DST/NTP clock changes
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

# Pipeline event helper — NEVER use queue.put() directly (blocks when full):
def emit_progress(event_queue: queue.Queue, event: dict):
    """Non-blocking event push. Drops non-terminal events on full queue.
    Terminal events (complete/error) drain one old event first to guarantee delivery."""
    try:
        event_queue.put_nowait(event)
    except queue.Full:
        if event["step"] in ("complete", "error"):
            try:
                event_queue.get_nowait()  # Drop oldest to make room
            except queue.Empty:
                pass
            event_queue.put_nowait(event)
        else:
            logger.warning("Event queue full, dropping: %s", event["step"])

# POST /api/remix → saves files, creates session, submits to executor
# Pipeline runs in thread, pushes events to session.events queue
# GET /api/remix/{id}/progress → async wrapper reads from session.events queue
# GET /api/remix/{id}/status → returns current status (for reconnection)
```

### Why ThreadPoolExecutor (not ProcessPoolExecutor)

- `threading.Queue` works across threads natively. `multiprocessing.Queue` requires pickling, which fails with many audio objects.
- GIL is not a problem: Modal cloud GPU calls are network I/O (GIL released during `await`/blocking network). Local audio processing releases the GIL during file operations. The thread worker and FastAPI event loop can run concurrently.
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

    # On connect: send current state and drain stale events from the queue.
    # When a client disconnects and reconnects, events pushed during the gap
    # sit in the queue. Without draining, the client receives old events AFTER
    # last_event, causing the progress bar to jump backward.
    if session.last_event:
        yield f"data: {json.dumps(session.last_event)}\n\n"
        if session.last_event["step"] in ("complete", "error"):
            return
    # Drain stale events — last_event is the authoritative state
    while not session.events.empty():
        try:
            session.events.get_nowait()
        except queue.Empty:
            break

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
      error_ttl_seconds: int = 900  # 15 minutes — error sessions cleaned up faster (no user value, same disk cost)
      max_sse_duration_seconds: int = 1200  # 20 minutes — safety cap for hung pipelines

      # Processing
      stem_model: str = "bs_roformer_sw"  # BS-RoFormer SW (6-stem, cloud GPU). Fallback: "htdemucs_ft" (4-stem, local)
      # NOTE: The exact 6-stem checkpoint filename must be verified during implementation.
      # The 12.9755 checkpoint is the 2-stem variant — using it silently produces 2-stem output.
      # Pin audio-separator version in pyproject.toml to prevent silent model changes.
      stem_backend: str = "modal"  # "modal" (cloud GPU) or "local" (audio-separator, for dev/fallback)
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
  # 0. Disk space check — reject early if disk is critically low (~1 GB minimum free).
  #    Each session consumes ~500MB (uploads + stems + remix). Without this check,
  #    a steady stream of remixes can fill the disk before TTL cleanup reclaims space.
  usage = shutil.disk_usage(settings.data_dir)
  if usage.free < 1_000_000_000:  # 1 GB minimum free
      raise HTTPException(507, "Server storage is full. Try again later.")
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
      # If executor.submit() fails (e.g., RuntimeError after shutdown), the session
      # is already created with status "queued" and a client may be connecting to SSE.
      # Set error status and push an error event so the client learns about the failure.
      session.status = "error"
      session.events.put_nowait({"step": "error", "detail": "Server error — please try again", "progress": 0})
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

**Goal:** Split each uploaded song into 6 stems (vocals, drums, bass, guitar, piano, other) using BS-RoFormer SW on cloud GPU via Modal.

- [ ] Install `modal` SDK (`pip install modal`)
- [ ] Create Modal account and set up `modal token new` for authentication
- [ ] Create `services/separation.py` with a **backend-agnostic interface**:
  - `separate_stems(audio_path: Path, output_dir: Path, progress_callback: Callable | None = None) -> dict[str, Path]`
  - Returns mapping: `{"vocals": Path, "drums": Path, "bass": Path, "guitar": Path, "piano": Path, "other": Path}`
  - Output format: WAV (uncompressed, needed for quality processing downstream)
  - Dispatches to `_separate_modal()` or `_separate_local()` based on `settings.stem_backend`
- [ ] Create `services/separation_modal.py` — Modal cloud GPU backend:
  - Define Modal app with GPU container image (A10G):
    ```python
    import modal

    app = modal.App("musicmixer-separation")

    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install("audio-separator[gpu]", "torch", "soundfile")
        # Bake model weights into image via run_commands() to eliminate download on cold start.
        # Verify the correct 6-stem SW checkpoint (not the 12.9755 2-stem variant).
        # VRAM requirement: ~4-6 GB for BS-RoFormer SW.
    )

    @app.function(image=image, gpu="A10G", timeout=180)  # Match client-side per-song timeout to prevent ghost GPU executions
    def separate_stems_remote(audio_bytes: bytes, model_name: str = "bs_roformer_sw") -> dict[str, bytes]:
        """Run BS-RoFormer separation on cloud GPU. Returns stem WAV bytes."""
        # Write input, run audio-separator, read output stems, return as bytes.
        # IMPORTANT: Write stems with explicit subtype='FLOAT' to preserve 32-bit float precision:
        #   sf.write(stem_path, stem_audio, sr, subtype='FLOAT')
        # Default may be PCM_16, silently truncating to 16-bit (96 dB dynamic range loss).
        # Client-side validation after receiving bytes back:
        #   info = sf.info(io.BytesIO(wav_bytes))
        #   assert info.subtype == 'FLOAT', f"Expected float WAV, got {info.subtype}"
        ...
    ```
  - Upload audio file as bytes, receive stem WAV bytes back
  - Handle Modal errors (timeout, cold start, capacity) → raise `SeparationError`
  - **Parallel separation**: Song A and Song B stem separation should be run concurrently using `asyncio.gather` or `concurrent.futures`. Modal supports concurrent function calls, and these are independent operations. This cuts separation time roughly in half (from 60-120s sequential to 35-65s concurrent).
  - **Cold start mitigation**: Use `modal.Function.keep_warm(1)` during active sessions to avoid 10-30s cold starts on consecutive remixes. Commit to `@modal.enter()` (required, not optional) to pre-load the model into GPU memory on container start -- without it, the first `separate_stems_remote` call pays an additional 5-15s model load penalty on top of the cold start. **Pre-warming strategy:** On page load or upload start, fire a lightweight ping to the Modal function (health check) to begin container warm-up before the user finishes uploading. This hides cold start behind upload time at zero cost. **Cold-start-specific progress message:** "Starting cloud GPU... (first remix takes a bit longer)" when separation takes >15s before returning. **Cost estimate:** `keep_warm(1)` on A10G costs ~$2/hour of idle time. If perpetual: ~$50/month. If only during active sessions (4 hrs/day): ~$8/month.
- [ ] Create `services/separation_local.py` — local fallback backend:
  - Uses `audio-separator` + `htdemucs_ft` locally (4-stem: vocals/drums/bass/other)
  - When using 4-stem fallback, return `{"vocals": Path, "drums": Path, "bass": Path, "guitar": None, "piano": None, "other": Path}` — downstream code handles `None` stems gracefully (treated as silent)
  - Requires local PyTorch installation (CPU or MPS for Mac dev)
- [ ] **Post-separation output validation**: After receiving stems back from Modal or local separation, assert stem count:
  ```python
  assert len(stems) == 6, f"Expected 6 stems from {settings.stem_model}, got {len(stems)} — check checkpoint file"
  ```
  This catches the silent failure mode where the wrong checkpoint produces 2-stem output.
- [ ] Handle errors: corrupt files, unsupported formats, Modal timeout, network failure → raise `SeparationError`
- [ ] Progress callback: called at start (upload to Modal), processing, and end (download stems)
- [ ] Write unit test with a short (5-second) test audio fixture. Mock Modal calls in tests.

**Reversibility:** The separation service interface (`separate_stems() -> dict[str, Path]`) is the same regardless of backend. Swap Modal→local or BS-RoFormer→htdemucs by changing `settings.stem_backend` and `settings.stem_model`. No downstream code changes.

**Disk note:** Each song produces ~6 stems as WAV. A 4-minute song ≈ 40MB per stem = ~240MB per song, ~480MB for two songs. Acknowledged in TTL cleanup design.

**Cost note:** Modal A10G is ~$0.001-0.005/remix (pay-per-second, billed only during GPU execution). At typical usage, costs are negligible. `htdemucs_ft` local fallback costs $0 but produces 4 stems at lower quality.

**Latency note:** ~30-60s/song on Modal cloud GPU (A10G) + ~5s network transfer. With parallel separation (both songs concurrently): ~35-65s total. Cold start adds ~10-30s on first request (mitigated with keep_warm + pre-warming on upload).

**Files created:** `src/musicmixer/services/separation.py`, `src/musicmixer/services/separation_modal.py`, `src/musicmixer/services/separation_local.py`, `tests/test_separation.py`, `tests/fixtures/test_short.wav`

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
                                  # After tempo matching via rubberband, re-run librosa.beat.beat_track
                                  # on the stretched audio with start_bpm=target_bpm for an accurate
                                  # post-stretch beat grid. Proportional scaling (beat_frames * ratio)
                                  # accumulates 0.5-1.5s of drift per minute (from compounding BPM detection
                                  # error + tempo variation within the song) and should be used ONLY as a
                                  # last-resort fallback. When the fallback is used, log a warning:
                                  # section boundaries may be audibly misaligned.
        key: str                  # e.g., "C"
        scale: str                # "major" or "minor"
        key_confidence: float
        has_modulation: bool = False  # True if key changes mid-song (detected by comparing first 60%
                                      # and last 40% via essentia.KeyExtractor). 15-25% of pop songs
                                      # modulate (truck driver's modulation in final chorus is common).
                                      # When True, key matching is skipped (key_source set to "none").
        duration_seconds: float
        total_beats: int          # total beats in the song (rounded to nearest bar boundary)
        vocal_prominence_db: float = 0.0  # vocal stem LUFS minus full mix LUFS (dB difference).
                                  # 0 dB = very prominent, -6 dB = moderate, -12 dB = faint,
                                  # < -20 dB = effectively instrumental.
                                  # Measured with 300 Hz - 5 kHz bandpass on vocal stem to reduce bleed inflation.
                                  # Default 0.0, filled by pipeline AFTER stem separation (B3) and BEFORE LLM call (B5).
                                  # Pipeline step: bandpass vocal stem to 300 Hz - 5 kHz, measure LUFS vs full mix using pyloudnorm.
                                  # Critical signal for LLM's vocal_source decision when prompt is vague.
                                  # Sanity check: if both songs have vocal_prominence_db < -12, warn the LLM
                                  # that neither song has clear vocals.
        groove_type: str = "unknown"  # "straight" | "half-time" | "swung" | "unknown"
                                      # Detected from drum stem onset patterns after stem separation.
                                      # Straight-time: IOIs cluster at beat_period/2.
                                      # Half-time: IOIs at 2x beat_period.
                                      # Swung/triplet: IOIs at beat_period * 2/3 and 1/3.
                                      # Used by LLM for groove-aware arrangement decisions.
        energy_regions: list[EnergyRegion]  # sorted by time (temporal order), ~8s windows
    ```
  - BPM via `librosa.beat.beat_track` — **returns both BPM and beat frame positions; store BOTH**
  - **BPM reconciliation (expanded)**: This is a **cross-song reconciliation step** that runs AFTER both songs have been individually analyzed. It cannot happen inside the single-song `analyze_audio` function because it needs both BPMs to compare.
    - **Implementation:** Add a separate `reconcile_bpm(meta_a: AudioMetadata, meta_b: AudioMetadata) -> tuple[AudioMetadata, AudioMetadata]` function called by the pipeline after analyzing both songs.
    - **Expanded interpretation matrix**: For each song, generate all plausible BPM interpretations: {original, halved, doubled, 3/2, 2/3}, then filter to the **70-180 BPM** range (tighter than the naive 60-200 -- 60 BPM is barely a pulse, 200+ is speed metal edge). The 3/2 and 2/3 ratios capture triplet-feel tempos common in hip-hop and jazz (a track at 90 BPM with triplet hi-hats may be better interpreted as 135 BPM).
    - Evaluate all valid pairs across the two songs. Score each pair with interpretation penalties: original = 0% penalty, doubled/halved = 5% penalty, 3/2 and 2/3 = 15% penalty. Score = percentage_gap + penalty_a + penalty_b. Select the pair with the minimum score (not minimum gap). The original BPM should be preferred unless there is strong evidence it is wrong -- non-identity interpretations (especially 3/2 and 2/3) can produce false equivalences (e.g., hip-hop at 95 BPM gets reinterpreted as 142.5 via 3/2). This catches the ~29% of songs where librosa reports the wrong octave AND some additional cases the 2:1-only check misses (e.g., R&B at 72 doubled to 144 brings it within 13% of EDM at 128).
    - **What this doesn't rescue:** Hip-hop (85) + pop (120) = 41% gap has no half/double/triplet relationship that brings them into range. This is the most-attempted mashup pairing. See QR-3 for the alternative strategy (vocals-only stretching).
    - Return updated copies of the metadata (don't mutate originals).
  - Key via `essentia.standard.KeyExtractor` (or librosa chromagram fallback)
  - **Modulation detection**: Run `essentia.KeyExtractor` on the first 60% and last 40% of the song separately. If detected keys differ, set `has_modulation = True` on `AudioMetadata`. When `has_modulation` is True for either song, skip key matching (set `key_source` to `"none"`) and note in the explanation. ~10 lines of additional code, ~1s additional processing.
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
- [ ] **Groove detection** (runs after stem separation on the drum stem):
  - `detect_groove(drum_stem_path: Path, bpm: float) -> str`
  - Run `librosa.onset.onset_detect` on the drum stem
  - Compute inter-onset intervals (IOIs)
  - Classify: straight-time (IOIs cluster at beat_period/2), half-time (IOIs at 2x beat_period), swung/triplet (IOIs at beat_period * 2/3 and 1/3), unknown
  - ~40 lines of code, near-zero latency (drum stem is already loaded)
  - Store result in `AudioMetadata.groove_type`
  - Pass detected groove types to the LLM so it can make informed arrangement decisions (e.g., "Song A is half-time hip-hop, Song B is straight-time pop -- use sections where the instrumental has sparse drums")
  - Include groove mismatch in the compatibility score (QR-3 item 1)
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

### Phase B4b: Compatibility Analysis Endpoint

**Goal:** Implement the `POST /api/analyze` endpoint that runs lightweight compatibility analysis after both songs are uploaded. The frontend calls this automatically when both songs are selected (before the user submits a remix).

- [ ] Create `api/analyze.py` (or add to `api/remix.py`) with `POST /api/analyze` route handler:
  - Accept two audio files (`song_a`, `song_b`) as multipart form data (same format as `POST /api/remix` but without `prompt`)
  - Does NOT acquire the processing lock -- runs independently of remix processing
  - Uses a separate lightweight thread pool (`analyze_executor = ThreadPoolExecutor(max_workers=2)`) for parallel per-song analysis
  - 10-second overall timeout; return 408 if exceeded
- [ ] Implement `compute_compatibility(meta_a: AudioMetadata, meta_b: AudioMetadata) -> CompatibilityResult`:
  - Run BPM reconciliation (`reconcile_bpm()` from Phase B4) on both songs
  - Compute key distance using the key distance algorithm from Phase B6 (extract into a shared utility `services/key_utils.py` so both the compatibility scorer and the processing pipeline can use it)
  - Score based on reconciled BPM distance + circle-of-fifths key distance:
    - **Great** (BPM <10%, CoF 0-1): "These songs should blend really well together."
    - **Good** (BPM 10-20%, or key shift <= 3 semitones): "These songs have some differences, but we can work with it."
    - **Challenging** (BPM 20-35%): "These songs have a different energy. The remix will have more of a mashup feel."
    - **Tough** (BPM >35%, or key shift > 5 semitones): "These songs are very different -- the AI will do its best, but they may not sync up perfectly."
  - Return `CompatibilityResult` with `level`, `message`, and optional `detail` (raw BPM/key values for power users)
- [ ] Per-song analysis pipeline (fits in ~2-3 seconds):
  1. Load audio (`librosa.load` with `duration=60, sr=22050`) -- ~0.3s/song
  2. BPM detection (`librosa.beat.beat_track`) -- ~0.4s/song
  3. Key detection (`essentia.KeyExtractor`) -- ~0.3s/song
  4. Run both songs in parallel = ~1.2s total
  5. BPM reconciliation + key compatibility scoring -- ~0.01s
- [ ] Handle partial analysis failures gracefully: if BPM or key detection fails for one song, return 200 with best-effort partial data and degrade compatibility to "challenging"
- [ ] Return 422 for invalid audio files with `detail` identifying which song failed
- [ ] Response model: `AnalyzeResponse` (already defined in Pydantic Response Models section)
- [ ] Write unit test

**Files created:** `src/musicmixer/api/analyze.py` (or modified `api/remix.py`), `src/musicmixer/services/key_utils.py`
**Files modified:** `src/musicmixer/main.py` (register route)

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
    **MVP constraint:** `vocal_source` determines the split. Vocals come from that song; drums, bass, guitar, piano, and other all come from the other song. No cross-song stem mixing. Source time ranges (`_vocal`/`_instrumental`) select which region of each song to extract; `sections` arrange the extracted material over time.
  - **System prompt design** — the "product brain"; write and test as a first-class deliverable:
    - **MVP constraint and capability boundaries**: Vocals from one song, all instrumentals from the other. No cross-song stem mixing. Explicit capability section: "You can combine stems, adjust volumes per stem (vocals, drums, bass, guitar, piano, other), choose sections, and control arrangement structure. You CANNOT add effects, generate new sounds, or use vocals from both songs. Acknowledge limitations in `warnings`."
    - Available stems: vocals, drums, bass, guitar, piano, other. "Other" contains synths, strings, wind instruments, and anything not captured by the 5 named stems.
    - Song metadata: BPM, key, duration, `total_beats` (eliminates LLM arithmetic), `vocal_prominence_db` (critical signal for vocal_source decision when prompt is vague)
    - **Energy regions (condensed format)**: Do NOT pass raw `EnergyRegion` objects. Pre-process into condensed text summary (~10x fewer tokens):
      ```
      Song A (120 BPM, C major, 240s, 480 beats, vocal_prominence: -3 dB, groove: straight):
        Rhythmic (chorus/drop): 64s-80s [0.95], 128s-144s [0.76]
        Sustained (breakdown): 48s-64s [0.68]
        Sparse (verse): 16s-48s [0.42]
        Moderate (intro/outro): 0s-16s [0.25], 200s-240s [0.18]
        Structure: moderate(0-16s) -> sparse(16-48s) -> sustained(48-64s) -> rhythmic(64-80s) -> ...
      ```
      Groups by `character` label with relative energy values in brackets (allows the LLM to distinguish between a loud chorus [0.95] and a moderately energetic bridge [0.76] within the same character label). The temporal structure summary line (~20-30 additional tokens) gives the LLM the sequential arc that is lost in the character-grouped format. Includes conversion note: "1 beat = 0.5s at 120 BPM."
    - **Arrangement templates** as proportional guides with total-beat constraints (constrains creative space to proven structures while adapting to any source material length):
      ```
      Your sections must sum to approximately {total_available_beats} beats.
      Template A (Standard Mashup): intro(~15%) -> verse(~30%) -> breakdown(~15%) -> drop(~30%) -> outro(~10%)
      Template B (DJ Set): build(~25%) -> vocals in(~25%) -> peak(~25%) -> vocals out(~12%) -> outro(~13%)
      Template C (Quick Hit): intro(~15%) -> vocal drop(~70%) -> outro(~15%)
      Template D (Chill): intro(~25%) -> vocals(~50%) -> outro(~25%)

      If total beats < 48, use Template C (Quick Hit). If 48-96, use Standard Mashup. If > 96, you may use DJ Set or add a second verse.
      ```
      The system prompt must explicitly state "your sections must sum to approximately {total_available_beats} beats" with the actual value injected at call time. Fixed bar counts cause the LLM to produce sections that exceed available audio (triggering truncation) or compress awkwardly; proportional guides adapt to any source material length.
    - **Tempo guidance** (separate from key): `"average"` only when BPMs differ by <15%. 15-30%: prefer vocal source tempo. >30%: skip, explain in `explanation`.
    - **Key guidance** (separate from tempo): Pre-compute the key matching decision BEFORE the LLM call and pass it as a constraint in the LLM context: "Key matching: available (confidence 0.72)" or "Key matching: unavailable (low confidence)". This way the LLM's explanation always matches reality. The processing code uses a two-tier threshold (see Phase B6): >= 0.55 = full shift, 0.40-0.55 = half shift, < 0.40 = skip.
    - **4-stem fallback adaptation**: When `settings.stem_backend == 'local'` and 4-stem fallback is active, modify the system prompt to list only 4 available stems (vocals, drums, bass, other). Remove guitar/piano from the `stem_gains` schema for that call. Add `used_fallback_separation: bool = False` to `RemixPlan` so the frontend can display a note about reduced stem control. Add to the LLM explanation template: "Using simplified stem separation -- guitar and piano adjustments are not available for this remix."
    - **Stem artifact awareness**: "Stem separation is imperfect. Vocal stem may contain instrument traces. Instrumental stems may contain ghost vocals. Bleed is less noticeable during high-energy sections. When the instrumental source song has prominent vocals (`vocal_prominence_db > -6`), avoid purely-instrumental sections longer than 8 beats. Ghost vocals from the original song bleed through instrumental stems and become noticeable during extended instrumental-only passages with no masking from the intentional vocals."
    - **Mixing philosophy** (prevents "correct but boring" output):
      ```
      MIXING PRINCIPLES:
      - Contrast creates energy: if a section has drums at 0.0, the next section's drums at 1.0 will feel powerful
      - When vocals are active, reduce competing stems (guitar, piano, other) to 0.3-0.5 unless the user asks for a "full" sound
      - Every remix should have an energy arc: build, peak, resolve
      - Muted stems (0.0) are a tool, not a failure — silence in the right place is more powerful than sound
      - Use the full 0.0-1.0 range. Avoid keeping all stems at 0.5-0.8 throughout — that produces a flat, unengaging mix
      ```
    - **Transition guidance**:
      ```
      TRANSITION GUIDANCE:
      - "cut": Use between sections at similar energy levels for a punchy feel. Good for drop-to-verse or chorus-to-chorus.
      - "crossfade": Use when energy changes significantly between sections. Default choice for most transitions.
      - "fade": Use for the first section (intro) and last section (outro). Also good for bringing vocals in from silence.
      ```
    - **Handling ambiguous/contradictory/impossible prompts** (majority case):
      1. **Vague** ("make it cool"): Use energy profiles, pick vocals from the song with higher `vocal_prominence_db` (closer to 0 dB = more prominent), use standard template.
      2. **Contradictory** ("vocals from both"): Acknowledge in `warnings`, produce best plan within limits.
      3. **Genre jargon** ("trap", "lo-fi"): Translate to possible actions — tempo, gains, structure.
      4. **Inaudible references** ("guitar solo at 2:30"): Use time range, add warning.
    - **Duration guidance**: "Target remix duration: 60-120 seconds. Minimum 30 seconds. Maximum 180 seconds. Choose source regions and sections to hit this range. Shorter for Quick Hit templates, longer for DJ Set."
    - **Explanation quality**: Non-technical, 2-3 sentences, key creative decisions, no internal jargon.
    - **`warnings` field**: Populate for vague/contradictory/unverifiable prompts. Frontend displays distinctly.
    - **Few-shot examples**: 3 concrete, musically diverse examples with full condensed energy profiles. ~350-400 tokens each (~1,050-1,200 total):
      - **Example 1 (Clear directive, matched tempos):** Prompt "Put Song A's vocals over Song B's beat, boost the bass" with 120/118 BPM, both C major, Song A vocal_prominence -3 dB. Output: Standard mashup, 5 sections, bass at 1.0. Teaches basic schema compliance, stem gain adjustment from prompt.
      - **Example 2 (Vague prompt, tempo mismatch):** Prompt "mix them together, make it sound good" with 88 BPM hip-hop / 125 BPM pop, Song B vocal_prominence -4 dB, Song A vocal_prominence < -15 dB. Output: Song B vocals, instrumental tempo as target, DJ Set template adapted, explanation notes stretch, warning generated. Teaches vocal_prominence_db decision-making, tempo mismatch handling, template adaptation.
      - **Example 3 (Contradictory prompt, genre clash):** Prompt "I want the drums from Song A with the vocals from both songs" with 140 BPM EDM / 72 BPM R&B, EDM vocal_prominence -10 dB, R&B vocal_prominence -2 dB. Output: R&B vocals over EDM, R&B vocals stretched, warning about "vocals from both," Quick Hit template, breakdown sections used. Teaches constraint acknowledgment, extreme tempo handling, graceful degradation.
      - **At least one example MUST demonstrate dramatic gain dynamics** (drums at 0.0 in one section, 0.8+ in the adjacent section) to prevent the "all stems at 0.5-0.8" monotonous gain problem.
  - Uses Claude with tool_use for schema enforcement. **Default to Sonnet** (configurable via `settings.llm_model`). Section-based arrangement requires structured reasoning where Sonnet outperforms Haiku. At $0.003/call, cost is negligible for single-user.
  - **Token budget**: System prompt + few-shot + condensed context under ~3,500-4,000 tokens. Breakdown estimate: system prompt (~400), arrangement templates + adaptation guidance (~200), genre guidance (~250), ambiguity/contradiction handling (~200), mixing philosophy + transition guidance (~150), three few-shot examples (~350 each = ~1,050), song context (~160), and tool schema (~200) total ~2,610 before new additions (groove data, temporal structure summary, duration guidance). At Sonnet pricing ($3/M input), the additional ~1,000-1,500 tokens cost ~$0.003-0.005/call -- negligible.
- [ ] **Post-LLM validation** — two layers:
  - **Time range validation**: `0 <= start_time_X < end_time_X <= duration_X`, min 5.0s. Clamp out-of-bounds. **If clamping by >5s, append to `warnings`**. Log original vs. clamped.
  - **Section list validation**: Sections contiguous (`end_beat[N]` = `start_beat[N+1]`), starts at beat 0, last at/before `total_beats`, `transition_beats` < section length, gains in `[0.0, 1.0]` (see section validation point 7 — gains above 1.0 cause limiter distortion). On failure: **surgical code fix** (fix gaps, clamp, normalize), then re-prompt if still broken, then deterministic fallback. Log every violation.
- [ ] **Duration validation**: If total section duration (converted from beats to seconds at target BPM) is < 30s or > 180s, clamp the source time ranges and re-validate sections. Log the original vs. clamped duration.
- [ ] **Semantic validation**: Verify enum values (`vocal_source`, `tempo_source`, etc.). Check `response.stop_reason` — if `max_tokens`, fall back immediately.
- [ ] **LLM observability** — structured logging:
  ```python
  logger.info("llm_request", session_id=session_id, prompt=prompt,
              song_a_bpm=meta_a.bpm, song_b_bpm=meta_b.bpm,
              song_a_vocal_prominence=meta_a.vocal_prominence_db)
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
      # Higher vocal_prominence_db = more prominent vocals (0 dB = very prominent, -20 dB = faint)
      vocal_src = "song_a" if meta_a.vocal_prominence_db >= meta_b.vocal_prominence_db else "song_b"
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
      eighth = total_beats // 8
      quarter = total_beats // 4
      three_quarter = total_beats * 3 // 4
      seven_eighth = total_beats * 7 // 8
      return RemixPlan(
          vocal_source=vocal_src,
          start_time_vocal=v_start, end_time_vocal=v_end,
          start_time_instrumental=i_start, end_time_instrumental=i_end,
          sections=[
              Section("intro", 0, eighth,
                      {"vocals":0,"drums":0.8,"bass":0.8,"guitar":0.6,"piano":0.5,"other":1}, "fade", 4),
              Section("build", eighth, quarter,
                      {"vocals":0.6,"drums":0.7,"bass":0.8,"guitar":0.5,"piano":0.4,"other":0.5}, "crossfade", 4),
              Section("main", quarter, three_quarter,
                      {"vocals":1,"drums":0.7,"bass":0.8,"guitar":0.5,"piano":0.4,"other":0.5}, "crossfade", 2),
              Section("breakdown", three_quarter, seven_eighth,
                      {"vocals":0.8,"drums":0.0,"bass":0.6,"guitar":0.7,"piano":0.8,"other":0.7}, "crossfade", 4),
              Section("outro", seven_eighth, total_beats,
                      {"vocals":0,"drums":0.6,"bass":0.5,"guitar":0.5,"piano":0.6,"other":0.8}, "crossfade", 8),
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
    1. **Sample rate + channel + format standardization** (MUST be first): Validate and standardize all stems to 44.1kHz stereo 32-bit float before any processing:
       ```python
       def validate_stem(path: Path, expected_sr: int = 44100) -> tuple[np.ndarray, int]:
           info = sf.info(str(path))
           audio, sr = sf.read(str(path), dtype='float32')
           if sr != expected_sr:
               logger.warning(f"Stem {path.name} at {sr}Hz, resampling to {expected_sr}Hz")
               audio = librosa.resample(audio, orig_sr=sr, target_sr=expected_sr, res_type='soxr_hq')
               sr = expected_sr
           if audio.ndim == 1:
               audio = np.column_stack([audio, audio])
           return audio, sr
       ```
       BS-RoFormer and htdemucs_ft both output 44.1kHz but verify and resample if needed. A future model change could silently produce mismatched formats (e.g., 48kHz stems processed as 44.1kHz play 8.7% sharp). **Mono-to-stereo**: Separation models preserve input channel count, so a mono Song A and stereo Song B produce mismatched stems. Convert mono stems to stereo by duplicating the channel. This prevents shape mismatches during mixing and pitch/timing bugs from feeding mismatched sample rates to rubberband. **Missing stems (4-stem fallback)**: When using htdemucs_ft, guitar and piano stems are `None`. Treat as zero-filled arrays of matching length -- downstream code handles gracefully.
    2. **Trim stems** to the time ranges specified in RemixPlan (`start_time_a`/`end_time_a`, etc.). **Important:** Trimming happens BEFORE tempo stretch. The LLM's time ranges are in the original tempo domain. After stretching, the trimmed segment's duration will change proportionally to the tempo ratio (e.g., a 60s segment at 2x stretch = 120s). The processor must account for this when computing final output duration — trim first, then let rubberband change the length.
    3. **Tempo + key matching in a single Rubber Band pass**:
       Tempo stretch and pitch shift MUST be applied in a single `rubberband` CLI invocation per stem. Two separate passes compound artifacts (1-2 dB noise floor increase, phase smearing), especially on vocals.
       - **Tempo matching** with tiered limits. Slowing down vocals (ratio > 1.0) produces significantly more artifacts than speeding up (ratio < 1.0) at the same percentage because slowing requires interpolating new temporal information. Direction-aware thresholds:
         - < 10% BPM difference: stretch either/both silently
         - 10-25%: stretch vocals only (preserve instrumental tempo -- drums are the most stretch-sensitive element, sounding unnatural beyond 8-10% stretch; vocals tolerate 15-20% stretch with formant preservation)
         - 25-30% speedup / 25% slowdown: stretch vocals only, strongly warn in explanation
         - 30-45% speedup / 25-35% slowdown: **vocals-only stretch** -- stretch the vocal stem to match the instrumental's tempo. Do NOT skip tempo matching entirely. Vocals stretched this much sound noticeably different but remain intelligible; an instrumental stretched by the same amount sounds broken (drums especially). This rescues the critical hip-hop (85) + pop (120) pairing (41% gap) that the BPM sanity check cannot solve. Note the stretch in the explanation. For slowdowns, user-facing messaging should be explicit: "The vocals have been significantly slowed to match the beat, so they may sound different from the original recording."
         - > 45% speedup / > 35% slowdown: skip tempo matching entirely -- stretching vocals beyond these limits is unintelligible. Use alternative techniques from the LLM's arrangement (a cappella over breakdowns, loop-based phrases). Let LLM explain why.
         - Use Rubber Band R3 engine via `-3` flag (short form of `--fine`, requires Rubber Band v3.x; fall back to `--crisp 5` for v2.x)
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
         - **Key confidence gate with two-tier shifting**: Wrong pitch shifts are ~2x worse than correct shifts are good, so break-even accuracy is ~67%, not 50%. Use a two-tier system:
           - Confidence >= 0.55: shift by the full computed amount
           - Confidence 0.40-0.55: shift by half the computed distance (hedging -- reduces harm from wrong detections while still improving correct ones)
           - Confidence < 0.40: skip key matching entirely
           The threshold is 0.50 (not lower) because essentia reports 0.4-0.6 confidence for wrong keys ~20-30% of the time -- at lower thresholds, the system pitch-shifts based on detections that are correct only ~50% of the time, producing a *negative* expected value. Note: run a calibration experiment on the GiantSteps Key dataset before shipping to validate the thresholds empirically.
       - **Vocal stem pre-filtering (before Rubber Band):** Separation artifacts (instrumental bleed in vocal stem) compound through time stretching and pitch shifting. For the critical hip-hop + pop pairing (41% stretch), effective vocal SDR drops from 11.3 dB to ~7-8 dB. Before passing vocal stems to Rubber Band, apply a single bandpass filter:
         ```python
         # Single 2nd-order Butterworth bandpass: 200 Hz - 8 kHz
         # Uses a single sosfiltfilt call (one forward+backward pass) instead of two separate
         # filters cascaded. Two separate filters would double the effective order at crossover
         # frequencies (unintended 4th-order response) and double the computation (4 traversals).
         sos = butter(2, [200, 8000], btype='band', fs=sr, output='sos')
         vocal_filtered = sosfiltfilt(sos, vocal_audio, axis=0)
         ```
         - Removes bass/kick bleed below 200 Hz (stretching amplifies it most) and cymbal/hi-hat bleed above 8 kHz (pitch shifting makes it metallic)
         - Vocal quality loss is minimal (fundamentals above 200 Hz, harmonics above 8 kHz add "air" but are not essential)
       - **Skip-at-unity optimization:** When `abs(tempo_ratio - 1.0) < 0.001 and abs(semitones) < 0.01`, return the audio unmodified without routing through Rubber Band. This applies per-stem: if a stem's song is already at the target tempo and the key shift is 0, pass it through unchanged. In the typical case (instrumental song's tempo is the target), all 5-6 instrumental stems skip rubberband entirely. Only the vocal stem (+ any pitch-shifted instrumentals) needs processing. Best case: 1 rubberband call instead of 12.
       - **Parallelize rubberband invocations:** Each rubberband call is an independent subprocess on a different file. Use `concurrent.futures.ProcessPoolExecutor` or `asyncio.gather` with subprocess calls, with 4-6 parallel workers. This turns 6-12 sequential calls (60-144s) into 2-3 batches (15-48s). Emit per-stem progress events during rubberband: "Matching tempo (3/7 stems)..."
       - **Call the `rubberband` CLI directly via subprocess** (not pyrubberband's Python API, which constructs CLI commands internally in undocumented ways that make injecting `--pitch` via rbargs fragile):
         ```python
         import subprocess, soundfile as sf, numpy as np

         def rubberband_process(audio: np.ndarray, sr: int, source_bpm: float,
                                target_bpm: float, semitones: float = 0,
                                is_vocal: bool = False) -> np.ndarray:
             """Single-pass tempo + pitch via rubberband CLI. Guaranteed single invocation.
             Computes time_ratio = source_bpm / target_bpm internally."""
             in_path = tmp_dir / "rb_in.wav"
             out_path = tmp_dir / "rb_out.wav"
             sf.write(in_path, audio, sr)

             # Rubberband -t takes a TIME RATIO (output_duration / input_duration), NOT a speed ratio.
             # To go from 90 BPM to 120 BPM: time_ratio = 90/120 = 0.75 (shorter output = faster).
             # To go from 120 BPM to 90 BPM: time_ratio = 120/90 = 1.333 (longer output = slower).
             # Formula: time_ratio = source_bpm / target_bpm
             # CRITICAL: Inverting this (target/source) produces the opposite of the intended stretch.
             time_ratio = source_bpm / target_bpm  # e.g., 90/120 = 0.75 to speed up

             cmd = ["rubberband", "-t", str(time_ratio)]
             if semitones != 0:
                 cmd += ["-p", str(semitones)]
             cmd += ["-3"]  # R3 engine (v3.x); `-3` is the short form of `--fine` for R3. If only v2 is installed, use `--crisp 5` (closest R2 equivalent).
             if is_vocal and semitones != 0:
                 cmd += ["--formant"]
             cmd += [str(in_path), str(out_path)]

             subprocess.run(cmd, check=True, capture_output=True, timeout=120)
             result, _ = sf.read(out_path)
             return result
         ```
         This calls the `rubberband` CLI once per stem with both `-t` (tempo) and `-p` (pitch) flags in a single pass. Add a startup check that verifies `rubberband --version` returns v3.x; log a warning and fall back to `--crisp 5` if only v2.x is available.
    3c. **Beat phase alignment**: After tempo matching, both songs are at the same BPM but their downbeats may be offset. Even a 50ms offset is audible as a "flamming" effect. Compute the offset between the vocal's beat grid and the instrumental's beat grid using the first 16 beats, then shift the vocal audio by the median offset to align downbeats (~15 lines).
    4. **Cross-song level matching**:
       - Stems from different songs can have wildly different absolute loudness out of the separator. Measure the LUFS of the **unfiltered** vocal stem and the summed instrumental stems using `pyloudnorm` **before** applying vocal pre-filtering (step 3's bandpass). Pre-filtering removes low-frequency energy (chest resonance, fundamental below 200 Hz) which reduces measured LUFS by 1-3 dB, causing the system to over-boost vocals. Measure first, then filter, then apply the gain offset.
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
             # How much of the instrumental's energy competes with the vocal.
             # Use energy OUTSIDE the vocal band as denominator (not full spectrum)
             # to avoid sub-bass bias — bass-heavy hip-hop produces a low density_ratio
             # even when the mid-range is busy with hi-hats and synths.
             low_band = (freqs >= 80) & (freqs < 200)
             high_band = (freqs >= 5000) & (freqs <= 16000)
             inst_non_vocal_energy = np.mean(psd_inst[low_band]) + np.mean(psd_inst[high_band])
             density_ratio = inst_vocal_band_energy / (inst_non_vocal_energy + 1e-10)
             # Map: sparse instrumental (low ratio) = +1-2 dB, dense = +5-8 dB
             # Note: breakpoints [0.1, 0.3, 0.5, 0.7] should be calibrated empirically
             # against 10-15 real song pairings during implementation.
             offset_db = np.interp(density_ratio, [0.1, 0.3, 0.5, 0.7], [1.0, 3.0, 5.0, 7.0])
             return float(np.clip(offset_db, 1.0, 10.0))  # Raised upper clip from 8.0 to 10.0 for dense instrumentals
         ```
         This is ~15 lines, uses only scipy (already a transitive dependency via librosa), and directly measures what determines the correct balance: how much the instrumental competes with the vocal in its frequency range. No genre metadata needed. Per-stem volume adjustments are now embedded in `Section.stem_gains` (QR-1) and applied during arrangement rendering (step 7).
    5. *(Removed -- per-stem volume adjustments are now part of `Section.stem_gains` in QR-1, applied by the arrangement renderer in step 7, not as a separate pipeline step.)*
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
       - **Silence guard (critical):** If the final mix is near-silent, `integrated_loudness()` returns `-inf`, producing `inf` gain that corrupts the audio buffer to NaN. Guard before applying:
         ```python
         current_lufs = meter.integrated_loudness(mixed)
         if current_lufs < LUFS_FLOOR:  # -40.0
             logger.warning(f"Final mix near-silent ({current_lufs:.1f} LUFS), skipping normalization")
         else:
             gain_db = target_lufs - current_lufs
             gain_db = np.clip(gain_db, -12.0, 12.0)  # Safety cap prevents extreme amplification
             mixed = mixed * (10 ** (gain_db / 20.0))
         ```
       - **Important:** Normalize the final mix, NOT individual stems before summing. Per-stem normalization to -14 LUFS causes the sum to land around -11 LUFS, forcing the peak limiter to crush the audio and produce audible pumping/distortion.
    9. **Peak limiter**: After LUFS normalization, apply soft-knee clipping at -1.0 dBTP ceiling to prevent clipping during MP3 encoding while preserving average loudness. Use 4x oversampled true-peak measurement per ITU-R BS.1770 (not sample-peak via `np.max(np.abs())`), which misses intersample peaks that can exceed the ceiling after the limiter, causing DAC clipping:
       ```python
       from scipy.signal import resample_poly

       def true_peak(signal: np.ndarray) -> float:
           """Measure true peak via 4x oversampling (practical approximation of ITU-R BS.1770-4).
           NOTE: resample_poly uses a Hamming-windowed FIR, not the ITU-specified 4-phase 12-tap
           polyphase FIR coefficients. This can underestimate true peaks by up to ~0.3 dB.
           Acceptable for MVP; for post-MVP compliance, use pyloudnorm's true-peak measurement
           or implement the ITU filter coefficients directly."""
           if signal.ndim == 2:
               return max(true_peak(signal[:, ch]) for ch in range(signal.shape[1]))
           upsampled = resample_poly(signal, 4, 1)
           return float(np.max(np.abs(upsampled)))

       def soft_clip(signal: np.ndarray, ceiling: float, knee_db: float = 6.0) -> np.ndarray:
           """Soft-knee clipper. Below threshold: UNCHANGED. Knee region: quadratic compression
           (C1 continuous at both boundaries). Above ceiling: hard limit.
           knee_db=6.0 handles 3-6 dB of peak overshoot from LUFS normalization + section dynamics."""
           knee_linear = 10 ** (knee_db / 20.0)
           threshold = ceiling / knee_linear
           result = signal.copy()
           abs_signal = np.abs(signal)
           # Knee region: quadratic parabolic compression (C1 continuous).
           # At t=0 (threshold): output=threshold, derivative=1 (matches passband).
           # At t=1 (ceiling): output=ceiling, derivative=0 (smooth into hard limit).
           # This avoids the gain discontinuity that tanh mapping creates (tanh(1)=0.76,
           # causing output to jump from 0.76*(ceiling-threshold) to ceiling at the boundary).
           knee_mask = (abs_signal > threshold) & (abs_signal <= ceiling)
           if np.any(knee_mask):
               x = abs_signal[knee_mask]
               knee_width = ceiling - threshold
               t = (x - threshold) / knee_width  # 0 to 1 in knee region
               compressed = threshold + knee_width * (2*t - t*t)  # Parabolic: f(0)=0, f(1)=1, f'(0)=2, f'(1)=0
               result[knee_mask] = np.sign(signal[knee_mask]) * compressed
           # Hard limit above ceiling
           over_mask = abs_signal > ceiling
           result[over_mask] = np.sign(signal[over_mask]) * ceiling
           return result

       ceiling = 10 ** (-1.0 / 20.0)  # -1.0 dBTP ≈ 0.891
       peak = true_peak(mixed)
       if peak > ceiling:
           mixed = soft_clip(mixed, ceiling)
       ```
       **Why soft-knee clip, not global gain reduction or `np.tanh`:** Global gain reduction (`mixed * (ceiling / peak)`) pulls the *entire* mix down to tame a single transient peak. When remix stems have peaks 6-10 dB above average (common), this produces a quiet, thin mix that drifts far below the -14 LUFS target. `np.tanh` saturation is also wrong: it applies nonlinear compression to the ENTIRE signal, not just peaks above the ceiling. Every sample is distorted, introducing odd-harmonic distortion across the full dynamic range. The soft-knee clipper leaves samples below the threshold bit-identical (zero modification), smoothly compresses the knee region, and hard-limits above the ceiling. Post-MVP: a true lookahead limiter would be even cleaner for transient handling.
    10. **Fade-in/fade-out**: Apply 2-second fade-in and 3-second fade-out in numpy (not pydub) to avoid precision loss from pydub's 16-bit internal representation. Use equal-power (cosine-squared) fade curves:
        ```python
        fade_in = np.cos(np.linspace(np.pi/2, 0, n_in)) ** 2   # 0 -> 1
        fade_out = np.cos(np.linspace(0, np.pi/2, n_out)) ** 2  # 1 -> 0
        ```
        **Prevent double-fading:** Skip global fade-in if first section's `transition_in == "fade"`. Skip global fade-out if last section's `label == "outro"`.
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
  - **Cooperative shutdown**: Between each major pipeline step, check `_shutdown_requested.is_set()` and raise `PipelineError("Server shutting down")` if True. This prevents orphaned Modal GPU calls and partial file writes during SIGTERM. The lifespan uses `executor.shutdown(wait=True)` with a reasonable timeout (30s) so the pipeline can finish cleanly.
  - **Sets session attributes directly** in addition to pushing queue events. The queue may have no consumer (SSE client disconnected), so the pipeline must always update `session.status`, `session.remix_path`, and `session.explanation` on the `SessionState` object at key transitions (processing start, completion, error). The status endpoint reads these attributes, not the queue.
  - Pushes `ProgressEvent` dicts to the queue at each step:
    ```python
    class ProgressEvent(BaseModel):
        step: Literal["separating", "analyzing", "interpreting", "processing", "rendering", "complete", "error", "keepalive"]
        detail: str    # Human-readable description
        progress: float  # 0.0 to 1.0
        explanation: str | None = None     # Present when step == "complete"
        warnings: list[str] | None = None  # Present when step == "complete" — LLM caveats and fallback notes
        used_fallback: bool | None = None  # Present when step == "complete" — True when deterministic fallback fired
    ```
  - Progress sequence:
    ```
    {"step": "separating", "detail": "Extracting stems from both songs...", "progress": 0.10}
    {"step": "separating", "detail": "Analyzing audio patterns...", "progress": 0.18}   (synthetic, after ~15s)
    {"step": "separating", "detail": "Separating instruments...", "progress": 0.26}     (synthetic, after ~30s)
    {"step": "separating", "detail": "Finalizing stem extraction...", "progress": 0.35} (synthetic, after ~45s)
    {"step": "analyzing", "detail": "Detecting tempo and key...", "progress": 0.50}
    {"step": "interpreting", "detail": "Planning your remix...", "progress": 0.58}
    {"step": "processing", "detail": "Matching tempo (1/7 stems)...", "progress": 0.65}
    {"step": "processing", "detail": "Matching tempo (5/7 stems)...", "progress": 0.72}
    {"step": "processing", "detail": "Normalizing loudness...", "progress": 0.80}
    {"step": "processing", "detail": "Mixing stems...", "progress": 0.85}
    {"step": "rendering", "detail": "Building your remix...", "progress": 0.90}
    {"step": "rendering", "detail": "Rendering final mix...", "progress": 0.95}
    {"step": "complete", "detail": "Remix ready!", "progress": 1.0, "explanation": "...", "warnings": [...], "usedFallback": false}
    # Progress percentages re-weighted to match actual time distribution:
    # 0.00-0.50: Stem separation (50% of bar for 50-60% of real time)
    # 0.50-0.58: Analysis
    # 0.58-0.65: LLM interpretation
    # 0.65-0.85: Rubberband + processing
    # 0.85-0.95: Rendering (arrangement + ducking + mixing + export)
    # 0.95-1.00: Finalize
    ```
  - On any error: push `{"step": "error", "detail": "user-friendly message", "progress": 0}` and clean up session files
  - **Processing lock is acquired by the POST handler** (not the pipeline) and released via a `pipeline_wrapper` function's `try/finally` block (see Phase B2). The pipeline itself (`run_pipeline`) does not receive or manage the lock — the wrapper keeps it pure. `finally` executes even on `BaseException` subclasses (`KeyboardInterrupt`, `SystemExit`), preventing orphaned locks on thread death.
    ```python
    def pipeline_wrapper(session_id: str, ...):
        global _pipeline_start_mono
        try:
            _pipeline_start_mono = time.monotonic()
            session.status = "processing"
            run_pipeline(session_id, ...)
            # Pipeline sets session.status, session.remix_path, session.explanation
            # directly before pushing the complete event to the queue.
        except BaseException as e:
            # Push error event so SSE client learns about failure.
            # Map MusicMixerError subclasses to their messages; all other exceptions
            # get a generic message to avoid leaking file paths, stack traces, or API keys.
            try:
                session.status = "error"
                detail = str(e) if isinstance(e, MusicMixerError) else \
                    "Something went wrong while creating your remix. Please try again."
                if not isinstance(e, MusicMixerError):
                    logger.exception("Unhandled pipeline error", session_id=session_id)
                session.events.put_nowait({"step": "error", "detail": detail, "progress": 0})
            except Exception:
                pass  # Best effort — don't mask the original exception
            raise
        finally:
            _pipeline_start_mono = None
            # Unconditional release: this is the only code path that releases the lock.
            # Catch RuntimeError in case the lock was already released (e.g., by the watchdog
            # after a native crash — see watchdog below). Do NOT use `processing_lock.locked()`
            # as a guard: Lock.locked() is not ownership-aware and introduces a TOCTOU race.
            try:
                processing_lock.release()
            except RuntimeError:
                pass  # Already released (e.g., by watchdog)
    ```
- [ ] Create SSE endpoint `GET /api/remix/{session_id}/progress`:
  - Returns `StreamingResponse` with `text/event-stream` content type and anti-buffering headers:
    ```python
    StreamingResponse(
        event_stream(session),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # prevents nginx buffering
            "Connection": "keep-alive",
        }
    )
    ```
    These prevent reverse proxies (nginx, Cloudflare) from buffering SSE events, which would cause the frontend timeout to fire during normal operation.
  - On connect: check session status. If already complete, send final event immediately. If processing, send latest event then continue streaming.
  - Async generator reads from `session.events` queue via `run_in_executor`
  - Keepalive: The backend's 5-second queue timeout already sends keepalives frequently. The SSE data event format (`{"step":"keepalive"}`) must be a `data:` event, not an SSE comment -- `EventSource.onmessage` does not fire for comments.
- [ ] Create status endpoint `GET /api/remix/{session_id}/status`:
  - Returns current session state as JSON (not SSE)
  - Used by frontend for quick reconnection check after page refresh
  - **Note:** `queued` is an internal status. The status endpoint maps it to `processing` with `progress: 0` before returning — the frontend never sees `queued`.
  - **Explanation data flow:** Pipeline sets `session.explanation = remix_plan.explanation` before pushing the `complete` event. Status endpoint reads `session.explanation` directly (not from the queue). This guarantees the explanation is available even if the SSE `complete` event was consumed by a different client or never consumed.
  - Response: `{"status": "processing", "progress": 0, "detail": "Starting..."}` or `{"status": "processing", "progress": 0.45, "detail": "..."}` or `{"status": "complete", "explanation": "...", "warnings": [...], "used_fallback": false}` or `{"status": "error", "detail": "..."}`
  - **Note:** The `remix_url` field has been removed from `SessionStatusResponse` -- the audio URL is deterministic from the session ID (`/api/remix/{session_id}/audio`), and the frontend constructs it via `getAudioUrl(sessionId)`. Including it in the response was dead data that created a confusing API contract. Do NOT use `session.remix_path` (a filesystem path) in any response -- that would be a path disclosure vulnerability.
- [ ] Create audio endpoint `GET /api/remix/{session_id}/audio`:
  - Serves the rendered remix MP3 file via `FileResponse`
  - Returns 404 if remix doesn't exist or has expired
- [ ] Implement TTL cleanup via FastAPI lifespan:
  ```python
  # Track pipeline start time for watchdog (set by pipeline_wrapper, cleared on completion)
  _pipeline_start_mono: float | None = None

  @asynccontextmanager
  async def lifespan(app: FastAPI):
      cleanup_task = asyncio.create_task(cleanup_loop())
      watchdog_task = asyncio.create_task(lock_watchdog())
      _shutdown_requested.clear()
      yield
      _shutdown_requested.set()
      cleanup_task.cancel()
      watchdog_task.cancel()
      executor.shutdown(wait=True, cancel_futures=True)  # Wait for pipeline to finish (see M-6 cooperative shutdown)

  async def cleanup_loop():
      while True:
          await asyncio.sleep(settings.cleanup_interval_seconds)
          try:
              now = time.monotonic()
              with sessions_lock:
                  expired = [
                      (sid, s) for sid, s in sessions.items()
                      if s.status not in ("processing", "queued")
                      and (now - s.created_at_mono) > (
                          # Error sessions expire after 15 minutes (no user-visible value,
                          # but they consume the same ~500MB disk as completed sessions).
                          settings.error_ttl_seconds if s.status == "error"
                          else settings.remix_ttl_seconds + 300  # 5-min grace period for in-progress streaming
                      )
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

  # Cooperative shutdown flag — checked by the pipeline between major steps.
  # Prevents orphaned cloud GPU calls and partial file writes during SIGTERM.
  _shutdown_requested = threading.Event()

  async def lock_watchdog():
      """Safety net for edge cases that `finally` cannot cover: native segfault in a
      C extension (numpy, scipy, rubberband), os._exit() in a dependency, or OOM kill.
      If the processing lock is held longer than max_sse_duration + 60s (21 min),
      force-release it so the server can accept new remixes."""
      while True:
          await asyncio.sleep(60)
          if processing_lock.locked() and _pipeline_start_mono is not None:
              elapsed = time.monotonic() - _pipeline_start_mono
              if elapsed > settings.max_sse_duration_seconds + 60:  # 21 minutes
                  logger.error("Watchdog: processing lock held for %ds — forcing release", int(elapsed))
                  try:
                      processing_lock.release()
                  except RuntimeError:
                      pass
  ```
  - Cleanup uses `sessions_lock` when mutating the sessions dict (thread-safe with pipeline)
  - File deletion runs via `asyncio.to_thread` to avoid blocking the event loop
  - Cleanup skips sessions in `"processing"` or `"queued"` status
  - Error sessions expire after `error_ttl_seconds` (15 min) instead of the full 3-hour TTL — no user-visible value but same ~500MB disk cost
  - Completed sessions get a 5-minute grace period beyond `remix_ttl_seconds` to allow in-progress `FileResponse` streaming to finish
  - Log session count and lock state each cycle for operational visibility
  - **Intermediate artifact cleanup:** After a pipeline completes successfully, delete the uploads and stems directories for that session (keeping only the final remix MP3). This reduces per-session disk from ~500MB to ~10MB for completed sessions. Run in the pipeline's finally block or as a post-completion step:
    ```python
    # In pipeline, after successful completion:
    for subdir in ("uploads", "stems"):
        session_dir = settings.data_dir / subdir / session_id
        shutil.rmtree(session_dir, ignore_errors=True)
    ```
- [ ] **Per-step timeouts** (prevents hanging indefinitely on any single step):
  - Modal stem separation: 180 seconds per song (matches Modal server-side `timeout=180` to prevent ghost GPU executions). This is a per-*song* timeout on the `concurrent.futures.Future` wrapping each Modal call. With parallel separation, total wall time is `max(song_a_time, song_b_time)` which will be at most 180s.
  - LLM call: 30 seconds (already specified in Phase B5 -- reference it)
  - Each rubberband subprocess: 120 seconds (already specified in Phase B6 -- reference it)
  - Spectral ducking: 60 seconds (catches performance regressions if smoothing optimization is not applied)
  - Total pipeline: 20 minutes (already specified as SSE cap)
  If Modal hangs without the per-step timeout, the pipeline thread blocks indefinitely until the 20-minute SSE global cap fires.
- [ ] Error handling: any service raising a `MusicMixerError` is caught by the pipeline, logged, and emitted as an SSE error event

**Files created:** `src/musicmixer/services/pipeline.py`
**Files modified:** `src/musicmixer/api/remix.py`, `src/musicmixer/main.py`

---

### Backend Phase Summary

| Phase | Depends On | Can Parallelize With | Estimated Complexity |
|-------|-----------|---------------------|---------------------|
| B1: Scaffolding | — | F1 (frontend scaffolding) | Low |
| B2: Upload | B1 | F2 (upload UI) | Low-Medium (validation) |
| B3: Separation | B1 | B4, F2, F3 | Medium (Modal setup, BS-RoFormer config, local fallback) |
| B4: Analysis | B1 | B3, B5, F3 | Low-Medium (BPM sanity check) |
| B4b: Compatibility Endpoint | B1, B4 | B3, B5, F2 | Low (reuses B4 analysis + B6 key distance) |
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
modal                   # cloud GPU for stem separation (BS-RoFormer)
audio-separator         # stem separation — used inside Modal container AND for local fallback (brings PyTorch locally). Pin version in pyproject.toml.
librosa                 # BPM detection, audio loading
essentia                # key detection
pyrubberband            # tempo/key matching
pydub                   # audio mixing, fades, export
soundfile               # WAV I/O
pyloudnorm              # LUFS loudness normalization
numpy                   # float mixing, peak limiter (also a transitive dep of librosa)
soxr                    # high-quality resampling (explicit dep; without it, librosa falls back to inferior resampy)
scipy                   # spectral ducking bandpass filter (also a transitive dep of librosa, but explicit is safer)
anthropic               # LLM calls
```

**System dependencies (document in CLAUDE.md and README):**
```
ffmpeg                  # required by pydub
rubberband              # required by pyrubberband
libsndfile              # required by soundfile
```

**PyTorch installation (only needed for local fallback):**
```
# PyTorch is NOT required locally when using Modal (cloud GPU) for separation.
# Only install for local htdemucs_ft fallback or development:

# macOS development (CPU/MPS):
torch --extra-index-url https://download.pytorch.org/whl/cpu

# The Modal container image installs its own PyTorch + CUDA inside the cloud environment.
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
  export type ProgressStep = 'separating' | 'analyzing' | 'interpreting' | 'processing' | 'rendering' | 'complete' | 'error' | 'keepalive';

  export type ProgressEvent = {
    step: ProgressStep;
    detail: string;
    progress: number;
    explanation?: string;
    warnings?: string[];        // Present when step === 'complete' — LLM caveats and fallback notes
    usedFallback?: boolean;     // Present when step === 'complete' — true when deterministic fallback was used
  };

  export type CreateRemixResponse = {
    session_id: string;
  };

  export type SessionStatus = {
    status: 'processing' | 'complete' | 'error';  // Backend maps 'queued' → 'processing' before returning
    progress?: number;
    detail?: string;
    explanation?: string;
    warnings?: string[];        // Present when status === 'complete'
    usedFallback?: boolean;     // Present when status === 'complete'
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
    | { phase: 'ready'; sessionId: string; explanation: string; warnings: string[]; usedFallback: boolean }
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
    | { type: 'REMIX_READY'; explanation: string; warnings: string[]; usedFallback: boolean }
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

  export async function analyzeCompatibility(songA: File, songB: File, signal?: AbortSignal): Promise<AnalyzeResponse>
  // POST /api/analyze — lightweight compatibility check (~2-3s)
  // Called automatically after both songs are uploaded (non-blocking)
  // Uses fetch for the request. Accept optional AbortSignal so the caller can cancel
  // in-flight analysis when a song is replaced (prevents stale results overwriting fresh ones).
  // KNOWN UX ISSUE (MVP): Files are re-uploaded for this endpoint — the same files will be
  // uploaded again when the user submits POST /api/remix. On slow connections this doubles
  // upload time. Post-MVP: add a POST /api/upload endpoint that stores files by ID,
  // then both /api/analyze and /api/remix reference stored files.

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
  - **No-event timeout**: Step-aware timeouts to prevent false error dispatches during long processing steps. Default: 120 seconds. During `"separating"` step (which can take 90s per song + cold start): 180 seconds. During other steps: 60 seconds. Dispatch `ERROR` action ("Processing is taking longer than expected") only when the step-aware timeout is exceeded.
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
           OR: {"status":"complete","explanation":"...","warnings":[...],"used_fallback":false}
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
    step: Literal["separating", "analyzing", "interpreting", "processing", "rendering", "complete", "error", "keepalive"]
    detail: str
    progress: float
    explanation: str | None = None
    warnings: list[str] | None = None
    used_fallback: bool | None = None

class SessionStatusResponse(BaseModel):
    status: Literal["processing", "complete", "error"]
    progress: float | None = None
    detail: str | None = None
    explanation: str | None = None
    warnings: list[str] | None = None
    used_fallback: bool | None = None
    # NOTE: Use response_model_exclude_none=True on the endpoint (or model_config)
    # to omit null fields from JSON — prevents TypeScript null vs undefined issues.
    model_config = {"exclude_none": True}

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
| Stem granularity | 6-stem separation (vocals/drums/bass/guitar/piano/other) with BS-RoFormer SW on cloud GPU (Modal). MVP uses a **vocals + instrumentals model**: vocals from one song, all instrumentals from the other. No cross-song stem mixing. 6-stem gives the LLM fine-grained per-stem volume control (e.g., "boost bass", "mute piano", "bring up guitar"). Fallback: `htdemucs_ft` (4-stem, local) if Modal is unavailable. Upgrade path: cross-song stem mixing post-MVP. |
| File format/size limits | MP3 and WAV. Max 50MB per file, max 10 minutes. Validated by parsing the audio, not just MIME type. |
| Processing architecture | Server-side everything. Pipeline runs in ThreadPoolExecutor with threading.Queue for progress events. Single-worker, processing lock, one remix at a time. |
| Processing time | Cloud GPU (Modal): ~2-5 min total (median ~3 min with parallelized separation + optimized rubberband; worst case ~5 min with cold start + long songs + large tempo gap). Separation: ~35-65s (parallel) or ~60-120s (sequential). Rubberband: ~5-60s depending on how many stems need processing. Analysis + LLM + rendering: ~10-30s. Local fallback (CPU): ~12-15min total. Cold start adds ~10-30s on first remix (mitigated by pre-warming on upload). Progress UI keeps users informed either way. |
| First-time experience | Tagline + static example prompts above the input field + self-explanatory 2-upload + 1-text-field interface. |
| Error handling | Multi-layer: upload validation (413/422), pipeline errors via SSE, LLM fallback to default plan, audio 404 for expired remixes. Full error taxonomy defined. |
| Async model | ThreadPoolExecutor with threading.Queue. Pipeline pushes events, SSE endpoint reads them via async wrapper. |
| Concurrency | Single-worker uvicorn, global processing lock. Second request gets 429. |
| Output length | Duration emerges from the LLM's source region and section choices, constrained by system prompt guidance (target 60-120s, min 30s, max 180s) and post-LLM validation that clamps out-of-range durations. LLM also picks start/end times for each song. |
| Section selection | Two levels: (1) LLM picks source regions via `start_time_vocal`/`end_time_vocal`/`start_time_instrumental`/`end_time_instrumental` fields, guided by energy profile data. (2) LLM designs section-based arrangement (QR-1) over the extracted material — `sections: list[Section]` with per-stem gains and transitions. Energy profiles (RMS + onset density in ~8s windows) guide both levels. Post-MVP: add `allin1` package for labeled structural analysis. |
| Loudness matching | Per-stem gains now embedded in `Section.stem_gains` (QR-1) — applied per-section during arrangement rendering. Cross-song level matching (LUFS) applied before arrangement. Final LUFS normalization with `pyloudnorm` to -14 LUFS on the **final summed mix** (not per-stem). Spectral ducking (QR-1) carves frequency space for vocals. |
| Frequency clashing | Deferred to post-MVP. Not needed in vocals+instrumentals model — all instrumental stems come from the same song (already balanced). When cross-song stem mixing is added, will use scipy role-based filtering (high-pass on non-bass, low-pass on competing bass). |
| Build vs buy | Run BS-RoFormer on Modal cloud GPU (~$0.001-0.005/remix). All other processing local. `htdemucs_ft` available as fully-local fallback. No API exists for the full prompt-based pipeline — this is genuinely novel. |

---

## Risk Analysis

| Risk | Impact | Mitigation |
|------|--------|------------|
| Modal cloud GPU latency or downtime | Separation blocked | Automatic fallback to `htdemucs_ft` local (4-stem, slower but functional). Monitor Modal status. Cold start mitigated with `keep_warm`. |
| Modal cost spike at scale | Unexpected bills | Pay-per-second billing caps exposure. At ~$0.005/remix, even 1000 remixes/month = $5. Set Modal spending limits. |
| Local-only fallback is too slow (12+ min on CPU) | Users abandon if Modal unavailable | Set expectations with progress UI. Document that GPU/Apple Silicon recommended for local fallback. |
| PyTorch installation issues (local fallback only) | Blocks local development | PyTorch only needed for local fallback, not for Modal path. Pin exact version + variant in pyproject.toml. Document setup steps. |
| Essentia installation fails | Blocks key detection | Pluggable interface: auto-fallback to librosa chromagram. Designed in from the start. |
| BPM half/double-time error (~29% of songs) | Wrong tempo = terrible remix | Expanded BPM reconciliation: 5 interpretations per song (original, half, double, 3/2, 2/3) in 70-180 BPM range. Reduces failure from ~73% to ~53% for cross-genre pairings. |
| Tempo stretch >30% sounds bad | Artifact-ridden output | Vocals-only stretching for 30-50% gaps (vocals tolerate more stretch than instruments). Default to instrumental source tempo. Skip entirely only above 50%. Pre-upload compatibility signal sets expectations. |
| Pitch-shifted vocals sound chipmunk/demonic | Unnatural output | Formant preservation enabled by default for vocal stems. Cap at +/- 4 semitones for vocals. |
| Two tracks layered sound like two tracks playing | Unmusical output | Section-based arrangement (QR-1): LLM-generated section timeline with per-stem gains, beat-aligned transitions, spectral ducking. Vocals+instrumentals model keeps it simple. See QR-1 for full solution. |
| LLM produces bad remix plan | Bad remixes | Default to Sonnet for section-based arrangement quality. Condensed energy data + arrangement templates constrain LLM creative space. Few-shot examples for Haiku/Sonnet reliability. Surgical code-level validation fixes common schema errors. `warnings` field surfaces uncertainty. Deterministic data-driven fallback with `used_fallback` flag and honest user messaging. Explanation shown to user. Structured LLM observability logging for debugging. |
| Disk fills up (~500MB per session) | Server crashes | TTL cleanup every 5 minutes. Disk space check (1 GB min free) before accepting uploads — returns 507. Error sessions cleaned after 15 min (not 3 hours). Intermediate artifacts (uploads + stems) deleted after successful pipeline completion, reducing per-session disk from ~500MB to ~10MB. |
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
    stem_gains: dict[str, float]  # {"vocals": 1.0, "drums": 0.7, "bass": 0.8, "guitar": 0.6, "piano": 0.5, "other": 0.4}
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
    {"label": "intro",     "start_beat": 0,  "end_beat": 16, "stem_gains": {"vocals": 0.0, "drums": 0.9, "bass": 0.8, "guitar": 0.7, "piano": 0.6, "other": 1.0}, "transition_in": "fade", "transition_beats": 4},
    {"label": "verse",     "start_beat": 16, "end_beat": 48, "stem_gains": {"vocals": 1.0, "drums": 0.7, "bass": 0.8, "guitar": 0.5, "piano": 0.4, "other": 0.5}, "transition_in": "crossfade", "transition_beats": 4},
    {"label": "breakdown", "start_beat": 48, "end_beat": 64, "stem_gains": {"vocals": 0.8, "drums": 0.0, "bass": 0.6, "guitar": 0.7, "piano": 0.8, "other": 0.7}, "transition_in": "crossfade", "transition_beats": 2},
    {"label": "drop",      "start_beat": 64, "end_beat": 96, "stem_gains": {"vocals": 1.0, "drums": 1.0, "bass": 1.0, "guitar": 0.8, "piano": 0.6, "other": 0.8}, "transition_in": "cut", "transition_beats": 0},
    {"label": "outro",     "start_beat": 96, "end_beat": 112, "stem_gains": {"vocals": 0.0, "drums": 0.6, "bass": 0.5, "guitar": 0.5, "piano": 0.7, "other": 0.8}, "transition_in": "crossfade", "transition_beats": 8}
  ]
}
```

**Mixer logic** — deterministic renderer of the LLM's plan:
1. Convert beat positions to sample positions using the **beat grid** from `AudioMetadata.beat_frames` (NOT constant-BPM math — real songs have tempo drift; the beat grid captures actual beat positions). After tempo matching, scale beat frame positions proportionally by the tempo ratio.
2. For each section, extract the sample range from each stem
3. Apply per-stem gain from `stem_gains`
4. Apply transition envelopes at section boundaries:
   - **"fade"**: Cosine fade-in over `transition_beats` at section start. Each stem's gain ramps from 0 to the section's `stem_gains` value.
   - **"crossfade"**: **Per-stem equal-power gain interpolation** (NOT summed-audio crossfade). During the overlap region (`transition_beats`), each stem's gain uses equal-power interpolation: `gain(t) = sqrt(gain_prev * (1-t) + gain_next * t)` where `t` is the cosine-smoothed transition progress. This interpolates power (gain-squared) linearly and takes the square root, ensuring constant perceived loudness. Linear gain interpolation causes volume bumps/dips at transition midpoints, especially when all stems change in the same direction.
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
6. All `stem_gains` keys must be in `{"vocals", "drums", "bass", "guitar", "piano", "other"}` — add missing keys with default 0.0, remove unknown keys. When using 4-stem fallback (htdemucs_ft), "guitar" and "piano" keys are ignored.
7. All `stem_gains` values in range `[0.0, 1.0]` — clamp if out of range. Do NOT allow gains above 1.0: when 5-6 stems are summed at >1.0x gain each, the signal entering the limiter can be 10+ dB hot, causing audible distortion. If the LLM wants a stem louder, it should attenuate other stems instead (standard mixing practice, per the "Use the full 0.0-1.0 range" mixing philosophy).
8. Total beat range must be within available audio duration (after tempo matching, converted using `beat_frames`) — truncate last section if exceeded
9. At least 2 sections — if only 1, split it into intro (instrumental only) + main section
10. `end_beat` of last section should be a multiple of 4 (bar boundary) — extend or truncate by up to 2 beats

**On validation failure:** Fix automatically where possible (clamp, merge, extend). Log all corrections with severity for LLM prompt quality debugging. Only raise an error if the sections are completely unrecoverable (e.g., 0 sections after merging).

**Default arrangement template** (used when LLM fails entirely or produces unrecoverable output):
```python
def default_arrangement(total_beats: int) -> list[Section]:
    """Generate a safe 5-section arrangement when LLM output is unusable.
    Adds vocal entry buildup and a breakdown before the outro for energy dynamics."""
    eighth = total_beats // 8
    quarter = total_beats // 4
    three_quarter = total_beats * 3 // 4
    seven_eighth = total_beats * 7 // 8
    return [
        Section(label="intro", start_beat=0, end_beat=eighth,
                stem_gains={"vocals": 0.0, "drums": 0.8, "bass": 0.7, "guitar": 0.6, "piano": 0.5, "other": 1.0},
                transition_in="fade", transition_beats=4),
        Section(label="build", start_beat=eighth, end_beat=quarter,
                stem_gains={"vocals": 0.6, "drums": 0.7, "bass": 0.8, "guitar": 0.5, "piano": 0.4, "other": 0.5},
                transition_in="crossfade", transition_beats=4),
        Section(label="main", start_beat=quarter, end_beat=three_quarter,
                stem_gains={"vocals": 1.0, "drums": 0.7, "bass": 0.8, "guitar": 0.5, "piano": 0.4, "other": 0.5},
                transition_in="crossfade", transition_beats=2),
        Section(label="breakdown", start_beat=three_quarter, end_beat=seven_eighth,
                stem_gains={"vocals": 0.8, "drums": 0.0, "bass": 0.6, "guitar": 0.7, "piano": 0.8, "other": 0.7},
                transition_in="crossfade", transition_beats=4),
        Section(label="outro", start_beat=seven_eighth, end_beat=total_beats,
                stem_gains={"vocals": 0.0, "drums": 0.6, "bass": 0.5, "guitar": 0.5, "piano": 0.6, "other": 0.8},
                transition_in="crossfade", transition_beats=4),
    ]
```

This 5-section fallback provides vocal entry buildup, a main section, a breakdown (drums at 0.0 for contrast), and an instrumental outro. It is vastly better than a flat 3-section template where a single "verse" section is 67% of the remix at identical gains -- which produces exactly the "two songs playing at once" problem QR-1 was designed to solve. When the fallback is used, set `RemixPlan.used_fallback = True` so the frontend can display the explanation with warning styling.

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
    # Bandpass vocal stem to 300 Hz - 5 kHz before computing RMS
    # (isolates vocal formants, reduces instrumental bleed inflation)
    # Bandpass vocal to 300-3500 Hz for energy detection — aligned with the ducking range
    # (300-3000 Hz) plus a small margin for harmonic masking. Using the same filter order (4th)
    # as the mid-band extraction for consistent skirt slopes. Previously 300-5000 Hz with 2nd
    # order, which caused sibilance (4-8 kHz) to trigger ducking of mid-range content that was
    # not actually competing with those high-frequency vocal components — resulting in over-ducking
    # during sibilant passages.
    sos_bp = butter(4, [300, 3500], btype='band', fs=sr, output='sos')
    vocal_filtered = sosfiltfilt(sos_bp, vocal_mono)
    vocal_energy = np.array([
        np.sqrt(np.mean(vocal_filtered[i:i+frame_len]**2))
        for i in range(0, len(vocal_filtered) - frame_len, frame_len)
    ])
    # Noise-floor-relative threshold with hysteresis (replaces fragile 40th percentile).
    # The percentile approach fails with real-world stem separation output:
    # instrumental bleed raises the noise floor, causing over-ducking during pauses
    # (pumping artifacts), and quiet vocal passages (whispered verses) are misclassified.
    noise_floor = np.percentile(vocal_energy, 10)
    onset_threshold = noise_floor * 4.0
    offset_threshold = noise_floor * 2.0
    # Frame becomes active when energy > onset_threshold, stays active until < offset_threshold
    vocal_active = np.zeros_like(vocal_energy)
    active = False
    for i in range(len(vocal_energy)):
        if active:
            active = vocal_energy[i] >= offset_threshold
        else:
            active = vocal_energy[i] > onset_threshold
        vocal_active[i] = float(active)

    # 2. Upsample mask to sample rate, smooth with asymmetric attack/release
    #    30ms time constant for attack, 150ms for release (~90ms/450ms to 95%)
    #    Prevents pumping artifacts from rapid on/off between words
    #
    #    PERFORMANCE: The naive Python loop is O(n) with serial data dependency, taking
    #    5-15s for a 2-minute track and 10-30s for a 4-minute track. Use the two-pass
    #    scipy.signal.lfilter approximation (runs in <0.05s, no new dependency):
    mask = np.repeat(vocal_active, frame_len)[:len(instrumental)]
    # Use the correct exponential IIR alpha: alpha = 1 - exp(-1 / (tau * sr))
    # The linear approximation (1 / (tau * sr)) diverges at short time constants
    # and can produce alpha > 1 (unstable filter) when tau * sr < 1.
    import math
    attack_alpha = 1 - math.exp(-1.0 / (0.03 * sr))   # 30ms attack
    release_alpha = 1 - math.exp(-1.0 / (0.15 * sr))  # 150ms release
    from scipy.signal import lfilter
    attack_smoothed = lfilter([attack_alpha], [1, -(1 - attack_alpha)], mask)
    release_smoothed = lfilter([release_alpha], [1, -(1 - release_alpha)], mask[::-1])[::-1]
    mask = np.maximum(attack_smoothed, release_smoothed)
    # NOTE: This two-pass lfilter + max() approach is an approximation of true asymmetric
    # attack/release smoothing. It has fast attack behavior but the release may exhibit
    # a plateau artifact near transitions (the max() keeps the mask elevated longer than
    # a true conditional-alpha loop would). For MVP this is acceptable — perceptually close
    # and runs in <0.05s vs 5-15s for the naive Python loop. For exact behavior, use
    # numba.jit(nopython=True) on the conditional-alpha serial loop (~100x faster, <0.1s).

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
1.  Separate stems (BS-RoFormer SW via Modal; fallback: htdemucs_ft local)
2.  Trim stems to source time ranges (start_time_vocal/instrumental from RemixPlan)
3a. Tempo match all stems to common BPM
3b. Re-run beat detection on stretched audio: `librosa.beat.beat_track` on the stretched audio with
    `start_bpm=target_bpm` for an accurate post-stretch beat grid. This replaces proportional scaling
    (`beat_frames * (original_bpm / target_bpm)`) which accumulates 0.5-1.5s of drift per minute due to
    compounding BPM detection error and tempo variation within the song. Cost: ~0.4s per stem.
    Proportional scaling as last-resort fallback only — log a warning when used (section boundaries
    may be audibly misaligned). This MUST happen after rubberband and BEFORE the arrangement renderer uses the grid.
3c. Beat-phase-align vocal to instrumental beat grid (see Phase B6 step 3c)
4.  Key match if needed
5.  Cross-song level matching (LUFS with silence guard)
6.  Validate LLM sections (section validation — see 2b above).
    Available beats = len(beat_frames_stretched). Truncate sections exceeding this.
6b. Instrumental bleed attenuation ("bleed tax"): Before rendering, measure cross-correlation
    between the vocal stem and each instrumental stem in the 300 Hz - 3 kHz band. If significant
    correlation is detected (suggesting bleed), apply a permanent gentle attenuation (-2 to -4 dB)
    to the instrumental stems in the vocal band. This reduces ghost vocals at the cost of slightly
    thinner instrumentals. ~15 lines of scipy. Ghost vocals from the instrumental source song are
    the most user-noticeable artifact; spectral ducking makes it worse by releasing during vocal
    pauses, exposing ghost vocals exactly when there is no masking.
7.  Render arrangement into TWO BUSES: vocal bus + instrumental bus ← NEW
    - Place stems per LLM section timeline using beat grid
    - Apply per-stem gains from Section.stem_gains
    - Apply transition envelopes (per-stem equal-power interpolation for crossfade, micro-crossfade for cut)
    - Vocal bus = rendered vocal stem with arrangement gains applied
    - Instrumental bus = sum of rendered drums + bass + guitar + piano + other stems with arrangement gains
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
- ~~Fallback loses all intelligence~~ → Phase B5 now includes a concrete deterministic fallback algorithm using `vocal_prominence_db`, energy profiles, and a 5-section arrangement with energy dynamics
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

   Each test case includes: input prompt, mock song metadata (BPM/key/energy/vocal_prominence_db), expected `vocal_source`, and qualitative expectations. Save as `tests/fixtures/llm_test_prompts.json`.

2. **Deterministic data-driven fallback** — now specified concretely in Phase B5 with `vocal_prominence_db`-based vocal source selection, energy-profile-guided section selection, BPM-confidence-based tempo decision, and honest user-facing explanation with `used_fallback=True`.

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
- Confident-but-wrong key detection (~20-30% of songs) — now mitigated by two-tier key confidence system (see Phase B6): >= 0.55 = full shift, 0.40-0.55 = half shift (hedging), < 0.40 = skip
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

See Phase B6 step 4 for the full implementation. Uses `scipy.signal.welch` to measure instrumental spectral density in the vocal frequency band (200 Hz - 5 kHz) and maps to an offset: sparse instrumental = +1-2 dB, dense = +5-8 dB. ~15 lines, no new dependencies. Per-stem volume adjustments are now embedded in `Section.stem_gains` (QR-1) and applied during arrangement rendering.

#### 3. Expanded BPM reconciliation (70-180 range, 3/2 and 2/3 ratios)

See Phase B4 for the expanded algorithm. Generates {original, halved, doubled, 3/2, 2/3} interpretations per song, filtered to 70-180 BPM, selects the pair with minimum gap. Reduces the cross-genre failure rate from ~73% to ~53% by rescuing R&B (72 doubled to 144) + pop/rock/EDM pairings.

**What it doesn't rescue:** Hip-hop (85) + pop (120) = 41% gap has no half/double/triplet relationship. This is solved by solution #4 below.

#### 4. Vocals-only stretching for 30-50% tempo gaps

See Phase B6 tempo matching tiered limits. For gaps between 30-50%, the system now stretches ONLY the vocal stem to match the instrumental's tempo, instead of skipping tempo matching entirely. Vocals tolerate 30-50% stretch (noticeably different but intelligible); instrumentals do not (drums sound broken beyond 10%). This rescues the hip-hop + pop pairing.

**Default tempo target changed to instrumental source song's tempo** (not "average"). Rationale: drums are the most stretch-sensitive element; preserving the instrumental's tempo produces a more natural rhythmic foundation.

#### 5. Key confidence two-tier system (replaces single 0.3 threshold)

See Phase B6 key matching. Uses a two-tier system: >= 0.55 = full shift, 0.40-0.55 = half shift (hedging to reduce harm from wrong detections), < 0.40 = skip. Wrong pitch shifts are ~2x worse than correct shifts are good, so break-even accuracy is ~67%. The hedging tier improves expected value for moderate-confidence detections.

#### 6. Transparency framework (inform → transparent → honest)

Instead of hard-blocking, soft-warning, or silently proceeding, the system follows a three-stage transparency approach:

**Before submission** (compatibility signal): Informational, non-blocking. Describes what the remix will be like, not what's wrong with the songs.

**During processing** (progress events): Transparent about what's happening. When tempo or key matching is skipped or limited, the progress detail says so:
- "Mixing stems... (adjusting vocal speed to match the beat — these songs have different tempos)"
- "Keeping original keys (key detection wasn't confident enough to shift safely)"
- "Matching tempo... (stretching vocals to match the instrumental's rhythm)"

**After processing** (LLM explanation): Honest about what was done and why. The LLM explains any compatibility challenges and what the system did about them: "These songs had very different tempos (Song A is much slower than Song B), so I stretched the vocals to match Song B's beat. The vocal may sound slightly different from the original."

**Hard-block conditions (return 422):** None in MVP. The system always attempts a remix. Even a rough remix is better than "we can't do this."

#### 7. Groove feel (addressed — drum-stem detection + LLM guidance)

**The problem:** Two songs can have matching BPMs but incompatible grooves (half-time vs straight-time, swung vs quantized, 6/8 vs 4/4). The plan treats tempo as a scalar number. A half-time hip-hop beat at 170 BPM has its snare on beat 3, while a straight-time pop beat at 128 BPM has snares on beats 2 and 4 — even after tempo matching, the rhythmic feel clashes. 40-55% of cross-genre pairings have groove mismatches.

**MVP solution:** Detect groove classification from drum stem onset patterns after stem separation. `detect_groove()` in Phase B4 runs `librosa.onset.onset_detect` on the drum stem, computes inter-onset intervals, and classifies as straight-time, half-time, swung, or unknown (~40 lines). Stored in `AudioMetadata.groove_type` and passed to the LLM with specific groove-aware arrangement instructions: "Song A is half-time hip-hop, Song B is straight-time pop -- use sections where the instrumental has sparse drums." Groove mismatch is included in the compatibility score (QR-3 item 1).

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
- ~~Swing/straight detection from drum stem inter-onset-interval analysis~~ — now addressed in MVP (see Phase B4 groove detection)
- Recovery flow: "Create New Remix" should pre-fill previously uploaded songs (keep Song A, replace only Song B)
- Post-remix compatibility insight with suggestion: "For a smoother blend, try a song with a similar rhythm"

---

## Recommended Implementation Order

> **Constraint:** 4 days to demo (Saturday). Single developer. Tracer bullet approach — build vertically (thin end-to-end slices), not horizontally (complete layers). Every day ends with working software that's better than the day before.
>
> **Core principle:** Use **BS-RoFormer SW via Modal** (cloud GPU, 6-stem) as the primary separation backend from Day 1. This gives best-in-class separation quality and cloud GPU speed (~30-60s/song vs. minutes on local CPU). Fall back to local `htdemucs_ft` (4-stem) only if Modal setup blocks progress for >2 hours.

### Day 1: "Hardcoded Frankenmix" — End-to-End Plumbing

**Exit criteria:** Upload two songs in a browser, hear them remixed, press play.

**Morning — Backend skeleton + Modal cloud GPU setup:**
- B1 (stripped): FastAPI app, CORS, health check, minimal `config.py` with `BaseSettings`, `data/` directories. No structlog, no exceptions hierarchy — just get the server running.
- B2 (stripped): `POST /api/remix` accepting two files + prompt. Validate file extension only (skip magic bytes, duration check, MIME). Save to `data/uploads/{session_id}/`. Return session ID.
- **Modal + BS-RoFormer setup:**
  - Create Modal account, run `modal token new`
  - Build Modal app with GPU container image (A10G): install `audio-separator[gpu]`, `torch`, `soundfile`
  - Bake BS-RoFormer SW model weights into image via `run_commands()` (eliminates cold-start model download)
  - **Verify the correct 6-stem SW checkpoint** (not the 12.9755 2-stem variant)
  - Test with one song: upload bytes → separate → receive 6 stem WAV bytes back
  - Validate output: assert 6 stems returned, verify float32 WAV format
  - **Time-box: 2 hours max.** If Modal blocks (account issues, GPU capacity, wrong checkpoint), pivot to local `htdemucs_ft` and revisit Modal on Day 4.
- Hardcoded audio pipeline: take vocal stem from Song A + all instrumental stems from Song B → pydub overlay → export MP3 → serve via `GET /api/remix/{session_id}/audio`. **Synchronous** — no ThreadPoolExecutor, no SSE, no processing lock yet.

**Afternoon — Rubberband validation + pipeline wiring:**
- Install `rubberband`. Verify `rubberband --version` returns v3.x. Test a single tempo stretch via CLI with a known input.
- Wire the full separation → overlay → export chain: upload two songs → separate both via Modal (6 stems each) → take vocals from Song A, drums+bass+guitar+piano+other from Song B → overlay with pydub → export MP3. Still synchronous, still hardcoded.
- **Also install `htdemucs_ft` locally** as the fallback backend. Verify it runs on a test song and produces 4 stems. Wire the `settings.stem_backend` toggle ("modal" vs "local") so switching is a one-line config change.

**Evening — Single HTML page frontend:**
- One HTML file served by FastAPI (or a minimal Vite page): two file inputs, a text input for prompt, a submit button, and an `<audio>` tag. Form POST via fetch, poll `/api/remix/{id}/audio` until it's ready (or just wait for the sync response). No React, no state machine, no SSE.

**Draws from phases:** B1 (partial), B2 (partial), B3 (Modal + local fallback), B6 steps 1+7+11 (overlay + export)

---

### Day 2: "The Real Remix" — Tempo Matching + Arrangement + SSE

**Exit criteria:** Upload two songs → hear a tempo-matched, level-balanced, 3-section remix with per-stem volume control (6 stems). Progress updates in the browser.

**Morning — Async processing + SSE:**
- Add `ThreadPoolExecutor(max_workers=1)` + `threading.Queue` for progress events. Move pipeline to background thread.
- Add `GET /api/remix/{session_id}/progress` SSE endpoint (basic — no reconnection, no keepalive tuning, no tab backgrounding). Just streams events from the queue.
- Add `GET /api/remix/{session_id}/status` endpoint (basic — returns current status JSON).
- Update HTML page to use `EventSource` and show progress text + a progress bar.

**Afternoon — BPM detection + tempo matching:**
- B4 (stripped): `analyze_audio()` returning BPM + beat_frames + duration only. Skip key detection, energy profiles, groove detection, vocal_energy_ratio for now.
- BPM reconciliation: implement `reconcile_bpm()` with the expanded interpretation matrix (original, half, double, 3/2, 2/3 in 70-180 range).
- B6 step 3: Rubberband tempo matching. Use `rubberband_process()` CLI wrapper. Stretch vocals toward instrumental tempo (default: instrumental source tempo). Tiered limits (< 10%, 10-25%, 25-50%, > 50%).
- B6 step 1: Sample rate + channel standardization (44.1kHz stereo float32). Mono-to-stereo handling.

**Evening — LUFS normalization + deterministic arrangement:**
- B6 step 4: Cross-song level matching with `pyloudnorm`. Use fixed +3 dB vocal offset for now (defer spectral-density-adaptive offset). Include silence guard (`LUFS_FLOOR = -40.0`).
- B6 steps 8-9: LUFS normalization on final mix + peak limiter (soft-knee clip).
- B6 steps 10-11: Fade-in/fade-out + MP3 export via ffmpeg.
- Build the **deterministic fallback plan** (`generate_fallback_plan()`) — this becomes the remix plan for today. No LLM yet.
- Build the **section-based arrangement renderer** (QR-1 mixer logic). Test with the 3-section deterministic fallback. Convert beats → samples via beat grid. Apply per-stem gains across all 6 stems (vocals, drums, bass, guitar, piano, other). Apply transition envelopes (fade, crossfade, cut with micro-crossfade).
- Wire it all together: separate (Modal 6-stem) → analyze → deterministic plan → tempo match → arrange → level match → LUFS → peak limit → fade → export.

**Draws from phases:** B4 (partial), B6 (most of it), B7 (partial — async + SSE), QR-1 arrangement renderer

---

### Day 3: "Intelligence" — LLM + React Frontend

**Exit criteria:** Type a prompt like "put the rap vocals over the pop beat, drop the drums in the middle" → get an LLM-driven remix with custom arrangement → play it back in a real React UI.

**Morning — LLM prompt interpretation:**
- B5: `interpret_prompt()` with Claude Sonnet via `tool_use`. System prompt with: MVP constraint explanation, song metadata (BPM, key, duration, total_beats), arrangement templates (Standard Mashup, DJ Set, Quick Hit, Chill), ambiguity handling, 3 few-shot examples.
- Post-LLM validation: section validation (the 10-point checklist from QR-1 section 2b). Time range validation. Clamp out-of-bounds values.
- Deterministic fallback on any LLM failure. Set `used_fallback = True`.
- Wire LLM into pipeline: replace hardcoded deterministic plan with LLM output. Keep deterministic fallback as safety net.
- Test with 5-10 prompts against real songs. Listen to every output.

**Afternoon — React frontend:**
- F1: `bun create vite frontend --template react-ts`. Install Tailwind. Set up Vite proxy to backend.
- Types in `types/index.ts` (ProgressEvent, AppState discriminated union, AppAction).
- API client in `api/client.ts` (createRemix with XHR for upload progress, connectProgress with EventSource, getSessionStatus, getAudioUrl).
- `RemixSession.tsx` with `useReducer` state machine.
- F2: `SongUpload.tsx` — drag-and-drop + file input, file name display, basic client-side validation (extension, 50MB).
- F3: `PromptInput.tsx` + `RemixForm.tsx` — textarea, example prompts as static text, submit button with disable-on-click.

**Evening — Progress + Player:**
- `useRemixProgress.ts` hook — EventSource connection, dispatches PROGRESS_EVENT/REMIX_READY/ERROR. Basic no-event timeout (120s). No reconnection, no tab backgrounding yet.
- F4: `ProgressDisplay.tsx` — progress bar, step description, cancel button.
- F5: `RemixPlayer.tsx` — `<audio>` with native controls, LLM explanation display, "Create New Remix" button, expiration notice.
- End-to-end test: upload songs in React UI → see progress → hear remix → see explanation.

**Draws from phases:** B5 (full), F1-F5 (compressed into one session, built against real working API)

---

### Day 4: "Demo Ready" — Polish + Quality + Error Handling

**Exit criteria:** App handles errors gracefully, audio quality is tuned across 3-5 song pairings, ready to demo.

**Morning — Key matching + audio quality:**
- B4: Add key detection (essentia with librosa fallback). Add key_confidence.
- B6 step 3: Key matching via rubberband (single-pass with tempo). Formant preservation for vocals. Key confidence gate at 0.45. Cap at +/- 4 semitones for vocals.
- B6 step 3 (new): Vocal stem pre-filtering before rubberband (200 Hz high-pass, 8 kHz low-pass).
- Spectral ducking (QR-1 section 3): implement `spectral_duck()`. Wire between arrangement rendering and final mix.
- Listen to 3-5 real song pairings. Tune parameters by ear.

**Afternoon — Error handling + robustness:**
- Processing lock: `processing_lock.acquire(blocking=False)` → 429 on busy. Fail-fast check. Pipeline wrapper with `try/finally` release.
- Upload validation: add pydub/ffprobe audio parse check (the real validation). Duration check (max 10 min). Prompt validation (5-1000 chars).
- Error handling in pipeline: catch `MusicMixerError` subtypes, push SSE error events with user-friendly messages.
- TTL cleanup via FastAPI lifespan (cleanup_loop, 5-minute interval, delete expired sessions).
- Frontend error states: network failure, 413, 422, 429, 500 — all with appropriate user-facing messages.

**Evening — Demo prep:**
- Test full flow with 3-5 diverse song pairings (hip-hop + pop, R&B + EDM, rock + electronic, similar-tempo pair, mismatched pair).
- LLM prompt tuning: adjust system prompt based on listening results.
- UI copy: add tagline ("Upload two songs. Describe your mashup. AI does the rest.") and constraint explanation ("musicMixer takes the vocals from one song and layers them over the other song's beat.").
- Add `warnings` display in RemixPlayer (amber info boxes). Add `used_fallback` indicator.
- Smoke test: upload → prompt → progress → playback → new remix. Repeat 3x. Fix any crashes.

**Draws from phases:** B4 (key detection), B6 (key matching, vocal pre-filtering, spectral ducking), B7 (processing lock, error handling, TTL), QR-1 (spectral ducking), F1-F5 (error states, polish)

---

### Day-by-Day Summary

| Day | Milestone | What You Can Demo |
|-----|-----------|-------------------|
| 1 | Hardcoded Frankenmix | Upload 2 songs → BS-RoFormer 6-stem separation via Modal → vocals over instrumentals in browser |
| 2 | The Real Remix | Tempo-matched, level-balanced, 3-section remix with per-stem control (6 stems) + progress bar |
| 3 | Intelligence | Prompt-driven LLM remixes ("boost the guitar, mute the piano") in a real React UI |
| 4 | Demo Ready | Key matching, spectral ducking, error handling, tested across 5 song pairings |

### What's Deferred to Post-Demo

These features are fully specified in the plan above but cut from the 4-day demo scope:

| Feature | Plan Reference | Why Deferred |
|---------|---------------|--------------|
| Compatibility analysis endpoint | QR-3 (POST /api/analyze) | Nice-to-have UX — not needed to produce a remix. |
| Energy profiles for LLM | B4 (energy_regions, onset_density) | LLM works fine with BPM + key + duration. Add when tuning prompt quality post-demo. |
| Groove detection | B4 (detect_groove) | Requires drum stem analysis. Layer in post-demo for cross-genre arrangement quality. |
| Vocal energy ratio | B4 (vocal_energy_ratio) | LLM falls back to reasonable heuristics without it. Add post-demo. |
| Spectral-density-adaptive vocal offset | B6 step 4 (compute_vocal_offset) | Fixed +3 dB offset is good enough for demo. Calibrate adaptive version post-demo with 10+ pairings. |
| SSE reconnection + tab backgrounding | F4 (useRemixProgress.ts) | Remixes take 2-5 min — users won't refresh during a demo. Add post-demo. |
| sessionStorage persistence | F1 (RemixSession.tsx) | Same rationale — crash recovery matters for real users, not demos. |
| Full upload validation (magic bytes, MIME) | B2 | Extension + pydub parse is sufficient. Magic bytes add security, not functionality. |
| Server-level request size limit | B2 (uvicorn --limit-max-body-size) | Single-user demo — won't hit this. |
| Mobile-responsive polish | F2, F3 | Demo on a laptop. Mobile polish post-demo. |
| Structured logging (structlog) | B1 | Python logging is fine for 4 days. Add structlog when debugging production issues. |

---

## What's NOT in This Plan (Deferred)

Per the brainstorm and PRD, these are explicitly out of scope for the MVP entirely:

- Cross-song stem mixing (e.g., drums from Song A + bass from Song B) — post-MVP. 6-stem separation is already in place, so the infrastructure supports this.
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
