# API Contract Reference

> Implementation reference extracted from the full plan. Canonical source: `docs/plans/2026-02-23-feat-mvp-prompt-based-remix-plan.md`

## Endpoints

### POST /api/remix

Creates a new remix session. Accepts two audio files and a prompt.

```
Body: multipart/form-data
  - song_a: File (MP3/WAV, max 50MB)
  - song_b: File (MP3/WAV, max 50MB)
  - prompt: string (min 5 chars, max 1000 chars)

Response 200: { "session_id": "uuid" }
Response 413: File too large (server-level: uvicorn --limit-max-body-size 104857600)
Response 422: { "detail": "Validation error message" }
Response 429: { "detail": "Server is busy processing another remix" }
```

### GET /api/remix/{session_id}/progress

SSE stream of processing events.

```
Response: text/event-stream

data: {"step":"separating","detail":"Extracting stems from Song A...","progress":0.10}
data: {"step":"separating","detail":"Extracting stems from Song B...","progress":0.30}
data: {"step":"analyzing","detail":"Detecting tempo and key...","progress":0.45}
data: {"step":"interpreting","detail":"Planning your remix...","progress":0.55}
data: {"step":"processing","detail":"Matching tempo...","progress":0.65}
data: {"step":"processing","detail":"Normalizing loudness...","progress":0.75}
data: {"step":"processing","detail":"Mixing stems...","progress":0.85}
data: {"step":"processing","detail":"Rendering final mix...","progress":0.95}
data: {"step":"complete","detail":"Remix ready!","progress":1.0,"explanation":"..."}
  OR
data: {"step":"error","detail":"Could not match these songs...","progress":0}

Keepalive (every 5s timeout with no event):
data: {"step":"keepalive","detail":"","progress":-1}
```

**On connect when already complete:** sends final event immediately, then closes.
**On connect when processing:** sends `last_event`, then continues streaming.

### GET /api/remix/{session_id}/status

Quick JSON status check for reconnection.

```
Response 200:
  {"status":"processing","progress":0,"detail":"Starting..."}
  {"status":"processing","progress":0.45,"detail":"Matching tempo..."}
  {"status":"complete","remix_url":"/api/remix/{id}/audio","explanation":"..."}
  {"status":"error","detail":"..."}

Response 404: Session not found or expired
```

**Note:** Internal `queued` status is mapped to `processing` before returning — frontend never sees `queued`.

### GET /api/remix/{session_id}/audio

```
Response 200: audio/mpeg (MP3 file via FileResponse)
Response 404: Remix not found or expired
```

### POST /api/analyze

Lightweight compatibility check (~2-3s). Runs independently of remix processing (does NOT acquire processing_lock).

```
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
Response 408: { "detail": "Analysis timed out" }
```

**Partial failure:** If BPM/key detection fails for one song, return 200 with failed fields as `null`, degrade compatibility to `"challenging"`.

### GET /health

```
Response 200: {"status":"ok"}
```

---

## Pydantic Response Models

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

class SongAnalysis(BaseModel):
    bpm: float | None = None
    key: str | None = None
    scale: str | None = None

class CompatibilityResult(BaseModel):
    level: str               # "great" | "good" | "challenging" | "tough"
    message: str             # Plain-language, user-facing
    detail: str | None = None

class AnalyzeResponse(BaseModel):
    song_a: SongAnalysis
    song_b: SongAnalysis
    compatibility: CompatibilityResult

class HealthResponse(BaseModel):
    status: str

class ErrorResponse(BaseModel):
    detail: str
```

---

## Processing Lock Behavior

Single-worker uvicorn with global processing lock. One remix at a time; second request gets 429.

```python
# In POST /api/remix handler:

# 1. Fail-fast check (non-mutating — never acquires the lock)
if processing_lock.locked():
    raise HTTPException(429, "Server is busy processing another remix")

# 2. Accept upload, validate files, save to disk

# 3. Acquire lock once (authoritative gate) and submit:
if not processing_lock.acquire(blocking=False):
    raise HTTPException(429, "Server is busy")

# 4. Submit to executor via wrapper that guarantees lock release:
def pipeline_wrapper():
    try:
        run_pipeline(session_id, song_a_path, song_b_path, prompt, session.events)
    finally:
        if processing_lock.locked():
            processing_lock.release()

try:
    future = executor.submit(pipeline_wrapper)
    future.add_done_callback(lambda f: _log_unhandled(f, session_id))
except Exception:
    processing_lock.release()
    raise
```

---

## Session State

```python
sessions: dict[str, SessionState] = {}
sessions_lock = threading.Lock()   # Guards sessions dict mutations (add/delete)
processing_lock = threading.Lock() # One remix at a time
executor = ThreadPoolExecutor(max_workers=1)

@dataclass
class SessionState:
    status: str                      # "queued" | "processing" | "complete" | "error"
    events: queue.Queue              # maxsize=100
    created_at: datetime
    remix_path: str | None = None
    explanation: str | None = None
    last_event: dict | None = None   # Most recent ProgressEvent (for reconnecting SSE clients)
```

**Thread safety:** `sessions_lock` guards dict mutations. `SessionState` attribute assignments are atomic under CPython GIL (always assign new objects, never mutate in place). `events` queue is inherently thread-safe.

---

## SSE Implementation

```python
sse_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="sse-reader")

async def event_stream(session: SessionState):
    loop = asyncio.get_running_loop()
    start = time.monotonic()
    while True:
        if time.monotonic() - start > settings.max_sse_duration_seconds:  # default: 1200 (20 min)
            yield 'data: {"step":"error","detail":"Processing timed out","progress":0}\n\n'
            break
        try:
            event = await loop.run_in_executor(
                sse_executor, functools.partial(session.events.get, timeout=5)
            )
        except queue.Empty:
            yield 'data: {"step":"keepalive","detail":"","progress":-1}\n\n'
            continue
        except asyncio.CancelledError:
            break
        session.last_event = event
        yield f"data: {json.dumps(event)}\n\n"
        if event["step"] in ("complete", "error"):
            break
```

---

## CORS Config

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,  # ["http://localhost:5173"]
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Upload Validation Layers

1. Check file extension (`.mp3`, `.wav`)
2. Check MIME type / magic bytes
3. Parse with pydub/ffprobe — real validation
4. Check file size (max 50MB, enforced during streaming read)
5. Check duration (max 10 minutes, checked after audio parse)
6. Validate resolved paths within `data/` directory (path traversal defense via `pathlib.Path.resolve()`)

---

## TTL Cleanup

- Runs via `asyncio.create_task` in FastAPI `lifespan`
- Every `cleanup_interval_seconds` (300s): find sessions where `status != "processing"` and age > `remix_ttl_seconds` (10800s)
- Delete session from `sessions` dict under `sessions_lock`, then delete files (`uploads/`, `stems/`, `remixes/` subdirs) via `asyncio.to_thread(shutil.rmtree)`
- On shutdown: `cleanup_task.cancel()`, `executor.shutdown(wait=False, cancel_futures=True)`

---

## Settings

```python
class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["http://localhost:5173"]
    max_file_size_mb: int = 50
    max_duration_seconds: int = 600
    allowed_extensions: set[str] = {".mp3", ".wav"}
    data_dir: Path = Path("data")
    remix_ttl_seconds: int = 10800
    cleanup_interval_seconds: int = 300
    max_sse_duration_seconds: int = 1200
    stem_model: str = "htdemucs_ft"
    max_tempo_adjustment_pct: float = 0.30
    max_pitch_shift_semitones: int = 5
    output_format: str = "mp3"
    output_bitrate: str = "320k"
    target_lufs: float = -14.0
    anthropic_api_key: str
    llm_model: str = "claude-sonnet-4-20250514"
    llm_max_retries: int = 2
    llm_timeout_seconds: int = 30
    model_config = SettingsConfigDict(env_file=".env")
```

---

## Exception Hierarchy

```python
class MusicMixerError(Exception): ...
class ValidationError(MusicMixerError): ...
class SeparationError(MusicMixerError): ...
class AnalysisError(MusicMixerError): ...
class ProcessingError(MusicMixerError): ...
class PipelineError(MusicMixerError): ...
```

---

## File Storage Layout

```
data/
├── uploads/{session_id}/song_a.{ext}, song_b.{ext}
├── stems/{session_id}/...
└── remixes/{session_id}/...
```

Session ID: `uuid.uuid4()` (cryptographically random).

---

## Compatibility Levels

| Level | BPM Condition | Key Condition | Message |
|-------|--------------|---------------|---------|
| great | <10% gap | CoF distance 0-1 | "These songs should blend really well together." |
| good | 10-20% gap | shift <= 3 semitones | "These songs have some differences, but we can work with it." |
| challenging | 20-35% gap | — | "These songs have a different energy. The remix will have more of a mashup feel." |
| tough | >35% gap | shift > 5 semitones | "These songs are very different — the AI will do its best, but they may not sync up perfectly." |

---

## Cross-References

- SSE event format consumed by frontend: see `frontend.md` (useRemixProgress hook)
- Pipeline that generates progress events: see `audio-pipeline.md` (pipeline orchestrator)
- LLM that generates `explanation` field: see `llm-integration.md`
