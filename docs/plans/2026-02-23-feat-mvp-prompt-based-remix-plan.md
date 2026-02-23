---
title: "feat: MVP Prompt-Based Remix"
type: feat
status: active
date: 2026-02-23
brainstorm: docs/brainstorms/2026-02-23-prompt-based-remix-brainstorm.md
prd: docs/PRD.md
---

# MVP: Prompt-Based Music Remix

## Overview

Build the end-to-end musicMixer MVP: upload two songs, describe a mashup in plain English, get an AI-generated remix. Server renders the final audio file; browser plays it back. Remix expires after 3 hours or when a new one is created.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        FRONTEND                              │
│              React + TypeScript + Vite (Bun)                 │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐ │
│  │  Upload   │→ │  Prompt  │→ │ Progress │→ │   Player    │ │
│  │  2 songs  │  │  input   │  │  (SSE)   │  │ (<audio>)   │ │
│  └──────────┘  └──────────┘  └──────────┘  └─────────────┘ │
└──────────────────────┬──────────────────────────────────────┘
                       │ REST API + SSE
┌──────────────────────┴──────────────────────────────────────┐
│                        BACKEND                               │
│                   Python + FastAPI                            │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                Pipeline Orchestrator                   │   │
│  │                                                        │   │
│  │  1. File Upload & Validation                           │   │
│  │  2. Stem Separation (audio-separator + htdemucs_ft)    │   │
│  │  3. Audio Analysis (librosa BPM + essentia key)        │   │
│  │  4. LLM Prompt Interpretation (Claude Haiku)           │   │
│  │  5. Audio Processing (pyrubberband tempo/key match)    │   │
│  │  6. Mixing & Rendering (pydub → MP3)                   │   │
│  │  7. Serve remix + 3-hour TTL cleanup                   │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Key Architectural Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Backend language | Python | Entire audio ML ecosystem lives in Python. No point wrapping it. |
| Backend framework | FastAPI | Async, built-in SSE support, automatic OpenAPI docs, simple. |
| Stem separation | `audio-separator` + `htdemucs_ft` | Single package wrapping best models. 4-stem output (vocals/drums/bass/other). ~10s on GPU. |
| BPM detection | `librosa` | Simple, well-documented, good enough for MVP. |
| Key detection | `essentia` | Best accuracy for musical key estimation. |
| Tempo/key matching | `pyrubberband` | Industry-standard quality (Rubber Band Library). |
| Audio mixing | `pydub` + `ffmpeg` | High-level API for overlay, volume, export. |
| Remix delivery | Server-rendered single MP3 file | Massively simplifies frontend — just an `<audio>` tag. No Tone.js needed. |
| Browser playback | HTML5 `<audio>` element | Zero JS dependencies. Native controls. Streams large files. |
| LLM for prompts | Claude Haiku (structured outputs) | Fast, cheap (~$0.001/call), schema-guaranteed JSON. |
| Progress updates | Server-Sent Events (SSE) | Simpler than WebSockets for one-way server→client updates. |
| Frontend framework | React + Vite + TypeScript | Standard, fast, good DX with Bun. |
| Styling | Tailwind CSS | Rapid prototyping, no component library overhead. |
| State management | React state (useState/useReducer) | App has one linear flow — no need for Redux/Zustand. |

### Design Principles

- **Orthogonality**: Each backend service (separation, analysis, LLM, mixing) is a standalone module with a clean interface. Swap any piece without touching others.
- **Reversibility**: Every technology choice has a clear migration path. Switch stem models by changing one config line. Swap LLM providers by changing one module. Replace pydub with direct ffmpeg if needed.
- **Server-side rendering**: The single biggest simplification. The frontend never touches raw audio data — it uploads files and plays back a URL. This makes the frontend trivially simple and keeps all complexity in one place.
- **Minimal API surface**: One endpoint to create a remix, one SSE endpoint for progress, one endpoint to serve the audio file. That's it.

---

## Backend Execution Plan

> All backend work happens in `backend/`. Python project using FastAPI.

### Phase B1: Project Scaffolding

**Goal:** Runnable FastAPI server with health check.

- [ ] Initialize Python project with `pyproject.toml` (use `uv` or `poetry` — TBD based on preference)
- [ ] Create project structure:
  ```
  backend/
  ├── pyproject.toml
  ├── CLAUDE.md
  ├── src/
  │   └── musicmixer/
  │       ├── __init__.py
  │       ├── main.py              # FastAPI app entry point
  │       ├── config.py            # Settings (env vars, file paths, limits)
  │       ├── api/
  │       │   ├── __init__.py
  │       │   └── routes.py        # API endpoint definitions
  │       └── services/            # Business logic modules
  │           └── __init__.py
  ├── data/                        # Runtime data (gitignored)
  │   ├── uploads/                 # Uploaded song files
  │   ├── stems/                   # Separated stems
  │   └── remixes/                 # Final rendered remixes
  └── tests/
      └── __init__.py
  ```
- [ ] Install core dependencies: `fastapi`, `uvicorn`, `python-multipart`
- [ ] Implement health check endpoint (`GET /health`)
- [ ] Add `CLAUDE.md` for backend repo with conventions
- [ ] Verify server starts and responds

**Reversibility:** Standard project structure. Any dependency can be swapped without restructuring.

**Files created:** `pyproject.toml`, `src/musicmixer/main.py`, `src/musicmixer/config.py`, `src/musicmixer/api/routes.py`, `CLAUDE.md`

---

### Phase B2: File Upload & Validation

**Goal:** Accept two audio files via multipart upload. Validate and store them.

- [ ] Create `POST /api/remix` endpoint accepting:
  - Two audio files (`song_a`, `song_b`) as multipart form data
  - A `prompt` text field
- [ ] Validate files:
  - Accepted formats: MP3, WAV (check MIME type + file extension)
  - Max file size: 50MB per file (configurable)
  - Max duration: 10 minutes per song (configurable — check after saving)
- [ ] Generate unique session ID (UUID) for each remix request
- [ ] Save uploaded files to `data/uploads/{session_id}/song_a.{ext}`, `data/uploads/{session_id}/song_b.{ext}`
- [ ] Return session ID to client immediately (processing continues async)
- [ ] Create `services/upload.py` with validation logic

**Orthogonality:** Upload handling is isolated in its own service. The rest of the pipeline receives file paths, not upload objects.

**Files created:** `src/musicmixer/services/upload.py`
**Files modified:** `src/musicmixer/api/routes.py`

---

### Phase B3: Stem Separation Service

**Goal:** Split each uploaded song into 4 stems (vocals, drums, bass, other).

- [ ] Install `audio-separator` (brings in PyTorch, etc.)
- [ ] Create `services/separation.py`:
  - `separate_stems(audio_path: str, output_dir: str) -> dict[str, str]`
  - Uses `htdemucs_ft` model (best quality among Demucs variants)
  - Returns mapping: `{"vocals": "/path/to/vocals.wav", "drums": "...", "bass": "...", "other": "..."}`
  - Model auto-downloads on first run
- [ ] Handle errors: corrupt files, unsupported formats, out-of-memory
- [ ] Add progress callback hook (for SSE integration later)
- [ ] Write unit test with a short test audio file

**Reversibility:** The service exposes a simple function signature. Swap `htdemucs_ft` for any other model (BS-RoFormer, Mel-Band RoFormer) by changing one line. Swap `audio-separator` for direct Demucs if needed.

**GPU note:** This is the most compute-intensive step. ~10s/song on GPU, ~3-6min/song on CPU. MVP can work on CPU but the UX will be slow.

**Files created:** `src/musicmixer/services/separation.py`, `tests/test_separation.py`

---

### Phase B4: Audio Analysis

**Goal:** Detect BPM and musical key for each uploaded song.

- [ ] Install `librosa`, `essentia`
- [ ] Create `services/analysis.py`:
  - `analyze_audio(audio_path: str) -> AudioMetadata`
  - `AudioMetadata` dataclass: `bpm: float`, `key: str`, `scale: str`, `duration: float`
  - BPM via `librosa.beat.beat_track`
  - Key via `essentia.standard.KeyExtractor`
  - Duration via `librosa.get_duration`
- [ ] Write unit test

**Orthogonality:** Analysis is pure — takes a file path, returns metadata. No side effects. Can be called independently of separation.

**Note:** Analysis runs on the original full songs, not the stems. BPM and key are properties of the whole track.

**Files created:** `src/musicmixer/services/analysis.py`, `tests/test_analysis.py`

---

### Phase B5: LLM Prompt Interpretation

**Goal:** Convert user's natural language prompt into a structured stem-selection plan.

- [ ] Install `anthropic` SDK (or `openai` — whichever is preferred)
- [ ] Create `services/interpreter.py`:
  - `interpret_prompt(prompt: str, song_a_metadata: AudioMetadata, song_b_metadata: AudioMetadata) -> RemixPlan`
  - `RemixPlan` dataclass:
    ```python
    @dataclass
    class StemSelection:
        stem: str            # "vocals" | "drums" | "bass" | "other"
        include: bool
        volume_db: float     # relative volume adjustment

    @dataclass
    class RemixPlan:
        song_a_stems: list[StemSelection]
        song_b_stems: list[StemSelection]
        tempo_source: str    # "song_a" | "song_b" | "average"
        key_source: str      # "song_a" | "song_b" | "none"
        explanation: str     # LLM's reasoning (shown to user)
    ```
  - System prompt explains available stems, song metadata, and output schema
  - Uses Claude Haiku with structured outputs (or tool_use for schema enforcement)
- [ ] Handle edge cases: vague prompts ("mix these two"), contradictory requests, requests for instruments in "other" stem
- [ ] Write unit test with mocked LLM response

**Reversibility:** Interface is `(prompt, metadata, metadata) → RemixPlan`. Swap Claude for OpenAI by changing one module. Swap Haiku for Sonnet if interpretation quality needs improvement.

**Files created:** `src/musicmixer/services/interpreter.py`, `tests/test_interpreter.py`

---

### Phase B6: Audio Processing (Tempo/Key Matching + Mixing)

**Goal:** Time-stretch and pitch-shift stems to match, then mix them into a final audio file.

- [ ] Install `pyrubberband`, `pydub`, `soundfile`
- [ ] System dependencies: `ffmpeg`, `rubberband` (document in README)
- [ ] Create `services/processor.py`:
  - `process_remix(remix_plan: RemixPlan, song_a_stems: dict, song_b_stems: dict, song_a_meta: AudioMetadata, song_b_meta: AudioMetadata) -> str`
  - Steps:
    1. Determine target BPM (from `remix_plan.tempo_source`)
    2. Time-stretch all included stems to target BPM using pyrubberband
    3. Determine target key (from `remix_plan.key_source`)
    4. Pitch-shift stems if key adjustment needed
    5. Apply volume adjustments from remix plan
    6. Overlay all included stems using pydub
    7. Export final mix as MP3 (320kbps) to `data/remixes/{session_id}/remix.mp3`
  - Returns path to rendered remix file
- [ ] Write unit test with pre-separated test stems

**Orthogonality:** Processing takes a plan + stems + metadata and produces an audio file. Doesn't know about uploads, LLMs, or the web layer.

**Files created:** `src/musicmixer/services/processor.py`, `tests/test_processor.py`

---

### Phase B7: Pipeline Orchestrator + SSE Progress

**Goal:** Wire all services into an end-to-end pipeline with real-time progress updates.

- [ ] Create `services/pipeline.py`:
  - `run_pipeline(session_id: str, prompt: str) -> str`
  - Orchestrates: upload validation → stem separation (2x) → analysis (2x) → LLM interpretation → processing → remix path
  - Emits progress events at each step:
    ```
    {"step": "separating", "detail": "Extracting stems from Song A...", "progress": 0.15}
    {"step": "separating", "detail": "Extracting stems from Song B...", "progress": 0.35}
    {"step": "analyzing", "detail": "Detecting tempo and key...", "progress": 0.50}
    {"step": "interpreting", "detail": "Planning your remix...", "progress": 0.60}
    {"step": "processing", "detail": "Matching tempo...", "progress": 0.70}
    {"step": "processing", "detail": "Mixing stems...", "progress": 0.85}
    {"step": "complete", "detail": "Remix ready!", "progress": 1.0}
    ```
  - Parallelizes where possible: stem separation of song A and B can run concurrently (if resources allow); analysis of both songs can run concurrently
- [ ] Create `GET /api/remix/{session_id}/progress` SSE endpoint
  - Streams progress events as they occur
  - Client connects after receiving session ID from upload
- [ ] Create `GET /api/remix/{session_id}/audio` endpoint
  - Serves the rendered remix MP3 file
  - Returns 404 if remix doesn't exist or has expired
- [ ] Implement TTL management:
  - Background task that runs every 5 minutes
  - Deletes remix data (uploads, stems, remix) older than 3 hours
  - Also deletes when a new remix is created for the same client (track by session cookie or similar)
- [ ] Error handling: if any step fails, emit an error event via SSE with a user-friendly message

**Files created:** `src/musicmixer/services/pipeline.py`
**Files modified:** `src/musicmixer/api/routes.py`

---

### Backend Phase Summary

| Phase | Depends On | Can Parallelize With | Estimated Complexity |
|-------|-----------|---------------------|---------------------|
| B1: Scaffolding | — | F1 (frontend scaffolding) | Low |
| B2: Upload | B1 | F2 (upload UI) | Low |
| B3: Separation | B1 | B4, F2, F3 | Medium (ML dependencies) |
| B4: Analysis | B1 | B3, B5, F3 | Low |
| B5: Interpreter | B1 | B3, B4 | Low |
| B6: Processing | B3, B4, B5 | F4 | Medium (audio math) |
| B7: Orchestrator | B2, B3, B4, B5, B6 | F4, F5 | Medium (integration) |

---

## Frontend Execution Plan

> All frontend work happens in `frontend/`. React + TypeScript + Vite with Bun.

### Phase F1: Project Scaffolding

**Goal:** Runnable React app with basic layout.

- [ ] Initialize project: `bun create vite frontend --template react-ts`
- [ ] Install Tailwind CSS: `bun add -d tailwindcss @tailwindcss/vite`
- [ ] Create project structure:
  ```
  frontend/
  ├── package.json
  ├── CLAUDE.md
  ├── vite.config.ts
  ├── tailwind.config.ts
  ├── tsconfig.json
  ├── index.html
  ├── public/
  └── src/
      ├── main.tsx
      ├── App.tsx
      ├── index.css
      ├── components/           # UI components
      ├── hooks/                # Custom React hooks
      ├── api/                  # API client functions
      └── types/                # TypeScript types
  ```
- [ ] Set up Vite proxy to backend (`/api` → `http://localhost:8000`)
- [ ] Create basic app shell with centered layout
- [ ] Add `CLAUDE.md` for frontend repo with conventions

**Files created:** Standard Vite scaffold + `CLAUDE.md`, `src/api/client.ts`, `src/types/index.ts`

---

### Phase F2: Song Upload UI

**Goal:** Two upload areas where users can add their songs.

- [ ] Create `components/SongUpload.tsx`:
  - Drag-and-drop zone OR click-to-browse (use native `<input type="file">`)
  - Accepts MP3/WAV only (validate client-side)
  - Shows file name and size after selection
  - Visual indicator for "Song A" and "Song B" slots
- [ ] Create `components/UploadSection.tsx` composing two `SongUpload` components
- [ ] Client-side validation: file type, max 50MB
- [ ] Disabled state when processing is in progress

**Design note:** Keep it dead simple. Two boxes, drag or click. No waveform previews, no pre-playback, no metadata display. Those are post-MVP polish.

**Files created:** `src/components/SongUpload.tsx`, `src/components/UploadSection.tsx`

---

### Phase F3: Prompt Input

**Goal:** Text field for the remix description.

- [ ] Create `components/PromptInput.tsx`:
  - Single text input (or textarea for longer prompts)
  - Placeholder text with inspiring examples that rotate:
    - "Hendrix guitar with MF Doom rapping over it"
    - "Just the drums from Song A with everything from Song B"
    - "Slow down Song A and layer the vocals over Song B's beat"
  - Submit button ("Create Remix" or similar)
  - Disabled when no songs uploaded or when processing
- [ ] Create `components/RemixForm.tsx` composing uploads + prompt + submit
- [ ] Wire up form submission: collect files + prompt, POST to `/api/remix`

**Files created:** `src/components/PromptInput.tsx`, `src/components/RemixForm.tsx`
**Files modified:** `src/api/client.ts`

---

### Phase F4: Progress Display

**Goal:** Real-time progress updates while the remix is being created.

- [ ] Create `hooks/useRemixProgress.ts`:
  - Connects to SSE endpoint `GET /api/remix/{sessionId}/progress`
  - Returns current step, detail message, and progress percentage
  - Handles connection errors and reconnection
- [ ] Create `components/ProgressDisplay.tsx`:
  - Shows current step with descriptive text
  - Progress bar (or step indicators)
  - Animated/loading state
  - Error state with "Try again" option
- [ ] Transition from form view → progress view on submission
- [ ] Transition from progress view → player view on completion

**Design note:** The progress display is important for trust and perceived speed. Keep messages human and specific ("Extracting vocals from Song A..." not "Processing step 2 of 7").

**Files created:** `src/hooks/useRemixProgress.ts`, `src/components/ProgressDisplay.tsx`

---

### Phase F5: Audio Player

**Goal:** Play back the rendered remix.

- [ ] Create `components/RemixPlayer.tsx`:
  - HTML5 `<audio>` element with controls
  - Points to `/api/remix/{sessionId}/audio`
  - Shows the LLM's explanation of what it did (from the remix plan)
  - "Create New Remix" button → resets to upload/prompt form
  - Expiration notice ("This remix expires in X hours")
- [ ] Implement app state flow:
  ```
  IDLE → UPLOADING → PROCESSING → READY → IDLE (on new remix)
  ```
- [ ] Handle expired remix: if audio 404s, show friendly message and reset

**Files created:** `src/components/RemixPlayer.tsx`
**Files modified:** `src/App.tsx`

---

### Frontend Phase Summary

| Phase | Depends On | Can Parallelize With | Estimated Complexity |
|-------|-----------|---------------------|---------------------|
| F1: Scaffolding | — | B1 (backend scaffolding) | Low |
| F2: Upload UI | F1 | B2, B3, B4, F3 | Low |
| F3: Prompt Input | F1 | B3, B4, B5, F2 | Low |
| F4: Progress | F1 | B7 | Low-Medium (SSE) |
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
  Response: { "session_id": "uuid" }

GET /api/remix/{session_id}/progress
  Response: SSE stream
    data: { "step": "separating", "detail": "...", "progress": 0.15 }
    ...
    data: { "step": "complete", "detail": "Remix ready!", "progress": 1.0, "explanation": "..." }
    OR
    data: { "step": "error", "detail": "Could not match these songs...", "progress": 0 }

GET /api/remix/{session_id}/audio
  Response: audio/mpeg (MP3 file)
  404 if expired or not found

GET /health
  Response: { "status": "ok" }
```

### Integration Testing Order

1. Backend health check works
2. File upload stores files correctly
3. Each service works in isolation (unit tests)
4. Full pipeline runs end-to-end with test audio files
5. Frontend can upload and receive session ID
6. Frontend receives SSE progress events
7. Frontend plays back rendered audio
8. TTL cleanup works

---

## Open Questions Resolved

| Question (from brainstorm) | Resolution |
|---------------------------|------------|
| Stem granularity | 4-stem separation (vocals/drums/bass/other) using htdemucs_ft. "Guitar" lives in "other" — the LLM prompt interpreter maps user requests like "guitar" to the "other" stem and explains this in its response. |
| File format/size limits | MP3 and WAV. Max 50MB per file, max 10 minutes per song. Configurable via environment variables. |
| Processing architecture | Server-side for everything. Stems separated on server, remix rendered on server, single MP3 served to client. |
| Processing time | With GPU: ~10s separation per song + ~2s analysis + ~1s LLM + ~5s processing = ~30s total. Without GPU: ~5-6min separation per song = ~12-15min total. |
| First-time experience | User lands on the upload/prompt page immediately. Rotating placeholder prompts provide inspiration. No onboarding, no demo — the interface is self-explanatory (two upload boxes + one text field). |
| Error handling | Pipeline reports failure via SSE. User sees friendly message ("These songs were too different to blend — try a different pair or prompt"). User can retry from scratch. No automated alternatives. |

---

## Risk Analysis

| Risk | Impact | Mitigation |
|------|--------|------------|
| CPU-only processing is too slow (12+ min) | Users abandon | Deploy with GPU, or set expectations with progress UI. Consider offering a "fast mode" with lower-quality model. |
| `audio-separator` / PyTorch installation issues | Blocks development | Pin versions, document exact setup in CLAUDE.md. Consider Docker for reproducibility. |
| Essentia installation complexity | Blocks development | Essentia has C++ dependencies. If installation is painful, fall back to librosa chromagram for key detection (less accurate but pure Python). |
| LLM produces bad stem selections | Bad remixes | Include the LLM's explanation in the response so users understand what happened. Iterate on the system prompt. |
| Time-stretching large BPM differences sounds bad | Bad remixes | Cap tempo adjustment to +/- 30% of original. If songs differ by more, warn user. |
| Two songs in very different keys | Bad remixes | Pitch-shifting by more than a few semitones degrades quality. Cap at +/- 5 semitones. If larger shift needed, skip key matching and let the LLM explain. |
| File upload size (50MB x 2 = 100MB) | Slow uploads, memory | Stream uploads to disk, don't buffer in memory. Consider chunked uploads later. |

---

## Recommended Implementation Order

The backend and frontend can be developed **in parallel** since they communicate through a well-defined API contract. Here's the optimal order:

```
Week 1: Foundation
  Backend:  B1 (scaffold) → B2 (upload) → B3 (separation) → B4 (analysis)
  Frontend: F1 (scaffold) → F2 (upload UI) → F3 (prompt input)

Week 2: Intelligence + Integration
  Backend:  B5 (LLM interpreter) → B6 (processing) → B7 (orchestrator + SSE)
  Frontend: F4 (progress display) → F5 (player)

Week 3: Integration + Polish
  End-to-end testing
  Error handling edge cases
  UI polish
```

Backend B3 (separation) and B4 (analysis) can be built in parallel since they're independent services. Similarly, B5 (interpreter) can be built in parallel with B3/B4 since it only needs the metadata interface, not real data.

Frontend phases F2 and F3 can be built in parallel. F4 and F5 can be built against mock data before the backend SSE is ready.

---

## What's NOT in This Plan (Deferred)

Per the brainstorm and PRD, these are explicitly out of scope:

- Iterative refinement ("make the drums louder")
- Export/download/sharing
- Multi-song mixing (3+ songs)
- Voice cloning
- Mobile app
- User accounts / authentication
- Social features
- Search-based song input
- Streaming integration
- Waveform visualization
- Volume/EQ controls in the player
