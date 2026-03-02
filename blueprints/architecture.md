# musicMixer Architecture

> High-level system architecture. Last updated: 2026-02-28.

## System Overview

musicMixer is a web app that creates AI-driven music remixes. Users upload two songs (or paste YouTube URLs), describe the mashup they want, and the system separates stems, generates an LLM-powered arrangement plan, processes the audio, and streams the result back for playback.

```
┌─────────────────────────────────────────────────────────────┐
│                        Browser (SPA)                        │
│                                                             │
│  React 19 + TypeScript + Tailwind v4                       │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────────┐ │
│  │ RemixForm│→ │ProgressDisplay│→ │    RemixPlayer        │ │
│  │ (upload) │  │ (SSE stream) │  │ (turntable + audio)   │ │
│  └──────────┘  └──────────────┘  └───────────────────────┘ │
│       │              ▲                      │               │
│       │ POST         │ SSE                  │ GET /audio    │
└───────┼──────────────┼──────────────────────┼───────────────┘
        │              │                      │
        ▼              │                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Server (:8000)                    │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                   API Layer                          │    │
│  │  POST /api/remix         → file upload + queue       │    │
│  │  POST /api/remix/youtube → URL download + queue      │    │
│  │  GET  /api/remix/:id/progress → SSE stream           │    │
│  │  GET  /api/remix/:id/audio    → MP3 download         │    │
│  │  GET  /api/remix/:id/status   → session snapshot     │    │
│  │  GET  /api/remix/:id/stems    → list available stems │    │
│  └──────────────────────┬──────────────────────────────┘    │
│                         │                                    │
│  ┌──────────────────────▼──────────────────────────────┐    │
│  │              Pipeline (pipeline.py, 16 steps)        │    │
│  │                                                      │    │
│  │  INTAKE                                   ┌────────┐ │    │
│  │   1. Separate stems ─┐ (parallel)         │ Claude │ │    │
│  │      Lookup lyrics ──┘                    │  API   │ │    │
│  │   2. Analyze (BPM, key, structure)        │   ▲    │ │    │
│  │   3. Generate remix plan (LLM) ───────────┼───┘    │ │    │
│  │   3.5 Taste stage (candidate scoring)     └────────┘ │    │
│  │                                                      │    │
│  │  PROCESSING                                          │    │
│  │   5-7.  Load, trim, filter stems                     │    │
│  │   7.75 Corrective EQ per stem                        │    │
│  │   8-9.  Tempo match (rubberband)                     │    │
│  │   10-11. Vocal compression + level matching          │    │
│  │   11.8 Pre-limit drum/bass transients                │    │
│  │   12.  Render arrangement (per-stem gains)           │    │
│  │   12.5 Spectral ducking (vocal pocket)               │    │
│  │   13.  Sum buses                                     │    │
│  │   13.7 Auto-leveler (4s/1.5dB/2.5dB)               │    │
│  │                                                      │    │
│  │  MASTERING                                           │    │
│  │   14. Static mastering (LUFS → limiter → correction) │    │
│  │   14.6 Safety soft clip                              │    │
│  │   15. Fades (2s in / 3s out)                         │    │
│  │   16. Export MP3 (ffmpeg, 320kbps, no pre-dither)    │    │
│  │                                                      │    │
│  │  Each step emits SSE progress events                 │    │
│  └──────────────────────────────────────────────────────┘    │
│                         │                                    │
│  ┌──────────────────────▼──────────────────────────────┐    │
│  │                   Storage (data/)                    │    │
│  │  uploads/{session}/  → original files                │    │
│  │  stems/{session}/    → separated WAVs (~240MB/song)  │    │
│  │  remixes/{session}/  → final MP3 (320kbps)           │    │
│  │  stem_cache/{sha256}/→ content-hash cached stems     │    │
│  └──────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   External Services                         │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  Modal (GPU) │  │ Anthropic    │  │ yt-dlp           │  │
│  │  BS-RoFormer │  │ Claude API   │  │ YouTube download  │  │
│  │  6-stem sep  │  │ Remix plans  │  │ (optional input)  │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐                        │
│  │ syncedlyrics │  │ noembed.com  │                        │
│  │ LRCLIB /     │  │ YouTube      │                        │
│  │ Musixmatch   │  │ oEmbed       │                        │
│  └──────────────┘  └──────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

## Frontend

**Stack:** React 19, TypeScript (strict), Vite, Tailwind CSS v4

### Phase-Based State Machine

The UI is a linear state machine driven by `useReducer`:

```
idle → uploading  → processing → ready
idle → submitting → processing → ready
uploading/submitting/processing → error
error → idle (RETRY)
processing → idle (CANCEL)
ready → idle (RESET)
```

| Phase | Component | What's Happening |
|-------|-----------|-----------------|
| `idle` | RemixForm | Collecting songs (file upload or YouTube URL) + prompt |
| `uploading` | RemixForm | XHR upload with progress bar |
| `submitting` | RemixForm | YouTube URL submission |
| `processing` | ProgressDisplay | SSE stream, vinyl merge animation |
| `ready` | RemixPlayer | Turntable UI, HTML5 audio playback |
| `error` | Error display | Message + retry button |

### Key Components

```
App
└── RemixSession (useReducer state machine)
    ├── RemixForm
    │   ├── SongUpload (x2) — file drag-drop OR YouTube URL
    │   └── PromptInput — textarea with examples
    ├── ProgressDisplay
    │   └── VinylMergeAnimation — two records merge into marbled disc
    │       ├── VinylRecord (x2) — colored vinyl disc SVG
    │       └── PaintMeldFilter — SVG gooey merge filter
    └── RemixPlayer
        └── RecordPlayerView
            ├── TurntableScene — wood plinth, platter, spinning record
            │   ├── RecordLabel
            │   └── Tonearm — pivots with playback progress
            └── FloatingControls — play/pause/rewind
```

### Hooks

| Hook | Purpose |
|------|---------|
| `remixReducer` | Centralized state (18 action types, 6 phases) — plain function, not a hook |
| `useRemixProgress` | SSE connection, timeout handling, monotonic progress |
| `useAudioPlayer` | HTML5 audio element control (play, pause, seek) |
| `useFormPersistence` | Survives refresh: files in IndexedDB, URLs/prompt in sessionStorage |

### API Communication

- **File uploads:** XMLHttpRequest (for `upload.onprogress` tracking)
- **YouTube submissions:** `fetch()` with JSON body
- **Progress streaming:** `EventSource` (SSE) with auto-reconnect
- **Audio playback:** HTML5 `<audio>` element pointed at `/api/remix/:id/audio`
- **Proxy:** Vite dev server proxies `/api` to `localhost:8000`

## Backend

**Stack:** Python 3.11, FastAPI, uv (package manager)

### API Layer

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/remix` | POST | Accept file uploads + prompt, start pipeline |
| `/api/remix/youtube` | POST | Accept YouTube URLs + prompt, start pipeline |
| `/api/remix/{id}/progress` | GET | SSE stream of processing events |
| `/api/remix/{id}/status` | GET | Current session state snapshot |
| `/api/remix/{id}/audio` | GET | Serve finished MP3 |
| `/api/remix/{id}/stems` | GET | List separated stems |
| `/api/remix/{id}/stems/{song}/{name}` | GET | Download individual stem WAV |
| `/health` | GET | Health check |

**Concurrency:** Single-remix lock. One pipeline runs at a time; concurrent requests get 409.

### Feature Flags

After the flag cleanup sprint (Feb 2026), the pipeline was reduced from 9 A/B flags to 1. Eight flags were merged into baseline behavior (their "on" paths became the unconditional code path) and two entire processing stages were deleted (multiband compression, resonance detection). The single remaining flag:

| Flag | Default | Purpose |
|------|---------|---------|
| `ab_taste_model_v1` | `False` | Gate for taste stage (step 4.5): candidate generation + scoring |

The taste model flag is the only conditional in the pipeline. When off (default), the LLM/fallback plan is used directly — the LLM decides the arrangement. When on, the plan goes through candidate generation, constraint filtering, and scoring before selection.

### Processing Pipeline

The pipeline runs in a background thread and emits SSE events at each step. After the flag cleanup (Feb 2026), there is a single unconditional code path — no feature flag conditionals remain except the taste model gate (`ab_taste_model_v1`, default ON).

```
Upload/Download
     │
     ▼
┌─────────────────────────────────────────────┐
│ 1. STEM SEPARATION ──┐                      │
│    Modal (cloud GPU): │ BS-RoFormer → 6     │
│    Local (CPU):       │ htdemucs_ft → 4     │
│    Output: vocals, drums, bass, guitar,     │
│            piano, other (WAV, float32)       │
│    ~60% of total time │                      │
│                       │ (parallel)           │
│ 1'. LYRICS LOOKUP ───┘                      │
│    syncedlyrics → LRCLIB / Musixmatch       │
│    Runs concurrently with stem separation   │
└─────────────────┬───────────────────────────┘
                  ▼
┌─────────────────────────────────────────────┐
│ 2-3. ANALYSIS                               │
│    - BPM detection (librosa beat tracking)  │
│    - Key detection (essentia + fallback)    │
│    - Song structure (section boundaries)    │
│    - Per-bar energy + vocal activity        │
│    - Cross-song BPM reconciliation          │
│    - Map lyrics to bars                     │
└─────────────────┬───────────────────────────┘
                  ▼
┌─────────────────────────────────────────────┐
│ 4. REMIX PLAN (LLM)                        │
│    Anthropic Claude (tool_use)              │
│    5-layer system prompt:                   │
│      1. Role definition                     │
│      2. Song metadata (BPM, key, duration)  │
│      3. Stem character descriptions         │
│      4. Cross-song relationships            │
│      5. Lyrics (if available)               │
│    Output: vocal/instrumental sources,      │
│            time ranges, sections with gains, │
│            tempo strategy, explanation       │
│    Fallback: deterministic plan on failure   │
└─────────────────┬───────────────────────────┘
                  ▼
┌─────────────────────────────────────────────┐
│ 4.5. TASTE STAGE (candidate scoring)       │
│    Generates 8-12 candidate remix plans    │
│    Hard-filters invalid candidates          │
│      (gains in [0,2], sections align, etc.) │
│    Scores survivors (model-based/heuristic) │
│    Returns best plan (400ms timeout,        │
│      circuit breaker fallback)              │
└─────────────────┬───────────────────────────┘
                  ▼
┌─────────────────────────────────────────────┐
│ 5-13. AUDIO PROCESSING                      │
│    5.  Load + standardize (44.1kHz, stereo) │
│    6.  Trim to plan time ranges             │
│    7.  Detect silent stems (< -50 LUFS)     │
│    7.7 Vocal bandpass filter (150Hz-16kHz)  │
│    7.75 Corrective EQ per stem              │
│         (broad Q, ±0.75 dB, halved for      │
│          lossy sources)                      │
│    8.  Compute tempo plan                   │
│    9.  Tempo match (rubberband R3, 6 thr)   │
│    10. Post-stretch beat re-detection       │
│    11. Vocal compression + level matching   │
│    11.5 Cross-song level matching (LUFS)    │
│    11.8 Pre-limit drum/bass transients      │
│    12. Render arrangement (gain curves)     │
│    12.5 Spectral ducking (vocal pocket)     │
│    13. Sum buses + auto-leveling            │
└─────────────────┬───────────────────────────┘
                  ▼
┌─────────────────────────────────────────────┐
│ 14-16. MASTERING + EXPORT                   │
│    14. Static mastering chain               │
│        Optional LPF (16kHz, lossy sources)  │
│        LUFS normalization (-12 LUFS target) │
│        True-peak limiter (-1.0 dBTP, 5ms)  │
│    14.5 Post-mastering LUFS correction      │
│         (iterate-and-converge, +3dB cap,    │
│          second limiter 5ms lookahead)      │
│    14.6 Safety soft clip (ISP guard)        │
│    15. Fade in (2s) / fade out (3s)         │
│    16. Export MP3 (ffmpeg, 320kbps,         │
│        no pre-dither)                        │
└─────────────────────────────────────────────┘
```

### Stem Separation Backends

| | Modal (Cloud GPU) | Local (CPU) |
|-|-------------------|-------------|
| **Model** | BS-RoFormer-SW | htdemucs_ft |
| **Stems** | 6 (vocals, drums, bass, guitar, piano, other) | 4 (vocals, drums, bass, other) |
| **Speed** | 3-5 min/song | 10-20 min/song |
| **Hardware** | A10G GPU (Modal-managed) | CPU only |
| **Cold start** | +60-90s first run | None |
| **Config** | `STEM_BACKEND=modal` (default) | `STEM_BACKEND=local` |

### Key Services

| Module | Lines | Responsibility |
|--------|-------|---------------|
| `analysis.py` | ~1,670 | BPM, key, structure detection, energy profiling |
| `interpreter.py` | ~1,570 | LLM remix plan generation + 8-guard fallback |
| `pipeline.py` | ~990 | 16-step orchestrator + SSE emission |
| `processor.py` | ~860 | DSP: tempo stretch, compression, limiting, export |
| `taste_model.py` | ~820 | Taste scoring model for candidate plans |
| `taste_constraints.py` | ~735 | Hard constraint validation (gains, section alignment) |
| `taste_features.py` | ~710 | Feature extraction for taste scoring |
| `candidate_planner.py` | ~650 | Generates 8-12 remix plan candidates |
| `lyrics.py` | ~485 | Lyrics lookup + bar mapping |
| `youtube.py` | ~346 | yt-dlp wrapper with SSRF prevention |
| `taste_stage.py` | ~272 | Candidate generation + filtering + scoring orchestrator |
| `renderer.py` | ~266 | Section-based arrangement rendering with gain curves |
| `stem_cache.py` | ~206 | Content-hash caching (SHA-256 keyed) |
| `ducking.py` | ~175 | Spectral ducking (vocal pocket carve) |
| `eq.py` | ~161 | Per-stem corrective EQ (presets, lossy-aware, always on) |
| `separation.py` | ~123 | Stem separation dispatcher (modal vs local) |
| `tempo.py` | ~99 | BPM estimation logic |
| `mastering.py` | ~97 | Static mastering chain (LUFS normalize + limiter) |
| `separation_modal.py` | ~81 | Modal cloud GPU wrapper |
| `separation_local.py` | ~69 | Local CPU fallback (htdemucs_ft) |
| `audio_utils.py` | — | Shared audio utilities |
| `remix.py` | ~589 | HTTP endpoints + validation (api/) |

## Data Flow

```
                    ┌─────────┐
                    │  User   │
                    └────┬────┘
                         │ uploads 2 songs + prompt
                         ▼
              ┌──────────────────────┐
              │  data/uploads/{sid}/ │
              │  song_a.mp3          │
              │  song_b.mp3          │
              └──────────┬───────────┘
                         │ stem separation
                         ▼
              ┌──────────────────────┐
              │  data/stems/{sid}/   │
              │  song_a/             │
              │    vocals.wav        │  ~240MB
              │    drums.wav         │  per song
              │    bass.wav          │  (float32
              │    guitar.wav        │   WAV)
              │    piano.wav         │
              │    other.wav         │
              │  song_b/             │
              │    (same structure)  │
              └──────────┬───────────┘
                         │ analysis + LLM plan + DSP
                         ▼
              ┌──────────────────────┐
              │  data/remixes/{sid}/ │
              │  remix.mp3           │  ~8-12MB
              │  (320kbps)           │  (MP3)
              └──────────────────────┘

              ┌──────────────────────┐
              │  data/stem_cache/    │  (persistent across sessions)
              │  {sha256_hex}/       │  keyed by input file hash
              │    vocals.wav        │  avoids re-separating
              │    drums.wav         │  same songs
              │    ...               │
              └──────────────────────┘
```

**Per-session disk usage:** ~500MB (mostly WAV stems)

**Stem cache:** Content-hash (SHA-256 of input file) → cached stem directory. Avoids re-separating the same song. Configurable max size (default 10GB) with LRU pruning.

**Session lifecycle:** In-memory dict (no database). Sessions survive the process lifetime only. Planned: 3-hour TTL with background cleanup.

## SSE Progress Protocol

Events flow from the pipeline through a per-session queue to the client:

```
Pipeline step → emit_progress() → session.events queue → SSE generator → EventSource
```

**Event schema:**
```json
{
  "step": "downloading|separating|analyzing|interpreting|processing|rendering|complete|error|keepalive",
  "detail": "Human-readable status message",
  "progress": 0.0,
  "explanation": "...",   // complete event only
  "warnings": [],         // complete event only
  "usedFallback": false   // complete event only
}
```

**Progress ranges:**
| Range | Phase |
|-------|-------|
| 0.05-0.45 | Downloading (YouTube only) |
| 0.10-0.50 | Separating stems |
| 0.52-0.57 | Analyzing audio |
| 0.58-0.90 | Processing + rendering |
| 0.90-0.95 | Mastering + export |

**Reliability:**
- Keepalive every 5s (prevents proxy timeouts)
- Server replays `last_event` on new connections (client catch-up)
- Queue overflow: non-terminal events dropped, terminal events guaranteed
- 20-minute safety timeout

## External Dependencies

| Service | Purpose | Required? |
|---------|---------|-----------|
| **Modal** | Cloud GPU stem separation (BS-RoFormer) | No (local fallback) |
| **Anthropic Claude API** | LLM-powered remix plan generation | No (deterministic fallback) |
| **yt-dlp** | YouTube audio download | No (file upload always works) |
| **syncedlyrics** | Lyrics lookup (LRCLIB, Musixmatch) | No (pipeline continues without) |
| **noembed.com** | YouTube oEmbed metadata (frontend) | No (cosmetic only) |
| **essentia** | Key detection (primary) | No (librosa fallback) |
| **ffmpeg** | MP3 export | Yes (system dependency) |
| **rubberband** | Tempo stretching | Yes (system dependency) |

## Concurrency Model

```
Main Thread (uvicorn)
  │
  ├── Request handlers (async FastAPI)
  │     ├── processing_lock (1 remix at a time)
  │     └── sessions_lock (thread-safe session dict)
  │
  ├── Pipeline executor (ThreadPoolExecutor, max_workers=1)
  │     └── 16-step pipeline runs here (serial, 1 remix at a time)
  │           ├── Stem separation pool (max_workers=2, per-request)
  │           │     └── Both songs separated concurrently
  │           └── YouTube download pool (max_workers=2, per-request)
  │                 └── Both songs downloaded concurrently
  │
  ├── SSE reader pool (ThreadPoolExecutor, max_workers=4)
  │     └── Concurrent SSE connections
  │
  └── Lyrics worker pool (ThreadPoolExecutor, max_workers=2, per-request)
        └── Runs concurrently with stem separation
```

## Audio Processing Conventions

- **Sample rate:** 44.1kHz throughout
- **Channels:** Stereo (2ch)
- **Bit depth:** float32 from stem separation through final mix
- **Gain scale:** Linear (0.0 = mute, 1.0 = unity)
- **Time signature:** 4/4 assumed (bars = beats / 4)
- **Beat grid:** Librosa beat tracking (not constant-BPM math)
- **Loudness target:** -12 LUFS
- **Peak ceiling:** -1.0 dBTP (true-peak limited)
- **Export format:** MP3, 320kbps via ffmpeg

## Resilience Patterns

Every non-trivial stage has a timeout + fallback:

| Stage | Timeout | Fallback |
|-------|---------|----------|
| LLM remix plan | 20s | Deterministic arrangement |
| Taste scoring | 400ms | Pass-through (use LLM plan as-is) |
| Key detection (essentia) | — | Librosa fallback |
| Stem separation (Modal) | — | Local CPU backend |
| Lyrics lookup | 15s/song | Pipeline continues without lyrics |
| DSP steps (stretch, master) | 120s | Error (no safe fallback) |
| SSE connection | 20min | Emit error event, close stream |

Monotonic progress is enforced in the SSE stream (high-water mark prevents backward values). Queue overflow drops non-terminal events; terminal events (complete/error) are always preserved.

## Directory Structure

```
musicMixer/                     ← workspace root (git repo)
├── CLAUDE.md                   ← workspace instructions
├── blueprints/                 ← architectural docs (this file)
├── docs/impl/                  ← implementation plans
├── notes/                      ← session notes (gitignored)
├── examples/                   ← test songs
│
├── frontend/                   ← independent git repo
│   ├── src/
│   │   ├── api/client.ts       ← API client (fetch + XHR)
│   │   ├── components/         ← React components
│   │   ├── hooks/              ← useReducer, SSE, audio, persistence
│   │   ├── types/index.ts      ← discriminated unions
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── vite.config.ts
│   ├── package.json
│   └── CLAUDE.md
│
└── backend/                    ← independent git repo
    ├── src/musicmixer/
    │   ├── main.py             ← FastAPI app + lifespan
    │   ├── config.py           ← Pydantic Settings
    │   ├── models.py           ← dataclasses
    │   ├── api/
    │   │   ├── health.py
    │   │   └── remix.py        ← all remix endpoints
    │   └── services/
    │       ├── pipeline.py          ← 16-step orchestrator
    │       ├── separation.py        ← stem separation dispatcher
    │       ├── analysis.py          ← BPM, key, structure
    │       ├── interpreter.py       ← LLM remix plan + fallback
    │       ├── processor.py         ← DSP (tempo, compression, limiting)
    │       ├── renderer.py          ← arrangement rendering
    │       ├── mastering.py         ← LUFS normalize + true-peak limiter
    │       ├── ducking.py           ← spectral ducking
    │       ├── eq.py                ← per-stem corrective EQ
    │       ├── taste_stage.py       ← taste candidate orchestrator
    │       ├── taste_model.py       ← taste scoring model
    │       ├── taste_constraints.py ← hard constraint validation
    │       ├── taste_features.py    ← feature extraction for scoring
    │       ├── candidate_planner.py ← remix plan candidate generation
    │       ├── lyrics.py            ← lyrics lookup + bar mapping
    │       ├── youtube.py           ← yt-dlp + SSRF prevention
    │       ├── stem_cache.py        ← content-hash stem caching
    │       ├── tempo.py             ← BPM estimation
    │       ├── audio_utils.py       ← shared audio utilities
    │       ├── separation_modal.py  ← Modal cloud GPU wrapper
    │       └── separation_local.py  ← local CPU fallback
    ├── pyproject.toml
    └── CLAUDE.md
```
