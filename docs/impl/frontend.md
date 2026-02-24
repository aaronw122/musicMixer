# Frontend Reference

> Implementation reference extracted from the full plan. Canonical source: `docs/plans/2026-02-23-feat-mvp-prompt-based-remix-plan.md`

## TypeScript Types

```typescript
// === API Contract Types ===

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
  status: 'processing' | 'complete' | 'error';  // Backend maps 'queued' -> 'processing'
  progress?: number;
  detail?: string;
  remix_url?: string;
  explanation?: string;
};

// === Compatibility Analysis Types (QR-3) ===

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

// === App State (Discriminated Union) ===

export type AppState =
  | { phase: 'idle'; songA: File | null; songB: File | null; prompt: string; analyzing: boolean; compatibility: CompatibilityResult | null; analyzeError: string | null }
  | { phase: 'uploading'; songA: File; songB: File; prompt: string; uploadProgress: number }
  | { phase: 'processing'; sessionId: string; progress: ProgressEvent }
  | { phase: 'ready'; sessionId: string; explanation: string }
  | { phase: 'error'; message: string; songA: File | null; songB: File | null; prompt: string };

// === Reducer Actions (Discriminated Union) ===

export type AppAction =
  | { type: 'SET_SONG_A'; file: File | null }
  | { type: 'SET_SONG_B'; file: File | null }
  | { type: 'SET_PROMPT'; prompt: string }
  | { type: 'ANALYZE_STARTED' }
  | { type: 'ANALYZE_COMPLETE'; result: CompatibilityResult }
  | { type: 'ANALYZE_FAILED'; reason: 'transient' }
  | { type: 'ANALYZE_FAILED'; reason: 'invalid_file'; detail: string }
  | { type: 'START_UPLOAD' }
  | { type: 'UPLOAD_PROGRESS'; percent: number }
  | { type: 'UPLOAD_SUCCESS'; sessionId: string }
  | { type: 'PROGRESS_EVENT'; event: ProgressEvent }
  | { type: 'REMIX_READY'; explanation: string }
  | { type: 'ERROR'; message: string }
  | { type: 'RESET' };

// === API Error Types ===

export type CreateRemixError =
  | { type: 'network' }
  | { type: 'timeout' }
  | { type: 'http'; status: number; body: { detail: string } };
```

---

## API Client Functions

```typescript
// api/client.ts

export async function createRemix(
  songA: File, songB: File, prompt: string,
  onUploadProgress?: (pct: number) => void
): Promise<CreateRemixResponse>
// Uses XMLHttpRequest for upload progress (fetch doesn't support it reliably)
// XHR error handling — rejects with CreateRemixError:
//   - onerror: reject { type: 'network' }
//   - ontimeout: reject { type: 'timeout' }
//   - onload status >= 400: reject { type: 'http', status, body: JSON.parse(xhr.responseText) }
//   - onload status 200: resolve with parsed JSON

export async function analyzeCompatibility(songA: File, songB: File): Promise<AnalyzeResponse>
// POST /api/analyze — lightweight (~2-3s)
// Uses fetch (no upload progress needed)
// Called automatically after both songs are uploaded (non-blocking)

export function connectProgress(sessionId: string): EventSource
// Returns native EventSource connected to /api/remix/{sessionId}/progress

export async function getSessionStatus(sessionId: string): Promise<SessionStatus>
// GET /api/remix/{sessionId}/status — for reconnection after refresh

export function getAudioUrl(sessionId: string): string
// Returns /api/remix/{sessionId}/audio URL string
```

---

## State Machine (useReducer)

### Phase Transitions

```
idle -> uploading         (START_UPLOAD)
uploading -> processing   (UPLOAD_SUCCESS)
processing -> ready       (REMIX_READY)
processing -> error       (ERROR)
uploading -> error        (ERROR)
error -> idle             (RESET)
ready -> idle             (RESET)
```

### Idle Phase Actions

- `SET_SONG_A` / `SET_SONG_B`: Update file, clear compatibility if replacing a song
- `SET_PROMPT`: Update prompt text
- `ANALYZE_STARTED`: Set `analyzing: true`
- `ANALYZE_COMPLETE`: Set `compatibility`, `analyzing: false`
- `ANALYZE_FAILED` (transient): Silently ignore, `analyzing: false`
- `ANALYZE_FAILED` (invalid_file): Set `analyzeError` with detail, `analyzing: false`
- `START_UPLOAD`: Transition to uploading (requires songA, songB, prompt all set)

### Processing Phase Actions

- `PROGRESS_EVENT`: Update `progress` (skip keepalives — `progress: -1`)
- `REMIX_READY`: Transition to ready with explanation
- `ERROR`: Transition to error

---

## sessionStorage Persistence

- **Write** session ID on entering `processing` phase
- **Clear** on entering `idle` phase
- **On mount** (RemixSession): Check sessionStorage for active session ID
  - Call `getSessionStatus()`:
    - `processing` -> enter processing phase, reopen EventSource
    - `complete` -> enter ready phase with explanation (skip SSE)
    - `error` -> enter error phase (File objects lost after refresh — songA/songB = null)
    - 404 -> reset to idle

---

## Component Tree

```
App
└── RemixSession (state machine orchestrator, owns useReducer)
    ├── RemixForm (idle + uploading phases)
    │   ├── SongUpload (Song A)
    │   ├── SongUpload (Song B)
    │   ├── CompatibilitySignal
    │   └── PromptInput (textarea + submit button)
    ├── ProgressDisplay (processing phase)
    └── RemixPlayer (ready phase)
```

---

## Component Specs

### SongUpload

- Drag-and-drop zone with native `<input type="file">` as accessible fallback
- `accept=".mp3,.wav,audio/mpeg,audio/wav"`
- Visually-hidden input (sr-only, not `display: none`) — preserves keyboard accessibility
- Shows file name + size after selection
- Visual indicator for "Song A" / "Song B" slots
- Client-side validation: file type, max 50MB. Inline error for rejected files.
- Disabled state when processing

### CompatibilitySignal

- Renders `CompatibilityResult` inline between upload zones and prompt input
- Color accent: green (great), neutral/amber (good), orange (challenging/tough)
- Shows plain-language `message` (never raw BPM/key values)
- Optional collapsed "Details" disclosure for power users
- While analyzing: "Checking compatibility..." subtle text
- If not yet run or failed: renders nothing

**UX rules:**
- Never show raw BPM/key values to user
- Never block or gate submission flow — informational only
- Celebrate good matches ("These songs should blend really well")
- Reframe from judgment to description (no "warning" / "bad" / "rough")

### PromptInput

- Textarea for longer prompts
- Static example prompts **above** the input (not as placeholder):
  - "Put the vocals from Song A over Song B's beat"
  - "Song B vocals over Song A instrumentals, boost the bass"
  - "Slow it down and layer the singing over the other track"
- Submit button ("Create Remix")
- Disabled when: no songs, empty prompt, or processing
- **Disables immediately on click** (before server response) to prevent double submission

### ProgressDisplay

- Progress bar showing percentage
- Step description with human-readable text
- Animated/loading indicator
- Cancel button: closes SSE, resets to idle (backend continues processing, cleaned up by TTL)
- Error state: error message + "Try Again" button (resets to idle)
- Loading micro-state: progress bar at 0% with "Starting..." immediately on entering processing (before first SSE event)

### RemixPlayer

- HTML5 `<audio>` element with native controls
- `src` = `getAudioUrl(sessionId)`
- Shows LLM explanation of what it did
- "Create New Remix" button -> reset to idle (brief confirmation since current remix will be replaced)
- Expiration notice: "This remix will expire in ~3 hours"
- Handle expired remix: `<audio>` error event or 404 from status -> friendly message + reset to idle

---

## SSE Hook (useRemixProgress)

```typescript
// hooks/useRemixProgress.ts
function useRemixProgress(
  sessionId: string | null,
  dispatch: React.Dispatch<AppAction>
): { isConnected: boolean }
```

**Data flow:** Hook dispatches actions directly (`PROGRESS_EVENT`, `REMIX_READY`, `ERROR`). All progress state flows through the reducer. Hook does NOT maintain its own progress state. Returns only `{ isConnected }`.

**Behaviors:**
- Uses native `EventSource` API
- **Cleanup on unmount:** `.close()` the EventSource
- **No-event timeout:** 60 seconds with no event -> dispatch `ERROR`
- **Tab backgrounding:** Listen to `document.visibilitychange`. On refocus, check connection. If dead, call `getSessionStatus()` first, then reopen EventSource.
- **Reconnection:** On disconnect, call `getSessionStatus()`. If complete, dispatch `REMIX_READY`. If processing, reopen EventSource.
- **Max-retry:** After 5 consecutive `onerror` without any `onmessage`, close and dispatch `ERROR` ("Lost connection to the server").
- **Keepalive events:** When `step === "keepalive"`, reset no-event timeout but do NOT dispatch or update progress bar. `progress: -1` is the sentinel.
- **Error events:** Parse `step === "error"`, dispatch `ERROR` with detail message.
- **Malformed events:** Log and ignore. Don't crash.

---

## Compatibility Analysis Trigger

When both songs are selected (`songA` and `songB` non-null):
1. Dispatch `ANALYZE_STARTED`
2. Call `analyzeCompatibility(songA, songB)`
3. On success: dispatch `ANALYZE_COMPLETE`
4. On failure: dispatch `ANALYZE_FAILED` (silently for transient, inline warning for 422 invalid_file)
5. If user replaces a song: clear existing compatibility data and re-trigger

If user submits before analysis completes, proceed without compatibility signal.

---

## Error Taxonomy

| Error | Response | User Sees |
|-------|----------|-----------|
| Network failure | No response | "Upload failed. Check your connection and try again." |
| 413 File too large | Server rejects | "File too large. Maximum 50MB per song." |
| 422 Validation error | Server message | Display server's error message |
| 429 Server busy | Processing lock held | "Another remix is being created. Please wait and try again." |
| 500 Server error | Generic | "Something went wrong. Please try again." |

---

## Layout

- Mobile-first single-column
- Upload zones stack vertically on < 768px
- Side by side on desktop
- Centered max-width container on desktop
- Tagline above form: "Upload two songs. Describe your mashup. AI does the rest."
- UI copy near prompt: "musicMixer takes the vocals from one song and layers them over the other song's beat."

---

## Tech Stack

- React + TypeScript + Vite (Bun)
- Tailwind CSS (verify Preflight is active)
- Vite proxy: `/api` -> `http://localhost:8000`

---

## Cross-References

- API endpoints consumed by client: see `api-contract.md`
- SSE event format and progress sequence: see `api-contract.md`
- LLM explanation field shown in RemixPlayer: see `llm-integration.md`
- Audio pipeline progress steps: see `audio-pipeline.md`
