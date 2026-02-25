# Day 2: "The Real Remix" — Tempo Matching + Arrangement + SSE

**Date:** 2026-02-24
**Source plan:** `docs/plans/2026-02-23-feat-mvp-prompt-based-remix-plan.md` (revision 22)
**Builds on:** Day 1 output (FastAPI skeleton, Modal BS-RoFormer 6-stem separation, pydub overlay, hardcoded sync pipeline, single HTML page)

## Progress

- [ ] Step 1: Async Pipeline Orchestrator + SSE Progress Events
- [ ] Step 2: BPM Detection + Reconciliation
- [ ] Step 3: Sample Rate/Channel Standardization + Tempo Matching via Rubberband
- [ ] Step 4: LUFS Normalization + Peak Limiter
- [ ] Step 5: Deterministic Fallback Plan + Section Data Structures
- [ ] Step 6: Section-Based Arrangement Renderer
- [ ] Step 7: Wire the Full Pipeline
- [ ] Step 8: Update the HTML Test Page with SSE Progress

---

## Exit Criteria

Upload two songs at different BPMs. Hear a tempo-matched, level-balanced, multi-section remix with per-stem volume control (6 stems). See real-time progress updates in the browser.

**Specific verification tests:**

1. Upload a 90 BPM hip-hop track and a 120 BPM pop track. The remix plays at the instrumental's tempo with vocals time-stretched to match. No audible drift between vocal rhythm and instrumental beat.
2. Upload two songs with different loudness levels. The output sounds balanced, not one track drowning the other.
3. The remix has audible arrangement dynamics: instrumental intro, vocals enter, breakdown (drums drop out), vocals return, instrumental outro. Not a flat "two songs at once."
4. The progress bar advances through separation, analysis, processing, rendering stages. Events appear within ~5 seconds of each pipeline step starting.
5. Two rapid-fire submissions: second one gets a proper error, not a crash or silent hang.

---

## Dependencies to Install

### Python packages (add to `backend/pyproject.toml`)

```
librosa              # BPM detection, audio loading, beat tracking
pyloudnorm           # LUFS loudness normalization
soxr                 # High-quality resampling (explicit; librosa falls back to inferior resampy without it)
scipy                # Spectral analysis for vocal offset + bandpass filters (transitive dep of librosa, but pin explicitly)
```

`pyrubberband`, `pydub`, `soundfile`, `numpy` should already be installed from Day 1.

### System dependencies (verify from Day 1)

```bash
# These should already be installed from Day 1, but verify:
rubberband --version    # Must be v3.x for -3 (R3/fine engine) flag
ffmpeg -version         # Required by pydub + MP3 export
```

If `rubberband --version` returns v2.x, the `-3` flag won't work. Fall back to `--crisp 5` (closest R2 equivalent). See Risk Items.

---

## Implementation Order

The steps are sequenced by dependency. Each step lists the files to create/modify, what to build, key decisions, and gotchas.

---

### Step 1: Async Pipeline Orchestrator + SSE Progress Events

**Phase reference:** B7 (partial)
**Why first:** Everything else plugs into this. Moving from Day 1's synchronous pipeline to async+SSE is the architectural backbone.

#### Files to create

| File | Description |
|------|-------------|
| `backend/src/musicmixer/services/pipeline.py` | Pipeline orchestrator: `run_pipeline()` + `emit_progress()` helper |

#### Files to modify

| File | Description |
|------|-------------|
| `backend/src/musicmixer/main.py` | Add `ThreadPoolExecutor`, `sessions` dict, `sessions_lock`, `processing_lock`, lifespan hooks, SSE executor |
| `backend/src/musicmixer/api/remix.py` | Refactor POST handler to use executor + lock, add SSE endpoint, add status endpoint, add audio endpoint |
| `backend/src/musicmixer/models.py` | Add `SessionState` dataclass, `ProgressEvent` model |

#### What to build

**1a. In-memory session state (`models.py`):**

```python
@dataclass
class SessionState:
    status: str                      # "queued" | "processing" | "complete" | "error"
    events: queue.Queue              # Pipeline pushes via emit_progress()
    created_at_mono: float           # time.monotonic()
    remix_path: str | None = None
    explanation: str | None = None
    last_event: dict | None = None   # Most recent event (for reconnecting SSE clients)
```

**1b. Non-blocking event helper (`pipeline.py`):**

```python
def emit_progress(event_queue: queue.Queue, event: dict):
    """Non-blocking event push. Drops non-terminal events on full queue.
    Terminal events (complete/error) drain one old event first."""
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
```

**1c. Pipeline skeleton (`pipeline.py`):**

`run_pipeline(session_id, song_a_path, song_b_path, prompt, event_queue, session)` that:
- Emits progress events at each step
- Sets `session.status`, `session.remix_path`, `session.explanation` directly (queue may have no consumer)
- Catches `MusicMixerError` subclasses, emits SSE error event
- For today: plug in existing Day 1 separation logic, add new analysis/processing/rendering steps as stubs, then fill them in subsequent steps

Progress event sequence (with percentages re-weighted to actual time distribution):

```
{"step": "separating",   "detail": "Extracting stems from both songs...", "progress": 0.10}
{"step": "separating",   "detail": "Separating instruments...",           "progress": 0.26}  (synthetic, after ~30s)
{"step": "separating",   "detail": "Finalizing stem extraction...",       "progress": 0.35}  (synthetic, after ~45s)
{"step": "analyzing",    "detail": "Detecting tempo and key...",          "progress": 0.50}
{"step": "processing",   "detail": "Matching tempo (1/7 stems)...",       "progress": 0.65}
{"step": "processing",   "detail": "Normalizing loudness...",             "progress": 0.80}
{"step": "rendering",    "detail": "Building your remix...",              "progress": 0.90}
{"step": "rendering",    "detail": "Rendering final mix...",              "progress": 0.95}
{"step": "complete",     "detail": "Remix ready!",                        "progress": 1.0, "explanation": "...", "warnings": [...], "usedFallback": true}
```

**1d. POST handler refactor (`remix.py`):**

- Add `processing_lock = threading.Lock()`, `sessions = {}`, `sessions_lock = threading.Lock()`
- Add `executor = ThreadPoolExecutor(max_workers=1)`
- Fail-fast: check `processing_lock.locked()` before accepting upload
- Authoritative gate: `processing_lock.acquire(blocking=False)` after validation
- Submit `pipeline_wrapper` to executor:

```python
def pipeline_wrapper():
    try:
        session.status = "processing"
        run_pipeline(session_id, song_a_path, song_b_path, prompt, session.events, session)
    except BaseException as e:
        try:
            session.status = "error"
            detail = str(e) if isinstance(e, MusicMixerError) else \
                "Something went wrong while creating your remix. Please try again."
            session.events.put_nowait({"step": "error", "detail": detail, "progress": 0})
        except Exception:
            pass
        raise
    finally:
        try:
            processing_lock.release()
        except RuntimeError:
            pass  # Already released
```

**1e. SSE endpoint (`GET /api/remix/{session_id}/progress`):**

```python
sse_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="sse-reader")

async def event_stream(session: SessionState):
    loop = asyncio.get_running_loop()
    start = time.monotonic()

    # On connect: send current state, drain stale events
    if session.last_event:
        yield f"data: {json.dumps(session.last_event)}\n\n"
        if session.last_event["step"] in ("complete", "error"):
            return
    while not session.events.empty():
        try:
            session.events.get_nowait()
        except queue.Empty:
            break

    while True:
        if time.monotonic() - start > 1200:  # 20 min safety cap
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

Return as `StreamingResponse` with headers: `Cache-Control: no-cache`, `X-Accel-Buffering: no`, `Connection: keep-alive`.

**1f. Status endpoint (`GET /api/remix/{session_id}/status`):**

Returns current state as JSON. Maps internal "queued" to "processing" for the client. Reads `session.explanation` directly (not from queue).

**1g. Audio endpoint (`GET /api/remix/{session_id}/audio`):**

Serves the rendered MP3 via `FileResponse`. Returns 404 if not found.

#### Key decisions

- Keepalives use `data:` events (not SSE comments) because `EventSource.onmessage` does not fire for comments.
- Queue `maxsize=100` to bound memory.
- `pipeline_wrapper` wraps `run_pipeline` to keep the pipeline pure (no lock management).

#### Gotchas

- `run_in_executor` with 5-second timeout means SSE reader threads can linger up to 5 seconds after client disconnect. 4 SSE threads is sufficient for single-user MVP.
- Must use `time.monotonic()` (not `time.time()`) for all timing, because `time.time()` is affected by NTP/DST clock changes.

---

### Step 2: BPM Detection + Reconciliation

**Phase reference:** B4 (partial -- BPM + beat_frames + duration only; skip key, energy, groove, vocal prominence)

#### Files to create

| File | Description |
|------|-------------|
| `backend/src/musicmixer/services/analysis.py` | `analyze_audio()` returning BPM, beat_frames, duration; `reconcile_bpm()` cross-song correction |

#### Files to modify

| File | Description |
|------|-------------|
| `backend/src/musicmixer/models.py` | Add `AudioMetadata` dataclass (stripped version for Day 2) |

#### What to build

**2a. AudioMetadata (stripped for Day 2):**

```python
@dataclass
class AudioMetadata:
    bpm: float
    bpm_confidence: float
    beat_frames: np.ndarray   # Beat frame positions from librosa.beat.beat_track
    duration_seconds: float
    total_beats: int          # Total beats (rounded to nearest bar boundary)
    # Day 3+: key, scale, key_confidence, energy_regions, groove_type, vocal_prominence_db
```

**2b. `analyze_audio(audio_path: Path) -> AudioMetadata`:**

- Load with `librosa.load(audio_path, sr=22050)` (22050 is sufficient for BPM detection, saves memory)
- `tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units='frames')` -- store BOTH tempo and beat_frames
- `duration = librosa.get_duration(y=y, sr=sr)`
- `total_beats = round(tempo * duration / 60 / 4) * 4` (round to nearest bar of 4)
- `bpm_confidence`: use `librosa.beat.beat_track` with `start_bpm` parameter, compare results at 60, 90, 120, 150 to gauge stability. For Day 2, a simpler approach: compute `tempogram = librosa.feature.tempogram(y=y, sr=sr)` and measure peak sharpness as a confidence proxy.

**2c. `reconcile_bpm(meta_a, meta_b) -> tuple[AudioMetadata, AudioMetadata]`:**

This is a CROSS-SONG reconciliation. It cannot be inside `analyze_audio()`.

Algorithm (from the plan's expanded interpretation matrix):
1. For each song, generate all plausible BPM interpretations: `{original, halved, doubled, 3/2, 2/3}`
2. Filter to the **70-180 BPM** range (tighter than naive 60-200)
3. Evaluate all valid pairs across both songs
4. Score: `percentage_gap + penalty_a + penalty_b`
   - Original = 0% penalty
   - Doubled/halved = 5% penalty
   - 3/2 and 2/3 = 15% penalty
5. Select the pair with the **minimum score** (not minimum gap)
6. Return updated copies of metadata (don't mutate originals)

```python
def reconcile_bpm(meta_a: AudioMetadata, meta_b: AudioMetadata) -> tuple[AudioMetadata, AudioMetadata]:
    def interpretations(bpm):
        candidates = {
            "original": bpm,
            "halved": bpm / 2,
            "doubled": bpm * 2,
            "3/2": bpm * 3 / 2,
            "2/3": bpm * 2 / 3,
        }
        penalties = {"original": 0.0, "halved": 0.05, "doubled": 0.05, "3/2": 0.15, "2/3": 0.15}
        return {k: (v, penalties[k]) for k, v in candidates.items() if 70 <= v <= 180}

    interps_a = interpretations(meta_a.bpm)
    interps_b = interpretations(meta_b.bpm)

    best_score = float("inf")
    best_pair = (meta_a.bpm, meta_b.bpm)

    for (_, (bpm_a, pen_a)) in interps_a.items():
        for (_, (bpm_b, pen_b)) in interps_b.items():
            gap = abs(bpm_a - bpm_b) / max(bpm_a, bpm_b)
            score = gap + pen_a + pen_b
            if score < best_score:
                best_score = score
                best_pair = (bpm_a, bpm_b)

    # Return copies with reconciled BPMs
    ...
```

#### Key decisions

- Run analysis on the **original full songs**, not the stems. BPM is a property of the whole track.
- Load at 22050 Hz for analysis (not 44100). Half the memory, same BPM accuracy.
- Store `beat_frames` in frame units (not samples or seconds) -- convert to samples at 44100 Hz when needed for the arrangement renderer.

#### Gotchas

- librosa reports the wrong tempo octave ~29% of the time (e.g., 60 instead of 120). The reconciliation step with the expanded interpretation matrix catches most of these.
- The 3/2 and 2/3 ratios can produce false equivalences (e.g., hip-hop at 95 BPM reinterpreted as 142.5 via 3/2). The 15% penalty prevents this from being preferred over the original unless the gap is large.
- **Hip-hop (85) + pop (120) = 41% gap has no half/double/triplet relationship that brings them into range.** This is the most-attempted mashup pairing. The vocals-only stretch strategy in Step 3 handles this.

---

### Step 3: Sample Rate/Channel Standardization + Tempo Matching via Rubberband

**Phase reference:** B6 steps 1, 2, 3

#### Files to create

| File | Description |
|------|-------------|
| `backend/src/musicmixer/services/processor.py` | `process_remix()` orchestrator, `rubberband_process()`, `validate_stem()`, stem trimming |

#### What to build

**3a. Sample rate + channel + format standardization (MUST be first):**

```python
def validate_stem(path: Path, expected_sr: int = 44100) -> tuple[np.ndarray, int]:
    info = sf.info(str(path))
    audio, sr = sf.read(str(path), dtype='float32')
    if sr != expected_sr:
        logger.warning(f"Stem {path.name} at {sr}Hz, resampling to {expected_sr}Hz")
        audio = librosa.resample(audio, orig_sr=sr, target_sr=expected_sr, res_type='soxr_hq')
        sr = expected_sr
    if audio.ndim == 1:
        audio = np.column_stack([audio, audio])  # Mono -> stereo
    return audio, sr
```

**3b. Trim stems to source time ranges:**

Trimming happens BEFORE tempo stretch. The LLM's time ranges (or the deterministic fallback's time ranges) are in the original tempo domain. After stretching, the trimmed segment's duration changes proportionally.

```python
def trim_audio(audio: np.ndarray, sr: int, start_sec: float, end_sec: float) -> np.ndarray:
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)
    return audio[start_sample:end_sample]
```

**3c. Rubberband tempo matching -- the core of Day 2:**

```python
def rubberband_process(audio: np.ndarray, sr: int, source_bpm: float,
                       target_bpm: float, semitones: float = 0,
                       is_vocal: bool = False) -> np.ndarray:
    """Single-pass tempo + pitch via rubberband CLI."""
    # Write input to temp file
    in_path = tmp_dir / f"rb_in_{uuid4().hex[:8]}.wav"
    out_path = tmp_dir / f"rb_out_{uuid4().hex[:8]}.wav"
    sf.write(in_path, audio, sr)

    # CRITICAL: -t takes a TIME RATIO (output_duration / input_duration), NOT speed ratio.
    # Formula: time_ratio = source_bpm / target_bpm
    #   90 BPM -> 120 BPM: time_ratio = 90/120 = 0.75 (shorter output = faster)
    #   120 BPM -> 90 BPM: time_ratio = 120/90 = 1.333 (longer output = slower)
    # INVERTING THIS (target/source) PRODUCES THE OPPOSITE OF THE INTENDED STRETCH.
    time_ratio = source_bpm / target_bpm

    cmd = ["rubberband", "-t", str(time_ratio)]
    if semitones != 0:
        cmd += ["-p", str(semitones)]
    cmd += ["-3"]  # R3 engine (v3.x); fall back to --crisp 5 for v2.x
    if is_vocal and semitones != 0:
        cmd += ["--formant"]
    cmd += [str(in_path), str(out_path)]

    subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    result, _ = sf.read(out_path, dtype='float32')

    # Cleanup temp files
    in_path.unlink(missing_ok=True)
    out_path.unlink(missing_ok=True)

    return result
```

**3d. Tiered stretch limits (direction-aware):**

Slowing down vocals (ratio > 1.0) produces more artifacts than speeding up at the same percentage.

| BPM Difference | Action |
|---|---|
| < 10% | Stretch either/both silently |
| 10-25% | Stretch vocals only (preserve instrumental tempo) |
| 25-30% speedup / 25% slowdown | Stretch vocals only, strongly warn |
| 30-45% speedup / 25-35% slowdown | Vocals-only stretch (rescues hip-hop + pop pairing) |
| > 45% speedup / > 35% slowdown | Skip tempo matching entirely |

**Default tempo target: instrumental source song's tempo.** When `tempo_source` is `"average"`, only use the midpoint if the gap is <15%.

Implementation:

```python
def compute_tempo_plan(vocal_bpm: float, instrumental_bpm: float, tempo_source: str):
    """Decide target BPM and which stems to stretch."""
    target_bpm = instrumental_bpm  # Default: match to instrumental

    if tempo_source == "average":
        gap_pct = abs(vocal_bpm - instrumental_bpm) / max(vocal_bpm, instrumental_bpm)
        if gap_pct < 0.15:
            target_bpm = (vocal_bpm + instrumental_bpm) / 2
        # else: keep instrumental as target

    vocal_ratio = vocal_bpm / target_bpm  # time_ratio for rubberband
    inst_ratio = instrumental_bpm / target_bpm

    # Compute stretch percentage
    vocal_stretch_pct = abs(1.0 - vocal_ratio)
    is_slowdown = vocal_ratio > 1.0  # Vocals need to be stretched longer = slowed down

    warnings = []
    stretch_vocals = True
    stretch_instrumentals = abs(1.0 - inst_ratio) > 0.001

    if vocal_stretch_pct > 0.45 or (is_slowdown and vocal_stretch_pct > 0.35):
        stretch_vocals = False
        warnings.append("Tempo difference too large for stretching. Songs play at original tempos.")
    elif vocal_stretch_pct > 0.30 or (is_slowdown and vocal_stretch_pct > 0.25):
        warnings.append("Vocals stretched significantly to match the beat -- they may sound different.")
    elif vocal_stretch_pct > 0.25:
        warnings.append("Vocals adjusted to match the instrumental tempo.")

    return target_bpm, stretch_vocals, stretch_instrumentals, warnings
```

**3e. Skip-at-unity optimization:**

When `abs(tempo_ratio - 1.0) < 0.001 and abs(semitones) < 0.01`, return audio unmodified. In the typical case (instrumental's tempo is the target), all 5-6 instrumental stems skip rubberband entirely.

**3f. Parallel rubberband invocations:**

Each rubberband call is an independent subprocess on a different file. Use `concurrent.futures.ThreadPoolExecutor` with 4 workers. This turns 6-12 sequential calls into 2-3 batches. Emit per-stem progress events: "Matching tempo (3/7 stems)..."

**3g. Post-stretch beat grid re-detection:**

After tempo matching, re-run `librosa.beat.beat_track` on the stretched audio with `start_bpm=target_bpm` for an accurate post-stretch beat grid. This replaces proportional scaling (`beat_frames * ratio`) which accumulates 0.5-1.5s of drift per minute. Cost: ~0.4s per stem. This MUST happen after rubberband and BEFORE the arrangement renderer uses the grid.

Fallback: if beat detection fails on stretched audio, use proportional scaling and log a warning (section boundaries may be audibly misaligned).

#### Key decisions

- Call the `rubberband` CLI directly via subprocess (not pyrubberband's Python API, which constructs CLI commands internally in undocumented ways).
- On startup, check `rubberband --version` for v3.x. If v2.x, log a warning and use `--crisp 5` instead of `-3`.
- Temp files use unique names (uuid prefix) to prevent collision during parallel execution.

#### Gotchas -- CRITICAL rubberband details from the plan

1. **`-t` takes a TIME RATIO, not a speed ratio.** `time_ratio = source_bpm / target_bpm`. To go from 90 to 120 BPM: `90/120 = 0.75` (shorter output = faster). Inverting this produces the opposite stretch.
2. **`-3` is the short form of `--fine` for the R3 engine.** Requires Rubber Band v3.x. If only v2 is installed, use `--crisp 5`.
3. **Slowing down vocals (ratio > 1.0) produces significantly more artifacts** than speeding up at the same percentage, because slowing requires interpolating new temporal information. The tiered limits are asymmetric for this reason.
4. **Formant preservation (`--formant`) only when pitch-shifting vocals.** Do not include when pitch shift is zero (Day 2 has no pitch shifting, so omit for now).
5. **Single pass per stem.** Tempo + pitch in one invocation. Two passes compound artifacts.
6. **120-second timeout per subprocess.** Prevents hanging on corrupt input.

---

### Step 4: LUFS Normalization + Peak Limiter

**Phase reference:** B6 steps 4, 8, 9, 10, 11

#### Files to modify

| File | Description |
|------|-------------|
| `backend/src/musicmixer/services/processor.py` | Add `cross_song_level_match()`, `lufs_normalize()`, `soft_clip()`, `true_peak()`, fade functions, MP3 export |

#### What to build

**4a. Cross-song level matching:**

Measure LUFS of the vocal stem and the summed instrumental stems BEFORE any filtering. Apply a fixed +3 dB vocal offset for Day 2 (the spectral-density-adaptive offset is deferred to post-demo).

```python
LUFS_FLOOR = -40.0  # Below this, the stem is effectively silence/noise

def cross_song_level_match(vocal_audio, instrumental_sum, sr):
    meter = pyloudnorm.Meter(sr)
    vocal_lufs = meter.integrated_loudness(vocal_audio)
    instrumental_lufs = meter.integrated_loudness(instrumental_sum)

    if vocal_lufs < LUFS_FLOOR or instrumental_lufs < LUFS_FLOOR:
        logger.warning(f"Skipping level matching: vocal={vocal_lufs:.1f}, instrumental={instrumental_lufs:.1f}")
        return vocal_audio

    vocal_offset_db = 3.0  # Fixed for Day 2; spectral-density-adaptive in post-demo
    target_vocal_lufs = instrumental_lufs + vocal_offset_db
    gain_db = target_vocal_lufs - vocal_lufs
    gain_db = np.clip(gain_db, -12.0, 12.0)  # Safety cap
    return vocal_audio * (10 ** (gain_db / 20.0))
```

**4b. LUFS normalization on final mix:**

```python
def lufs_normalize(mixed: np.ndarray, sr: int, target_lufs: float = -14.0) -> np.ndarray:
    meter = pyloudnorm.Meter(sr)
    current_lufs = meter.integrated_loudness(mixed)
    if current_lufs < LUFS_FLOOR:
        logger.warning(f"Final mix near-silent ({current_lufs:.1f} LUFS), skipping normalization")
        return mixed
    gain_db = target_lufs - current_lufs
    gain_db = np.clip(gain_db, -12.0, 12.0)
    return mixed * (10 ** (gain_db / 20.0))
```

Important: normalize the FINAL MIX, not individual stems. Per-stem normalization to -14 LUFS causes the sum to land around -11 LUFS, forcing the limiter to crush the audio.

**4c. True peak measurement + soft-knee clipper:**

```python
from scipy.signal import resample_poly

def true_peak(signal: np.ndarray) -> float:
    """4x oversampled true-peak measurement (practical ITU-R BS.1770-4 approximation)."""
    if signal.ndim == 2:
        return max(true_peak(signal[:, ch]) for ch in range(signal.shape[1]))
    upsampled = resample_poly(signal, 4, 1)
    return float(np.max(np.abs(upsampled)))

def soft_clip(signal: np.ndarray, ceiling: float, knee_db: float = 6.0) -> np.ndarray:
    """Soft-knee clipper at -1.0 dBTP ceiling.
    Below threshold: UNCHANGED (bit-identical).
    Knee region: quadratic compression (C1 continuous at both boundaries).
    Above ceiling: hard limit."""
    knee_linear = 10 ** (knee_db / 20.0)
    threshold = ceiling / knee_linear
    result = signal.copy()
    abs_signal = np.abs(signal)

    # Knee region: parabolic compression
    knee_mask = (abs_signal > threshold) & (abs_signal <= ceiling)
    if np.any(knee_mask):
        x = abs_signal[knee_mask]
        knee_width = ceiling - threshold
        t = (x - threshold) / knee_width  # 0 to 1
        compressed = threshold + knee_width * (2*t - t*t)
        result[knee_mask] = np.sign(signal[knee_mask]) * compressed

    # Hard limit above ceiling
    over_mask = abs_signal > ceiling
    result[over_mask] = np.sign(signal[over_mask]) * ceiling
    return result
```

Apply after LUFS normalization:

```python
ceiling = 10 ** (-1.0 / 20.0)  # -1.0 dBTP ~ 0.891
peak = true_peak(mixed)
if peak > ceiling:
    mixed = soft_clip(mixed, ceiling)
```

**4d. Fade-in / fade-out (equal-power cosine-squared curves):**

```python
def apply_fades(audio: np.ndarray, sr: int, fade_in_sec: float = 2.0, fade_out_sec: float = 3.0,
                skip_fade_in: bool = False, skip_fade_out: bool = False) -> np.ndarray:
    """Apply equal-power fades. Skip flags prevent double-fading with arrangement transitions."""
    result = audio.copy()
    if not skip_fade_in:
        n_in = int(fade_in_sec * sr)
        fade_in = np.cos(np.linspace(np.pi/2, 0, n_in)) ** 2  # 0 -> 1
        if result.ndim == 2:
            result[:n_in] *= fade_in[:, np.newaxis]
        else:
            result[:n_in] *= fade_in
    if not skip_fade_out:
        n_out = int(fade_out_sec * sr)
        fade_out = np.cos(np.linspace(0, np.pi/2, n_out)) ** 2  # 1 -> 0
        if result.ndim == 2:
            result[-n_out:] *= fade_out[:, np.newaxis]
        else:
            result[-n_out:] *= fade_out
    return result
```

Skip global fade-in if first section's `transition_in == "fade"`. Skip global fade-out if last section's `label == "outro"`.

**4e. MP3 export via ffmpeg (NOT pydub):**

```python
# Write float WAV (preserves full precision)
sf.write(tmp_wav_path, mixed, sr, subtype="FLOAT")
# Encode to MP3 via ffmpeg
subprocess.run(["ffmpeg", "-y", "-i", str(tmp_wav_path), "-codec:a", "libmp3lame",
                "-b:a", "320k", str(output_path)], check=True, capture_output=True, timeout=120)
```

Do NOT use pydub for export. Pydub's `AudioSegment` quantizes to 16-bit integers internally, destroying 32-bit float headroom.

#### Key decisions

- Fixed +3 dB vocal offset for Day 2 (spectral-density-adaptive deferred).
- LUFS normalization target: -14.0 (Spotify/YouTube standard).
- Soft-knee clipper with 6 dB knee width handles 3-6 dB of peak overshoot.

#### Gotchas

- `pyloudnorm.Meter.integrated_loudness()` returns `-inf` for silent/near-silent audio. The `LUFS_FLOOR = -40.0` guard is critical to prevent NaN/inf corruption.
- The `resample_poly` true peak measurement uses a Hamming-windowed FIR (not the ITU-specified polyphase FIR). Can underestimate true peaks by ~0.3 dB. Acceptable for MVP.
- Global gain reduction (`mixed * (ceiling / peak)`) would pull the ENTIRE mix down. The soft-knee clipper only touches samples above the threshold.

---

### Step 5: Deterministic Fallback Plan + Section Data Structures

**Phase reference:** B5 (deterministic fallback only -- LLM integration is Day 3)

#### Files to modify

| File | Description |
|------|-------------|
| `backend/src/musicmixer/models.py` | Add `Section` and `RemixPlan` dataclasses |
| `backend/src/musicmixer/services/interpreter.py` (create) | `generate_fallback_plan()` + `default_arrangement()` |

#### What to build

**5a. Section and RemixPlan data structures:**

```python
@dataclass
class Section:
    label: str              # "intro" | "verse" | "build" | "breakdown" | "drop" | "main" | "outro"
    start_beat: int         # Beat-aligned (snapped to grid)
    end_beat: int
    stem_gains: dict[str, float]  # {"vocals": 1.0, "drums": 0.7, "bass": 0.8, ...}
    transition_in: str      # "fade" | "crossfade" | "cut"
    transition_beats: int   # Length of transition envelope

@dataclass
class RemixPlan:
    vocal_source: str                    # "song_a" | "song_b"
    start_time_vocal: float              # Seconds, original tempo
    end_time_vocal: float
    start_time_instrumental: float
    end_time_instrumental: float
    sections: list[Section]              # Beat-aligned arrangement
    tempo_source: str                    # "song_a" | "song_b" | "average"
    key_source: str                      # "song_a" | "song_b" | "none"
    explanation: str
    warnings: list[str]
    used_fallback: bool = False
```

**5b. Deterministic fallback (5-section arrangement):**

This is the remix plan for all of Day 2 (no LLM until Day 3). The 5 sections from QR-1:

```python
def generate_fallback_plan(meta_a: AudioMetadata, meta_b: AudioMetadata) -> RemixPlan:
    """Deterministic fallback using analysis data. No LLM required."""
    # Pick vocal source: for Day 2 without vocal_prominence_db, default to song_a
    # (Day 3+ uses vocal_prominence_db comparison)
    vocal_src = "song_a"
    vocal_meta = meta_a
    inst_meta = meta_b

    # Use most energetic 90-second region (simplified: start from 25% into the song)
    v_start = vocal_meta.duration_seconds * 0.15
    v_end = min(v_start + 90.0, vocal_meta.duration_seconds)
    i_start = inst_meta.duration_seconds * 0.15
    i_end = min(i_start + 90.0, inst_meta.duration_seconds)

    tempo_src = "song_b"  # Default: match to instrumental's tempo
    total_beats = int(inst_meta.bpm * 90 / 60)  # Beats in 90 seconds at inst tempo

    return RemixPlan(
        vocal_source=vocal_src,
        start_time_vocal=v_start,
        end_time_vocal=v_end,
        start_time_instrumental=i_start,
        end_time_instrumental=i_end,
        sections=default_arrangement(total_beats),
        tempo_source=tempo_src,
        key_source="none",  # No key matching on Day 2
        explanation="We created a remix using the strongest sections of each song. "
                    "Vocals from Song A layered over Song B's instrumentals.",
        warnings=["Using automatic remix layout (no prompt interpretation yet)."],
        used_fallback=True,
    )

def default_arrangement(total_beats: int) -> list[Section]:
    """5-section fallback: intro -> build -> main -> breakdown -> outro."""
    eighth = total_beats // 8
    quarter = total_beats // 4
    three_quarter = total_beats * 3 // 4
    seven_eighth = total_beats * 7 // 8

    return [
        Section(label="intro", start_beat=0, end_beat=eighth,
                stem_gains={"vocals": 0.0, "drums": 0.8, "bass": 0.7, "guitar": 0.6,
                            "piano": 0.5, "other": 1.0},
                transition_in="fade", transition_beats=4),
        Section(label="build", start_beat=eighth, end_beat=quarter,
                stem_gains={"vocals": 0.6, "drums": 0.7, "bass": 0.8, "guitar": 0.5,
                            "piano": 0.4, "other": 0.5},
                transition_in="crossfade", transition_beats=4),
        Section(label="main", start_beat=quarter, end_beat=three_quarter,
                stem_gains={"vocals": 1.0, "drums": 0.7, "bass": 0.8, "guitar": 0.5,
                            "piano": 0.4, "other": 0.5},
                transition_in="crossfade", transition_beats=2),
        Section(label="breakdown", start_beat=three_quarter, end_beat=seven_eighth,
                stem_gains={"vocals": 0.8, "drums": 0.0, "bass": 0.6, "guitar": 0.7,
                            "piano": 0.8, "other": 0.7},
                transition_in="crossfade", transition_beats=4),
        Section(label="outro", start_beat=seven_eighth, end_beat=total_beats,
                stem_gains={"vocals": 0.0, "drums": 0.6, "bass": 0.5, "guitar": 0.5,
                            "piano": 0.6, "other": 0.8},
                transition_in="crossfade", transition_beats=4),
    ]
```

The 5 sections provide: (1) instrumental-only intro establishing the beat, (2) vocals gradually entering, (3) full main section, (4) breakdown with drums at 0.0 for contrast, (5) instrumental outro. This is vastly better than a flat overlay.

#### Key decisions

- Vocal source defaults to `song_a` on Day 2 (no `vocal_prominence_db` yet).
- Target ~90 seconds of remix output.
- Key matching set to "none" for Day 2.
- `total_beats` calculated from the instrumental's BPM and the 90-second target duration.

---

### Step 6: Section-Based Arrangement Renderer

**Phase reference:** QR-1 mixer logic

This is the most complex new code on Day 2. It converts the `RemixPlan.sections` into actual audio.

#### Files to modify

| File | Description |
|------|-------------|
| `backend/src/musicmixer/services/processor.py` | Add `render_arrangement()`, `snap_to_bar()`, transition envelope functions |

#### What to build

**6a. Beat-to-sample conversion using beat grid:**

```python
def beats_to_samples(beat_index: int, beat_frames: np.ndarray, sr: int, hop_length: int = 512) -> int:
    """Convert beat index to sample position using the actual beat grid.
    Uses beat_frames from librosa (NOT constant-BPM math, which drifts)."""
    if beat_index >= len(beat_frames):
        # Extrapolate beyond last detected beat
        if len(beat_frames) < 2:
            return beat_index * sr  # Fallback for degenerate case
        avg_beat_len = np.mean(np.diff(beat_frames[-8:])) * hop_length
        overshoot = beat_index - len(beat_frames) + 1
        return int(beat_frames[-1] * hop_length + overshoot * avg_beat_len)
    return int(beat_frames[beat_index] * hop_length)
```

**6b. Transition envelopes:**

```python
def make_transition_envelope(n_samples: int, transition_type: str, sr: int = 44100) -> np.ndarray:
    """Generate transition envelope for section boundary."""
    if transition_type == "fade":
        # Cosine fade-in: 0 -> 1
        return np.cos(np.linspace(np.pi/2, 0, n_samples)) ** 2
    elif transition_type == "crossfade":
        # Returns the "in" curve; the "out" curve is (1 - in_curve)
        return np.cos(np.linspace(np.pi/2, 0, n_samples)) ** 2
    elif transition_type == "cut":
        # Micro-crossfade: ~2ms (88 samples at 44.1kHz) to eliminate clicks
        micro_len = min(88, n_samples)
        env = np.ones(n_samples)
        env[:micro_len] = np.linspace(0, 1, micro_len)
        return env
    else:
        return np.ones(n_samples)
```

**6c. Arrangement renderer (outputs TWO buses: vocal + instrumental):**

```python
def render_arrangement(
    sections: list[Section],
    vocal_stems: dict[str, np.ndarray],     # {"vocals": array}
    instrumental_stems: dict[str, np.ndarray],  # {"drums": array, "bass": array, ...}
    beat_frames: np.ndarray,  # Post-stretch beat grid
    sr: int,
    hop_length: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Render sections into vocal bus + instrumental bus."""

    # Compute total output length from last section's end_beat
    last_beat = max(s.end_beat for s in sections)
    total_samples = beats_to_samples(last_beat, beat_frames, sr, hop_length)

    vocal_bus = np.zeros((total_samples, 2), dtype=np.float32)
    instrumental_bus = np.zeros((total_samples, 2), dtype=np.float32)

    for i, section in enumerate(sections):
        start_sample = beats_to_samples(section.start_beat, beat_frames, sr, hop_length)
        end_sample = beats_to_samples(section.end_beat, beat_frames, sr, hop_length)
        section_len = end_sample - start_sample

        if section_len <= 0:
            continue

        # Compute transition envelope
        trans_samples = beats_to_samples(
            section.start_beat + section.transition_beats, beat_frames, sr, hop_length
        ) - start_sample
        trans_samples = min(trans_samples, section_len // 2)

        # For crossfade: also apply fade-out to previous section's tail
        if section.transition_in == "crossfade" and i > 0 and trans_samples > 0:
            prev_end = end_sample  # Already at this position in output buffer
            fade_out = np.cos(np.linspace(0, np.pi/2, trans_samples)) ** 2
            # Apply fade-out to both buses in the overlap region
            vocal_bus[start_sample:start_sample + trans_samples] *= fade_out[:, np.newaxis]
            instrumental_bus[start_sample:start_sample + trans_samples] *= fade_out[:, np.newaxis]

        # Build transition-in envelope
        in_env = make_transition_envelope(trans_samples, section.transition_in, sr)
        full_env = np.ones(section_len)
        if trans_samples > 0:
            full_env[:trans_samples] = in_env

        # Apply per-stem gains and add to appropriate bus
        for stem_name, gain in section.stem_gains.items():
            if gain < 0.001:
                continue  # Skip effectively silent stems

            if stem_name == "vocals":
                stem_audio = vocal_stems.get("vocals")
            else:
                stem_audio = instrumental_stems.get(stem_name)

            if stem_audio is None:
                continue  # Missing stem (e.g., guitar/piano in 4-stem fallback)

            # Extract section range from stem (with bounds checking)
            stem_section = stem_audio[start_sample:end_sample]
            actual_len = len(stem_section)
            if actual_len < section_len:
                # Pad with silence if stem is shorter than section
                pad = np.zeros((section_len - actual_len, stem_section.shape[1] if stem_section.ndim == 2 else 1), dtype=np.float32)
                stem_section = np.concatenate([stem_section, pad])

            # Apply gain and transition envelope
            stem_section = stem_section[:section_len] * gain * full_env[:, np.newaxis]

            if stem_name == "vocals":
                vocal_bus[start_sample:end_sample] += stem_section
            else:
                instrumental_bus[start_sample:end_sample] += stem_section

    return vocal_bus, instrumental_bus
```

**6d. `snap_to_bar()` validation utility:**

```python
def snap_to_bar(sample_position: int, beat_positions: np.ndarray, beats_per_bar: int = 4) -> int:
    """Snap a sample position to the nearest bar (downbeat)."""
    bar_positions = beat_positions[::beats_per_bar]
    idx = np.argmin(np.abs(bar_positions - sample_position))
    return int(bar_positions[idx])
```

#### Key decisions

- Output TWO separate buses (vocal + instrumental), not a single summed buffer. This enables spectral ducking on Day 4 (using the vocal bus for activity detection).
- For Day 2, immediately sum the buses after rendering (spectral ducking is Day 4).
- Per-stem equal-power gain interpolation for crossfades (not summed-audio crossfade).
- "Cut" transitions include a 2ms micro-crossfade (88 samples at 44.1kHz) to prevent clicks.

#### Gotchas

- Beat grid extrapolation: if `end_beat` exceeds detected beats, extrapolate using the average of the last 8 beat intervals. Without this, the renderer crashes on the outro section.
- Stems may be shorter than the total arrangement length (especially after trimming + stretching). Pad with silence rather than crashing.
- All stems must be stereo (ensured by Step 3a). The renderer assumes 2-channel audio.

---

### Step 7: Wire the Full Pipeline

**Phase reference:** Connects Steps 1-6

#### Files to modify

| File | Description |
|------|-------------|
| `backend/src/musicmixer/services/pipeline.py` | Fill in `run_pipeline()` with the complete processing chain |

#### What to build

Wire the complete pipeline inside `run_pipeline()`:

```
1. Separate stems (existing Day 1 Modal/local code)
2. Analyze both songs (BPM, beat_frames, duration)
3. Reconcile BPM between songs
4. Generate deterministic fallback plan
5. Standardize stems (44.1kHz, stereo, float32)
6. Trim stems to source time ranges
7. Tempo match vocals to instrumental tempo via rubberband
8. Re-run beat detection on stretched audio (post-stretch beat grid)
9. Cross-song level match (LUFS with +3 dB vocal offset)
10. Render arrangement into vocal bus + instrumental bus
11. Sum buses into final mix
12. LUFS normalize final mix to -14.0 LUFS
13. Peak limiter (soft-knee clip at -1.0 dBTP)
14. Fade-in / fade-out
15. Export to MP3 via ffmpeg
```

Emit progress events between each major step. The event percentages map to actual time distribution:

- 0.00-0.50: Stem separation (50% of time)
- 0.50-0.58: Analysis
- 0.58-0.65: Tempo matching prep + rubberband
- 0.65-0.80: Rubberband execution (per-stem progress updates)
- 0.80-0.85: Level matching
- 0.85-0.95: Arrangement rendering + export
- 0.95-1.0: Finalize

#### Gotchas

- Separation returns `dict[str, Path]` (file paths). Each must be loaded via `validate_stem()` before processing.
- The vocal stems come from the `vocal_source` song, all instrumental stems from the other song.
- `beat_frames` from analysis is at 22050 Hz. Convert to 44100 Hz sample positions: `beat_frames_44k = beat_frames * (44100 / 22050)`. Or better: re-analyze at 44100 Hz (more accurate but slower -- for Day 2, use the scaling approach with a note to revisit).

---

### Step 8: Update the HTML Test Page with SSE Progress

**Phase reference:** F4 (simplified -- Day 3 builds the real React frontend)

#### Files to modify

| File | Description |
|------|-------------|
| `backend/static/index.html` (or wherever Day 1's HTML page lives) | Add EventSource-based progress bar |

#### What to build

Enhance the Day 1 HTML test page:

1. After form submission, open an `EventSource` connection to `/api/remix/{session_id}/progress`
2. Show a progress bar (HTML `<progress>` element) that updates from events
3. Show the current step description text
4. On `complete` event: show the `<audio>` player and explanation text
5. On `error` event: show the error message
6. Handle `keepalive` events (reset timeout, don't update progress)

```html
<div id="progress-section" style="display:none">
    <progress id="progress-bar" value="0" max="100"></progress>
    <p id="progress-text">Starting...</p>
</div>
```

```javascript
const evtSource = new EventSource(`/api/remix/${sessionId}/progress`);
evtSource.onmessage = (e) => {
    const data = JSON.parse(e.data);
    if (data.step === 'keepalive') return;
    if (data.step === 'error') {
        progressText.textContent = `Error: ${data.detail}`;
        evtSource.close();
        return;
    }
    if (data.step === 'complete') {
        // Show audio player
        audioElement.src = `/api/remix/${sessionId}/audio`;
        audioElement.style.display = 'block';
        explanationText.textContent = data.explanation;
        evtSource.close();
        return;
    }
    progressBar.value = Math.round(data.progress * 100);
    progressText.textContent = data.detail;
};
```

---

## Full File Summary

### Files to create

| File | Step | Description |
|------|------|-------------|
| `backend/src/musicmixer/services/pipeline.py` | 1, 7 | Pipeline orchestrator with `run_pipeline()` + `emit_progress()` |
| `backend/src/musicmixer/services/analysis.py` | 2 | BPM detection, beat tracking, duration, BPM reconciliation |
| `backend/src/musicmixer/services/processor.py` | 3, 4, 6 | Audio processing: rubberband, LUFS, limiter, arrangement renderer |
| `backend/src/musicmixer/services/interpreter.py` | 5 | Deterministic fallback plan (LLM integration is Day 3) |

### Files to modify

| File | Step | Description |
|------|------|-------------|
| `backend/src/musicmixer/main.py` | 1 | ThreadPoolExecutor, sessions state, processing lock, SSE executor, lifespan |
| `backend/src/musicmixer/api/remix.py` | 1 | POST handler refactor (async), SSE endpoint, status endpoint, audio endpoint |
| `backend/src/musicmixer/models.py` | 1, 2, 5 | `SessionState`, `AudioMetadata`, `Section`, `RemixPlan`, `ProgressEvent` |
| `backend/static/index.html` | 8 | EventSource progress bar + step text |

---

## Risk Items

### 1. Rubberband integration (HIGH)

**Risk:** Rubberband v3.x may not be installed, or the CLI may behave differently than expected.

**Mitigations:**
- Verify `rubberband --version` at startup. Log warning and fall back to `--crisp 5` for v2.x.
- Test with a known input file FIRST (before wiring into the pipeline): stretch a 5-second WAV from 90 to 120 BPM and verify the output is ~3.75 seconds (0.75 ratio).
- If rubberband CLI is missing entirely, `subprocess.run` will raise `FileNotFoundError`. Catch this and raise a clear `ProcessingError("rubberband CLI not found. Install with: brew install rubberband")`.
- 120-second timeout per subprocess prevents infinite hangs.

### 2. BPM detection accuracy (MEDIUM)

**Risk:** librosa reports wrong tempo octave ~29% of the time.

**Mitigations:**
- The reconciliation step with expanded interpretation matrix catches most octave errors.
- For Day 2, if both songs are detected at the same BPM (or very close), the remix will sound right regardless. The risk materializes when songs are at different tempos AND librosa picks the wrong octave.
- Listen to the output. If BPM sounds wrong, manually verify with `librosa.beat.beat_track` and check if halving/doubling fixes it.

### 3. Memory usage with 6 stems x 2 songs (MEDIUM)

**Risk:** Loading all 12 stems as 44.1kHz stereo float32 arrays simultaneously could consume 2-3 GB of RAM.

**Mitigations:**
- Only load stems that are needed: vocal stems from the vocal source song, instrumental stems from the other.
- After processing each stem through rubberband, write it back to disk and free the array.
- If memory is tight, process stems sequentially rather than holding all in memory.

### 4. Beat grid accuracy after tempo stretching (LOW-MEDIUM)

**Risk:** The post-stretch beat grid from `librosa.beat.beat_track` may not align perfectly with the rubberband output.

**Mitigations:**
- Pass `start_bpm=target_bpm` to `librosa.beat.beat_track` on the stretched audio for better accuracy.
- Proportional scaling fallback: `beat_frames * (source_bpm / target_bpm)`. Acceptable drift for 90-second remixes (~0.5-1.5s cumulative), but log a warning.
- Listen for section boundaries that sound off-beat. If systematic, adjust the beat grid approach.

### 5. SSE event delivery timing (LOW)

**Risk:** Progress events may not reach the client if the queue fills up during long processing steps.

**Mitigations:**
- Queue maxsize=100 provides ample buffer.
- `emit_progress()` drops non-terminal events on full queue (terminal events always delivered).
- 5-second keepalive prevents client timeout during long steps.

---

## Deferred to Day 3+

These are explicitly scoped OUT of Day 2:

| Feature | Day | Why deferred |
|---|---|---|
| LLM prompt interpretation | Day 3 | Deterministic fallback is sufficient for Day 2 |
| Key detection + key matching | Day 4 | Tempo matching alone produces a usable remix |
| Energy profiles / groove detection | Post-demo | LLM works fine with BPM + duration |
| Vocal prominence detection | Day 3 | Affects vocal_source selection (hardcoded for Day 2) |
| Spectral ducking | Day 4 | Requires vocal/instrumental bus separation (built on Day 2, used on Day 4) |
| Spectral-density-adaptive vocal offset | Post-demo | Fixed +3 dB is sufficient for demo |
| Vocal pre-filtering before rubberband | Day 4 | Improves quality but not blocking |
| Beat phase alignment | Day 4 | Fixes subtle "flamming" -- not noticeable in demo |
| Processing lock fail-fast + error handling | Day 4 | Basic lock works for Day 2 |
| TTL cleanup | Day 4 | Manual cleanup for Day 2 |
| React frontend | Day 3 | HTML test page is sufficient for Day 2 validation |
| SSE reconnection / tab backgrounding | Post-demo | Users won't refresh during testing |
| Bleed attenuation ("bleed tax") | Day 4 | Quality refinement |
