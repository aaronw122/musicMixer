# musicMixer — Audio Engineering Architecture

> Generated from full source analysis of all audio-related code in both backend and frontend
> Perspective: Sound engineering and DSP

---

## 1. Audio Pipeline Overview

End-to-end flow from raw uploaded audio to final remix playback:

```
Upload (MP3/WAV/YouTube) → Separation → Analysis → BPM Reconciliation →
Song Structure → Lyrics Lookup → LLM Prompt Interpretation →
[optional: Taste Stage] → Stem Standardization →
Trim → Silent Stem Filter → Vocal Bandpass Pre-filter → Preset EQ →
Tempo Plan → Rubberband Time-Stretch → Post-stretch Beat Re-detection →
Vocal Compression → Cross-song Level Match → Drum/Bass Pre-limit →
Arrangement Render → Spectral Ducking → Bus Sum → Auto-leveler →
Static Mastering → Post-master LUFS Correction → Safety Soft Clip →
Fades → MP3 Export (ffmpeg 320kbps) → FileResponse serve
```

**Pipeline entry point:** `backend/src/musicmixer/services/pipeline.py` — `run_pipeline()`

Steps 1-2 (separation + analysis) run concurrently. Rubberband time-stretching (step 9) runs all stems in parallel. Everything else is sequential.

---

## 2. Stem Separation

### 2.1 Backend-Agnostic Dispatcher

`separation.py` → `separate_stems(audio_path, output_dir)`:
1. Check SHA-256 stem cache
2. On miss: dispatch to Modal GPU or local CPU based on `STEM_BACKEND` setting
3. Write result to cache

### 2.2 Modal (Primary) — BS-Roformer-SW

- **Model:** `BS-Roformer-SW.ckpt` (by jarredou) — state-of-the-art 6-stem separation
- **Stems:** vocals, drums, bass, guitar, piano, other
- **Hardware:** NVIDIA A10G GPU on Modal.com
- **Cold start:** 60-90s first invocation; model weights pre-baked into container image
- **Output:** `dict[str, bytes]` — each stem as float32 WAV bytes
- **Stem name mapping:** Token-based filename matching (splits on `_-.\s()`, matches stem names as whole tokens)

### 2.3 Local Fallback — htdemucs_ft

- **Model:** `htdemucs_ft.yaml` — 4-stem separation
- **Stems:** vocals, drums, bass, other (no guitar/piano — set to `None`)
- **Speed:** 10-20 min per song on CPU
- **Post-processing:** Re-encodes to float32 WAV (htdemucs_ft outputs 16-bit)

### 2.4 Stem Cache

- **Key:** SHA-256 of input file bytes (chunked 64KiB reads)
- **Storage:** `data/stem_cache/{sha256_hex}/*.wav`
- **Atomic writes:** temp dir + `os.rename()`
- **LRU eviction:** oldest by mtime, triggered when total exceeds 10GB default
- **Validation:** checks recognized stem set (4 or 6) with non-zero file sizes

---

## 3. Tempo Detection and Matching

### 3.1 Detection

- **Library:** librosa `beat.beat_track()` at 22050 Hz with `hop_length=512`
- **Confidence:** tempogram peak sharpness ratio (`max / mean`), normalized to 0-1
- **Total beats:** rounded to nearest bar (4-beat grid)

### 3.2 BPM Reconciliation

For each song, 5 BPM interpretations: original (0% penalty), halved (5%), doubled (5%), x3/2 (15%), x2/3 (15%). All outside 70-180 BPM discarded. Best pair minimizes `gap + penalty_a + penalty_b`.

When interpretation is "halved", `beat_frames` is subsampled `[::2]`. When "doubled", midpoints interpolated and interleaved.

### 3.3 Target BPM Computation

Tiered weighted midpoint, instrumental-biased:

| BPM gap | Strategy |
|---------|----------|
| ≤ 4% | Use instrumental BPM exactly (DJ-transparent) |
| 4-10% | `inst * 0.65 + vocal * 0.35` |
| 10-20% | `inst * 0.70 + vocal * 0.30` |
| > 20% | 70/30 with clamping: max vocal stretch at 12% |

### 3.4 Time-Stretching

- **Tool:** rubberband CLI (v3+) via subprocess
- **Engine:** `-3` (R3/fine mode) for v3+; `--crisp 5` for v2
- **Time ratio:** `source_bpm / target_bpm` (duration ratio, not speed ratio)
- **Formant preservation:** `--formant` applied to all vocal stems
- **Parallelism:** all stems stretched concurrently in `ThreadPoolExecutor(max_workers=6)`
- **Skip-at-unity:** stems within 0.001 of ratio 1.0 returned unchanged
- **Stretch limits:** ≤25% slowdown / ≤30% speedup (silent); 25-35%/30-45% (warn); beyond that, stretching disabled

### 3.5 Post-stretch Beat Re-detection

librosa `beat.beat_track()` on summed mono stretched instrumental at 22050 Hz with `start_bpm=target_bpm`. Falls back to proportionally-scaled original grid if < 10 beats detected.

---

## 4. Key Detection and Matching

### 4.1 Detection

Two-tier approach:

- **Primary (essentia):** `es.KeyExtractor()` at 44100 Hz — professional chromagram + key profile matching. Optional; Python < 3.12 only.
- **Fallback (librosa):** `librosa.feature.chroma_cqt()` at 22050 Hz with Krumhansl-Kessler correlation profiles.

Modulation detection: overlapping 60s windows with 30s hop; flagged if consecutive windows disagree on key.

### 4.2 Key-based Pitch Shifting

Designed in the system (`key_source` field in RemixPlan, Camelot wheel distance in taste scoring) but **not yet applied to audio**. `rubberband_process()` always receives `semitones=0` in current pipeline.

---

## 5. Mixing Engine

### 5.1 Corrective EQ

Applied per-stem before tempo stretching. Uses **pedalboard** (Spotify's audio plugin library).

| Stem | Filters |
|------|---------|
| `vocals` | HPF@80Hz, peak -1.5dB@250Hz, peak -1.5dB@800Hz, peak +0.75dB@3kHz, LPF@16kHz |
| `drums` | HPF@30Hz, peak -1.5dB@400Hz, peak -2dB@800Hz, peak +0.75dB@5kHz, high-shelf -1dB@12kHz |
| `bass` | HPF@30Hz, peak -2dB@250Hz, peak -2dB@800Hz, LPF@8kHz |
| `guitar` | HPF@80Hz, peak -2dB@200Hz, peak -1.5dB@1.2kHz, peak +0.75dB@3.5kHz, LPF@14kHz |
| `piano` | HPF@60Hz, peak -1.5dB@300Hz, peak +0.75dB@2.5kHz, LPF@16kHz |
| `other` | HPF@80Hz, peak -2dB@400Hz, peak +0.5dB@2.5kHz |

All boosts capped at +0.75 dB. Q values: 1.5 (cuts, wider), 2.0 (boosts, narrower). Zero-phase via pedalboard.

### 5.2 Vocal Bandpass Pre-filter

150Hz–16kHz Butterworth bandpass (order 2, zero-phase `sosfiltfilt`). Removes bass rumble bleed and high-frequency separation artifacts before rubberband. 150Hz preserves bass vocal fundamentals.

### 5.3 Vocal Compression

RMS-based feed-forward compressor with noise gate:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Threshold | -20 dBFS | Moderate — only loud phrases |
| Ratio | 3:1 | Moderate compression |
| Attack | 10ms | Preserves plosive transients |
| Release | 80ms | Fast recovery between phrases |
| Makeup gain | +3.0 dB | Compensates for compression |
| Gate floor | -50 dBFS | True silence only |

Uses 10ms analysis frames with leaky integrator smoothing. Gate mask smoothed with attack/release coefficients to prevent clicks.

### 5.4 Cross-song Level Matching

Measures integrated LUFS (via pyloudnorm) of vocal stem and summed instrumental. Target: `instrumental_lufs + 2.0 dB` for vocals. The +2 dB offset balances "vocals buried" (0 dB) vs. "vocals fighting instrumentals" (+3 dB after compression). Gain capped to [-12, +12] dB.

### 5.5 Drum/Bass Pre-limiting

Pre-arrangement brick-wall limiting:

| Stem | Ceiling | Lookahead | Release |
|------|---------|-----------|---------|
| Drums | -6.0 dBTP | 3ms | 30ms |
| Bass | -4.0 dBTP | 5ms | 50ms |

Purpose: drum transients have 12-15 dB crest factor; without pre-limiting, the LUFS normalizer can't boost the mix to target.

### 5.6 Arrangement Renderer

Each `Section` in the `RemixPlan` carries per-stem gains (0.0-1.0 linear) and transition metadata.

**Gain curve construction:**
1. First pass: fill each section's body with flat gain
2. Second pass: cosine interpolation across transition regions (capped at 8 beats and 1/3 of section length)
3. Transition split: 1 beat before boundary, remainder after (asymmetric: quick fade-out, smooth fade-in)
4. Cosine envelope: `prev + (curr - prev) * (1 - cos(t * pi)) / 2`

**Beat-to-sample conversion:** `beats_to_samples()` uses the actual detected beat grid, not constant-BPM math. Extrapolates beyond last beat using average interval of last 8 beats.

Output: `(vocal_bus, instrumental_bus)` — both float32 stereo `(N, 2)` — kept separate for spectral ducking.

### 5.7 Spectral Ducking

Carves a mid-range pocket (300-3000 Hz) in the instrumental when vocals are active:

1. Bandpass vocal to 300-3500 Hz (4th-order Butterworth, zero-phase; detection band wider than ducking band to capture harmonics)
2. Per-frame RMS in 50ms frames
3. Hysteresis: onset = `max(noise_floor * 4.0, 1e-5)`, offset = `max(noise_floor * 2.0, ...)` (noise_floor = 10th percentile)
4. Smooth mask: 30ms attack, 150ms release (exponential IIR)
5. Extract mid-band from instrumental (4th-order Butterworth zero-phase bandpass — **critical:** zero-phase prevents comb-filter artifacts)
6. Reduce mid-band by **-3.5 dB** where mask is active: `result = instrumental - mid_band * (1 - gain) * mask`

Default cut depth: -3.5 dB (conservative, creates space without making instrumental hollow).

**Important:** Uses a copy of the array; never mutates input `instrumental_bus`. The auto-leveler needs the un-ducked version as its detector signal.

### 5.8 Auto-leveler (Slow AGC)

Window-based gain control with 75% overlap:

| Parameter | Value |
|-----------|-------|
| Window | 4 seconds |
| Max boost | +1.5 dB |
| Max cut | -2.5 dB |
| Target | 50th percentile RMS of active windows |
| Active floor | -50 dBFS |

**Critical:** `detector_audio` is set to `instrumental_bus` (not the mixed signal). Without this, vocals trigger gain reduction on themselves, causing a volume-dip regression at ~11s/~22s.

---

## 6. Audio Formats

### 6.1 Input Formats

- File extensions: `.mp3`, `.wav`
- Size limit: 50 MB per file
- YouTube: any video ≤15 minutes; audio extracted via yt-dlp to WAV

### 6.2 Intermediate Formats

| Stage | Format | Sample Rate | Channels | Bit Depth |
|-------|--------|-------------|----------|-----------|
| Stems | WAV | 44100 Hz | Stereo | float32 |
| Stem cache | WAV | 44100 Hz | Stereo | float32 |
| Pipeline arrays | numpy ndarray | 44100 Hz | Stereo (N, 2) | float32 |
| Analysis (BPM/beat) | in-memory | 22050 Hz | Mono | float32 |
| Key detection (essentia) | in-memory | 44100 Hz | Mono | float32 |
| rubberband I/O | WAV (temp) | 44100 Hz | Stereo | float32 |

### 6.3 Output Format

- Container: MP3
- Encoder: libmp3lame at **320 kbps**
- Dithering: `use_s16_dither=False`
- Target loudness: **-12 LUFS** integrated
- Peak ceiling: **-1.0 dBTP**

### 6.4 Standardization

`validate_stem()` enforces 44100 Hz stereo float32 on load. Resampling via `librosa.resample(..., res_type="soxr_hq")`. Mono duplicated to stereo via `np.column_stack([audio, audio])`.

---

## 7. FFmpeg and CLI Tool Usage

### FFmpeg

**MP3 export** (`processor.py` — `export_mp3()`):
```
ffmpeg -y -i <float32_wav> -codec:a libmp3lame -b:a 320k <output.mp3>
```
120-second timeout. Temp WAV cleaned up in `finally` block.

**YouTube audio extraction:** Via yt-dlp's `FFmpegExtractAudio` postprocessor (yt-dlp manages subprocess internally). Output is PCM WAV at native YouTube sample rate.

### rubberband

Separate CLI tool invoked via subprocess:
- Version detected at startup and cached
- `-3` flag for R3/fine mode (v3+)
- `--formant` for vocal stems
- Float32 WAV I/O via soundfile

---

## 8. AI-Driven Mix Decisions

### 8.1 LLM Interpretation

`interpreter.py` → `interpret_prompt()`:

- **Model:** `claude-sonnet-4-20250514` via Anthropic `tool_use` API
- Produces a structured `RemixPlan` with beat-aligned sections, per-stem gains, transitions
- **Context fed:** BPM, key, duration, section map with energy labels, stem character, vocal gaps, cross-song relationships, bar-synced lyrics

### 8.2 5-Layer Song Data

1. Song overview: BPM, key, duration, beat count, energy profile
2. Song structure: Sections with energy levels, vocal status, annotations (GOOD INSTRUMENTAL SOURCE, DROP, BUILD)
3. Stem character: Per-stem energy and vocal prominence
4. Cross-song relationships: LUFS diff, stretch %, Camelot distance
5. Lyrics: Bar-mapped for arrangement-aware decisions

### 8.3 Deterministic Fallback

On LLM failure: 5-section arrangement (intro → verse → breakdown → drop → outro), 210-second target. `plan.used_fallback = True`.

### 8.4 Taste Training (Experimental, Off by Default)

When `AB_TASTE_MODEL_V1=True`:
1. Generate 8-16 candidates across 4 arrangement families
2. Filter via 14 hard constraints
3. Score via 7-dimension heuristic (arrangement quality, energy arc, vocal intelligibility, harmonic fit, transition quality, groove coherence, loudness/fatigue)
4. 400ms timeout with circuit breaker (5 failures → 10-min cooldown)

---

## 9. Energy Profiling

### 9.1 Per-bar RMS

`analysis.py` → `analyze_stems()`:
- Bar grid: every 4th beat from librosa beat frames
- Per-stem per-bar RMS computed for all stems individually

### 9.2 Vocal Activity Detection

Dual-threshold hysteresis on vocal stem bar-level RMS:
- Onset: 15% of stem peak
- Sustain: 8% of stem peak
- Minimum duration: 2 bars

Produces `vocal_active: np.ndarray` (boolean per bar).

### 9.3 Adaptive Energy Bucketing

Pre-normalization noise floor: -60 dBFS. Combined energy = equal-weighted sum of all stem RMS, normalized to p99 = 1.0. Five buckets: silent (<0.02), low (<p10), medium (p10-p50), high (p50-p85), peak (>p85).

### 9.4 Section Detection

1. Energy derivative boundary detection: `find_peaks` on smoothed gradient, snapped to 4-bar phrase grid
2. Labeling: based on energy level, trajectory, vocal activity, density
3. Merging: adjacent same-energy sections merged, minimum 4-bar section length
4. Labels: intro, verse, chorus, instrumental, breakdown, build, outro

---

## 10. Playback Architecture

### 10.1 Server-side

`GET /api/remix/{session_id}/audio` → `FileResponse` with `media_type="audio/mpeg"`. Direct file serving, no chunked streaming or range requests.

### 10.2 Client-side

Pure HTML5 `<audio>` element. No Web Audio API, no AudioContext. The frontend has zero awareness of stems, buses, or DSP — it receives a finished MP3 URL and plays it.

`useAudioPlayer` hook manages playback state and controls. `RecordPlayerView` coordinates audio state with turntable animation (tonearm position tracks playback progress).

### 10.3 Progress Updates

SSE stream (`GET /api/remix/{session_id}/progress`) with keepalive every 5 seconds. Frontend enforces monotonic progress via `maxProgressRef`.

---

## 11. Quality Considerations and Known Limitations

### Stem Separation
- BS-Roformer-SW is state-of-the-art but separation artifacts ("bleed") remain the primary quality floor
- Vocal bandpass pre-filter (150Hz-16kHz) partially compensates for bass bleed

### Tempo Matching
- rubberband R3/fine mode is highest quality available
- Formant preservation on vocals prevents "chipmunk" effect
- Timbre artifacts audible above ~15% stretch
- Beyond ~35%/~45% slowdown/speedup, stretching disabled entirely

### Key Compatibility
- Pitch shifting designed but **not yet active** (always 0 semitones)
- Harmonically incompatible songs (large Camelot distance) will produce dissonant mixes
- LLM is informed of Camelot distance but cannot fix without pitch-shifting

### Mastering
- pyloudnorm integrated LUFS can be inaccurate on short remixes or silence-heavy material
- -40 LUFS floor guard prevents extreme normalization
- True-peak measurement (4x oversampled) can underestimate by ~0.3 dB

### Spectral Ducking
- -3.5 dB cut is intentionally conservative; optimal depth is material-dependent
- No sidechain compression; frequency-selective ducking only

### Storage
- No session TTL enforcement — `data/` grows unbounded except for stem cache LRU
- Each session leaves ~500 MB of stems without automated cleanup
- Stem cache (10 GB limit) is the only automated cleanup

### Playback
- No streaming: entire MP3 must download before seeking works in most browsers
- No client-side audio manipulation — all processing server-side and baked in

### Concurrency
- One remix at a time (enforced by threading.Semaphore, HTTP 409 if busy)

---

## 12. Mastering Chain

### Step 14: Static Mastering (`mastering.py` → `master_static()`)

1. Optional gentle LPF (order 2 Butterworth, zero-phase) at `lossy_lpf_hz` for YouTube sources (16 kHz for Opus — rolls off codec artifacts)
2. Constrained LUFS normalization to -12 LUFS with +3 dB headroom allowance (lets signal overshoot ceiling; limiter follows)
3. True-peak limiter at -1.0 dBTP

### Step 14.5: Post-master LUFS Correction

Limiter eats 2-3 dB of integrated loudness. If measured LUFS > 1 dB below target, applies correction gain (capped at +3 dB) + second lighter limiter pass.

### Step 14.6: Safety Soft Clip

tanh waveshaper at -1.0 dBTP with 2 dB knee. Handles inter-sample true peaks that MP3 encoding reconstructs above limiter ceiling. Below threshold: bit-identical. Knee region: `threshold + knee_width * tanh((x - threshold) / knee_width)`.

### True-peak Measurement

4x oversampled via `scipy.signal.resample_poly(signal, 4, 1)` with Hamming-windowed FIR.

### True-peak Limiter

Block-based (64 samples), lookahead-capable:
1. Compute per-block gain
2. Apply lookahead (minimum gain over lookahead window)
3. Smooth with attack/release IIR at block rate
4. Expand to sample rate
5. Apply

---

## 13. File Lifecycle

### Per-session Footprint (~500 MB)

- 6 stems x ~40 MB x 2 songs = ~480 MB
- Upload files: ≤100 MB (2 x 50 MB limit)
- Remix MP3: ~8 MB (320 kbps x 210 seconds)

### Ephemeral Model

Sessions are in-memory (`dict[str, SessionState]`) with no database. No TTL cleanup implemented — the "3 hour expiry" is informational only. Remix MP3 survives server restart; session metadata does not.

---

## 14. Data Flow Diagram

```
User uploads MP3/WAV (or YouTube URLs)
    |
    v
FastAPI saves to data/uploads/{session_id}/
    |
    v
[ThreadPool: parallel]
  separation_a -----> 6 stem WAVs (44100Hz float32)
  separation_b -----> 6 stem WAVs (44100Hz float32)
  analysis_a -------> AudioMetadata (BPM, beats, key, scale)
  analysis_b -------> AudioMetadata
  [optional] lyrics_a, lyrics_b
    |
    v
reconcile_bpm() --> adjusted BPMs
    |
    v
analyze_stems() --> per-bar RMS, vocal_active[], section map
    |
    v
interpret_prompt() --> Anthropic API --> RemixPlan
    |   (or taste_stage candidates --> score --> select)
    v
validate_stem() --> standardize 44100Hz stereo float32
trim_audio() --> slice to plan time windows
_filter_inactive() --> drop -50 LUFS stems
    |
    v
bandpass_filter(vocals, 150Hz-16kHz)
apply_corrective_eq(all stems)
    |
    v
compute_tempo_plan() --> target BPM
    |
    v
[ThreadPool: parallel]
  rubberband_process(stem, src_bpm, tgt_bpm) x N_stems
    |
    v
re-detect beat grid on stretched instrumental
    |
    v
compress_dynamic_range(vocals)
cross_song_level_match(vocals, instrumental)
true_peak_limit(drums, -6dBTP)
true_peak_limit(bass, -4dBTP)
    |
    v
render_arrangement() --> vocal_bus (N,2), instrumental_bus (N,2)
    |
    v
spectral_duck(instrumental_bus, vocal_bus) --> ducked_instrumental
    |
    v
mixed = vocal_bus + ducked_instrumental
    |
    v
auto_level(mixed, detector=instrumental_bus)
    |
    v
master_static() --> LUFS normalize + limiter
    |
    v
[if LUFS < target-1dB] correction gain + re-limit
    |
    v
soft_clip(mixed, -1.0dBTP, knee=2dB)
    |
    v
apply_fades(2s in, 3s out)
    |
    v
export_mp3(44100Hz, 320kbps, no-dither)
    --> data/remixes/{session_id}/remix.mp3
    |
    v
SSE "complete" event --> client fetches audio URL
    |
    v
<audio src="/api/remix/{id}/audio"> --> browser plays MP3
```

---

## 15. Key Libraries

| Library | Role |
|---------|------|
| `librosa` | BPM, beat tracking, resampling, chroma features |
| `essentia` (optional) | Professional key detection |
| `scipy` | Butterworth filters (sosfiltfilt), resampling, peak finding |
| `numpy` | All audio array operations |
| `soundfile` | WAV I/O (float32 subtype) |
| `pedalboard` | Parametric EQ (Spotify) |
| `pyloudnorm` | Integrated LUFS (ITU-R BS.1770-4) |
| `soxr` | High-quality resampling |
| `audio-separator` | ML stem separation (BS-Roformer, htdemucs) |
| `rubberband` (CLI) | Time-stretching |
| `ffmpeg` (CLI) | MP3 export, YouTube extraction |
| `anthropic` | LLM mix planning |
