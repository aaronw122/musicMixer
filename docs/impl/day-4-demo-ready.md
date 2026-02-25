# Day 4: "Demo Ready" -- Implementation Plan

> Canonical source: `docs/plans/2026-02-23-feat-mvp-prompt-based-remix-plan.md` (revision 22)
>
> **Builds on:** Days 1-3 (full pipeline with Modal 6-stem separation, rubberband tempo matching,
> section-based arrangement renderer, LLM prompt interpretation with Claude Sonnet, React UI with
> SSE progress). By end of Day 3, the app produces tempo-matched, LLM-driven remixes with per-stem
> volume control, playable in a React frontend.
>
> **Day 4 goal:** Polish to demo quality. Add key matching, spectral ducking, comprehensive error
> handling, TTL cleanup, and test across 5+ song pairings. By end of day, the app handles edge
> cases gracefully and produces good-sounding remixes across genres.

---

## Progress

- [ ] Step 1: Key Detection with Essentia
- [ ] Step 2: Key Distance Algorithm + Confidence Gate
- [ ] Step 3: Key/Pitch Shifting via Rubberband
- [ ] Step 4: Vocal Stem Pre-Filtering
- [ ] Step 5: Spectral Ducking
- [ ] Step 6: Wire Ducking into Pipeline
- [ ] Step 7: Processing Lock + Fail-Fast 429
- [ ] Step 8: Upload Validation Hardening
- [ ] Step 9: Pipeline Error Handling + User-Friendly Messages
- [ ] Step 10: TTL Cleanup + Lock Watchdog
- [ ] Step 11: Intermediate Artifact Cleanup
- [ ] Step 12: Frontend Error States
- [ ] Step 13: Warnings Display + used_fallback Indicator
- [ ] Step 14: Expiration Notice + Expired Remix Handling
- [ ] Step 15: UI Copy
- [ ] Step 16: End-to-End Testing Matrix
- [ ] Step 17: LLM Prompt Tuning

---

## Dependencies to Install

### Python packages (backend)

```bash
cd backend
# essentia should already be installed from Day 2 (B4 partial) but verify:
uv pip list | grep essentia || uv add essentia

# scipy is already a transitive dep of librosa, but make it explicit for spectral ducking:
uv add scipy  # if not already explicit in pyproject.toml
```

Verify essentia works:

```python
python -c "from essentia.standard import KeyExtractor; print('essentia OK')"
```

If essentia install fails on macOS ARM64, the librosa chromagram fallback (already designed into the
interface from Day 2) kicks in automatically. Do not block on essentia -- proceed with fallback.

### System dependencies (verify, should already be installed)

```bash
rubberband --version  # Must return v3.x for R3 engine (-3 flag)
ffmpeg -version       # Required for MP3 export
```

### No new frontend dependencies

Day 4 frontend work is pure React/TypeScript -- error states, warning display, expiration notices.

---

## Implementation Order

Priority sequence: audio quality features first (key matching, ducking) because they affect every
test pairing, then error handling (catches failures during testing), then testing matrix.

```
Morning (audio quality):
  Step 1: Key detection with essentia (B4 completion)          ~45 min
  Step 2: Key distance algorithm + confidence gate (B6)         ~30 min
  Step 3: Key/pitch shifting via rubberband (B6)                ~45 min
  Step 4: Vocal stem pre-filtering before rubberband (B6)       ~20 min
  Step 5: Spectral ducking (QR-1 section 3)                     ~60 min
  Step 6: Wire ducking into pipeline between steps 9 and 11     ~20 min

Afternoon (error handling + robustness):
  Step 7:  Processing lock + fail-fast 429 (B7)                 ~30 min
  Step 8:  Upload validation hardening (B2)                     ~30 min
  Step 9:  Pipeline error handling + user-friendly messages (B7) ~45 min
  Step 10: TTL cleanup + lock watchdog (B7 lifespan)            ~45 min
  Step 11: Intermediate artifact cleanup (B7)                   ~15 min

Evening (frontend polish + testing):
  Step 12: Frontend error states (all HTTP error codes)         ~30 min
  Step 13: Warnings display + used_fallback indicator (F5)      ~20 min
  Step 14: Expiration notice + expired remix handling (F5)      ~15 min
  Step 15: UI copy (tagline + constraint explanation)           ~10 min
  Step 16: End-to-end testing matrix (5+ song pairings)         ~90 min
  Step 17: LLM prompt tuning based on listening results         ~30 min
```

---

## Step 1: Key Detection with Essentia

**Files to modify:**
- `backend/src/musicmixer/services/analysis.py`
- `backend/src/musicmixer/models.py` (if `key_confidence` and `has_modulation` not already on `AudioMetadata`)

**What to build:**

Add key detection to `analyze_audio()`. On Day 2-3, this function returned BPM + beat_frames +
duration only. Now add `key`, `scale`, `key_confidence`, and `has_modulation`.

```python
def detect_key(audio_path: Path) -> tuple[str, str, float]:
    """Returns (key, scale, confidence). Uses essentia if available, else librosa."""
    try:
        from essentia.standard import KeyExtractor, MonoLoader
        audio = MonoLoader(filename=str(audio_path), sampleRate=44100)()
        key_extractor = KeyExtractor()
        key, scale, confidence = key_extractor(audio)
        return key, scale, float(confidence)
    except ImportError:
        logger.warning("essentia not available, falling back to librosa chromagram")
        return _detect_key_librosa(audio_path)
    except Exception as e:
        logger.warning(f"essentia key detection failed: {e}, falling back to librosa")
        return _detect_key_librosa(audio_path)

def _detect_key_librosa(audio_path: Path) -> tuple[str, str, float]:
    """Fallback key detection using librosa chromagram. Lower accuracy than essentia."""
    y, sr = librosa.load(str(audio_path), sr=22050)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    # Sum chroma bins across time
    chroma_sum = chroma.mean(axis=1)
    # Map to key names
    key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    key_idx = int(np.argmax(chroma_sum))
    # Rough major/minor determination using relative minor offset
    major_profile = chroma_sum[key_idx]
    minor_idx = (key_idx + 9) % 12  # relative minor
    minor_profile = chroma_sum[minor_idx]
    scale = "major" if major_profile >= minor_profile else "minor"
    if scale == "minor":
        key_idx = minor_idx
    # Confidence is normalized peak strength (0-1, but typically lower than essentia)
    confidence = float(chroma_sum[key_idx] / (chroma_sum.sum() + 1e-10))
    return key_names[key_idx], scale, confidence
```

**Modulation detection** -- run essentia on first 60% and last 40% separately:

```python
def detect_modulation(audio_path: Path) -> bool:
    """Check if the song modulates (key changes mid-song). ~1s additional processing."""
    try:
        from essentia.standard import KeyExtractor, MonoLoader
        audio = MonoLoader(filename=str(audio_path), sampleRate=44100)()
        split = int(len(audio) * 0.6)
        ke = KeyExtractor()
        key_first, scale_first, _ = ke(audio[:split])
        key_last, scale_last, _ = ke(audio[split:])
        return (key_first, scale_first) != (key_last, scale_last)
    except Exception:
        return False  # If detection fails, assume no modulation
```

Wire into `analyze_audio()`:

```python
key, scale, key_confidence = detect_key(audio_path)
has_modulation = detect_modulation(audio_path)
# Set on AudioMetadata
metadata.key = key
metadata.scale = scale
metadata.key_confidence = key_confidence
metadata.has_modulation = has_modulation
```

**Key decisions:**
- Essentia is primary, librosa is automatic fallback -- no manual switching needed
- Modulation detection costs ~1s but prevents bad pitch shifts on 15-25% of pop songs
- `detect_key` interface is pluggable: `(Path) -> (str, str, float)`

**Gotchas:**
- Essentia's `MonoLoader` loads the entire file at 44.1kHz -- memory-safe for songs up to 10 min
- Librosa fallback has lower accuracy (~50-60% vs essentia's ~75%); confidence values are not
  directly comparable between the two. This is acceptable because the confidence gate (Step 2)
  will skip key matching at low confidence regardless of the detector.

---

## Step 2: Key Distance Algorithm + Confidence Gate

**Files to create:**
- `backend/src/musicmixer/services/key_utils.py`

**Files to modify:**
- `backend/src/musicmixer/services/processor.py`

**What to build:**

Extract key distance computation into a shared utility (used by both the processing pipeline
and the future compatibility analysis endpoint).

```python
# Pitch class mapping (handles enharmonic equivalence)
PITCH_CLASS = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4, 'Fb': 4, 'F': 5, 'E#': 5, 'F#': 6, 'Gb': 6,
    'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10,
    'B': 11, 'Cb': 11,
}

# Circle-of-fifths order for harmonic compatibility
COF_ORDER = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]  # C, G, D, A, E, B, F#, C#, Ab, Eb, Bb, F

def normalize_to_major(key: str, scale: str) -> int:
    """Normalize to relative major pitch class. A minor -> C major (0)."""
    pc = PITCH_CLASS[key]
    if scale == "minor":
        pc = (pc + 3) % 12  # relative major is 3 semitones up
    return pc

def compute_key_shift(source_key: str, source_scale: str,
                      target_key: str, target_scale: str) -> int:
    """Compute minimal semitone shift from source to target key.
    Returns signed integer: positive = shift up, negative = shift down."""
    source_pc = PITCH_CLASS[source_key]
    target_pc = PITCH_CLASS[target_key]
    interval = (target_pc - source_pc) % 12
    shift = interval if interval <= 6 else interval - 12
    return shift

def circle_of_fifths_distance(key_a: str, scale_a: str,
                               key_b: str, scale_b: str) -> int:
    """Compute circle-of-fifths distance between two keys (normalized to major).
    Returns 0-6 (0 = same key, 1 = perfect fifth apart, 6 = tritone)."""
    major_a = normalize_to_major(key_a, scale_a)
    major_b = normalize_to_major(key_b, scale_b)
    pos_a = COF_ORDER.index(major_a)
    pos_b = COF_ORDER.index(major_b)
    dist = abs(pos_a - pos_b)
    return min(dist, 12 - dist)

def compute_effective_shift(source_key: str, source_scale: str,
                            target_key: str, target_scale: str,
                            source_confidence: float,
                            target_confidence: float,
                            is_vocal: bool = False) -> float:
    """Compute the effective pitch shift in semitones, applying confidence gating.

    Two-tier confidence system:
      - min(confidences) >= 0.55: full shift
      - min(confidences) 0.40-0.55: half shift (hedging)
      - min(confidences) < 0.40: skip (return 0)

    Harmonic compatibility gate:
      - CoF distance 0 or 1: skip (already compatible)

    Cap: +/- 4 semitones for vocals, +/- 5 for instruments.
    """
    min_confidence = min(source_confidence, target_confidence)

    # Confidence gate
    if min_confidence < 0.40:
        return 0.0

    # Harmonic compatibility gate
    cof_dist = circle_of_fifths_distance(source_key, source_scale, target_key, target_scale)
    if cof_dist <= 1:
        return 0.0  # Already harmonically compatible

    # Compute raw shift
    raw_shift = compute_key_shift(source_key, source_scale, target_key, target_scale)

    # Apply two-tier confidence scaling
    if min_confidence >= 0.55:
        effective_shift = float(raw_shift)  # Full shift
    else:
        # 0.40-0.55: half shift (hedging)
        effective_shift = float(raw_shift) / 2.0
        # Round to nearest 0.5 semitone for rubberband
        effective_shift = round(effective_shift * 2) / 2

    # Cap based on stem type
    cap = 4.0 if is_vocal else 5.0
    if abs(effective_shift) > cap:
        return 0.0  # Exceeds cap, skip entirely

    return effective_shift
```

**Key decisions:**
- Use `min(source_confidence, target_confidence)` -- the weakest link determines gate
- The 0.55/0.40 thresholds come directly from the plan's analysis of essentia's error rates
- Half-shift rounding to 0.5 semitones keeps rubberband happy
- Return 0.0 (skip) when shift exceeds cap rather than clamping -- a clamped shift to a
  non-target key sounds worse than no shift

**Gotcha from the plan:**
> "essentia reports 0.4-0.6 confidence for wrong keys ~20-30% of the time"
>
> This is why the threshold is 0.55 not 0.50 -- at 0.50, too many wrong detections pass through.
> The plan recommends a calibration experiment on GiantSteps Key dataset before shipping, but
> for demo day, these thresholds are the best available defaults.

---

## Step 3: Key/Pitch Shifting via Rubberband

**Files to modify:**
- `backend/src/musicmixer/services/processor.py`

**What to build:**

Modify the existing `rubberband_process()` function (built on Day 2) to accept a `semitones`
parameter and apply pitch shifting in the same single-pass invocation as tempo stretching.

The Day 2 implementation already has:
```python
def rubberband_process(audio, sr, source_bpm, target_bpm, ...) -> np.ndarray
```

Update the signature and body:

```python
def rubberband_process(audio: np.ndarray, sr: int, source_bpm: float,
                       target_bpm: float, semitones: float = 0,
                       is_vocal: bool = False, tmp_dir: Path = None) -> np.ndarray:
    """Single-pass tempo + pitch via rubberband CLI. Guaranteed single invocation."""

    time_ratio = source_bpm / target_bpm  # CRITICAL: source/target, NOT target/source

    # Skip-at-unity optimization
    if abs(time_ratio - 1.0) < 0.001 and abs(semitones) < 0.01:
        return audio

    in_path = tmp_dir / f"rb_in_{uuid.uuid4().hex[:8]}.wav"
    out_path = tmp_dir / f"rb_out_{uuid.uuid4().hex[:8]}.wav"
    sf.write(str(in_path), audio, sr)

    cmd = ["rubberband", "-t", str(time_ratio)]
    if abs(semitones) >= 0.01:
        cmd += ["-p", str(semitones)]
    cmd += ["-3"]  # R3 engine (v3.x)
    if is_vocal and abs(semitones) >= 0.01:
        cmd += ["--formant"]  # Prevents chipmunk/demonic artifacts on vocals
    cmd += [str(in_path), str(out_path)]

    subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    result, _ = sf.read(str(out_path), dtype='float32')

    # Cleanup temp files
    in_path.unlink(missing_ok=True)
    out_path.unlink(missing_ok=True)

    return result
```

**Wire key matching into the processing pipeline:**

In `process_remix()`, after computing the target BPM (already done in Day 2 tempo matching),
compute the pitch shift:

```python
# Determine key shift (if key_source != "none" and neither song has modulation)
semitones = 0.0
if remix_plan.key_source != "none" and not meta_a.has_modulation and not meta_b.has_modulation:
    vocal_meta = meta_a if remix_plan.vocal_source == "song_a" else meta_b
    inst_meta = meta_b if remix_plan.vocal_source == "song_a" else meta_a

    # Shift vocals toward instrumental key
    semitones = compute_effective_shift(
        source_key=vocal_meta.key, source_scale=vocal_meta.scale,
        target_key=inst_meta.key, target_scale=inst_meta.scale,
        source_confidence=vocal_meta.key_confidence,
        target_confidence=inst_meta.key_confidence,
        is_vocal=True,
    )

    if abs(semitones) >= 0.01:
        logger.info(f"Key matching: shifting vocals {semitones:+.1f} semitones "
                    f"({vocal_meta.key} {vocal_meta.scale} -> {inst_meta.key} {inst_meta.scale}, "
                    f"confidence: {min(vocal_meta.key_confidence, inst_meta.key_confidence):.2f})")
    elif remix_plan.key_source != "none":
        logger.info("Key matching: skipped (harmonically compatible or low confidence)")

# Pass semitones to rubberband_process for vocal stems
# Instrumental stems get semitones=0 (or their own shift if key_source targets them)
```

**Key decisions:**
- Tempo + pitch in a single rubberband call (plan is emphatic: "Two separate passes compound
  artifacts -- 1-2 dB noise floor increase, phase smearing")
- `--formant` flag ONLY when pitch shift is non-zero AND stem is vocal
- Skip-at-unity: when no tempo or key change needed, return audio unmodified (avoids unnecessary
  rubberband processing -- especially important for instrumental stems at target tempo)
- Use unique temp file names (uuid) to enable parallel rubberband invocations

**Gotchas:**
- `time_ratio = source_bpm / target_bpm` -- inverting this produces the OPPOSITE stretch
- The `-3` flag requires rubberband v3.x. If only v2 is installed, fall back to `--crisp 5`.
  Add a startup check:
  ```python
  def _check_rubberband_version() -> int:
      result = subprocess.run(["rubberband", "--version"], capture_output=True, text=True)
      # Parse major version from output
      match = re.search(r'(\d+)\.', result.stdout + result.stderr)
      return int(match.group(1)) if match else 2
  ```

---

## Step 4: Vocal Stem Pre-Filtering

**Files to modify:**
- `backend/src/musicmixer/services/processor.py`

**What to build:**

Apply a bandpass filter (200 Hz - 8 kHz) to the vocal stem BEFORE passing it to rubberband.
This removes separation artifacts that compound through time stretching and pitch shifting.

```python
from scipy.signal import butter, sosfiltfilt

def prefilter_vocal_stem(vocal_audio: np.ndarray, sr: int) -> np.ndarray:
    """Bandpass filter vocal stem to remove separation artifacts before rubberband.

    Removes:
    - Bass/kick bleed below 200 Hz (stretching amplifies it most)
    - Cymbal/hi-hat bleed above 8 kHz (pitch shifting makes it metallic)

    Vocal quality loss is minimal: fundamentals are above 200 Hz,
    harmonics above 8 kHz add "air" but are not essential.
    """
    # Single 2nd-order Butterworth bandpass: 200 Hz - 8 kHz
    # Uses a single sosfiltfilt call (one forward+backward pass) instead of two
    # separate filters cascaded (which would double the effective order at crossover).
    sos = butter(2, [200, 8000], btype='band', fs=sr, output='sos')
    return sosfiltfilt(sos, vocal_audio, axis=0)
```

**Where it goes in the pipeline:**

```
... (after cross-song level matching measures LUFS on UNFILTERED vocal) ...
vocal_audio = prefilter_vocal_stem(vocal_audio, sr)
... (then pass to rubberband_process with tempo + pitch) ...
```

**Critical ordering:** Measure LUFS on the unfiltered vocal FIRST (Step 4 of the 13-step pipeline
from Day 2). Pre-filtering removes low-frequency energy (chest resonance below 200 Hz) which
reduces measured LUFS by 1-3 dB. Measuring after filtering would cause over-boosting vocals.

**Key decision:** 2nd-order Butterworth is gentle enough to avoid audible artifacts on the vocal
itself. The 200 Hz cutoff is safe -- vocal fundamentals are above 200 Hz for all but the deepest
bass singers.

---

## Step 5: Spectral Ducking

**Files to create:**
- `backend/src/musicmixer/services/ducking.py`

**What to build:**

This is the QR-1 spectral ducking implementation. It reduces instrumental energy in the 300Hz-3kHz
range when vocals are active, carving a frequency pocket so vocals sit *inside* the instrumental
rather than on top.

```python
"""Spectral ducking: carve a mid-range pocket in the instrumental for vocals."""

import math
import numpy as np
from scipy.signal import butter, sosfiltfilt, lfilter
import logging

logger = logging.getLogger(__name__)


def spectral_duck(instrumental: np.ndarray, vocal: np.ndarray, sr: int,
                  cut_db: float = -3.5, lo: float = 300, hi: float = 3000) -> np.ndarray:
    """Duck instrumental mid-range energy where vocals are active.

    CRITICAL: Uses sosfiltfilt (zero-phase filtering), NOT sosfilt. Causal sosfilt
    introduces frequency-dependent phase delay. Subtracting a phase-shifted mid-band
    from the original creates comb-filter artifacts (metallic/hollow coloration).
    sosfiltfilt applies the filter forward+backward, eliminating phase shift.

    Args:
        instrumental: Stereo instrumental audio (N, 2). Pipeline step 1 standardizes to stereo.
        vocal: Stereo vocal audio (N, 2).
        sr: Sample rate (44100).
        cut_db: Gain reduction in dB when vocals are active. Default -3.5 dB is conservative.
        lo: Low cutoff frequency for mid-band extraction (Hz).
        hi: High cutoff frequency for mid-band extraction (Hz).

    Returns:
        Ducked instrumental audio (same shape as input).
    """
    assert instrumental.ndim == 2, "Expected stereo (pipeline step 1 standardizes to stereo)"

    # Ensure same length (truncate to shorter)
    min_len = min(len(instrumental), len(vocal))
    instrumental = instrumental[:min_len]
    vocal = vocal[:min_len]

    # 1. Detect vocal activity via RMS envelope (50ms frames)
    vocal_mono = vocal.mean(axis=1) if vocal.ndim == 2 else vocal
    frame_len = int(0.05 * sr)  # 50ms = 2205 samples at 44.1kHz

    # Bandpass vocal to 300-3500 Hz for energy detection
    # (aligned with the ducking range + small margin for harmonic masking)
    # 4th order Butterworth for consistent skirt slopes with mid-band extraction
    sos_bp = butter(4, [300, 3500], btype='band', fs=sr, output='sos')
    vocal_filtered = sosfiltfilt(sos_bp, vocal_mono)

    # Compute per-frame RMS energy
    n_frames = (len(vocal_filtered) - frame_len) // frame_len + 1
    vocal_energy = np.array([
        np.sqrt(np.mean(vocal_filtered[i * frame_len:(i + 1) * frame_len] ** 2))
        for i in range(n_frames)
    ])

    # Noise-floor-relative threshold with hysteresis
    # (replaces fragile 40th percentile approach that fails with stem separation bleed)
    noise_floor = np.percentile(vocal_energy, 10)
    onset_threshold = noise_floor * 4.0
    offset_threshold = noise_floor * 2.0

    # Hysteresis state machine: active when > onset, stays active until < offset
    vocal_active = np.zeros(n_frames, dtype=np.float64)
    active = False
    for i in range(n_frames):
        if active:
            active = vocal_energy[i] >= offset_threshold
        else:
            active = vocal_energy[i] > onset_threshold
        vocal_active[i] = float(active)

    # 2. Upsample mask to sample rate + smooth with asymmetric attack/release
    mask = np.repeat(vocal_active, frame_len)[:min_len]

    # Exponential IIR smoothing: 30ms attack, 150ms release
    attack_alpha = 1 - math.exp(-1.0 / (0.03 * sr))   # ~30ms attack
    release_alpha = 1 - math.exp(-1.0 / (0.15 * sr))  # ~150ms release

    # Two-pass lfilter approximation (runs in <0.05s vs 5-15s for naive Python loop)
    attack_smoothed = lfilter([attack_alpha], [1, -(1 - attack_alpha)], mask)
    release_smoothed = lfilter([release_alpha], [1, -(1 - release_alpha)], mask[::-1])[::-1]
    mask = np.maximum(attack_smoothed, release_smoothed)

    # 3. Extract mid-band from instrumental (ZERO-PHASE -- critical)
    sos = butter(4, [lo, hi], btype='band', fs=sr, output='sos')
    mid_band = sosfiltfilt(sos, instrumental, axis=0)

    # 4. Reduce mid-band energy where vocals are active (stereo-safe)
    gain = 10 ** (cut_db / 20)  # -3.5 dB -> ~0.668
    reduction = mid_band * (1 - gain) * mask[:, np.newaxis]
    return instrumental - reduction
```

**Key parameters (from the plan):**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `cut_db` | -3.5 dB | Conservative -- creates space without making instrumental hollow |
| `lo` / `hi` | 300 Hz / 3 kHz | Vocal fundamental + formant range |
| Vocal detection band | 300-3500 Hz | Aligned with ducking range + harmonic margin |
| Attack time | 30ms | Fast enough to duck when vocals start |
| Release time | 150ms | Slow enough to avoid pumping between words |
| Noise floor | 10th percentile | Below bleed floor, above absolute silence |
| Onset threshold | 4x noise floor | Confident vocal onset detection |
| Offset threshold | 2x noise floor | Hysteresis prevents rapid toggling |

**Gotchas from the plan:**
- MUST use `sosfiltfilt` (zero-phase), NOT `sosfilt`. Causal filtering creates phase-shifted
  mid-band; subtracting it produces comb-filter artifacts (metallic/hollow coloration).
- The two-pass `lfilter` + `max()` smoothing is an approximation of true asymmetric attack/release.
  It may produce a plateau artifact near transitions. For MVP this is acceptable and runs in
  <0.05s vs 5-15s for the naive Python loop.
- Spectral ducking timeout is 60 seconds (per-step timeout from B7). Add timeout tracking.

---

## Step 6: Wire Ducking into Pipeline

**Files to modify:**
- `backend/src/musicmixer/services/processor.py` (or `pipeline.py` depending on Day 2-3 structure)

**What to build:**

Spectral ducking goes AFTER arrangement rendering (step 9 in the 13-step pipeline) and BEFORE
the final sum (step 11). The arrangement renderer outputs two buses:

```
Step 9:  Render arrangement -> vocal_bus, instrumental_bus
Step 10: spectral_duck(instrumental_bus, vocal_bus, sr) -> ducked_instrumental
Step 11: final_mix = vocal_bus + ducked_instrumental
```

This ordering is critical -- ducking operates on the rendered instrumental (with per-section
gains already applied), and the vocal activity mask reflects the actual arrangement (sections
where `stem_gains["vocals"] == 0.0` will NOT trigger ducking).

```python
# In process_remix(), after arrangement rendering:
vocal_bus, instrumental_bus = render_arrangement(...)

# Spectral ducking
logger.info("Applying spectral ducking...")
if progress_callback:
    progress_callback({"step": "rendering", "detail": "Applying spectral ducking...", "progress": 0.88})

ducked_instrumental = spectral_duck(instrumental_bus, vocal_bus, sr)

# Sum buses
final_mix = vocal_bus + ducked_instrumental
```

---

## Step 7: Processing Lock + Fail-Fast 429

**Files to modify:**
- `backend/src/musicmixer/api/remix.py`
- `backend/src/musicmixer/main.py` (if lock is defined at module level)

**What to build:**

The processing lock was likely partially implemented on Day 2 (single-worker executor). Now
make it robust with the plan's two-stage lock pattern:

```python
import threading

processing_lock = threading.Lock()

# In POST /api/remix handler:

# 0. Disk space check
usage = shutil.disk_usage(settings.data_dir)
if usage.free < 1_000_000_000:  # 1 GB minimum free
    raise HTTPException(507, "Server storage is full. Try again later.")

# 1. Fail-fast check (non-mutating -- never acquires the lock)
if processing_lock.locked():
    raise HTTPException(429, "Another remix is being created. Please wait and try again.")

# 2. Accept upload, validate files, save to disk
# ... (existing upload code) ...

# 3. Acquire lock once (authoritative gate)
if not processing_lock.acquire(blocking=False):
    raise HTTPException(429, "Server is busy processing another remix. Please wait and try again.")

# 4. Submit to executor via wrapper that guarantees lock release
def pipeline_wrapper():
    try:
        _pipeline_start_mono = time.monotonic()
        session.status = "processing"
        run_pipeline(session_id, song_a_path, song_b_path, prompt, session.events)
    except BaseException as e:
        try:
            session.status = "error"
            detail = str(e) if isinstance(e, MusicMixerError) else \
                "Something went wrong while creating your remix. Please try again."
            if not isinstance(e, MusicMixerError):
                logger.exception("Unhandled pipeline error", extra={"session_id": session_id})
            emit_progress(session.events,
                         {"step": "error", "detail": detail, "progress": 0})
        except Exception:
            pass  # Best effort
        raise
    finally:
        try:
            processing_lock.release()
        except RuntimeError:
            pass  # Already released (e.g., by watchdog)

try:
    future = executor.submit(pipeline_wrapper)
except Exception:
    session.status = "error"
    emit_progress(session.events,
                 {"step": "error", "detail": "Server error -- please try again", "progress": 0})
    processing_lock.release()
    raise
```

**Key decision:** The two-stage pattern (read-only `.locked()` check + single `.acquire()` gate)
avoids blocking on the upload when the server is busy. The `.locked()` check is a TOCTOU race
but never acquires the lock, so it cannot leak.

---

## Step 8: Upload Validation Hardening

**Files to modify:**
- `backend/src/musicmixer/services/upload.py` (or wherever upload validation lives from Day 1)
- `backend/src/musicmixer/api/remix.py`

**What to build:**

Day 1 had extension-only validation. Add the real validation:

1. **Audio parse check** -- the actual test that the file is valid audio:
   ```python
   from pydub import AudioSegment

   def validate_audio_file(file_path: Path) -> float:
       """Parse audio file and return duration in seconds. Raises ValidationError if invalid."""
       try:
           audio = AudioSegment.from_file(str(file_path))
       except Exception as e:
           raise ValidationError(f"Could not read audio file: {file_path.name}. "
                                "Please upload a valid MP3 or WAV file.") from e

       duration_sec = len(audio) / 1000.0

       if duration_sec > settings.max_duration_seconds:
           raise ValidationError(
               f"Song is too long ({duration_sec:.0f}s). Maximum duration is "
               f"{settings.max_duration_seconds // 60} minutes."
           )

       if duration_sec < 10:
           raise ValidationError(
               f"Song is too short ({duration_sec:.0f}s). Minimum duration is 10 seconds."
           )

       return duration_sec
   ```

2. **Prompt validation** (5-1000 characters):
   ```python
   if len(prompt.strip()) < 5:
       raise HTTPException(422, "Prompt must be at least 5 characters.")
   if len(prompt) > 1000:
       raise HTTPException(422, "Prompt must be 1000 characters or fewer.")
   ```

3. **File size check** (already partially done on Day 1, verify it's streaming-safe):
   ```python
   MAX_FILE_SIZE = settings.max_file_size_mb * 1024 * 1024  # 50MB

   # During streaming upload read:
   total_read = 0
   while chunk := await file.read(8192):
       total_read += len(chunk)
       if total_read > MAX_FILE_SIZE:
           raise HTTPException(413, f"File too large. Maximum size is {settings.max_file_size_mb}MB.")
       out_file.write(chunk)
   ```

**Error responses:**
- 413: File too large
- 422: Invalid audio file, duration too long/short, prompt too short/long
- 429: Server busy (from Step 7)
- 507: Disk full (from Step 7)

---

## Step 9: Pipeline Error Handling + User-Friendly Messages

**Files to modify:**
- `backend/src/musicmixer/services/pipeline.py`
- `backend/src/musicmixer/exceptions.py`

**What to build:**

Ensure every pipeline step has try/except that maps to user-friendly SSE error messages.
The plan requires that `MusicMixerError` subtypes pass through their messages; all other
exceptions get a generic message (no file paths, stack traces, or API keys leaked).

```python
def run_pipeline(session_id: str, song_a_path: Path, song_b_path: Path,
                 prompt: str, event_queue: queue.Queue) -> None:
    """Main pipeline orchestrator. Called in background thread."""

    try:
        # Check cooperative shutdown between every major step
        def check_shutdown():
            if _shutdown_requested.is_set():
                raise PipelineError("Server shutting down -- please try again shortly.")

        # Step 1: Stem separation
        emit_progress(event_queue, {"step": "separating",
                      "detail": "Extracting stems from both songs...", "progress": 0.10})
        check_shutdown()
        try:
            stems_a, stems_b = separate_both_songs(song_a_path, song_b_path)
        except SeparationError:
            raise  # Already has user-friendly message
        except Exception as e:
            raise SeparationError("Failed to separate stems. The audio file may be corrupted.") from e

        # Step 2: Analysis
        emit_progress(event_queue, {"step": "analyzing",
                      "detail": "Detecting tempo and key...", "progress": 0.50})
        check_shutdown()
        try:
            meta_a = analyze_audio(song_a_path)
            meta_b = analyze_audio(song_b_path)
            meta_a, meta_b = reconcile_bpm(meta_a, meta_b)
        except AnalysisError:
            raise
        except Exception as e:
            raise AnalysisError("Failed to analyze audio properties.") from e

        # Step 3: LLM interpretation
        emit_progress(event_queue, {"step": "interpreting",
                      "detail": "Planning your remix...", "progress": 0.58})
        check_shutdown()
        try:
            # Pre-compute key matching decision for LLM context
            key_info = _compute_key_guidance(meta_a, meta_b)
            remix_plan = interpret_prompt(prompt, meta_a, meta_b, key_info)
        except Exception as e:
            logger.warning(f"LLM interpretation failed: {e}, using deterministic fallback")
            remix_plan = generate_fallback_plan(meta_a, meta_b)

        # Step 4: Audio processing
        emit_progress(event_queue, {"step": "processing",
                      "detail": "Matching tempo and key...", "progress": 0.65})
        check_shutdown()
        try:
            result_path = process_remix(remix_plan, stems_a, stems_b,
                                       meta_a, meta_b, output_path,
                                       progress_callback=lambda e: emit_progress(event_queue, e))
        except ProcessingError:
            raise
        except Exception as e:
            raise ProcessingError("Failed to process the remix. Please try again.") from e

        # Success
        session.status = "complete"
        session.remix_path = str(result_path)
        session.explanation = remix_plan.explanation
        emit_progress(event_queue, {
            "step": "complete",
            "detail": "Remix ready!",
            "progress": 1.0,
            "explanation": remix_plan.explanation,
            "warnings": remix_plan.warnings or [],
            "usedFallback": remix_plan.used_fallback,
        })

    except MusicMixerError as e:
        session.status = "error"
        session.last_event = {"step": "error", "detail": str(e), "progress": 0}
        emit_progress(event_queue, {"step": "error", "detail": str(e), "progress": 0})
        logger.error(f"Pipeline error: {e}", extra={"session_id": session_id})

    except Exception as e:
        session.status = "error"
        detail = "Something went wrong while creating your remix. Please try again."
        session.last_event = {"step": "error", "detail": detail, "progress": 0}
        emit_progress(event_queue, {"step": "error", "detail": detail, "progress": 0})
        logger.exception("Unhandled pipeline error", extra={"session_id": session_id})
```

**Key guidance pre-computation for LLM context:**

```python
def _compute_key_guidance(meta_a: AudioMetadata, meta_b: AudioMetadata) -> str:
    """Pre-compute key matching decision to pass as LLM context."""
    min_conf = min(meta_a.key_confidence, meta_b.key_confidence)
    if meta_a.has_modulation or meta_b.has_modulation:
        return "Key matching: unavailable (one or both songs change key mid-song)"
    if min_conf < 0.40:
        return f"Key matching: unavailable (low confidence: {min_conf:.2f})"
    if min_conf < 0.55:
        return f"Key matching: partial (moderate confidence: {min_conf:.2f}, will apply half shift)"
    return f"Key matching: available (confidence: {min_conf:.2f})"
```

This ensures the LLM's explanation always matches what the processing code actually does.

---

## Step 10: TTL Cleanup + Lock Watchdog

**Files to modify:**
- `backend/src/musicmixer/main.py`

**What to build:**

Implement the FastAPI lifespan with cleanup loop and lock watchdog.

```python
import asyncio
import shutil
import time
import threading
from contextlib import asynccontextmanager

# Module-level state
_pipeline_start_mono: float | None = None
_shutdown_requested = threading.Event()

@asynccontextmanager
async def lifespan(app: FastAPI):
    cleanup_task = asyncio.create_task(cleanup_loop())
    watchdog_task = asyncio.create_task(lock_watchdog())
    _shutdown_requested.clear()
    yield
    _shutdown_requested.set()
    cleanup_task.cancel()
    watchdog_task.cancel()
    executor.shutdown(wait=True, cancel_futures=True)


async def cleanup_loop():
    """Periodic cleanup of expired sessions and their files."""
    while True:
        await asyncio.sleep(settings.cleanup_interval_seconds)  # 5 minutes
        try:
            now = time.monotonic()
            with sessions_lock:
                expired = [
                    (sid, s) for sid, s in sessions.items()
                    if s.status not in ("processing", "queued")
                    and (now - s.created_at_mono) > (
                        settings.error_ttl_seconds if s.status == "error"  # 15 min
                        else settings.remix_ttl_seconds + 300  # 3h + 5 min grace
                    )
                ]
                for sid, _ in expired:
                    del sessions[sid]

            # Delete files OUTSIDE the lock, in a thread to avoid blocking event loop
            for sid, _ in expired:
                for subdir in ("uploads", "stems", "remixes"):
                    session_dir = settings.data_dir / subdir / sid
                    await asyncio.to_thread(shutil.rmtree, session_dir, ignore_errors=True)

            if expired:
                logger.info(f"Cleanup: removed {len(expired)} sessions, "
                           f"{len(sessions)} active, lock_held={processing_lock.locked()}")
        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("Cleanup cycle failed, will retry next interval")


async def lock_watchdog():
    """Safety net: force-release lock if held longer than 21 minutes.

    Catches edge cases that finally{} cannot cover: native segfault in a C extension
    (numpy, scipy, rubberband), os._exit() in a dependency, or OOM kill.
    """
    while True:
        await asyncio.sleep(60)
        try:
            if processing_lock.locked() and _pipeline_start_mono is not None:
                elapsed = time.monotonic() - _pipeline_start_mono
                if elapsed > settings.max_sse_duration_seconds + 60:  # 21 minutes
                    logger.error(f"Watchdog: processing lock held for {int(elapsed)}s -- forcing release")
                    try:
                        processing_lock.release()
                    except RuntimeError:
                        pass
        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("Watchdog cycle failed")
```

**TTL values (from plan's Settings):**
- `remix_ttl_seconds`: 10800 (3 hours) -- completed sessions
- `error_ttl_seconds`: 900 (15 minutes) -- error sessions (no user value, same disk cost)
- `cleanup_interval_seconds`: 300 (5 minutes) -- how often the loop runs
- `max_sse_duration_seconds`: 1200 (20 minutes) -- SSE safety cap

**Gotcha:** File deletion via `shutil.rmtree` runs in `asyncio.to_thread` to avoid blocking
the event loop. Each session directory is ~500MB -- `rmtree` can take 100s of ms.

---

## Step 11: Intermediate Artifact Cleanup

**Files to modify:**
- `backend/src/musicmixer/services/pipeline.py`

**What to build:**

After a successful pipeline completion, delete uploads and stems (keeping only the final remix MP3).
This reduces per-session disk from ~500MB to ~10MB.

```python
# In run_pipeline(), after successful completion (before the except blocks):

# Cleanup intermediate artifacts (uploads + stems, keep only remix MP3)
for subdir in ("uploads", "stems"):
    session_dir = settings.data_dir / subdir / session_id
    shutil.rmtree(session_dir, ignore_errors=True)
logger.info(f"Cleaned up intermediate artifacts for session {session_id}")
```

Place this in a `finally`-like structure or right after the `session.status = "complete"` line.
Use `ignore_errors=True` so cleanup failure doesn't crash the pipeline after a successful remix.

---

## Step 12: Frontend Error States

**Files to modify:**
- `frontend/src/components/RemixSession.tsx` (or wherever the reducer lives)
- `frontend/src/components/ProgressDisplay.tsx`
- `frontend/src/components/RemixForm.tsx`

**What to build:**

Map every HTTP error code to a user-friendly message. The reducer already has an `error` phase
from Day 3; now make the error messages specific.

**Error taxonomy (from the plan):**

| Error Source | HTTP Status | User Sees |
|-------------|-------------|-----------|
| Network failure | No response | "Upload failed. Check your connection and try again." |
| File too large | 413 | "File too large. Maximum 50MB per song." |
| Invalid audio file | 422 | Server's error message (e.g., "Could not read audio file") |
| Duration too long | 422 | Server's error message (e.g., "Song is too long") |
| Prompt too short/long | 422 | Server's error message |
| Server busy | 429 | "Another remix is being created. Please wait and try again." |
| Disk full | 507 | "Server storage is full. Try again later." |
| Server error | 500 | "Something went wrong. Please try again." |
| SSE error event | N/A | Display the `detail` from the error event |
| SSE timeout | N/A | "Processing is taking longer than expected. Please try again." |

**In the API client** (`api/client.ts`), the XHR error handler should parse these:

```typescript
// In createRemix() XHR handler:
xhr.onload = () => {
  if (xhr.status >= 400) {
    let detail = "Something went wrong. Please try again.";
    try {
      const body = JSON.parse(xhr.responseText);
      if (body.detail) detail = body.detail;
    } catch {}

    // Override generic messages for known status codes
    if (xhr.status === 413) detail = "File too large. Maximum 50MB per song.";
    if (xhr.status === 429) detail = "Another remix is being created. Please wait and try again.";
    if (xhr.status === 507) detail = "Server storage is full. Try again later.";

    reject({ type: 'http', status: xhr.status, body: { detail } });
  }
  // ... success handling
};

xhr.onerror = () => reject({ type: 'network' });
xhr.ontimeout = () => reject({ type: 'timeout' });
```

**In the error phase rendering** (ProgressDisplay or a dedicated ErrorDisplay):

```tsx
function ErrorDisplay({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div className="rounded-lg border border-red-200 bg-red-50 p-6 text-center">
      <p className="text-red-800 mb-4">{message}</p>
      <button
        onClick={onRetry}
        className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
      >
        Try Again
      </button>
    </div>
  );
}
```

The "Try Again" button dispatches `RESET` to return to idle with previous songs/prompt preserved
where possible (from the plan: `error` state carries `songA`, `songB`, `prompt`).

---

## Step 13: Warnings Display + used_fallback Indicator

**Files to modify:**
- `frontend/src/components/RemixPlayer.tsx`
- `frontend/src/types/index.ts` (if `warnings` and `usedFallback` not already in types)

**What to build:**

When the remix is complete, display any `warnings` from the LLM as amber info boxes below
the explanation. When `usedFallback` is true, render the explanation with a "generated
automatically" indicator.

```tsx
// In RemixPlayer.tsx:

{state.usedFallback && (
  <div className="rounded-lg border border-amber-200 bg-amber-50 p-3 mb-3">
    <p className="text-amber-800 text-sm">
      This remix was generated automatically because the AI couldn't interpret your prompt.
      Try rephrasing for a custom result.
    </p>
  </div>
)}

{state.warnings && state.warnings.length > 0 && (
  <div className="space-y-2 mb-3">
    {state.warnings.map((warning, i) => (
      <div key={i} className="rounded-lg border border-amber-200 bg-amber-50 p-3">
        <p className="text-amber-700 text-sm">{warning}</p>
      </div>
    ))}
  </div>
)}

<div className={`rounded-lg p-4 ${state.usedFallback ? 'bg-gray-50 border border-gray-200' : 'bg-green-50 border border-green-200'}`}>
  <p className="text-gray-700">{state.explanation}</p>
</div>
```

**Key decisions:**
- Warnings use amber (not red) -- they're informational, not errors
- Fallback indicator is honest: "generated automatically... try rephrasing"
- Explanation styling differs: green border for LLM-generated, gray for fallback

---

## Step 14: Expiration Notice + Expired Remix Handling

**Files to modify:**
- `frontend/src/components/RemixPlayer.tsx`

**What to build:**

```tsx
// Expiration notice
<p className="text-gray-500 text-sm mt-2">
  This remix will expire in approximately 3 hours.
</p>

// Handle expired remix: if <audio> fires error event or status returns 404
<audio
  src={getAudioUrl(state.sessionId)}
  controls
  onError={(e) => {
    // Audio file may have expired
    dispatch({ type: 'ERROR', message: 'This remix has expired. Create a new one to listen again.' });
  }}
/>
```

Also handle the case where the user returns to a stale session (from sessionStorage, if
implemented, or from a bookmarked URL): the status endpoint returns 404, which maps to
a reset to idle with a friendly message.

---

## Step 15: UI Copy

**Files to modify:**
- `frontend/src/components/RemixSession.tsx` (or the main layout component)
- `frontend/src/components/PromptInput.tsx` or `RemixForm.tsx`

**What to add:**

1. **Tagline** (above the form):
   > "Upload two songs. Describe your mashup. AI does the rest."

2. **Constraint explanation** (near the prompt input):
   > "musicMixer takes the vocals from one song and layers them over the other song's beat."

These set expectations and eliminate the "two songs playing at once" perception gap at zero cost.

---

## Step 16: End-to-End Testing Matrix

### Test Pairings

Test with 5+ real song pairings covering different genres, tempos, and keys. For each pairing,
run the full flow: upload -> prompt -> progress -> playback. Listen to the output critically.

| # | Song A (Vocals) | Song B (Instrumentals) | BPM Gap | Key Gap | What to Listen For |
|---|-----------------|------------------------|---------|---------|-------------------|
| 1 | **Hip-hop** (~90 BPM, e.g., Nas "N.Y. State of Mind") | **Pop** (~120 BPM, e.g., Dua Lipa "Levitating") | ~33% | Likely different | Vocals-only stretch (30-45% tier). Vocal intelligibility after stretch. Beat alignment. Does the vocal sit in the instrumental pocket (ducking working)? |
| 2 | **R&B** (~70 BPM, e.g., SZA "Kill Bill") | **EDM** (~128 BPM, e.g., Disclosure "Latch") | ~83% (but 70 doubled = 140, closer to 128 = ~9%) | Variable | BPM reconciliation (70 -> 140 via doubling). After reconciliation, the stretch should be small. Key matching if confident. |
| 3 | **Pop** (~120 BPM, e.g., The Weeknd "Blinding Lights") | **Pop** (~118 BPM, e.g., Doja Cat "Say So") | ~2% | Likely close | Near-unity tempo (should skip rubberband for instrumental). Focus on: key matching accuracy, ducking clarity, arrangement quality. This is the "golden path" -- should sound great. |
| 4 | **Rock** (~130 BPM, e.g., Arctic Monkeys "Do I Wanna Know") | **Electronic** (~140 BPM, e.g., Tame Impala "The Less I Know the Better") | ~8% | Variable | Moderate stretch. Rock vocals over electronic beat. Ghost vocal handling (rock songs often have heavy reverb that bleeds). Ducking effectiveness with dense instrumentals. |
| 5 | **Rap** (~140 BPM, e.g., Eminem "Lose Yourself") | **Lo-fi/Chill** (~85 BPM, e.g., Nujabes "Feather") | ~65% (but 140 halved = 70, closer to 85 = ~21%) | Variable | BPM reconciliation (140 -> 70 via halving). After reconciliation, ~17-21% stretch. Half-time rap over chill beat. Arrangement should use sparse sections. |

### Additional Edge Cases to Test

| Scenario | What to Test |
|----------|-------------|
| **Same song twice** | Should produce a listenable (if odd) remix. No crash. |
| **Very short song** (~30s) + normal song | Minimum duration handling. Section arrangement should compress gracefully. |
| **Instrumental-only song** (no vocals) | `vocal_prominence_db` should be very negative. LLM or fallback should handle gracefully. Warnings should surface. |
| **Server busy (429)** | Start a remix, then immediately try another upload. Should get "Another remix is being created" error. |
| **Large files** (~45MB each) | Upload progress should display. No timeout during upload. |

### What to Listen For (Across All Pairings)

- [ ] **Beat alignment**: Do vocals land on downbeats? Is there audible flamming (50ms+ offset)?
- [ ] **Key matching**: Does the pitch shift sound natural or artificial? Are there chipmunk artifacts?
- [ ] **Spectral ducking**: Can you hear the instrumental "breathe" when vocals enter? Is it pumping (too aggressive) or not noticeable (too subtle)?
- [ ] **Arrangement**: Does the remix have an energy arc (intro -> build -> peak -> resolve)?
  Are there sections with dramatic gain changes (drums at 0 then 0.8+)?
- [ ] **Transitions**: Are crossfades smooth? Are cuts click-free (micro-crossfade working)?
- [ ] **Overall loudness**: Is the mix at a reasonable volume? Not too quiet, not clipping?
- [ ] **Vocal clarity**: Can you understand the lyrics over the instrumental?
- [ ] **Ghost vocals**: Do you hear the instrumental source song's original vocals bleeding through?

---

## Step 17: LLM Prompt Tuning

**Files to modify:**
- `backend/src/musicmixer/services/interpreter.py` (system prompt)

**What to do:**

After testing all 5+ pairings, review the LLM's output quality and adjust the system prompt:

1. **If arrangement is too flat** (all stems at 0.5-0.8): Reinforce the mixing philosophy in the
   system prompt -- "Use the full 0.0-1.0 range. Muted stems (0.0) are a tool, not a failure."
   Add or strengthen the few-shot example that demonstrates dramatic gain dynamics.

2. **If sections are too long** (monotonous 32-beat sections): Remind the LLM that "Sections
   should be 4, 8, or 16 beats long -- shorter sections create more movement."

3. **If key matching decisions are wrong**: Check whether the key guidance string
   (`_compute_key_guidance`) accurately reflects what the processing code does. Adjust
   thresholds if needed.

4. **If explanation quality is poor**: Review the 3 few-shot examples. Ensure at least one
   demonstrates a good explanation for a challenging pairing.

5. **If warnings are missing on edge cases**: Ensure the system prompt's ambiguity handling
   section covers the cases you encountered.

---

## Exit Criteria

All of the following must be true:

- [ ] All 5 test pairings produce listenable remixes (vocals are intelligible, beat is aligned,
  arrangement has an energy arc, no hard clipping)
- [ ] Key matching works correctly when confidence is high (>= 0.55) -- listen for improved
  harmonic fit compared to no key matching
- [ ] Key matching is skipped when confidence is low (< 0.40) -- no bad pitch shifts
- [ ] Spectral ducking creates audible vocal clarity improvement on at least 3/5 pairings
- [ ] 429 error is returned when server is busy (test with concurrent requests)
- [ ] Invalid audio file upload returns 422 with specific error message
- [ ] TTL cleanup loop runs (check logs after 5+ minutes)
- [ ] Frontend displays error messages for all error types (network, 413, 422, 429, 500, 507)
- [ ] Frontend displays warnings in amber info boxes when present
- [ ] Frontend displays fallback indicator when `usedFallback` is true
- [ ] Expiration notice shows on completed remixes
- [ ] No crashes during 3x consecutive full-flow smoke tests
- [ ] Intermediate artifacts (uploads + stems) are cleaned up after successful remix

---

## Risk Items

### 1. Key Detection Accuracy (Medium Risk)

**Risk:** Essentia's key detection is accurate ~75% of the time. For the remaining 25%, the
two-tier confidence gate mitigates damage (half-shift at 0.40-0.55, skip below 0.40). But some
wrong detections happen at confidence 0.55-0.70, where the system applies a full shift.

**Mitigation:** The plan recommends a GiantSteps Key dataset calibration experiment, but this is
not feasible on Day 4. Accept the current thresholds (0.55/0.40) and note that ~5-10% of pairings
may get a wrong full shift. Listen for this during testing. If a test pairing sounds worse with
key matching, lower the full-shift threshold to 0.60 for safety.

**Fallback:** If essentia is completely unavailable, librosa chromagram has lower accuracy (~50-60%)
but the confidence gate will catch most bad detections (librosa confidence values tend to be lower).

### 2. Spectral Ducking Tuning (Medium Risk)

**Risk:** The -3.5 dB cut default may be too subtle for dense instrumentals or too aggressive for
sparse ones. The 300-3000 Hz band may not capture all vocal energy for very high or very low voices.

**Mitigation:** Listen critically during testing. If ducking is:
- **Too subtle** (vocals still buried): Increase `cut_db` to -5.0 or -6.0 dB
- **Too aggressive** (instrumental sounds hollow): Decrease to -2.0 dB
- **Pumping** (audible volume fluctuation between words): Increase release time from 150ms to 250ms
- **Missing high vocal range**: Widen band to 200-4000 Hz

These are single-parameter changes in `spectral_duck()`. Budget 15-20 min of tuning time.

### 3. Rubberband v3 Availability (Low Risk)

**Risk:** If only rubberband v2 is installed, the `-3` flag will fail.

**Mitigation:** Add version detection at startup (Step 3). Fall back to `--crisp 5` for v2.x.
Quality difference is minor for stretches under 20%.

### 4. Cross-Song Level Matching (Low Risk)

**Risk:** The fixed +3 dB vocal offset (from Day 2) may produce poor vocal-instrumental balance
on some pairings. The plan specifies a spectral-density-adaptive offset, but the Day 4 schedule
explicitly defers this ("Fixed +3 dB offset is good enough for demo").

**Mitigation:** If a test pairing has obviously buried or overpowering vocals, manually adjust
the offset for that genre combination. The full adaptive system is post-demo work.

### 5. Processing Time (Low Risk)

**Risk:** Adding key detection (~1s), modulation detection (~1s), and spectral ducking
(~0.05-0.5s) increases total pipeline time by ~2-3 seconds. This is negligible relative to
the 2-5 minute total pipeline time.

**Mitigation:** None needed. The spectral ducking timeout (60s from B7) is a safety net.

---

## File Summary

### Files to Create

| File | Step | Purpose |
|------|------|---------|
| `backend/src/musicmixer/services/key_utils.py` | 2 | Key distance algorithm, confidence gate, circle-of-fifths |
| `backend/src/musicmixer/services/ducking.py` | 5 | Spectral ducking implementation |

### Files to Modify

| File | Steps | Changes |
|------|-------|---------|
| `backend/src/musicmixer/services/analysis.py` | 1 | Add `detect_key()`, `detect_modulation()`, wire into `analyze_audio()` |
| `backend/src/musicmixer/models.py` | 1 | Add `key`, `scale`, `key_confidence`, `has_modulation` to AudioMetadata (if not present) |
| `backend/src/musicmixer/services/processor.py` | 3, 4, 6 | Key shift in rubberband, vocal pre-filtering, ducking wiring |
| `backend/src/musicmixer/api/remix.py` | 7, 8 | Processing lock, disk check, upload validation, prompt validation |
| `backend/src/musicmixer/services/upload.py` | 8 | Audio parse check, duration validation |
| `backend/src/musicmixer/services/pipeline.py` | 9, 11 | Error handling taxonomy, cooperative shutdown, intermediate cleanup |
| `backend/src/musicmixer/exceptions.py` | 9 | Verify exception hierarchy (should exist from Day 1) |
| `backend/src/musicmixer/main.py` | 10 | TTL cleanup loop, lock watchdog, lifespan |
| `frontend/src/api/client.ts` | 12 | HTTP error parsing with specific messages |
| `frontend/src/components/RemixSession.tsx` | 12, 15 | Error state rendering, UI copy |
| `frontend/src/components/RemixForm.tsx` | 15 | Constraint explanation copy |
| `frontend/src/components/PromptInput.tsx` | 15 | Tagline |
| `frontend/src/components/ProgressDisplay.tsx` | 12 | Error display with retry |
| `frontend/src/components/RemixPlayer.tsx` | 13, 14 | Warnings, fallback indicator, expiration notice, expired handling |
| `frontend/src/types/index.ts` | 13 | `warnings`, `usedFallback` on relevant types |
| `backend/src/musicmixer/services/interpreter.py` | 17 | System prompt adjustments based on testing |

---

## Quick Reference: Key Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Key confidence: full shift | >= 0.55 | Plan B6 |
| Key confidence: half shift | 0.40-0.55 | Plan B6 |
| Key confidence: skip | < 0.40 | Plan B6 |
| Vocal pitch cap | +/- 4 semitones | Plan B6 |
| Instrument pitch cap | +/- 5 semitones | Plan B6 |
| Harmonic compatibility skip | CoF distance 0-1 | Plan B6 |
| Vocal pre-filter band | 200 Hz - 8 kHz | Plan B6 |
| Vocal pre-filter order | 2nd-order Butterworth | Plan B6 |
| Ducking cut | -3.5 dB | QR-1 |
| Ducking band | 300 Hz - 3 kHz | QR-1 |
| Ducking vocal detection band | 300 Hz - 3.5 kHz | QR-1 |
| Ducking attack | 30ms | QR-1 |
| Ducking release | 150ms | QR-1 |
| Ducking noise floor | 10th percentile | QR-1 |
| Ducking onset threshold | 4x noise floor | QR-1 |
| Ducking offset threshold | 2x noise floor | QR-1 |
| Remix TTL | 3 hours (10800s) | Plan B7 |
| Error TTL | 15 min (900s) | Plan B7 |
| Cleanup interval | 5 min (300s) | Plan B7 |
| Lock watchdog timeout | 21 min (SSE cap + 60s) | Plan B7 |
| Disk space minimum | 1 GB | Plan B2 |
| Fade-in | 2s cosine-squared | Plan B6 |
| Fade-out | 3s cosine-squared | Plan B6 |
| Cross-fade (transitions) | Per-stem equal-power interpolation | QR-1 |
| Cut (transitions) | 2ms micro-crossfade (88 samples at 44.1kHz) | QR-1 |
