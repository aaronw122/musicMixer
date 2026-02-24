# Audio Pipeline Reference

> Implementation reference extracted from the full plan. Canonical source: `docs/plans/2026-02-23-feat-mvp-prompt-based-remix-plan.md`

## Revised 13-Step Pipeline Order

```
1.  Separate stems (htdemucs_ft) — both songs
2.  Analyze audio (BPM, key, energy, beat grid) — both songs
    2b. Reconcile BPM across both songs
    2c. Measure vocal_energy_ratio (vocal stem LUFS vs full mix LUFS)
3.  LLM prompt interpretation → RemixPlan (see llm-integration.md)
4.  Trim stems to source time ranges (start_time_vocal/instrumental from RemixPlan)
5a. Tempo match all stems to common BPM (rubberband single-pass)
5b. Scale beat_frames: beat_frames_stretched = beat_frames * (original_bpm / target_bpm)
    MUST happen after rubberband and BEFORE arrangement renderer uses the grid.
6.  Key match if needed (same rubberband pass as 5a when possible)
7.  Cross-song level matching (LUFS with silence guard)
8.  Validate LLM sections (10-point checklist — see llm-integration.md)
9.  Render arrangement into TWO BUSES: vocal bus + instrumental bus
    - Place stems per LLM section timeline using beat grid
    - Apply per-stem gains from Section.stem_gains
    - Apply transition envelopes
    - Vocal bus = rendered vocal stem with arrangement gains
    - Instrumental bus = sum of rendered drums + bass + other stems
10. Spectral ducking on instrumental bus (using vocal bus for activity detection)
11. Sum vocal bus + instrumental bus into final mix
12. LUFS normalize final mix → peak limiter (tanh soft-clip)
13. Fade-in/fade-out → export MP3 via ffmpeg
```

---

## Data Models

### EnergyRegion

```python
@dataclass
class EnergyRegion:
    start_sec: float
    end_sec: float
    relative_energy: float   # 0.0-1.0 normalized (1.0 = loudest region)
    onset_density: float     # 0.0-1.0 normalized — rhythmic activity level
    label: str               # "high" | "medium" | "low" — composite energy label
    character: str           # "rhythmic" | "sustained" | "sparse" | "moderate"
    # Character derived from energy x onset density:
    #   high energy + high onsets = "rhythmic" (chorus, drop)
    #   high energy + low onsets  = "sustained" (breakdown, build, pad section)
    #   low energy + high onsets  = "sparse" (verse with sparse percussion)
    #   low energy + low onsets   = "moderate" (intro, outro, quiet passage)
```

### AudioMetadata

```python
@dataclass
class AudioMetadata:
    bpm: float
    bpm_confidence: float
    beat_frames: np.ndarray   # Beat frame positions from librosa.beat.beat_track — the beat GRID.
                              # After tempo matching, scale: beat_frames_stretched = beat_frames * (original_bpm / target_bpm)
    key: str                  # e.g., "C"
    scale: str                # "major" or "minor"
    key_confidence: float
    duration_seconds: float
    total_beats: int          # total beats (rounded to nearest bar boundary)
    vocal_energy_ratio: float = 0.0  # 0.0-1.0 — vocal stem LUFS relative to full mix.
                              # Default 0.0, filled AFTER stem separation and BEFORE LLM call.
                              # ~0.8+ = clear prominent vocals. ~0.1 = effectively instrumental.
    energy_regions: list[EnergyRegion]  # sorted by time (temporal order), ~8s windows
```

### Section

```python
@dataclass
class Section:
    label: str              # "intro" | "verse" | "breakdown" | "drop" | "outro"
    start_beat: int         # beat-aligned (snapped to grid)
    end_beat: int
    stem_gains: dict[str, float]  # {"vocals": 1.0, "drums": 0.7, "bass": 0.8, "other": 0.6}
    transition_in: str      # "fade" | "crossfade" | "cut"
    transition_beats: int   # length of transition envelope
```

### RemixPlan

```python
@dataclass
class RemixPlan:
    vocal_source: str                    # "song_a" | "song_b"
    start_time_vocal: float              # seconds, original tempo domain
    end_time_vocal: float
    start_time_instrumental: float
    end_time_instrumental: float
    sections: list[Section]              # Beat-aligned arrangement
    tempo_source: str                    # "song_a" | "song_b" | "average"
    key_source: str                      # "song_a" | "song_b" | "none"
    explanation: str                     # LLM reasoning (shown to user)
    warnings: list[str]                  # Caveats about what couldn't be fulfilled
    used_fallback: bool = False          # True when deterministic fallback was used
```

**MVP constraint:** `vocal_source` determines the split. Vocals come from that song; drums, bass, and other all come from the other song. No cross-song stem mixing.

---

## Stem Separation Service

```python
def separate_stems(audio_path: Path, output_dir: Path,
                   progress_callback: Callable | None = None) -> dict[str, Path]:
    """Separate audio into 4 stems using htdemucs_ft.

    Returns: {"vocals": Path, "drums": Path, "bass": Path, "other": Path}
    Output format: WAV (uncompressed).
    Model auto-downloads on first run.
    Raises SeparationError on failure.
    """
```

- Uses `audio-separator` package with `htdemucs_ft` model (configurable via `settings.stem_model`)
- ~10s/song on GPU, ~3-6min/song on CPU
- ~160MB WAV output per song (4 stems x ~40MB each)

---

## Audio Analysis Service

```python
def analyze_audio(audio_path: Path) -> AudioMetadata:
    """Detect BPM, key, energy profile, and beat grid.
    Analysis runs on original full songs, not stems.
    """

def detect_key(audio_path: Path) -> tuple[str, str, float]:
    """Returns (key, scale, confidence). Uses essentia if available, else librosa."""

def reconcile_bpm(meta_a: AudioMetadata, meta_b: AudioMetadata) -> tuple[AudioMetadata, AudioMetadata]:
    """Cross-song BPM reconciliation. Returns updated copies (no mutation)."""
```

### Energy Profile

```python
def compute_energy_profile(y: np.ndarray, sr: int, window_sec: float = 8.0) -> list[EnergyRegion]:
    """Compute RMS energy + onset density in fixed windows. Returns temporal order."""
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
    # Normalize to 0.0-1.0
    max_energy = max(r.relative_energy for r in regions) if regions else 1.0
    for r in regions:
        r.relative_energy = r.relative_energy / max_energy if max_energy > 0 else 0.0
        r.label = "high" if r.relative_energy >= 0.75 else "low" if r.relative_energy <= 0.25 else "medium"
    regions.sort(key=lambda r: r.start_sec)
    return regions
```

**Onset density:** Combine with `librosa.onset.onset_strength` — multiply normalized RMS by normalized onset density to populate `character` field.

### BPM Reconciliation

Cross-song reconciliation (runs after both songs analyzed). For each song, generate `{original, halved, doubled, 3/2, 2/3}` interpretations, filter to **70-180 BPM**, evaluate all valid pairs, select pair with minimum percentage difference. Return updated copies (no mutation).

---

## Tempo Matching — Tiered Limits

| BPM Gap | Action |
|---------|--------|
| < 10% | Stretch either/both silently |
| 10-25% | Stretch vocals only (preserve instrumental tempo) |
| 25-30% | Stretch vocals only, strongly warn in explanation |
| 30-50% | **Vocals-only stretch** — stretch vocal stem to match instrumental's tempo. Vocals remain intelligible; instrumentals would sound broken. |
| > 50% | Skip tempo matching entirely. Use LLM alternative techniques (a cappella over breakdowns, loop-based phrases). |

**Default tempo target:** Instrumental source song's tempo (not average). When `tempo_source` is `"average"`, split stretch only if gap <15%.

---

## Rubberband CLI Invocation

Single-pass tempo + pitch per stem. Do NOT use pyrubberband Python API — call CLI directly.

```python
import subprocess, soundfile as sf, numpy as np

def rubberband_process(audio: np.ndarray, sr: int, tempo_ratio: float,
                       semitones: float = 0, is_vocal: bool = False) -> np.ndarray:
    """Single-pass tempo + pitch via rubberband CLI."""
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

**Startup check:** Verify `rubberband --version` returns v3.x; fall back to `--fine` if v2.x.

---

## Key Matching Algorithm

1. Normalize both keys to relative major (e.g., A minor -> C major)
2. Compute chromatic semitone interval: `interval = (target_root - source_root) % 12`
3. Choose minimal shift: `shift = interval if interval <= 6 else interval - 12`
4. **Harmonic compatibility gate:** Circle-of-fifths distance between normalized keys. If 0 or 1 (same key, relative major/minor, or perfect fifth apart), skip key matching entirely.
5. Handle enharmonic equivalence (F# = Gb) via integer pitch classes

**Caps:** +/- 4 semitones for vocals, +/- 5 for instruments.
**Formant preservation:** `--formant` flag for vocal stems (only when pitch shift is non-zero).
**Key confidence gate:** Skip key matching entirely if `key_confidence < 0.45` for either song.

---

## Cross-Song Level Matching

```python
LUFS_FLOOR = -40.0  # Below this, stem is effectively silence/noise

vocal_lufs = meter.integrated_loudness(vocal_audio)
instrumental_lufs = meter.integrated_loudness(instrumental_sum)

if vocal_lufs < LUFS_FLOOR or instrumental_lufs < LUFS_FLOOR:
    logger.warning(f"Skipping level matching: vocal={vocal_lufs:.1f}, instrumental={instrumental_lufs:.1f}")
else:
    vocal_offset_db = compute_vocal_offset(vocal_audio, instrumental_sum, sr)
    target_vocal_lufs = instrumental_lufs + vocal_offset_db
    gain_db = target_vocal_lufs - vocal_lufs
    gain_db = np.clip(gain_db, -12.0, 12.0)  # Safety cap
    vocal_audio = vocal_audio * (10 ** (gain_db / 20.0))
```

### Spectral-Density-Adaptive Vocal Offset

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
    density_ratio = inst_vocal_band_energy / (inst_full_energy + 1e-10)
    # Map: sparse instrumental (low ratio) = +1-2 dB, dense = +5-8 dB
    offset_db = np.interp(density_ratio, [0.1, 0.3, 0.5, 0.7], [1.0, 3.0, 5.0, 7.0])
    return float(np.clip(offset_db, 1.0, 8.0))
```

---

## Arrangement Renderer

Deterministic renderer of the LLM's section plan. Outputs TWO buses (vocal + instrumental) for spectral ducking.

1. Convert beat positions to sample positions using **beat grid** (`beat_frames`, NOT constant-BPM math)
2. For each section, extract sample range from each stem, apply per-stem gain from `stem_gains`
3. Apply transition envelopes:
   - **"fade"**: Cosine fade-in over `transition_beats` (0 -> section gain)
   - **"crossfade"**: Per-stem gain interpolation (NOT summed-audio). Cosine curve from previous gain to current. Avoids volume dip.
   - **"cut"**: Immediate + **micro-crossfade ~2ms (88 samples at 44.1kHz)** to eliminate clicks

```python
def snap_to_bar(sample_position: int, beat_positions: np.ndarray, beats_per_bar: int = 4) -> int:
    """Snap a sample position to the nearest bar (downbeat). Validation utility."""
    bar_positions = beat_positions[::beats_per_bar]
    return int(bar_positions[np.argmin(np.abs(bar_positions - sample_position))])
```

---

## Spectral Ducking

```python
from scipy.signal import butter, sosfiltfilt

def spectral_duck(instrumental: np.ndarray, vocal: np.ndarray, sr: int,
                  cut_db: float = -3.5, lo: float = 300, hi: float = 3000) -> np.ndarray:
    """Duck instrumental mid-range where vocals are active. Zero-phase. Expects stereo (2D)."""
    assert instrumental.ndim == 2

    # 1. Vocal activity mask: RMS in 50ms frames, 40th percentile threshold
    vocal_mono = vocal.mean(axis=1) if vocal.ndim == 2 else vocal
    frame_len = int(0.05 * sr)
    vocal_energy = np.array([
        np.sqrt(np.mean(vocal_mono[i:i+frame_len]**2))
        for i in range(0, len(vocal_mono) - frame_len, frame_len)
    ])
    threshold = np.percentile(vocal_energy[vocal_energy > 0], 40) if np.any(vocal_energy > 0) else 0
    vocal_active = (vocal_energy > threshold).astype(float)

    # 2. Upsample + smooth (30ms attack / 150ms release — use scipy.signal.lfilter in prod)
    mask = np.repeat(vocal_active, frame_len)[:len(instrumental)]
    attack_alpha, release_alpha = 1.0 / int(0.03 * sr), 1.0 / int(0.15 * sr)
    smoothed = np.copy(mask)
    for i in range(1, len(smoothed)):
        alpha = attack_alpha if smoothed[i] > smoothed[i-1] else release_alpha
        smoothed[i] = smoothed[i-1] + alpha * (smoothed[i] - smoothed[i-1])

    # 3. Extract mid-band (MUST use sosfiltfilt, not sosfilt — zero-phase prevents comb artifacts)
    sos = butter(4, [lo, hi], btype='band', fs=sr, output='sos')
    mid_band = sosfiltfilt(sos, instrumental, axis=0)

    # 4. Reduce mid-band where vocals active
    gain = 10 ** (cut_db / 20)
    reduction = mid_band * (1 - gain) * smoothed[:, np.newaxis]
    return instrumental - reduction
```

Applied after arrangement rendering (step 10) so vocal activity mask reflects actual arrangement gains.

---

## LUFS Normalization + Peak Limiter

Normalize final summed mix (NOT per-stem) to -14.0 LUFS. Then apply soft-clip:

```python
ceiling = 10 ** (-1.0 / 20.0)  # -1.0 dBTP ~ 0.891
peak = np.max(np.abs(mixed))
if peak > ceiling:
    mixed = np.tanh(mixed / ceiling) * ceiling
```

---

## Fade + Export + Standardization

**Sample rate (before all processing):** All stems -> 44.1kHz stereo 32-bit float. Resample with `librosa.resample`. Mono-to-stereo: `stereo = np.column_stack([mono, mono])`.

**Fades:** 2-second fade-in, 3-second fade-out (in numpy, not pydub).

**Export** (do NOT use pydub — quantizes to 16-bit internally):
```python
sf.write(tmp_wav_path, mixed, sr, subtype="FLOAT")
subprocess.run(["ffmpeg", "-y", "-i", str(tmp_wav_path), "-codec:a", "libmp3lame",
                "-b:a", "320k", str(output_path)], check=True, capture_output=True, timeout=120)
```

---

## Pipeline Orchestrator

```python
def run_pipeline(session_id: str, song_a_path: Path, song_b_path: Path,
                 prompt: str, event_queue: queue.Queue) -> None:
    """End-to-end pipeline. Sets session attributes directly AND pushes queue events."""

def process_remix(remix_plan: RemixPlan, song_a_stems: dict[str, Path], song_b_stems: dict[str, Path],
                  song_a_meta: AudioMetadata, song_b_meta: AudioMetadata,
                  output_path: Path, progress_callback: Callable | None = None) -> Path:
    """Execute steps 4-13 of the pipeline. Returns path to rendered MP3."""
```

- Pipeline sets `session.status`, `session.remix_path`, `session.explanation` at key transitions
- On error: pushes `{"step": "error", "detail": str(e), "progress": 0}` and cleans up

**Dependencies:** `audio-separator`, `librosa`, `essentia`, `pyrubberband`, `pydub`, `soundfile`, `pyloudnorm`, `numpy`, `scipy`, `anthropic` | System: `ffmpeg`, `rubberband` (v3.x preferred), `libsndfile`

---

## Cross-References

- LLM that generates RemixPlan: see `llm-integration.md`
- API endpoints that expose the pipeline: see `api-contract.md`
- Frontend that consumes progress events: see `frontend.md`
