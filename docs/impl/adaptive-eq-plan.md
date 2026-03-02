# Adaptive EQ System

**Date:** 2026-02-28
**Goal:** Replace blind static EQ presets with spectral-analysis-driven adaptive corrections that respond to actual stem content and cross-stem frequency conflicts.

---

## Context

The current EQ (`eq.py`) applies identical hardcoded filter chains per stem type — guitar always gets -2 dB at 200 Hz, -1.5 dB at 1200 Hz, etc. regardless of what's actually in the audio. This is fine as a safety net but misses the core problem mashups have: stems from different recordings, studios, and eras collide in unpredictable frequency ranges. A Biggie vocal with heavy proximity effect paired with a Grateful Dead guitar has different conflict patterns than the same vocal with a modern pop track. Static EQ can't adapt.

The adaptive system analyzes each stem's actual frequency content, detects where stems mask each other, and applies proportional corrections — while preserving the static presets as a safe foundation.

---

## Design

### Core Idea

1. Compute a **spectral profile** per stem (averaged magnitude spectrum in 1/3-octave bands)
2. Compare profiles to **reference envelopes** to detect per-stem anomalies (e.g., 10 dB mud peak at 250 Hz)
3. Compare profiles **across stems** to detect masking conflicts (e.g., guitar and vocal both peaking at 300 Hz)
4. Generate **adaptive correction parameters** (frequency, gain, Q) proportional to the problems found
5. Apply adaptive corrections **on top of** the existing static presets

### Key Constraints

- **Cuts only.** No adaptive boosts — boosts amplify noise and artifacts. Static preset boosts remain.
- **Conservative thresholds.** +6 dB deviation (smoothed) to trigger a per-stem correction. Musical character rarely produces isolated 6+ dB peaks in 1/3-octave spectra; separation artifacts do.
- **Cap corrections.** Max -4 dB per-stem cut, max -3 dB cross-stem cut, max 4 adaptive filters per stem.
- **Vocals always win.** Cross-stem conflicts in the 2-5 kHz presence range always cut the instrumental stem, never the vocal.

### Analysis Method

**Welch's method** (`scipy.signal.welch`, nperseg=4096, 50% overlap) → dB scale → 1/3-octave band smoothing.

- ~60ms per stem (60s audio at 44.1kHz)
- ~400ms total for 6 stems
- 31 bands from 20 Hz to 20 kHz (ISO 266 standard centers)
- Peak detection via `scipy.signal.find_peaks(prominence=3.0)`

### Adaptive Correction Logic

**Per-stem:** `correction_db = max(-4.0, -(deviation - threshold) * 0.5)` — correct half the excess, floored at -4 dB (most aggressive allowed cut).

**Cross-stem:** For each 1/3-octave band where both stems exceed the per-stem anomaly threshold (i.e., both have relative deviation > +6 dB from their reference envelope in the same band), compute overlap severity. Cut the lower-priority stem proportionally, capped at -3 dB. Priority: vocals > bass > drums > guitar/piano/other.

> **Note:** The cross-stem threshold uses relative deviations (same as `band_energies_db`), not absolute dBFS values. Since v1 reference envelopes are flat (0 dB), a +6 dB relative deviation means the band is 6 dB above the flat reference. This keeps the cross-stem detection consistent with per-stem anomaly detection and avoids mixing absolute/relative scales.

**Q adaptation:** `Q = max(1.5, min(3.0, deviation_db / 4))` — wider for broad problems, narrower for localized peaks.

### Reference Envelopes

31-float arrays per stem type representing expected spectral shape after BS-RoFormer separation. For v1, use flat references (0 dB) and rely purely on absolute thresholds. Build real references later from accumulated test data.

---

## Implementation

### New File: `services/spectral.py` (~250 LOC)

```
compute_spectral_profile(audio, sr) -> SpectralProfile
    - mono conversion, Welch PSD, dB conversion, 1/3-octave smoothing, peak detection

detect_conflicts(vocal_profiles: list[SpectralProfile], inst_profiles: list[SpectralProfile]) -> list[FrequencyConflict]
    - compares each vocal-source stem against each instrumental-source stem
    - per-band overlap check (both stems > +6 dB relative deviation), priority-based cut assignment

compute_adaptive_corrections(conflicts, vocal_profiles, inst_profiles)
    -> tuple[dict[str, list[tuple[freq, gain, q]]], dict[str, list[tuple[freq, gain, q]]]]
    - returns (vocal_corrections, inst_corrections) — two separate dicts keyed by stem name
    - merges per-stem anomaly corrections + cross-stem conflict corrections per source
    - caps at 4 corrections per stem, clamps gain to [-4, 0] dB
```

### New Dataclasses in `models.py` (~35 LOC)

```python
@dataclass
class SpectralProfile:
    stem_type: str
    band_centers_hz: np.ndarray       # (31,) ISO 266 centers
    band_energies_db: np.ndarray      # (31,) smoothed energy per band
    peak_frequencies_hz: np.ndarray   # detected peaks
    peak_magnitudes_db: np.ndarray

@dataclass
class FrequencyConflict:
    stem_a: str
    stem_b: str
    center_hz: float
    severity_db: float
    recommended_cut_stem: str
    recommended_cut_db: float
    recommended_q: float
```

### Modified: `eq.py` (+~20 LOC)

Add `adaptive_corrections: list[tuple[float, float, float]] | None = None` parameter to `apply_corrective_eq`. When provided, append `PeakFilter` plugins to the existing preset board. Single pedalboard pass — no extra audio processing cost.

### Modified: `pipeline.py` (+~25 LOC)

Insert adaptive analysis between step 7.7 (vocal bandpass) and step 7.75 (per-stem EQ):

```
Between 7.7 and 7.75:
  [IF adaptive_eq_enabled] Compute spectral profiles for all stems
      Detect cross-stem conflicts (vocal_profiles vs inst_profiles)
      Generate adaptive corrections (vocal_corrections, inst_corrections)
7.75: Apply preset EQ + adaptive corrections (or preset-only if flag off / analysis fails)
```

Per-stem correction wiring at step 7.75:

```python
# Step 7.75 — pass per-stem adaptive corrections into the EQ call
for stem_type, audio in vocal_audio.items():
    adaptive = vocal_corrections.get(stem_type, [])
    vocal_audio[stem_type] = apply_corrective_eq(audio, sr, stem_type, ..., adaptive_corrections=adaptive)

for stem_type, audio in inst_audio.items():
    adaptive = inst_corrections.get(stem_type, [])
    inst_audio[stem_type] = apply_corrective_eq(audio, sr, stem_type, ..., adaptive_corrections=adaptive)
```

> **Bandpass interaction:** Spectral analysis runs after the vocal bandpass filter (step 7.7), so adaptive EQ only operates on post-bandpass frequencies (150 Hz - 16 kHz for vocals). This is intentional — the bandpass already handles out-of-range energy, and adaptive EQ should not duplicate that work.

### Modified: `config.py` (+1 LOC)

```python
adaptive_eq_enabled: bool = False  # ADAPTIVE_EQ_ENABLED env var
```

### New: `tests/test_spectral.py`

- Sine wave at 400 Hz → verify peak detected at correct band
- Two overlapping sines (different "stems") → verify conflict detected
- Non-overlapping sines → verify no conflict
- Silent/short/mono edge cases
- Performance: 12 stems under 2 seconds

### Extended: `tests/test_eq.py`

- `adaptive_corrections=None` matches current behavior (backward compat)
- Adaptive corrections produce measurable dB change at target frequency
- Extreme corrections are clamped

---

## Files Changed

| File | Change | LOC |
|------|--------|-----|
| `services/spectral.py` | **New** — spectral analysis + conflict detection | ~250 |
| `models.py` | Add `SpectralProfile`, `FrequencyConflict` | ~35 |
| `services/eq.py` | Add `adaptive_corrections` param, `_build_adaptive_board` helper | ~20 |
| `services/pipeline.py` | Wire spectral analysis between step 7.7 and 7.75 behind flag | ~25 |
| `config.py` | Add `adaptive_eq_enabled` flag | ~1 |
| `tests/test_spectral.py` | **New** — unit tests for spectral module | ~150 |
| `tests/test_eq.py` | Extend with adaptive correction tests | ~50 |

---

## Implementation Order

1. `models.py` — dataclasses (no dependencies)
2. `spectral.py` — analysis module (depends only on numpy/scipy)
3. `test_spectral.py` — validate in isolation
4. `eq.py` — add adaptive parameter
5. `test_eq.py` — backward compat + adaptive tests
6. `config.py` — feature flag
7. `pipeline.py` — integration, gated by flag

Steps 1-5 can be developed and tested without touching the pipeline.

---

## Performance Budget

| Operation | Per stem | 6 stems total |
|-----------|----------|---------------|
| Welch PSD + band mapping | ~60ms | ~360ms |
| Peak detection | ~1ms | ~6ms |
| Conflict detection | — | ~5ms |
| Correction computation | — | ~1ms |
| Adaptive filter application | ~15ms | ~90ms |
| **Total** | | **~460ms** |

0.4% increase on a typical 120-second pipeline. Peak memory: ~21 MB (STFT of longest stem).

---

## Fallback

If spectral analysis fails (exception, NaN, timeout), log the error and fall back to preset-only EQ. The `adaptive_corrections` parameter defaults to `None`, so the fallback path is identical to current production behavior. Per-stem failures are caught individually — a bad "other" stem doesn't block adaptive EQ on vocals.

---

## Verification

1. Run backend tests: `cd backend && uv run pytest tests/test_eq.py tests/test_spectral.py -v`
2. Run full test suite: `cd backend && uv run pytest tests/ -m "not slow" -v`
3. Manual pipeline test with `ADAPTIVE_EQ_ENABLED=true`: run a remix on example songs, compare output LUFS and listen for artifacts vs. the flag-off baseline
4. Check logs for spectral analysis timing and detected conflicts
