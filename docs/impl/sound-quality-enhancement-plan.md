# Sound Quality Enhancement Plan: Per-Stem EQ, Multiband Compression, Static Mastering

revision: 13

## Context

The remix pipeline already has solid foundations (single-band compression, spectral ducking, LUFS normalization, true-peak limiting), but mixes still suffer from:
- **Echoey vocals** — source recordings have reverb that gets amplified by stem separation, then clashes with new instrumentals
- **Frequency buildup** when stems overlap (muddy low-mids, harsh highs)
- **One-size-fits-all dynamics** — the compressor treats all frequencies the same
- **No spectral shaping** — stems are mixed raw without corrective EQ
- **No reference matching** — output tonal balance varies wildly between remix combos

This plan adds three processing layers: per-stem corrective EQ, multiband compression, and static mastering. All behind feature flags for A/B comparison. (Vocal de-reverb is deferred to post-v1.)

## New Dependencies

Add to `backend/pyproject.toml`:
```
"pedalboard>=0.9.0",    # Spotify's DSP lib: fast EQ, compressor, limiter
```

> **matchering removed for v1.** Reference-based mastering has been replaced with a static mastering chain (see Step 6). matchering can be re-added in a future version if genre-appropriate reference tracks are curated.

## New Files

| File | Purpose |
|------|---------|
| `backend/src/musicmixer/services/audio_utils.py` | Shared DSP utilities (`process_with_pedalboard` helper, used by eq/multiband/mastering) |
| `backend/src/musicmixer/services/dereverb.py` | *(Deferred to post-v1)* Vocal de-reverb via audio-separator + noise gate cleanup |
| `backend/src/musicmixer/services/eq.py` | Per-stem corrective EQ + resonance detection |
| `backend/src/musicmixer/services/multiband.py` | 4-band multiband compressor |
| `backend/src/musicmixer/services/mastering.py` | Static mastering chain (LUFS normalization + limiter) |
| `backend/tests/test_dereverb.py` | *(Deferred to post-v1)* De-reverb unit tests |
| `backend/tests/test_eq.py` | EQ unit tests |
| `backend/tests/test_multiband.py` | Multiband compression tests |
| `backend/tests/test_mastering.py` | Mastering wrapper tests |

## Files to Modify

| File | Changes |
|------|---------|
| `backend/pyproject.toml` | Add pedalboard dep |
| `backend/src/musicmixer/config.py` | Add 4 feature flags (de-reverb deferred to post-v1) |
| `backend/src/musicmixer/services/pipeline.py` | Wire in steps 7.75, 9.5, 13.3, 13.7, 14.5; add `source_quality_a` / `source_quality_b` params to `run_pipeline()` and derive `is_lossy_source_a/b` inside the function (see Step 7) |
| `backend/src/musicmixer/api/remix.py` | In `_youtube_pipeline_wrapper`, pass `source_quality_a` and `source_quality_b` through to `run_pipeline()`. Without this, the values are always `None`. Example call: `run_pipeline(..., source_quality_a=source_quality_a, source_quality_b=source_quality_b)`. Note: `_pipeline_wrapper` (file upload path) requires no changes since the new parameters default to `None`. |
| `backend/src/musicmixer/services/processor.py` | Add `headroom_db: float = 0.0` parameter to `lufs_normalize_constrained` signature (see Step 6 note on signature change) |

## Enhanced Pipeline

New steps marked `[NEW]`:

```
 7.7  Vocal bandpass (150Hz-16kHz)  # Upper cutoff 16kHz — CHANGED from 8kHz; gated behind ab_per_stem_eq_v1 (see note below)
 7.75 Broad preset EQ cuts (Q~1-3) .............. [NEW]
 8    Compute tempo plan
 9    Rubberband tempo match
 9.5  Narrow resonance notch cuts ................ [NEW] (moved from pre-stretch)
11    Vocal compression
11.5  Cross-song level match
11.8  Pre-limit drums/bass transients
12    Render arrangement
12.5  Spectral ducking
13    Sum buses → final mix
13.3  Multiband compression (4-band) ............. [NEW]
13.7  Auto-leveler (gentler when multiband on)
      ┌─ if ab_static_mastering_v1 ──────────────────────┐
14.5  │ Static mastering chain .................. [NEW]      │
      │ (LUFS normalize -12 LUFS + limiter -1.0 dBTP)       │
14.6  │ Soft clip safety (-1.0 dBTP, knee 2dB)              │
      ├─ else ──────────────────────────────────────────────┤
14    │ True-peak limiter                                    │
15    │ LUFS normalize                                       │
15.5  │ Soft clip safety                                     │
      └────────────────────────────────────────────────────┘
16    Fades
17    Export MP3
```

> **Vocal bandpass change (step 7.7):** The existing code in `pipeline.py` (line ~481-483) uses an 8kHz upper cutoff for the vocal bandpass. This plan changes it to **16kHz** to preserve vocal air, breathiness, and sibilance that 8kHz cuts. **This change MUST be gated behind `ab_per_stem_eq_v1`** to keep the control group on the original 8kHz cutoff for valid A/B comparison. Implementation: when `ab_per_stem_eq_v1` is True, use `LowpassFilter(cutoff_frequency_hz=16000)`; when False, retain the existing `LowpassFilter(cutoff_frequency_hz=8000)`. Without this gating, the control group is modified and A/B results are invalid. The corrective EQ preset for vocals (Step 4) includes its own LPF at 16kHz, which provides the final HF shaping — these are not redundant because the bandpass runs before EQ and acts as a wider safety net.

> **Why mutual exclusion?** The static mastering chain applies its own limiter and LUFS normalization. Running the standard limiter + LUFS normalize + soft-clip before or after would create gain-chain conflicts (double-limiting, loudness overshoots). The two paths are mutually exclusive.

---

## Step 1: Add Dependencies

Add `pedalboard>=0.9.0` to `pyproject.toml` dependencies. Run `uv sync`.

## Step 2: Add Feature Flags

In `config.py`, add to the `Settings` class alongside existing `ab_*` flags:

```python
# Sound quality enhancement flags (default-off for A/B)
ab_per_stem_eq_v1: bool = False
ab_resonance_detection_v1: bool = False
ab_multiband_comp_v1: bool = False
ab_static_mastering_v1: bool = False
```

> **Deferred:** `ab_vocal_dereverb_v1` is deferred to post-v1. Do not add this flag in v1.

## Step 3: Create `dereverb.py` — Vocal De-Reverb

> **Deferred to post-v1.** This entire step is not included in the current processing chain. The de-reverb feature, its ML model dependency, the post-de-reverb expander, and the `ab_vocal_dereverb_v1` flag are all out of scope for v1. The section below is retained for future reference only.

### The Problem

Source recordings contain room reverb. Stem separation amplifies this because the model can't cleanly split "dry vocal" from "vocal reverb tail." When the reverb-laden vocal is layered over different instrumentals, the original room sound clashes and compounds — making the remix sound echoey.

### Solution: ML De-Reverb via audio-separator

`audio-separator` (already a dependency) supports de-reverb models. Use **`Reverb_HQ_By_FoxJoy.onnx`** (MDX-Net architecture, 67MB, ONNX runtime already installed). The model auto-downloads on first use.

The API is identical to existing stem separation:
```python
from audio_separator.separator import Separator

separator = Separator()
separator.load_model("Reverb_HQ_By_FoxJoy.onnx")
output_files = separator.separate(vocal_stem_path)
# Returns list of output file paths — select by filename pattern, NOT by index.
# The dry vocal output contains "No Reverb" or "dry" in its filename.
# Index order may vary across model versions.
dry_vocal_path = next(f for f in output_files if "No Reverb" in f or "dry" in f.lower())
```

> **Output file selection:** Always select the de-reverbed output by matching filename substrings (`"No Reverb"` or `"dry"`), not by list index. The order of `output_files` is not guaranteed across model versions. Verify this pattern against the specific `Reverb_HQ_By_FoxJoy.onnx` model's output naming convention during implementation.

### Pipeline Placement

Applied at **step 7.72** — after vocal bandpass (7.7), before EQ (7.75). Rationale:
- Bandpass first removes sub-bass rumble and high-freq artifacts, giving the de-reverb model cleaner input
- De-reverb before EQ means EQ is shaping the dry vocal, not the reverb tail
- De-reverb before tempo stretch avoids stretching reverb artifacts

### Post-De-Reverb Expander (optional, off by default in v1)

> **Design decision:** A hard noise gate after ML de-reverb damages vocal naturalness — abrupt gain changes clip reverb tails mid-decay, creating audible pumping on sustained notes. An expander with a gentle ratio is preferable: it attenuates low-level reverb tails gradually rather than slamming them shut.
>
> **v1 default:** Expander is **off by default**. Enable only if listening tests reveal problematic residual reverb tails after ML de-reverb alone. The ML model handles the heavy lifting; the expander is a surgical cleanup tool, not a primary processor.

If enabled, apply a custom RMS-based expander. **Do NOT use `pedalboard.NoiseGate`** — it does not support a `ratio` parameter and operates as a hard gate (fully open or fully closed), which damages vocal naturalness with abrupt gain changes.

Instead, implement a manual expander with RMS-based gain reduction and attack/release smoothing:

```python
def rms_expander(
    audio: np.ndarray,
    sr: int,
    threshold_db: float = -50.0,
    ratio: float = 2.0,
    attack_ms: float = 5.0,
    release_ms: float = 300.0,
    rms_window_ms: float = 20.0,
) -> np.ndarray:
    """
    Gentle expander: when RMS falls below threshold, attenuate by (threshold - rms) * (1 - 1/ratio).
    Uses smoothed gain envelope with attack/release to avoid abrupt transitions.
    """
    # 1. Compute per-sample RMS in sliding window
    # 2. Convert to dB
    # 3. For samples below threshold: gain_reduction = (threshold_db - rms_db) * (1 - 1/ratio)
    # 4. Smooth gain envelope: attack_coeff for increasing gain, release_coeff for decreasing
    # 5. Apply gain reduction
    ...
```

Parameters chosen to preserve vocal naturalness:
- **Threshold -50dB:** Only acts on very low-level content (reverb tails in gaps)
- **Ratio 2:1:** Gentle expansion — attenuates but doesn't silence
- **Attack 5ms:** Slow enough to avoid clipping vocal onsets
- **Release 300ms:** Slow release preserves natural decay of sustained notes
- **RMS window 20ms:** Short enough for responsiveness, long enough to avoid modulation artifacts

### Function Signature

```python
def dereverb_vocal(
    vocal_path: Path,
    output_dir: Path,
    sr: int = 44100,
    apply_expander: bool = False,  # off by default in v1
    expander_threshold_db: float = -50.0,
) -> np.ndarray
```

Takes the vocal stem file path (already on disk from separation), runs de-reverb, optionally applies gentle expander, returns float32 stereo numpy array. The reverb output is discarded.

**File I/O note:** audio-separator operates on files, not numpy arrays. The vocal stem WAV is already on disk from step 1 (separation). The de-reverb model writes output files to `output_dir`. We read the dry vocal back with `soundfile.read(dtype="float32")` and clean up intermediate files.

### Feature Flag

Gated by `ab_vocal_dereverb_v1`. When disabled, the pipeline skips straight to EQ as before.

## Step 4: Create `eq.py` — Per-Stem Corrective EQ

### Pedalboard Helper

Shared utility for channel convention conversion. Our pipeline uses `(samples, channels)`, pedalboard expects `(channels, samples)`. **This helper lives in `backend/src/musicmixer/services/audio_utils.py`** (shared module), not in `eq.py`, because it is also needed by `multiband.py` and `mastering.py`. All three modules import from `audio_utils`:

```python
# In backend/src/musicmixer/services/audio_utils.py
def process_with_pedalboard(audio: np.ndarray, board: Pedalboard, sr: int) -> np.ndarray
```

> **Note:** The function is public (`process_with_pedalboard`, no leading underscore) since it is a shared utility imported across modules.

### Preset EQ Profiles

> **Philosophy: first, do no harm.** These are corrective EQ profiles for stems that have already been through ML separation and may have artifacts. The goal is gentle cleanup, not creative shaping. All boosts are capped at +0.75dB — it is far better to under-process than to strip character from the source material. Cuts can be slightly more aggressive since removing problems is lower-risk than adding coloration.

Gentle corrective cuts per stem type using pedalboard's `PeakFilter`, `HighpassFilter`, `LowpassFilter`, `HighShelfFilter`.

**Default Q values:** All `PeakFilter` cuts default to **Q=1.5** (wider for gentle correction); boosts default to **Q=2.0** (narrower for surgical enhancement). These defaults apply unless overridden per-filter in the table below.

| Stem | Key Corrections |
|------|----------------|
| **vocals** | HPF 80Hz, cut mud 250Hz (-2.5dB), cut box 800Hz (-1.5dB), boost presence 3kHz (+0.75dB), LPF 16kHz |
| **drums** | HPF 30Hz, cut box 400Hz (-3dB), cut ring 800Hz (-2dB), boost snap 5kHz (+0.75dB), shelf cut highs 12kHz (-1dB) |
| **bass** | HPF 30Hz, boost fundamental 60Hz (+0.75dB), cut mud 250Hz (-2dB), cut bleed 800Hz (-3dB), LPF 8kHz |
| **guitar** | HPF 80Hz, cut mud 200Hz (-2dB), cut honk 1.2kHz (-1.5dB), boost clarity 3.5kHz (+0.75dB) |
| **piano** | HPF 60Hz, cut mud 300Hz (-1.5dB), boost clarity 2.5kHz (+0.75dB) |
| **other** | HPF 80Hz, cut mud 400Hz (-2dB), mild clarity 2.5kHz (+0.5dB) |

### Resonance Detection (gated by `ab_resonance_detection_v1`, OFF by default in v1)

> **Design decision:** Automated resonance detection is a **rescue feature for bad sources**, not a standard processing step. Aggressive notch cutting strips character and warmth from stems — the very frequencies that make a guitar sound like *that* guitar or a voice sound like *that* voice. For v1, this flag defaults to `False` in `config.py` and should only be enabled selectively when source material has known problematic resonances.

**Stem type restriction:** Resonance detection is only applied to `stem_type in ("vocals", "drums", "other")`. It is **not applied to pitched instruments** (bass, guitar, piano) because the algorithm cannot distinguish intentional harmonic content (fundamentals, overtones) from problematic resonances in pitched sources. Applying it to pitched instruments produces false positives that strip tonal character.

When enabled (and stem type is eligible), the detection algorithm:

1. FFT analysis → averaged magnitude spectrum over multiple windows
2. Smooth baseline via moving average
3. Find peaks >10dB above baseline (restricted to **200–600Hz** and **2–4kHz** ranges, max 3 resonances)
4. Apply narrow notch cuts (Q=4-6, max depth **-3dB**) at detected frequencies

Conservative parameters to minimize character loss:
- **Threshold 10-12dB above baseline** (was 6dB — too sensitive, catches intentional tonal features)
- **Q=4-6** (was 8-12 — narrower Q is more surgical but also more audible as coloration)
- **Max depth -3dB** (was -3 to -6dB — capped to prevent over-cutting)
- **Detection range restricted** to 200–600Hz (mud/box) and 2–4kHz (harshness) — the two ranges where resonances are most commonly problematic. Full 100Hz–10kHz scanning catches too many legitimate spectral features.

> **Important: narrow notch filters must be applied after tempo stretch.** Narrow notch cuts (Q=4-6) confuse rubberband's transient detector and cause ringing artifacts when the audio is subsequently time-stretched. Only broad preset EQ cuts (Q~1-3) are safe to apply before stretching. The `apply_corrective_eq` function accepts an `apply_resonance_cuts` flag to control this — call it with `apply_resonance_cuts=False` before stretch and `apply_resonance_cuts=True` (preset=False) after stretch.

```python
RESONANCE_ELIGIBLE_STEMS = {"vocals", "drums", "other"}

def detect_resonances(audio, sr, threshold_db=10.0, max_resonances=3,
                      freq_ranges=((200, 600), (2000, 4000))) -> list[tuple[float, float]]

def apply_corrective_eq(audio, sr, stem_type, apply_preset=True, apply_resonance_cuts=True,
                        halve_hf_boosts: bool = False) -> np.ndarray
    # When apply_resonance_cuts=True, internally checks:
    #   if stem_type not in RESONANCE_ELIGIBLE_STEMS: skip resonance detection
```

## Step 5: Create `multiband.py` — 4-Band Multiband Compression

### Crossover Frequencies

150Hz / 600Hz / 3000Hz — standard 4-band split separating sub-bass, low-mids, vocal presence, and highs.

### Band Splitting

**Topology: Cascaded 2-way tree.** The 4-band split is built from three complementary LR4 (Linkwitz-Riley 4th-order) crossovers arranged as a binary tree:

```
                        Input
                          │
                    split @ 600Hz
                     /         \
               low-half       high-half
                 │                │
           split @ 150Hz    split @ 3000Hz
            /       \        /        \
         Low    Low-Mid    Mid       High
       0-150   150-600   600-3k    3k-20k
```

Each split is a complementary LR4 pair: `butter(2, fc, btype='low', output='sos')` applied twice in cascade (two 2nd-order sections = 4th-order LR4). This is correct because LR4 is defined as two cascaded Butterworth-2 filters. Do **NOT** use `butter(4, fc, output='sos')` — that produces a Butterworth-4, not an LR4. The key property: at each crossover frequency, the low-pass and high-pass outputs are each at -6dB, and they sum to unity (allpass).

Explicit implementation of double-`sosfilt` cascade:
```python
from scipy.signal import butter, sosfilt

def lr4_split(x: np.ndarray, fc: float, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Split signal into low and high bands using LR4 crossover at fc Hz."""
    sos_lp = butter(2, fc, btype='low', fs=sr, output='sos')
    sos_hp = butter(2, fc, btype='high', fs=sr, output='sos')
    # LR4 = two cascaded Butterworth-2 filters
    y_low = sosfilt(sos_lp, sosfilt(sos_lp, x))
    y_high = sosfilt(sos_hp, sosfilt(sos_hp, x))
    return y_low, y_high
```

`sosfiltfilt` (zero-phase) cannot be used here because it destroys the complementary magnitude relationship needed for unity reconstruction — when low-pass and high-pass `sosfiltfilt` outputs are summed, they don't add back to the original signal. Causal `sosfilt` with LR4 crossovers guarantees unity reconstruction (allpass sum). LR4 crossovers introduce frequency-dependent group delay (~4ms at 150Hz crossover, negligible above 600Hz). Since multiband compression applies to the already-summed mix bus, no inter-stem alignment issues arise, and the absolute delay is below audibility thresholds. The planned unity-reconstruction unit test (test_multiband.py) confirms correctness.

> **Phase caveat:** The unity-reconstruction test validates the crossover filter bank only (split then sum without processing). After per-band compression, reconstruction will **not** be bit-perfect unity because: (1) compression changes the envelope of each band independently, and (2) causal filters introduce group delay that interacts differently with compressed vs. uncompressed envelopes at crossover boundaries. This is expected and normal for multiband compressors — the crossover unity test validates the filter math, not the full compress-then-sum pipeline.

### Per-Band Compression Settings

| Band | Range | Threshold | Ratio | Attack | Release | Makeup |
|------|-------|-----------|-------|--------|---------|--------|
| **Low** | 0–150Hz | -14dB | 4:1 | 20ms | 200ms | 0dB |
| **Low-Mid** | 150–600Hz | -20dB | 3:1 | 10ms | 120ms | 0dB |
| **Mid** | 600–3kHz | -20dB | 2.5:1 | 5ms | 100ms | 0dB |
| **High** | 3k–20kHz | -24dB | 2:1 | 3ms | 80ms | 0dB |

Tighter on bass (4:1), gentler on highs (2:1) to avoid dullness. Fast attack on highs catches harsh transients.

> **Low band threshold note:** Raised from -18dB to -14dB to account for pre-limited content common in hip-hop/electronic. If low end sounds flat or pumpy during A/B testing, reduce ratio to 2.5:1 or raise threshold further.

> **Threshold rationale:** After band splitting, per-band levels are significantly lower than the full-band signal because each band contains only a portion of the spectrum's energy. The previous thresholds (-4 to -8 dBFS) were set relative to the full-band level and would rarely engage after splitting. These thresholds (-18 to -24 dBFS) are set to realistic per-band RMS levels. **Implementation must include per-band gain reduction logging** (`logger.debug("band=%s gain_reduction_dB=%.1f", band_name, gr)`) to verify compressors are actually engaging during testing. If per-band RMS varies significantly across input material, a future improvement would be to measure per-band peak/RMS after splitting and set thresholds dynamically (e.g., `threshold = per_band_peak_dB - 6`).
>
> **Makeup gain:** All bands default to **0 dB** makeup gain. Non-zero makeup gain defaults risk adding unwanted level to bands that weren't compressed (e.g., sparse arrangements where a band has little content). Makeup gain should be tuned during listening tests, not defaulted.

Each band compressed with `pedalboard.Compressor` (threshold, ratio, attack, release only — **`Compressor` has no `makeup_gain` parameter**). Makeup gain is applied manually after compression as a multiplication: `compressed *= 10 ** (makeup_db / 20.0)`. Bands are then summed back.

**Internal gain staging — output-level normalization:** Compression with 0 dB makeup gain on all bands results in 3-4 dB net level loss (compression reduces peaks but no makeup compensates). The auto-leveler is capped at 1.0 dB max_boost when multiband is active, so this loss goes uncompensated. To fix this, `multiband_compress` must internally restore the original integrated loudness:

1. Measure integrated LUFS of the input audio **before** band splitting
2. Split, compress per-band, and recombine
3. Measure integrated LUFS of the recombined output
4. Apply makeup gain to restore the pre-compression integrated LUFS: `gain_db = input_lufs - output_lufs; output *= 10 ** (gain_db / 20.0)`

This is internal gain staging, not creative makeup — it ensures multiband compression is level-neutral and doesn't shift the overall loudness balance before the auto-leveler and mastering stages.

```python
def multiband_compress(audio, sr, crossovers=(150, 600, 3000), settings=None) -> np.ndarray
```

### Auto-Leveler Tuning

When multiband is active, reduce auto-leveler aggressiveness since multiband handles per-frequency dynamics. **Precedence rule:** `ab_multiband_comp_v1` takes priority — when it is True, always use multiband-aware tuning regardless of `ab_autolvl_tune_v1` state.

All four flag-combination states:

| `ab_multiband_comp_v1` | `ab_autolvl_tune_v1` | Auto-leveler behavior |
|:-:|:-:|---|
| False | False | Default tuning: max_boost=2.0 dB, max_cut=3.0 dB, window_sec=3.0 |
| False | True | `ab_autolvl_tune_v1` tuning: max_boost=1.5 dB, max_cut=2.5 dB, window_sec=4.0 (**Intentional change from deployed values:** changes `max_boost_db` from 1.0->1.5 and `max_cut_db` from 2.0->2.5 to allow more dynamic range correction. This is a deliberate tweak; if A/B results are ambiguous, revert to original values.) |
| True | False | **Multiband-aware tuning: max_boost=1.0 dB, max_cut=1.5 dB, window_sec=3.0** (multiband takes precedence for boost/cut) |
| True | True | **Multiband-aware tuning: max_boost=1.0 dB, max_cut=1.5 dB, window_sec=4.0** (multiband takes precedence for boost/cut; window_sec from autolvl_tune) |

> **Unit test requirement:** All 4 flag-combination states above must be covered by unit tests in `test_pipeline_wiring.py`. The AB test suite only exercises all-off (control) and all-on (enhanced), so the intermediate states (multiband on / autolvl off, and multiband off / autolvl on) are never exercised there. Unit tests are the correct place to verify the precedence logic for each combination.

## Step 6: Create `mastering.py` — Static Mastering Chain

### v1 Approach: Static Mastering Chain

> **Design decision:** Reference-based mastering (matchering) uses the uploaded source song as a spectral/loudness target. The problem is that the remix is intentionally *different* from the source — it combines elements from two songs in a new way. Matching the remix's spectral profile to the source fights the creative intent and produces a tonal balance that serves neither song well. For v1, a simple static mastering chain produces more predictable, reliable results.

The mastering step applies operations in sequence (standard mastering order):
1. **Low-pass filter** (optional) — if `lossy_lpf_hz` is set, apply a gentle LPF at the given frequency to roll off codec artifacts above the source's spectral ceiling
2. **Constrained LUFS normalization** — target **-12 LUFS** (matches standard path). The LUFS gain is **capped** so that post-gain peaks stay within ~3 dB of the ceiling, preventing the limiter from doing excessive gain reduction. Specifically: `gain_to_apply = min(lufs_gain_needed, gain_that_keeps_peaks_within_3dB_of_ceiling)`. This avoids the failure mode where quiet input mixes push +6 to +10 dB of gain into the limiter, causing massive gain reduction and audible distortion. The standard path handles this via `lufs_normalize_constrained`; this is the `master_static` equivalent.
3. **True-peak limiter** — ceiling at **-1.0 dBTP** (ITU-R BS.1770-4 compliant). With constrained normalization, the limiter handles at most ~3-4 dB of gain reduction.

> **Why this order matters:** LUFS normalization adjusts gain to hit the loudness target, which may push peaks above the true-peak ceiling. The limiter runs last to catch any peaks introduced by the LUFS gain stage and enforce the -1.0 dBTP ceiling. Reversing this order (limit then normalize) allows the normalizer to push peaks above the limiter's ceiling, defeating its purpose. The constrained normalization ensures the limiter is never driven into excessive gain reduction.

```python
def master_static(
    audio: np.ndarray,
    sr: int,
    target_lufs: float = -12.0,
    ceiling_dbtp: float = -1.0,
    lossy_lpf_hz: float | None = None,
) -> np.ndarray
```

Inside the function:
```python
# 1. Optional low-pass filter for lossy sources
if lossy_lpf_hz is not None:
    # Apply gentle LPF at lossy_lpf_hz (e.g. 16kHz for Opus 128kbps)

# 2. Constrained LUFS normalization — delegate to existing utility.
#    REQUIRED SIGNATURE CHANGE in processor.py: add `headroom_db: float = 0.0` parameter
#    to `lufs_normalize_constrained`. Inside the function, apply headroom as:
#        ceiling_with_headroom = 10 ** ((ceiling_dbtp + headroom_db) / 20.0)
#    and use ceiling_with_headroom (instead of the raw ceiling) for peak-constraint math.
#    Default of 0.0 preserves backward compatibility — all existing call sites are unaffected.
#
#    The +3.0 dB headroom allowance is intentional: in the static mastering chain
#    a true-peak limiter immediately follows (step 3 below), so we can afford to let
#    peaks land up to 3 dB above the final ceiling — the limiter will catch them.
#    In the *standard* pipeline path, lufs_normalize_constrained is called with
#    headroom_db=0.0 (default) because no limiter follows immediately.
#    Do NOT "harmonize" these two call sites — the different headroom values are
#    correct for their respective signal chains.
audio = lufs_normalize_constrained(audio, sr, target_lufs=target_lufs,
                                   ceiling_dbtp=ceiling_dbtp, headroom_db=3.0)

# 3. True-peak limiter at ceiling
audio = true_peak_limit(audio, sr, ceiling_dbtp=ceiling_dbtp)
```

Returns the mastered float32 stereo array. This replaces the standard limiter chain (steps 14/15/15.5) when the flag is enabled — the static mastering chain handles both limiting and loudness normalization in one step.

> **Future improvement:** Reference-based mastering with curated, genre-appropriate reference tracks (not the user's uploaded source) could be reintroduced in a future version. This would require building a reference library keyed by genre/mood, which is out of scope for v1. The matchering dependency (`matchering>=1.0.0`) can be removed from `pyproject.toml` for v1.

## Step 7: Wire into Pipeline

In `pipeline.py`, add gated steps:

**Step 7.75** — Broad preset EQ only (before tempo stretch, after vocal bandpass). Resonance notch cuts are deferred to after stretch to avoid ringing artifacts (see eq.py note):
```python
if settings.ab_per_stem_eq_v1:
    for stem_type, audio in vocal_audio.items():
        vocal_audio[stem_type] = apply_corrective_eq(audio, sr, stem_type,
            apply_preset=True, apply_resonance_cuts=False)
    for stem_type, audio in inst_audio.items():
        inst_audio[stem_type] = apply_corrective_eq(audio, sr, stem_type,
            apply_preset=True, apply_resonance_cuts=False)
```

**Step 9.5** — Narrow resonance notch cuts (after tempo stretch is complete):
```python
if settings.ab_per_stem_eq_v1 and settings.ab_resonance_detection_v1:
    for stem_type, audio in vocal_audio.items():
        vocal_audio[stem_type] = apply_corrective_eq(audio, sr, stem_type,
            apply_preset=False, apply_resonance_cuts=True)
    for stem_type, audio in inst_audio.items():
        inst_audio[stem_type] = apply_corrective_eq(audio, sr, stem_type,
            apply_preset=False, apply_resonance_cuts=True)
```

**Step 13.3** — Multiband compression (after bus sum, before auto-leveler):
```python
if settings.ab_multiband_comp_v1:
    mixed = multiband_compress(mixed, sr)
```

**Step 13.7** — Auto-leveler with flag-aware tuning. The precedence logic must be explicit in the pipeline wiring:
```python
# Auto-leveler tuning: ab_multiband_comp_v1 takes priority over ab_autolvl_tune_v1
# window_sec always follows ab_autolvl_tune_v1 regardless of multiband state
window_sec = 4.0 if settings.ab_autolvl_tune_v1 else 3.0

if settings.ab_multiband_comp_v1:
    # Multiband already handles per-frequency dynamics — reduce auto-leveler aggressiveness
    auto_level_kwargs = dict(max_boost_db=1.0, max_cut_db=1.5, window_sec=window_sec)
elif settings.ab_autolvl_tune_v1:
    # ab_autolvl_tune_v1 tuning: wider window, slightly tighter limits
    auto_level_kwargs = dict(max_boost_db=1.5, max_cut_db=2.5, window_sec=window_sec)
else:
    # Default tuning
    auto_level_kwargs = dict(max_boost_db=2.0, max_cut_db=3.0, window_sec=window_sec)

# CRITICAL: detector_audio must be set to instrumental_bus to avoid the volume-dip
# regression (the ~11s/~22s dip bug). Without this, auto_level uses the mixed signal
# for detection, which causes vocals to trigger gain reduction on themselves.
auto_level_kwargs["detector_audio"] = instrumental_bus
auto_level_kwargs["target_percentile"] = 50.0
auto_level_kwargs["active_floor_db"] = -40.0

mixed = auto_level(mixed, sr, **auto_level_kwargs)
```

> **`window_sec` note:** The `window_sec` parameter follows `ab_autolvl_tune_v1` regardless of whether multiband is active: `window_sec=4.0` when `ab_autolvl_tune_v1=True`, `3.0` otherwise. Multiband-aware tuning only overrides `max_boost_db` and `max_cut_db`.

**Steps 14/14.5/15/15.5** — Mutual exclusion between static mastering and standard limiter chain:
```python
if settings.ab_static_mastering_v1:
    # Static mastering chain: constrained LUFS normalize (-12 LUFS) then limiter (-1.0 dBTP)
    mixed = master_static(mixed, sr, target_lufs=-12.0, ceiling_dbtp=-1.0)
    # Safety soft clip — catches inter-sample true peaks that can exceed -1.0 dBTP
    # after MP3 encoding. Without this, lossy codecs can reconstruct peaks above the
    # limiter ceiling, causing audible distortion. The standard path has this at step 15.5.
    safety_ceiling = 10 ** (-1.0 / 20.0)
    mixed = soft_clip(mixed, safety_ceiling, knee_db=2.0)
else:
    # Standard chain: limiter → LUFS normalize → soft-clip
    mixed = true_peak_limit(mixed, sr, ceiling_dbtp=-7.0, lookahead_ms=5.0, release_ms=50.0)  # step 14
    mixed = lufs_normalize_constrained(mixed, sr, target_lufs=-12.0, ceiling_dbtp=-1.0)        # step 15
    safety_ceiling = 10 ** (-1.0 / 20.0)
    mixed = soft_clip(mixed, safety_ceiling, knee_db=2.0)                                      # step 15.5
```

Add LUFS checkpoint logs after each new step for debugging.

### Source-Quality-Aware Processing

When input audio is flagged as a lossy YouTube source (via `source_quality` metadata from `AudioMetadata` — see youtube-input-plan.md), the enhancement chain should adapt to avoid amplifying compression artifacts. **Each song's source quality is tracked independently** — song A (vocals source) and song B (instrumentals source) may have different quality levels. Lossy-source EQ adjustments are applied only to stems from the lossy song, not blanket to all stems:

| Processing step | Adjustment for YouTube sources |
|----------------|-------------------------------|
| **Per-stem EQ** | Halve all HF boost gains (multiply by 0.5) — **only for stems from the lossy song**. Stems from a lossless source get normal EQ. |
| **Mastering** | Apply a gentle low-pass at the detected spectral ceiling (typically ~16kHz for Opus 128kbps) if **either** source is lossy |

> **Note:** No lossy-source adjustment is needed for the multiband compressor because all makeup gains already default to 0 dB (see Step 5 threshold rationale). If makeup gains are later tuned above 0 dB during listening tests, a lossy-source override for the high band should be revisited at that point.

> **Design decision: always-on for lossy sources (no feature flag).** Source-quality-aware adjustments are **protective** — they prevent degradation by avoiding operations that amplify compression artifacts. They are not enhancements that change the character of the audio. Therefore, these adjustments are always active when a song's `source_quality` indicates a lossy source (`"youtube"` prefix), regardless of which `ab_*` feature flags are enabled. Each song is evaluated independently — `is_lossy_source_a` and `is_lossy_source_b` are separate booleans derived from `source_quality_a` and `source_quality_b` respectively. Rationale: gating protective processing behind a flag risks shipping a configuration that actively degrades YouTube-sourced audio by boosting codec artifacts.

#### Pipeline wiring for source_quality

**Signature change required:** `run_pipeline()` in `pipeline.py` currently has no `source_quality` parameters. The following changes are needed across two files:

1. **`pipeline.py` — add params to `run_pipeline()`:**
```python
def run_pipeline(
    ...,
    source_quality_a: str | None = None,  # e.g. "youtube-opus-128kbps" or None
    source_quality_b: str | None = None,
) -> None:
```

2. **`remix.py` — pass values through from `_youtube_pipeline_wrapper`:** The `_youtube_pipeline_wrapper` function computes `source_quality_a` and `source_quality_b` but currently does NOT pass them to `run_pipeline()`. Without this wiring, the params always receive `None` and lossy-source processing is dead code. Add the pass-through:
```python
# In _youtube_pipeline_wrapper:
run_pipeline(..., source_quality_a=source_quality_a, source_quality_b=source_quality_b)
```

3. **`pipeline.py` — derive lossy flags inside `run_pipeline()`:** The `is_lossy_source_a/b` booleans must be computed inside `run_pipeline()` from the new params (they are NOT available on `AudioMetadata`):
```python
is_lossy_source_a = source_quality_a is not None and source_quality_a.startswith("youtube")
is_lossy_source_b = source_quality_b is not None and source_quality_b.startswith("youtube")
```

4. **`_pipeline_wrapper` (file upload path) — no changes needed.** The new params default to `None`, so file upload callers are unaffected.

Inside `run_pipeline()`, determine per-stem lossy status based on which song each stem comes from. **Important:** The vocal/instrumental source assignment is dynamic — `plan.vocal_source` determines which song provides vocals. Do not hardcode vocals=song_a:

```python
# At the start of the enhancement chain in pipeline.py:
is_lossy_source_a = source_quality_a is not None and source_quality_a.startswith("youtube")
is_lossy_source_b = source_quality_b is not None and source_quality_b.startswith("youtube")

# Dynamically map lossy flags to vocal/instrumental based on plan.vocal_source.
# When vocals come from song B, the lossy flags must be swapped.
is_lossy_vocal_source = is_lossy_source_a if plan.vocal_source == "song_a" else is_lossy_source_b
is_lossy_inst_source = is_lossy_source_b if plan.vocal_source == "song_a" else is_lossy_source_a

# Step 7.75 — Per-stem EQ (source-quality-aware)
# Apply lossy-source EQ adjustments only to stems from the lossy song.
if settings.ab_per_stem_eq_v1:
    vocal_eq_kwargs = {"halve_hf_boosts": True} if is_lossy_vocal_source else {}
    inst_eq_kwargs = {"halve_hf_boosts": True} if is_lossy_inst_source else {}
    for stem_type, audio in vocal_audio.items():
        vocal_audio[stem_type] = apply_corrective_eq(audio, sr, stem_type,
            apply_preset=True, apply_resonance_cuts=False, **vocal_eq_kwargs)
    for stem_type, audio in inst_audio.items():
        inst_audio[stem_type] = apply_corrective_eq(audio, sr, stem_type,
            apply_preset=True, apply_resonance_cuts=False, **inst_eq_kwargs)

# Step 13.3 — Multiband compression
# No lossy-source adjustment needed here — all makeup gains default to 0 dB.
# If makeup gains are later tuned above 0 dB, add a lossy-source HF override.
if settings.ab_multiband_comp_v1:
    mixed = multiband_compress(mixed, sr)

# Step 14.5 — Static mastering (source-quality-aware)
# If EITHER source is lossy, apply gentle LPF at spectral ceiling
if settings.ab_static_mastering_v1:
    master_kwargs = dict(target_lufs=-12.0, ceiling_dbtp=-1.0)
    if is_lossy_source_a or is_lossy_source_b:
        master_kwargs["lossy_lpf_hz"] = 16000  # gentle LPF at spectral ceiling
    mixed = master_static(mixed, sr, **master_kwargs)
```

Function signatures accept these optional parameters:
- `apply_corrective_eq(..., halve_hf_boosts: bool = False)` — when True, multiplies all HF boost gains by 0.5
- `master_static(..., lossy_lpf_hz: float | None = None)` — when set, applies a gentle LPF before mastering

The `source_quality` field is `None` for file uploads (no adjustments needed).

## Step 8: Tests

All tests use synthetic audio (sine waves), following existing patterns in `test_processor.py`.

### test_dereverb.py *(Deferred to post-v1)*

### test_eq.py
- Each stem type produces valid filtered output
- Float32 stereo shape preserved
- Frequency attenuation verifiable (250Hz sine through vocals preset → quieter)
- Unknown stem type falls back to "other"
- Resonance detection finds known peaks in synthetic signal
- Max resonances limit respected

### test_multiband.py
- Band split + recombine = unity (within tolerance) for flat signal
- Each band compresses independently
- Float32 stereo preserved
- Different crossover frequencies work

### test_mastering.py
- Synthetic signal produces mastered output with correct shape/dtype
- Output LUFS is within tolerance of target (-12 LUFS)
- Output true-peak does not exceed ceiling (-1.0 dBTP)
- Float32 stereo preserved

### test_pipeline_wiring.py (extend existing)
- Add cases with new flags enabled
- Verify LUFS within target range, true peak within ceiling
- **Auto-leveler 4-state test:** Verify all 4 combinations of `ab_multiband_comp_v1` x `ab_autolvl_tune_v1` produce the correct auto-leveler kwargs (see precedence table in Step 5). The AB suite only tests all-off and all-on; the intermediate states must be covered here.

### A/B Integration Test Suite (`scripts/run_ab_test_suite.py`)

Full-pipeline A/B comparison across 5 genre-diverse song pairs. See `docs/impl/ab-test-suite-plan.md` for details.

| # | Vocals | Instrumentals | Genre test |
|---|--------|---------------|------------|
| 1 | Hypnotize (Biggie) | Althea (Grateful Dead) | Rap + classic rock |
| 2 | Ghost Town (Kid Cudi/Kanye) | Khala My Friend (Amanaz) | Alt-hip-hop + Zamrock |
| 3 | Encore (Jay-Z) | Numb (Linkin Park) | Rap + nu-metal |
| 4 | Adventure of a Lifetime (Coldplay) | Give Life Back to Music (Daft Punk) | Pop vocals + French house |
| 5 | Air (MF DOOM) | Scarlet Begonias (Grateful Dead) | Underground rap + classic rock |

Each pair runs twice (all flags off, all flags on). Output: `mashupTests/{pair}/control.mp3` and `enhanced.mp3`.

## Verification

1. Run existing tests: `cd backend && uv run pytest` — all pass (no regressions)
2. Run new tests: `uv run pytest tests/test_eq.py tests/test_multiband.py tests/test_mastering.py`
3. Run the A/B test suite: `cd backend && uv run python ../scripts/run_ab_test_suite.py`
   - Runs 5 song pairs (control vs enhanced) automatically
   - Outputs to `mashupTests/` with labeled control/enhanced MP3s per pair
   - See `docs/impl/ab-test-suite-plan.md` for full test matrix and output structure
4. Listen to all 5 pairs, compare control vs enhanced for:
   - Vocal clarity and presence
   - Frequency balance and low-end tightness
   - Overall loudness consistency (check LUFS in `mashupTests/results.txt`)
   - Genre-specific issues (electronic sub-bass, rock dynamics, rap vocal intelligibility)

## A/B Test Suite

Batch test runner for evaluating sound quality flags across genre-diverse song pairs. Downloads audio from YouTube, runs the pipeline with different flag configurations, and measures LUFS/true-peak/crest factor on outputs.

- **Full plan:** `docs/impl/ab-test-suite-plan.md`
- **Script:** `scripts/run_ab_test_suite.py`
- **Test matrix:** 5 song pairs x 2 modes (compare: control vs enhanced; sweep: per-flag isolation)
- **Metrics:** Integrated LUFS, true-peak dBTP (ITU-R BS.1770-4 via 4x oversampling), crest factor
- **Output:** `mashupTests/{pair}/{variant}.mp3` + `mashupTests/results.txt` (pipe-delimited)

## Notes

- **matchering is GPLv3** — fine for personal project, would require backend to be GPLv3 if distributed. Feature flag lets it be disabled entirely.
- **pedalboard** is Apache 2.0, no license concerns.
- All new processing maintains float32 throughout, no int16 conversions.
- Research notes with full tool comparison: `notes/2026-02-26-ai-mastering-research.md`
