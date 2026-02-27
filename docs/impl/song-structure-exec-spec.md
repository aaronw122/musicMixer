# Song Structure Analysis -- Execution Spec

Pure build spec. Read this, implement in order, touch only the listed files.

## 1. Data Structures

Add to `backend/src/musicmixer/models.py`:

```python
@dataclass
class VocalGap:
    start_bar: int; end_bar: int; length_bars: int

@dataclass
class EnergyBuckets:
    noise_floor: float  # default 0.02; bars below = "silent"
    p10: float; p50: float; p85: float
    # silent=<noise_floor, low=<p10, medium=p10-p50, high=p50-p85, peak=>p85

@dataclass
class StemAnalysis:
    bar_rms: dict[str, np.ndarray]    # stem_name -> per-bar RMS (raw, NOT normalized)
    combined_energy: np.ndarray       # per-bar combined energy (normalized to p99=1.0)
    vocal_active: np.ndarray          # per-bar bool
    vocal_gaps: list[VocalGap]
    bucket_thresholds: EnergyBuckets

@dataclass
class SectionInfo:
    start_bar: int; end_bar: int; bar_count: int
    start_time: float; end_time: float  # seconds
    label: str            # intro|verse|chorus|instrumental|breakdown|build|outro
    energy_level: str     # low|medium|high|peak
    energy_trajectory: str  # e.g. "medium->high" (thirds, deduped)
    density: str          # sparse|mid|full|full+extra
    vocal_status: str     # vox:yes|vox:no|vox:fading
    annotations: list[str]  # ["DROP","BUILD","GOOD INSTRUMENTAL SOURCE"]

@dataclass
class SongStructure:
    sections: list[SectionInfo]; vocal_gaps: list[VocalGap]; total_bars: int

@dataclass
class CrossSongRelationships:
    loudness_diff_db: float         # 20*log10(rms_a/rms_b), positive=A louder
    energy_profile_a: str; energy_profile_b: str  # "consistent high" / "wide dynamic range"
    vocal_source: str               # "song_a" or "song_b"
    vocal_prominence_a_db: float; vocal_prominence_b_db: float  # dB above accompaniment
    instrumental_sections: list[str]  # bar ranges from recommended source
    frequency_conflicts: str        # warning text or empty
    stretch_pct: float
```

Add to existing `AudioMetadata`:
```python
    key: Optional[str] = None; scale: Optional[str] = None
    key_confidence: Optional[float] = None; has_modulation: bool = False
    mean_rms: Optional[float] = None  # from original mix audio (NOT summed stems)
    stem_analysis: Optional[StemAnalysis] = None
    song_structure: Optional[SongStructure] = None
```

**AudioMetadata expansion note:** The `stem_analysis` field stores the full per-stem analysis results (`bar_rms` dict with per-stem arrays, `combined_energy`, `vocal_active` mask, etc.). This data is produced by the structure analysis step (3.5) and consumed by the interpreter for building the system prompt. When adding new analysis features (e.g., per-stem spectral centroid, per-stem dynamic range), add them as fields on `StemAnalysis` rather than creating parallel data structures on `AudioMetadata`. This keeps all stem-level analysis co-located and avoids fragmented data flows through the pipeline.

## 2. Algorithm Specs

### 2.0 Stem Loading & Bar Grid

Load each stem WAV at **22050 Hz**. Bar boundaries from reconciled beat grid: `beat_frames[::4]`. Partial final bar: discard if <4 beats remain, else extend last bar boundary to audio end.

### 2.1 Normalization Pipeline

Two separate noise floors — don't confuse them:

1. Store **raw** per-stem bar-level RMS (feeds density, vocal detection)
2. Apply noise floor filter at **0.001 / -60 dBFS** BEFORE normalization (pre-filter)
3. Compute combined energy (equal-weighted sum), normalize to **p99=1.0** (feeds section detection)
4. Compute adaptive bucket thresholds on the normalized combined energy, filtering bars below **0.02 RMS** (bucket noise floor)

Percentiles are computed on the **weighted-total-energy-per-bar array** (the normalized combined energy from step 3), NOT on flattened per-stem values. Remaining active bars (above 0.02 RMS): silent=<0.02, low=<p10, medium=p10-p50, high=p50-p85, peak=>p85.

### 2.2 Vocal Activity Detection (Dual-Threshold Hysteresis)

On vocal stem RMS (raw, per-bar). Onset: >15% of stem peak RMS -> start region. Sustain: stay active while >8% of stem peak RMS. Min duration: 2 bars (discard shorter).

**Threshold rationale:** The onset threshold of 15% of peak RMS (approximately -16.5 dB below peak) is chosen as a relative threshold to adapt to different vocal recording levels. A range of 15-20% of peak RMS is acceptable -- 15% catches quieter vocal passages (spoken word, whispered sections), while 20% is more conservative and avoids false positives from vocal bleed in the stem. Tune within this range based on test song results. The key property is that thresholds are **relative to the stem's own peak**, not absolute dB values, so they adapt automatically to different source levels.

### 2.3 Vocal Gap Detection

Contiguous runs of 2+ bars where `vocal_active == False` -> `VocalGap(start, end, length)`.

**MVP scope:** Song A vocals + Song B instrumentals. Gaps used for arrangement (e.g., boost instrumental gain in vocal-free sections), NOT for cross-song stem switching.

### 2.4 Section Detection (3-stage)

**Stage 1: Boundary detection**
```
combined_energy = equal-weighted sum of all stem bar energies
smoothed = moving_average(combined_energy, window=4)
deriv = abs(diff(smoothed))

per_stem_derivs = [abs(diff(smooth(stem))) for stem in stems]
max_stem_deriv = element_wise_max(per_stem_derivs)

change_signal = np.maximum(deriv, max_stem_deriv)

threshold = max(median(change_signal) * 3.0, 0.05)
boundaries = peaks above threshold, min 4 bars apart
```

**Stage 1b: Phrase quantization**

Snap boundaries to nearest 4-bar grid. Deduplicate. Remove segments < 4 bars.

**Stage 2: Label segments**

Use adaptive energy buckets (p10/p50/p85) for classification. Decision tree (checked in order):
```
IF first segment AND (low energy OR mid energy + no vocals): "intro"
ELIF last segment AND (low energy OR position > 85% of song):  "outro"
ELIF first-quarter energy NOT high AND steady rise (1.5x first-to-last quarter, epsilon > 0.01): "build"
ELIF high energy AND vocals active:     "chorus"
ELIF high energy AND no vocals:         "instrumental"
ELIF drums low AND vocals active:       "breakdown"
ELIF mid energy AND vocals active:      "verse"
ELIF low energy AND no vocals:          "instrumental"
ELIF mid energy AND no vocals:          "instrumental"
ELSE:                                    "verse"
```

**Arrangement density** (raw per-stem RMS). Stem "active" = segment mean > p25 of that stem's own bars. sparse=0-2 active, mid=3-4, full=5-6 (none at peak), full+extra=5-6 with at least one stem's segment-max > p90 of own bars.

**`GOOD INSTRUMENTAL SOURCE` annotation:** After labeling, annotate any section where vocal stem RMS stays below the sustain threshold (8% stem peak) for the entire section. These are clean instrumental sections with no vocal bleed.

**Build/drop:** Build=1.5x rise first-to-last quarter, first quarter NOT high (epsilon>0.01). Drop=1.5x jump prev-section-last-bar to this-section-first-bar, annotate `DROP`.

**Energy trajectory:** Section thirds -> bucket each -> format `"low->medium->high"`, dedupe adjacent same.

**Stage 3:** Merge adjacent same-label. Absorb sections <4 bars into louder neighbor.

### 2.5 Key Detection

Interface: `detect_key(path) -> (key, scale, confidence)`. Primary: `essentia.KeyExtractor` at 44.1kHz. Fallback: `librosa.chroma_cqt`. If essentia fails, use librosa silently.

**Modulation:** Run `KeyExtractor` on first 60% and last 40%. If keys differ, `has_modulation = True`.

**Circle-of-fifths compatibility:** When reporting key analysis results to the LLM, include a note about harmonic compatibility. Keys that are adjacent on the circle of fifths (e.g., C major and G major, A minor and E minor) are generally compatible for mixing. Keys a tritone apart (e.g., C and F#) are maximally dissonant. Consider flagging key relationships in `CrossSongRelationships` as `compatible` (same key, relative major/minor, or adjacent on circle of fifths), `neutral` (2-3 steps apart), or `conflicting` (4+ steps apart) to help the LLM make informed mixing decisions.

### 2.6 Cross-Song Loudness

`loudness_diff_db = 20 * log10(max(mean_rms_a, 1e-10) / max(mean_rms_b, 1e-10))`. Skip if either < 0.001. Guidance: <2dB=similar, 2-6dB=reduce louder by ~N dB, >6dB=will overpower.

### 2.7 Vocal Prominence

`prominence_db = 20 * log10(mean_vocal_rms / mean_non_vocal_rms)`. **Both means computed over vocal-active bars only** (not all bars). `mean_vocal_rms` = vocal stem mean across bars where `vocal_active == True`. `mean_non_vocal_rms` = sum of non-vocal stems across those **same** vocal-active bars only. Higher = cleaner separation = better vocal source. If non-vocal sections of the vocal stem have energy above the onset threshold (15% stem peak), flag as "bleed expected."

## 3. Calibration Values

Smoothing=4bars, threshold=3.0x median (floor=0.05, may need 0.03 for compressed), phrase grid=4bars, min section=4bars, build ratio=1.5x (epsilon>0.01), drop ratio=1.5x bar-to-bar, noise floor=0.02 RMS, vocal onset=15% stem peak, vocal sustain=8% stem peak, vocal min duration=2bars, normalization=p99.

**Calibration budget:** 1 hour max on test songs. **Escape hatch:** If boundary detection fails after 1hr, fall back to per-bar classification but keep adaptive bucketing + vocal hysteresis.

## 4. Example Output (illustrative bar counts/timestamps)

Replaces current `SONG DATA` section in system prompt. Layer 3 MVP: template `"{stem}: {energy_bucket}-energy, {density} density"`.

```
=== LAYER 1: SONG OVERVIEW ===
Song A: "Hypnotize" -- Hip-hop, 95 BPM, Emin, 4:00, 96 bars.
Vocals: dominant rap, +8 dB above instrumental, clean separation. Energy: compressed. Tempo: metronomic.
Song B: "Althea" -- Rock/jam, 118 BPM, Amaj, 6:52, 203 bars.
Vocals: moderate, +2 dB above instrumental, guitar bleed expected. Energy: wide dynamic range. Tempo: live (~3% drift).

=== LAYER 2: SECTION MAP ===
Song A (96 bars):
  1-4    4b  0:00-0:10 | intro     | medium       | mid       | vox:no  | GOOD INSTRUMENTAL SOURCE
  5-20   16b 0:10-0:50 | verse     | high         | full      | vox:yes
  21-28  8b  0:50-1:10 | chorus    | high         | full+extra| vox:yes | DROP
  29-44  16b 1:10-1:51 | verse     | high         | full      | vox:yes
  45-52  8b  1:51-2:11 | chorus    | high         | full+extra| vox:yes
  53-60  8b  2:11-2:31 | breakdown | high->medium | mid       | vox:yes
  61-64  4b  2:31-2:41 | build     | medium->high | mid       | vox:no  | GOOD INSTRUMENTAL SOURCE
  65-80  16b 2:41-3:22 | verse     | high         | full      | vox:yes
  81-88  8b  3:22-3:42 | chorus    | high         | full+extra| vox:yes | DROP
  89-96  8b  3:42-4:02 | outro     | high->medium | mid       | vox:fading
Vocal gaps: 1-4, 61-64
Song B (203 bars):
  1-8    8b  0:00-0:16 | intro        | low->medium      | sparse    | vox:no  | GOOD INSTRUMENTAL SOURCE
  9-24   16b 0:16-0:48 | verse        | medium           | mid       | vox:yes
  25-36  12b 0:48-1:13 | chorus       | medium->high     | full      | vox:yes
  37-60  24b 1:13-2:02 | instrumental | high->peak       | full+extra| vox:no  | GOOD INSTRUMENTAL SOURCE
  61-76  16b 2:02-2:34 | verse        | medium           | mid       | vox:yes
  77-88  12b 2:34-2:58 | chorus       | medium->high     | full      | vox:yes
  89-136 48b 2:58-4:36 | instrumental | high->peak->high | full+extra| vox:no  | GOOD INSTRUMENTAL SOURCE
  137-152 16b 4:36-5:09 | verse       | medium->low      | mid       | vox:yes
  153-168 16b 5:09-5:41 | chorus      | medium->high     | full      | vox:yes
  169-184 16b 5:41-6:14 | instrumental | high             | full      | vox:no  | GOOD INSTRUMENTAL SOURCE
  185-203 19b 6:14-6:52 | outro        | medium->low      | sparse    | vox:no
Vocal gaps: 1-8, 37-60, 89-136, 169-203

=== LAYER 3: STEM CHARACTER ===
Song A: vocals: high-energy, full. drums: high-energy, full. bass: high-energy, full, 40-100Hz. other: high-energy, full. (guitar: negligible | piano: minor)
Song B: guitar: high-energy, variable, 200Hz-8kHz. drums: high-energy, full. bass: medium-energy, mid, 60-250Hz. vocals: medium-energy, mid (bleed). piano: low-energy, sparse. (other: minimal)

=== LAYER 4: CROSS-SONG ===
Loudness: Song A ~6 dB louder. Reduce Song A stems ~6 dB.
Energy: A=consistent high (compressed). B=wide dynamics, provides the arc.
Vocal source: Song A (+8 dB, clean). Song B has guitar bleed (+2 dB).
Instrumental source: Song B jams (bars 37-60, 89-136). Intro (1-8) for sparse.
Conflict: Song A "other" may mask Song B guitar at 1-4 kHz.
```

## 5. Files to Modify

- **`models.py`** -- Add all dataclasses above. Add new fields to `AudioMetadata`. Remove `energy_regions` placeholder.
- **`services/analysis.py`** -- Add `analyze_stems()`, `detect_key()`, `_detect_key_librosa()`, `detect_modulation()`, `detect_sections()`, `quantize_to_phrases()`, `label_sections()`, `merge_sections()`, `compute_relationships()`
- **`services/interpreter.py`** -- Rewrite `_build_song_info()`. Delete `_condense_energy_profile()` (and its label vocabulary: `rhythmic`/`sustained`/`sparse`/`moderate`). Add failure mode guards to `_build_system_prompt()`. Update few-shot examples. Fix `"weighted_midpoint"` -> `"average"`. Modify `compute_tempo_plan()` to expose stretch % to LLM. Add post-plan validation warning for vocal stretch limits.
- **`services/pipeline.py`** -- Add Step 3.5 after `reconcile_bpm`, before `interpret_prompt`. SSE: "Analyzing song structure..."

All paths relative to `backend/src/musicmixer/`.

## 6. Implementation Order

**Step 3.5 MUST use the reconciled beat grid from Step 3** (`reconcile_bpm()` may halve/double `beat_frames`). Bar boundaries: `beat_frames[::4]`.

Pipeline dependency chain:
```
Step 1 (separate stems) -> Step 2 (analyze originals) -> Step 3 (reconcile_bpm) -> Step 3.5 (structure analysis) -> Step 4 (interpret_prompt)
```

1. Adaptive percentile bucketing (0.5h)
2. Dual-threshold vocal hysteresis (0.75h)
3. Key detection + modulation (1h)
4. Derivative boundary detection + per-stem max pool (1.5h)
5. Phrase quantization (0.25h)
6. Percentile labeling decision tree (1.5h)
7. Arrangement density + build/drop (1h)
8. Energy trajectory strings (0.25h)
9. Section merge/cleanup (0.5h)
10. **Calibration checkpoint** -- test both songs, 1hr max (1h) -- running total: 8.25h
11. Cross-song RMS loudness (0.25h)
12. Vocal gap annotation (0.25h)
13. Vocal prominence + stretch limits (0.5h). **New data flow:** `compute_tempo_plan()` must expose stretch % to the LLM. Add plan validation: warn (don't truncate) if any vocal section exceeds bar limits at computed stretch ratio.
14. Stem formatting, 6-stem, suppress silent (0.25h)
15. System prompt failure mode guards (0.5h)
16. LLM prompt integration + few-shot updates (1.5h)
17. Fix `"weighted_midpoint"` -> `"average"` in interpreter.py fallback plan (0.1h)
18. End-to-end test both songs (1h)
19. Buffer (2.4h) -- **total: 15h**

**Cut order if behind:** 1) trajectory strings 2) build/drop 3) per-stem derivative (combined-only) 4) density 5) nuclear: per-bar classification, keep bucketing+hysteresis.

**Never cut:** adaptive bucketing, vocal hysteresis, phrase quantization, failure mode guards, cross-song RMS.

## 7. System Prompt Additions (Phase 4)

Add to `_build_system_prompt()`. **7 failure mode guards:**
1. Prefer no-vocal instrumental sections. "other" stem low-energy->0.2; medium+->0.4-0.6 (genre identity).
2. Vocals active: reduce mid-freq stems (guitar/piano/other) 30%+. Only drums+bass at full vol.
3. Never drums from both songs simultaneously. No overlapping bass lines.
4. Match vocal energy to instrumental energy. Exception: vocal over minimal beat = intentional.
5. Must have 1+ contrast moment, min 3 energy levels.
6. End with 4-8 bars reduced energy or natural outro. Never cut at full energy.
7. Vary gain profiles. Strip to drums+bass+vocals for contrast, full arrangement for impact.

**Additional prompt notes:** Stagger stem entries over 2-4 bars. Begin vocal sections 1-2 beats early for pickup notes. Stretch >12%: 8 bars max at 15%, 4 bars max above 15%, prefer stretching instruments over vocals. Two peak stems at full vol will clip -- reduce one by 3-6 dB. Add advisory: "Section labels are approximate guidance, not rigid constraints."

## 8. Implementation Notes

- 1 bar = 4 beats. Analysis=bars, plan schema=beats. Conversion explicit.
- Always 6-stem (drums/bass/guitar/piano/vocals/other). Drop 4-stem logic.
- Include "other" stem in ALL energy computations.
- `mean_rms` from original mix in `analyze_audio()`, NOT summed stems.
- Analysis labels (chorus/breakdown/verse) differ from plan labels (intro/verse/breakdown/drop/outro). Intentional.
- BPM outside 70-170: re-run `beat_track()` on drum stem.
- **BPM half/double-time sanity check:** When two songs have detected BPMs where one is approximately 2x the other (within 5% tolerance), verify whether the faster BPM is a double-time detection artifact. Common case: a 140 BPM detection on a song that is actually 70 BPM with busy hi-hats. Compare the detected BPM against the drum stem's kick pattern periodicity. If the kick pattern suggests the lower BPM, use half the detected value. Similarly, if a song detects at ~75 BPM but has a driving feel, it may be 150 BPM with a half-time kick. Log a warning when BPM is halved or doubled for transparency.
- Phase coherence: all stems per-song stretched together. No per-stem stretching.
- Essentia new dep. If install fails, librosa fallback. Do not block.
